import datetime as dt
import itertools
import math
from typing import Optional, Sequence, Union

import numpy as np

from exchange import (
    BookView,
    CancelOrderMsg,
    DoneMsg,
    FillMsg,
    LimitOrderMsg,
    OrderAcceptedMsg,
    OrderRejectedMsg,
    Side,
)
from kernel import AgentId, Kernel, Oracle


class MarketMaker:
    def __init__(
        self,
        my_id: AgentId,
        exchange_id: AgentId,
        kernel: Kernel,
        book: BookView,
        rng: np.random.Generator,
        quote_rates: Union[Sequence[float], np.ndarray],
        cancel_rates: Union[Sequence[float], np.ndarray],
        quote_size: float,
        tick_size: float = 0.01,
        initial_mid: float = 100.0,
        oracle: Optional[Oracle] = None,
        obi_cutoffs: Optional[Sequence[float]] = None,
    ):
        self.my_id = my_id
        self.exchange_id = exchange_id
        self.kernel = kernel
        self.book = book
        self.rng = rng

        qr = np.asarray(quote_rates, dtype=float)
        cr = np.asarray(cancel_rates, dtype=float)

        if qr.ndim == 1:
            qr = qr[np.newaxis, np.newaxis, :]
        elif qr.ndim != 3:
            raise ValueError("quote_rates must be a 1D or 3D numpy array")

        if cr.ndim == 1:
            cr = cr[np.newaxis, np.newaxis, :]
        elif cr.ndim != 3:
            raise ValueError("cancel_rates must be a 1D or 3D numpy array")

        self.quote_rates = qr
        self.cancel_rates = cr
        self.s_max_q, self.n_obi_q, self.n_levels = self.quote_rates.shape
        self.s_max_c, self.n_obi_c, _ = self.cancel_rates.shape
        self.n_obi_bins = max(self.n_obi_q, self.n_obi_c)
        self._is_static = (
            self.s_max_q == 1
            and self.n_obi_q == 1
            and self.s_max_c == 1
            and self.n_obi_c == 1
        )

        if obi_cutoffs is not None:
            self.obi_cutoffs = np.asarray(obi_cutoffs, dtype=float)
        elif self.n_obi_bins > 1:
            self.obi_cutoffs = np.linspace(-1.0, 1.0, self.n_obi_bins + 1)[1:-1]
        else:
            self.obi_cutoffs = np.empty(0, dtype=float)

        self.quote_size = quote_size
        self.tick_size = tick_size
        self.oracle = oracle
        self.m = initial_mid / tick_size  # reference price, in ticks

        self.r_lo_init = float(2 * self.quote_rates[0, self.n_obi_q // 2].sum())
        self.cancel_rates_max = float(self.cancel_rates.max())
        self._bound = self.r_lo_init
        self._last_state: tuple[Optional[int], Optional[int], int, int] = (
            None,
            None,
            0,
            0,
        )

        # Pre-allocate event intensity buffers:
        # [0 : L)         -> BUY limit order at level l
        # [L : 2L)        -> SELL limit order at level l
        # [2L : 3L)       -> BUY cancellation at level l
        # [3L : 4L)       -> SELL cancellation at level l
        self._rates_buf = np.empty(4 * self.n_levels, dtype=float)
        if self._is_static:
            self._rates_buf[: self.n_levels] = self.quote_rates[0, 0]
            self._rates_buf[self.n_levels : 2 * self.n_levels] = self.quote_rates[0, 0]
        self._cum_buf = np.empty(4 * self.n_levels, dtype=float)

        self._order_seq = itertools.count(1)
        self.pending: dict[int, tuple[Side, int]] = {}  # client id -> (side, ticks)
        self.resting: dict[int, tuple[Side, int]] = {}  # exchange id -> (side, ticks)
        self.orders_at: dict[
            tuple[Side, int], list[int]
        ] = {}  # (side, ticks) -> [exchange id]

    def kernel_start(self, start_time: dt.datetime) -> None:
        if self.oracle is not None:
            self.oracle.initialize(start_time)
            self.m = self.oracle.get_price(start_time) / self.tick_size
        self._schedule(start_time, self.r_lo_init)

    def kernel_stop(self, stop_time: dt.datetime) -> None:
        pass

    def _get_state(
        self, bid: Optional[int], ask: Optional[int]
    ) -> tuple[Optional[int], Optional[int], int, int]:
        if self._is_static or bid is None or ask is None or ask <= bid:
            return bid, ask, 0, self.n_obi_bins // 2

        s_idx = ask - bid - 1
        if self.n_obi_bins <= 1:
            return bid, ask, s_idx, 0

        v_b = self.book.size_at_ticks(Side.BUY, bid)
        v_a = self.book.size_at_ticks(Side.SELL, ask)
        tot = v_b + v_a
        if tot <= 0:
            i_idx = self.n_obi_bins // 2
        else:
            obi = (v_b - v_a) / tot
            i_idx = int(np.searchsorted(self.obi_cutoffs, obi, side="right"))
        return bid, ask, s_idx, i_idx

    def _compute_intensities(
        self,
        current_time: dt.datetime,
        state: Optional[tuple[Optional[int], Optional[int], int, int]] = None,
    ) -> np.ndarray:
        if state is None:
            bid, ask = self.book.best_bid_ticks(), self.book.best_ask_ticks()
            state = self._get_state(bid, ask)
        else:
            bid, ask, _, _ = state

        self._last_state = state
        _, _, s_idx, i_idx = state

        if self.oracle is not None:
            self.m = self.oracle.get_price(current_time) / self.tick_size
        elif bid is not None and ask is not None:
            self.m = (bid + ask) / 2

        n_levels = self.n_levels
        rates_buf = self._rates_buf

        if not self._is_static:
            sq = min(s_idx, self.s_max_q - 1)
            iq_b = i_idx if self.n_obi_q > 1 else 0
            iq_a = (self.n_obi_q - 1 - i_idx) if self.n_obi_q > 1 else 0
            rates_buf[:n_levels] = self.quote_rates[sq, iq_b]
            rates_buf[n_levels : 2 * n_levels] = self.quote_rates[sq, iq_a]

            sc = min(s_idx, self.s_max_c - 1)
            ic_b = i_idx if self.n_obi_c > 1 else 0
            ic_a = (self.n_obi_c - 1 - i_idx) if self.n_obi_c > 1 else 0
            cr_buy = self.cancel_rates[sc, ic_b]
            cr_sell = self.cancel_rates[sc, ic_a]
        else:
            cr_buy = self.cancel_rates[0, 0]
            cr_sell = cr_buy

        ex = self.book._ex
        bids = ex.bids
        asks = ex.asks
        lot_size = ex.lot_size

        b_base = ask if (self.oracle is None and ask is not None) else math.ceil(self.m)
        a_base = (
            bid if (self.oracle is None and bid is not None) else math.floor(self.m)
        )

        cancel_buy_slice = rates_buf[2 * n_levels : 3 * n_levels]
        cancel_sell_slice = rates_buf[3 * n_levels :]
        for i in range(n_levels):
            lvl = i + 1
            b_lvl = bids.get(b_base - lvl)
            a_lvl = asks.get(a_base + lvl)
            v_b = 0.0 if b_lvl is None else b_lvl.total_size * lot_size
            v_a = 0.0 if a_lvl is None else a_lvl.total_size * lot_size
            cancel_buy_slice[i] = cr_buy[i] * v_b
            cancel_sell_slice[i] = cr_sell[i] * v_a

        return np.cumsum(rates_buf, out=self._cum_buf)

    def on_wakeup(self, current_time: dt.datetime) -> None:
        cum = self._compute_intensities(current_time)
        rate = float(cum[-1])
        n_levels = self.n_levels
        if rate > 0 and self.rng.random() < min(1.0, rate / self._bound):
            u = self.rng.uniform(0, rate)
            k = int(np.searchsorted(cum, u, side="right"))
            if k < n_levels:
                self._submit(Side.BUY, k + 1)
            elif k < 2 * n_levels:
                self._submit(Side.SELL, k - n_levels + 1)
            elif k < 3 * n_levels:
                self._cancel(Side.BUY, k - 2 * n_levels + 1)
            else:
                self._cancel(Side.SELL, k - 3 * n_levels + 1)
        self._schedule(current_time, rate)

    def on_message(self, current_time: dt.datetime, msg) -> None:
        if isinstance(msg, OrderAcceptedMsg):
            key = self.pending.pop(msg.client_order_id)
            self.resting[msg.order_id] = key
            self.orders_at.setdefault(key, []).append(msg.order_id)
        elif isinstance(msg, OrderRejectedMsg):
            self.pending.pop(msg.client_order_id, None)
        elif isinstance(msg, DoneMsg):
            key = self.resting.pop(msg.order_id, None)
            if key is not None and key in self.orders_at:
                ids = self.orders_at[key]
                try:
                    ids.remove(msg.order_id)
                    if not ids:
                        del self.orders_at[key]
                except ValueError:
                    pass
        elif isinstance(msg, FillMsg):
            # Only reschedule if best quotes or (s, i) state actually changed
            bid, ask = self.book.best_bid_ticks(), self.book.best_ask_ticks()
            new_state = self._get_state(bid, ask)
            if new_state != self._last_state:
                cum = self._compute_intensities(current_time, new_state)
                self._schedule(current_time, float(cum[-1]))

    def _schedule(self, now: dt.datetime, rate: float) -> None:
        self._bound = rate + self.cancel_rates_max * self.quote_size * len(self.pending)
        if self._bound <= 0:
            return
        wait = self.rng.exponential(1.0 / self._bound)
        self.kernel.set_wakeup(self.my_id, now + dt.timedelta(seconds=wait))

    def _submit(self, side: Side, level: int) -> None:
        ticks = self._ticks(side, level)
        client_id = self.my_id * 1_000_000 + next(self._order_seq)
        self.pending[client_id] = (side, ticks)
        order = LimitOrderMsg(
            self.my_id,
            client_id,
            side,
            self._usd(ticks),
            self.quote_size,
            post_only=(self.oracle is None),
        )
        self.kernel.send_message(self.my_id, self.exchange_id, order)

    def _cancel(self, side: Side, level: int) -> None:
        key = (side, self._ticks(side, level))
        ids = self.orders_at.get(key)
        if not ids:
            return
        idx = int(self.rng.integers(len(ids)))
        order_id = ids.pop(idx)
        if not ids:
            del self.orders_at[key]
        del self.resting[order_id]
        self.kernel.send_message(
            self.my_id, self.exchange_id, CancelOrderMsg(self.my_id, order_id)
        )

    def _ticks(self, side: Side, level: int) -> int:
        if self.oracle is None:
            best_bid = self.book.best_bid_ticks()
            best_ask = self.book.best_ask_ticks()

            if side is Side.BUY and best_ask is not None:
                return best_ask - level

            if side is Side.SELL and best_bid is not None:
                return best_bid + level

        # fallback
        if side is Side.BUY:
            return math.ceil(self.m) - level

        return math.floor(self.m) + level

    def _usd(self, ticks: int) -> float:
        return round(ticks * self.tick_size, 8)
