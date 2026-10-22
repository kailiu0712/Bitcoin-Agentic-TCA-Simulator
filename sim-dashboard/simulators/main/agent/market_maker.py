import datetime as dt
import itertools
import math
from typing import Optional

import numpy as np

from exchange import (
    BookView,
    CancelOrderMsg,
    DoneMsg,
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
        quote_rates: list[float],
        cancel_rates: list[float],
        quote_size: float,
        tick_size: float = 0.01,
        initial_mid: float = 100.0,
        oracle: Optional[Oracle] = None,
    ):
        self.my_id = my_id
        self.exchange_id = exchange_id
        self.kernel = kernel
        self.book = book
        self.rng = rng
        self.quote_rates = np.asarray(quote_rates, dtype=float)
        self.cancel_rates = np.asarray(cancel_rates, dtype=float)
        self.n_levels = len(self.quote_rates)
        self.quote_size = quote_size
        self.tick_size = tick_size
        self.oracle = oracle
        self.m = initial_mid / tick_size  # reference price, in ticks

        self.r_lo = 2 * self.quote_rates.sum()  # total limit order arrival rate
        self.cancel_rates_max = self.cancel_rates.max()
        self._bound = self.r_lo

        # pre-allocate buffers to avoid per-wakeup array allocations
        self._rates_buf = np.empty(2 * self.n_levels, dtype=float)
        self._rates_buf[: self.n_levels] = 2 * self.quote_rates
        self._cum_buf = np.empty(2 * self.n_levels, dtype=float)

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
        self._schedule(start_time, self.r_lo)

    def kernel_stop(self, stop_time: dt.datetime) -> None:
        pass

    def on_wakeup(self, current_time: dt.datetime) -> None:
        if self.oracle is not None:
            self.m = self.oracle.get_price(current_time) / self.tick_size
        else:
            bid, ask = self.book.best_bid_ticks(), self.book.best_ask_ticks()
            if bid is not None and ask is not None:
                self.m = (bid + ask) / 2  # keep the last mid if a side is empty

        n_levels = self.n_levels
        ex = self.book._ex
        bids = ex.bids
        asks = ex.asks
        lot_size = ex.lot_size
        ceil_m = math.ceil(self.m)
        floor_m = math.floor(self.m)

        cancel_slice = self._rates_buf[n_levels:]
        cancel_rates = self.cancel_rates
        for i in range(n_levels):
            lvl = i + 1
            b_lvl = bids.get(ceil_m - lvl)
            a_lvl = asks.get(floor_m + lvl)
            lots = (0 if b_lvl is None else b_lvl.total_size) + (
                0 if a_lvl is None else a_lvl.total_size
            )
            cancel_slice[i] = cancel_rates[i] * (lots * lot_size)

        cum = np.cumsum(self._rates_buf, out=self._cum_buf)
        rate = cum[-1]  # true total intensity r_lo + sum(nu_l * V_l)
        if self.rng.random() < min(1.0, rate / self._bound):  # otherwise thinned
            u = self.rng.uniform(0, rate)
            k = int(np.searchsorted(cum, u, side="right"))
            if k < n_levels:
                self._submit(k + 1)
            else:
                self._cancel(k - n_levels + 1)
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

    def _schedule(self, now: dt.datetime, rate: float) -> None:
        self._bound = rate + self.cancel_rates_max * self.quote_size * len(self.pending)
        wait = float(self.rng.exponential(1 / self._bound))
        self.kernel.set_wakeup(self.my_id, now + dt.timedelta(seconds=wait))

    def _submit(self, level: int) -> None:
        side = Side.BUY if self.rng.random() < 0.5 else Side.SELL
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

    def _cancel(self, level: int) -> None:
        buy_key = (Side.BUY, self._ticks(Side.BUY, level))
        sell_key = (Side.SELL, self._ticks(Side.SELL, level))
        buy_ids = self.orders_at.get(buy_key, [])
        sell_ids = self.orders_at.get(sell_key, [])
        n_buy = len(buy_ids)
        n_total = n_buy + len(sell_ids)
        if n_total == 0:
            return
        idx = int(self.rng.integers(n_total))
        if idx < n_buy:
            order_id = buy_ids.pop(idx)
            if not buy_ids:
                del self.orders_at[buy_key]
        else:
            order_id = sell_ids.pop(idx - n_buy)
            if not sell_ids:
                del self.orders_at[sell_key]
        del self.resting[order_id]
        self.kernel.send_message(
            self.my_id, self.exchange_id, CancelOrderMsg(self.my_id, order_id)
        )

    def _volume(self, level: int) -> float:
        return sum(
            self.book.size_at(side, self._usd(self._ticks(side, level)))
            for side in (Side.BUY, Side.SELL)
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
