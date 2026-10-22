import datetime as dt
import itertools
from typing import Any, Optional, Sequence, Union

import numpy as np

from exchange import BookView, MarketOrderMsg, Side
from kernel import AgentId, Kernel


class NoiseTrader:
    def __init__(
        self,
        my_id: AgentId,
        exchange_id: AgentId,
        kernel: Kernel,
        rng: np.random.Generator,
        trade_rate: Union[float, Sequence[float], np.ndarray] = 1.0,
        trade_size: float = 1.0,
        book: Optional[BookView] = None,
        obi_cutoffs: Optional[Sequence[float]] = None,
    ):
        self.my_id = my_id
        self.exchange_id = exchange_id
        self.kernel = kernel
        self.rng = rng
        self.trade_size = trade_size
        self.book = book
        self._order_seq = itertools.count(1)

        tr = np.asarray(trade_rate, dtype=float)
        if tr.ndim == 0:
            self.trade_rates = tr.reshape(1, 1)
        elif tr.ndim == 2:
            self.trade_rates = tr
        else:
            raise ValueError(
                "trade_rate must be a scalar float or a 2D array (s_max, n_obi_bins)"
            )

        self.s_max, self.n_obi_bins = self.trade_rates.shape
        if obi_cutoffs is not None:
            self.obi_cutoffs = np.asarray(obi_cutoffs, dtype=float)
        elif self.n_obi_bins > 1:
            self.obi_cutoffs = np.linspace(-1.0, 1.0, self.n_obi_bins + 1)[1:-1]
        else:
            self.obi_cutoffs = np.empty(0, dtype=float)

        # Upper bound of total market order arrival rate (buy + sell) for thinning
        self._bound = float((self.trade_rates + self.trade_rates[:, ::-1]).max())

    def kernel_start(self, start_time: dt.datetime) -> None:
        self._schedule_next_wakeup(start_time)

    def kernel_stop(self, stop_time: dt.datetime) -> None:
        pass

    def _current_rates(self) -> tuple[float, float]:
        """Return (buy_trade_rate, sell_trade_rate) for the current orderbook state."""
        if self.book is None or (self.s_max == 1 and self.n_obi_bins == 1):
            r = float(self.trade_rates[0, 0])
            return r, r

        bid = self.book.best_bid_ticks()
        ask = self.book.best_ask_ticks()
        if bid is None or ask is None or ask <= bid:
            s_idx = 0
            i_idx = self.n_obi_bins // 2
        else:
            s_idx = min(ask - bid - 1, self.s_max - 1)
            if self.n_obi_bins <= 1:
                i_idx = 0
            else:
                v_b = self.book.size_at_ticks(Side.BUY, bid)
                v_a = self.book.size_at_ticks(Side.SELL, ask)
                tot = v_b + v_a
                if tot <= 0:
                    i_idx = self.n_obi_bins // 2
                else:
                    obi = (v_b - v_a) / tot
                    i_idx = int(np.searchsorted(self.obi_cutoffs, obi, side="right"))

        neg_i_idx = (self.n_obi_bins - 1 - i_idx) if self.n_obi_bins > 1 else 0
        return (
            float(self.trade_rates[s_idx, i_idx]),
            float(self.trade_rates[s_idx, neg_i_idx]),
        )

    def on_wakeup(self, current_time: dt.datetime) -> None:
        r_buy, r_sell = self._current_rates()
        total_rate = r_buy + r_sell
        if total_rate > 0 and self.rng.random() < (total_rate / self._bound):
            side = Side.BUY if self.rng.random() < (r_buy / total_rate) else Side.SELL
            order_id = self.my_id * 1_000_000 + next(self._order_seq)
            self.kernel.send_message(
                self.my_id,
                self.exchange_id,
                MarketOrderMsg(self.my_id, order_id, side, self.trade_size),
            )
        self._schedule_next_wakeup(current_time)

    def on_message(self, current_time: dt.datetime, msg: Any) -> None:
        pass

    def _schedule_next_wakeup(self, now: dt.datetime) -> None:
        if self._bound <= 0:
            return
        dt_sec = float(self.rng.exponential(1.0 / self._bound))
        self.kernel.set_wakeup(self.my_id, now + dt.timedelta(seconds=dt_sec))
