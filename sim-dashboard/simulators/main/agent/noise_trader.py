import datetime as dt
import itertools
from typing import Any

import numpy as np

from exchange import MarketOrderMsg, Side
from kernel import AgentId, Kernel


class NoiseTrader:
    def __init__(
        self,
        my_id: AgentId,
        exchange_id: AgentId,
        kernel: Kernel,
        rng: np.random.Generator,
        trade_rate: float = 1.0,  # trades per second (buy / sell)
        trade_size: float = 1.0,
    ):
        self.my_id = my_id
        self.exchange_id = exchange_id
        self.kernel = kernel
        self.rng = rng
        self.trade_rate = trade_rate
        self.trade_size = trade_size
        self._order_seq = itertools.count(1)

    def kernel_start(self, start_time: dt.datetime) -> None:
        self._schedule_next_wakeup(start_time)

    def kernel_stop(self, stop_time: dt.datetime) -> None:
        pass

    def on_wakeup(self, current_time: dt.datetime) -> None:
        side = Side.BUY if self.rng.random() < 0.5 else Side.SELL
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
        if self.trade_rate <= 0:
            return
        dt_sec = float(self.rng.exponential(1.0 / (2.0 * self.trade_rate)))
        self.kernel.set_wakeup(self.my_id, now + dt.timedelta(seconds=dt_sec))
