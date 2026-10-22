import bisect
import datetime as dt
from typing import Optional

import numpy as np


class HistoricalOracle:
    def __init__(
        self,
        ref_times: list[float] | np.ndarray,
        ref_prices: list[float] | np.ndarray,
    ):
        self.ref_times = np.asarray(ref_times, dtype=float)
        self.ref_prices = np.asarray(ref_prices, dtype=float)
        self._start_time: Optional[dt.datetime] = None

    def initialize(self, start_time: dt.datetime) -> None:
        self._start_time = start_time

    def get_price(self, current_time: dt.datetime) -> float:
        if self._start_time is None:
            self._start_time = current_time
        t = (current_time - self._start_time).total_seconds()
        idx = bisect.bisect_right(self.ref_times, t) - 1
        if idx < 0:
            idx = 0
        elif idx >= len(self.ref_prices):
            idx = len(self.ref_prices) - 1
        return float(self.ref_prices[idx])
