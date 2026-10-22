import datetime as dt
from typing import Optional

import numpy as np


class BMOracle:
    def __init__(
        self,
        rng: np.random.Generator,
        init_price: float = 100.0,
        volatility: float = 0.0,
    ):
        self.rng = rng
        self.init_price = float(init_price)
        self.price = float(init_price)
        self.volatility = float(volatility)
        self._last_time: Optional[dt.datetime] = None

    def initialize(self, start_time: dt.datetime) -> None:
        self.price = self.init_price
        self._last_time = start_time

    def get_price(self, current_time: dt.datetime) -> float:
        if self._last_time is None:
            self._last_time = current_time
            return self.price
        dt_sec = (current_time - self._last_time).total_seconds()
        if dt_sec > 0 and self.volatility > 0:
            z = self.rng.standard_normal()
            self.price += self.volatility * (dt_sec**0.5) * z
        self._last_time = current_time
        return self.price
