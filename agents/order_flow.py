import numpy as np
from market.orders import Order


class OrderFlowAgent:
    def __init__(self, cfg, rng, agent_id="order_flow"):
        self.cfg, self.rng, self.agent_id = cfg, rng, agent_id
        self.sign = 1
        self._next_id = 10_000_000

    def start(self, kernel):
        self._schedule(kernel)

    def _schedule(self, kernel):
        rate = self.cfg["market_order_arrival_rate"]
        burst_probability=self.cfg.get("order_flow_burst_probability",0.0)
        burst_mean=self.cfg.get("order_flow_burst_interval_seconds",.01)
        # Preserve E[interarrival] = 1/rate while allowing a mixture with a
        # short-interval burst component. Previously `rate` was not the actual
        # arrival rate, making calibration poorly identified.
        regular_mean=max(burst_mean,(1/rate-burst_probability*burst_mean)/max(1e-9,1-burst_probability))
        wait=self.rng.exponential(burst_mean if self.rng.random()<burst_probability else regular_mean)
        kernel.schedule(kernel.time+wait,self.wake)

    def wake(self, kernel):
        if self.rng.random() < self.cfg.get("sign_regime_switch_probability", 0.0):
            self.sign *= -1
        observed_sign = self.sign if self.rng.random() < self.cfg["trade_sign_persistence"] else -self.sign
        qty = self.rng.lognormal(np.log(self.cfg["trade_size_median"]), self.cfg["trade_size_sigma"])
        oid = self._next_id; self._next_id += 1
        kernel.exchange.submit(Order(oid, self.agent_id, "BUY" if observed_sign > 0 else "SELL", qty, kernel.time))
        self._schedule(kernel)
