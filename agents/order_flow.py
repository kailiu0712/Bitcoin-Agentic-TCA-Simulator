import numpy as np
from collections import deque
from market.orders import Order


class OrderFlowAgent:
    def __init__(self, cfg, rng, agent_id="order_flow"):
        self.cfg, self.rng, self.agent_id = cfg, rng, agent_id
        self.sign = 1
        self._next_id = 10_000_000
        self._excitation = 0.0
        self._last_intensity_time = 0.0
        self._sign_history = deque(maxlen=int(cfg.get("order_flow_long_memory_max_lag", 2000)))
        components = int(cfg.get("order_flow_memory_components", 8))
        self._memory_components = self.rng.choice(np.array([-1, 1]), size=max(2, components))
        alpha = float(cfg.get("order_flow_long_memory_exponent", 0.54))
        rates = np.logspace(-3.5, -0.25, len(self._memory_components))
        self._memory_flip_rates = np.clip(rates ** max(0.2, alpha), 1e-4, 0.45)

    def start(self, kernel):
        self._schedule(kernel)

    def _schedule(self, kernel):
        rate = self.cfg["market_order_arrival_rate"]
        # A stationary Hawkes process is the default arrival model.  Its
        # branching ratio controls the observed over-dispersion while the
        # baseline is adjusted so that the long-run mean remains `rate`.
        branching = float(self.cfg.get("order_flow_hawkes_branching", 0.0))
        decay = float(self.cfg.get("order_flow_hawkes_decay", 1.0))
        if branching > 0.0:
            branching = min(branching, 0.95)
            baseline = rate * (1.0 - branching)
            elapsed = max(0.0, kernel.time - self._last_intensity_time)
            self._excitation *= np.exp(-decay * elapsed)
            self._last_intensity_time = kernel.time
            intensity = max(1e-9, baseline + self._excitation)
            wait = self.rng.exponential(1.0 / intensity)
            self._excitation += branching * decay
            kernel.schedule(kernel.time + wait, self.wake)
            return
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
        memory_p = float(self.cfg.get("order_flow_long_memory_probability", 0.0))
        if memory_p > 0.0:
            flips = self.rng.random(len(self._memory_components)) < self._memory_flip_rates
            self._memory_components[flips] *= -1
        if memory_p > 0.0 and self.rng.random() < memory_p:
            observed_sign = 1 if np.sum(self._memory_components) >= 0 else -1
        else:
            observed_sign = self.sign if self.rng.random() < self.cfg["trade_sign_persistence"] else -self.sign
        self._sign_history.append(observed_sign)
        qty = self.rng.lognormal(np.log(self.cfg["trade_size_median"]), self.cfg["trade_size_sigma"])
        # A small population of unusually large taker orders is essential for
        # the P0 size tail and for realistic queue depletion.  The switch is
        # explicit so experiments can disable it without changing the model.
        tail_p = float(self.cfg.get("large_order_probability", 0.0))
        if tail_p > 0.0 and self.rng.random() < tail_p:
            qty *= float(self.cfg.get("large_order_multiplier", 1.0)) * self.rng.pareto(float(self.cfg.get("large_order_pareto", 2.0)) + 1.0)
        oid = self._next_id; self._next_id += 1
        kernel.exchange.submit(Order(oid, self.agent_id, "BUY" if observed_sign > 0 else "SELL", qty, kernel.time))
        self._schedule(kernel)
