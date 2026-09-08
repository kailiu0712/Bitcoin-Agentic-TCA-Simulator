import numpy as np
from market.orders import Order


class MarketMakerAgent:
    def __init__(self, cfg, rng, agent_id="market_maker"):
        self.cfg, self.rng, self.agent_id = cfg, rng, agent_id
        self.inventory, self.cash, self.order_ids = 0.0, 0.0, []
        self.inventory_signal, self.last_inventory = 0.0, 0.0
        self.flow_signal,self.last_trade_index=0.0,0
        self.reference = float(cfg["initial_price"])
        self._next_id = 1

    def start(self, kernel):
        kernel.schedule(0.0, self.wake)

    def wake(self, kernel):
        ex = kernel.exchange
        # Measure the live queue before this maker refreshes its own quotes;
        # measuring after cancellation would make the imbalance identically 0.
        pre_state = ex.book.state()
        for oid in self.order_ids:
            ex.cancel(kernel.time, self.agent_id, oid)
        self.order_ids.clear()
        dt = self.cfg["market_maker_refresh_seconds"]
        state = ex.book.state()
        center = state["mid"] if state["mid"] is not None else self.reference
        # Quote imbalance is an endogenous OFI-to-price channel.  It gives
        # book events predictive power instead of making price movement depend
        # only on the maker's exogenous reference innovation (P1 Fact 8).
        bid_size = pre_state.get("bid_size", 0.0)
        ask_size = pre_state.get("ask_size", 0.0)
        imbalance = (bid_size - ask_size) / max(1e-12, bid_size + ask_size)
        center += self.cfg.get("market_maker_queue_imbalance_response", 0.0) * imbalance
        center += self.rng.normal(0, self.cfg["reference_price_volatility"] * np.sqrt(dt))
        self.reference = center
        self.inventory = ex.inventory[self.agent_id]; self.cash = ex.cash[self.agent_id]
        delta_inventory=self.inventory-self.last_inventory;self.last_inventory=self.inventory
        decay=np.exp(-self.cfg.get("market_maker_inventory_signal_decay",0.5)*dt)
        self.inventory_signal=self.inventory_signal*decay+delta_inventory
        center -= self.cfg["market_maker_inventory_skew"] * self.inventory_signal
        new_trades=ex.trades[self.last_trade_index:];self.last_trade_index=len(ex.trades)
        # An anonymous concave response: experimental and background orders
        # affect liquidity through exactly the same mechanism.
        flow=sum((1 if t["aggressor_side"]=="BUY" else -1)*np.sqrt(t["size"]) for t in new_trades)
        flow_decay=np.exp(-self.cfg.get("market_maker_flow_signal_decay",1.0)*dt)
        self.flow_signal=self.flow_signal*flow_decay+flow
        center += self.cfg.get("market_maker_flow_response",0.0)*self.flow_signal
        tick = self.cfg["tick_size"]
        spread_pressure=self.cfg.get("market_maker_spread_sensitivity",0.0)*abs(self.flow_signal)
        spread_ticks = max(1, int(round((self.cfg["market_maker_target_spread"]+spread_pressure) / tick)))
        bid = np.floor((center - spread_ticks * tick / 2) / tick) * tick
        ask = bid + spread_ticks * tick
        levels=int(self.cfg.get("market_maker_levels",1));spacing=int(self.cfg.get("market_maker_level_spacing_ticks",1))
        sigma=self.cfg.get("market_maker_quote_size_sigma",0.0);median=self.cfg["market_maker_quote_size"]
        for level in range(levels):
            for side, price in (("BUY", bid-level*spacing*tick),("SELL",ask+level*spacing*tick)):
                oid=self._next_id;self._next_id+=1;qty=median if sigma<=0 else self.rng.lognormal(np.log(median),sigma)
                # Visible depth should grow away from the touch and then
                # taper at the edge of the quoting range (P0 Fact 20).
                slope=float(self.cfg.get("market_maker_depth_slope",0.35))
                peak=float(self.cfg.get("market_maker_depth_peak",4.0))
                collapse=float(self.cfg.get("market_maker_depth_collapse",0.08))
                qty *= (1.0 + slope * level) * np.exp(-collapse * max(0.0, level - peak) ** 2)
                # Predictable buy flow attracts ask liquidity (and vice versa),
                # an interpretable asymmetric-dynamic-liquidity response.
                pressure=np.clip(self.flow_signal,-3,3);same_side=1 if side=="SELL" else -1
                qty*=np.exp(self.cfg.get("market_maker_asymmetric_liquidity",0.0)*same_side*pressure)
                order=Order(oid,self.agent_id,side,float(qty),kernel.time,float(price));before=len(ex.trades);ex.submit(order)
                for t in ex.trades[before:]:
                    signed=t["size"] if t["buyer_agent"]==self.agent_id else -t["size"]
                    self.inventory+=signed;self.cash-=signed*t["price"]
                if order.remaining>1e-12:self.order_ids.append(oid)
        ex.record_quote(kernel.time,self.agent_id)
        kernel.schedule(kernel.time + dt, self.wake)
