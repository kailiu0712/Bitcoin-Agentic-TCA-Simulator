import numpy as np
from market.orders import Order


class ExecutionAgent:
    def __init__(self,side,total_quantity,start_time,end_time,n_slices,agent_id):
        self.side=side.upper();self.sign=1 if self.side=="BUY" else -1
        self.total_quantity=float(total_quantity);self.remaining=float(total_quantity)
        self.start_time=float(start_time);self.end_time=float(end_time);self.n_slices=int(n_slices);self.agent_id=agent_id
        self.next_order_id=50_000_000;self.arrival_mid=None;self.filled=0.0;self.notional=0.0;self.records=[]

    def start(self,kernel):
        for i,t in enumerate(np.linspace(self.start_time,self.end_time,self.n_slices)):
            kernel.schedule(float(t),lambda k,i=i:self.wake(k,i))

    def desired_quantity(self,kernel,step):
        return self.remaining/max(1,self.n_slices-step)

    def wake(self,kernel,step):
        state=kernel.exchange.book.state()
        if self.arrival_mid is None:self.arrival_mid=state["mid"]
        if self.remaining<=1e-12:return
        requested=min(self.remaining,max(0.0,self.desired_quantity(kernel,step)))
        if requested<=0:return
        oid=self.next_order_id;self.next_order_id+=1;before=len(kernel.exchange.trades)
        kernel.exchange.submit(Order(oid,self.agent_id,self.side,requested,kernel.time))
        fills=[t for t in kernel.exchange.trades[before:] if t["buyer_agent"]==self.agent_id or t["seller_agent"]==self.agent_id]
        filled=sum(t["size"] for t in fills);notional=sum(t["size"]*t["price"] for t in fills)
        self.filled+=filled;self.remaining=max(0.0,self.total_quantity-self.filled);self.notional+=notional
        after=kernel.exchange.book.state()
        self.records.append({"timestamp":kernel.time,"step":step,"requested":requested,"filled":filled,"fill_notional":notional,
            "cumulative_filled":self.filled,"remaining":self.remaining,"progress":self.filled/self.total_quantity,
            "mid":after["mid"] if after["mid"] is not None else state["mid"],"best_bid":after["best_bid"],"best_ask":after["best_ask"]})

    @property
    def average_fill_price(self):return self.notional/self.filled if self.filled else None


class TWAPLiquidationAgent(ExecutionAgent):
    pass


class ImmediateExecutionAgent(ExecutionAgent):
    """Worst-case urgency benchmark: submit the entire parent order at arrival."""
    def start(self,kernel):kernel.schedule(self.start_time,lambda k:self.wake(k,0))
    def desired_quantity(self,kernel,step):return self.remaining


class DepthParticipationAgent(ExecutionAgent):
    def __init__(self,*args,participation=0.5,**kwargs):super().__init__(*args,**kwargs);self.participation=participation
    def desired_quantity(self,kernel,step):
        state=kernel.exchange.book.state();depth=state["ask_size"] if self.side=="BUY" else state["bid_size"]
        schedule=self.remaining/max(1,self.n_slices-step)
        return min(self.remaining,max(schedule*.25,self.participation*(depth or 0)))


class FrontLoadedExecutionAgent(ExecutionAgent):
    """Deterministic urgency schedule analogous to a simple AC trajectory."""
    def __init__(self,*args,decay=2.0,**kwargs):
        super().__init__(*args,**kwargs);w=np.exp(-decay*np.linspace(0,1,self.n_slices));self.weights=w/w.sum()
    def desired_quantity(self,kernel,step):
        return self.remaining if step==self.n_slices-1 else min(self.remaining,self.total_quantity*self.weights[step])


class LiquidityAdaptiveExecutionAgent(ExecutionAgent):
    """Interpretable schedule using urgency, spread, imbalance and displayed depth."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.profile_decay=None
        self.profile_weights=None

    def _schedule_profile(self,kernel):
        s=kernel.exchange.book.state()
        depth=(s["ask_size"] if self.side=="BUY" else s["bid_size"]) or 0.0
        ratio=self.total_quantity/max(depth,1e-9)
        spread=s["spread"] or 0.0
        # Use only observable liquidity regime. Thin books keep the online
        # liquidity-seeking policy; ordinary and large parents are mildly or
        # strongly front-loaded to reduce exposure to later price impact.
        self.profile_decay=0.0 if ratio>=18 else (2.0 if ratio>=8 else 1.0)
        weights=np.exp(-self.profile_decay*np.linspace(0,1,self.n_slices))
        self.profile_weights=weights/weights.sum()

    def desired_quantity(self,kernel,step):
        if step==self.n_slices-1:return self.remaining
        if self.profile_weights is None:self._schedule_profile(kernel)
        s=kernel.exchange.book.state();schedule=self.remaining/max(1,self.n_slices-step)
        bid,ask=s["bid_size"] or 0.0,s["ask_size"] or 0.0;depth=ask if self.side=="BUY" else bid
        imbalance=(bid-ask)/(bid+ask+1e-9);adverse_pressure=self.sign*imbalance
        mid=s.get("mid"); favorable_move=0.0 if self.arrival_mid is None or mid is None else self.sign*(mid-self.arrival_mid)
        time_fraction=step/max(1,self.n_slices-1);urgency=.65+1.1*time_fraction
        spread_multiplier=.65 if (s["spread"] or 0)>.1 and time_fraction<.8 else 1.0
        # Adverse queue pressure means the next move is more likely to be
        # against the liquidation. Slow down there and spend the schedule
        # when pressure is favorable; the previous sign did the opposite.
        signal_multiplier=float(np.clip(1-.6*adverse_pressure+.35*np.tanh(favorable_move),.55,1.45))
        online=min(self.remaining,max(.25*schedule,min(.8*depth,urgency*signal_multiplier*spread_multiplier*schedule)))
        planned=self.total_quantity*self.profile_weights[step]
        profile_multiplier=float(np.clip(planned/max(schedule,1e-9),.70,1.40))
        return min(self.remaining,online*profile_multiplier)


class TabularRLExecutionAgent(ExecutionAgent):
    ACTIONS=(.5,1.0,2.0)
    def __init__(self,*args,q_table=None,epsilon=0.0,rng=None,**kwargs):
        super().__init__(*args,**kwargs);self.q_table={} if q_table is None else q_table;self.epsilon=epsilon;self.rng=rng or np.random.default_rng(0);self.actions=[]
    def state_key(self,kernel,step):
        s=kernel.exchange.book.state();imb=((s["bid_size"] or 0)-(s["ask_size"] or 0))/((s["bid_size"] or 0)+(s["ask_size"] or 0)+1e-9)
        return (min(3,int(4*step/max(1,self.n_slices))),min(3,int(4*self.remaining/self.total_quantity)),int((s["spread"] or 0)>.1),int(np.clip(np.floor((imb+1)*2),0,3)))
    def desired_quantity(self,kernel,step):
        state=self.state_key(kernel,step);values=self.q_table.get(state,[0.0]*len(self.ACTIONS))
        action=int(self.rng.integers(len(self.ACTIONS))) if self.rng.random()<self.epsilon else int(np.argmax(values));self.actions.append((state,action))
        return min(self.remaining,self.remaining/max(1,self.n_slices-step)*self.ACTIONS[action])


class EnhancedRLExecutionAgent(ExecutionAgent):
    """Tabular policy whose state maps directly to an executable child-order instruction."""
    ACTIONS=(.5,.75,1.0,1.5,2.0)
    def __init__(self,*args,q_table=None,epsilon=0.0,rng=None,**kwargs):
        super().__init__(*args,**kwargs);self.q_table={} if q_table is None else q_table;self.epsilon=epsilon;self.rng=rng or np.random.default_rng(0);self.actions=[]
    def observe(self,kernel,step):
        s=kernel.exchange.book.state();bid,ask=s["bid_size"] or 0.0,s["ask_size"] or 0.0
        imbalance=(bid-ask)/(bid+ask+1e-9);side_imbalance=self.sign*imbalance
        schedule=self.remaining/max(1,self.n_slices-step);depth=ask if self.side=="BUY" else bid
        mid=s["mid"] if s["mid"] is not None else self.arrival_mid;momentum=0.0 if not self.arrival_mid or mid is None else self.sign*(mid-self.arrival_mid)/self.arrival_mid
        return {"time_fraction":step/max(1,self.n_slices-1),"inventory_fraction":self.remaining/self.total_quantity,
            "spread":s["spread"] or 0.0,"side_adjusted_imbalance":side_imbalance,"opposite_depth_ratio":depth/(schedule+1e-12),"side_adjusted_momentum":momentum}
    @staticmethod
    def encode(obs):
        return (min(3,int(4*obs["time_fraction"])),min(2,int(3*obs["inventory_fraction"])),
            int(obs["spread"]>.1),min(2,max(0,int(np.floor((obs["side_adjusted_imbalance"]+1)*1.5)))),
            int(np.digitize(obs["opposite_depth_ratio"],[.5,1.5,3.])),int(obs["side_adjusted_momentum"]>0))
    @classmethod
    def instruction_from_observation(cls,obs,remaining,steps_remaining,q_table,is_final=False):
        state=cls.encode(obs);values=q_table.get(state,[0.0]*len(cls.ACTIONS))
        action=cls.ACTIONS.index(1.0) if np.ptp(values)<1e-15 else int(np.argmax(values));multiplier=cls.ACTIONS[action]
        schedule=remaining/max(1,steps_remaining);quantity=remaining if is_final else min(remaining,schedule*multiplier)
        return {"state":state,"features":obs,"action":action,"multiplier":multiplier,"quantity":quantity}
    def instruction(self,kernel,step):
        obs=self.observe(kernel,step);state=self.encode(obs);values=self.q_table.get(state,[0.0]*len(self.ACTIONS))
        if self.rng.random()>=self.epsilon:return self.instruction_from_observation(obs,self.remaining,self.n_slices-step,self.q_table,step==self.n_slices-1)
        action=int(self.rng.integers(len(self.ACTIONS)));multiplier=self.ACTIONS[action];schedule=self.remaining/max(1,self.n_slices-step);quantity=self.remaining if step==self.n_slices-1 else min(self.remaining,schedule*multiplier)
        return {"state":state,"features":obs,"action":action,"multiplier":multiplier,"quantity":quantity}
    def desired_quantity(self,kernel,step):
        instruction=self.instruction(kernel,step);self.actions.append((instruction["state"],instruction["action"]));return instruction["quantity"]
