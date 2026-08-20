import numpy as np
from agents.market_maker import MarketMakerAgent
from agents.order_flow import OrderFlowAgent
from market.exchange import Exchange
from market.kernel import Kernel
from liquidation.agents import TWAPLiquidationAgent,ImmediateExecutionAgent,DepthParticipationAgent,FrontLoadedExecutionAgent,LiquidityAdaptiveExecutionAgent,TabularRLExecutionAgent,EnhancedRLExecutionAgent


def midpoint_at(events,time,direction="before"):
    # Post-trade exchange events carry the immediate book state. Restricting
    # this lookup to periodic quote refreshes hides a market order's footprint.
    rows=[e for e in events if (e.get("event_type")=="QUOTE_UPDATE" or str(e.get("event_type","")).endswith("EXECUTION")) and e.get("mid") is not None and (e["timestamp"]<=time if direction=="before" else e["timestamp"]>=time)]
    if not rows:return None
    return (rows[-1] if direction=="before" else rows[0])["mid"]


def midpoint_path(events):
    rows=[(e["timestamp"],e["mid"]) for e in events if (e.get("event_type")=="QUOTE_UPDATE" or str(e.get("event_type","")).endswith("EXECUTION")) and e.get("mid") is not None]
    if not rows:return np.array([]),np.array([])
    rows.sort(key=lambda x:x[0]);return np.asarray([x[0] for x in rows]),np.asarray([x[1] for x in rows])


def midpoint_from_path(path,time,direction="before"):
    ts,mid=path
    if not len(ts):return None
    ix=np.searchsorted(ts,time,side="right")-1 if direction=="before" else np.searchsorted(ts,time,side="left")
    return None if ix<0 or ix>=len(ts) else float(mid[ix])


def run_market(cfg,seed,total_duration,liquidation=None):
    seq=np.random.SeedSequence(seed);mm_seed,of_seed,liq_seed=seq.spawn(3);ex=Exchange();kernel=Kernel(ex,seed)
    mm=MarketMakerAgent(cfg,np.random.default_rng(mm_seed));flow=OrderFlowAgent(cfg,np.random.default_rng(of_seed));mm.start(kernel);flow.start(kernel)
    if liquidation:
        liquidation.rng=np.random.default_rng(liq_seed) if hasattr(liquidation,"rng") else None
        liquidation.start(kernel)
    kernel.run(total_duration);return ex,liquidation


def make_agent(strategy,side,quantity,start,end,n_slices,q_table=None,seed=0):
    kw=dict(side=side,total_quantity=quantity,start_time=start,end_time=end,n_slices=n_slices,agent_id=f"liquidation_{strategy}")
    if strategy=="twap":return TWAPLiquidationAgent(**kw)
    if strategy=="immediate":return ImmediateExecutionAgent(**kw)
    if strategy=="pov":return DepthParticipationAgent(**kw,participation=.5)
    if strategy=="frontload":return FrontLoadedExecutionAgent(**kw)
    if strategy=="adaptive":return LiquidityAdaptiveExecutionAgent(**kw)
    if strategy=="rl":return TabularRLExecutionAgent(**kw,q_table=q_table,rng=np.random.default_rng(seed))
    if strategy=="rl2":return EnhancedRLExecutionAgent(**kw,q_table=q_table,rng=np.random.default_rng(seed))
    raise ValueError(strategy)
