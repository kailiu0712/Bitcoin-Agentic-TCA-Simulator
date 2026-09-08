import json,threading,traceback,uuid
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
import yaml
import numpy as np
from liquidation.runner import make_agent,midpoint_from_path,midpoint_path,run_market

# The tuned PDF-priority parameters are now versioned in the default config;
# calibration artifacts are reports, not a runtime dependency.
# Bumped whenever the API contract changes. `app/run_app.py` and the frontend both
# compare against it so a stale server left running on the port is reported as
# such instead of silently answering a newer UI with an older contract.
BUILD_ID="impact-simulator-v4-adaptive"
CONFIG_PATH=ROOT/"config/default.yaml"
EVIDENCE_DIR=ROOT/"outputs/latest_p0p1"
STATIC=Path(__file__).resolve().parent/"static"
JOBS={};LOCK=threading.Lock()

# Registry of liquidation algorithms exposed by the impact simulator. Any policy
# that `liquidation.runner.make_agent` can build is pluggable: add its key here
# and it is simulated, measured and charted with no other change. The shipped
# panel keeps the two reference benchmarks so the reader compares algorithms
# against a common yardstick rather than against a tuned in-house policy.
ALGORITHMS=(
    {"key":"immediate","label":"Immediate","summary":"母单在到达时一次性市价成交，代表最激进的执行方式。","role":"urgency benchmark"},
    {"key":"twap","label":"TWAP","summary":"按剩余时间均匀拆分子单，是行业标准的中性执行基准。","role":"schedule benchmark"},
    {"key":"adaptive","label":"Adaptive","summary":"根据剩余时间、盘口深度、价差与不平衡动态调整子单节奏。","role":"liquidity-aware policy"},
)
REFERENCE_ALGORITHM="twap"


class SimulationRequest(BaseModel):
    side:str=Field(pattern="^(BUY|SELL)$")
    parent_quantity:float=Field(gt=0,le=100)
    duration_seconds:int=Field(ge=30,le=3600)
    volatility:float=Field(ge=0,le=100,description="Reference-price USD volatility per square-root second")
    spread:float=Field(gt=0,le=100)
    arrival_rate:float=Field(gt=0,le=20)
    displayed_depth:float=Field(gt=0,le=100)
    seed:int=Field(ge=0,le=2_147_483_647)


def _set_job(job_id,**values):
    with LOCK:JOBS[job_id].update(values)


def _step_sample(times,values,grid):
    """Zero-order hold: a midpoint holds until the next book event.

    Linear interpolation is wrong here and not harmlessly so. The control market
    and the execution market emit events at different instants (the execution
    market has extra MARKET_EXECUTION events), so a linear fill between two
    samples straddling a background price move puts the two series at different
    points of the same jump. That produced transient spikes in the paired
    difference as large as the background move itself - up to 0.38 bp - which
    are pure sampling artifact and are present even for a parent order too small
    to move any price. Sampling both series as step functions removes them and
    matches how per-fill costs are already looked up.
    """
    index=np.clip(np.searchsorted(times,grid,side="right")-1,0,len(times)-1)
    return values[index]


def _causal_impact_bp(sign,arrival,mid,baseline_mid):
    if mid is None or baseline_mid is None:return None
    return sign*(mid-baseline_mid)/arrival*1e4


def _impact_summary(trajectory,duration,reaction_seconds):
    """Peak, terminal and post-trade impact read off the paired control grid.

    These are the three shapes the market-impact literature reports for a
    metaorder: how far the price is pushed, where it stands when the parent is
    finished, and how much of that push survives after the child orders stop.
    """
    grid=[(row["elapsed_seconds"],row["impact_bp"]) for row in trajectory
        if row.get("impact_bp") is not None and (row.get("filled") or 0)==0]
    if not grid:return {"peak_impact_bp":0.0,"terminal_impact_bp":0.0,"post_trade_impact_bp":0.0,"decay_ratio":None}
    # Prices are tick-quantised, so a fully recovered midpoint lands on an exact
    # zero that floating point can sign as -0.0; clamp it so the reported
    # retention ratio reads 0% rather than -0%.
    zero=lambda value:0.0 if abs(value)<1e-9 else value
    during=[value for elapsed,value in grid if elapsed<=duration]
    peak=zero(max(during,key=abs) if during else 0.0)
    terminal=zero(min(((abs(elapsed-duration),value) for elapsed,value in grid),key=lambda item:item[0])[1])
    settle=duration+reaction_seconds
    post=zero(min(((abs(elapsed-settle),value) for elapsed,value in grid),key=lambda item:item[0])[1])
    return {"peak_impact_bp":peak,"terminal_impact_bp":terminal,"post_trade_impact_bp":post,
        "decay_ratio":(post/peak) if abs(peak)>1e-12 else None}


def run_execution_simulation(req,progress=lambda value,message:None):
    cfg=dict(yaml.safe_load(CONFIG_PATH.read_text())["simulation"])
    # Keep the algorithm benchmark comparable across runs; the queue-response
    # calibration is a market-generation control and is not part of the
    # client-side liquidation-algorithm comparison.
    cfg["market_maker_queue_imbalance_response"]=0.0
    cfg.update(reference_price_volatility=req.volatility,market_maker_target_spread=req.spread,
        market_order_arrival_rate=req.arrival_rate,market_maker_quote_size=req.displayed_depth)
    warmup=60.;post=30.;end=warmup+req.duration_seconds;total=end+post;n_slices=max(4,int(req.duration_seconds/10))
    # The post-trade window is what makes decay measurable: keep measuring for a
    # full 30 seconds after the parent finishes, not just one quote refresh.
    settle_seconds=post
    progress(10,"生成同种子无执行对照市场")
    baseline,_=run_market(cfg,req.seed,total);base_path=midpoint_path(baseline.events)
    sign=1 if req.side=="BUY" else -1;comparisons={};execution_paths={};trajectories={"no_execution":[]}
    baseline_rows=[]
    bts,bmid=base_path
    baseline_arrival=midpoint_from_path(base_path,warmup)
    baseline_ix=[ix for ix in np.linspace(0,max(0,len(bts)-1),min(400,len(bts))).astype(int) if warmup<=bts[ix]<=end+settle_seconds] if len(bts) else []
    for ix in baseline_ix:
        control_change=0.0 if baseline_arrival in (None,0) else sign*(bmid[ix]-baseline_arrival)/baseline_arrival*1e4
        baseline_rows.append({"elapsed_seconds":float(bts[ix]-warmup),"impact_bp":0.0,"mid_change_bp":control_change,"executed_fraction":0.0,"filled":0.0,"fill_price":None})
    trajectories["no_execution"]=baseline_rows
    span=100//max(1,len(ALGORITHMS))
    for index,algorithm in enumerate(ALGORITHMS):
        strategy=algorithm["key"]
        progress(25+index*span,f"测量 {algorithm['label']} 的因果冲击")
        agent=make_agent(strategy,req.side,req.parent_quantity,warmup,end,n_slices,seed=req.seed)
        exchange,agent=run_market(cfg,req.seed,total,agent);arrival=agent.arrival_mid
        if arrival is None:raise RuntimeError("market did not establish a valid arrival midpoint")
        paired_cost=0.0;strategy_path=[]
        for record in agent.records:
            baseline_mid=midpoint_from_path(base_path,record["timestamp"]);fill_price=record["fill_notional"]/record["filled"] if record["filled"] else None
            if record["filled"] and baseline_mid is not None:paired_cost+=sign*(record["fill_notional"]-baseline_mid*record["filled"])/(arrival*req.parent_quantity)
            causal_impact_bp=_causal_impact_bp(sign,arrival,record["mid"],baseline_mid)
            strategy_path.append({"elapsed_seconds":round(record["timestamp"]-warmup,6),"requested":record["requested"],"filled":record["filled"],
                "cumulative_filled":record["cumulative_filled"],"remaining":record["remaining"],"executed_fraction":record["progress"],"fill_price":fill_price,
                "mid":record["mid"],"baseline_mid":baseline_mid,"causal_impact_bp":causal_impact_bp,
                "mid_change_bp":sign*(record["mid"]-agent.arrival_mid)/agent.arrival_mid*1e4})
        # Compare every execution run with its same-seeded control on a common
        # regular time grid. This is a true paired midpoint time series rather
        # than an asynchronous cloud of quote events.
        strategy_ts,strategy_mid=midpoint_path(exchange.events)
        trajectory_path=[]
        if len(strategy_ts) and len(bts):
            grid=np.linspace(0.0,req.duration_seconds+settle_seconds,
                min(600,max(2,int(req.duration_seconds+settle_seconds)+1)))
            abs_grid=warmup+grid
            control_mid=_step_sample(bts,bmid,abs_grid)
            executed_mid=_step_sample(strategy_ts,strategy_mid,abs_grid)
            for elapsed,mid_value,control_value in zip(grid,executed_mid,control_mid):
                trajectory_path.append({"elapsed_seconds":round(float(elapsed),6),
                    "impact_bp":_causal_impact_bp(sign,arrival,float(mid_value),float(control_value)),
                    "mid_change_bp":sign*(float(mid_value)-arrival)/arrival*1e4,
                    "executed_fraction":0.0,"filled":0.0,"fill_price":None})
        impact=_impact_summary(trajectory_path,req.duration_seconds,settle_seconds)
        # Add the actual fills after constructing the line so the frontend can
        # render them as execution markers without distorting the time series.
        trajectory_path.extend(strategy_path)
        trajectory_path.sort(key=lambda row:row["elapsed_seconds"])
        completion=agent.filled/req.parent_quantity;penalized=paired_cost+.0005*(1-completion)
        comparisons[strategy]={"key":strategy,"label":algorithm["label"],"role":algorithm["role"],"summary":algorithm["summary"],
            "impact_bp":impact["peak_impact_bp"],"terminal_impact_bp":impact["terminal_impact_bp"],
            "post_trade_impact_bp":impact["post_trade_impact_bp"],"decay_ratio":impact["decay_ratio"],
            "execution_cost_bp":paired_cost*1e4,"penalized_cost_bp":penalized*1e4,"completion":completion,
            "average_fill_price":agent.average_fill_price,"filled_quantity":agent.filled,
            "execution_notional":agent.notional,"instructions":len(strategy_path)}
        trajectories[strategy]=trajectory_path;execution_paths[strategy]=strategy_path
    progress(96,"汇总冲击与成本指标")
    reference=comparisons[REFERENCE_ALGORITHM];path=execution_paths[REFERENCE_ALGORITHM]
    return {"request":req.model_dump(),"algorithms":[dict(item) for item in ALGORITHMS],
        "summary":{"strategy":REFERENCE_ALGORITHM,"arrival_mid":arrival,"average_fill_price":reference["average_fill_price"],
            "filled_quantity":path[-1]["cumulative_filled"] if path else 0,"completion":reference["completion"],
            "paired_cost_bp":reference["execution_cost_bp"],"penalized_cost_bp":reference["penalized_cost_bp"],
            "peak_impact_bp":reference["impact_bp"],"instructions":reference["instructions"]},
        "comparison":comparisons,"execution_path":path,"execution_paths":execution_paths,"trajectories":trajectories}


def _worker(job_id,req):
    try:
        result=run_execution_simulation(req,lambda p,m:_set_job(job_id,progress=p,message=m));_set_job(job_id,status="complete",progress=100,message="完成",result=result)
    except Exception as exc:_set_job(job_id,status="failed",message=str(exc),error=traceback.format_exc())


def load_stylized_facts():
    """Serve the held-out calibration evidence that backs every impact number.

    The panel is the product claim: the market the algorithm trades against is
    not decorative, it reproduces the microstructure facts that determine
    execution cost. Values come from the versioned validation artifacts.
    """
    facts=json.loads((EVIDENCE_DIR/"stylized_facts.json").read_text(encoding="utf-8"))
    return facts


app=FastAPI(title="Stylized-Fact Calibrated Liquidation Impact Simulator",version="2.0.0")
app.mount("/static",StaticFiles(directory=STATIC),name="static")

@app.get("/",include_in_schema=False)
def index():return FileResponse(STATIC/"index.html")

@app.get("/api/health")
def health():return {"status":"ok","config_available":CONFIG_PATH.exists(),
    "algorithms":[item["key"] for item in ALGORITHMS],"build_id":BUILD_ID}

@app.get("/api/algorithms")
def algorithms():return {"algorithms":[dict(item) for item in ALGORITHMS],"reference":REFERENCE_ALGORITHM}

@app.get("/api/stylized-facts")
def stylized_facts():
    try:return load_stylized_facts()
    except FileNotFoundError:raise HTTPException(503,"stylized-fact evidence has not been generated")

@app.post("/api/jobs",status_code=202)
def create_job(req:SimulationRequest):
    job_id=uuid.uuid4().hex
    with LOCK:JOBS[job_id]={"id":job_id,"status":"queued","progress":0,"message":"排队中"}
    threading.Thread(target=_worker,args=(job_id,req),daemon=True).start();return JOBS[job_id]

@app.get("/api/jobs/{job_id}")
def get_job(job_id:str):
    with LOCK:job=JOBS.get(job_id);snapshot=None if job is None else dict(job)
    if snapshot is None:raise HTTPException(404,"unknown job")
    snapshot.pop("error",None);return snapshot
