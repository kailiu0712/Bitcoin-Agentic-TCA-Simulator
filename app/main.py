import threading,traceback,uuid
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
import yaml
from liquidation.runner import make_agent,midpoint_from_path,midpoint_path,run_market

CONFIG_PATH=ROOT/"outputs/calibration/best_config.yaml"
STATIC=Path(__file__).resolve().parent/"static"
JOBS={};LOCK=threading.Lock()


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


def run_execution_simulation(req,progress=lambda value,message:None):
    cfg=yaml.safe_load(CONFIG_PATH.read_text())["simulation"]
    cfg.update(reference_price_volatility=req.volatility,market_maker_target_spread=req.spread,
        market_order_arrival_rate=req.arrival_rate,market_maker_quote_size=req.displayed_depth)
    warmup=60.;post=30.;end=warmup+req.duration_seconds;total=end+post;n_slices=max(4,int(req.duration_seconds/10))
    progress(10,"Generating paired control market")
    baseline,_=run_market(cfg,req.seed,total);base_path=midpoint_path(baseline.events)
    sign=1 if req.side=="BUY" else -1;comparisons={};path=[]
    for strategy,pct,message in (("immediate",35,"Estimating immediate execution"),("twap",58,"Estimating TWAP execution"),("adaptive",80,"Optimizing adaptive execution")):
        progress(pct,message);agent=make_agent(strategy,req.side,req.parent_quantity,warmup,end,n_slices,seed=req.seed);exchange,agent=run_market(cfg,req.seed,total,agent);arrival=agent.arrival_mid
        if arrival is None:raise RuntimeError("market did not establish a valid arrival midpoint")
        paired_cost=0.0;strategy_path=[]
        for record in agent.records:
            baseline_mid=midpoint_from_path(base_path,record["timestamp"]);fill_price=record["fill_notional"]/record["filled"] if record["filled"] else None
            if record["filled"] and baseline_mid is not None:paired_cost+=sign*(record["fill_notional"]-baseline_mid*record["filled"])/(arrival*req.parent_quantity)
            strategy_path.append({"elapsed_seconds":round(record["timestamp"]-warmup,6),"requested":record["requested"],"filled":record["filled"],
                "cumulative_filled":record["cumulative_filled"],"remaining":record["remaining"],"executed_fraction":record["progress"],"fill_price":fill_price,
                "mid":record["mid"],"baseline_mid":baseline_mid,"causal_impact_bp":None if baseline_mid is None else sign*(record["mid"]-baseline_mid)/arrival*1e4})
        reaction_time=end+cfg["market_maker_refresh_seconds"];causal=[]
        for event in exchange.events:
            stable=event.get("event_type")=="QUOTE_UPDATE" or str(event.get("event_type","")).endswith("EXECUTION")
            if stable and warmup<=event["timestamp"]<=reaction_time and event.get("mid") is not None:
                bm=midpoint_from_path(base_path,event["timestamp"])
                if bm is not None:causal.append(sign*(event["mid"]-bm)/arrival*1e4)
        completion=agent.filled/req.parent_quantity;penalized=paired_cost+.0005*(1-completion)
        comparisons[strategy]={"label":{"immediate":"Immediate","twap":"TWAP","adaptive":"Adaptive"}[strategy],"impact_bp":max(causal) if causal else 0.0,
            "execution_cost_bp":paired_cost*1e4,"penalized_cost_bp":penalized*1e4,"completion":completion,"average_fill_price":agent.average_fill_price,"instructions":len(strategy_path)}
        if strategy=="adaptive":path=strategy_path
    progress(96,"Finalizing result")
    adaptive=comparisons["adaptive"]
    return {"request":req.model_dump(),"summary":{"strategy":"adaptive","arrival_mid":arrival,"average_fill_price":adaptive["average_fill_price"],"filled_quantity":path[-1]["cumulative_filled"] if path else 0,
        "completion":adaptive["completion"],"paired_cost_bp":adaptive["execution_cost_bp"],"penalized_cost_bp":adaptive["penalized_cost_bp"],"instructions":adaptive["instructions"]},"comparison":comparisons,"execution_path":path}


def _worker(job_id,req):
    try:
        result=run_execution_simulation(req,lambda p,m:_set_job(job_id,progress=p,message=m));_set_job(job_id,status="complete",progress=100,message="Complete",result=result)
    except Exception as exc:_set_job(job_id,status="failed",message=str(exc),error=traceback.format_exc())


app=FastAPI(title="BTC Execution Simulator",version="1.0.0")
app.mount("/static",StaticFiles(directory=STATIC),name="static")

@app.get("/",include_in_schema=False)
def index():return FileResponse(STATIC/"index.html")

@app.get("/api/health")
def health():return {"status":"ok","config_available":CONFIG_PATH.exists(),"execution_policy":"adaptive"}

@app.post("/api/jobs",status_code=202)
def create_job(req:SimulationRequest):
    job_id=uuid.uuid4().hex
    with LOCK:JOBS[job_id]={"id":job_id,"status":"queued","progress":0,"message":"Queued"}
    threading.Thread(target=_worker,args=(job_id,req),daemon=True).start();return JOBS[job_id]

@app.get("/api/jobs/{job_id}")
def get_job(job_id:str):
    with LOCK:job=JOBS.get(job_id);snapshot=None if job is None else dict(job)
    if snapshot is None:raise HTTPException(404,"unknown job")
    snapshot.pop("error",None);return snapshot
