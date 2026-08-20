import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,yaml
from liquidation.runner import make_agent,midpoint_at,midpoint_from_path,midpoint_path,run_market


def load_q(path):
    raw=json.loads(Path(path).read_text());return {tuple(map(int,k.split("|"))):v for k,v in raw.items()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="outputs/calibration/best_config.yaml");ap.add_argument("--q-table",default="outputs/liquidation/rl/q_table.json");ap.add_argument("--output-dir",default="outputs/liquidation");ap.add_argument("--seeds",type=int,default=5);ap.add_argument("--strategies",nargs="+",default=["twap","pov","frontload","adaptive","rl2"])
    args=ap.parse_args();cfgdoc=yaml.safe_load(Path(args.config).read_text());cfg=cfgdoc["simulation"];q=load_q(args.q_table);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    ratios=[.001,.0025,.005,.01,.02,.05,.1,.2];durations=[120.,300.,600.];strategies=args.strategies;sides=["BUY","SELL"];start=180.;post=300.;runs=[];paths=[];decays=[];began=time.perf_counter();baseline_cache={}
    for duration in durations:
        for seed in range(args.seeds):
            total=start+duration+post;ex0,_=run_market(cfg,1000+seed+int(duration),total);baseline_cache[(duration,seed)]=ex0;base_path=midpoint_path(ex0.events)
            volume=sum(t["size"] for t in ex0.trades if start<=t["timestamp"]<=start+duration);base_end=midpoint_at(ex0.events,start+duration);base_start=midpoint_at(ex0.events,start)
            for ratio in ratios:
                quantity=max(1e-6,ratio*volume)
                for side in sides:
                    sign=1 if side=="BUY" else -1
                    for strategy in strategies:
                        agent=make_agent(strategy,side,quantity,start,start+duration,max(4,int(duration/10)),q_table=q,seed=seed)
                        ex,agent=run_market(cfg,1000+seed+int(duration),total,agent);m0=agent.arrival_mid or midpoint_at(ex.events,start);reaction_time=start+duration+cfg["market_maker_refresh_seconds"];mend=midpoint_at(ex.events,reaction_time);base_end=midpoint_from_path(base_path,reaction_time);avg=agent.average_fill_price
                        observed=None if None in (m0,mend) else sign*(mend-m0)/m0;end_causal=None if None in (m0,mend,base_end) else sign*(mend-base_end)/m0;shortfall=None if avg is None or m0 is None else sign*(avg-m0)/m0
                        causal_samples=[]
                        for event in ex.events:
                            stable=event.get("event_type")=="QUOTE_UPDATE" or str(event.get("event_type","")).endswith("EXECUTION")
                            if stable and start<=event["timestamp"]<=reaction_time and event.get("mid") is not None:
                                bm=midpoint_from_path(base_path,event["timestamp"])
                                if bm is not None:causal_samples.append(sign*(event["mid"]-bm)/m0)
                        peak_causal=max(causal_samples) if causal_samples else end_causal
                        paired_cost=0.0
                        for rec in agent.records:
                            bm=midpoint_from_path(base_path,rec["timestamp"])
                            if rec["filled"] and bm is not None:paired_cost+=sign*(rec["fill_notional"]-bm*rec["filled"])/(m0*quantity)
                        completion=agent.filled/quantity;penalized_cost=paired_cost+.0005*(1-completion)
                        run_id=f"{strategy}_{int(duration)}_{seed}_{side}_{ratio:g}";runs.append({"run_id":run_id,"strategy":strategy,"duration":duration,"seed":seed,"side":side,"participation":ratio,"market_volume":volume,"quantity":quantity,"filled":agent.filled,"completion":completion,"arrival_mid":m0,"end_mid":mend,"baseline_end_mid":base_end,"peak_impact_observed":observed,"peak_impact_causal":peak_causal,"end_impact_causal":end_causal,"shortfall":shortfall,"paired_execution_cost":paired_cost,"penalized_execution_cost":penalized_cost})
                        for rec in agent.records:
                            baseline_mid=midpoint_from_path(base_path,rec["timestamp"])
                            paths.append({"run_id":run_id,"strategy":strategy,"duration":duration,"seed":seed,"side":side,"participation":ratio,**rec,
                                "impact":None if m0 is None or rec["mid"] is None else sign*(rec["mid"]-m0)/m0,
                                "baseline_mid":baseline_mid,"impact_causal":None if None in (m0,rec["mid"],baseline_mid) else sign*(rec["mid"]-baseline_mid)/m0})
                        for lag in [0,30,60,120,300]:
                            mi=midpoint_at(ex.events,start+duration+lag,direction="after");mb=midpoint_from_path(base_path,start+duration+lag,direction="after")
                            decays.append({"run_id":run_id,"strategy":strategy,"duration":duration,"seed":seed,"side":side,"participation":ratio,"seconds_after":lag,"impact_observed":None if None in (mi,m0) else sign*(mi-m0)/m0,"impact_causal":None if None in (mi,mb,m0) else sign*(mi-mb)/m0})
    pd.DataFrame(runs).to_parquet(out/"runs.parquet",index=False);pd.DataFrame(runs).to_csv(out/"runs.csv",index=False);pd.DataFrame(paths).to_parquet(out/"impact_paths.parquet",index=False);pd.DataFrame(decays).to_parquet(out/"impact_decay.parquet",index=False)
    meta={"exploratory_warning":"Stage 1 response/flow-impact validation failed; do not treat Stage 2 as validated TCA.","background_config_frozen_for_grid":True,"config":cfg,"ratios":ratios,"durations":durations,"sides":sides,"strategies":strategies,"seeds":list(range(args.seeds)),"runs":len(runs),"runtime_seconds":time.perf_counter()-began,"common_random_numbers":True}
    (out/"experiment_metadata.json").write_text(json.dumps(meta,indent=2));print(json.dumps({"runs":len(runs),"runtime_seconds":meta["runtime_seconds"]},indent=2))


if __name__=="__main__":main()
