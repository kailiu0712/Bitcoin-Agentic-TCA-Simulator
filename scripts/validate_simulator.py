import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,yaml
from analytics.plots import make_plots
from analytics.stylized_facts import compute,scalar_metrics
from calibration.objective import calibration_loss
from scripts.run_simulation import simulate


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="outputs/calibration/best_config.yaml");ap.add_argument("--base-config",default="config/default.yaml");ap.add_argument("--targets",default="outputs/baseline_market/real_stylized_facts/pooled_targets.json");ap.add_argument("--output-dir",default="outputs/validation");ap.add_argument("--seeds",type=int,default=10);ap.add_argument("--p0p1-only",action="store_true")
    args=ap.parse_args();base=yaml.safe_load(Path(args.base_config).read_text());cfg=yaml.safe_load(Path(args.config).read_text())["simulation"];target_path=Path(args.targets);targets=json.loads(target_path.read_text()) if target_path.exists() else {"calibration":{},"validation":{}};targets["calibration"].update(base["calibration"].get("priority_targets", {}));targets["validation"].update(base["calibration"].get("priority_targets", {}));weights=base["calibration"]["weights"];out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    if args.p0p1_only:
        targets={period:{k:v for k,v in values.items() if k in base["calibration"].get("priority_targets", {})} for period,values in targets.items()}
    records=[];sim_metrics=[];metadata=[];started=time.perf_counter()
    for i in range(args.seeds):
        df,meta=simulate(cfg,2_000_000+i);m=compute(df);sim_metrics.append(m);metadata.append(meta)
        for period in ("calibration","validation"):
            loss,rows=calibration_loss(m,targets[period],weights)
            for row in rows:records.append({"seed":i,"period":period,"total_loss":loss,**row})
        if i==0:
            df.to_parquet(out/"representative_simulated_events.parquet",index=False);make_plots(df,m,out/"plots","simulated")
            # Event-level add/cancel/execution rows with contemporaneous L1
            # and depth-profile state: the simulator's reproducible L3-style
            # event-book artifact for downstream TCA diagnostics.
            l3_cols=[c for c in ["timestamp","event_type","agent_id","side","price","size","order_id","trade_id","best_bid","best_ask","mid","spread","bid_depth_profile","ask_depth_profile"] if c in df]
            df[l3_cols].to_parquet(out/"simulated_l3_event_book.parquet",index=False)
    raw=pd.DataFrame(records);raw.to_csv(out/"seed_metric_errors.csv",index=False)
    summary=raw.groupby(["period","metric"],as_index=False).agg(normalized_distance=("normalized_distance","mean"),seed_std=("normalized_distance","std"),weight=("weight","first"),contribution=("contribution","mean"))
    wide=summary.pivot(index="metric",columns="period",values=["normalized_distance","seed_std","contribution"]);wide.columns=[f"{a}_{b}" for a,b in wide.columns];wide=wide.reset_index().sort_values("normalized_distance_validation",ascending=False);wide.to_csv(out/"validation_metrics.csv",index=False)
    variability_path=Path("outputs/baseline_market/real_stylized_facts/daily_variability.json");variability=json.loads(variability_path.read_text()) if variability_path.exists() else {"calibration":{},"validation":{}};scalar_rows=[]
    for period in ("calibration","validation"):
        real=scalar_metrics(targets[period])
        for name,rv in real.items():
            vals=[scalar_metrics(m).get(name) for m in sim_metrics];vals=[v for v in vals if v is not None]
            if rv is not None and vals:
                day_std=variability.get(period,{}).get(name,{}).get("std")
                scalar_rows.append({"period":period,"metric":name,"real":rv,"sim_mean":float(np.mean(vals)),"sim_std":float(np.std(vals)),"relative_error":float(abs(np.mean(vals)-rv)/(abs(rv)+1e-8)),"empirical_daily_std":day_std,"z_score":None if not day_std else float((np.mean(vals)-rv)/(day_std+1e-8))})
    pd.DataFrame(scalar_rows).to_csv(out/"scalar_comparison.csv",index=False)
    key=wide.set_index("metric");gate_metrics=( ["spread_one_tick_share","trade_size_tail_p99_to_median","trade_size_tail_top_1pct_volume_share","arrival_clustering","sign_memory_exponent","response_monotone_fraction","depth_shape_bid","depth_shape_ask","ofi_slope","ofi_r2"] if args.p0p1_only else ["spread_distribution","relative_spread_distribution","spread_one_tick_share","trade_size_distribution","trade_size_tail_p99_to_median","trade_size_tail_top_1pct_volume_share","arrival_clustering","interarrival_distribution","sign_acf_curve","ofi_slope","ofi_r2","response_curve","clock_response_curve","size_impact_curve","flow_impact_curve","return_volatility","bid_depth_distribution","ask_depth_distribution"] )
    failures=[m for m in gate_metrics if m not in key.index or key.loc[m,"normalized_distance_validation"]>1.0]
    gate={"passed":not failures,"threshold":"validation normalized distance <= 1.0 for every key metric","failures":failures}
    report={"seeds":args.seeds,"runtime_seconds":time.perf_counter()-started,"config":cfg,"calibration_dates":base["data"]["calibration_days"],"validation_dates":base["data"]["validation_days"],"gate":gate,"run_metadata":metadata}
    (out/"simulated_metrics_by_seed.json").write_text(json.dumps(sim_metrics,indent=2));(out/"validation_summary.json").write_text(json.dumps(report,indent=2));print(json.dumps({"gate":gate,"metrics":str(out/"validation_metrics.csv")},indent=2))


if __name__=="__main__":main()
