import argparse,json,time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np,pandas as pd


def fit_strategy(g,bootstrap=500,seed=7,min_participation=.01):
    pts=g.groupby("participation",as_index=False).agg(impact=("peak_impact_causal","mean"),std=("peak_impact_causal","std"),n=("peak_impact_causal","count"));pts["impact_bp"]=pts.impact*1e4
    all_ok=(pts.participation>0)&(pts.impact>0);ax=pts.loc[all_ok,"participation"].to_numpy();ay=pts.loc[all_ok,"impact"].to_numpy();all_delta=float(np.polyfit(np.log(ax),np.log(ay),1)[0]) if len(ax)>=3 else None
    ok=all_ok&(pts.participation>=min_participation);x=pts.loc[ok,"participation"].to_numpy();y=pts.loc[ok,"impact"].to_numpy()
    if len(x)<3:return pts,{"A":None,"delta":None,"ci_low":None,"ci_high":None,"power_r2":None,"linear_r2":None,"n":len(g)}
    coef=np.polyfit(np.log(x),np.log(y),1);delta,A=float(coef[0]),float(np.exp(coef[1]));pred=A*x**delta;power_r2=1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2)
    lin=np.polyfit(x,y,1);lp=np.polyval(lin,x);linear_r2=1-np.sum((y-lp)**2)/np.sum((y-y.mean())**2);rng=np.random.default_rng(seed);slopes=[]
    seeds=g.seed.unique()
    for _ in range(bootstrap):
        chosen=rng.choice(seeds,len(seeds),replace=True);sample=pd.concat([g[g.seed==s] for s in chosen]);p=sample.groupby("participation",as_index=False).peak_impact_causal.mean();p=p[(p.participation>=min_participation)&(p.peak_impact_causal>0)]
        if len(p)>=3:slopes.append(np.polyfit(np.log(p.participation),np.log(p.peak_impact_causal),1)[0])
    ci=np.quantile(slopes,[.025,.975]) if slopes else [np.nan,np.nan]
    return pts,{"A":A,"delta":delta,"ci_low":float(ci[0]),"ci_high":float(ci[1]),"power_r2":float(power_r2),"linear_r2":float(linear_r2),"n":int(len(g)),"fit_min_participation":min_participation,"all_grid_delta":all_delta,"distance_from_half":abs(delta-.5),"impact_at_1pct":A*.01**delta,"impact_at_5pct":A*.05**delta}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input-dir",default="outputs/liquidation");ap.add_argument("--output-dir",default="outputs/liquidation/comparison");ap.add_argument("--fit-min-participation",type=float,default=.01);args=ap.parse_args();root=Path(args.input_dir);out=Path(args.output_dir);plots=out/"plots";plots.mkdir(parents=True,exist_ok=True);runs=pd.read_parquet(root/"runs.parquet");paths=pd.read_parquet(root/"impact_paths.parquet");decay=pd.read_parquet(root/"impact_decay.parquet");fits=[];point_tables={}
    for strategy,g in runs.groupby("strategy"):
        pts,fit=fit_strategy(g.dropna(subset=["peak_impact_causal"]),min_participation=args.fit_min_participation);fit["strategy"]=strategy;fit["mean_shortfall"]=float(g.shortfall.mean());fit["shortfall_std"]=float(g.shortfall.std());fit["mean_paired_cost"]=float(g.paired_execution_cost.mean());fit["mean_penalized_cost"]=float(g.penalized_execution_cost.mean());fit["completion_rate"]=float(g.completion.mean());fit["minimum_completion"]=float(g.completion.min());fits.append(fit);point_tables[strategy]=pts;pts.to_csv(out/f"{strategy}_impact_by_size.csv",index=False)
        plt.figure();plt.errorbar(pts.participation,pts.impact,yerr=pts["std"],marker="o",capsize=3,label="paired causal mean")
        if fit["A"]:xx=np.logspace(np.log10(args.fit_min_participation),np.log10(pts.participation.max()),100);plt.plot(xx,fit["A"]*xx**fit["delta"],label=f"power delta={fit['delta']:.3f}, Q/V >= {args.fit_min_participation:g}")
        plt.axhline(0,color="black",lw=.7);plt.xlabel("Q / baseline volume");plt.ylabel("Peak causal impact");plt.legend();plt.tight_layout();plt.savefig(plots/f"{strategy}_impact_size.png",dpi=160);plt.xscale("log")
        if fit["A"]:plt.yscale("log")
        else:plt.text(.5,.9,"Power law not estimable: no positive causal mean",transform=plt.gca().transAxes,ha="center")
        plt.tight_layout();plt.savefig(plots/f"{strategy}_impact_size_loglog.png",dpi=160);plt.close()
        pg=paths[paths.strategy==strategy].dropna(subset=["impact"]);pg["progress_bin"]=pd.cut(pg.progress,np.linspace(0,1,21),include_lowest=True);curve=pg.groupby("progress_bin",observed=True).agg(progress=("progress","mean"),impact_observed=("impact","mean"),observed_std=("impact","std"),impact_causal=("impact_causal","mean"),causal_std=("impact_causal","std"));curve.to_csv(out/f"{strategy}_impact_path.csv",index=False);plt.figure();plt.plot(curve.progress,curve.impact_observed,label="observed");plt.plot(curve.progress,curve.impact_causal,label="paired causal");plt.fill_between(curve.progress,curve.impact_causal-curve.causal_std,curve.impact_causal+curve.causal_std,alpha=.2);plt.xlabel("Executed fraction");plt.ylabel("Impact");plt.legend();plt.tight_layout();plt.savefig(plots/f"{strategy}_impact_path.png",dpi=160);plt.close()
        dg=decay[decay.strategy==strategy].groupby("seconds_after",as_index=False).agg(impact=("impact_causal","mean"),std=("impact_causal","std"));dg.to_csv(out/f"{strategy}_impact_decay.csv",index=False);plt.figure();plt.plot(dg.seconds_after,dg.impact,marker="o");plt.fill_between(dg.seconds_after,dg.impact-dg["std"],dg.impact+dg["std"],alpha=.2);plt.xlabel("Seconds after completion");plt.ylabel("Causal impact");plt.tight_layout();plt.savefig(plots/f"{strategy}_impact_decay.png",dpi=160);plt.close()
        cost=g.groupby("participation",as_index=False).agg(shortfall=("shortfall","mean"),std=("shortfall","std"),paired_cost=("paired_execution_cost","mean"),paired_std=("paired_execution_cost","std"),penalized_cost=("penalized_execution_cost","mean"),completion=("completion","mean"));cost["shortfall_bp"]=cost.shortfall*1e4;cost["std_bp"]=cost["std"]*1e4;cost["paired_cost_bp"]=cost.paired_cost*1e4;cost["penalized_cost_bp"]=cost.penalized_cost*1e4;cost.to_csv(out/f"{strategy}_cost_by_size.csv",index=False);plt.figure();plt.errorbar(cost.participation,cost.paired_cost,yerr=cost.paired_std,marker="o");plt.xlabel("Q / baseline volume");plt.ylabel("Paired execution cost");plt.tight_layout();plt.savefig(plots/f"{strategy}_shortfall.png",dpi=160);plt.close()
        detail=g.groupby(["duration","side","participation"],as_index=False).agg(peak_observed=("peak_impact_observed","mean"),peak_causal=("peak_impact_causal","mean"),shortfall=("shortfall","mean"),completion=("completion","mean"),peak_causal_std=("peak_impact_causal","std"));detail.to_csv(out/f"{strategy}_detail_by_duration_side.csv",index=False)
        fit["impact_at_1pct"]=float(pts.loc[np.isclose(pts.participation,.01),"impact"].mean());fit["impact_at_5pct"]=float(pts.loc[np.isclose(pts.participation,.05),"impact"].mean())
    table=pd.DataFrame(fits);table["runtime_seconds"]=json.loads((root/"experiment_metadata.json").read_text())["runtime_seconds"];table.to_csv(out/"strategy_comparison.csv",index=False);(out/"fits.json").write_text(json.dumps(fits,indent=2))
    plt.figure()
    for s,p in point_tables.items():plt.plot(p.participation,p.impact,marker="o",label=s)
    plt.xlabel("Q / baseline volume");plt.ylabel("Peak causal impact");plt.legend();plt.tight_layout();plt.savefig(plots/"all_strategies_impact.png",dpi=160);plt.close();print(table.to_string(index=False))


if __name__=="__main__":main()
