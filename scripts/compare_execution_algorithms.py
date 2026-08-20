import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--benchmark",default="outputs/execution_algorithms/benchmark/runs.parquet");ap.add_argument("--tuned-rl",default="outputs/execution_algorithms/rl2_tuned_benchmark/runs.parquet");ap.add_argument("--immediate",default="outputs/execution_algorithms/immediate_benchmark/runs.parquet");ap.add_argument("--output-dir",default="outputs/execution_algorithms/comparison");args=ap.parse_args()
    base=pd.read_parquet(args.benchmark);tuned=pd.read_parquet(args.tuned_rl);immediate=pd.read_parquet(args.immediate);data=pd.concat([base[base.strategy!="rl2"],tuned,immediate],ignore_index=True);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    rank=data.groupby("strategy",as_index=False).agg(paired_cost=("paired_execution_cost","mean"),penalized_cost=("penalized_execution_cost","mean"),cost_std=("paired_execution_cost","std"),completion=("completion","mean"),minimum_completion=("completion","min"),mean_shortfall=("shortfall","mean"))
    rank["paired_cost_bp"]=rank.paired_cost*1e4;rank["penalized_cost_bp"]=rank.penalized_cost*1e4;rank["cost_std_bp"]=rank.cost_std*1e4;rank=rank.sort_values("penalized_cost");rank.to_csv(out/"algorithm_ranking.csv",index=False)
    by_size=data.groupby(["strategy","participation"],as_index=False).agg(paired_cost=("paired_execution_cost","mean"),penalized_cost=("penalized_execution_cost","mean"),completion=("completion","mean"));by_size["paired_cost_bp"]=by_size.paired_cost*1e4;by_size.to_csv(out/"cost_by_strategy_and_size.csv",index=False)
    plt.figure()
    for strategy,g in by_size.groupby("strategy"):plt.plot(g.participation,g.paired_cost_bp,marker="o",label=strategy)
    plt.xlabel("Q / baseline volume");plt.ylabel("Paired execution cost (bp)");plt.legend();plt.tight_layout();plt.savefig(out/"cost_comparison.png",dpi=170);plt.close();print(rank.to_string(index=False))
    impact=data.groupby(["strategy","participation"],as_index=False).peak_impact_causal.mean();impact["impact_bp"]=impact.peak_impact_causal*1e4;impact.to_csv(out/"impact_by_strategy_and_size.csv",index=False)
    plt.figure()
    for strategy,g in impact.groupby("strategy"):plt.plot(g.participation,g.impact_bp,marker="o",label=strategy)
    plt.xscale("log");plt.xlabel("Parent size / baseline volume");plt.ylabel("Peak paired causal impact (bp)");plt.legend(ncol=2);plt.tight_layout();plt.savefig(out/"impact_comparison.png",dpi=170);plt.close()


if __name__=="__main__":main()
