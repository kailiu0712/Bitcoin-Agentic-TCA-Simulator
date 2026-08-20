import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import yaml
from agents.market_maker import MarketMakerAgent
from agents.order_flow import OrderFlowAgent
from analytics.stylized_facts import compute
from market.exchange import Exchange
from market.kernel import Kernel


def simulate(cfg, seed=None):
    seed = cfg["seed"] if seed is None else seed
    root = np.random.SeedSequence(seed); mm_seed, of_seed = root.spawn(2)
    ex, started = Exchange(), time.perf_counter(); kernel = Kernel(ex, seed)
    mm = MarketMakerAgent(cfg, np.random.default_rng(mm_seed)); flow = OrderFlowAgent(cfg, np.random.default_rng(of_seed))
    mm.start(kernel); flow.start(kernel); kernel.run(float(cfg["duration_seconds"]))
    rows = ex.output(); df = pd.DataFrame(rows)
    if len(df):
        df["timestamp_ns"] = (df.pop("timestamp") * 1e9).round().astype("int64")
        df["sign_source"] = np.where(df.event_type.eq("TRADE"), "DIRECT_AGGRESSOR_SIDE", None)
        df["source_timestamp"] = pd.to_datetime(df.timestamp_ns, unit="ns", utc=True)
        df["order_type"] = np.where(df.event_type.eq("TRADE"), "market", None)
    metadata = {"code_version": "compact-btc-sim-v1", "random_seed": seed, "config": cfg, "simulation_duration": cfg["duration_seconds"],
                "number_of_events": int(len(df)), "number_of_trades": int(len(ex.trades)),
                "runtime_seconds": time.perf_counter() - started}
    return df, metadata


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config/default.yaml"); ap.add_argument("--seed",type=int)
    ap.add_argument("--output-dir",default="outputs/simulation"); args=ap.parse_args()
    config=yaml.safe_load(Path(args.config).read_text())["simulation"]; out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    df, meta=simulate(config,args.seed); df.to_parquet(out/"events.parquet",index=False)
    metrics=compute(df); (out/"metrics.json").write_text(json.dumps(metrics,indent=2)); (out/"run_metadata.json").write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))


if __name__=="__main__": main()
