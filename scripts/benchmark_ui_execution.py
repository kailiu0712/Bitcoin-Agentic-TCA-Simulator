"""Multi-seed impact benchmark across the registered liquidation algorithms.

This used to select a default execution policy by counting wins against TWAP.
That question is no longer the product's: the simulator measures impact, it does
not recommend a schedule. What is worth checking repeatedly is that the *market*
behaves sensibly across seeds and regimes - that a one-shot parent order pushes
the price further than the same parent spread over the same horizon, and that the
paired cost ordering follows. Any newly registered algorithm is picked up
automatically from `app.main.ALGORITHMS`.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import ALGORITHMS, REFERENCE_ALGORITHM, SimulationRequest, run_execution_simulation

PRESETS = {
    "standard": dict(side="SELL", parent_quantity=1., duration_seconds=300, volatility=2.02,
                     spread=.10, arrival_rate=.536, displayed_depth=.30, seed=2026),
    "thin_depth": dict(side="BUY", parent_quantity=4.5, duration_seconds=600, volatility=2.40,
                       spread=.13, arrival_rate=.580, displayed_depth=.18, seed=2027),
    "large_sell": dict(side="SELL", parent_quantity=5., duration_seconds=900, volatility=2.60,
                       spread=.14, arrival_rate=.700, displayed_depth=.42, seed=2028),
}
FIELDS = ("impact_bp", "terminal_impact_bp", "post_trade_impact_bp", "decay_ratio",
          "execution_cost_bp", "penalized_cost_bp", "completion", "filled_quantity",
          "execution_notional")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", default="outputs/latest_p0p1/impact_benchmark.csv")
    args = parser.parse_args()

    rows = []
    for preset, base in PRESETS.items():
        for offset in range(args.seeds):
            request = SimulationRequest(**dict(base, seed=base["seed"] + offset))
            result = run_execution_simulation(request)
            for strategy, value in result["comparison"].items():
                rows.append({"preset": preset, "seed": request.seed, "strategy": strategy,
                             **{field: value[field] for field in FIELDS}})
    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    reference = REFERENCE_ALGORITHM
    others = [item["key"] for item in ALGORITHMS if item["key"] != reference]
    summary = []
    for preset, group in frame.groupby("preset"):
        pivot = group.pivot(index="seed", columns="strategy", values=["impact_bp", "penalized_cost_bp"])
        record = {"preset": preset, "seeds": len(pivot)}
        for strategy in others:
            impact = pivot["impact_bp"][strategy].abs()
            cost = pivot["penalized_cost_bp"][strategy]
            record[f"{strategy}_impact_above_{reference}"] = int((impact > pivot["impact_bp"][reference].abs()).sum())
            record[f"{strategy}_mean_impact_bp"] = float(impact.mean())
            record[f"{reference}_mean_impact_bp"] = float(pivot["impact_bp"][reference].abs().mean())
            record[f"{strategy}_mean_cost_bp"] = float(cost.mean())
            record[f"{reference}_mean_cost_bp"] = float(pivot["penalized_cost_bp"][reference].mean())
        summary.append(record)
    table = pd.DataFrame(summary)
    table.to_csv(output.with_name("impact_benchmark_summary.csv"), index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
