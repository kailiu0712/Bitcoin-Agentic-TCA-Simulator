"""Replay liquidation schedules on reconstructed real L3 top-of-book data.

This is an execution-cost replay proxy: it does not insert historical orders
into the venue and therefore cannot identify a true counterfactual impact.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def schedule(name, n):
    if name == "immediate":
        w = np.zeros(n); w[0] = 1.0
    elif name == "twap":
        w = np.ones(n) / n
    else:
        # Patient in wide/fragile books; mild front-loading otherwise.
        w = np.exp(-0.5 * np.linspace(0, 1, n)); w /= w.sum()
    return w


def replay(frame, side="SELL", quantity=1.0, slices=20):
    frame = frame.dropna(subset=["mid", "best_bid", "best_ask"]).copy()
    frame = frame.sort_values("timestamp_ns")
    if len(frame) < slices * 3:
        raise ValueError("not enough reconstructed quote observations")
    # Use evenly spaced ten-minute windows from the real session.
    ts = frame.timestamp_ns.to_numpy() / 1e9
    starts = np.arange(ts[0], ts[-1] - 600, 600.0)
    starts = starts[::max(1, len(starts) // 24)][:24]
    rows = []
    sign = 1 if side == "BUY" else -1
    for start in starts:
        ix = np.searchsorted(ts, start + np.arange(slices) * 600.0 / slices)
        ix = np.clip(ix, 0, len(frame) - 1)
        quotes = frame.iloc[ix]
        arrival = float(quotes.iloc[0].mid)
        for name in ("immediate", "twap", "adaptive"):
            weights = schedule(name, slices)
            # A replay fill uses the observed touch. This intentionally does
            # not invent hidden depth beyond the recorded top-of-book.
            touch = quotes.best_ask.to_numpy() if side == "BUY" else quotes.best_bid.to_numpy()
            fill_notional = float(np.sum(weights * quantity * touch))
            avg = fill_notional / quantity
            shortfall_bp = sign * (avg - arrival) / arrival * 1e4
            terminal = float(quotes.iloc[-1].mid)
            drift_bp = sign * (terminal - arrival) / arrival * 1e4
            rows.append({"window_start":pd.to_datetime(start, unit="s", utc=True),"strategy":name,
                "arrival_mid":arrival,"average_touch_fill":avg,"shortfall_bp":shortfall_bp,
                "terminal_mid_drift_bp":drift_bp,"mean_spread":float(quotes.spread.mean()),
                "mean_touch_depth":float((quotes.ask_size if side=="BUY" else quotes.bid_size).mean())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.input, columns=["timestamp_ns","mid","best_bid","best_ask","spread","bid_size","ask_size"])
    result = replay(frame)
    result.to_csv(out / "real_replay_observations.csv", index=False)
    summary = result.groupby("strategy").agg(windows=("strategy","size"),mean_shortfall_bp=("shortfall_bp","mean"),
        median_shortfall_bp=("shortfall_bp","median"),mean_terminal_drift_bp=("terminal_mid_drift_bp","mean"),
        mean_spread=("mean_spread","mean"),mean_touch_depth=("mean_touch_depth","mean")).reset_index()
    summary.to_csv(out / "real_replay_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__": main()
