"""
Turn the tapes of a run pair (with / without liquidation) into replay.json, the frames the page plays.

    python replay.py <job dir>        (the job dir holds liquidation/ and counterfactual/, from scenario.py)

Reads only the exchange's LOBSTER message tape and the bot's fills, never a model's code, so it is
the same for every model. The book is rebuilt by aggregating the tape: every row's Size is the lots
added to (type 1) or removed from (types 2, 3, 4) the level at Price on side Direction (+1 bid, -1 ask).
"""

import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = 10                 # book levels per side shown in the heatmap
MAX_FRAMES = 1800           # frame step = 1 s, longer if the run has more seconds than this
EPOCH = dt.datetime(1970, 1, 1)


def load(run_dir: Path) -> tuple[np.ndarray, dict, pd.DataFrame]:
    return (np.load(run_dir / "message.npy"), json.loads((run_dir / "meta.json").read_text()),
            pd.read_csv(run_dir / "fills.csv"))


def us(iso: str) -> int:
    """ISO time -> microseconds since 1970, the tape's time unit."""
    return round((dt.datetime.fromisoformat(iso) - EPOCH).total_seconds() * 1e6)


def book_frames(tape: np.ndarray, frame_us: np.ndarray, levels: int = LEVELS) -> dict:
    """The book just before each frame time (every tape row strictly earlier), as the impact
    analysis samples it (impact_analysis/reduce.py): the top `levels` bid / ask levels, and the mid,
    which a one-sided book keeps at its last two-sided value (nan before the first one).

    Prices stay in the tape's fixed point (price x 10^4), sizes in lots; empty levels are 0."""
    n = len(frame_us)
    out = {k: np.zeros((n, levels), np.int64) for k in ("bid_px", "bid_sz", "ask_px", "ask_sz")}
    out["mid"] = np.full(n, np.nan)
    book = {1: {}, -1: {}}                      # direction -> {price: lots}
    last_mid = np.nan
    row, n_rows = 0, len(tape)
    for k, t in enumerate(frame_us):
        while row < n_rows and tape[row, 0] < t:
            _, typ, _, size, price, direction = tape[row]
            side = book[direction]
            lots = side.get(price, 0) + (size if typ == 1 else -size)
            if lots <= 0 and len(side) == 1 and book[-direction]:
                last_mid = (max(book[1]) + min(book[-1])) / 2      # this row empties a side
            if lots > 0:
                side[price] = lots
            else:
                side.pop(price, None)
            row += 1
        for name, side, best_first in (("bid", book[1], True), ("ask", book[-1], False)):
            prices = sorted(side, reverse=best_first)[:levels]
            out[f"{name}_px"][k, :len(prices)] = prices
            out[f"{name}_sz"][k, :len(prices)] = [side[p] for p in prices]
        both = book[1] and book[-1]
        out["mid"][k] = (out["bid_px"][k, 0] + out["ask_px"][k, 0]) / 2 if both else last_mid
    return out


def trades(tape: np.ndarray, t0_us: int) -> pd.DataFrame:
    """One row per aggressive order: the execution rows (type 4) at one time against one side.
    The aggressor is the opposite of the resting side: hitting bids = a sell."""
    ex = pd.DataFrame(tape[tape[:, 1] == 4][:, [0, 3, 4, 5]], columns=["t", "lots", "px", "dir"])
    ex["notional"] = ex.lots * ex.px
    g = ex.groupby(["t", "dir"], sort=True).agg(lots=("lots", "sum"), notional=("notional", "sum")).reset_index()
    return pd.DataFrame({"t": (g.t - t0_us) / 1e6, "px": g.notional / g.lots, "lots": g.lots,
                         "side": np.where(g.dir == 1, "sell", "buy")})


def build(job: Path) -> dict:
    tape_l, meta, fills = load(job / "liquidation")
    tape_c, meta_c, _ = load(job / "counterfactual")
    t0, t1 = us(meta["start_time"]), us(meta["stop_time"])
    liq0, liq1 = us(meta["liq_start"]), us(meta["liq_end"])
    total_s = (t1 - t0) / 1e6
    step = max(1, math.ceil(total_s / MAX_FRAMES))
    frame_s = np.arange(0, total_s + 1e-9, step)
    frame_us = t0 + np.round(frame_s * 1e6).astype(np.int64)

    px = 1e-4                                    # tape price -> USD
    lot = meta["lot_size"]                       # lots -> SOL
    tick = meta["tick_size"]
    tick_fp = round(tick * 1e4)                  # one tick in the tape's fixed point

    def run_frames(tape: np.ndarray) -> dict:
        b = book_frames(tape, frame_us)
        has = (b["bid_px"][:, 0] > 0) & (b["ask_px"][:, 0] > 0)
        bid = np.where(has, b["bid_px"][:, 0] * px, np.nan)
        ask = np.where(has, b["ask_px"][:, 0] * px, np.nan)
        tr = trades(tape, t0)
        # mid in ticks x tick, the arithmetic of reduce.py, so the numbers match the impact analysis
        mid = b["mid"] / tick_fp * tick
        return {"book": b, "bid": bid, "ask": ask, "mid": mid,
                "depth_bid": b["bid_sz"].sum(1) * lot, "depth_ask": b["ask_sz"].sum(1) * lot, "trades": tr}

    runs = {"liquidation": run_frames(tape_l), "counterfactual": run_frames(tape_c)}

    # one price grid (in ticks) for both heatmaps, so their axes match
    all_px = np.concatenate([r["book"][f"{s}_px"].ravel() for r in runs.values() for s in ("bid", "ask")])
    all_px = all_px[all_px > 0] // tick_fp
    lo, hi = int(all_px.min()), int(all_px.max())
    grid = np.arange(lo, hi + 1)

    def heatmap(b: dict) -> list:
        """signed volume per (price, frame): bids < 0, asks > 0, None where the level is empty."""
        z = np.full((len(grid), len(frame_s)), np.nan)
        cols = np.arange(len(frame_s))[:, None].repeat(LEVELS, 1)
        for name, sign in (("bid", -1), ("ask", 1)):
            p, s = b[f"{name}_px"], b[f"{name}_sz"]
            ok = p > 0
            z[p[ok] // tick_fp - lo, cols[ok]] = sign * s[ok] * lot
        return [[None if math.isnan(v) else round(v, 6) for v in r] for r in z]

    def series(a: np.ndarray, nd: int | None = 6) -> list:
        """JSON list: None for nan, rounded to nd decimals (nd=None: full precision)."""
        return [None if math.isnan(v) else (float(v) if nd is None else round(float(v), nd)) for v in a]

    # the impact analysis's definitions (impact_analysis/analyze.py):
    #   P0 = m_liq(t_start-), IS = eps (VWAP - P0) / P0 1e4 (> 0: a cost), eps = +1 buy / -1 sell,
    #   I(t) = eps (m_liq(t-) - m_base(t-)) / m_base(t_start-) 1e4 (> 0: the price moved against the trade)
    eps = 1 if meta["side"] == "buy" else -1
    i0 = int(np.searchsorted(frame_us, liq0))    # the frame at t_start (sampled just before it)
    ml, mb = runs["liquidation"]["mid"], runs["counterfactual"]["mid"]
    p0 = float(ml[i0])
    impact = eps * (ml - mb) / mb[i0] * 1e4
    executed = float(fills["size"].sum()) if len(fills) else 0.0
    vwap = float((fills.price * fills["size"]).sum() / executed) if executed else None
    pre_l, pre_c = tape_l[tape_l[:, 0] < liq0], tape_c[tape_c[:, 0] < liq0]
    fills_t = [(us(t) - t0) / 1e6 for t in fills["time"]]

    out = {
        "meta": {**meta, "total_s": total_s, "step_s": step, "liq_start_s": (liq0 - t0) / 1e6,
                 "liq_end_s": (liq1 - t0) / 1e6, "tick": tick, "p0": p0, "executed": executed, "vwap": vwap,
                 "is_bps": eps * (vwap - p0) / p0 * 1e4 if vwap else None,
                 "identical_before_liquidation": bool(pre_l.shape == pre_c.shape and (pre_l == pre_c).all())},
        "t": series(frame_s, 3),
        "impact": series(impact, None),
        "price_grid": series(grid * tick, 8),
        "fills": {"t": fills_t, "px": series(fills.price.to_numpy(), 8),
                  "size": series(fills["size"].to_numpy(), 6),
                  "cum": series(np.cumsum(fills["size"].to_numpy()), 6)},
    }
    for name, r in runs.items():
        tr = r["trades"]
        out[name] = {
            "mid": series(r["mid"], None), "bid": series(r["bid"], 8), "ask": series(r["ask"], 8),
            "depth_bid": series(r["depth_bid"], 4), "depth_ask": series(r["depth_ask"], 4),
            "heatmap": heatmap(r["book"]),
            "trades": {"t": series(tr.t.to_numpy(), 6), "px": series(tr.px.to_numpy() * px, 8),
                       "size": series(tr.lots.to_numpy() * lot, 6), "side": tr.side.tolist()},
        }
    return out


if __name__ == "__main__":
    job = Path(sys.argv[1])
    (job / "replay.json").write_text(json.dumps(build(job), separators=(",", ":")))
    print(f"wrote {job / 'replay.json'}")
