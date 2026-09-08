"""
Stylized fact 5 (P0-C) - Instantaneous liquidity-cost curve / cumulative depth.

Empirical claim
    Bitcoin books are shallow and the cost of a virtual market order rises
    steeply with size.  Schnaubelt, Rende & Krauss study exactly this on
    BTC/USD: bid-ask and VWAP spreads at increasing order sizes.

Why it matters for the simulator
    This is the mechanical starting point of liquidation slippage.  The
    calibration target is the whole map Q -> VWAP cost, not the mean depth at
    the touch.

Method
    The kernel walks the full visible book at **every** packet and accumulates
    time-weighted histograms of the cost of a virtual market order at each size
    on the configured grid.  The curve reported here is therefore the
    unconditional, time-weighted distribution over every coherent book state in
    the regime - not a sample of states, and not conditioned on trades
    happening.  ``feasible_share`` records how often the visible book (100
    published levels per side) could absorb the size at all: costs are computed
    only over the feasible states, so a curve that flattens at the top end
    should be read together with that column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfcore import plotting as pl
from sfcore import stats as st

FACT = "fact05"
FACT_ID = 5
PRIORITY = "P0-C"
TITLE = "Instantaneous liquidity-cost curve / cumulative LOB depth"

#: Sides of the book, in accumulator column order.
SIDE_LABEL = {0: "sell into bids", 1: "buy from asks"}
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def _curve_table(rd) -> pd.DataFrame:
    """Time-weighted cost statistics per virtual order size and direction."""
    acc = rd.accumulators
    sizes = acc["sizes"]
    cost_tw = acc["cost_tw"]              # [size, side, bin]
    stats = acc["cost_stats"]             # [stat, size, side]
    lo, hi = acc["cost_hist_range"]
    nbins = int(acc["hist_bins"][0])
    edges = st.log_bin_edges(float(lo), float(hi), nbins)

    rows = []
    for i, size in enumerate(sizes):
        for side in (0, 1):
            feas_w = stats[1, i, side]
            tot_w = stats[3, i, side]
            mean = stats[0, i, side] / feas_w if feas_w > 0 else np.nan
            var = (stats[2, i, side] / feas_w - mean ** 2) if feas_w > 0 else np.nan
            qs = st.hist_quantiles(cost_tw[i, side], edges, QUANTILES)
            rows.append({
                "size_btc": float(size),
                "direction": SIDE_LABEL[side],
                "mean_cost_bps": mean,
                "std_cost_bps": np.sqrt(max(var, 0.0)) if np.isfinite(var) else np.nan,
                **{f"p{int(q*100)}_cost_bps": v for q, v in zip(QUANTILES, qs)},
                "feasible_share": (feas_w / tot_w) if tot_w > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def _depth_table(rd) -> pd.DataFrame:
    """Time-weighted mean cumulative depth inside each price band."""
    acc = rd.accumulators
    bands = acc["bands"]
    depth = acc["depth_sum"]
    total_w = float(acc["total_weight_ns"][0])
    rows = []
    for j, band in enumerate(bands):
        rows.append({
            "band_bps": float(band),
            "mean_bid_depth_btc": depth[j, 0] / total_w if total_w > 0 else np.nan,
            "mean_ask_depth_btc": depth[j, 1] / total_w if total_w > 0 else np.nan,
        })
    df = pd.DataFrame(rows)
    df["mean_total_depth_btc"] = df.mean_bid_depth_btc + df.mean_ask_depth_btc
    return df


def run(ctx) -> dict:
    pl.apply_style()
    cfg = ctx.cfg
    metrics: list[dict] = []
    curves: dict[str, pd.DataFrame] = {}
    depths: dict[str, pd.DataFrame] = {}

    for name, rd in ctx.each():
        curve = _curve_table(rd)
        depth = _depth_table(rd)
        curves[name] = curve
        depths[name] = depth

        ctx.out.save_table(curve, FACT, "cost_curve", name,
                           caption=f"Virtual market-order cost, time-weighted over all book states - {rd.label}")
        ctx.out.save_table(depth, FACT, "depth_bands", name,
                           caption=f"Mean cumulative visible depth by price band - {rd.label}")

        # --- per-regime figure ------------------------------------------
        fig, axes = pl.new_figure(1, 3, width=5.2, height=4.2)
        ax = axes[0]
        for side in (0, 1):
            sub = curve[curve.direction == SIDE_LABEL[side]]
            ax.plot(sub.size_btc, sub.mean_cost_bps, "o-", lw=1.6, ms=4,
                    label=f"mean, {SIDE_LABEL[side]}")
            ax.fill_between(sub.size_btc, sub.p25_cost_bps, sub.p75_cost_bps, alpha=0.15)
        ax.set_xscale("log")
        ax.set_yscale("log")
        pl.finish(ax, "Cost of a virtual market order", "order size (BTC)",
                  "cost vs mid (bps)", legend=True)

        ax = axes[1]
        for side in (0, 1):
            sub = curve[curve.direction == SIDE_LABEL[side]]
            ax.plot(sub.size_btc, sub.feasible_share * 100, "o-", lw=1.6, ms=4,
                    label=SIDE_LABEL[side])
        ax.set_xscale("log")
        ax.set_ylim(0, 105)
        pl.finish(ax, "Share of time the visible book can absorb the size",
                  "order size (BTC)", "% of time", legend=True,
                  note="The feed publishes the top 100 price levels per side, so "
                       "this is visible liquidity, not total resting liquidity.")

        ax = axes[2]
        ax.plot(depth.band_bps, depth.mean_bid_depth_btc, "o-", lw=1.6, ms=4, label="bid side")
        ax.plot(depth.band_bps, depth.mean_ask_depth_btc, "s-", lw=1.6, ms=4, label="ask side")
        ax.set_xscale("log")
        pl.finish(ax, "Cumulative depth within a price band", "band from mid (bps)",
                  "mean depth (BTC)", legend=True)
        fig.suptitle(f"Fact 5 - liquidity cost and depth | {rd.label}", x=0.02,
                     ha="left", fontsize=12.5, fontweight="semibold")
        pl.layout(fig)
        ctx.out.save_figure(fig, FACT, "liquidity_cost_and_depth", name)

        # --- concavity of the cost curve --------------------------------
        for side in (0, 1):
            sub = curve[curve.direction == SIDE_LABEL[side]]
            fit = st.fit_loglog(sub.size_btc.to_numpy(), sub.mean_cost_bps.to_numpy())
            metrics.append({
                "metric": f"cost-curve log-log slope ({SIDE_LABEL[side]})",
                "regime": name, "value": fit["exponent"],
                "detail": f"R2={fit['r2']:.3f}",
            })
        ref = curve[(curve.size_btc == 1.0)]
        for _, row in ref.iterrows():
            metrics.append({"metric": f"cost of 1 BTC ({row.direction}), bps",
                            "regime": name, "value": row.mean_cost_bps, "detail": ""})
        d10 = depth[depth.band_bps == 10.0]
        if len(d10):
            metrics.append({"metric": "mean depth within 10 bps (BTC, both sides)",
                            "regime": name, "value": float(d10.mean_total_depth_btc.iloc[0]),
                            "detail": ""})

    # --- comparison figure ------------------------------------------------
    fig, axes = pl.new_figure(1, 2, width=5.8, height=4.4)
    for name, rd in ctx.each():
        curve = curves[name]
        for side, marker in ((0, "o-"), (1, "s--")):
            sub = curve[curve.direction == SIDE_LABEL[side]]
            axes[0].plot(sub.size_btc, sub.mean_cost_bps, marker, color=rd.color,
                         lw=1.6, ms=4, alpha=0.95 if side == 0 else 0.7,
                         label=f"{rd.label} - {SIDE_LABEL[side]}")
        d = depths[name]
        axes[1].plot(d.band_bps, d.mean_total_depth_btc, "o-", color=rd.color,
                     lw=1.7, ms=4, label=rd.label)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    pl.finish(axes[0], "Liquidity-cost curve: normal vs stress", "order size (BTC)",
              "cost vs mid (bps)", legend=True)
    axes[1].set_xscale("log")
    pl.finish(axes[1], "Cumulative visible depth: normal vs stress",
              "band from mid (bps)", "mean depth, both sides (BTC)", legend=True)
    fig.suptitle("Fact 5 - liquidity cost and depth by regime", x=0.02, ha="left",
                 fontsize=12.5, fontweight="semibold")
    pl.layout(fig, rect=(0, 0, 1, 0.94))
    ctx.out.save_figure(fig, FACT, "liquidity_cost_and_depth", "comparison")

    combined = pd.concat([c.assign(regime=n) for n, c in curves.items()], ignore_index=True)
    ctx.out.save_table(combined, FACT, "cost_curve", "comparison",
                       caption="Virtual market-order cost curve, both regimes")

    return {
        "fact": FACT, "id": FACT_ID, "priority": PRIORITY, "title": TITLE,
        "metrics": metrics,
        "literature": "Schnaubelt, Rende & Krauss (2019): BTC/USD books are shallow "
                      "and VWAP cost rises steeply with size.",
    }
