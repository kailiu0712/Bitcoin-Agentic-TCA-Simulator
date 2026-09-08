"""
Stylized fact 16 (P1) - BTC trade-size distribution: many small trades, heavy upper tail.

Empirical claim
    Schnaubelt, Rende & Krauss report very many small BTC trades with a heavy
    upper tail, fit a q-Gamma over the relevant upper region, and also observe
    round-number clustering of trade sizes.

Why it matters for the simulator
    Organic taker flow has to look like this or the background market is wrong.
    The source document is explicit that the *current* venue distribution should
    be re-estimated rather than historical parameters imported.

Method
    Two size series are reported because they are different objects and the
    liquidation simulator needs both:

    * **fill size** - one printed trade, i.e. one maker order consumed;
    * **aggressive-order size** - the sum over a burst, i.e. what a taker
      actually sent.

    The upper tail is characterised by a Hill estimator over the top 5% and by
    the log-log complementary CDF, whose local slope is the tail index if the
    tail is Pareto-like.  Round-number clustering is measured as the share of
    sizes landing exactly on 0.01 / 0.1 / 0.5 / 1 BTC grids.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfcore import plotting as pl
from sfcore import stats as st

FACT = "fact16"
FACT_ID = 16
PRIORITY = "P1"
TITLE = "BTC trade-size distribution"

ROUND_GRIDS = (0.01, 0.1, 0.5, 1.0)
CCDF_POINTS = 400


def _ccdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(x[np.isfinite(x) & (x > 0)])
    if x.size == 0:
        return np.array([]), np.array([])
    if x.size > CCDF_POINTS:
        idx = np.unique(np.geomspace(1, x.size, CCDF_POINTS).astype(int) - 1)
    else:
        idx = np.arange(x.size)
    surv = 1.0 - idx / x.size
    return x[idx], surv


def _round_share(x: np.ndarray) -> pd.DataFrame:
    rows = []
    for grid in ROUND_GRIDS:
        r = np.abs(x / grid - np.round(x / grid))
        rows.append({"grid_btc": grid,
                     "share_on_grid": float(np.mean(r < 1e-9)),
                     "share_within_0.1pct": float(np.mean(r < 1e-3))})
    return pd.DataFrame(rows)


def run(ctx) -> dict:
    pl.apply_style()
    metrics: list[dict] = []
    ccdfs = {}

    for name, rd in ctx.each():
        fill = rd.fills["qty"].to_numpy(np.float64)
        order = rd.trades["qty"].to_numpy(np.float64)
        notional = rd.trades["notional"].to_numpy(np.float64)

        rows = []
        for label, x in (("fill", fill), ("aggressive order", order)):
            hill = st.hill_tail_index(x, 0.05)
            qs = np.nanquantile(x, [0.01, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999])
            rows.append({
                "series": label, "n": int(x.size), "mean_btc": float(np.nanmean(x)),
                **{f"p{q}": v for q, v in zip([1, 25, 50, 75, 90, 99, 99.9], qs)},
                "max_btc": float(np.nanmax(x)),
                "hill_alpha_top5pct": hill["alpha"],
                "hill_threshold_btc": hill["threshold"],
                "share_below_0.01_btc": float(np.mean(x < 0.01)),
            })
        summary = pd.DataFrame(rows)
        rounds = _round_share(order)
        ctx.out.save_table(summary, FACT, "size_distribution", name,
                           caption=f"Trade-size distribution - {rd.label}")
        ctx.out.save_table(rounds, FACT, "round_number_clustering", name,
                           caption=f"Round-number clustering of aggressive-order size - {rd.label}")

        cf, sf = _ccdf(fill)
        co, so = _ccdf(order)
        ccdfs[name] = (co, so)

        fig, axes = pl.new_figure(1, 3, width=5.2, height=4.2)
        ax = axes[0]
        ax.plot(cf, sf, lw=1.8, label="fill size")
        ax.plot(co, so, lw=1.8, ls="--", label="aggressive-order size")
        hill = st.hill_tail_index(order, 0.05)
        if np.isfinite(hill["alpha"]):
            xs = np.geomspace(hill["threshold"], np.nanmax(order), 40)
            ref = 0.05 * (xs / hill["threshold"]) ** (-hill["alpha"])
            ax.plot(xs, ref, lw=1.4, color="#B45309",
                    label=f"Hill tail alpha={hill['alpha']:.2f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        pl.finish(ax, "Complementary CDF of trade size", "size (BTC)", "P(size > x)",
                  legend=True,
                  note="A straight line on log-log is a Pareto tail; its slope is -alpha.")

        ax = axes[1]
        bins = np.geomspace(max(np.nanmin(order[order > 0]), 1e-6), np.nanmax(order), 60)
        ax.hist(order, bins=bins, color="#6B8FBF", alpha=0.9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        pl.finish(ax, "Aggressive-order size histogram", "size (BTC)", "count")

        ax = axes[2]
        ax.bar(np.arange(len(rounds)), rounds["share_within_0.1pct"],
               color="#7A4FA3", width=0.6)
        ax.set_xticks(np.arange(len(rounds)))
        ax.set_xticklabels([f"{g:g}" for g in rounds.grid_btc])
        pl.finish(ax, "Round-number clustering", "size grid (BTC)",
                  "share of orders on the grid")
        fig.suptitle(f"Fact 16 - trade-size distribution | {rd.label}", x=0.02,
                     ha="left", fontsize=12.5, fontweight="semibold")
        pl.layout(fig)
        ctx.out.save_figure(fig, FACT, "trade_size", name)

        metrics += [
            {"metric": "median aggressive-order size (BTC)", "regime": name,
             "value": float(np.nanmedian(order)), "detail": ""},
            {"metric": "share of aggressive orders below 0.01 BTC", "regime": name,
             "value": float(np.mean(order < 0.01)), "detail": "many small trades"},
            {"metric": "Hill tail index alpha (top 5%, aggressive orders)",
             "regime": name, "value": hill["alpha"],
             "detail": f"threshold {hill['threshold']:.4g} BTC; alpha < 3 = heavy tail"},
            {"metric": "99.9th pct / median size ratio", "regime": name,
             "value": float(np.nanquantile(order, 0.999) / np.nanmedian(order)),
             "detail": ""},
            {"metric": "mean aggressive-order notional (USD)", "regime": name,
             "value": float(np.nanmean(notional)), "detail": ""},
            {"metric": "share of orders on the 0.1 BTC grid", "regime": name,
             "value": float(rounds.loc[rounds.grid_btc == 0.1, "share_within_0.1pct"].iloc[0]),
             "detail": "round-number clustering"},
        ]

    fig, ax = pl.new_figure(1, 1, width=7.4, height=4.6)
    for name, rd in ctx.each():
        co, so = ccdfs[name]
        ax.plot(co, so, lw=2.0, color=rd.color, label=rd.label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    pl.finish(ax, "Fact 16 - aggressive-order size CCDF by regime", "size (BTC)",
              "P(size > x)", legend=True)
    pl.layout(fig, rect=(0, 0, 1, 1.0))
    ctx.out.save_figure(fig, FACT, "trade_size", "comparison")

    return {
        "fact": FACT, "id": FACT_ID, "priority": PRIORITY, "title": TITLE,
        "metrics": metrics,
        "literature": "Schnaubelt et al.: many small trades, heavy upper tail, "
                      "round-number clustering.",
    }
