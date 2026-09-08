"""
Stylized fact 8 (P0-C) - Order-flow imbalance (OFI) -> short-horizon price response.

Empirical claim
    Cont, Kukanov & Stoikov (2014) show short-horizon mid-price changes are
    approximately **linear in OFI**, with an impact coefficient **inversely
    related to market depth**.  The relation is more robust than raw
    volume-impact regressions.

Why it matters for the simulator
    A liquidation is an extreme signed-flow shock.  The simulator must
    reproduce dm | (OFI, depth), and in particular that a thinner book produces
    a larger response to the same imbalance.

Method
    OFI is built exactly as in Cont-Kukanov-Stoikov, from consecutive
    top-of-book states on the quote tape::

        e_n = 1{P_b(n) >= P_b(n-1)} q_b(n) - 1{P_b(n) <= P_b(n-1)} q_b(n-1)
            - 1{P_a(n) <= P_a(n-1)} q_a(n) + 1{P_a(n) >= P_a(n-1)} q_a(n-1)

    The quote tape carries **every** change in the top of book, so this is the
    exact event-by-event OFI, not a sampled approximation.  It is then summed
    over non-overlapping blocks of ``OFI_BLOCK_EVENTS`` quote events and
    regressed on the contemporaneous mid change, overall and within depth
    quartiles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfcore import plotting as pl
from sfcore import stats as st

FACT = "fact08"
FACT_ID = 8
PRIORITY = "P0-C"
TITLE = "Order-flow imbalance -> short-horizon price response"

#: Quote events per OFI block.  Cont et al. use fixed clock intervals; in event
#: time the natural analogue is a fixed number of book updates, which keeps the
#: blocks comparable across quiet and busy periods.
OFI_BLOCK_EVENTS = 50
N_OFI_BINS = 25
N_DEPTH_GROUPS = 4


def _ofi_series(rd) -> np.ndarray:
    """Event-by-event order-flow imbalance from the quote tape."""
    q = rd.quotes
    pb = q["bid_tick"].to_numpy(np.int64)
    pa = q["ask_tick"].to_numpy(np.int64)
    qb = q["bid_qty"].to_numpy(np.float64)
    qa = q["ask_qty"].to_numpy(np.float64)

    pb0, pb1 = pb[:-1], pb[1:]
    pa0, pa1 = pa[:-1], pa[1:]
    qb0, qb1 = qb[:-1], qb[1:]
    qa0, qa1 = qa[:-1], qa[1:]

    bid_term = np.where(pb1 >= pb0, qb1, 0.0) - np.where(pb1 <= pb0, qb0, 0.0)
    ask_term = np.where(pa1 <= pa0, qa1, 0.0) - np.where(pa1 >= pa0, qa0, 0.0)
    e = bid_term - ask_term
    return np.concatenate([[0.0], e])


def _blocks(rd, block: int):
    """Aggregate OFI and mid changes over non-overlapping event blocks."""
    ofi = _ofi_series(rd)
    mid = rd.q_mid
    depth = (rd.quotes["bid_depth"].to_numpy(np.float64)
             + rd.quotes["ask_depth"].to_numpy(np.float64))
    n = (len(ofi) // block) * block
    if n < block * 20:
        return None
    ofi_b = ofi[:n].reshape(-1, block).sum(axis=1)
    mid_b = mid[:n].reshape(-1, block)
    dmid = mid_b[:, -1] - mid_b[:, 0]
    ref = mid_b[:, 0]
    dmid_bps = dmid / ref * 1e4
    depth_b = depth[:n].reshape(-1, block).mean(axis=1)
    return ofi_b, dmid_bps, depth_b


def run(ctx) -> dict:
    pl.apply_style()
    metrics: list[dict] = []
    curves, depth_tables = {}, {}

    for name, rd in ctx.each():
        packed = _blocks(rd, OFI_BLOCK_EVENTS)
        if packed is None:
            continue
        ofi, dmid_bps, depth = packed

        overall = st.ols(dmid_bps, ofi[:, None])
        corr = st.safe_corr(ofi, dmid_bps)

        # Linearity check: binned conditional mean.
        bins = st.quantile_bins(ofi, N_OFI_BINS)
        mean_dmid, counts = st.group_mean(dmid_bps, bins, N_OFI_BINS)
        sem = st.group_sem(dmid_bps, bins, N_OFI_BINS)
        mean_ofi, _ = st.group_mean(ofi, bins, N_OFI_BINS)
        curve = pd.DataFrame({"bin": np.arange(N_OFI_BINS), "mean_ofi_btc": mean_ofi,
                              "mean_dmid_bps": mean_dmid, "sem_bps": sem, "n": counts})
        curves[name] = curve

        # Depth dependence of the impact coefficient.
        dgroup = st.quantile_bins(depth, N_DEPTH_GROUPS)
        rows = []
        for g in range(N_DEPTH_GROUPS):
            m = dgroup == g
            if m.sum() < 200:
                continue
            fit = st.ols(dmid_bps[m], ofi[m][:, None])
            rows.append({
                "depth_quartile": g + 1,
                "mean_depth_btc": float(np.nanmean(depth[m])),
                "impact_coefficient_bps_per_btc": float(fit["beta"][1]),
                "std_error": float(fit["se"][1]),
                "r2": fit["r2"], "n": fit["n"],
            })
        dtab = pd.DataFrame(rows)
        depth_tables[name] = dtab

        ctx.out.save_table(curve, FACT, "ofi_response_curve", name,
                           caption=f"Mid change vs order-flow imbalance - {rd.label}")
        ctx.out.save_table(dtab, FACT, "impact_vs_depth", name,
                           caption=f"OFI impact coefficient by depth quartile - {rd.label}")

        fig, axes = pl.new_figure(1, 3, width=5.2, height=4.2)
        ax = axes[0]
        ax.errorbar(curve.mean_ofi_btc, curve.mean_dmid_bps, yerr=curve.sem_bps,
                    fmt="o", ms=4, lw=1.4, capsize=2)
        xs = np.linspace(np.nanmin(curve.mean_ofi_btc), np.nanmax(curve.mean_ofi_btc), 50)
        ax.plot(xs, overall["beta"][0] + overall["beta"][1] * xs, **pl.REFERENCE_KW)
        pl.finish(ax, "Linearity of the OFI response",
                  f"OFI over {OFI_BLOCK_EVENTS} quote events (BTC)", "mid change (bps)",
                  note=f"OLS slope {overall['beta'][1]:.4g} bps/BTC, R2={overall['r2']:.3f}, "
                       f"corr={corr:.3f}")

        ax = axes[1]
        if len(dtab):
            ax.errorbar(dtab.mean_depth_btc, dtab.impact_coefficient_bps_per_btc,
                        yerr=dtab.std_error, fmt="o-", ms=5, lw=1.6, capsize=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        pl.finish(ax, "Impact coefficient vs depth", "mean visible depth (BTC)",
                  "bps per BTC of OFI",
                  note="Cont-Kukanov-Stoikov: the coefficient should fall as depth rises.")

        ax = axes[2]
        if len(dtab):
            ax.bar(dtab.depth_quartile, dtab.r2, color="#6B8FBF", width=0.6)
            ax.set_xticks(dtab.depth_quartile)
        pl.finish(ax, "Explanatory power by depth quartile",
                  "depth quartile (1 = thinnest)", "regression R-squared")
        fig.suptitle(f"Fact 8 - order-flow imbalance | {rd.label}", x=0.02, ha="left",
                     fontsize=12.5, fontweight="semibold")
        pl.layout(fig)
        ctx.out.save_figure(fig, FACT, "ofi_response", name)

        ratio = np.nan
        if len(dtab) >= 2:
            ratio = float(dtab.impact_coefficient_bps_per_btc.iloc[0]
                          / dtab.impact_coefficient_bps_per_btc.iloc[-1])
        metrics += [
            {"metric": "OFI impact coefficient (bps per BTC)", "regime": name,
             "value": float(overall["beta"][1]),
             "detail": f"R2={overall['r2']:.3f}, n={overall['n']:,}"},
            {"metric": "OFI / mid-change correlation", "regime": name, "value": corr,
             "detail": f"blocks of {OFI_BLOCK_EVENTS} quote events"},
            {"metric": "impact coefficient, thinnest / thickest depth quartile",
             "regime": name, "value": ratio,
             "detail": "CKS predicts > 1 (inverse in depth)"},
        ]

    fig, axes = pl.new_figure(1, 2, width=5.8, height=4.4)
    for name, rd in ctx.each():
        if name in curves:
            c = curves[name]
            axes[0].plot(c.mean_ofi_btc, c.mean_dmid_bps, "o-", ms=4, lw=1.7,
                         color=rd.color, label=rd.label)
        if name in depth_tables and len(depth_tables[name]):
            d = depth_tables[name]
            axes[1].plot(d.mean_depth_btc, d.impact_coefficient_bps_per_btc, "o-",
                         ms=5, lw=1.7, color=rd.color, label=rd.label)
    pl.finish(axes[0], "OFI response by regime",
              f"OFI over {OFI_BLOCK_EVENTS} quote events (BTC)", "mid change (bps)",
              legend=True)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    pl.finish(axes[1], "Impact coefficient vs depth by regime",
              "mean visible depth (BTC)", "bps per BTC of OFI", legend=True)
    fig.suptitle("Fact 8 - order-flow imbalance by regime", x=0.02, ha="left",
                 fontsize=12.5, fontweight="semibold")
    pl.layout(fig, rect=(0, 0, 1, 0.94))
    ctx.out.save_figure(fig, FACT, "ofi_response", "comparison")

    return {
        "fact": FACT, "id": FACT_ID, "priority": PRIORITY, "title": TITLE,
        "metrics": metrics,
        "literature": "Cont, Kukanov & Stoikov (2014): mid changes linear in OFI, "
                      "coefficient inversely related to depth.",
    }
