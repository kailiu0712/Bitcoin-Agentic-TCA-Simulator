"""
Stylized fact 9 (P0-C) - Trade-response function R(l).

Empirical claim
    Signed trades move the price in their own direction, but the observable
    response is shaped by the correlated order flow that follows and by adaptive
    liquidity.  Bouchaud, Gefen, Potters & Wyart (2004) document a relatively
    stable, slowly-growing aggregate response despite strongly persistent order
    flow.

Why it matters for the simulator
    This tests whether individual liquidation slices have realistic immediate
    and delayed consequences, before any complete-metaorder experiment is run.

Method
    In aggressive-order (event) time,

        R(l) = E[ eps_0 * (m_l - m_0) ] / m_0 * 1e4   [bps]

    where ``eps_0`` is the aggressor sign of order 0 and ``m_l`` is the
    pre-trade mid of the l-th subsequent aggressive order.  Because every mid
    here is the exchange-time-aligned pre-trade mid carried on the trade tape,
    R(l) is measured on a single consistent clock.

    R(l) is also computed within size quintiles, and the immediate response
    R(1) is split into its "price moved" and "spread widened" parts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfcore import plotting as pl
from sfcore import stats as st

FACT = "fact09"
FACT_ID = 9
PRIORITY = "P0-C"
TITLE = "Trade-response function R(l)"

N_SIZE_GROUPS = 5


def _response(sign: np.ndarray, mid: np.ndarray, horizons, mask=None) -> pd.DataFrame:
    n = mid.size
    idx = np.arange(n) if mask is None else np.flatnonzero(mask)
    rows = []
    for l in horizons:
        j = idx[idx + l < n]
        if j.size < 50:
            continue
        r = sign[j] * (mid[j + l] - mid[j]) / mid[j] * 1e4
        rows.append({"horizon_orders": int(l), "response_bps": float(np.nanmean(r)),
                     "sem_bps": float(np.nanstd(r) / np.sqrt(j.size)), "n": int(j.size)})
    return pd.DataFrame(rows)


def run(ctx) -> dict:
    pl.apply_style()
    cfg = ctx.cfg
    metrics: list[dict] = []
    overall, by_size = {}, {}

    for name, rd in ctx.each():
        trades = rd.trades
        sign = trades["sign"].to_numpy(np.float64)
        mid = trades["mid"].to_numpy(np.float64)
        qty = trades["qty"].to_numpy(np.float64)

        resp = _response(sign, mid, cfg.EVENT_HORIZONS)
        overall[name] = resp
        ctx.out.save_table(resp, FACT, "response_function", name,
                           caption=f"Trade response R(l) in aggressive-order time - {rd.label}")

        groups = st.quantile_bins(qty, N_SIZE_GROUPS)
        frames = []
        for g in range(N_SIZE_GROUPS):
            sub = _response(sign, mid, cfg.EVENT_HORIZONS, mask=(groups == g))
            if len(sub):
                sub["size_quintile"] = g + 1
                sub["median_qty_btc"] = float(np.nanmedian(qty[groups == g]))
                frames.append(sub)
        sized = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        by_size[name] = sized
        if len(sized):
            ctx.out.save_table(sized, FACT, "response_by_size", name,
                               caption=f"R(l) by aggressive-order size quintile - {rd.label}")

        fig, axes = pl.new_figure(1, 3, width=5.2, height=4.2)
        ax = axes[0]
        ax.errorbar(resp.horizon_orders, resp.response_bps, yerr=resp.sem_bps,
                    fmt="o-", ms=4, lw=1.7, capsize=2)
        ax.axhline(0.0, **pl.REFERENCE_KW)
        ax.set_xscale("log")
        pl.finish(ax, "Response function R(l)", "aggressive orders after the trade",
                  "signed mid change (bps)",
                  note="A response that keeps growing slowly - rather than jumping and "
                       "reverting - is the Bouchaud et al. signature.")

        ax = axes[1]
        if len(sized):
            for g, sub in sized.groupby("size_quintile"):
                ax.plot(sub.horizon_orders, sub.response_bps, "o-", ms=3.5, lw=1.4,
                        label=f"Q{int(g)} (median {sub.median_qty_btc.iloc[0]:.3g} BTC)")
        ax.axhline(0.0, **pl.REFERENCE_KW)
        ax.set_xscale("log")
        pl.finish(ax, "R(l) by order size", "aggressive orders after the trade",
                  "signed mid change (bps)", legend=True)

        ax = axes[2]
        immediate = _response(sign, mid, [1])
        eff_spread = (trades["sign"] * (trades["vwap"] - trades["mid"])
                      / trades["mid"] * 1e4).to_numpy()
        groups_q = st.quantile_bins(qty, 10)
        eff_mean, _ = st.group_mean(eff_spread, groups_q, 10)
        r1 = np.full(10, np.nan)
        for g in range(10):
            sub = _response(sign, mid, [1], mask=(groups_q == g))
            if len(sub):
                r1[g] = sub.response_bps.iloc[0]
        qmed, _ = st.group_mean(qty, groups_q, 10)
        ax.plot(qmed, eff_mean, "o-", ms=4, lw=1.6, label="effective half-spread paid")
        ax.plot(qmed, r1, "s-", ms=4, lw=1.6, label="R(1): mid displacement")
        ax.set_xscale("log")
        pl.finish(ax, "Cost paid vs price displaced", "aggressive-order size (BTC)",
                  "bps", legend=True)
        fig.suptitle(f"Fact 9 - trade response function | {rd.label}", x=0.02,
                     ha="left", fontsize=12.5, fontweight="semibold")
        pl.layout(fig)
        ctx.out.save_figure(fig, FACT, "trade_response", name)

        def _at(l):
            row = resp[resp.horizon_orders == l]
            return float(row.response_bps.iloc[0]) if len(row) else np.nan

        growth = _at(100) / _at(1) if np.isfinite(_at(1)) and _at(1) != 0 else np.nan
        metrics += [
            {"metric": "R(1) immediate response (bps)", "regime": name,
             "value": _at(1), "detail": ""},
            {"metric": "R(100) (bps)", "regime": name, "value": _at(100), "detail": ""},
            {"metric": "R(1000) (bps)", "regime": name, "value": _at(1000), "detail": ""},
            {"metric": "R(100)/R(1) growth factor", "regime": name, "value": growth,
             "detail": "slow growth = adaptive liquidity absorbing persistent flow"},
            {"metric": "mean effective half-spread paid (bps)", "regime": name,
             "value": float(np.nanmean(eff_spread)), "detail": ""},
        ]

    fig, ax = pl.new_figure(1, 1, width=7.4, height=4.6)
    for name, rd in ctx.each():
        r = overall[name]
        ax.errorbar(r.horizon_orders, r.response_bps, yerr=r.sem_bps, fmt="o-",
                    ms=4, lw=1.8, capsize=2, color=rd.color, label=rd.label)
    ax.axhline(0.0, **pl.REFERENCE_KW)
    ax.set_xscale("log")
    pl.finish(ax, "Fact 9 - trade response function by regime",
              "aggressive orders after the trade", "signed mid change (bps)", legend=True)
    pl.layout(fig, rect=(0, 0, 1, 1.0))
    ctx.out.save_figure(fig, FACT, "trade_response", "comparison")

    return {
        "fact": FACT, "id": FACT_ID, "priority": PRIORITY, "title": TITLE,
        "metrics": metrics,
        "literature": "Bouchaud, Gefen, Potters & Wyart (2004): stable, slowly "
                      "growing aggregate response despite persistent order flow.",
    }
