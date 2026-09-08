"""
Stylized fact 9b (P0-C) - lag-l impact in ZJ's units, R(v, 1), and R+-(1).

Source
    ``Stylied facts ZJ.pdf`` p12, p13 and p14:

        R(l)     = < eps_t * (m_{t+l} - m_t) >
        R(v, l)  = < eps_t * (m_{t+l} - m_t) | v_t = v >
        R+-(1)   = < eps_t * (m_{t+1} - m_t) | eps_{t-1} * eps_t = +-1 >

    ``fact09_response`` already reports R(l), but in bps on a log lag axis and
    conditioned on size *quintiles*.  Two of ZJ's three panels are therefore
    not reproducible from it:

    * p13 conditions on volume normalised by the mean size resting at the best
      quote, ``v / V_best``, over three decades - a quintile split cannot show
      that shape;
    * p14, the sign-conditioned immediate response, is absent entirely, and it
      is the one ZJ flag as contradicting the literature.

    This module adds those and re-plots R(l) on ZJ's axes (dollars, linear lag,
    log response) so the two documents can be laid side by side.

On the R+-(1) claim
    The textbook expectation is R+(1) < R-(1): a trade that continues the
    previous direction is largely anticipated, so it moves the price less than
    one that reverses it.  ZJ find R+(1) > R-(1) on three of four pairs and
    read it as cryptos trending more than expected.

    That reading needs care.  R+ and R- are averages over *different
    subsamples*, and those subsamples differ in more than sign history:
    continuation trades arrive in bursts, and so are conditioned on a state
    with a thinner queue and a wider spread.  This module therefore reports the
    raw split, the size and book state of each subsample, and the split after
    the immediate response is scaled by the pre-trade spread - so a difference
    that is really a state difference is visible as one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfcore import plotting as pl
from sffacts.fact19_arrivals import _book_averages

FACT = "fact09b"
FACT_ID = 9
PRIORITY = "P0-C"
TITLE = "Lag-l impact in price units, R(v,1) and sign-conditioned R+-(1)"

#: ZJ p12 plots lags 1..80 on a linear axis.
MAX_LAG = 80

#: Log-spaced bins of normalised trade volume for ZJ p13.
N_VOLUME_BINS = 12

#: Lags at which the sign-conditioned response is followed out.
CONDITIONAL_LAGS: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 40, 80)


def _response(sign, mid, lags, mask=None) -> pd.DataFrame:
    """R(l) in USD and bps, over the rows selected by ``mask``."""
    n = mid.size
    base = np.arange(n) if mask is None else np.flatnonzero(mask)
    rows = []
    for lag in lags:
        j = base[base + lag < n]
        if j.size < 50:
            continue
        moved = sign[j] * (mid[j + lag] - mid[j])
        rows.append({
            "lag": int(lag), "n": int(j.size),
            "response_usd": float(np.mean(moved)),
            "sem_usd": float(np.std(moved) / np.sqrt(j.size)),
            "response_bps": float(np.mean(moved / mid[j] * 1e4)),
            "sem_bps": float(np.std(moved / mid[j] * 1e4) / np.sqrt(j.size)),
        })
    return pd.DataFrame(rows)


def _volume_conditioned(sign, mid, qty, best_volume, lag=1) -> pd.DataFrame:
    """R(v, lag) against v / V_best, on log-spaced volume bins (ZJ p13)."""
    n = mid.size
    usable = np.arange(n - lag)
    normalised = qty[usable] / best_volume
    positive = normalised[normalised > 0]
    if positive.size < 500:
        return pd.DataFrame()
    edges = np.logspace(np.log10(np.percentile(positive, 0.5)),
                        np.log10(np.percentile(positive, 99.5)), N_VOLUME_BINS + 1)
    which = np.digitize(normalised, edges) - 1
    moved = sign[usable] * (mid[usable + lag] - mid[usable])
    rows = []
    for b in range(N_VOLUME_BINS):
        pick = which == b
        if pick.sum() < 50:
            continue
        rows.append({
            "volume_bin": b + 1,
            "mean_normalised_volume": float(np.mean(normalised[pick])),
            "mean_qty_btc": float(np.mean(qty[usable][pick])),
            "response_usd": float(np.mean(moved[pick])),
            "sem_usd": float(np.std(moved[pick]) / np.sqrt(pick.sum())),
            "response_bps": float(np.mean(moved[pick] / mid[usable][pick] * 1e4)),
            "n": int(pick.sum()),
        })
    return pd.DataFrame(rows)


def _sign_conditioned(sign, mid, spread_bps, qty, lags) -> pd.DataFrame:
    """R+-(l), plus the state each subsample was conditioned into (ZJ p14)."""
    continuation = np.zeros(sign.size, dtype=bool)
    reversal = np.zeros(sign.size, dtype=bool)
    same = sign[1:] * sign[:-1]
    continuation[1:] = same > 0
    reversal[1:] = same < 0
    frames = []
    for label, mask in (("continuation (+)", continuation), ("reversal (-)", reversal)):
        table = _response(sign, mid, lags, mask=mask)
        if table.empty:
            continue
        table["condition"] = label
        table["share_of_trades"] = float(mask.mean())
        table["median_qty_btc"] = float(np.median(qty[mask]))
        table["mean_pre_trade_spread_bps"] = float(np.nanmean(spread_bps[mask]))
        # Same response measured in units of the spread it faced: if the split
        # is really a state difference, this ratio narrows the gap.
        frames.append(table)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["response_over_half_spread"] = (out["response_bps"]
                                        / (out["mean_pre_trade_spread_bps"] / 2.0))
    return out


def run(ctx) -> dict:
    pl.apply_style()
    metrics: list[dict] = []
    responses, volumes, conditionals = {}, {}, {}

    for name, rd in ctx.each():
        trades = rd.trades
        sign = trades["sign"].to_numpy(np.float64)
        mid = trades["mid"].to_numpy(np.float64)
        qty = trades["qty"].to_numpy(np.float64)
        spread_bps = trades["spread_bps"].to_numpy(np.float64)
        best_volume = _book_averages(rd, ctx.cfg.TICK_SIZE).get(
            "mean_total_volume_at_best_btc", np.nan)

        curve = _response(sign, mid, np.arange(1, MAX_LAG + 1))
        volume = _volume_conditioned(sign, mid, qty, best_volume)
        conditional = _sign_conditioned(sign, mid, spread_bps, qty, CONDITIONAL_LAGS)
        responses[name], volumes[name], conditionals[name] = curve, volume, conditional

        ctx.out.save_table(curve, FACT, "lag_impact_usd", name,
                           caption=f"R(l) in USD and bps, lags 1-{MAX_LAG} - {rd.label}")
        if len(volume):
            ctx.out.save_table(volume, FACT, "impact_by_normalised_volume", name,
                               caption=f"R(v,1) vs v / V_best - {rd.label}")
        if len(conditional):
            ctx.out.save_table(conditional, FACT, "sign_conditioned_response", name,
                               caption=f"R+-(l) by previous-sign agreement - {rd.label}")

        fig, axes = pl.new_figure(1, 3, width=5.2, height=4.2)
        ax = axes[0]
        ax.plot(curve["lag"], curve["response_usd"], "o-", ms=3.5, lw=1.6,
                color=rd.color)
        ax.set_yscale("log")
        pl.finish(ax, r"Lag-$l$ impact $R(l)$", "lag $l$ (aggressive orders)",
                  "signed mid change (USD)",
                  note="Axes follow ZJ p12: linear lag, log response, price units. "
                       "Growth that flattens is the Bouchaud et al. signature.")

        ax = axes[1]
        if len(volume):
            ax.errorbar(volume["mean_normalised_volume"], volume["response_usd"],
                        yerr=volume["sem_usd"], fmt="o-", ms=5, lw=1.6, capsize=3,
                        color=rd.color)
        ax.axhline(0.0, **pl.REFERENCE_KW)
        ax.set_xscale("log")
        pl.finish(ax, r"Impact conditioned on volume $R(v,1)$",
                  r"normalised volume $v / \overline{V}_{best}$",
                  "lag-1 impact (USD)",
                  note=f"V_best = {best_volume:.4g} BTC, the time-weighted mean "
                       "total volume at the best quote. ZJ p13 see a rise only "
                       "for BTC.")

        ax = axes[2]
        if len(conditional):
            for label, colour, marker in (("continuation (+)", "#2C6FBB", "o"),
                                          ("reversal (-)", "#C0392B", "s")):
                sub = conditional.loc[conditional.condition == label]
                ax.errorbar(sub["lag"], sub["response_usd"], yerr=sub["sem_usd"],
                            fmt=marker + "-", ms=5, lw=1.6, capsize=3, color=colour,
                            label=label)
            ax.axhline(0.0, **pl.REFERENCE_KW)
            ax.set_xscale("log")
            first = conditional.loc[conditional.lag == 1].set_index("condition")
            note = ("R+(1)={:.4g} vs R-(1)={:.4g} USD. The two subsamples are "
                    "conditioned into different book states (spreads {:.3g} vs "
                    "{:.3g} bps), so the gap is not by itself a statement about "
                    "trend continuation.").format(
                first.loc["continuation (+)", "response_usd"],
                first.loc["reversal (-)", "response_usd"],
                first.loc["continuation (+)", "mean_pre_trade_spread_bps"],
                first.loc["reversal (-)", "mean_pre_trade_spread_bps"])
        else:
            note = ""
        pl.finish(ax, r"Response by previous-sign agreement",
                  "lag $l$ (aggressive orders)", "signed mid change (USD)",
                  legend=len(conditional) > 0, note=note)
        fig.suptitle(f"Fact 9b - conditioned trade response | {rd.label}", x=0.02,
                     ha="left", fontsize=12.5, fontweight="semibold")
        pl.layout(fig)
        ctx.out.save_figure(fig, FACT, "conditional_response", name)

        if len(conditional):
            first = conditional.loc[conditional.lag == 1].set_index("condition")
            plus = float(first.loc["continuation (+)", "response_usd"])
            minus = float(first.loc["reversal (-)", "response_usd"])
            metrics += [
                {"metric": "R+(1) after a same-sign trade (USD)", "regime": name,
                 "value": plus, "detail": "ZJ p14 BTC: 0.1157"},
                {"metric": "R-(1) after an opposite-sign trade (USD)", "regime": name,
                 "value": minus, "detail": "ZJ p14 BTC: 0.0041"},
                {"metric": "R+(1) - R-(1) (USD)", "regime": name,
                 "value": plus - minus,
                 "detail": "literature expects < 0; ZJ found > 0 on 3 of 4 pairs"},
            ]
        metrics.append(
            {"metric": f"R({MAX_LAG}) / R(1) growth factor", "regime": name,
             "value": float(curve["response_usd"].iloc[-1] / curve["response_usd"].iloc[0])
                      if len(curve) and curve["response_usd"].iloc[0] != 0 else np.nan,
             "detail": "slow growth = adaptive liquidity absorbing persistent flow"})

    fig, axes = pl.new_figure(1, 2, width=5.8, height=4.4)
    for name, rd in ctx.each():
        curve = responses.get(name)
        if curve is not None and len(curve):
            axes[0].plot(curve["lag"], curve["response_usd"], "o-", ms=3.5, lw=1.7,
                         color=rd.color, label=rd.label)
        volume = volumes.get(name)
        if volume is not None and len(volume):
            axes[1].plot(volume["mean_normalised_volume"], volume["response_usd"],
                         "o-", ms=5, lw=1.7, color=rd.color, label=rd.label)
    axes[0].set_yscale("log")
    pl.finish(axes[0], r"Lag-$l$ impact by regime", "lag $l$ (aggressive orders)",
              "signed mid change (USD)", legend=True)
    axes[1].axhline(0.0, **pl.REFERENCE_KW)
    axes[1].set_xscale("log")
    pl.finish(axes[1], r"$R(v,1)$ by regime",
              r"normalised volume $v / \overline{V}_{best}$", "lag-1 impact (USD)",
              legend=True)
    fig.suptitle("Fact 9b - conditioned trade response by regime", x=0.02,
                 ha="left", fontsize=12.5, fontweight="semibold")
    pl.layout(fig, rect=(0, 0, 1, 0.94))
    ctx.out.save_figure(fig, FACT, "conditional_response", "comparison")

    return {
        "fact": FACT, "id": FACT_ID, "priority": PRIORITY, "title": TITLE,
        "metrics": metrics,
        "literature": "ZJ p12-p14 (2026-08-25): R(l) rises and settles; R(v,1) rises "
                      "with volume for BTC only; R+(1) > R-(1), against the usual "
                      "expectation.",
    }
