"""
Stylized fact 13 (P1) - Queue imbalance predicts the near-term price move.

Empirical claim
    Top-of-book bid/ask queue imbalance has statistically significant predictive
    power for the next mid-price move.  Gould & Bonart (2016) find the effect
    strongest for large-tick instruments and weaker but still meaningful for
    small-tick ones.  This is *not* Bitcoin-specific evidence, so the source
    document is explicit that it must be verified on the venue before any
    numerical target is set.

Why it matters for the simulator
    Once a liquidation policy reacts to book state - and certainly once it is
    trained by RL - the agent will exploit whatever predictability the simulator
    contains.  If that predictability is an artefact, the policy is worthless.

Method
    Queue imbalance is

        I = (Q_bid - Q_ask) / (Q_bid + Q_ask)

    at the touch, from the quote tape (which records every top-of-book change).
    Predictability is measured as the mean forward mid change by imbalance bin
    at several quote-event horizons, plus the sign-agreement rate ("directional
    accuracy") over moves that are actually non-zero.

    A caveat this framework can measure and therefore reports: BTC/USD on this
    venue is a **small-tick** instrument (the spread is one 0.1-USD tick ~99% of
    the time but the touch queue is tiny relative to depth), which is the regime
    where Gould-Bonart find the weaker effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfcore import plotting as pl
from sfcore import stats as st

FACT = "fact13"
FACT_ID = 13
PRIORITY = "P1"
TITLE = "Queue imbalance predicts the near-term price direction"

N_IMBALANCE_BINS = 20


def _forward_returns(mid: np.ndarray, horizon: int) -> np.ndarray:
    out = np.full(mid.size, np.nan)
    if horizon < mid.size:
        out[:-horizon] = (mid[horizon:] - mid[:-horizon]) / mid[:-horizon] * 1e4
    return out


def run(ctx) -> dict:
    pl.apply_style()
    cfg = ctx.cfg
    metrics: list[dict] = []
    curves, accuracy = {}, {}

    for name, rd in ctx.each():
        imb = rd.q_imbalance
        mid = rd.q_mid

        bins = st.quantile_bins(imb[np.isfinite(imb)], N_IMBALANCE_BINS)
        finite = np.isfinite(imb)
        bin_all = np.full(imb.size, -1)
        bin_all[finite] = bins
        mean_imb, counts = st.group_mean(imb, bin_all, N_IMBALANCE_BINS)

        frames, acc_rows = [], []
        for h in cfg.QUOTE_HORIZONS:
            fwd = _forward_returns(mid, h)
            means, n = st.group_mean(fwd, bin_all, N_IMBALANCE_BINS)
            sem = st.group_sem(fwd, bin_all, N_IMBALANCE_BINS)
            frames.append(pd.DataFrame({
                "horizon_quote_events": h,
                "imbalance_bin": np.arange(N_IMBALANCE_BINS),
                "mean_imbalance": mean_imb,
                "mean_forward_return_bps": means,
                "sem_bps": sem, "n": n,
            }))
            ok = np.isfinite(fwd) & np.isfinite(imb) & (fwd != 0)
            hit = float(np.mean(np.sign(imb[ok]) == np.sign(fwd[ok]))) if ok.sum() else np.nan
            acc_rows.append({
                "horizon_quote_events": h,
                "directional_accuracy": hit,
                "correlation": st.safe_corr(imb, fwd),
                "n": int(ok.sum()),
            })
        curve = pd.concat(frames, ignore_index=True)
        acc = pd.DataFrame(acc_rows)
        curves[name], accuracy[name] = curve, acc

        ctx.out.save_table(curve, FACT, "imbalance_response", name,
                           caption=f"Forward mid move by queue-imbalance bin - {rd.label}")
        ctx.out.save_table(acc, FACT, "predictive_power", name,
                           caption=f"Queue-imbalance predictive power by horizon - {rd.label}")

        fig, axes = pl.new_figure(1, 3, width=5.2, height=4.2)
        ax = axes[0]
        for h in (1, 10, 100, 1000):
            sub = curve[curve.horizon_quote_events == h]
            if len(sub):
                ax.plot(sub.mean_imbalance, sub.mean_forward_return_bps, "o-", ms=3.5,
                        lw=1.5, label=f"h={h} quote events")
        ax.axhline(0.0, **pl.REFERENCE_KW)
        ax.axvline(0.0, color="#9CA3AF", lw=0.8)
        pl.finish(ax, "Forward mid move vs queue imbalance",
                  "top-of-book queue imbalance", "mean forward mid change (bps)",
                  legend=True)

        ax = axes[1]
        ax.plot(acc.horizon_quote_events, acc.directional_accuracy * 100, "o-", ms=5, lw=1.8)
        ax.axhline(50.0, **pl.REFERENCE_KW)
        ax.set_xscale("log")
        pl.finish(ax, "Directional accuracy", "horizon (quote events)",
                  "sign agreement (%)",
                  note="50% is the no-information baseline.")

        ax = axes[2]
        hist_counts = rd.accumulators["qi"]
        qbins = hist_counts.size
        centres = np.linspace(-1, 1, qbins + 1)
        centres = 0.5 * (centres[:-1] + centres[1:])
        ax.fill_between(centres, hist_counts / hist_counts.sum(), alpha=0.5,
                        color="#6B8FBF")
        pl.finish(ax, "Distribution of queue imbalance (time-weighted)",
                  "queue imbalance", "density")
        fig.suptitle(f"Fact 13 - queue imbalance | {rd.label}", x=0.02, ha="left",
                     fontsize=12.5, fontweight="semibold")
        pl.layout(fig)
        ctx.out.save_figure(fig, FACT, "queue_imbalance", name)

        best = acc.iloc[acc.directional_accuracy.astype(float).idxmax()] if len(acc) else None
        metrics += [
            {"metric": "directional accuracy at h=1 quote event", "regime": name,
             "value": float(acc.directional_accuracy.iloc[0]), "detail": "0.5 = no information"},
            {"metric": "best directional accuracy", "regime": name,
             "value": float(best.directional_accuracy) if best is not None else np.nan,
             "detail": f"at h={int(best.horizon_quote_events)} quote events" if best is not None else ""},
            {"metric": "imbalance / forward-return correlation at h=10", "regime": name,
             "value": float(acc.loc[acc.horizon_quote_events == 10, "correlation"].iloc[0])
             if (acc.horizon_quote_events == 10).any() else np.nan, "detail": ""},
        ]

    fig, axes = pl.new_figure(1, 2, width=5.8, height=4.4)
    for name, rd in ctx.each():
        sub = curves[name][curves[name].horizon_quote_events == 10]
        axes[0].plot(sub.mean_imbalance, sub.mean_forward_return_bps, "o-", ms=4,
                     lw=1.8, color=rd.color, label=rd.label)
        a = accuracy[name]
        axes[1].plot(a.horizon_quote_events, a.directional_accuracy * 100, "o-", ms=4,
                     lw=1.8, color=rd.color, label=rd.label)
    axes[0].axhline(0.0, **pl.REFERENCE_KW)
    pl.finish(axes[0], "Forward mid move vs imbalance (h=10)",
              "top-of-book queue imbalance", "mean forward mid change (bps)", legend=True)
    axes[1].axhline(50.0, **pl.REFERENCE_KW)
    axes[1].set_xscale("log")
    pl.finish(axes[1], "Directional accuracy by horizon", "horizon (quote events)",
              "sign agreement (%)", legend=True)
    fig.suptitle("Fact 13 - queue imbalance by regime", x=0.02, ha="left",
                 fontsize=12.5, fontweight="semibold")
    pl.layout(fig, rect=(0, 0, 1, 0.94))
    ctx.out.save_figure(fig, FACT, "queue_imbalance", "comparison")

    return {
        "fact": FACT, "id": FACT_ID, "priority": PRIORITY, "title": TITLE,
        "metrics": metrics,
        "literature": "Gould & Bonart (2016): significant predictive power, strongest "
                      "for large-tick instruments (not Bitcoin-specific evidence).",
    }
