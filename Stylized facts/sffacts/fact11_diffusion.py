"""
Stylized fact 11 (P0-C) - Persistent order flow with near-diffusive prices.

Empirical claim
    Despite strongly persistent order signs (fact 10), returns stay close to
    unpredictable because liquidity adapts.  Lillo-Farmer and Bouchaud et al.
    show how adaptive liquidity offsets predictable flow.  For Bitcoin,
    Schnaubelt et al. report little return autocorrelation **from minutes to
    days**, while *tick-level trade-price changes* show a negative first-lag
    autocorrelation from bid-ask bounce.

Why it matters for the simulator
    This is a **joint** constraint on the agents.  If persistent sell flow
    mechanically produces strongly predictable negative returns, the liquidity
    providers in the model are too passive.  Facts 10 and 11 must be matched
    together or not at all.

Method
    Four diagnostics, on the clocks the claims are actually made on.

    1. **Mid-return autocorrelation** in quote-event time, in aggressive-order
       time, and on 1 s / 60 s clock bars.  The literature statement is the
       clock-time one; the event-time versions are reported because that is the
       clock the simulator's agents live in, and the two need not agree.
    2. **Variance ratio** Var(sum of k returns) / (k Var(r)), in event time and
       on clock bars.  Near 1 is diffusive, below 1 mean-reverting, above 1
       trending.  This is the sharpest single test of the joint constraint.
    3. **Bid-ask bounce**: first-lag autocorrelation of trade-price changes,
       measured on the **aggressive-order** price series.  The individual-fill
       series is reported too but is *not* the right object for this test: one
       sweep prints several fills walking the book in one direction, which
       injects mechanical positive autocorrelation that swamps the bounce.
    4. **Signed-flow predictability**: the correlation between the aggressor
       sign and the following return, which is the quantity that must stay
       small even though the sign series itself is strongly persistent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfcore import plotting as pl
from sfcore import stats as st

FACT = "fact11"
FACT_ID = 11
PRIORITY = "P0-C"
TITLE = "Persistent order flow with approximately diffusive prices"

VR_HORIZONS = (2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 1000, 2000)
ACF_LAGS_SHOWN = 60
CLOCK_BARS_S = (1.0, 60.0)


def run(ctx) -> dict:
    pl.apply_style()
    metrics: list[dict] = []
    quote_acfs, clock_acfs, vrs = {}, {}, {}

    for name, rd in ctx.each():
        mid_q = rd.q_mid
        ret_q = np.diff(np.log(mid_q))
        mid_t = rd.trades["mid"].to_numpy(np.float64)
        ret_t = np.diff(np.log(mid_t))

        # Bid-ask bounce: one price per aggressive order, not per fill.
        px_order = rd.trades["vwap"].to_numpy(np.float64)
        ret_order_px = np.diff(np.log(px_order))
        px_fill = rd.fills["price"].to_numpy(np.float64)
        ret_fill_px = np.diff(np.log(px_fill))

        acf_q = st.acf(ret_q, ACF_LAGS_SHOWN)
        acf_t = st.acf(ret_t, ACF_LAGS_SHOWN)
        acf_order_px = st.acf(ret_order_px, ACF_LAGS_SHOWN)
        acf_fill_px = st.acf(ret_fill_px, ACF_LAGS_SHOWN)
        quote_acfs[name] = acf_q

        # Clock-time bars: the scale the literature claim is made on.
        bar_acfs, bar_rows = {}, []
        for seconds in CLOCK_BARS_S:
            bars = st.clock_bars(rd.q_recv, mid_q, seconds)
            if bars.size < 200:
                continue
            ret_bar = np.diff(np.log(bars))
            ret_bar = ret_bar[np.isfinite(ret_bar)]
            a = st.acf(ret_bar, min(ACF_LAGS_SHOWN, max(ret_bar.size // 10, 5)))
            bar_acfs[seconds] = a
            vr_bar = st.variance_ratio(ret_bar, VR_HORIZONS)
            far = [v for k, v in zip(VR_HORIZONS, vr_bar) if k >= 10 and np.isfinite(v)]
            bar_rows.append({
                "bar_seconds": seconds,
                "n_bars": int(ret_bar.size),
                "acf_lag1": float(a[0]),
                "mean_abs_acf_lags_2_20": float(np.nanmean(np.abs(a[1:20]))),
                "noise_floor": float(1.96 / np.sqrt(ret_bar.size)),
                "variance_ratio_k10plus": float(np.mean(far)) if far else np.nan,
            })
        clock_acfs[name] = bar_acfs
        clock_table = pd.DataFrame(bar_rows)

        vr_q = st.variance_ratio(ret_q, VR_HORIZONS)
        vr_t = st.variance_ratio(ret_t, VR_HORIZONS)
        vrs[name] = (vr_q, vr_t)

        acf_table = pd.DataFrame({
            "lag": np.arange(1, ACF_LAGS_SHOWN + 1),
            "acf_mid_return_quote_time": acf_q,
            "acf_mid_return_order_time": acf_t,
            "acf_trade_price_change_order_level": acf_order_px,
            "acf_trade_price_change_fill_level": acf_fill_px,
        })
        vr_table = pd.DataFrame({"k": VR_HORIZONS,
                                 "variance_ratio_quote_time": vr_q,
                                 "variance_ratio_order_time": vr_t})
        ctx.out.save_table(acf_table, FACT, "return_autocorrelation", name,
                           caption=f"Return autocorrelation - {rd.label}")
        ctx.out.save_table(vr_table, FACT, "variance_ratio", name,
                           caption=f"Variance ratios in event time - {rd.label}")
        if len(clock_table):
            ctx.out.save_table(clock_table, FACT, "clock_time_diffusivity", name,
                               caption=f"Clock-bar return diagnostics - {rd.label}")

        # Bid-ask bounce, split by whether the aggressor side flipped.  With
        # order signs as persistent as they are here (fact 10), consecutive
        # trades sit on the same side often enough to mask the bounce in the
        # raw series; conditioning on a sign flip recovers it.
        sign_all = rd.trades["sign"].to_numpy(np.float64)
        dp = np.diff(px_order)
        flip = sign_all[2:] != sign_all[1:-1]
        bounce_flip = st.safe_corr(dp[1:][flip], dp[:-1][flip])
        bounce_same = st.safe_corr(dp[1:][~flip], dp[:-1][~flip])

        # --- figure ------------------------------------------------------
        noise_q = 1.96 / np.sqrt(ret_q.size)
        lags = np.arange(1, ACF_LAGS_SHOWN + 1)
        fig, axes = pl.new_figure(2, 2, width=5.4, height=3.9)

        ax = axes[0][0]
        ax.bar(lags - 0.2, acf_q, width=0.4, label="mid, quote time", color="#6B8FBF")
        ax.bar(lags + 0.2, acf_t, width=0.4, label="mid, order time", color="#A3B8D4")
        ax.axhspan(-noise_q, noise_q, color="#D1D5DB", alpha=0.55, zorder=0)
        ax.axhline(0.0, color="#4B5563", lw=0.8)
        pl.finish(ax, "Mid-return autocorrelation (event time)", "lag (events)",
                  "autocorrelation", legend=True,
                  note="Shaded: 95% white-noise band for the quote series.")

        ax = axes[0][1]
        for seconds, a in bar_acfs.items():
            ax.plot(np.arange(1, a.size + 1), a, "o-", ms=3, lw=1.3,
                    label=f"{seconds:g}s bars")
        ax.axhline(0.0, color="#4B5563", lw=0.8)
        pl.finish(ax, "Mid-return autocorrelation (clock time)", "lag (bars)",
                  "autocorrelation", legend=True,
                  note="Schnaubelt et al.: little autocorrelation from minutes to days.")

        ax = axes[1][0]
        ax.bar(lags - 0.2, acf_order_px, width=0.4, color="#C0392B",
               label="aggressive-order price")
        ax.bar(lags + 0.2, acf_fill_px, width=0.4, color="#E8A0A0",
               label="individual fill price")
        ax.axhline(0.0, color="#4B5563", lw=0.8)
        pl.finish(ax, "Trade-price change autocorrelation", "lag (trades)",
                  "autocorrelation", legend=True,
                  note=f"Both series are positive at lag 1 because order signs are "
                       f"persistent (fact 10). Conditioning on a sign flip recovers the "
                       f"bounce: {bounce_flip:+.4f} after a flip vs {bounce_same:+.4f} "
                       f"after a repeat.")

        ax = axes[1][1]
        ax.plot(VR_HORIZONS, vr_q, "o-", ms=4, lw=1.7, label="quote-event time")
        ax.plot(VR_HORIZONS, vr_t, "s-", ms=4, lw=1.7, label="aggressive-order time")
        for seconds in CLOCK_BARS_S:
            bars = st.clock_bars(rd.q_recv, mid_q, seconds)
            if bars.size < 200:
                continue
            vr_bar = st.variance_ratio(np.diff(np.log(bars)), VR_HORIZONS)
            ax.plot(VR_HORIZONS, vr_bar, "^--", ms=4, lw=1.4, alpha=0.85,
                    label=f"{seconds:g}s clock bars")
        ax.axhline(1.0, **pl.REFERENCE_KW)
        ax.set_xscale("log")
        pl.finish(ax, "Variance ratio", "aggregation k", "Var(sum of k) / (k Var)",
                  legend=True,
                  note="1.0 = diffusive. Event-time and clock-time answers can differ: "
                       "event time compresses busy, trending periods.")
        fig.suptitle(f"Fact 11 - diffusivity under persistent flow | {rd.label}",
                     x=0.02, ha="left", fontsize=12.5, fontweight="semibold")
        pl.layout(fig)
        ctx.out.save_figure(fig, FACT, "diffusion", name)

        # --- metrics -----------------------------------------------------
        # Predictability, not impact: pair the sign of order t with the return
        # from t+1 to t+2, i.e. *after* order t has already moved the mid.
        # Pairing it with the t -> t+1 return would just re-measure R(1).
        flow_pred = st.safe_corr(sign_all[:-2], ret_t[1:])
        contemporaneous = st.safe_corr(sign_all[:-1], ret_t)
        far_q = [v for k, v in zip(VR_HORIZONS, vr_q) if k >= 100 and np.isfinite(v)]
        metrics += [
            {"metric": "mid-return ACF lag 1 (quote-event time)", "regime": name,
             "value": float(acf_q[0]), "detail": f"95% band +/-{noise_q:.4f}"},
            {"metric": "mid-return ACF lag 1 (aggressive-order time)", "regime": name,
             "value": float(acf_t[0]), "detail": ""},
            {"metric": "trade-price change ACF lag 1, order level (raw)",
             "regime": name, "value": float(acf_order_px[0]),
             "detail": "sign persistence pushes this positive; see the split below"},
            {"metric": "trade-price change ACF lag 1, after a sign flip (bid-ask bounce)",
             "regime": name, "value": bounce_flip,
             "detail": "the Roll bounce, isolated from order-sign persistence"},
            {"metric": "trade-price change ACF lag 1, after a repeated sign",
             "regime": name, "value": bounce_same,
             "detail": "continuation of a sweep in the same direction"},
            {"metric": "trade-price change ACF lag 1, fill level", "regime": name,
             "value": float(acf_fill_px[0]),
             "detail": "positive by construction: sweeps walk the book one way"},
            {"metric": "variance ratio at k>=100 (quote-event time)", "regime": name,
             "value": float(np.mean(far_q)) if far_q else np.nan,
             "detail": "1.0 = diffusive; event time compresses busy trending periods"},
            {"metric": "aggressor sign / contemporaneous mid-return correlation",
             "regime": name, "value": contemporaneous,
             "detail": "this is impact (R(1)), not predictability"},
            {"metric": "aggressor sign / *subsequent* mid-return correlation",
             "regime": name, "value": flow_pred,
             "detail": "the predictability that must stay small even though the sign "
                       "series itself is strongly persistent"},
        ]
        for _, row in clock_table.iterrows():
            metrics += [
                {"metric": f"mid-return ACF lag 1 ({row.bar_seconds:g}s bars)",
                 "regime": name, "value": row.acf_lag1,
                 "detail": f"noise floor +/-{row.noise_floor:.4f}, n={int(row.n_bars):,}"},
                {"metric": f"variance ratio at k>=10 ({row.bar_seconds:g}s bars)",
                 "regime": name, "value": row.variance_ratio_k10plus,
                 "detail": "the clock-time diffusivity the literature refers to"},
            ]

    # --- comparison ------------------------------------------------------
    fig, axes = pl.new_figure(1, 3, width=5.2, height=4.2)
    for name, rd in ctx.each():
        axes[0].plot(np.arange(1, ACF_LAGS_SHOWN + 1), quote_acfs[name], "o-", ms=3,
                     lw=1.3, color=rd.color, label=rd.label)
        axes[1].plot(VR_HORIZONS, vrs[name][0], "o-", ms=4, lw=1.8, color=rd.color,
                     label=rd.label)
        bars = clock_acfs[name].get(60.0)
        if bars is not None:
            axes[2].plot(np.arange(1, bars.size + 1), bars, "o-", ms=3, lw=1.3,
                         color=rd.color, label=rd.label)
    axes[0].axhline(0.0, color="#4B5563", lw=0.8)
    pl.finish(axes[0], "Mid-return ACF (quote-event time)", "lag (quote events)",
              "autocorrelation", legend=True)
    axes[1].axhline(1.0, **pl.REFERENCE_KW)
    axes[1].set_xscale("log")
    pl.finish(axes[1], "Variance ratio (quote-event time)", "aggregation k",
              "Var(sum of k) / (k Var)", legend=True)
    axes[2].axhline(0.0, color="#4B5563", lw=0.8)
    pl.finish(axes[2], "Mid-return ACF (60 s clock bars)", "lag (minutes)",
              "autocorrelation", legend=True)
    fig.suptitle("Fact 11 - price diffusivity by regime", x=0.02, ha="left",
                 fontsize=12.5, fontweight="semibold")
    pl.layout(fig, rect=(0, 0, 1, 0.94))
    ctx.out.save_figure(fig, FACT, "diffusion", "comparison")

    return {
        "fact": FACT, "id": FACT_ID, "priority": PRIORITY, "title": TITLE,
        "metrics": metrics,
        "literature": "Lillo-Farmer / Bouchaud et al.: adaptive liquidity keeps prices "
                      "near-diffusive despite persistent signs; Schnaubelt et al.: little "
                      "return autocorrelation from minutes to days, negative first-lag "
                      "autocorrelation in tick trade prices.",
    }
