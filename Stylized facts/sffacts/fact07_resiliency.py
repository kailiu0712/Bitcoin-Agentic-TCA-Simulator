"""
Stylized fact 7 (P0-C) - Liquidity resiliency after large aggressive trades.

Empirical claim
    Large BTC trades widen the spread and liquidity then recovers.  In the
    Coinbase BTC/USD evidence of Schnaubelt, Rende & Krauss the spread-recovery
    exponential has a time constant of about **6.3 seconds**, and deeper-book
    VWAP liquidity recovers more slowly than the top-of-book spread.

Why it matters for the simulator
    This is one of the most important TWAP calibration targets.  Recover too
    fast and order splitting is unrealistically cheap; too slow and TWAP is
    unrealistically expensive.

Method
    Large orders are those above the ``LARGE_TRADE_QUANTILE`` of aggressive-order
    size within the regime.  Two clocks are used:

    * **event time** (primary) - the state seen by the k-th subsequent
      aggressive order.  This is the clock a liquidation algorithm actually
      lives in, and it is immune to the feed's latency jitter.
    * **clock time** (secondary) - the quote tape sampled at fixed second
      offsets after the event, because the literature's 6.3 s target is a
      clock-time statement.  Offsets are taken on the exchange clock.

    Both spread and *consumed-side* visible depth are tracked, since the claim
    is specifically that depth recovers more slowly than spread.

    One thing to read carefully: large orders do not arrive into an average
    book.  They arrive when the spread is already wider and the consumed side
    already thinner than usual, so "recovery" runs *past* the pre-event level
    and settles at the unconditional mean.  Every recovery panel therefore also
    draws the unconditional, time-weighted level, which is the steady state the
    book is actually returning to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfcore import plotting as pl
from sfcore import stats as st

FACT = "fact07"
FACT_ID = 7
PRIORITY = "P0-C"
TITLE = "Liquidity resiliency after large aggressive trades"

LITERATURE_TAU_S = 6.3


def _unconditional(rd, cfg) -> tuple[float, float]:
    """Time-weighted mean spread (bps) and one-sided depth (BTC) over all states."""
    acc = rd.accumulators
    hour = acc["hour"]
    spread = float(hour[1].sum() / hour[0].sum()) if hour[0].sum() > 0 else np.nan
    total_w = float(acc["total_weight_ns"][0])
    band_idx = cfg.QUOTE_TAPE_DEPTH_BAND_INDEX
    depth = acc["depth_sum"]
    one_sided = float((depth[band_idx, 0] + depth[band_idx, 1]) / (2 * total_w))
    return spread, one_sided


def _event_time_profile(rd, cfg, mask: np.ndarray) -> pd.DataFrame:
    """Spread and consumed-side depth at the k-th following aggressive order."""
    trades = rd.trades
    n = len(trades)
    spread = trades["spread_bps"].to_numpy(np.float64)
    sign = trades["sign"].to_numpy(np.float64)
    band = cfg.DEPTH_BANDS_BPS[cfg.QUOTE_TAPE_DEPTH_BAND_INDEX]
    bid_d = trades[f"depth_bid_{band:g}bps"].to_numpy(np.float64)
    ask_d = trades[f"depth_ask_{band:g}bps"].to_numpy(np.float64)
    # Depth on the side the aggressive order consumes.
    consumed = np.where(sign > 0, ask_d, bid_d)

    idx = np.flatnonzero(mask)
    rows = []
    base_spread = np.nanmean(spread[idx])
    base_depth = np.nanmean(consumed[idx])
    for k in [0] + list(cfg.EVENT_HORIZONS):
        j = idx + k
        j = j[j < n]
        if j.size < 30:
            continue
        rows.append({
            "horizon_orders": k,
            "mean_spread_bps": float(np.nanmean(spread[j])),
            "mean_consumed_depth_btc": float(np.nanmean(consumed[j])),
            "n": int(j.size),
        })
    df = pd.DataFrame(rows)
    df["spread_vs_pre"] = df.mean_spread_bps / base_spread
    df["depth_vs_pre"] = df.mean_consumed_depth_btc / base_depth
    return df


def _clock_time_profile(rd, cfg, mask: np.ndarray) -> pd.DataFrame:
    """Quote-tape state at fixed second offsets after the aggressive order."""
    trades = rd.trades
    q_exch = rd.q_exch
    q_spread = rd.q_spread_bps
    q_bid_d = rd.quotes["bid_depth"].to_numpy(np.float64)
    q_ask_d = rd.quotes["ask_depth"].to_numpy(np.float64)

    ts = trades["exch_ns"].to_numpy(np.int64)[mask]
    sign = trades["sign"].to_numpy(np.float64)[mask]
    if ts.size < 30:
        return pd.DataFrame()

    rows = []
    for offset in (0.0,) + tuple(cfg.CLOCK_HORIZONS_S):
        pos = st.asof_left(q_exch, ts + int(offset * 1e9))
        ok = pos >= 0
        pos = pos[ok]
        sgn = sign[ok]
        if pos.size < 30:
            continue
        consumed = np.where(sgn > 0, q_ask_d[pos], q_bid_d[pos])
        rows.append({
            "horizon_s": offset,
            "mean_spread_bps": float(np.nanmean(q_spread[pos])),
            "mean_consumed_depth_btc": float(np.nanmean(consumed)),
            "n": int(pos.size),
        })
    return pd.DataFrame(rows)


def run(ctx) -> dict:
    pl.apply_style()
    cfg = ctx.cfg
    metrics: list[dict] = []
    event_large, clock_large, event_typ = {}, {}, {}

    for name, rd in ctx.each():
        trades = rd.trades
        qty = trades["qty"].to_numpy(np.float64)
        threshold = float(np.nanquantile(qty, cfg.LARGE_TRADE_QUANTILE))
        large = qty >= threshold
        typical = (qty >= np.nanquantile(qty, 0.45)) & (qty <= np.nanquantile(qty, 0.55))

        uncond_spread, uncond_depth = _unconditional(rd, cfg)
        ev_l = _event_time_profile(rd, cfg, large)
        ev_t = _event_time_profile(rd, cfg, typical)
        ck_l = _clock_time_profile(rd, cfg, large)
        event_large[name], event_typ[name], clock_large[name] = ev_l, ev_t, ck_l

        ctx.out.save_table(ev_l, FACT, "event_time_recovery_large", name,
                           caption=f"Recovery after large orders, event time - {rd.label}")
        ctx.out.save_table(ck_l, FACT, "clock_time_recovery_large", name,
                           caption=f"Recovery after large orders, clock time - {rd.label}")

        # --- exponential fit on the clock-time spread path ---------------
        # The exponential is fitted from the *peak* of the dislocation onward.
        # The spread needs a moment to reach its widest - the book channel has
        # to publish the sweep - so fitting from t=0 would have the decay
        # constant absorb that rise and badly understate the recovery speed.
        tau = {"tau_s": np.nan, "r2": np.nan, "amplitude": np.nan, "t0": 0.0}
        if len(ck_l) > 4:
            far = ck_l[ck_l.horizon_s >= 120.0]
            y_inf = float(far.mean_spread_bps.mean()) if len(far) else float(ck_l.mean_spread_bps.iloc[-1])
            early = ck_l[ck_l.horizon_s <= 5.0]
            t_peak = float(early.loc[early.mean_spread_bps.idxmax(), "horizon_s"]) if len(early) else 0.0
            fit_rows = ck_l[(ck_l.horizon_s >= t_peak) & (ck_l.horizon_s <= 60.0)]
            tau = st.fit_exponential_recovery(fit_rows.horizon_s.to_numpy() - t_peak,
                                              fit_rows.mean_spread_bps.to_numpy(), y_inf)
            tau["y_inf"] = y_inf
            tau["t0"] = t_peak

        fig, axes = pl.new_figure(1, 3, width=5.2, height=4.2)
        ax = axes[0]
        ax.plot(np.maximum(ev_l.horizon_orders, 0.5), ev_l.spread_vs_pre, "o-",
                lw=1.7, ms=4, label="large orders (top 1%)")
        if len(ev_t):
            ax.plot(np.maximum(ev_t.horizon_orders, 0.5), ev_t.spread_vs_pre, "s--",
                    lw=1.3, ms=3.5, alpha=0.8, label="median-size orders")
        ax.axhline(1.0, **pl.REFERENCE_KW)
        base_spread = ev_l.mean_spread_bps.iloc[0]
        ax.axhline(uncond_spread / base_spread, **pl.LITERATURE_KW)
        ax.set_xscale("log")
        pl.finish(ax, "Spread recovery (event time)", "aggressive orders since event",
                  "spread / pre-event spread", legend=True,
                  note=f"Dashed grey: the pre-event level. Dotted green: the "
                       f"unconditional time-weighted mean ({uncond_spread:.3f} bps), which "
                       f"is the steady state - large orders arrive when the spread is "
                       f"already {base_spread / uncond_spread:.1f}x wider than usual.")

        ax = axes[1]
        ax.plot(np.maximum(ev_l.horizon_orders, 0.5), ev_l.depth_vs_pre, "o-",
                lw=1.7, ms=4, color="#7A4FA3", label="large orders")
        if len(ev_t):
            ax.plot(np.maximum(ev_t.horizon_orders, 0.5), ev_t.depth_vs_pre, "s--",
                    lw=1.3, ms=3.5, alpha=0.8, color="#B08BD1", label="median-size orders")
        ax.axhline(1.0, **pl.REFERENCE_KW)
        base_depth = ev_l.mean_consumed_depth_btc.iloc[0]
        ax.axhline(uncond_depth / base_depth, **pl.LITERATURE_KW)
        ax.set_xscale("log")
        pl.finish(ax, "Consumed-side depth recovery (event time)",
                  "aggressive orders since event", "depth / pre-event depth", legend=True,
                  note=f"Dotted green: the unconditional mean one-sided depth "
                       f"({uncond_depth:.1f} BTC). Large orders arrive into a book "
                       f"{base_depth / uncond_depth:.2f}x the usual depth.")

        ax = axes[2]
        if len(ck_l):
            ax.plot(np.maximum(ck_l.horizon_s, 0.02), ck_l.mean_spread_bps, "o-",
                    lw=1.7, ms=4, label="mean spread")
            if np.isfinite(tau.get("tau_s", np.nan)):
                t0 = tau.get("t0", 0.0)
                tt = np.geomspace(max(t0, 0.02), 60, 60)
                ax.plot(tt, tau["y_inf"] + tau["amplitude"] * np.exp(-(tt - t0) / tau["tau_s"]),
                        lw=1.5, ls="--", color="#B45309",
                        label=f"exp fit from peak, tau={tau['tau_s']:.1f}s")
            ax.axvline(LITERATURE_TAU_S, **pl.LITERATURE_KW)
            ax.axhline(uncond_spread, color="#6B7280", lw=1.1, ls="-.", alpha=0.9)
        ax.set_xscale("log")
        pl.finish(ax, "Spread recovery (clock time)", "seconds since event",
                  "mean spread (bps)", legend=True,
                  note=f"Dotted green vertical: the 6.3 s Coinbase literature target. "
                       f"Dash-dot horizontal: the unconditional mean spread "
                       f"({uncond_spread:.3f} bps).")
        fig.suptitle(f"Fact 7 - resiliency after large trades | {rd.label}", x=0.02,
                     ha="left", fontsize=12.5, fontweight="semibold")
        pl.layout(fig)
        ctx.out.save_figure(fig, FACT, "resiliency", name)

        peak = float(ev_l.spread_vs_pre.iloc[1]) if len(ev_l) > 1 else np.nan
        metrics += [
            {"metric": "large-order size threshold (BTC)", "regime": name,
             "value": threshold, "detail": f"top {100*(1-cfg.LARGE_TRADE_QUANTILE):.0f}%"},
            {"metric": "spread at the next order / pre-event", "regime": name,
             "value": peak, "detail": "event-time peak dislocation"},
            {"metric": "spread recovery time constant (s)", "regime": name,
             "value": tau.get("tau_s", np.nan),
             "detail": f"literature {LITERATURE_TAU_S}s; fit R2={tau.get('r2', np.nan):.2f}"},
        ]
        if len(ev_l) > 2:
            # Skip horizon 0: that row *is* the pre-event baseline, so its
            # ratio is 1 by construction.
            half = ev_l[(ev_l.horizon_orders > 0) & (ev_l.depth_vs_pre >= 0.99)]
            spread_back = ev_l[(ev_l.horizon_orders > 0) & (ev_l.spread_vs_pre <= 1.01)]
            metrics.append({
                "metric": "orders until spread is back within 1% of pre-event",
                "regime": name,
                "value": float(spread_back.horizon_orders.iloc[0]) if len(spread_back) else np.nan,
                "detail": "event-time spread resiliency",
            })
            metrics.append({
                "metric": "orders until consumed-side depth is back to 99%",
                "regime": name,
                "value": float(half.horizon_orders.iloc[0]) if len(half) else np.nan,
                "detail": "compare with the spread row above: depth recovering later is "
                          "the Schnaubelt et al. ordering",
            })

    fig, axes = pl.new_figure(1, 2, width=5.8, height=4.4)
    for name, rd in ctx.each():
        ev = event_large[name]
        axes[0].plot(np.maximum(ev.horizon_orders, 0.5), ev.spread_vs_pre, "o-",
                     lw=1.8, ms=4, color=rd.color, label=f"{rd.label} - spread")
        axes[0].plot(np.maximum(ev.horizon_orders, 0.5), ev.depth_vs_pre, "s--",
                     lw=1.3, ms=3.5, alpha=0.75, color=rd.color,
                     label=f"{rd.label} - depth")
        ck = clock_large[name]
        if len(ck):
            axes[1].plot(np.maximum(ck.horizon_s, 0.02), ck.mean_spread_bps, "o-",
                         lw=1.8, ms=4, color=rd.color, label=rd.label)
    axes[0].axhline(1.0, **pl.REFERENCE_KW)
    axes[0].set_xscale("log")
    pl.finish(axes[0], "Recovery after large orders (event time)",
              "aggressive orders since event", "level / pre-event level", legend=True,
              note="Curves settle below 1 for the spread because large orders arrive "
                   "when the spread is already wide: the pre-event level is not the "
                   "steady state. See the per-regime panels for the unconditional mean.")
    axes[1].axvline(LITERATURE_TAU_S, **pl.LITERATURE_KW)
    axes[1].set_xscale("log")
    pl.finish(axes[1], "Spread recovery (clock time)", "seconds since event",
              "mean spread (bps)", legend=True,
              note="Dotted vertical line: 6.3 s Coinbase BTC/USD literature target.")
    fig.suptitle("Fact 7 - resiliency by regime", x=0.02, ha="left", fontsize=12.5,
                 fontweight="semibold")
    pl.layout(fig, rect=(0, 0, 1, 0.94))
    ctx.out.save_figure(fig, FACT, "resiliency", "comparison")

    return {
        "fact": FACT, "id": FACT_ID, "priority": PRIORITY, "title": TITLE,
        "metrics": metrics,
        "literature": f"Schnaubelt et al.: spread recovery tau ~ {LITERATURE_TAU_S}s on "
                      "Coinbase BTC/USD; deep-book liquidity recovers more slowly.",
    }
