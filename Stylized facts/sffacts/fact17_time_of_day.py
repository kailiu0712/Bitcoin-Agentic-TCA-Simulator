"""
Stylized fact 17 (P2) - 24/7 time-of-day liquidity and activity structure.

Empirical claim
    Unlike equities, BTC has no open or close and does not exhibit one robust
    universal U-shaped intraday liquidity pattern.  Hourly activity and
    liquidity vary across venues and time zones.

Why it matters for the simulator
    Useful for 24/7 realism and time-conditioned stress testing, but the source
    document is clear that it matters less than the actual state variables
    (spread, depth, OFI, volatility).  It is included because it is nearly free:
    the hourly aggregates are already accumulated during the single ingest pass.

Method
    Liquidity variables (spread, depth, cost of 1 BTC) come from the kernel's
    **time-weighted** hourly accumulators, so they describe the book you would
    see sampling that hour at random.  Activity variables (order count, volume,
    realised volatility) are aggregated from the trade tape.  The "U-shape test"
    reported is simply whether the hourly profile has the two-peaked shape of an
    equity session; a flat-ish or single-peaked profile is the expected crypto
    result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfcore import plotting as pl
from sfcore import stats as st

FACT = "fact17"
FACT_ID = 17
PRIORITY = "P2"
TITLE = "24/7 time-of-day liquidity and activity structure"

NS_PER_HOUR = 3_600_000_000_000


def _hourly(rd) -> pd.DataFrame:
    acc = rd.accumulators
    hour = acc["hour"]
    weight = hour[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = np.where(weight > 0, hour[1] / weight, np.nan)
        depth = np.where(weight > 0, (hour[2] + hour[3]) / weight, np.nan)
        cost = np.where(hour[5] > 0, hour[4] / hour[5], np.nan)

    trades = rd.trades
    hr = ((trades["recv_ns"].to_numpy(np.int64) // NS_PER_HOUR) % 24).astype(int)
    counts = np.bincount(hr, minlength=24).astype(float)
    volume = np.bincount(hr, weights=trades["qty"].to_numpy(np.float64), minlength=24)
    notional = np.bincount(hr, weights=trades["notional"].to_numpy(np.float64), minlength=24)

    # Realised volatility per hour-of-day, from quote-event mid returns.
    q_hr = ((rd.q_recv // NS_PER_HOUR) % 24).astype(int)
    logmid = np.log(rd.q_mid)
    ret = np.diff(logmid)
    ret_hr = q_hr[1:]
    sq = np.bincount(ret_hr, weights=ret ** 2, minlength=24)

    hours_observed = weight / NS_PER_HOUR
    # Realised volatility per hour of wall-clock time in that hour-of-day slot,
    # so a busy hour is not credited with extra volatility purely for having
    # more quote updates.
    with np.errstate(invalid="ignore", divide="ignore"):
        rv = np.where(hours_observed > 0, np.sqrt(sq / hours_observed), np.nan) * 1e4

    with np.errstate(invalid="ignore", divide="ignore"):
        orders_per_hour = np.where(hours_observed > 0, counts / hours_observed, np.nan)
        volume_per_hour = np.where(hours_observed > 0, volume / hours_observed, np.nan)

    return pd.DataFrame({
        "hour_utc": np.arange(24),
        "hours_observed": hours_observed,
        "mean_spread_bps": spread,
        "mean_total_depth_btc": depth,
        "mean_cost_1btc_bps": cost,
        "aggressive_orders_per_hour": orders_per_hour,
        "volume_btc_per_hour": volume_per_hour,
        "notional_usd": notional,
        "realised_vol_bps": rv,
    })


def _u_shape_score(profile: np.ndarray) -> float:
    """How much larger the session edges are than the middle.

    For a 24/7 market there is no session, so this is computed against the
    UTC day purely as a descriptive statistic: >1 would mean edge-heavy
    (equity-like U), ~1 means flat.
    """
    x = np.asarray(profile, dtype=np.float64)
    if not np.isfinite(x).all():
        x = pd.Series(x).interpolate(limit_direction="both").to_numpy()
    edges = np.concatenate([x[:4], x[-4:]])
    middle = x[8:16]
    return float(np.nanmean(edges) / np.nanmean(middle))


def run(ctx) -> dict:
    pl.apply_style()
    metrics: list[dict] = []
    profiles = {}

    for name, rd in ctx.each():
        prof = _hourly(rd)
        profiles[name] = prof
        ctx.out.save_table(prof, FACT, "hourly_profile", name,
                           caption=f"Hour-of-day liquidity and activity - {rd.label}")

        fig, axes = pl.new_figure(2, 2, width=5.4, height=3.6)
        panels = [
            (axes[0][0], "mean_spread_bps", "Mean spread (bps, time-weighted)", "#2C6FBB"),
            (axes[0][1], "mean_total_depth_btc", "Visible depth within 10 bps (BTC)", "#7A4FA3"),
            (axes[1][0], "aggressive_orders_per_hour", "Aggressive orders per hour", "#B45309"),
            (axes[1][1], "volume_btc_per_hour", "Traded volume per hour (BTC)", "#0E7C66"),
        ]
        for ax, column, title, color in panels:
            ax.bar(prof.hour_utc, prof[column], color=color, alpha=0.85, width=0.75)
            ax.axhline(np.nanmean(prof[column]), **pl.REFERENCE_KW)
            pl.finish(ax, title, "hour (UTC)", "")
        fig.suptitle(f"Fact 17 - 24/7 intraday structure | {rd.label}", x=0.02,
                     ha="left", fontsize=12.5, fontweight="semibold")
        pl.layout(fig)
        ctx.out.save_figure(fig, FACT, "time_of_day", name)

        for column, label in (("mean_spread_bps", "spread"),
                              ("mean_total_depth_btc", "depth"),
                              ("volume_btc_per_hour", "volume")):
            series = prof[column].to_numpy()
            metrics.append({
                "metric": f"hourly {label}: max/min ratio", "regime": name,
                "value": float(np.nanmax(series) / np.nanmin(series)), "detail": "",
            })
        metrics.append({
            "metric": "edge-vs-middle ratio of hourly volume (U-shape score)",
            "regime": name,
            "value": _u_shape_score(prof.volume_btc_per_hour.to_numpy()),
            "detail": "~1 = no equity-like U-shape",
        })
        busiest = int(prof.volume_btc_per_hour.idxmax())
        metrics.append({"metric": "busiest UTC hour by volume", "regime": name,
                        "value": float(busiest), "detail": "hour of day"})

    fig, axes = pl.new_figure(1, 3, width=5.2, height=4.2)
    for name, rd in ctx.each():
        prof = profiles[name]
        axes[0].plot(prof.hour_utc, prof.mean_spread_bps, "o-", ms=4, lw=1.7,
                     color=rd.color, label=rd.label)
        axes[1].plot(prof.hour_utc, prof.mean_total_depth_btc, "o-", ms=4, lw=1.7,
                     color=rd.color, label=rd.label)
        norm = prof.volume_btc_per_hour / np.nanmean(prof.volume_btc_per_hour)
        axes[2].plot(prof.hour_utc, norm, "o-", ms=4, lw=1.7, color=rd.color,
                     label=rd.label)
    pl.finish(axes[0], "Hourly spread", "hour (UTC)", "mean spread (bps)", legend=True)
    pl.finish(axes[1], "Hourly visible depth", "hour (UTC)", "depth within 10 bps (BTC)",
              legend=True)
    axes[2].axhline(1.0, **pl.REFERENCE_KW)
    pl.finish(axes[2], "Hourly volume (normalised)", "hour (UTC)",
              "volume / daily mean", legend=True)
    fig.suptitle("Fact 17 - 24/7 intraday structure by regime", x=0.02, ha="left",
                 fontsize=12.5, fontweight="semibold")
    pl.layout(fig, rect=(0, 0, 1, 0.94))
    ctx.out.save_figure(fig, FACT, "time_of_day", "comparison")

    return {
        "fact": FACT, "id": FACT_ID, "priority": PRIORITY, "title": TITLE,
        "metrics": metrics,
        "literature": "Schnaubelt et al.: hourly variation exists but no universal "
                      "U-shape; BTC has no session open/close.",
    }
