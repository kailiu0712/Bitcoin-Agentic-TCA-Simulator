"""
Stylized fact 6 (P0-C) - Bid-ask spread distribution and state dependence.

Empirical claim
    BTC spreads and liquidity costs vary materially through time and market
    state.  Crypto does *not* show the single universal U-shaped intraday
    spread pattern of traditional equities; Schnaubelt, Rende & Krauss find
    substantial hourly variation on Coinbase but no universal shape.

Why it matters for the simulator
    The spread is the first cost an aggressive liquidation pays.  The
    calibration target is the whole distribution and its conditional tails, not
    the average.

Method
    The unconditional distribution comes from the kernel's streaming histogram
    over every packet, in two weightings:
      * **time-weighted** - the spread you would observe sampling the book at a
        uniformly random instant (the right weighting for a simulator's steady
        state);
      * **event-weighted** - one observation per book update (the right
        weighting for "what does an event see").
    State dependence is then measured three ways on the tapes: spread against
    visible depth, against the top-of-book queue imbalance, and against
    realised volatility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfcore import plotting as pl
from sfcore import stats as st

FACT = "fact06"
FACT_ID = 6
PRIORITY = "P0-C"
TITLE = "Bid-ask spread distribution and state dependence"

QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999)
N_STATE_BINS = 20


def _distribution(rd) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    acc = rd.accumulators
    lo, hi = acc["spread_hist_range"]
    nbins = int(acc["hist_bins"][0])
    edges = st.log_bin_edges(float(lo), float(hi), nbins)
    tw = acc["spread_tw"]
    ew = acc["spread_ew"]
    rows = []
    for label, counts in (("time-weighted", tw), ("event-weighted", ew)):
        qs = st.hist_quantiles(counts, edges, QUANTILES)
        rows.append({"weighting": label, "mean_bps": st.hist_mean(counts, edges),
                     **{f"p{q*100:g}": v for q, v in zip(QUANTILES, qs)}})
    return pd.DataFrame(rows), edges, tw, ew


def _hourly(rd) -> pd.DataFrame:
    acc = rd.accumulators
    hour = acc["hour"]
    weight = hour[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = np.where(weight > 0, hour[1] / weight, np.nan)
        depth_bid = np.where(weight > 0, hour[2] / weight, np.nan)
        depth_ask = np.where(weight > 0, hour[3] / weight, np.nan)
        cost = np.where(hour[5] > 0, hour[4] / hour[5], np.nan)
    return pd.DataFrame({
        "hour_utc": np.arange(24),
        "mean_spread_bps": spread,
        "mean_bid_depth_btc": depth_bid,
        "mean_ask_depth_btc": depth_ask,
        "mean_cost_1btc_bps": cost,
        "time_share": weight / weight.sum() if weight.sum() > 0 else np.nan,
        "packets": hour[6],
    })


def _state_dependence(rd, cfg) -> pd.DataFrame:
    """Spread conditional on depth, queue imbalance and short-horizon volatility."""
    q = rd.quotes
    spread = rd.q_spread_bps
    depth = (q["bid_depth"].to_numpy(np.float64) + q["ask_depth"].to_numpy(np.float64))
    imbalance = np.abs(rd.q_imbalance)
    mid = rd.q_mid
    # Realised volatility in quote-event time: rolling absolute log return.
    logmid = np.log(mid)
    ret = np.diff(logmid, prepend=logmid[0])
    window = 200
    vol = pd.Series(np.abs(ret)).rolling(window, min_periods=window // 2).mean().to_numpy() * 1e4

    frames = []
    for label, driver in (("visible depth (BTC)", depth),
                          ("|queue imbalance|", imbalance),
                          ("realised vol (bps/event, 200-event MA)", vol)):
        ok = np.isfinite(driver) & np.isfinite(spread)
        if ok.sum() < 1000:
            continue
        bins = st.quantile_bins(driver[ok], N_STATE_BINS)
        mean_spread, counts = st.group_mean(spread[ok], bins, N_STATE_BINS)
        mean_driver, _ = st.group_mean(driver[ok], bins, N_STATE_BINS)
        frames.append(pd.DataFrame({
            "state_variable": label,
            "bin": np.arange(N_STATE_BINS),
            "mean_state": mean_driver,
            "mean_spread_bps": mean_spread,
            "n": counts,
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run(ctx) -> dict:
    pl.apply_style()
    metrics: list[dict] = []
    dists, hourlies, states = {}, {}, {}

    for name, rd in ctx.each():
        dist, edges, tw, ew = _distribution(rd)
        hourly = _hourly(rd)
        state = _state_dependence(rd, ctx.cfg)
        dists[name] = (dist, edges, tw, ew)
        hourlies[name] = hourly
        states[name] = state

        ctx.out.save_table(dist, FACT, "spread_distribution", name,
                           caption=f"Quoted-spread distribution (bps) - {rd.label}")
        ctx.out.save_table(hourly, FACT, "hourly_profile", name,
                           caption=f"Time-weighted hourly spread / depth profile - {rd.label}")
        if len(state):
            ctx.out.save_table(state, FACT, "state_dependence", name,
                               caption=f"Spread conditional on book state - {rd.label}")

        fig, axes = pl.new_figure(1, 3, width=5.2, height=4.2)
        centres = np.sqrt(edges[:-1] * edges[1:])
        ax = axes[0]
        ax.step(centres, tw / tw.sum(), where="mid", lw=1.6, label="time-weighted")
        ax.step(centres, ew / ew.sum(), where="mid", lw=1.4, ls="--", label="event-weighted")
        ax.set_xscale("log")
        ax.set_yscale("log")
        pl.finish(ax, "Spread distribution", "quoted spread (bps)", "density",
                  legend=True,
                  note="One tick (0.1 USD) is ~0.015 bps at these prices, so the spread is quantised: the comb of spikes at the left is 1, 2, 3 ... ticks, and the empty bins between them are real, not missing data.")

        ax = axes[1]
        ax.plot(hourly.hour_utc, hourly.mean_spread_bps, "o-", lw=1.7, ms=4)
        pl.finish(ax, "Intraday profile (24/7 market)", "hour (UTC)",
                  "time-weighted mean spread (bps)",
                  note="Crypto has no open/close; a universal U-shape is not expected.")

        ax = axes[2]
        if len(state):
            for label, sub in state.groupby("state_variable"):
                x = np.arange(len(sub))
                ax.plot(x, sub.mean_spread_bps, "o-", lw=1.5, ms=3.5, label=label)
            ax.set_xlabel("state-variable quantile bin")
        pl.finish(ax, "Spread vs book state", "quantile bin of state variable",
                  "mean spread (bps)", legend=True)
        fig.suptitle(f"Fact 6 - spread distribution and state dependence | {rd.label}",
                     x=0.02, ha="left", fontsize=12.5, fontweight="semibold")
        pl.layout(fig)
        ctx.out.save_figure(fig, FACT, "spread_distribution", name)

        med = float(dist.loc[dist.weighting == "time-weighted", "p50"].iloc[0])
        p99 = float(dist.loc[dist.weighting == "time-weighted", "p99"].iloc[0])
        hs = hourly.mean_spread_bps
        metrics += [
            {"metric": "median quoted spread (bps, time-weighted)", "regime": name,
             "value": med, "detail": f"{med * 65000 / 1e4:.2f} USD at 65k"},
            {"metric": "99th pct quoted spread (bps)", "regime": name, "value": p99,
             "detail": f"{p99 / med:.1f}x the median"},
            {"metric": "hourly spread max/min ratio", "regime": name,
             "value": float(np.nanmax(hs) / np.nanmin(hs)), "detail": "U-shape test"},
            {"metric": "hourly spread coefficient of variation", "regime": name,
             "value": float(np.nanstd(hs) / np.nanmean(hs)), "detail": ""},
        ]

    fig, axes = pl.new_figure(1, 3, width=5.2, height=4.2)
    for name, rd in ctx.each():
        dist, edges, tw, ew = dists[name]
        centres = np.sqrt(edges[:-1] * edges[1:])
        axes[0].step(centres, np.cumsum(tw) / tw.sum(), where="mid", lw=1.8,
                     color=rd.color, label=rd.label)
        axes[1].plot(hourlies[name].hour_utc, hourlies[name].mean_spread_bps, "o-",
                     lw=1.7, ms=4, color=rd.color, label=rd.label)
        state = states[name]
        if len(state):
            sub = state[state.state_variable == "visible depth (BTC)"]
            axes[2].plot(sub.mean_state, sub.mean_spread_bps, "o-", lw=1.6, ms=4,
                         color=rd.color, label=rd.label)
    axes[0].set_xscale("log")
    pl.finish(axes[0], "Spread CDF (time-weighted)", "quoted spread (bps)",
              "P(spread <= x)", legend=True)
    pl.finish(axes[1], "Intraday spread profile", "hour (UTC)",
              "mean spread (bps)", legend=True)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    pl.finish(axes[2], "Spread vs visible depth", "visible depth within 10 bps (BTC)",
              "mean spread (bps)", legend=True)
    fig.suptitle("Fact 6 - spread by regime", x=0.02, ha="left", fontsize=12.5,
                 fontweight="semibold")
    pl.layout(fig, rect=(0, 0, 1, 0.94))
    ctx.out.save_figure(fig, FACT, "spread_distribution", "comparison")

    return {
        "fact": FACT, "id": FACT_ID, "priority": PRIORITY, "title": TITLE,
        "metrics": metrics,
        "literature": "Schnaubelt, Rende & Krauss (2019): substantial hourly "
                      "variation, no universal U-shape across venues.",
    }
