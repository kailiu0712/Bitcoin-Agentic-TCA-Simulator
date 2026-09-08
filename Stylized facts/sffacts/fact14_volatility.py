"""
Stylized fact 14 (P1) - Volatility clustering at short horizons.

Empirical claim
    Bitcoin squared / absolute returns show positive, slowly decaying
    autocorrelation.  Schnaubelt, Rende & Krauss estimate squared-return ACF
    decay exponents of about **0.16 at minute scale** and **0.24 hourly**.

Why it matters for the simulator
    A liquidation started in a volatile state should stay risky for a while.
    Without clustering, stress conditions evaporate unrealistically fast and
    the tail of the implementation-shortfall distribution is understated.

Method
    Event time is this framework's primary clock, so the absolute-return ACF is
    first computed over quote events.  But the literature's exponents are
    clock-time statements, so mid prices are additionally sampled onto 1 s,
    60 s and 3600 s grids by last-observation-carried-forward and the
    squared-return ACF is fitted there.  Bar construction is used *only* for
    this comparison; no event is discarded anywhere in the pipeline.

    The decay exponent is estimated by OLS on log ACF vs log lag over the lag
    window where the ACF is still above its noise floor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfcore import plotting as pl
from sfcore import stats as st

FACT = "fact14"
FACT_ID = 14
PRIORITY = "P1"
TITLE = "Volatility clustering at short horizons"

LITERATURE = {60.0: 0.16, 3600.0: 0.24}
FIT_LAG_MIN = 2
FIT_LAG_MAX_EVENT = 500


def _clock_bars(recv_ns: np.ndarray, mid: np.ndarray, seconds: float) -> np.ndarray:
    """Last mid in each fixed clock bucket (last observation carried forward)."""
    step = int(seconds * 1e9)
    bucket = (recv_ns - recv_ns[0]) // step
    n_bars = int(bucket[-1]) + 1
    if n_bars < 50:
        return np.array([])
    last = np.full(n_bars, np.nan)
    # The final observation inside a bucket wins; assigning in order does that.
    last[bucket] = mid
    # Carry forward across empty buckets.
    idx = np.where(np.isfinite(last), np.arange(n_bars), 0)
    np.maximum.accumulate(idx, out=idx)
    return last[idx]


def _acf_table(series: np.ndarray, max_lag: int, transform: str) -> pd.DataFrame:
    ret = np.diff(np.log(series))
    ret = ret[np.isfinite(ret)]
    # A one-week regime yields only ~168 hourly bars, which is few but is the
    # sample the literature's hourly exponent has to be compared against; the
    # bar count is carried in the output table so the reader can weigh it.
    if ret.size < 100:
        return pd.DataFrame()
    x = np.abs(ret) if transform == "abs" else ret ** 2
    lags = np.arange(1, min(max_lag, max(ret.size // 10, 5)) + 1)
    values = st.acf(x, lags.size)
    return pd.DataFrame({"lag": lags[: values.size], "acf": values[: lags.size]})


def run(ctx) -> dict:
    pl.apply_style()
    cfg = ctx.cfg
    metrics: list[dict] = []
    event_acfs, clock_tables = {}, {}

    for name, rd in ctx.each():
        recv = rd.q_recv
        mid = rd.q_mid

        ev = _acf_table(mid, cfg.ACF_MAX_LAG_EVENT, "abs")
        event_acfs[name] = ev
        ev_fit = st.fit_loglog(
            ev.lag[(ev.lag >= FIT_LAG_MIN) & (ev.lag <= FIT_LAG_MAX_EVENT)].to_numpy(),
            ev.acf[(ev.lag >= FIT_LAG_MIN) & (ev.lag <= FIT_LAG_MAX_EVENT)].to_numpy(),
        ) if len(ev) else {"exponent": np.nan, "r2": np.nan, "prefactor": np.nan}

        rows, clock_acfs = [], {}
        for seconds in cfg.VOL_BAR_SECONDS:
            bars = _clock_bars(recv, mid, seconds)
            if bars.size < 200:
                continue
            tab = _acf_table(bars, cfg.ACF_MAX_LAG_CLOCK, "square")
            if not len(tab):
                continue
            clock_acfs[seconds] = tab
            fit_hi = int(min(200, tab.lag.max()))
            window = tab[(tab.lag >= FIT_LAG_MIN) & (tab.lag <= fit_hi)]
            fit = st.fit_loglog(window.lag.to_numpy(), window.acf.to_numpy())
            rows.append({
                "bar_seconds": seconds,
                "n_bars": int(bars.size),
                "acf_lag1": float(tab.acf.iloc[0]),
                "acf_lag10": float(tab.acf.iloc[9]) if len(tab) > 9 else np.nan,
                "decay_exponent": -fit["exponent"],
                "fit_r2": fit["r2"],
                "fit_lag_min": FIT_LAG_MIN,
                "fit_lag_max": fit_hi,
                "literature_exponent": LITERATURE.get(seconds, np.nan),
            })
        clock = pd.DataFrame(rows)
        clock_tables[name] = (clock, clock_acfs)

        if len(ev):
            ctx.out.save_table(ev, FACT, "abs_return_acf_event_time", name,
                               caption=f"Absolute-return ACF in quote-event time - {rd.label}")
        if len(clock):
            ctx.out.save_table(clock, FACT, "clock_time_decay", name,
                               caption=f"Squared-return ACF decay by bar length - {rd.label}")

        fig, axes = pl.new_figure(1, 3, width=5.2, height=4.2)
        ax = axes[0]
        if len(ev):
            ok = ev.acf > 0
            ax.plot(ev.lag[ok], ev.acf[ok], ".", ms=3, alpha=0.6)
            if np.isfinite(ev_fit["exponent"]):
                xs = np.geomspace(FIT_LAG_MIN, FIT_LAG_MAX_EVENT, 50)
                ax.plot(xs, ev_fit["prefactor"] * xs ** ev_fit["exponent"], lw=2.0,
                        color="#B45309",
                        label=f"exponent {-ev_fit['exponent']:.3f} (R2={ev_fit['r2']:.2f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        pl.finish(ax, "|return| ACF, quote-event time", "lag (quote events)",
                  "autocorrelation", legend=True)

        ax = axes[1]
        for seconds, tab in clock_acfs.items():
            ok = tab.acf > 0
            ax.plot(tab.lag[ok], tab.acf[ok], "o-", ms=3, lw=1.2, alpha=0.85,
                    label=f"{seconds:g}s bars")
        ax.set_xscale("log")
        ax.set_yscale("log")
        pl.finish(ax, "Squared-return ACF, clock time", "lag (bars)",
                  "autocorrelation", legend=True)

        ax = axes[2]
        if len(clock):
            x = np.arange(len(clock))
            ax.bar(x - 0.18, clock.decay_exponent, width=0.36, label="measured",
                   color="#6B8FBF")
            ax.bar(x + 0.18, clock.literature_exponent, width=0.36,
                   label="literature", color="#0E7C66", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{s:g}s" for s in clock.bar_seconds])
        pl.finish(ax, "Decay exponent vs literature", "bar length",
                  "power-law decay exponent", legend=True,
                  note="Schnaubelt et al.: ~0.16 at minute scale, ~0.24 hourly.")
        fig.suptitle(f"Fact 14 - volatility clustering | {rd.label}", x=0.02,
                     ha="left", fontsize=12.5, fontweight="semibold")
        pl.layout(fig)
        ctx.out.save_figure(fig, FACT, "volatility_clustering", name)

        metrics.append({"metric": "|return| ACF decay exponent (quote-event time)",
                        "regime": name, "value": -ev_fit["exponent"],
                        "detail": f"R2={ev_fit['r2']:.3f}"})
        for _, row in clock.iterrows():
            metrics.append({
                "metric": f"squared-return ACF decay exponent ({row.bar_seconds:g}s bars)",
                "regime": name, "value": row.decay_exponent,
                "detail": (f"lags {int(row.fit_lag_min)}-{int(row.fit_lag_max)}, "
                           f"n={int(row.n_bars):,} bars, R2={row.fit_r2:.2f}; "
                           + (f"literature {row.literature_exponent:.2f}"
                              if np.isfinite(row.literature_exponent)
                              else "no literature target")),
            })
        if len(ev):
            metrics.append({"metric": "|return| ACF at lag 100 (quote-event time)",
                            "regime": name,
                            "value": float(ev.acf.iloc[99]) if len(ev) > 99 else np.nan,
                            "detail": "positive and slowly decaying = clustering"})

    fig, axes = pl.new_figure(1, 2, width=5.8, height=4.4)
    for name, rd in ctx.each():
        ev = event_acfs[name]
        if len(ev):
            ok = ev.acf > 0
            axes[0].plot(ev.lag[ok], ev.acf[ok], ".", ms=3, alpha=0.5, color=rd.color,
                         label=rd.label)
        clock, clock_acfs = clock_tables[name]
        if 60.0 in clock_acfs:
            tab = clock_acfs[60.0]
            ok = tab.acf > 0
            axes[1].plot(tab.lag[ok], tab.acf[ok], "o-", ms=3, lw=1.3, color=rd.color,
                         label=rd.label)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    pl.finish(axes[0], "|return| ACF, quote-event time", "lag (quote events)",
              "autocorrelation", legend=True)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    pl.finish(axes[1], "Squared-return ACF, 60 s bars", "lag (minutes)",
              "autocorrelation", legend=True)
    fig.suptitle("Fact 14 - volatility clustering by regime", x=0.02, ha="left",
                 fontsize=12.5, fontweight="semibold")
    pl.layout(fig, rect=(0, 0, 1, 0.94))
    ctx.out.save_figure(fig, FACT, "volatility_clustering", "comparison")

    return {
        "fact": FACT, "id": FACT_ID, "priority": PRIORITY, "title": TITLE,
        "metrics": metrics,
        "literature": "Schnaubelt et al.: squared-return ACF decay ~0.16 (minute), "
                      "~0.24 (hourly).",
    }
