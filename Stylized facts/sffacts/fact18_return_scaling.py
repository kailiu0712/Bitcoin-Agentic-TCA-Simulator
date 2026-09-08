"""
Stylized fact 18 (P0-C) - price-diffusion scaling sigma(tau) ~ tau^H.

Source
    ``proxy.ipynb`` cells 7 and 9.  The notebook resamples the mid onto a 100 ms
    grid, measures the standard deviation of the log return at every lag from
    0.2 s to 60 s, and reads the Hurst exponent off a weighted log-log fit.  It
    then repeats the fit on tau >= 2 s only, on the grounds that the shortest
    lags are dominated by microstructure noise rather than by diffusion.

Empirical claim
    Over the diffusive range the mid behaves close to a random walk, H ~ 0.5.
    Below that range bid-ask bounce and discreteness flatten the curve, so the
    all-lag fit understates H.

Why it matters here
    H is not only a description of the price process.  It is the input to the
    notebook's closure relation for metaorder impact,

        gamma_theory = H - delta_2 * nu

    where delta_2 is the impact concavity exponent and nu the local
    volume-clock exponent (both measured in ``fact_meta_notebook``).  Measuring
    H here, once, on the same data, is what makes that closure checkable rather
    than assumed.

Relation to fact 11
    ``fact11_diffusion`` reports Lo-MacKinlay variance ratios, which answer
    "is the k-step variance k times the one-step variance" at a handful of
    horizons.  This module reports the *continuous* scaling exponent of the
    standard deviation across three decades of lag.  They are consistent views
    of the same process, not substitutes: a variance ratio is a level, H is a
    slope, and neither is evidence of long-range dependence on its own.

Convention notes
    * The grid is built per UTC day and increments are never taken across a day
      boundary, so a gap between days cannot masquerade as a large return.
    * ``n`` in the error bars counts *overlapping* increments, so it overstates
      the independent sample size at long lags.  The error bars weight the fit;
      they are not confidence statements.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfcore import fits
from sfcore import plotting as pl

FACT = "fact18"
FACT_ID = 18
PRIORITY = "P0-C"
TITLE = "Price-diffusion scaling sigma(tau) ~ tau^H"

#: Resampling grid, seconds.  The notebook's choice; also about the coarsest
#: grid that still resolves the microstructure-dominated short end.
GRID_S = 0.1

#: Lag range, seconds.  Matches proxy.ipynb cell 7.
LAG_MIN_S, LAG_MAX_S, N_LAGS = 0.2, 60.0, 30

#: Lower edge of the "clean" diffusive range (proxy.ipynb cell 9).
DIFFUSIVE_MIN_S = 2.0

DAY_NS = 86_400_000_000_000


def _daily_grid(ts_ns: np.ndarray, mid: np.ndarray, step_s: float):
    """Last mid in each fixed bucket, carried forward, restarted every UTC day.

    Returns ``(log_price, segment_starts)`` where ``segment_starts`` holds the
    first index of each day, so that ``fits.volatility_scaling`` never differs
    two prices that straddle a day boundary.
    """
    step = int(round(step_s * 1e9))
    days = ts_ns // DAY_NS                     # ts_ns is sorted, so days is too
    edges = np.searchsorted(days, np.unique(days))
    edges = np.append(edges, len(days))
    segments, starts, offset = [], [], 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        day_ts, day_mid = ts_ns[lo:hi], mid[lo:hi]
        if day_ts.size < 100:
            continue
        bucket = (day_ts - day_ts[0]) // step
        n_bars = int(bucket[-1]) + 1
        if n_bars < 100:
            continue
        bars = np.full(n_bars, np.nan)
        bars[bucket] = day_mid            # last observation in each bucket wins
        idx = np.where(np.isfinite(bars), np.arange(n_bars), 0)
        np.maximum.accumulate(idx, out=idx)
        bars = bars[idx]
        bars = bars[np.isfinite(bars)]    # drop the lead-in before the first quote
        if bars.size < 100:
            continue
        starts.append(offset)
        segments.append(bars)
        offset += bars.size
    if not segments:
        return np.array([]), np.array([0], dtype=np.int64)
    return np.log(np.concatenate(segments)), np.asarray(starts, dtype=np.int64)


def measure(rd) -> dict:
    """sigma(tau) plus the two log-log fits.  Reused by ``fact_meta_notebook``."""
    quotes = rd.quotes
    log_price, starts = _daily_grid(quotes["recv_ns"].to_numpy(np.int64),
                                    rd.q_mid, GRID_S)
    if log_price.size < 1000:
        return {"table": pd.DataFrame(), "all_lags": {}, "diffusive": {}}
    lags = np.unique(np.round(
        np.logspace(np.log10(LAG_MIN_S), np.log10(LAG_MAX_S), N_LAGS) / GRID_S
    ).astype(np.int64))
    table = fits.volatility_scaling(log_price, lags, segment_starts=starts)
    if table.empty:
        return {"table": table, "all_lags": {}, "diffusive": {}}
    table["tau_s"] = table["lag_steps"] * GRID_S
    clean = table.loc[table["tau_s"] >= DIFFUSIVE_MIN_S]
    return {
        "table": table,
        "all_lags": fits.wls_loglog(table["tau_s"], table["sigma"], table["sem_iid"]),
        "diffusive": fits.wls_loglog(clean["tau_s"], clean["sigma"], clean["sem_iid"]),
    }


def run(ctx) -> dict:
    pl.apply_style()
    metrics: list[dict] = []
    measured = {}

    for name, rd in ctx.each():
        result = measure(rd)
        table = result["table"]
        if table.empty:
            continue
        measured[name] = result
        every, clean = result["all_lags"], result["diffusive"]

        out = table.assign(
            hurst_all_lags=every.get("slope", np.nan),
            hurst_diffusive=clean.get("slope", np.nan),
        )
        ctx.out.save_table(out, FACT, "volatility_scaling", name,
                           caption=f"sigma(tau) on a {GRID_S * 1000:.0f} ms grid - {rd.label}")

        fig, axes = pl.new_figure(1, 2, width=5.6, height=4.4)
        ax = axes[0]
        ax.errorbar(table["tau_s"], table["sigma"], yerr=table["sem_iid"], fmt="o",
                    ms=4.5, lw=1.2, capsize=2.5, color=rd.color,
                    label=r"empirical $\sigma(\tau)$")
        grid = np.logspace(np.log10(LAG_MIN_S), np.log10(LAG_MAX_S), 100)
        if np.isfinite(every.get("slope", np.nan)):
            ax.plot(grid, np.exp(every["intercept"]) * grid ** every["slope"],
                    lw=2.0, color=rd.color,
                    label=rf"all lags: $\tau^{{{every['slope']:.2f}}}$ ($R^2={every['r2']:.3f}$)")
        if np.isfinite(clean.get("slope", np.nan)):
            ax.plot(grid, np.exp(clean["intercept"]) * grid ** clean["slope"],
                    lw=1.8, ls="-.", color="#0E7C66",
                    label=rf"$\tau\geq{DIFFUSIVE_MIN_S:g}$s: $\tau^{{{clean['slope']:.2f}}}$")
        middle = table.iloc[len(table) // 2]
        ax.plot(grid, middle["sigma"] / middle["tau_s"] ** 0.5 * grid ** 0.5,
                **pl.REFERENCE_KW, label="standard diffusion ($H=0.50$)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        pl.finish(ax, "Price diffusion scaling", r"time scale $\tau$ (seconds)",
                  r"$\sigma(\tau) = \mathrm{std}(r_\tau)$", legend=True,
                  note="Axes follow proxy.ipynb cell 7. Flattening below ~1 s is "
                       "microstructure noise, which is why the notebook refits "
                       "above 2 s.")

        ax = axes[1]
        local = np.gradient(np.log(table["sigma"].to_numpy()),
                            np.log(table["tau_s"].to_numpy()))
        ax.plot(table["tau_s"], local, "o-", ms=4, lw=1.5, color=rd.color)
        ax.axhline(0.5, **pl.REFERENCE_KW)
        ax.axvline(DIFFUSIVE_MIN_S, color="#0E7C66", ls=":", lw=1.3)
        ax.set_xscale("log")
        pl.finish(ax, "Local slope of the scaling curve",
                  r"time scale $\tau$ (seconds)", r"$d\log\sigma / d\log\tau$",
                  note="A single Hurst exponent is only meaningful where this "
                       "curve is flat. Reported H is a summary of the fitted "
                       "range, not evidence of self-similarity everywhere.")
        fig.suptitle(f"Fact 18 - price-diffusion scaling | {rd.label}", x=0.02,
                     ha="left", fontsize=12.5, fontweight="semibold")
        pl.layout(fig)
        ctx.out.save_figure(fig, FACT, "volatility_scaling", name)

        metrics += [
            {"metric": "Hurst exponent H (0.2-60 s, all lags)", "regime": name,
             "value": every.get("slope", np.nan),
             "detail": f"se {every.get('se', np.nan):.4f}, R2={every.get('r2', np.nan):.3f}"},
            {"metric": f"Hurst exponent H (tau >= {DIFFUSIVE_MIN_S:g} s)",
             "regime": name, "value": clean.get("slope", np.nan),
             "detail": f"se {clean.get('se', np.nan):.4f}, "
                       "feeds the metaorder scaling closure"},
        ]

    if measured:
        fig, ax = pl.new_figure(1, 1, width=7.4, height=4.6)
        for name, rd in ctx.each():
            if name not in measured:
                continue
            table = measured[name]["table"]
            ax.plot(table["tau_s"], table["sigma"], "o-", ms=4, lw=1.7,
                    color=rd.color,
                    label=f"{rd.label} (H={measured[name]['diffusive'].get('slope', np.nan):.2f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        pl.finish(ax, "Fact 18 - price-diffusion scaling by regime",
                  r"time scale $\tau$ (seconds)", r"$\sigma(\tau)$", legend=True,
                  note="Legend H is the tau >= 2 s fit. A higher level is more "
                       "volatile; a steeper slope is more trending.")
        pl.layout(fig, rect=(0, 0, 1, 1.0))
        ctx.out.save_figure(fig, FACT, "volatility_scaling", "comparison")

    return {
        "fact": FACT, "id": FACT_ID, "priority": PRIORITY, "title": TITLE,
        "metrics": metrics,
        "literature": "proxy.ipynb cells 7/9: sigma(tau) ~ tau^H on a 100 ms grid; "
                      "H ~ 0.5 over the diffusive range, lower at sub-second lags.",
    }
