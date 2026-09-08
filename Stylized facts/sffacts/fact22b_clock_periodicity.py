"""
Stylized fact 22b (P1) - clock-time periodicity in the published convention.

Why this exists next to fact22
    ``fact22`` searches for *any* periodicity using tools imported from pulsar
    timing and spike-train analysis (Rayleigh test, point-process spectrum,
    jitter null).  Those are stronger than what the finance literature uses, but
    they are not what the finance literature *reports*, so its numbers cannot be
    quoted against published work.

    This module adds nothing to the search.  It re-expresses the same event
    times in the three descriptive conventions the published papers use, so each
    of their claims can be checked directly on this dataset:

    * **Muravyev & Picard (2022)**, Financial Management - US equities.  Trades
      and quote updates are much more frequent within the **first 100 ms of a
      second**, and activity spikes at intervals of exactly one second.
      Convention: deciles of the second.
    * **Shynkevich (2026)**, Applied Economics Letters - spot BTC and ETH.
      Periodicity at the one-second frequency, with trade count rising sharply
      during the **first and last seconds of the minute**.  Convention:
      second-of-minute profile.
    * **"The Quarter-Hour Effect"**, arXiv:2607.09426 - Binance perpetuals.
      Nested periodicity at 1, 5, 15 and 60 minute marks, with quarter-hour
      openings carrying ~26% more trades.  Convention: second-of-hour and
      minute-of-hour profiles, read visually.

    None of those papers runs a formal periodicity test - the Quarter-Hour paper
    says so explicitly.  So this module deliberately does not either: it reports
    the profiles and a replication verdict per claim, and leaves the inference
    to fact22, which has the statistical machinery and the controls.

The null band
    A Poisson band would be far too narrow: order arrivals are clustered (fact
    19), so bin counts are over-dispersed.  Instead every profile is computed
    **per UTC day** and the reported band is the standard error of the mean
    across days.  Days are the natural quasi-independent replicate here, and
    this absorbs the clustering without any model.

Reading the profiles
    A bin's value is (orders in that bin) / (mean orders per bin **within the
    same day**), so a flat line at 1.0 is "no clock" and the level is unaffected
    by how busy a day was.  Bins are labelled by their **start**: "second 0"
    means [0, 1) seconds past the minute.

Clock
    Exchange timestamps, sorted; see fact22 for the verification that this feed's
    exchange clock is microsecond-resolution and not snapped to any coarser grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfcore import plotting as pl

FACT = "fact22b"
FACT_ID = 22
PRIORITY = "P1"
TITLE = "Clock-time periodicity in the published convention (replication)"

DAY_NS = 86_400_000_000_000

#: Muravyev & Picard report deciles of the second; a 1 ms profile is added
#: because on this venue the peak turns out to sit well inside one decile.
SECOND_DECILES = 10
SECOND_MILLISECOND_BINS = 1000

#: Quarter-hour openings, per the Quarter-Hour Effect paper.
QUARTER_HOUR_MINUTES = (0, 15, 30, 45)


def _profile(seconds_of_day: np.ndarray, days: np.ndarray, modulus: float,
             n_bins: int) -> pd.DataFrame:
    """Per-day folded profile, normalised within each day, averaged over days.

    Normalising inside the day before averaging is what makes a busy day and a
    quiet day count equally, and it is what allows the across-day standard error
    to serve as the null band.
    """
    rows = []
    for day in np.unique(days):
        within = seconds_of_day[days == day]
        if within.size < 1000:
            continue
        counts, _ = np.histogram(np.mod(within, modulus), bins=n_bins,
                                 range=(0.0, modulus))
        mean = counts.mean()
        if mean <= 0:
            continue
        rows.append(counts / mean)
    if not rows:
        return pd.DataFrame()
    matrix = np.vstack(rows)
    edges = np.linspace(0.0, modulus, n_bins + 1)
    counts_total, _ = np.histogram(np.mod(seconds_of_day, modulus), bins=n_bins,
                                   range=(0.0, modulus))
    return pd.DataFrame({
        "bin_start": edges[:-1], "bin_end": edges[1:],
        "orders": counts_total,
        "ratio_to_uniform": matrix.mean(axis=0),
        "sem_across_days": matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0]),
        "n_days": matrix.shape[0],
    })


def _verdict(observed, baseline, sem, claim, source) -> dict:
    """Replication verdict for one published claim, on this dataset."""
    excess = observed - baseline
    sigma = excess / sem if sem and sem > 0 else np.nan
    return {"source": source, "claim": claim, "observed_ratio": observed,
            "baseline_ratio": baseline, "excess_pct": 100.0 * (observed / baseline - 1.0)
            if baseline else np.nan, "sem_across_days": sem,
            "sigma_vs_across_day_noise": sigma,
            "replicated": bool(np.isfinite(sigma) and sigma > 2.0)}


def run(ctx) -> dict:
    pl.apply_style()
    metrics: list[dict] = []
    stored = {}

    for name, rd in ctx.each():
        times = np.sort(rd.trades["exch_ns"].to_numpy(np.int64))
        days = times // DAY_NS
        seconds = (times - days * DAY_NS) / 1e9

        minute = _profile(seconds, days, 60.0, 60)          # second-of-minute
        hour_minutes = _profile(seconds, days, 3600.0, 60)  # minute-of-hour
        hour_seconds = _profile(seconds, days, 3600.0, 3600)  # second-of-hour
        decile = _profile(seconds, days, 1.0, SECOND_DECILES)
        millisecond = _profile(seconds, days, 1.0, SECOND_MILLISECOND_BINS)
        if minute.empty:
            continue

        decile = decile.assign(
            decile_of_second=np.arange(SECOND_DECILES),
            ms_from=(decile.bin_start * 1000).round().astype(int),
            ms_to=(decile.bin_end * 1000).round().astype(int))

        # ---- replication verdicts against each published claim -------------
        others_minute = minute.drop(index=[0, 59])
        quarter = hour_minutes.iloc[list(QUARTER_HOUR_MINUTES)]
        ordinary = hour_minutes.drop(index=list(QUARTER_HOUR_MINUTES))
        rest_of_second = decile.drop(index=[0])
        verdicts = pd.DataFrame([
            _verdict(float(decile.ratio_to_uniform.iloc[0]),
                     float(rest_of_second.ratio_to_uniform.mean()),
                     float(decile.sem_across_days.iloc[0]),
                     "activity concentrated in the first 100 ms of the second",
                     "Muravyev & Picard (2022), Financial Management"),
            _verdict(float(minute.ratio_to_uniform.iloc[0]),
                     float(others_minute.ratio_to_uniform.mean()),
                     float(minute.sem_across_days.iloc[0]),
                     "trade count rises in the FIRST second of the minute",
                     "Shynkevich (2026), Applied Economics Letters"),
            _verdict(float(minute.ratio_to_uniform.iloc[59]),
                     float(others_minute.ratio_to_uniform.mean()),
                     float(minute.sem_across_days.iloc[59]),
                     "trade count rises in the LAST second of the minute",
                     "Shynkevich (2026), Applied Economics Letters"),
            _verdict(float(quarter.ratio_to_uniform.mean()),
                     float(ordinary.ratio_to_uniform.mean()),
                     float(np.sqrt(np.sum(quarter.sem_across_days ** 2)) / len(quarter)),
                     "quarter-hour openings carry ~26% more trades",
                     "The Quarter-Hour Effect, arXiv:2607.09426"),
            # The pooled quarter-hour figure above can be carried entirely by the
            # top of the hour, so the two are separated rather than left merged.
            _verdict(float(hour_minutes.ratio_to_uniform.iloc[0]),
                     float(ordinary.ratio_to_uniform.mean()),
                     float(hour_minutes.sem_across_days.iloc[0]),
                     "  ... of which: top of the hour (minute 0) alone",
                     "The Quarter-Hour Effect, arXiv:2607.09426"),
            _verdict(float(hour_minutes.ratio_to_uniform.iloc[[15, 30, 45]].mean()),
                     float(ordinary.ratio_to_uniform.mean()),
                     float(np.sqrt(np.sum(
                         hour_minutes.sem_across_days.iloc[[15, 30, 45]] ** 2)) / 3),
                     "  ... of which: minutes 15, 30, 45 alone",
                     "The Quarter-Hour Effect, arXiv:2607.09426"),
        ])

        for table, slug, caption in (
            (minute, "second_of_minute", "Trades by second of the minute"),
            (hour_minutes, "minute_of_hour", "Trades by minute of the hour"),
            (hour_seconds, "second_of_hour", "Trades by second of the hour"),
            (decile, "decile_of_second", "Trades by 100 ms decile of the second"),
            (millisecond, "millisecond_of_second", "Trades by millisecond of the second"),
            (verdicts, "replication_verdicts", "Published claims checked on this dataset"),
        ):
            ctx.out.save_table(table, FACT, slug, name, caption=f"{caption} - {rd.label}")

        stored[name] = dict(minute=minute, hour_minutes=hour_minutes,
                            hour_seconds=hour_seconds, decile=decile,
                            millisecond=millisecond, verdicts=verdicts)

        # ------------------------------ figure -----------------------------
        fig, axes = pl.new_figure(2, 2, width=5.9, height=4.4)

        ax = axes[0, 0]
        x = minute.bin_start.to_numpy()
        ax.bar(x, minute.ratio_to_uniform, width=0.85, color="#C7D3E3")
        ax.bar([0, 59], minute.ratio_to_uniform.iloc[[0, 59]], width=0.85,
               color=rd.color)
        ax.errorbar(x, minute.ratio_to_uniform, yerr=minute.sem_across_days,
                    fmt="none", ecolor="#6B7280", elinewidth=0.8)
        ax.axhline(1.0, **pl.REFERENCE_KW)
        pl.finish(ax, "Trades by second of the minute", "second of the minute",
                  "orders / uniform expectation",
                  note=f"Shynkevich (2026) reports a surge in the first AND last "
                       f"second for spot BTC. Here second 0 is "
                       f"{minute.ratio_to_uniform.iloc[0]:.2f}x but second 59 is "
                       f"{minute.ratio_to_uniform.iloc[59]:.2f}x. Bars are SEM "
                       f"across {int(minute.n_days.iloc[0])} days.")

        ax = axes[0, 1]
        xm = hour_minutes.bin_start.to_numpy() / 60.0
        ax.bar(xm, hour_minutes.ratio_to_uniform, width=0.85, color="#C7D3E3")
        ax.bar([m for m in QUARTER_HOUR_MINUTES],
               hour_minutes.ratio_to_uniform.iloc[list(QUARTER_HOUR_MINUTES)],
               width=0.85, color="#E67E22")
        ax.errorbar(xm, hour_minutes.ratio_to_uniform,
                    yerr=hour_minutes.sem_across_days, fmt="none",
                    ecolor="#6B7280", elinewidth=0.8)
        ax.axhline(1.0, **pl.REFERENCE_KW)
        pl.finish(ax, "Trades by minute of the hour", "minute of the hour",
                  "orders / uniform expectation",
                  note=f"Orange are the quarter-hour openings (0, 15, 30, 45). "
                       f"They average {quarter.ratio_to_uniform.mean():.3f}x "
                       f"against {ordinary.ratio_to_uniform.mean():.3f}x for the "
                       f"other 56 minutes; arXiv:2607.09426 report ~26% on "
                       f"Binance perpetuals.")

        ax = axes[1, 0]
        ax.bar(decile.decile_of_second * 100 + 50, decile.ratio_to_uniform,
               width=85, color="#C7D3E3", label="100 ms deciles")
        ax.bar(50, decile.ratio_to_uniform.iloc[0], width=85, color=rd.color,
               label="first 100 ms (Muravyev & Picard)")
        ax.plot(millisecond.bin_start * 1000 + 0.5, millisecond.ratio_to_uniform,
                lw=0.8, color="#C0392B", alpha=0.85, label="1 ms detail")
        ax.axhline(1.0, **pl.REFERENCE_KW)
        peak_ms = float(millisecond.bin_start.iloc[
            int(millisecond.ratio_to_uniform.idxmax())] * 1000)
        pl.finish(ax, "Trades within the second", "milliseconds past the second",
                  "orders / uniform expectation", legend=True,
                  note=f"Muravyev & Picard (2022) find the excess in the FIRST "
                       f"100 ms on US equities. Here that decile is "
                       f"{decile.ratio_to_uniform.iloc[0]:.2f}x while the peak is "
                       f"{decile.ratio_to_uniform.max():.2f}x at "
                       f"{int(decile.ms_from.iloc[int(decile.ratio_to_uniform.idxmax())])}"
                       f"-{int(decile.ms_to.iloc[int(decile.ratio_to_uniform.idxmax())])} ms "
                       f"(1 ms peak at {peak_ms:.0f} ms).")

        ax = axes[1, 1]
        ax.plot(hour_seconds.bin_start, hour_seconds.ratio_to_uniform, lw=0.5,
                color=rd.color)
        for mark, colour in ((60, "#9CA3AF"), (300, "#2E7D32"), (900, "#E67E22")):
            for pos in range(0, 3600, mark):
                ax.axvline(pos, color=colour, lw=0.4, alpha=0.25, zorder=0)
        ax.axhline(1.0, **pl.REFERENCE_KW)
        pl.finish(ax, "Trades by second of the hour", "second of the hour",
                  "orders / uniform expectation",
                  note=f"The convention of arXiv:2607.09426. Faint lines mark the "
                       f"minute, 5-minute and quarter-hour boundaries. Only "
                       f"{hour_seconds.orders.mean():.0f} orders per bin here "
                       f"against four years of data there, so this panel is noisy "
                       f"by construction - the aggregated panels above carry the "
                       f"evidence.")

        fig.suptitle(f"Fact 22b - clock periodicity, published conventions | "
                     f"{rd.label}", x=0.02, ha="left", fontsize=12.5,
                     fontweight="semibold")
        pl.layout(fig)
        ctx.out.save_figure(fig, FACT, "clock_conventions", name)

        for _, row in verdicts.iterrows():
            metrics.append({
                "metric": f"{row.claim} [{row.source.split('(')[0].strip()}]",
                "regime": name, "value": float(row.observed_ratio),
                "detail": f"{row.excess_pct:+.1f}% vs baseline "
                          f"{row.baseline_ratio:.3f}x, "
                          f"{row.sigma_vs_across_day_noise:.1f} sigma of across-day "
                          f"noise -> {'replicated' if row.replicated else 'NOT replicated'}"})

    if stored:
        fig, axes = pl.new_figure(1, 2, width=5.8, height=4.4)
        for name, rd in ctx.each():
            block = stored.get(name)
            if block is None:
                continue
            axes[0].plot(block["minute"].bin_start, block["minute"].ratio_to_uniform,
                         "o-", ms=3, lw=1.3, color=rd.color, label=rd.label)
            axes[1].plot(block["millisecond"].bin_start * 1000,
                         block["millisecond"].ratio_to_uniform, lw=0.8,
                         color=rd.color, label=rd.label)
        for ax in axes:
            ax.axhline(1.0, **pl.REFERENCE_KW)
        pl.finish(axes[0], "Second of the minute by regime", "second of the minute",
                  "orders / uniform expectation", legend=True)
        pl.finish(axes[1], "Within the second by regime",
                  "milliseconds past the second", "orders / uniform expectation",
                  legend=True)
        fig.suptitle("Fact 22b - clock periodicity by regime", x=0.02, ha="left",
                     fontsize=12.5, fontweight="semibold")
        pl.layout(fig, rect=(0, 0, 1, 0.94))
        ctx.out.save_figure(fig, FACT, "clock_conventions", "comparison")

    return {
        "fact": FACT, "id": FACT_ID, "priority": PRIORITY, "title": TITLE,
        "metrics": metrics,
        "literature": "Muravyev & Picard (2022, Financial Management): US equity "
                      "activity concentrated in the first 100 ms of a second, plus "
                      "one-second spikes. Shynkevich (2026, Applied Economics "
                      "Letters): spot BTC/ETH periodicity at one second, rising in "
                      "the first and last seconds of the minute. The Quarter-Hour "
                      "Effect (arXiv:2607.09426): nested 1/5/15/60-minute marks on "
                      "Binance perpetuals, quarter-hour openings ~26% busier. None "
                      "runs a formal periodicity test; fact22 supplies that.",
    }
