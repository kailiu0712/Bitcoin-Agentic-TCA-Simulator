"""
Stylized fact 22 (P1) - order-timing regularity: fixed intervals and
algorithmic clocks in the aggressive-order stream.

The question
    A metaorder executed by a scheduler (TWAP, or a POV with a fixed wake-up
    timer) emits child orders on a **clock**.  Real order flow carries no parent
    labels, but a clock leaves observable fingerprints, and each has a standard
    test:

    1. an excess of inter-arrival times at exactly the slice interval;
    2. phase concentration - events piling up at one phase of a fixed period;
    3. a narrow line in the spectrum of the point process;
    4. equal-sized clips arriving at regular intervals.

    Finding any of them would let child orders be identified directly, instead
    of assembled by the random mappings of ``fact_meta_notebook``, which have no
    way to know which trades belong together.

Method
    * **Inter-arrival fine structure** - a 1 ms histogram of gaps, with the count
      at each round candidate interval compared against the median of nearby
      bins (immediate neighbours excluded so a smeared spike cannot inflate its
      own baseline).  Model-free and direct.

    * **Rayleigh phase test** - for period T, map each event to
      theta = 2*pi*(t mod T)/T; the statistic z = |sum exp(i*theta)|^2 / N is
      Bartlett's periodogram of the point process at frequency 1/T.  Computed
      **per UTC day and averaged**, because an execution clock need not hold the
      same phase across days and the day is also the unit the periodogram
      averages over.  Under the iid null each daily z ~ Exp(1), so the mean over
      D days has mean 1 and standard deviation 1/sqrt(D).

    * **Bartlett periodogram** - events binned at 10 ms, mean-removed,
      Hann-windowed and transformed per day, then averaged.  Normalised so a
      Poisson process gives 1 at every frequency.  This is the blind search.

    * **Repeated-clip regularity** - orders sharing one exact quantity and side
      within a day are candidate children of one parent; the coefficient of
      variation of their inter-arrival times is compared against uniformly
      placed events of the same count (CV = 1 Poisson, CV -> 0 metronome).

Periods tested are PRE-DECLARED, and include controls
    ``CANDIDATE_PERIODS_S`` are round values a scheduler would plausibly use.
    ``CONTROL_PERIODS_S`` are deliberately incommensurate with those - no
    control is a rational harmonic of a round clock - so the null level can be
    read straight off the same table rather than assumed.

Why a raw p-value would be wrong, and what is used instead
    Both the Rayleigh statistic and the periodogram assume independent events.
    BTC order flow is violently clustered (fact 19: Fano factor 3 at 1 s rising
    to 28 at 300 s), which inflates both without any periodicity present.
    Calibrated on a simulated clustered-but-aperiodic process, an iid Rayleigh
    p-value called a non-existent 1 s period "p = 0.44", and a global-median
    spectral ratio reached 111.

    Two corrections, both validated on simulation before use here:

    * a **jitter null** - every event displaced by U(-5T, +5T), destroying
      structure at scale T while preserving coarser bursting.  On the simulated
      clustered process this returns p = 0.43 for an absent period and p = 0.005
      for a planted one.  Because an empirical p from B replicates cannot go
      below 1/(B+1), significance is flagged on the **standardised score**
      (observed - null mean) / null sd, which does not saturate;
    * a **local running median** as the spectral continuum, so a line is judged
      against its own neighbourhood.  On the simulated process the local ratio
      peaked at 21.0 against a theoretical expected maximum of 20.7 over 1.7M
      bins - correctly calibrated, where the global-median version was not.

Power, stated up front so a null result is readable
    The blind spectral search is weak: planting 2% of events on an exact 5 s
    clock inside clustered background produced a local ratio of only 4.4 against
    a ~31 detection threshold.  A negative spectral result therefore rules out
    only a *large* periodic subpopulation.  The phase and gap tests are far more
    powerful and carry the weight of any conclusion.

Clock
    Exchange timestamps, which carry the sender-side schedule; receipt time adds
    50-400 ms of collector jitter that smears periodicity.  The exchange clock
    is microsecond-resolution and shows no snapping to any coarser grid (the
    share of stamps divisible by 10 us / 100 us / 1 ms is 0.1002 / 0.0103 /
    0.00106, against random expectations of 0.1 / 0.01 / 0.001), so nothing here
    is a quantisation artifact.  Exchange time is not monotone as stored; it is
    sorted, and the number of reorderings is reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfcore import plotting as pl

FACT = "fact22"
FACT_ID = 22
PRIORITY = "P1"
TITLE = "Order-timing regularity: fixed intervals and algorithmic clocks"

DAY_NS = 86_400_000_000_000

#: Binning for the periodogram.  Nyquist is 50 Hz; results are reported below
#: 20 Hz, where the binning attenuation sinc(f*res) is under 7%.
RESOLUTION_S = 0.01
MIN_PERIOD_S, MAX_PERIOD_S = 0.05, 100.0

#: Round periods a scheduler would plausibly use.  Fixed before looking.
CANDIDATE_PERIODS_S: tuple[float, ...] = (
    0.1, 0.2, 0.25, 0.5, 1.0, 1.25, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0,
    60.0, 120.0, 300.0, 600.0)

#: Controls: none is a rational harmonic of a round clock, so their statistics
#: show what "no periodicity" looks like on this data.
CONTROL_PERIODS_S: tuple[float, ...] = (0.7, 1.1, 3.7, 7.0, 13.0, 41.0)

JITTER_REPLICATES = 100
JITTER_HALFWIDTH_PERIODS = 5.0

GAP_HIST_MAX_MS = 6000
ROUND_GAPS_MS: tuple[int, ...] = (50, 100, 200, 250, 500, 1000, 2000, 2500,
                                  3000, 5000)
GAP_BASELINE_HALFWIDTH_MS = 15
GAP_BASELINE_EXCLUDE_MS = 3

MIN_CLIP_REPEATS = 20
CLIP_QTY_DECIMALS = 8
CLIP_NULL_REPLICATES = 400
SPECTRAL_LINES_REPORTED = 15

#: Fixed bin width for the phase-fold panel, so the peak height is comparable
#: across periods (a fixed *number* of bins would not be).
PHASE_BIN_S = 0.01


def _clock_hygiene(rd) -> dict:
    exch = rd.trades["exch_ns"].to_numpy(np.int64)
    recv = rd.trades["recv_ns"].to_numpy(np.int64)
    gaps = np.diff(np.sort(exch))
    positive = gaps[gaps > 0]
    return {
        "orders": int(exch.size),
        "exchange_backsteps_as_stored": int((np.diff(exch) < 0).sum()),
        "share_multiple_of_10us": float((exch % 10_000 == 0).mean()),
        "share_multiple_of_1ms": float((exch % 1_000_000 == 0).mean()),
        "smallest_positive_gap_ns": int(positive.min()) if positive.size else -1,
        "share_zero_gap": float((gaps == 0).mean()),
        "median_gap_s": float(np.median(gaps) / 1e9),
        "median_recv_minus_exch_ms": float(np.median(recv - exch) / 1e6),
    }


def _gap_spikes(times_ns: np.ndarray):
    gaps_ms = np.diff(np.sort(times_ns)) / 1e6
    inside = gaps_ms[(gaps_ms >= 0) & (gaps_ms < GAP_HIST_MAX_MS)]
    counts = np.bincount(inside.astype(np.int64), minlength=GAP_HIST_MAX_MS)
    rows = []
    for target in ROUND_GAPS_MS:
        lo = max(target - GAP_BASELINE_HALFWIDTH_MS, 0)
        hi = min(target + GAP_BASELINE_HALFWIDTH_MS + 1, GAP_HIST_MAX_MS)
        window = np.arange(lo, hi)
        baseline_bins = window[np.abs(window - target) > GAP_BASELINE_EXCLUDE_MS]
        baseline = float(np.median(counts[baseline_bins])) if baseline_bins.size else np.nan
        observed = float(counts[target])
        rows.append({
            "gap_ms": target, "observed_count": observed,
            "local_baseline_count": baseline,
            "excess_ratio": observed / baseline if baseline > 0 else np.nan,
            "excess_sigma": ((observed - baseline) / np.sqrt(baseline)
                             if baseline > 0 else np.nan)})
    return pd.DataFrame(rows), counts


def _rayleigh(times_s: np.ndarray, period: float) -> float:
    """z = |sum exp(2.pi.i.t/T)|^2 / N.  Under iid uniform phases z ~ Exp(1)."""
    phase = np.mod(times_s, period) * (2.0 * np.pi / period)
    return float((np.sum(np.cos(phase)) ** 2
                  + np.sum(np.sin(phase)) ** 2) / times_s.size)


def _rayleigh_null_day(times_s, period, rng, replicates) -> np.ndarray:
    """Jitter null for one day, vectorised over replicates."""
    half = JITTER_HALFWIDTH_PERIODS * period
    shifted = times_s[None, :] + rng.uniform(-half, half, (replicates, times_s.size))
    phase = np.mod(shifted, period) * (2.0 * np.pi / period)
    return ((np.cos(phase).sum(axis=1) ** 2
             + np.sin(phase).sum(axis=1) ** 2) / times_s.size)


def _phase_tests(times_ns, continuum_at, rng) -> pd.DataFrame:
    """Per-day Rayleigh z with a per-day jitter null, plus the spectral ratio."""
    days = times_ns // DAY_NS
    per_day = [times_ns[days == u] - u * DAY_NS
               for u in np.unique(days) if (days == u).sum() >= 1000]
    n_days = len(per_day)
    rows = []
    for period, role in ([(p, "candidate") for p in CANDIDATE_PERIODS_S]
                         + [(p, "control") for p in CONTROL_PERIODS_S]):
        observed, null_means = [], []
        for block in per_day:
            seconds = block / 1e9
            observed.append(_rayleigh(seconds, period))
            null_means.append(_rayleigh_null_day(seconds, period, rng,
                                                 JITTER_REPLICATES))
        z = float(np.mean(observed))
        # Mean over days of the null, replicate by replicate, so the null of the
        # DAY-AVERAGED statistic is what the observation is judged against.
        null = np.mean(np.vstack(null_means), axis=0)
        sd = float(null.std(ddof=1))
        ratio = continuum_at(1.0 / period)
        rows.append({
            "period_s": period, "role": role, "n_days": n_days,
            "mean_z_per_day": z, "min_day_z": float(np.min(observed)),
            "max_day_z": float(np.max(observed)),
            "null_mean_z": float(null.mean()), "null_sd_z": sd,
            "score_vs_null": (z - float(null.mean())) / sd if sd > 0 else np.nan,
            "p_jitter": float(((null >= z).sum() + 1) / (JITTER_REPLICATES + 1)),
            "spectral_continuum": ratio,
            "ratio_to_continuum": z / ratio if ratio and ratio > 0 else np.nan,
        })
    table = pd.DataFrame(rows)
    # Bonferroni over every period tested, on the non-saturating score.
    threshold = float(-np.log(0.05 / len(table)))
    table["significant"] = table.score_vs_null > np.sqrt(2 * threshold)
    table.attrs["score_threshold"] = float(np.sqrt(2 * threshold))
    return table


def _periodogram(times_ns: np.ndarray):
    """Bartlett periodogram averaged over UTC days; Poisson -> 1 at every f.

    Verified on simulated Poisson data: mean 1.012, median 0.700 against the
    theoretical ln 2 = 0.693.
    """
    n_bins = int(round(86400.0 / RESOLUTION_S))
    window = np.hanning(n_bins)
    window_power = float(np.mean(window ** 2))
    total = np.zeros(n_bins // 2 + 1)
    days = 0
    for day in np.unique(times_ns // DAY_NS):
        within = times_ns[times_ns // DAY_NS == day] - day * DAY_NS
        if within.size < 1000:
            continue
        index = (within / (RESOLUTION_S * 1e9)).astype(np.int64)
        counts = np.bincount(index[(index >= 0) & (index < n_bins)],
                             minlength=n_bins).astype(np.float64)
        counts = (counts - counts.mean()) * window
        spectrum = np.fft.rfft(counts)
        total += (spectrum.real ** 2 + spectrum.imag ** 2) / (within.size * window_power)
        days += 1
    return np.fft.rfftfreq(n_bins, d=RESOLUTION_S), (total / max(days, 1)), days


def _continuum(freqs, power, window_bins: int = 501):
    """Local running-median continuum, and an interpolator onto any frequency."""
    band = (freqs >= 1.0 / MAX_PERIOD_S) & (freqs <= 1.0 / MIN_PERIOD_S)
    f, p = freqs[band], power[band]
    median = (pd.Series(p).rolling(window_bins, center=True,
                                   min_periods=window_bins // 4)
              .median().to_numpy())

    def at(frequency):
        if not (f[0] <= frequency <= f[-1]):
            return np.nan
        return float(np.interp(frequency, f, median))
    return f, p, median, at


def _spectral_lines(f, p, median) -> tuple[pd.DataFrame, dict]:
    ratio = p / np.maximum(median, 1e-12)
    threshold = float(np.log2(max(f.size, 2)))
    order = np.argsort(ratio)[::-1][:SPECTRAL_LINES_REPORTED]
    table = pd.DataFrame({
        "period_s": 1.0 / f[order], "frequency_hz": f[order], "power": p[order],
        "local_continuum": median[order], "ratio_to_continuum": ratio[order],
        "above_expected_max": ratio[order] > threshold})
    return (table.sort_values("ratio_to_continuum", ascending=False)
            .reset_index(drop=True),
            {"bins_searched": int(f.size), "expected_max_ratio": threshold,
             "max_ratio": float(np.nanmax(ratio))})


def _harmonic_comb(lines: pd.DataFrame) -> dict:
    """Is the set of detected lines a harmonic series of one fundamental?

    A single periodic pulse train produces power at every integer multiple of
    its fundamental, so a comb is the signature of one clock rather than many
    unrelated ones.  The fundamental is taken as the largest frequency that
    divides the detected lines to within 1%.
    """
    strong = lines.loc[lines.above_expected_max, "frequency_hz"].to_numpy()
    if strong.size < 3:
        return {"fundamental_hz": np.nan, "fundamental_period_s": np.nan,
                "lines_explained": 0, "lines_strong": int(strong.size)}
    best = None
    for base in np.unique(np.round(strong, 6)):
        for divisor in range(1, 21):
            f0 = base / divisor
            if f0 <= 0:
                continue
            multiples = strong / f0
            explained = int(np.sum(np.abs(multiples - np.round(multiples)) < 0.01
                                   * np.maximum(np.round(multiples), 1)))
            # Prefer the fundamental that explains most lines; break ties on the
            # LARGEST f0, since any submultiple trivially explains them too.
            key = (explained, f0)
            if best is None or key > best[0]:
                best = (key, f0)
    f0 = best[1]
    return {"fundamental_hz": float(f0), "fundamental_period_s": float(1.0 / f0),
            "lines_explained": int(best[0][0]), "lines_strong": int(strong.size)}


def _clip_regularity(orders: pd.DataFrame, rng):
    work = orders.assign(clip=np.round(orders.qty.to_numpy(), CLIP_QTY_DECIMALS))
    null_by_n: dict[int, np.ndarray] = {}
    rows = []
    for (day, sign, clip), block in work.groupby(["day", "sign", "clip"], sort=False):
        n = len(block)
        if n < MIN_CLIP_REPEATS:
            continue
        times = np.sort(block.ts_ns.to_numpy(np.int64)) / 1e9
        gaps = np.diff(times)
        if times[-1] - times[0] <= 0 or gaps.mean() <= 0:
            continue
        cv = float(gaps.std() / gaps.mean())
        if n not in null_by_n:
            draws = np.sort(rng.uniform(0.0, 1.0, (CLIP_NULL_REPLICATES, n)), axis=1)
            d = np.diff(draws, axis=1)
            null_by_n[n] = d.std(axis=1) / d.mean(axis=1)
        null = null_by_n[n]
        rows.append({
            "date": str(pd.Timestamp(int(day) * DAY_NS, tz="UTC").date()),
            "sign": int(sign), "clip_btc": float(clip), "n_orders": n,
            "span_s": float(times[-1] - times[0]), "mean_gap_s": float(gaps.mean()),
            "median_gap_s": float(np.median(gaps)), "cv_gap": cv,
            "null_cv_median": float(np.median(null)),
            "p_more_regular_than_uniform": float((null <= cv).mean())})
    table = pd.DataFrame(rows)
    if table.empty:
        return table, {"clip_groups": 0}
    table = table.sort_values("cv_gap").reset_index(drop=True)
    return table, {
        "clip_groups": len(table),
        "share_of_orders_in_clip_groups": float(table.n_orders.sum() / len(orders)),
        "median_cv": float(table.cv_gap.median()),
        "groups_more_regular_p01": int((table.p_more_regular_than_uniform < 0.01).sum()),
        "expected_false_positives_p01": 0.01 * len(table)}


def run(ctx) -> dict:
    pl.apply_style()
    metrics: list[dict] = []
    stored = {}

    for name, rd in ctx.each():
        rng = np.random.default_rng(20260905)
        hygiene = _clock_hygiene(rd)

        orders = rd.trades[["exch_ns", "sign", "qty"]].copy()
        orders["ts_ns"] = orders.exch_ns.astype("int64")
        orders = orders.sort_values("ts_ns", kind="stable").reset_index(drop=True)
        orders["day"] = orders.ts_ns // DAY_NS
        times_ns = orders.ts_ns.to_numpy(np.int64)

        gap_table, gap_counts = _gap_spikes(times_ns)
        freqs, power, days = _periodogram(times_ns)
        f, p, median, continuum_at = _continuum(freqs, power)
        lines, line_info = _spectral_lines(f, p, median)
        comb = _harmonic_comb(lines)
        lines["harmonic_number"] = (lines.frequency_hz / comb["fundamental_hz"]
                                    if np.isfinite(comb["fundamental_hz"]) else np.nan)
        phase = _phase_tests(times_ns, continuum_at, rng)
        clips, clip_info = _clip_regularity(orders, rng)

        quote_ns = np.sort(rd.quotes["exch_ns"].to_numpy(np.int64))
        q_freqs, q_power, _ = _periodogram(quote_ns)
        qf, qp, qmed, _ = _continuum(q_freqs, q_power)
        q_lines, q_info = _spectral_lines(qf, qp, qmed)

        candidates = phase.loc[phase.role == "candidate"]
        controls = phase.loc[phase.role == "control"]
        best = candidates.loc[candidates.mean_z_per_day.idxmax()]

        ctx.out.save_table(
            pd.DataFrame([{**hygiene, **line_info, **comb, **clip_info,
                           "control_median_z": float(controls.mean_z_per_day.median()),
                           "best_candidate_period_s": float(best.period_s),
                           "best_candidate_mean_z": float(best.mean_z_per_day)}]),
            FACT, "timing_diagnostics", name,
            caption=f"Clock hygiene, spectral search, harmonic comb and clip "
                    f"coverage - {rd.label}")
        ctx.out.save_table(gap_table, FACT, "round_gap_excess", name,
                           caption=f"Inter-arrival counts at round intervals vs a "
                                   f"local baseline - {rd.label}")
        ctx.out.save_table(phase, FACT, "phase_periodicity", name,
                           caption=f"Per-day Rayleigh phase test with jitter null; "
                                   f"controls included - {rd.label}")
        ctx.out.save_table(lines, FACT, "spectral_lines_orders", name,
                           caption=f"Strongest spectral lines, aggressive orders - {rd.label}")
        ctx.out.save_table(q_lines, FACT, "spectral_lines_quotes", name,
                           caption=f"Strongest spectral lines, top-of-book events - {rd.label}")
        if len(clips):
            ctx.out.save_table(clips.head(200), FACT, "repeated_clip_groups", name,
                               caption=f"Most regularly spaced repeated-clip groups "
                                       f"(top 200 of {len(clips)}) - {rd.label}")

        # Per-day phase profile at the winning period.
        fold = float(best.period_s)
        n_bins = max(int(round(fold / PHASE_BIN_S)), 20)
        profile_rows = []
        for day in np.unique(times_ns // DAY_NS):
            within = (times_ns[times_ns // DAY_NS == day]
                      - day * DAY_NS) / 1e9
            if within.size < 1000:
                continue
            hist, edges = np.histogram(np.mod(within, fold), bins=n_bins,
                                       range=(0, fold))
            profile_rows.append({
                "date": str(pd.Timestamp(int(day) * DAY_NS, tz="UTC").date()),
                "orders": int(within.size), "peak_over_mean": float(hist.max() / hist.mean()),
                "peak_phase_s": float(edges[np.argmax(hist)]),
                "rayleigh_z": _rayleigh(within, fold)})
        day_profile = pd.DataFrame(profile_rows)
        ctx.out.save_table(day_profile, FACT, "phase_profile_by_day", name,
                           caption=f"Phase concentration at {fold:g}s, day by day - {rd.label}")

        stored[name] = dict(phase=phase, clips=clips, f=f, p=p, qf=qf, qp=qp,
                            gap_counts=gap_counts, fold=fold, times_ns=times_ns,
                            days=days, line_info=line_info, comb=comb,
                            clip_info=clip_info, day_profile=day_profile)

        # ---------------- figure -------------------------------------------
        fig, axes = pl.new_figure(2, 2, width=5.9, height=4.4)

        ax = axes[0, 0]
        ax.plot(np.arange(GAP_HIST_MAX_MS), np.maximum(gap_counts, 0.5), lw=0.7,
                color=rd.color)
        for target in ROUND_GAPS_MS:
            ax.axvline(target, color="#E67E22", ls=":", lw=0.9, alpha=0.7)
        ax.set_yscale("log")
        top = gap_table.loc[gap_table.excess_ratio.idxmax()]
        pl.finish(ax, "Inter-arrival times, 1 ms bins",
                  "gap between aggressive orders (ms)", "count",
                  note=f"Dotted lines mark round intervals. Largest excess "
                       f"{top.excess_ratio:.2f}x the local baseline at "
                       f"{top.gap_ms:.0f} ms.")

        ax = axes[0, 1]
        for role, colour, marker in (("candidate", rd.color, "o"),
                                     ("control", "#9CA3AF", "s")):
            sub = phase.loc[phase.role == role]
            ax.plot(sub.period_s, sub.mean_z_per_day, marker, ms=6, color=colour,
                    ls="none", label=role)
        ax.axhline(1.0, **pl.REFERENCE_KW)
        ax.set_xscale("log")
        ax.set_yscale("log")
        pl.finish(ax, "Phase concentration (Rayleigh, per day)",
                  "tested period (s)", "mean z per day", legend=True,
                  note=f"Dashed line is the iid null (z=1). Controls are periods "
                       f"incommensurate with a round clock; their median is "
                       f"{controls.mean_z_per_day.median():.1f} against "
                       f"{best.mean_z_per_day:.0f} at {best.period_s:g}s.")

        ax = axes[1, 0]
        for label, xf, xp, colour in (("aggressive orders", f, p, rd.color),
                                      ("top-of-book events", qf, qp, "#9CA3AF")):
            ax.plot(1.0 / xf, xp, lw=0.6, color=colour, label=label)
        ax.axhline(1.0, **pl.REFERENCE_KW)
        ax.set_xscale("log")
        ax.set_yscale("log")
        pl.finish(ax, "Point-process spectrum", "period (s)",
                  "power (Poisson = 1)", legend=True,
                  note=f"Averaged over {days} days. Detected fundamental "
                       f"{comb['fundamental_period_s']:.3g}s explaining "
                       f"{comb['lines_explained']}/{comb['lines_strong']} strong "
                       f"lines as a harmonic comb; threshold ratio "
                       f"{line_info['expected_max_ratio']:.0f}, observed max "
                       f"{line_info['max_ratio']:.0f}.")

        ax = axes[1, 1]
        for day in np.unique(times_ns // DAY_NS):
            within = (times_ns[times_ns // DAY_NS == day] - day * DAY_NS) / 1e9
            if within.size < 1000:
                continue
            hist, edges = np.histogram(np.mod(within, fold), bins=n_bins,
                                       range=(0, fold))
            ax.plot(edges[:-1], hist / hist.mean(), lw=1.0, alpha=0.75)
        ax.axhline(1.0, **pl.REFERENCE_KW)
        pl.finish(ax, f"Arrival phase folded at {fold:g} s",
                  f"phase within the {fold:g} s cycle (s)",
                  "orders / uniform expectation",
                  note=f"One line per day, {PHASE_BIN_S*1000:.0f} ms bins. A flat "
                       "line is no clock. Peak reaches "
                       f"{day_profile.peak_over_mean.max():.1f}x uniform.")

        fig.suptitle(f"Fact 22 - order-timing regularity | {rd.label}", x=0.02,
                     ha="left", fontsize=12.5, fontweight="semibold")
        pl.layout(fig)
        ctx.out.save_figure(fig, FACT, "order_timing", name)

        metrics += [
            {"metric": "strongest periodic clock (s)", "regime": name,
             "value": float(best.period_s),
             "detail": f"per-day Rayleigh z={best.mean_z_per_day:.0f} vs iid null 1 "
                       f"and control median {controls.mean_z_per_day.median():.1f}"},
            {"metric": "spectral fundamental period (s)", "regime": name,
             "value": comb["fundamental_period_s"],
             "detail": f"explains {comb['lines_explained']} of "
                       f"{comb['lines_strong']} strong lines as a harmonic comb"},
            {"metric": "candidate periods significant vs jitter null", "regime": name,
             "value": float(candidates.significant.sum()),
             "detail": f"of {len(candidates)}; controls flagged: "
                       f"{int(controls.significant.sum())} of {len(controls)}"},
            {"metric": "peak arrival rate at the clock phase (x uniform)",
             "regime": name, "value": float(day_profile.peak_over_mean.max()),
             "detail": f"folded at {fold:g}s, best day; "
                       f"median across days {day_profile.peak_over_mean.median():.2f}"},
            {"metric": "largest round-gap excess (x local baseline)", "regime": name,
             "value": float(gap_table.excess_ratio.max()),
             "detail": f"at {top.gap_ms:.0f} ms"},
            {"metric": "repeated-clip groups more regular than uniform (p<0.01)",
             "regime": name, "value": float(clip_info.get("groups_more_regular_p01", 0)),
             "detail": f"of {clip_info.get('clip_groups', 0)} groups, "
                       f"{clip_info.get('expected_false_positives_p01', 0):.0f} expected "
                       f"by chance; median CV {clip_info.get('median_cv', float('nan')):.2f} "
                       "(1 = Poisson)"},
        ]

    if stored:
        fig, axes = pl.new_figure(1, 2, width=5.8, height=4.4)
        for name, rd in ctx.each():
            block = stored.get(name)
            if block is None:
                continue
            sub = block["phase"]
            axes[0].plot(sub.loc[sub.role == "candidate", "period_s"],
                         sub.loc[sub.role == "candidate", "mean_z_per_day"],
                         "o-", ms=5, lw=1.6, color=rd.color, label=f"{rd.label}")
            axes[0].plot(sub.loc[sub.role == "control", "period_s"],
                         sub.loc[sub.role == "control", "mean_z_per_day"],
                         "s", ms=5, color=rd.color, alpha=0.4)
            axes[1].plot(1.0 / block["f"], block["p"], lw=0.5, color=rd.color,
                         label=rd.label)
        axes[0].axhline(1.0, **pl.REFERENCE_KW)
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        pl.finish(axes[0], "Phase concentration by regime", "tested period (s)",
                  "mean Rayleigh z per day", legend=True,
                  note="Circles are round candidate periods, faint squares the "
                       "incommensurate controls.")
        axes[1].axhline(1.0, **pl.REFERENCE_KW)
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        pl.finish(axes[1], "Point-process spectrum by regime", "period (s)",
                  "power (Poisson = 1)", legend=True)
        fig.suptitle("Fact 22 - order-timing regularity by regime", x=0.02,
                     ha="left", fontsize=12.5, fontweight="semibold")
        pl.layout(fig, rect=(0, 0, 1, 0.94))
        ctx.out.save_figure(fig, FACT, "order_timing", "comparison")

    return {
        "fact": FACT, "id": FACT_ID, "priority": PRIORITY, "title": TITLE,
        "metrics": metrics,
        "literature": "Order splitting is the accepted source of persistent sign "
                      "flow (Lillo-Mike-Farmer 2005; Toth et al. 2015, with broker "
                      "labels), and Hasbrouck & Saar (2013) identify linked message "
                      "sequences from inter-message timing. Bartlett's periodogram "
                      "and the Rayleigh test are the standard periodicity tools for "
                      "point processes. No published estimate exists for "
                      "scheduled-execution prevalence on crypto venues, so the "
                      "numbers here are descriptive rather than a comparison.",
    }
