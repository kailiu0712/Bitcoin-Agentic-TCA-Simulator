"""
Facts 1-4 (P0-V proxy) - the proxy.ipynb metaorder battery, run over three
proxy constructions on identical prices, scales and filters.

What this module is for
    ``proxy.ipynb`` runs eight tests on synthetic metaorders that the framework
    did not have: a continuous broken power law in size, the duration
    dependence of the rescaled impact factor, the execution trajectory on
    log-log axes, a propagator fit to the post-completion decay on the rescaled
    clock z = t/T, a shuffled-sign null, the local volume-clock exponent, and
    the joint (Q, T) regressions that check whether the two-step fit is
    identified at all.  This module implements those tests.

The three proxies (facts 1-4 are proxy evidence, so the construction is a
variable, not a fixed choice)

    ``current_run``          the framework's own pseudo-metaorder: a maximal
                             run of same-sign aggressive orders separated by
                             less than ``PSEUDO_METAORDER_GAP_S``.
    ``random_n6``            proxy.ipynb's mapping and Maitrier et al.
                             arXiv:2503.18199: assign each aggressive order to
                             one of N=6 synthetic traders uniformly at random,
                             then cut each trader's own stream at its own sign
                             changes.
    ``maitrier_threshold``   the local ``Maitrier et al. (2025).pdf``
                             (arXiv:2509.05065) Algorithm 2, p17: split by
                             side, draw a child count from a truncated
                             ``s^-(1+mu)``, then walk forward by exponential
                             waiting times, stopping when the next same-sign
                             trade overshoots the target by more than C/phi.

    All three are built by ``sfcore.metaorder_proxies``, per UTC day, and then
    measured by exactly the same code below.  Differences between the columns
    are differences between grouping operators, not between measurements.

The correction this module carries
    proxy.ipynb pre-filters individual metaorders to ``I_norm > 0`` before
    fitting the size curve, the duration scaling, the trajectory and the decay.
    That is a selection on the outcome: it estimates the impact of metaorders
    *that happened to be followed by a favourable move*, which is a different
    and much larger quantity than the average signed impact of a metaorder.
    Every primary result here keeps negative and zero impacts.  The notebook's
    positive-only selection is still computed and reported as
    ``selection = positive_only`` so the size of the distortion is visible.

    A log-space fit still cannot consume a non-positive number.  So the fits
    run on *bin means* - which stay positive wherever the signed effect is
    real - and every bin whose mean is non-positive is dropped and counted.
    No individual observation is filtered.

Endpoint convention
    Primary is the notebook's: the impact of a parent is
    ``sign * (mid_last_child - mid_first_child) / mid_first_child``, where each
    mid is the exchange-time-aligned *pre-trade* mid carried on the trade tape.
    That excludes the last child's own effect, which is why
    ``impact_next_norm`` - measured to the pre-trade mid of the next aggressive
    order after completion, as ``fact_meta_compare`` does - is carried in the
    same table as a sensitivity.  Neither is "correct"; they answer different
    questions and both are reported.

Scaling closure
    The notebook closes the loop with ``gamma = H - delta_2 * nu``, taking H
    from the price-diffusion scaling (``fact18_return_scaling``) and nu from
    the local volume clock measured here.  That relation is a model
    consequence, not an independent stylized fact, and it is reported as a
    residual to look at rather than a test to pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sfcore import fits
from sfcore import plotting as pl
from sfcore.metaorder_proxies import build_parents
from sffacts import fact18_return_scaling as f18
from sffacts.fact_meta_compare import _daily_scales, asof_prices, load_analysis_rows

FACT = "fact_meta_nb"
FACT_ID = 1
PRIORITY = "P0-V proxy"
TITLE = "Metaorder impact battery (proxy.ipynb) over three proxy constructions"

DAY_NS = 86_400_000_000_000

#: Size bins.  proxy.ipynb cell 2 uses 45 log-spaced bins between the 0.05th
#: and 99.95th percentiles of Q/V_D.
N_SIZE_BINS = 45
SIZE_QUANTILES = (0.0005, 0.9995)
MIN_BIN_COUNT = 5

#: Duration bins (cells 3, 4, 11).
N_DURATION_BINS = 20
MIN_DURATION_S = 0.5

#: Volume-progress grid for the trajectory (cell 5), including exactly 0.5 and 1.
PHI_GRID = np.linspace(0.05, 1.0, 20)
MIN_TRAJECTORY_CHILDREN = 4

#: Rescaled-clock grid for the decay (cell 6).
Z_GRID = np.linspace(1.0, 2.0, 25)
MIN_DECAY_CHILDREN = 3
MIN_DECAY_DURATION_S = 1.0

MIN_CHILDREN = 2

#: Quote staleness guard for every as-of price lookup, seconds.
MAX_QUOTE_AGE_S = 30.0


def _variants(cfg):
    """(label, method, params, seeds) for the three proxy constructions."""
    settings = cfg.METAORDER_COMPARISON
    seeds = list(settings["seeds"])
    return [
        ("current_run", "local_runs",
         dict(gap_s=cfg.PSEUDO_METAORDER_GAP_S), [seeds[0]]),
        ("random_n6", "random_traders",
         dict(n_traders=6), seeds),
        ("maitrier_threshold", "time_threshold",
         dict(mean_wait_s=settings["mean_wait_seconds"][0], mu=settings["mu"],
              max_children=settings["max_children"],
              max_target_delay_s=settings["max_target_delay_s"]), seeds),
    ]


def _parents_for_day(day, parents, orders, scale):
    """Notebook-convention measurement of one day's parents.

    ``day`` holds that day's aggressive orders; ``parents`` are lists of row
    positions into it; ``orders`` is the whole regime, needed for the
    next-order endpoint and for the market volume inside each parent's window.
    """
    kept = [p for p in parents if len(p) >= MIN_CHILDREN]
    if not kept:
        return pd.DataFrame(), []
    ts = day.ts_ns.to_numpy(np.int64)
    mid = day.mid.to_numpy(np.float64)
    qty = day.qty.to_numpy(np.float64)
    sign = day.sign.to_numpy(np.float64)

    first = np.array([p[0] for p in kept])
    last = np.array([p[-1] for p in kept])
    side = sign[first]
    p_start, p_end = mid[first], mid[last]
    starts, ends = ts[first], ts[last]
    amounts = np.array([qty[p].sum() for p in kept])

    # Market volume inside the execution window, and the pre-trade mid of the
    # first aggressive order strictly after completion (the sensitivity
    # endpoint).  Both need the whole regime, not just this day.
    order_ts = orders.ts_ns.to_numpy(np.int64)
    order_mid = orders.mid.to_numpy(np.float64)
    order_day = orders.day.to_numpy(np.int64)
    cum_qty = np.concatenate([[0.0], np.cumsum(orders.qty.to_numpy(np.float64))])
    left = np.searchsorted(order_ts, starts, side="left")
    right = np.searchsorted(order_ts, ends, side="right")
    window_volume = cum_qty[right] - cum_qty[left]
    following = np.minimum(right, len(orders) - 1)
    has_next = (right < len(orders)) & (order_day[following] == scale.day)
    p_next = np.where(has_next, order_mid[following], np.nan)

    impact = side * (p_end - p_start) / p_start
    frame = pd.DataFrame({
        "t_start_ns": starts, "t_end_ns": ends, "sign": side,
        "n_child": [len(p) for p in kept], "Q_btc": amounts,
        "duration_s": (ends - starts) / 1e9,
        "mid_start": p_start, "mid_end": p_end,
        "impact": impact,
        "I_norm": impact / scale.sigma_range,
        "impact_next_norm": (side * (p_next - p_start) / p_start) / scale.sigma_range,
        "Q_norm": amounts / scale.volume,
        "V_window_btc": window_volume,
        "participation": np.where(window_volume > 0, amounts / window_volume, np.nan),
        "date": scale.date, "day": scale.day, "sigma_range": scale.sigma_range,
    })
    trajectories = [(k, p) for k, p in enumerate(kept)
                    if len(p) >= MIN_TRAJECTORY_CHILDREN]
    return frame, trajectories


def _trajectory_rows(day, trajectories, frame):
    """Per-parent impact path on a common volume-progress grid, equally weighted.

    proxy.ipynb pools every child point and then bins phi, which lets the
    parents that happened to have many children dominate a bin.  Interpolating
    each parent onto the shared grid first and averaging the parents gives each
    metaorder one vote, which is what "the average trajectory" should mean.
    """
    if not trajectories:
        return np.empty((0, PHI_GRID.size))
    mid = day.mid.to_numpy(np.float64)
    qty = day.qty.to_numpy(np.float64)
    sign = frame["sign"].to_numpy(np.float64)
    start = frame["mid_start"].to_numpy(np.float64)
    total = frame["Q_btc"].to_numpy(np.float64)
    paths = []
    for k, rows in trajectories:
        progress = np.cumsum(qty[rows]) / total[k]
        # Impact is read at each child's own PRE-trade mid, so the point at
        # progress phi is the price before the child that completes phi.
        walk = sign[k] * (mid[rows] - start[k]) / start[k]
        paths.append(np.interp(PHI_GRID, np.r_[0.0, progress], np.r_[0.0, walk]))
    return np.vstack(paths)


def _binned(frame, x, y, edges, min_count=MIN_BIN_COUNT):
    """Conditional mean of ``y`` on log-spaced bins of ``x``, with an iid SEM."""
    work = pd.DataFrame({"x": frame[x], "y": frame[y]}).dropna()
    work = work.loc[work.x > 0]
    if work.empty:
        return pd.DataFrame()
    work["bin"] = pd.cut(work.x, edges, labels=False, include_lowest=True)
    out = work.groupby("bin", observed=True).agg(
        x=("x", "mean"), y=("y", "mean"), sem_iid=("y", "sem"),
        n=("y", "count")).reset_index()
    return out.loc[out.n >= min_count].reset_index(drop=True)


def _size_edges(values):
    lo, hi = np.quantile(values, SIZE_QUANTILES)
    if not (hi > lo > 0):
        return None
    return np.logspace(np.log10(lo), np.log10(hi), N_SIZE_BINS)


def _decay_curve(frame, quote_ts, quote_mid, delta, max_age_ns):
    """Normalised impact on the rescaled clock z = (t - t_start)/T, complete cohort.

    Only parents whose entire z path stays inside the same UTC day and inside a
    fresh quote window are kept, and the same parents are used at every z, so
    the curve cannot bend because its sample changed underneath it.
    """
    pick = frame.loc[(frame.n_child >= MIN_DECAY_CHILDREN)
                     & (frame.duration_s >= MIN_DECAY_DURATION_S)
                     & (frame.Q_norm > 0)].copy()
    if len(pick) < 100 or not np.isfinite(delta):
        return pd.DataFrame()
    start = pick.t_start_ns.to_numpy(np.int64)
    span = (pick.t_end_ns.to_numpy(np.int64) - start).astype(np.float64)
    queries = start[:, None] + (span[:, None] * Z_GRID[None, :]).astype(np.int64)
    prices = asof_prices(quote_ts, quote_mid, queries, max_age_ns)
    prices[(queries // DAY_NS) != pick.day.to_numpy()[:, None]] = np.nan
    complete = np.isfinite(prices).all(axis=1)
    if complete.sum() < 50:
        return pd.DataFrame()
    good = pick.loc[complete]
    base = good.mid_start.to_numpy()[:, None]
    walk = good.sign.to_numpy()[:, None] * (prices[complete] - base) / base
    denominator = (good.sigma_range.to_numpy() * good.Q_norm.to_numpy() ** delta)
    normalised = walk / denominator[:, None]
    return pd.DataFrame({
        "z": Z_GRID,
        "normalised_impact": normalised.mean(axis=0),
        "sem_iid": normalised.std(axis=0, ddof=1) / np.sqrt(len(good)),
        "n": len(good), "n_before_complete_filter": len(pick),
    })


def _joint_regressions(frame):
    """Q-T endogeneity diagnostics and the joint log regressions (cells 12, 13).

    These are the one place a positive-impact subsample is unavoidable: the
    regressand is ``log I_norm``.  That is stated in the output rather than
    worked around, and the share of parents it keeps is reported next to it.
    """
    pick = frame.loc[(frame.Q_norm > 0) & (frame.duration_s > 0)
                     & frame.I_norm.notna()]
    if len(pick) < 200:
        return {}
    log_q_all = np.log(pick.Q_norm.to_numpy())
    log_t_all = np.log(pick.duration_s.to_numpy())
    correlation = float(np.corrcoef(log_q_all, log_t_all)[0, 1])
    positive = pick.loc[pick.I_norm > 0]
    result = {"rho_logQ_logT": correlation, "n_all": len(pick),
              "n_positive": len(positive),
              "positive_share": len(positive) / len(pick)}
    if len(positive) < 200:
        return result
    log_q = np.log(positive.Q_norm.to_numpy())
    log_t = np.log(positive.duration_s.to_numpy())
    log_i = np.log(positive.I_norm.to_numpy())
    additive = fits.ols_hc3(log_i, np.column_stack([log_q, log_t]),
                            names=["logQ", "logT"])
    qc, tc = log_q - log_q.mean(), log_t - log_t.mean()
    centred = fits.ols_hc3(log_i, np.column_stack([qc, tc, qc * tc]),
                           names=["logQ", "logT", "logQ:logT"])
    result.update({
        "additive_delta": float(additive["params"][1]),
        "additive_delta_se": float(additive["se"][1]),
        "additive_gamma": float(additive["params"][2]),
        "additive_gamma_se": float(additive["se"][2]),
        "additive_r2": additive["r2"],
        "centred_delta": float(centred["params"][1]),
        "centred_delta_se": float(centred["se"][1]),
        "centred_gamma": float(centred["params"][2]),
        "centred_gamma_se": float(centred["se"][2]),
        "centred_interaction": float(centred["params"][3]),
        "centred_interaction_se": float(centred["se"][3]),
        "centred_r2": centred["r2"],
    })
    return result


def _measure_variant(orders, scales, quote_ts, quote_mid, method, params,
                     seed, shuffle_signs=False):
    """Build and measure every parent of one variant/seed across the regime."""
    frames, paths, assigned, offered = [], [], 0, 0
    by_day = {day: block.reset_index(drop=True)
              for day, block in orders.groupby("day", sort=True)}
    for scale in scales.itertuples(index=False):
        day = by_day.get(scale.day)
        if day is None or scale.sigma_range <= 0 or len(day) < 50:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(scale.day)]))
        if shuffle_signs:
            # Destroy the sign sequence but keep every timestamp and quantity,
            # then rebuild parents from scratch: this null therefore changes the
            # parent population as well as the sign vector, which is exactly
            # what proxy.ipynb cell 6 does and must be read as such.
            day = day.assign(sign=rng.permutation(day.sign.to_numpy()))
        parents = build_parents(day, method, rng, **params)
        flat = np.concatenate(parents) if parents else np.array([], dtype=int)
        if flat.size != np.unique(flat).size:
            raise AssertionError(f"{method}: a child was assigned to two parents")
        offered += len(day)
        assigned += flat.size
        frame, trajectories = _parents_for_day(day, parents, orders, scale)
        if frame.empty:
            continue
        frames.append(frame)
        paths.append(_trajectory_rows(day, trajectories, frame))
    if not frames:
        return pd.DataFrame(), np.empty((0, PHI_GRID.size)), {}
    coverage = {"input_orders": offered, "assigned_orders": assigned,
                "assigned_share": assigned / max(offered, 1)}
    stacked = [p for p in paths if len(p)]
    return (pd.concat(frames, ignore_index=True),
            np.vstack(stacked) if stacked else np.empty((0, PHI_GRID.size)),
            coverage)


def _analyse(frame, trajectories, quote_ts, quote_mid, hurst, max_age_ns):
    """Every proxy.ipynb test on one already-measured parent population."""
    out = {"curves": [], "duration": pd.DataFrame(), "trajectory": pd.DataFrame(),
           "decay": pd.DataFrame(), "summary": {}}
    edges = _size_edges(frame.Q_norm.to_numpy())
    if edges is None:
        return out
    out["size_edges"] = edges

    # --- cell 2: continuous broken power law, signed and positive-only -------
    fitted, single = {}, {}
    for selection, subset in (("all_signed", frame),
                              ("positive_only", frame.loc[frame.I_norm > 0])):
        curve = _binned(subset, "Q_norm", "I_norm", edges)
        if curve.empty:
            continue
        fitted[selection] = fits.fit_broken_power_law(curve.x, curve.y,
                                                      curve["sem_iid"])
        # A single power law on the SAME bins and the same weights.  The broken
        # form has a free breakpoint, so on a noisy curve its second branch can
        # end up describing a sparse tail rather than the impact regime; the
        # one-slope companion makes that visible instead of leaving delta_2 to
        # be read at face value.
        single[selection] = fits.wls_loglog(curve.x, curve.y, curve["sem_iid"])
        out["curves"].append(curve.assign(selection=selection))
    if "all_signed" not in fitted:
        return out
    primary = fitted["all_signed"]
    delta = primary["delta_2"]
    out["summary"] = {
        "n_parents": len(frame),
        "median_children": float(frame.n_child.median()),
        "median_duration_s": float(frame.duration_s.median()),
        "median_participation": float(frame.participation.median()),
        "positive_impact_share": float((frame.I_norm > 0).mean()),
        "mean_I_norm": float(frame.I_norm.mean()),
        "mean_impact_next_norm": float(frame.impact_next_norm.mean()),
        "x_break": primary["x_break"], "y_break": primary["y_break"],
        "delta_1": primary["delta_1"],
        "delta_2": delta, "delta_2_se": primary["delta_2_se"],
        "prefactor_Y": primary["prefactor_Y"], "r2_log": primary["r2_log"],
        "delta_single_power_law": single["all_signed"]["slope"],
        "delta_single_se": single["all_signed"]["se"],
        "r2_single_power_law": single["all_signed"]["r2"],
        "size_break_at_boundary": primary["break_at_boundary"],
        "size_bins_dropped_nonpositive": primary["n_dropped_nonpositive"],
        "delta_2_positive_only": fitted.get("positive_only", {}).get("delta_2", np.nan),
        "Y_positive_only": fitted.get("positive_only", {}).get("prefactor_Y", np.nan),
    }
    if not np.isfinite(delta):
        return out

    # --- cells 3, 4: duration dependence of the rescaled impact factor -------
    signal = frame.loc[(frame.Q_norm >= primary["x_break"])
                       & (frame.duration_s >= MIN_DURATION_S)].copy()
    if len(signal) >= 200:
        signal["Y_scaled"] = signal.I_norm / signal.Q_norm ** delta
        span = np.quantile(signal.duration_s, [0.0, 0.995])
        if span[1] > span[0] > 0:
            duration_edges = np.logspace(np.log10(span[0]), np.log10(span[1]),
                                         N_DURATION_BINS)
            curve = _binned(signal, "duration_s", "Y_scaled", duration_edges)
            volume = _binned(signal, "duration_s", "V_window_btc", duration_edges)
            if len(curve) >= 4:
                gamma = fits.wls_loglog(curve.x, curve.y, curve["sem_iid"])
                out["duration"] = curve.assign(kind="Y_scaled")
                out["summary"].update({
                    "gamma_empirical": gamma["slope"],
                    "gamma_se": gamma["se"], "gamma_r2": gamma["r2"],
                    "gamma_bins_dropped_nonpositive": gamma["n_dropped_nonpositive"],
                    "gamma_theory_sqrt_rule": 0.5 - delta,
                })
            if len(volume) >= 4:
                nu = fits.wls_loglog(volume.x, volume.y, volume["sem_iid"])
                out["duration"] = pd.concat(
                    [out["duration"], volume.assign(kind="V_window")],
                    ignore_index=True)
                out["summary"].update({"nu": nu["slope"], "nu_se": nu["se"],
                                       "nu_r2": nu["r2"]})
    hurst_value = hurst if np.isfinite(hurst) else np.nan
    nu_value = out["summary"].get("nu", np.nan)
    out["summary"]["hurst_diffusive"] = hurst_value
    out["summary"]["gamma_theory_hurst"] = hurst_value - delta
    out["summary"]["gamma_theory_hurst_nu"] = hurst_value - delta * nu_value
    out["summary"]["gamma_residual"] = (out["summary"].get("gamma_empirical", np.nan)
                                        - out["summary"]["gamma_theory_hurst_nu"])

    # --- cell 5: execution trajectory ---------------------------------------
    if len(trajectories):
        # ``trajectories`` is built for exactly the rows with enough children,
        # in frame order, so the two line up row for row.  Assert it rather
        # than trust it: a silent misalignment here would normalise every path
        # by the wrong parent's scale.
        eligible = (frame.n_child >= MIN_TRAJECTORY_CHILDREN).to_numpy()
        if eligible.sum() != len(trajectories):
            raise AssertionError("trajectory rows are not aligned with parents")
        scale = (frame.sigma_range.to_numpy()[eligible]
                 * frame.Q_norm.to_numpy()[eligible] ** delta)
        usable = np.isfinite(scale) & (scale > 0)
        if usable.any():
            normalised = trajectories[usable] / scale[usable][:, None]
            out["trajectory"] = pd.DataFrame({
                "phi": PHI_GRID,
                "normalised_impact": normalised.mean(axis=0),
                "sem_iid": normalised.std(axis=0, ddof=1) / np.sqrt(usable.sum()),
                "n": int(usable.sum())})
            half = np.interp(0.5, PHI_GRID, normalised.mean(axis=0))
            end = normalised.mean(axis=0)[-1]
            out["summary"]["trajectory_half_over_end"] = (
                float(half / end) if abs(end) > 1e-12 else np.nan)

    # --- cell 6: decay on the rescaled clock, plus the propagator fit --------
    decay = _decay_curve(frame, quote_ts, quote_mid, delta, max_age_ns)
    if len(decay):
        # Two fits, because they answer different questions and the notebook
        # only asks the first.  Fitting from the observed peak describes how
        # fast the response falls once it has turned over; fitting from the
        # pre-declared completion point z=1 always yields a number, including
        # when the response has not turned over at all inside z<=2.  A peak
        # sitting on the last grid point is the latter case, and is flagged
        # rather than reported as a decay.
        from_peak = fits.fit_propagator_decay(decay.z, decay.normalised_impact,
                                              decay["sem_iid"])
        from_completion = fits.fit_propagator_decay(decay.z, decay.normalised_impact,
                                                    decay["sem_iid"], z_peak=1.0)
        peak_z = from_peak["z_peak"]
        top = decay.normalised_impact.max()
        out["decay"] = decay
        out["summary"].update({
            "decay_beta": from_peak["beta"],
            "decay_beta_resolution": from_peak["beta_resolution"],
            "decay_I0": from_peak["I_0"], "decay_z_peak": peak_z,
            "decay_r2": from_peak["r2"], "decay_n": int(decay.n.iloc[0]),
            "decay_beta_from_completion": from_completion["beta"],
            "decay_r2_from_completion": from_completion["r2"],
            "decay_still_rising_at_z2": bool(peak_z >= Z_GRID[-3]),
            "decay_z2_over_peak": float(decay.normalised_impact.iloc[-1] / top)
            if top != 0 else np.nan,
        })

    # --- cells 12, 13: joint (Q, T) regressions ------------------------------
    out["summary"].update(_joint_regressions(frame))
    return out


def _panel_figure(ctx, regime, label, variant, seed, analysis, shuffled_curve,
                  frame, colour):
    """Nine panels on proxy.ipynb's axes, for one regime/variant/seed."""
    summary = analysis["summary"]
    delta = summary.get("delta_2", np.nan)
    fig, axes = pl.new_figure(3, 3, width=5.0, height=4.0)

    ax = axes[0, 0]
    negatives = 0
    for selection, style, colour_ in (("all_signed", "o", colour),
                                      ("positive_only", "s", "#9CA3AF")):
        curve = next((c for c in analysis["curves"] if c.selection.iloc[0] == selection),
                     None)
        if curve is None:
            continue
        # Keeping signed observations means some bin means come out negative,
        # and a log axis cannot show them.  Plot them at |y| with a hollow
        # downward marker instead of silently folding them into the positive
        # cloud, which would make a noisy small-size regime look like impact.
        up, down = curve.loc[curve.y > 0], curve.loc[curve.y <= 0]
        ax.errorbar(up.x, up.y, yerr=up["sem_iid"], fmt=style, ms=4, lw=1.1,
                    capsize=2, color=colour_,
                    label=f"{selection} (N={curve.n.sum():,})")
        if len(down) and selection == "all_signed":
            negatives = len(down)
            ax.errorbar(down.x, down.y.abs(), yerr=down["sem_iid"], fmt="v", ms=5,
                        lw=1.1, capsize=2, mfc="none", color="#C0392B",
                        label=f"negative bin mean, shown at |y| ({negatives})")
    if np.isfinite(summary.get("x_break", np.nan)):
        grid = np.logspace(np.log10(analysis["size_edges"][0]),
                           np.log10(analysis["size_edges"][-1]), 200)
        broken = np.where(
            grid <= summary["x_break"],
            summary["y_break"] * (grid / summary["x_break"]) ** summary["delta_1"],
            summary["y_break"] * (grid / summary["x_break"]) ** delta)
        ax.plot(grid, broken, lw=2.0, color="#2C6FBB",
                label=rf"broken power law ($R^2={summary['r2_log']:.3f}$)")
        ax.axvline(summary["x_break"], color="#E67E22", ls=":", lw=1.3)
        slope = summary.get("delta_single_power_law", np.nan)
        if np.isfinite(slope):
            anchor = summary["y_break"] / summary["x_break"] ** slope
            ax.plot(grid, anchor * grid ** slope, ls="--", lw=1.5, color="#4B5563",
                    label=rf"single power law $\delta={slope:.2f}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    pl.finish(ax, "Impact vs size", r"$Q / V_D$", r"$|I(Q)| / \sigma_D$", legend=True,
              note=(rf"$\delta_1$={summary.get('delta_1', float('nan')):.2f}, "
                    rf"$\delta_2$={delta:.2f}$\pm${summary.get('delta_2_se', float('nan')):.2f}, "
                    rf"$Y$={summary.get('prefactor_Y', float('nan')):.3f}; single-slope "
                    rf"$\delta$={summary.get('delta_single_power_law', float('nan')):.2f}. "
                    f"{summary.get('size_bins_dropped_nonpositive', 0)} bins had a "
                    "non-positive mean and could not enter the log fit. A large "
                    "gap between the two slopes means the breakpoint, not the "
                    "data, is carrying the fit."))
    if negatives:
        ax.set_ylabel(r"$I(Q) / \sigma_D$   (negatives at $|y|$)")

    ax = axes[0, 1]
    duration = analysis["duration"]
    scaled = duration.loc[duration.kind == "Y_scaled"] if len(duration) else duration
    if len(scaled):
        ax.errorbar(scaled.x, scaled.y, yerr=scaled["sem_iid"], fmt="o", ms=5,
                    lw=1.3, capsize=3, color=colour, label="empirical $Y(T)$")
        grid = np.logspace(np.log10(scaled.x.min()), np.log10(scaled.x.max()), 100)
        gamma = summary.get("gamma_empirical", np.nan)
        if np.isfinite(gamma):
            middle = scaled.iloc[len(scaled) // 2]
            for exponent, style, name in (
                    (gamma, dict(color="#2E7D32", lw=2.0),
                     rf"fitted $T^{{{gamma:.2f}}}$"),
                    (summary.get("gamma_theory_sqrt_rule", np.nan),
                     dict(color="#C0392B", ls="--", lw=1.7),
                     rf"$T^{{0.5-\delta_2}}$"),
                    (summary.get("gamma_theory_hurst_nu", np.nan),
                     dict(color="#7B4FA8", ls="-.", lw=1.7),
                     rf"$T^{{H-\delta_2\nu}}$")):
                if np.isfinite(exponent):
                    ax.plot(grid, middle.y / middle.x ** exponent * grid ** exponent,
                            label=name, **style)
        ax.set_xscale("log")
        ax.set_yscale("log")
        twin = ax.twinx()
        twin.hist(frame.duration_s, bins=np.logspace(
            np.log10(max(scaled.x.min(), 1e-3)), np.log10(scaled.x.max()), 30),
            color="#9CA3AF", alpha=0.16, density=True)
        twin.set_ylabel("parent density", fontsize=9, color="#6B7280")
        twin.grid(False)
    pl.finish(ax, "Duration dependence", "duration $T$ (seconds)",
              r"$Y = I/(\sigma_D (Q/V_D)^{\delta_2})$", legend=len(scaled) > 0,
              note="Bin means of signed impact; no positive-only pre-filter. "
                   "The theory lines are model consequences, not pass criteria.")

    ax = axes[0, 2]
    path = analysis["trajectory"]
    if len(path):
        ax.errorbar(path.phi, path.normalised_impact, yerr=path["sem_iid"], fmt="o",
                    ms=4.5, lw=1.2, capsize=2.5, color=colour, label="empirical")
        terminal = path.normalised_impact.iloc[-1]
        grid = np.linspace(PHI_GRID[0], 1.0, 100)
        ax.plot(grid, terminal * grid ** delta, "--", lw=1.7, color="#C0392B",
                label=rf"$Y\phi^{{\delta_2}}$ ($\delta_2$={delta:.2f})")
        ax.plot(grid, terminal * np.sqrt(grid), "-.", lw=1.3, color="#2E7D32",
                label=r"$Y\sqrt{\phi}$")
        ax.plot(grid, terminal * grid, ":", lw=1.5, color="#4B5563",
                label=r"linear $Y\phi$")
        ax.set_xscale("log")
        ax.set_yscale("log")
    pl.finish(ax, "Execution trajectory", r"volume progress $\phi=\sum q_i/Q$",
              r"$\mathcal{I}(\phi Q)/(\sigma_D (Q/V_D)^{\delta_2})$",
              legend=len(path) > 0,
              note="Each parent contributes one interpolated path and one vote, "
                   "rather than pooling all child points.")

    ax = axes[1, 0]
    decay = analysis["decay"]
    if len(decay):
        ax.errorbar(decay.z, decay.normalised_impact, yerr=decay["sem_iid"], fmt="o",
                    ms=5, lw=1.3, capsize=3, color=colour,
                    label=f"empirical (N={decay.n.iloc[0]:,})")
        beta = summary.get("decay_beta", np.nan)
        peak = summary.get("decay_z_peak", np.nan)
        if np.isfinite(beta) and np.isfinite(peak):
            grid = np.linspace(peak, Z_GRID[-1], 150)
            shifted = 1.0 + (grid - peak)
            kernel = shifted ** (1 - beta) - np.where(
                shifted > 1, np.maximum(shifted - 1, 0) ** (1 - beta), 0.0)
            ax.plot(grid, summary["decay_I0"] * kernel, lw=2.0, color="#2E7D32",
                    label=rf"from peak $z={peak:.2f}$: $\beta={beta:.2f}$")
            ax.axvline(peak, color="#6B7280", ls=":", lw=1.2)
        rising = summary.get("decay_still_rising_at_z2", False)
        note = ("Same parents at every z, so the curve cannot bend because its "
                "sample changed. The peak is picked from this same sample, so "
                "the fitted beta does not price that selection in.")
        if rising:
            note = ("The response is still rising at z=2, so there is no decay "
                    "to fit from a peak here: only the fit anchored at the "
                    "pre-declared completion point z=1 is defined "
                    f"(beta={summary.get('decay_beta_from_completion', float('nan')):.2f}). "
                    + note)
    else:
        note = ""
    pl.finish(ax, "Post-completion decay", r"rescaled time $z=(t-t_{start})/T$",
              r"$\mathcal{I}(Q,z)/(\sigma_D (Q/V_D)^{\delta_2})$",
              legend=len(decay) > 0, note=note)

    ax = axes[1, 1]
    real = next((c for c in analysis["curves"] if c.selection.iloc[0] == "all_signed"),
                None)
    if real is not None:
        ax.errorbar(real.x, real.y, yerr=real["sem_iid"], fmt="o", ms=4.5, lw=1.2,
                    capsize=2.5, color="#E67E22", label="original signs")
    if shuffled_curve is not None and len(shuffled_curve):
        ax.errorbar(shuffled_curve.x, shuffled_curve.y, yerr=shuffled_curve["sem_iid"],
                    fmt="s", ms=4.5, lw=1.2, capsize=2.5, color="#2E7D32",
                    label="shuffled signs")
    ax.axhline(0.0, **pl.REFERENCE_KW)
    if np.isfinite(summary.get("x_break", np.nan)):
        ax.axvline(summary["x_break"], color="#E67E22", ls=":", lw=1.2)
    ax.set_xscale("log")
    pl.finish(ax, "Shuffled-sign null", r"$Q / V_D$", r"$I / \sigma_D$", legend=True,
              note="Signs are permuted within each day and parents rebuilt, so the "
                   "null changes the parent population too - it tests the whole "
                   "pipeline, not the sign-return link alone.")

    ax = axes[1, 2]
    volume = duration.loc[duration.kind == "V_window"] if len(duration) else duration
    if len(volume):
        ax.errorbar(volume.x, volume.y, yerr=volume["sem_iid"], fmt="o", ms=5, lw=1.3,
                    capsize=3, color="#7B4FA8", label=r"$V_{window}(T)$")
        grid = np.logspace(np.log10(volume.x.min()), np.log10(volume.x.max()), 100)
        nu = summary.get("nu", np.nan)
        middle = volume.iloc[len(volume) // 2]
        if np.isfinite(nu):
            ax.plot(grid, middle.y / middle.x ** nu * grid ** nu, lw=2.0,
                    color="#7B4FA8", label=rf"$T^{{{nu:.2f}}}$")
        ax.plot(grid, middle.y / middle.x * grid, ":", lw=1.5, color="#4B5563",
                label=r"linear clock ($\nu=1$)")
        ax.set_xscale("log")
        ax.set_yscale("log")
    pl.finish(ax, "Local volume clock", "duration $T$ (seconds)",
              "market volume in the window (BTC)", legend=len(volume) > 0,
              note=r"$\nu\neq1$ means the volume clock is not linear in wall time, "
                   r"which changes the closure to $\gamma=H-\delta_2\nu$.")

    ax = axes[2, 0]
    sample = frame.loc[(frame.Q_norm > 0) & (frame.duration_s > 0)]
    if len(sample):
        step = max(1, len(sample) // 20000)
        ax.scatter(sample.Q_norm.iloc[::step], sample.duration_s.iloc[::step], s=5,
                   alpha=0.12, color="#8C564B", linewidths=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
    pl.finish(ax, "Size-duration entanglement", r"$Q/V_D$", "duration $T$ (s)",
              note=rf"$\rho(\log Q,\log T)$="
                   rf"{summary.get('rho_logQ_logT', float('nan')):.2f}. A strong "
                   "correlation means the two-step fit cannot separate delta from "
                   "gamma cleanly, whichever construction produced it.")

    ax = axes[2, 1]
    names, values, errors = [], [], []
    for key, label_ in (("delta_2", r"$\delta_2$ two-step"),
                        ("additive_delta", r"$\delta$ joint"),
                        ("centred_delta", r"$\delta$ centred"),
                        ("gamma_empirical", r"$\gamma$ two-step"),
                        ("additive_gamma", r"$\gamma$ joint"),
                        ("centred_gamma", r"$\gamma$ centred"),
                        ("centred_interaction", "interaction")):
        if np.isfinite(summary.get(key, np.nan)):
            names.append(label_)
            values.append(summary[key])
            errors.append(summary.get(key.replace("delta", "delta_se")
                                      .replace("gamma", "gamma_se")
                                      .replace("interaction", "interaction_se"),
                                      np.nan))
    if names:
        ax.bar(range(len(names)), values, yerr=np.nan_to_num(errors, nan=0.0),
               color=["#2C6FBB"] * 3 + ["#2E7D32"] * 3 + ["#C0392B"], capsize=3)
        ax.set_xticks(range(len(names)), names, rotation=25, ha="right")
    ax.axhline(0.0, **pl.REFERENCE_KW)
    pl.finish(ax, "Two-step vs joint estimates", "", "exponent",
              note=f"Joint and centred regressions run on log I, so they use the "
                   f"positive subsample only ({summary.get('positive_share', float('nan')):.0%} "
                   "of parents). The two-step figure does not.")

    ax = axes[2, 2]
    closure = [("empirical $\\gamma$", summary.get("gamma_empirical", np.nan)),
               ("$0.5-\\delta_2$", summary.get("gamma_theory_sqrt_rule", np.nan)),
               ("$H-\\delta_2$", summary.get("gamma_theory_hurst", np.nan)),
               ("$H-\\delta_2\\nu$", summary.get("gamma_theory_hurst_nu", np.nan))]
    keep = [(n, v) for n, v in closure if np.isfinite(v)]
    if keep:
        ax.barh(range(len(keep)), [v for _, v in keep], color="#4B5563")
        ax.set_yticks(range(len(keep)), [n for n, _ in keep])
        ax.invert_yaxis()
    ax.axvline(0.0, **pl.REFERENCE_KW)
    pl.finish(ax, "Duration-scaling closure", "exponent", "",
              note=rf"$H$={summary.get('hurst_diffusive', float('nan')):.2f} "
                   rf"(fact 18, $\tau\geq2$s), $\nu$={summary.get('nu', float('nan')):.2f}. "
                   "These are model relations to inspect, not thresholds to hit.")

    fig.suptitle(
        f"Facts 1-4 proxy battery - {variant} (seed {seed}) | {label}\n"
        "Signed impacts retained throughout: no I_norm > 0 pre-filter",
        x=0.02, ha="left", fontsize=12.5, fontweight="semibold")
    pl.layout(fig, rect=(0, 0, 1, 0.955))
    ctx.out.save_figure(fig, FACT, f"battery_{variant}_seed{seed}", regime)


def run(ctx) -> dict:
    pl.apply_style()
    cfg = ctx.cfg
    pl.apply_style()
    max_age_ns = int(MAX_QUOTE_AGE_S * 1e9)
    metrics: list[dict] = []
    summaries, all_curves, all_duration, all_paths, all_decay, coverages = \
        [], [], [], [], [], []

    for name, rd in ctx.each():
        orders, _fills, quotes, _audit = load_analysis_rows(
            Path(cfg.CACHE_DIR) / name, cfg.REGIMES[name])
        scales = _daily_scales(orders)
        quote_ts = quotes.exch_ns.to_numpy(np.int64)
        quote_mid = (quotes.bid_tick.to_numpy() + quotes.ask_tick.to_numpy()) \
            * cfg.TICK_SIZE / 2
        hurst = f18.measure(rd)["diffusive"].get("slope", np.nan)

        for variant, method, params, seeds in _variants(cfg):
            for seed in seeds:
                frame, trajectories, coverage = _measure_variant(
                    orders, scales, quote_ts, quote_mid, method, params, seed)
                if frame.empty:
                    continue
                tags = dict(regime=name, variant=variant, method=method, seed=seed)
                coverages.append({**tags, **coverage})
                analysis = _analyse(frame, trajectories, quote_ts, quote_mid,
                                    hurst, max_age_ns)
                if not analysis["summary"]:
                    continue

                shuffled, _, _ = _measure_variant(
                    orders, scales, quote_ts, quote_mid, method, params, seed,
                    shuffle_signs=True)
                shuffled_curve = (_binned(shuffled, "Q_norm", "I_norm",
                                          analysis["size_edges"])
                                  if not shuffled.empty else pd.DataFrame())
                analysis["summary"]["shuffled_mean_I_norm"] = (
                    float(shuffled.I_norm.mean()) if not shuffled.empty else np.nan)

                summaries.append({**tags, **analysis["summary"]})
                for curve in analysis["curves"]:
                    all_curves.append(curve.assign(**tags))
                if len(shuffled_curve):
                    all_curves.append(shuffled_curve.assign(**tags,
                                                            selection="shuffled_signs"))
                for key, sink in (("duration", all_duration),
                                  ("trajectory", all_paths), ("decay", all_decay)):
                    if len(analysis[key]):
                        sink.append(analysis[key].assign(**tags))

                _panel_figure(ctx, name, rd.label, variant, seed, analysis,
                              shuffled_curve, frame, rd.color)
                # Every parent, with its children's endpoints and both impact
                # conventions, so any curve above can be re-derived or re-cut.
                destination = Path(cfg.RESULTS_DIR) / name
                destination.mkdir(parents=True, exist_ok=True)
                frame.assign(**tags).to_parquet(
                    destination / f"{FACT}_parents_{variant}_seed{seed}__{name}.parquet",
                    index=False)
                print(f"    {name:6} {variant:19} seed={seed}: "
                      f"n={len(frame):,}, delta_2={analysis['summary']['delta_2']:.3f}",
                      flush=True)

    summary = pd.DataFrame(summaries)
    if summary.empty:
        return {"fact": FACT, "id": FACT_ID, "priority": PRIORITY, "title": TITLE,
                "metrics": [], "literature": "no parents measured"}

    ctx.out.save_table(summary, FACT, "proxy_battery_summary", "comparison",
                       caption="Every proxy.ipynb estimate, per regime / proxy / seed")
    ctx.out.save_table(pd.DataFrame(coverages), FACT, "proxy_coverage", "comparison",
                       caption="Share of aggressive orders each construction assigns")
    for label, frames in (("size_curves", all_curves), ("duration_curves", all_duration),
                          ("trajectories", all_paths), ("decay_curves", all_decay)):
        if frames:
            ctx.out.save_table(pd.concat(frames, ignore_index=True), FACT, label,
                               "comparison", caption=f"{label} for all constructions")

    # Cross-construction comparison, one figure per regime.
    curves = pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()
    paths = pd.concat(all_paths, ignore_index=True) if all_paths else pd.DataFrame()
    decays = pd.concat(all_decay, ignore_index=True) if all_decay else pd.DataFrame()
    palette = {"current_run": "#4B5563", "random_n6": "#2C6FBB",
               "maitrier_threshold": "#E67E22"}
    for name, rd in ctx.each():
        if name not in set(summary.regime):
            continue
        fig, axes = pl.new_figure(2, 2, width=5.6, height=4.3)
        for variant, colour in palette.items():
            pick = dict(regime=name, variant=variant)
            sub = curves.loc[(curves.regime == name) & (curves.variant == variant)
                             & (curves.selection == "all_signed")]
            if len(sub):
                mean = sub.groupby("bin").agg(x=("x", "mean"), y=("y", "mean"))
                up = mean.loc[mean.y > 0]
                axes[0, 0].plot(up.x, up.y, "o-", ms=3.5, lw=1.6, color=colour,
                                label=variant)
                down = mean.loc[mean.y <= 0]
                if len(down):
                    axes[0, 0].plot(down.x, down.y.abs(), "v", ms=5, mfc="none",
                                    color=colour)
            sub = paths.loc[(paths.regime == name) & (paths.variant == variant)]
            if len(sub):
                mean = sub.groupby("phi").normalised_impact.mean()
                axes[0, 1].plot(mean.index, mean.values, lw=1.8, color=colour,
                                label=variant)
            sub = decays.loc[(decays.regime == name) & (decays.variant == variant)]
            if len(sub):
                mean = sub.groupby("z").normalised_impact.mean()
                axes[1, 0].plot(mean.index, mean.values, lw=1.8, color=colour,
                                label=variant)
        block = summary.loc[summary.regime == name]
        stats = block.groupby("variant").agg(
            delta_2=("delta_2", "mean"), delta_2_sd=("delta_2", "std"),
            single=("delta_single_power_law", "mean"),
            single_sd=("delta_single_power_law", "std"),
            positive_only=("delta_2_positive_only", "mean"))
        positions = np.arange(len(stats))
        for offset, column, spread, colour_, label_ in (
                (-0.26, "delta_2", "delta_2_sd", "#2C6FBB",
                 r"broken $\delta_2$, all signed"),
                (0.0, "single", "single_sd", "#2E7D32",
                 r"single-slope $\delta$, all signed"),
                (0.26, "positive_only", None, "#9CA3AF",
                 "positive-only (notebook filter)")):
            axes[1, 1].bar(positions + offset, stats[column], width=0.25,
                           yerr=None if spread is None
                           else np.nan_to_num(stats[spread], nan=0.0),
                           capsize=3, color=colour_, label=label_)
        axes[1, 1].axhline(0.5, **pl.LITERATURE_KW)
        axes[1, 1].set_xticks(positions, stats.index, rotation=12)

        axes[0, 0].set_xscale("log")
        axes[0, 0].set_yscale("log")
        pl.finish(axes[0, 0], "Impact vs size", r"$Q / V_D$", r"$I / \sigma_D$",
                  legend=True,
                  note="Seed-averaged bin means. Hollow triangles are negative "
                       "means drawn at |y|, which a log axis cannot otherwise "
                       "show.")
        axes[0, 1].set_xscale("log")
        axes[0, 1].set_yscale("log")
        pl.finish(axes[0, 1], "Execution trajectory", r"$\phi$",
                  "normalised impact", legend=True)
        pl.finish(axes[1, 0], "Decay on the rescaled clock", r"$z=(t-t_{start})/T$",
                  "normalised impact", legend=True)
        pl.finish(axes[1, 1], "Size exponent by construction and selection", "",
                  r"$\delta_2$", legend=True,
                  note="Error bars are the spread across seeds: construction "
                       "sensitivity on one market sample, not a confidence "
                       "interval. Where the broken and single-slope bars "
                       "disagree, the breakpoint has landed in a sparse tail "
                       "and only the single-slope figure is readable. The "
                       "dotted line marks the square-root value; it is a "
                       "reference, not a target.")
        fig.suptitle(f"Facts 1-4 - three proxy constructions | {rd.label}", x=0.02,
                     ha="left", fontsize=12.5, fontweight="semibold")
        pl.layout(fig, rect=(0, 0, 1, 0.94))
        ctx.out.save_figure(fig, FACT, "proxy_comparison", name)

    for (regime, variant), block in summary.groupby(["regime", "variant"]):
        metrics += [
            {"metric": f"{variant}: size exponent delta_2 (all signed)",
             "regime": regime, "value": float(block.delta_2.mean()),
             "detail": f"seed spread {block.delta_2.std():.3f}; "
                       f"positive-only filter gives "
                       f"{block.delta_2_positive_only.mean():.3f}"},
            {"metric": f"{variant}: single-slope size exponent (all signed)",
             "regime": regime,
             "value": float(block.delta_single_power_law.mean()),
             "detail": "one power law on the same bins and weights; stable "
                       "companion to the free-breakpoint fit"},
            {"metric": f"{variant}: duration exponent gamma",
             "regime": regime, "value": float(block.get("gamma_empirical",
                                                        pd.Series([np.nan])).mean()),
             "detail": f"H-delta_2*nu = "
                       f"{block.get('gamma_theory_hurst_nu', pd.Series([np.nan])).mean():.3f}"},
            {"metric": f"{variant}: decay exponent beta",
             "regime": regime, "value": float(block.get("decay_beta",
                                                        pd.Series([np.nan])).mean()),
             "detail": "propagator fitted from the observed peak on z"},
        ]

    return {
        "fact": FACT, "id": FACT_ID, "priority": PRIORITY, "title": TITLE,
        "metrics": metrics,
        "literature": "proxy.ipynb (DOGE, 2026-08-25) reports delta_2=0.247, "
                      "gamma=0.194, centred joint gamma=0.211, beta=0.114 - under "
                      "an I_norm>0 pre-filter that is deliberately not applied "
                      "here. Maitrier et al. arXiv:2503.18199 / arXiv:2509.05065 "
                      "supply the two published mapping functions.",
    }
