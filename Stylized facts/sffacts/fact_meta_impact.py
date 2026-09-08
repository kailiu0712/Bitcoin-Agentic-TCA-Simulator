"""
Stylized facts 1-3 (P0-V), measured through a clearly-labelled **proxy**.

What the literature claims
    1. Peak metaorder impact is concave in size, commonly
       ``I_peak ~ Y * sigma * (Q/V)^delta`` with delta near 1/2.  Donier & Bonart
       reconstruct >1M BTC/USD metaorders and report square-root impact over
       about four decades of size; Toth et al. supply the latent-liquidity
       theory.  Zarinelli et al. caution that the square root is a good
       approximation only over a limited range and that a logarithmic form fits
       wider.
    2. Impact builds **concavely along the execution trajectory**, not in
       proportion to executed quantity.
    3. Impact **relaxes** after completion; the permanent fraction is not
       universal (Zarinelli et al. ~2/3 of peak in US equities; Bucci et al.
       continued slow decay; Donier-Bonart find the uninformed component of
       Bitcoin impact can decay almost completely).

Why this is a proxy, and what that costs
    True metaorder identification needs parent-order labels, which no public
    feed carries.  What public data *does* contain is runs of same-signed
    aggressive orders that arrive close together - the observable footprint of
    order splitting.  This module builds "pseudo-metaorders" from those runs and
    measures the same three quantities on them.

    The honest limitations, which the outputs state rather than hide:
      * a run may merge several genuine parents that happened to trade together,
        or split one parent that paused longer than the gap tolerance;
      * because a run is defined by same-signed flow, its **participation rate
        is mechanically high** (contra-flow is admitted only up to
        ``OPPOSITE_TOLERANCE``), so the participation axis of the fact-4 impact
        surface is narrow.  Duration and size are measured properly; the
        achieved participation range is reported so the narrowness is visible.

    Per the source document's methodological caveat, no exponent is assumed:
    both a power law and a logarithmic form are fitted and compared by R-squared,
    and the estimated ``delta`` is reported with its standard error rather than
    being pinned at 0.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfcore import plotting as pl
from sfcore import stats as st

FACT = "fact01_03"
FACT_ID = 1
PRIORITY = "P0-V (proxy)"
TITLE = "Metaorder impact proxy: concavity, trajectory and decay"

#: Contra-flow volume tolerated inside a run, as a fraction of same-signed
#: volume accumulated so far.  Above this the run is considered over.
OPPOSITE_TOLERANCE = 0.5
N_SIZE_BINS = 12
N_TRAJECTORY_BINS = 10

#: Minimum children for a run to enter the trajectory analysis.  Two-child runs
#: dominate the population but carry no shape information.
TRAJECTORY_MIN_CHILDREN = 5
DECAY_EVENT_HORIZONS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)
DECAY_CLOCK_HORIZONS_S = (1, 5, 15, 30, 60, 180, 600, 1800)

#: Settle lag applied when reading the mid at completion.  The book updates
#: caused by the last fill carry essentially the same exchange timestamp as the
#: fill itself, so peak impact is read a beat later to be sure the sweep has
#: been absorbed.
POST_COMPLETION_LAG_MS = 100.0

#: Random-time control draws per pseudo-metaorder.  Over a directional week the
#: unconditional drift alone moves the mid, so a raw "impact after 10 minutes"
#: is not the residual impact of the metaorder.  The control estimates
#: E[sign * return] from randomly chosen times using the same sign vector, and
#: is subtracted to give the drift-adjusted decay.
CONTROL_DRAWS = 8
CONTROL_SEED = 20260718


def _build_runs(trades: pd.DataFrame, gap_s: float, min_children: int) -> pd.DataFrame:
    """Group aggressive orders into pseudo-metaorders (runs of same-signed flow)."""
    sign = trades["sign"].to_numpy(np.float64)
    qty = trades["qty"].to_numpy(np.float64)
    ts = trades["exch_ns"].to_numpy(np.int64)
    n = len(trades)
    gap_ns = int(gap_s * 1e9)

    starts, ends, signs, qs, q_opps, n_children = [], [], [], [], [], []
    i = 0
    while i < n:
        s = sign[i]
        q_same = qty[i]
        q_opp = 0.0
        children = 1
        j = i + 1
        # ``0 <=`` matters: the exchange clock steps backwards on ~0.02% of
        # consecutive aggressive orders, and a backstep of, say, -269 s would
        # otherwise satisfy the gap test and chain two orders that are minutes
        # apart in real time into one "parent" (with a negative duration).
        while j < n and 0 <= (ts[j] - ts[j - 1]) <= gap_ns:
            if sign[j] == s:
                q_same += qty[j]
                children += 1
            else:
                if q_opp + qty[j] > OPPOSITE_TOLERANCE * q_same:
                    break
                q_opp += qty[j]
            j += 1
        if children >= min_children:
            starts.append(i)
            ends.append(j - 1)
            signs.append(s)
            qs.append(q_same)
            q_opps.append(q_opp)
            n_children.append(children)
        i = j if j > i else i + 1

    return pd.DataFrame({
        "start_idx": starts, "end_idx": ends, "sign": signs,
        "qty_btc": qs, "contra_qty_btc": q_opps, "n_children": n_children,
    })


def _annotate(runs: pd.DataFrame, rd, adv_btc: float, sigma_daily_bps: float) -> pd.DataFrame:
    """Attach start/end mids, impact, duration and participation to each run."""
    trades = rd.trades
    ts = trades["exch_ns"].to_numpy(np.int64)
    mid = trades["mid"].to_numpy(np.float64)
    q_exch = rd.q_exch
    q_mid = rd.q_mid

    s_idx = runs.start_idx.to_numpy()
    e_idx = runs.end_idx.to_numpy()
    m0 = mid[s_idx]

    # Mid at completion, read POST_COMPLETION_LAG_MS after the last child so
    # the book updates caused by the sweep are certainly included.
    end_pos = np.searchsorted(q_exch, ts[e_idx] + int(POST_COMPLETION_LAG_MS * 1e6),
                              side="right") - 1
    end_pos = np.clip(end_pos, 0, q_mid.size - 1)
    m_end = q_mid[end_pos]

    duration_s = (ts[e_idx] - ts[s_idx]) / 1e9
    runs = runs.copy()
    runs["t_start_ns"] = ts[s_idx]
    runs["t_end_ns"] = ts[e_idx]
    runs["end_quote_pos"] = end_pos
    runs["mid_start"] = m0
    runs["mid_end"] = m_end
    runs["duration_s"] = duration_s
    runs["impact_bps"] = runs.sign * (m_end - m0) / m0 * 1e4
    runs["q_over_adv"] = runs.qty_btc / adv_btc
    runs["impact_over_sigma"] = runs.impact_bps / sigma_daily_bps
    total = runs.qty_btc + runs.contra_qty_btc
    runs["participation"] = np.where(total > 0, runs.qty_btc / total, np.nan)
    runs["rate_btc_per_s"] = np.where(duration_s > 0, runs.qty_btc / duration_s, np.nan)
    return runs


def _regime_scales(rd) -> tuple[float, float]:
    """Average daily volume (BTC) and daily volatility (bps) for the regime."""
    hours = rd.hours if np.isfinite(rd.hours) and rd.hours > 0 else 24.0
    days = hours / 24.0
    adv = float(rd.trades["qty"].sum() / days)

    # Daily volatility from 60 s mid bars.
    recv = rd.q_recv
    mid = rd.q_mid
    step = int(60 * 1e9)
    bucket = (recv - recv[0]) // step
    n_bars = int(bucket[-1]) + 1
    last = np.full(n_bars, np.nan)
    last[bucket] = mid
    idx = np.where(np.isfinite(last), np.arange(n_bars), 0)
    np.maximum.accumulate(idx, out=idx)
    bars = last[idx]
    ret = np.diff(np.log(bars))
    ret = ret[np.isfinite(ret)]
    sigma_daily = float(np.std(ret) * np.sqrt(1440.0) * 1e4) if ret.size > 10 else np.nan
    return adv, sigma_daily


def _trajectory(runs: pd.DataFrame, rd) -> pd.DataFrame:
    """Mean impact against the fraction of the parent executed so far.

    Each qualifying run is interpolated onto a common grid of executed
    fractions before averaging, so every point on the curve is built from the
    *same* set of parents.  Pooling raw (fraction, impact) points instead would
    make the curve a composition artefact: two-child runs contribute only near
    the endpoints, and their impact is by construction that of a single child,
    which drags the last bin down and manufactures a spurious late reversal.

    Only runs with at least ``TRAJECTORY_MIN_CHILDREN`` children are used,
    since a shape cannot be read off two points.
    """
    trades = rd.trades
    qty = trades["qty"].to_numpy(np.float64)
    mid = trades["mid"].to_numpy(np.float64)
    all_sign = trades["sign"].to_numpy(np.float64)

    grid = (np.arange(N_TRAJECTORY_BINS) + 0.5) / N_TRAJECTORY_BINS
    eligible = runs[runs.n_children >= TRAJECTORY_MIN_CHILDREN]
    paths = []
    for start, end, sign, total, peak in zip(eligible.start_idx, eligible.end_idx,
                                             eligible.sign, eligible.qty_btc,
                                             eligible.impact_bps):
        if total <= 0 or end <= start:
            continue
        idx = np.arange(start, end + 1)
        same = all_sign[idx] == sign
        cum = np.cumsum(np.where(same, qty[idx], 0.0)) / total
        imp = sign * (mid[idx] - mid[start]) / mid[start] * 1e4
        # Anchor at (0, 0) before any child trades, and at (1, peak) once the
        # parent is complete - the last child's own mid is still pre-trade.
        f = np.concatenate([[0.0], cum, [1.0]])
        y = np.concatenate([[0.0], imp, [peak]])
        order = np.argsort(f)
        paths.append(np.interp(grid, f[order], y[order]))
    if not paths:
        return pd.DataFrame()
    matrix = np.vstack(paths)
    return pd.DataFrame({
        "executed_fraction": grid,
        "mean_impact_bps": matrix.mean(axis=0),
        "sem_bps": matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0]),
        "n": matrix.shape[0],
    })


def _decay(runs: pd.DataFrame, rd) -> pd.DataFrame:
    """Impact after completion, in event time and clock time, relative to peak."""
    trades = rd.trades
    mid_t = trades["mid"].to_numpy(np.float64)
    n_t = mid_t.size
    q_exch = rd.q_exch
    q_mid = rd.q_mid

    sign = runs.sign.to_numpy(np.float64)
    m0 = runs.mid_start.to_numpy()
    peak = runs.impact_bps.to_numpy()
    e_idx = runs.end_idx.to_numpy()
    t_end = runs.t_end_ns.to_numpy()

    # Random-time control: the same signs, but starting from arbitrary moments.
    # Over a directional week this measures the drift a metaorder would have
    # "earned" by doing nothing, which has to be removed before any residual is
    # called permanent impact.
    rng = np.random.default_rng(CONTROL_SEED)
    ctrl_pos = rng.integers(0, max(q_mid.size - 1, 1), size=(CONTROL_DRAWS, sign.size))
    ctrl_sign = np.tile(sign, (CONTROL_DRAWS, 1))

    rows = []
    for h in DECAY_EVENT_HORIZONS:
        j = e_idx + h
        ok = j < n_t
        if ok.sum() < 30:
            continue
        imp = sign[ok] * (mid_t[j[ok]] - m0[ok]) / m0[ok] * 1e4
        rows.append({"clock": "event", "horizon": float(h),
                     "mean_impact_bps": float(np.nanmean(imp)),
                     "control_bps": np.nan,
                     "fraction_of_peak": float(np.nanmean(imp) / np.nanmean(peak[ok])),
                     "adjusted_fraction_of_peak": np.nan,
                     "n": int(ok.sum())})
    for seconds in DECAY_CLOCK_HORIZONS_S:
        step_ns = int(seconds * 1e9)
        pos = np.searchsorted(q_exch, t_end + step_ns, side="right") - 1
        ok = (pos >= 0) & (pos < q_mid.size)
        if ok.sum() < 30:
            continue
        imp = sign[ok] * (q_mid[pos[ok]] - m0[ok]) / m0[ok] * 1e4

        cpos_end = np.searchsorted(q_exch, q_exch[ctrl_pos] + step_ns, side="right") - 1
        cpos_end = np.clip(cpos_end, 0, q_mid.size - 1)
        cm0 = q_mid[ctrl_pos]
        control = float(np.nanmean(ctrl_sign * (q_mid[cpos_end] - cm0) / cm0 * 1e4))

        mean_imp = float(np.nanmean(imp))
        peak_mean = float(np.nanmean(peak[ok]))
        rows.append({"clock": "seconds", "horizon": float(seconds),
                     "mean_impact_bps": mean_imp,
                     "control_bps": control,
                     "fraction_of_peak": mean_imp / peak_mean,
                     "adjusted_fraction_of_peak": (mean_imp - control) / peak_mean,
                     "n": int(ok.sum())})
    return pd.DataFrame(rows)


def run(ctx) -> dict:
    pl.apply_style()
    cfg = ctx.cfg
    metrics: list[dict] = []
    size_curves, trajectories, decays, fits = {}, {}, {}, {}

    for name, rd in ctx.each():
        adv, sigma = _regime_scales(rd)
        raw = _build_runs(rd.trades, cfg.PSEUDO_METAORDER_GAP_S,
                          cfg.PSEUDO_METAORDER_MIN_CHILDREN)
        if raw.empty:
            continue
        runs = _annotate(raw, rd, adv, sigma)

        # --- size dependence: I/sigma against Q/V ------------------------
        bins = st.quantile_bins(runs.q_over_adv.to_numpy(), N_SIZE_BINS)
        mean_x, counts = st.group_mean(runs.q_over_adv.to_numpy(), bins, N_SIZE_BINS)
        mean_y, _ = st.group_mean(runs.impact_over_sigma.to_numpy(), bins, N_SIZE_BINS)
        sem_y = st.group_sem(runs.impact_over_sigma.to_numpy(), bins, N_SIZE_BINS)
        mean_dur, _ = st.group_mean(runs.duration_s.to_numpy(), bins, N_SIZE_BINS)
        curve = pd.DataFrame({"bin": np.arange(N_SIZE_BINS), "mean_q_over_adv": mean_x,
                              "mean_impact_over_sigma": mean_y, "sem_impact": sem_y,
                              "mean_duration_s": mean_dur, "n": counts})
        size_curves[name] = curve

        power = st.fit_loglog(curve.mean_q_over_adv.to_numpy(),
                              curve.mean_impact_over_sigma.to_numpy())
        logfit = st.fit_log_model(curve.mean_q_over_adv.to_numpy(),
                                  curve.mean_impact_over_sigma.to_numpy())
        fits[name] = (power, logfit)

        traj = _trajectory(runs, rd)
        dec = _decay(runs, rd)
        trajectories[name], decays[name] = traj, dec

        ctx.out.save_table(curve, FACT, "impact_vs_size", name,
                           caption=f"Pseudo-metaorder impact vs relative size - {rd.label}")
        if len(traj):
            ctx.out.save_table(traj, FACT, "impact_trajectory", name,
                               caption=f"Impact along the execution trajectory - {rd.label}")
        if len(dec):
            ctx.out.save_table(dec, FACT, "impact_decay", name,
                               caption=f"Post-completion impact decay - {rd.label}")
        ctx.out.save_table(
            runs[["n_children", "qty_btc", "contra_qty_btc", "duration_s",
                  "participation", "rate_btc_per_s", "q_over_adv", "impact_bps"]]
            .describe().reset_index().rename(columns={"index": "statistic"}),
            FACT, "pseudo_metaorder_population", name,
            caption=f"Pseudo-metaorder population summary - {rd.label}")

        # --- figure ------------------------------------------------------
        fig, axes = pl.new_figure(2, 3, width=5.0, height=3.9)
        ax = axes[0][0]
        ax.errorbar(curve.mean_q_over_adv, curve.mean_impact_over_sigma, yerr=curve.sem_impact,
                    fmt="o", ms=5, lw=1.5, capsize=2, label="measured")
        xs = np.geomspace(curve.mean_q_over_adv.min(), curve.mean_q_over_adv.max(), 60)
        if np.isfinite(power["exponent"]):
            ax.plot(xs, power["prefactor"] * xs ** power["exponent"], lw=1.8,
                    color="#B45309",
                    label=f"power law: delta={power['exponent']:.3f} "
                          f"+/-{power['se']:.3f} (R2={power['r2']:.2f})")
        ref_y = curve.mean_impact_over_sigma.median()
        ref_x = curve.mean_q_over_adv.median()
        ax.plot(xs, ref_y * np.sqrt(xs / ref_x), **pl.LITERATURE_KW)
        ax.set_xscale("log")
        ax.set_yscale("log")
        pl.finish(ax, "Peak impact vs relative size", "Q / ADV",
                  "impact / daily sigma", legend=True,
                  note="Dotted green: a square-root reference anchored at the median point.")

        ax = axes[0][1]
        if len(traj):
            ax.errorbar(traj.executed_fraction, traj.mean_impact_bps, yerr=traj.sem_bps,
                        fmt="o-", ms=5, lw=1.7, capsize=2, label="measured")
            end = traj.mean_impact_bps.iloc[-1]
            ax.plot(traj.executed_fraction, end * traj.executed_fraction, **pl.REFERENCE_KW)
            ax.plot(traj.executed_fraction, end * np.sqrt(traj.executed_fraction),
                    **pl.LITERATURE_KW)
        n_traj = int(traj.n.iloc[0]) if len(traj) else 0
        pl.finish(ax, "Impact along the trajectory", "fraction of parent executed",
                  "impact so far (bps)", legend=True,
                  note=f"Dashed grey: linear in executed quantity. Dotted green: square "
                       f"root. {n_traj:,} parents with >= {TRAJECTORY_MIN_CHILDREN} "
                       f"children, each interpolated onto the same grid.")

        ev = dec[dec.clock == "event"] if len(dec) else pd.DataFrame()
        ck = dec[dec.clock == "seconds"] if len(dec) else pd.DataFrame()

        # The two decay clocks get their own panels: "10" means ten orders on
        # one and ten seconds on the other, so sharing an axis would invite a
        # false comparison.
        ax = axes[0][2]
        if len(ck):
            ax.plot(ck.horizon, ck.fraction_of_peak, "o-", ms=5, lw=1.8, label="raw")
            ax.plot(ck.horizon, ck.adjusted_fraction_of_peak, "^:", ms=5, lw=1.6,
                    color="#7A4FA3", label="drift-adjusted")
        ax.axhline(1.0, **pl.REFERENCE_KW)
        ax.axhline(2 / 3, **pl.LITERATURE_KW)
        ax.set_xscale("log")
        pl.finish(ax, "Post-completion decay (clock time)", "seconds after completion",
                  "impact / peak impact", legend=True,
                  note="Dotted green: the 2/3-of-peak level Zarinelli et al. report for "
                       "US equities - not a universal constant.")

        ax = axes[1][0]
        if len(ev):
            ax.plot(ev.horizon, ev.fraction_of_peak, "o-", ms=5, lw=1.8,
                    color="#0E7C66", label="raw")
        ax.axhline(1.0, **pl.REFERENCE_KW)
        ax.axhline(2 / 3, **pl.LITERATURE_KW)
        ax.set_xscale("log")
        pl.finish(ax, "Post-completion decay (event time)",
                  "aggressive orders after completion", "impact / peak impact",
                  legend=True,
                  note="Rising above 1 is the correlated flow that follows a parent, "
                       "not a measurement error - compare facts 9 and 10.")

        ax = axes[1][1]
        ax.scatter(runs.duration_s.clip(lower=0.01), runs.impact_bps, s=4, alpha=0.15,
                   color="#6B8FBF")
        dbins = st.quantile_bins(runs.duration_s.to_numpy(), 12)
        dm, _ = st.group_mean(runs.duration_s.to_numpy(), dbins, 12)
        im, _ = st.group_mean(runs.impact_bps.to_numpy(), dbins, 12)
        ax.plot(np.clip(dm, 0.01, None), im, "o-", ms=5, lw=1.8, color="#B45309",
                label="binned mean")
        ax.set_xscale("log")
        pl.finish(ax, "Impact vs execution duration", "duration (s)", "impact (bps)",
                  legend=True)

        ax = axes[1][2]
        ax.hist(np.log10(np.clip(runs.q_over_adv, 1e-9, None)), bins=60,
                color="#6B8FBF", alpha=0.9)
        pl.finish(ax, "Range of relative size explored", "log10(Q / ADV)",
                  "pseudo-metaorders",
                  note="Donier & Bonart resolve square-root impact over ~4 decades of "
                       "metaorder size; the width of this histogram is what this proxy "
                       "can actually constrain.")
        fig.suptitle(f"Facts 1-3 (proxy) - metaorder impact | {rd.label}", x=0.02,
                     ha="left", fontsize=12.5, fontweight="semibold")
        pl.layout(fig)
        ctx.out.save_figure(fig, FACT, "metaorder_impact_proxy", name)

        perm = np.nan
        perm_adj = np.nan
        if len(ck):
            far = ck[ck.horizon >= 600]
            if len(far):
                perm = float(far.fraction_of_peak.iloc[0])
                perm_adj = float(far.adjusted_fraction_of_peak.iloc[0])
        metrics += [
            {"metric": "pseudo-metaorders identified", "regime": name,
             "value": float(len(runs)),
             "detail": f"gap {cfg.PSEUDO_METAORDER_GAP_S}s, >= "
                       f"{cfg.PSEUDO_METAORDER_MIN_CHILDREN} children"},
            {"metric": "impact concavity exponent delta (proxy)", "regime": name,
             "value": power["exponent"],
             "detail": f"se {power['se']:.3f}, R2={power['r2']:.3f}; square root = 0.5"},
            {"metric": "power-law R2 minus logarithmic R2", "regime": name,
             "value": (power["r2"] - logfit["r2"]) if np.isfinite(logfit["r2"]) else np.nan,
             "detail": "positive favours the power law (Zarinelli et al. caveat)"},
            {"metric": "trajectory concavity: impact at 50% executed / impact at 100%",
             "regime": name,
             "value": (float(traj.mean_impact_bps.iloc[N_TRAJECTORY_BINS // 2 - 1]
                             / traj.mean_impact_bps.iloc[-1]) if len(traj) else np.nan),
             "detail": "0.5 = linear build-up, ~0.71 = square-root build-up"},
            {"metric": "impact 10 min after completion / peak (raw)", "regime": name,
             "value": perm,
             "detail": "includes market drift and the correlated flow that follows"},
            {"metric": "impact 10 min after completion / peak (drift-adjusted)",
             "regime": name, "value": perm_adj,
             "detail": "random-time control removed; Zarinelli et al. ~2/3 in US equities, "
                       "Donier-Bonart near-full decay for the uninformed part of BTC impact"},
            {"metric": "median pseudo-metaorder size (BTC)", "regime": name,
             "value": float(runs.qty_btc.median()), "detail": ""},
            {"metric": "Q/ADV explored: 1st percentile", "regime": name,
             "value": float(runs.q_over_adv.quantile(0.01)),
             "detail": "Donier-Bonart span ~4 decades of metaorder size"},
            {"metric": "Q/ADV explored: 99th percentile", "regime": name,
             "value": float(runs.q_over_adv.quantile(0.99)),
             "detail": "decades covered = log10(p99/p01)"},
            {"metric": "decades of size covered by the proxy", "regime": name,
             "value": float(np.log10(runs.q_over_adv.quantile(0.99)
                                     / max(runs.q_over_adv.quantile(0.01), 1e-12))),
             "detail": "a narrow range flattens any concavity estimate"},
            {"metric": "median participation rate of the proxy", "regime": name,
             "value": float(runs.participation.median()),
             "detail": "mechanically high: runs are defined by same-signed flow"},
            {"metric": "average daily volume (BTC)", "regime": name, "value": adv,
             "detail": ""},
            {"metric": "daily volatility (bps)", "regime": name, "value": sigma,
             "detail": "from 60 s mid bars"},
        ]

    fig, axes = pl.new_figure(1, 3, width=5.2, height=4.2)
    for name, rd in ctx.each():
        if name not in size_curves:
            continue
        c = size_curves[name]
        axes[0].errorbar(c.mean_q_over_adv, c.mean_impact_over_sigma, yerr=c.sem_impact,
                         fmt="o", ms=4, lw=1.4, capsize=2, color=rd.color)
        power, _ = fits[name]
        xs = np.geomspace(c.mean_q_over_adv.min(), c.mean_q_over_adv.max(), 50)
        if np.isfinite(power["exponent"]):
            axes[0].plot(xs, power["prefactor"] * xs ** power["exponent"], lw=1.9,
                         color=rd.color, label=f"{rd.label}: delta={power['exponent']:.3f}")
        t = trajectories[name]
        if len(t):
            axes[1].plot(t.executed_fraction, t.mean_impact_bps / t.mean_impact_bps.iloc[-1],
                         "o-", ms=4, lw=1.8, color=rd.color, label=rd.label)
        d = decays[name]
        if len(d):
            ck = d[d.clock == "seconds"]
            axes[2].plot(ck.horizon, ck.fraction_of_peak, "o-", ms=4, lw=1.8,
                         color=rd.color, label=f"{rd.label} (raw)")
            axes[2].plot(ck.horizon, ck.adjusted_fraction_of_peak, "^:", ms=4, lw=1.5,
                         alpha=0.8, color=rd.color, label=f"{rd.label} (drift-adj.)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    pl.finish(axes[0], "Impact vs size", "Q / ADV", "impact / daily sigma", legend=True)
    f = np.linspace(0.05, 1.0, 40)
    axes[1].plot(f, f, **pl.REFERENCE_KW)
    axes[1].plot(f, np.sqrt(f), **pl.LITERATURE_KW)
    pl.finish(axes[1], "Trajectory shape (normalised)", "fraction executed",
              "impact / final impact", legend=True)
    axes[2].axhline(1.0, **pl.REFERENCE_KW)
    axes[2].axhline(2 / 3, **pl.LITERATURE_KW)
    axes[2].set_xscale("log")
    pl.finish(axes[2], "Post-completion decay", "seconds after completion",
              "impact / peak", legend=True)
    fig.suptitle("Facts 1-3 (proxy) - metaorder impact by regime", x=0.02, ha="left",
                 fontsize=12.5, fontweight="semibold")
    pl.layout(fig, rect=(0, 0, 1, 0.94))
    ctx.out.save_figure(fig, FACT, "metaorder_impact_proxy", "comparison")

    return {
        "fact": FACT, "id": FACT_ID, "priority": PRIORITY, "title": TITLE,
        "metrics": metrics,
        "literature": "Donier & Bonart (BTC square-root impact); Toth et al. "
                      "(latent liquidity); Zarinelli et al. (limited range, log "
                      "alternative, ~2/3 relaxation); Bucci et al. (slow decay).",
        "caveat": "Proxy only: pseudo-metaorders are runs of same-signed aggressive "
                  "orders, not labelled parent orders, and their participation rate "
                  "is mechanically high.",
    }
