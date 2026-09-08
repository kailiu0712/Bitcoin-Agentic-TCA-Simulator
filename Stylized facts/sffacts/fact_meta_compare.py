"""Compare two published proxies and the legacy run on identical measured prices.

Runs from cached parquet with numpy/pandas/matplotlib only: no book.py import,
no numba, no ingest. Public entry: run_comparison(cfg), or run(ctx) in the full
framework. All outputs go to a separate metaorder_comparison directory.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sfcore.metaorder_proxies import build_parents

DAY_NS = 86_400_000_000_000
SIZE_EDGES = np.logspace(-9, 0, 31)
PHI_GRID = np.linspace(0, 1, 21)  # includes exactly 0.5 and 1
Z_GRID = np.linspace(1, 3, 25)
SECONDS = np.array([0.1, 1, 5, 15, 30, 60, 180, 600], dtype=float)


def asof_prices(times, prices, queries, max_age_ns):
    """No lookahead; outside the observed range or stale => NaN, not last price."""
    queries = np.asarray(queries, dtype=np.int64)
    out = np.full(queries.shape, np.nan)
    if not len(times):
        return out
    pos = np.searchsorted(times, queries, side="right") - 1
    safe = np.clip(pos, 0, len(times) - 1)
    valid = ((pos >= 0) & (queries <= times[-1])
             & (queries - times[safe] <= max_age_ns))
    out[valid] = prices[safe[valid]]
    return out


def _load(root, spec):
    """Make one row per order and a matched fill-row sensitivity table.

    Cached fills can include warmup and obsolete burst offsets. Join groups on
    receipt time, side and occurrence, then VERIFY count, quantity and VWAP.
    A failed verification excludes the group; never index trades by burst_idx.
    """
    t = pd.read_parquet(root / "trades.parquet")
    f = pd.read_parquet(root / "fills.parquet")
    q = pd.read_parquet(root / "quotes.parquet")
    lo = pd.Timestamp(spec["start_date"], tz="UTC").value
    hi = pd.Timestamp(spec["end_date"], tz="UTC").value + DAY_NS
    audit = {"cached_trades": len(t), "cached_fills": len(f), "cached_quotes": len(q)}
    f = f.loc[(f.recv_ns >= lo) & (f.recv_ns < hi)].copy()
    t = t.loc[(t.recv_ns >= lo) & (t.recv_ns < hi)].copy()
    audit["fills_removed_outside_regime"] = audit["cached_fills"] - len(f)
    t["occ"] = t.groupby(["recv_ns", "sign"], sort=False).cumcount()
    t["source_order_id"] = np.arange(len(t))
    f["notional"] = f.price * f.qty
    groups = f.groupby("burst_idx", sort=True).agg(
        recv_ns=("recv_ns", "first"), sign=("sign", "first"),
        fill_qty=("qty", "sum"), fill_count=("qty", "size"),
        fill_notional=("notional", "sum"))
    groups["occ"] = groups.groupby(["recv_ns", "sign"], sort=False).cumcount()
    matched = groups.reset_index().merge(
        t[["recv_ns", "sign", "occ", "source_order_id", "qty", "n_fills", "vwap"]],
        on=["recv_ns", "sign", "occ"], how="left", validate="one_to_one")
    verified = (matched.source_order_id.notna()
                & np.isclose(matched.fill_qty, matched.qty, rtol=1e-7, atol=1e-10)
                & (matched.fill_count == matched.n_fills)
                & np.isclose(matched.fill_notional / matched.fill_qty, matched.vwap,
                             rtol=1e-9, atol=1e-6))
    audit["fill_groups_unverified"] = int((~verified).sum())
    links = matched.loc[verified, ["burst_idx", "source_order_id"]]
    # Restrict BOTH granularities to the same verified orders to isolate row splitting.
    t = t.loc[t.source_order_id.isin(links.source_order_id)].copy()
    valid = (np.isfinite(t.mid) & (t.mid > 0) & (t.qty > 0)
             & t.sign.isin([-1, 1]) & (t.exch_ns >= lo) & (t.exch_ns < hi))
    audit["orders_removed_invalid"] = int((~valid).sum())
    audit["exchange_backsteps_before_sort"] = int((t.exch_ns.diff() < 0).sum())
    t = t.loc[valid].sort_values(["exch_ns", "recv_ns", "source_order_id"], kind="stable")
    t = t.reset_index(drop=True)
    t["order_pos"] = np.arange(len(t))
    t["ts_ns"] = t.exch_ns.astype("int64")
    t["day"] = t.ts_ns // DAY_NS
    f = f.merge(links, on="burst_idx", how="inner", validate="many_to_one")
    f = f.merge(t[["source_order_id", "order_pos", "ts_ns", "mid", "day"]],
                on="source_order_id", how="inner", validate="many_to_one")
    f = f.sort_values(["ts_ns", "order_pos"], kind="stable").reset_index(drop=True)
    # Fill prices are execution prices, NOT mids. Each matched fill inherits its
    # aggressive order's timestamp and pre-trade mid; limitations are explicit.
    f["vwap"] = f.price
    q = q.loc[(q.recv_ns >= lo) & (q.recv_ns < hi) & (q.exch_ns >= lo) & (q.exch_ns < hi)].copy()
    audit["quote_exchange_backsteps_before_sort"] = int((q.exch_ns.diff() < 0).sum())
    q = q.sort_values(["exch_ns", "recv_ns"], kind="stable").drop_duplicates("exch_ns", keep="last")
    audit.update(matched_orders=len(t), matched_fills=len(f),
                 source_order_coverage=len(t) / max(audit["cached_trades"], 1),
                 fill_time="inherited verified aggressive-order exchange timestamp",
                 quote_time="stable exchange-time sort; same timestamp last receipt wins")
    if not len(t) or not len(f):
        raise ValueError("No verified matched trades/fills")
    return t, f, q, audit


def load_analysis_rows(root, spec):
    """Public wrapper over the verified-row loader, for other fact modules.

    ``fact_meta_notebook`` measures the same parent populations on the same
    rows, so it must not re-implement this validation: sharing one loader is
    what makes the two modules' numbers comparable.
    """
    return _load(root, spec)


def _daily_scales(orders):
    rows = []
    for day, d in orders.groupby("day", sort=True):
        prices = d.mid.to_numpy()
        sigma = (prices.max() - prices.min()) / prices[0]
        rows.append({"day": day, "date": str(pd.Timestamp(int(day) * DAY_NS, tz="UTC").date()),
                     "volume": d.qty.sum(), "sigma_range": sigma,
                     "orders": len(d), "first_ns": d.ts_ns.iloc[0], "last_ns": d.ts_ns.iloc[-1]})
    return pd.DataFrame(rows)


def _curve(frame, x, y, edges):
    work = pd.DataFrame({"x": frame[x], "y": frame[y]})
    work["bin"] = pd.cut(work.x, edges, include_lowest=True, labels=False)
    return work.groupby("bin", observed=True).agg(
        x=("x", "mean"), y=("y", "mean"), sem_iid=("y", "sem"), n=("y", "count")).reset_index()


def fit_size(curve, min_bin=20):
    """Power and log fit scored on SAME signed y values and bins, equal bin weight.

    Bounded 1D power exponent search avoids log-transforming negative responses.
    Boundary hits are flagged. This is a descriptive in-sample fit, no iid CI.
    """
    d = curve.loc[(curve.n >= min_bin) & (curve.x > 0)].dropna(subset=["x", "y"])
    empty = dict(delta=np.nan, power_r2=np.nan, log_r2=np.nan, prefactor=np.nan,
                 delta_at_boundary=False, n_fit_bins=len(d))
    if len(d) < 4:
        return empty
    x, y = d.x.to_numpy(), d.y.to_numpy()
    slopes = np.linspace(-0.2, 1.2, 1401)
    scale = np.exp(np.mean(np.log(x)))
    basis = (x[:, None] / scale) ** slopes[None, :]
    amplitudes = np.maximum(0, (basis.T @ y) / np.sum(basis ** 2, axis=0))
    error = np.sum((y[:, None] - basis * amplitudes) ** 2, axis=0)
    best = int(np.argmin(error))
    total = np.sum((y - y.mean()) ** 2)
    design = np.column_stack([np.ones(len(x)), np.log(x)])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return dict(delta=float(slopes[best]), prefactor=float(amplitudes[best] / scale ** slopes[best]),
                power_r2=float(1 - error[best] / total) if total > 0 else np.nan,
                log_r2=float(1 - np.sum((y - design @ beta) ** 2) / total) if total > 0 else np.nan,
                delta_at_boundary=best in (0, len(slopes)-1), n_fit_bins=len(d))


def _measure(day, parents, orders, quote_times, quote_mid, scale, max_age_ns):
    """Shared measurement. Keep signed outcomes; end=next distinct-time order's pre-mid."""
    kept = [p for p in parents if len(p) >= 2]
    if not kept:
        return pd.DataFrame(), []
    first = np.array([p[0] for p in kept])
    last = np.array([p[-1] for p in kept])
    starts = day.ts_ns.to_numpy()[first]
    ends = day.ts_ns.to_numpy()[last]
    order_times = orders.ts_ns.to_numpy()
    next_pos = np.searchsorted(order_times, ends, side="right")
    safe = np.minimum(next_pos, len(orders)-1)
    valid = (next_pos < len(orders)) & (orders.day.to_numpy()[safe] == scale.day)
    m0 = day.mid.to_numpy()[first]
    m1 = orders.mid.to_numpy()[safe].astype(float).copy()
    m1[~valid] = np.nan
    side = day.sign.to_numpy()[first]
    qty = day.qty.to_numpy()
    amounts = np.array([qty[p].sum() for p in kept])
    all_q = orders.qty.to_numpy()
    cum = np.concatenate([[0.0], np.cumsum(all_q)])
    left = np.searchsorted(order_times, starts, side="left")
    right = np.searchsorted(order_times, ends, side="right")
    window = cum[right] - cum[left]
    mids100 = asof_prices(quote_times, quote_mid, ends + 100_000_000, max_age_ns)
    mids100[(ends + 100_000_000) // DAY_NS != scale.day] = np.nan
    impact = side * (m1 - m0) / m0 * 1e4
    frame = pd.DataFrame({
        "start_ns": starts, "end_ns": ends, "sign": side, "n_children": [len(p) for p in kept],
        "qty_btc": amounts, "duration_s": (ends-starts)/1e9, "mid_start": m0, "mid_end": m1,
        "end_price_delay_s": (order_times[safe]-ends)/1e9,
        "impact_bps": impact, "impact_over_sigma": impact/(scale.sigma_range*1e4),
        "impact_100ms_bps": side*(mids100-m0)/m0*1e4,
        "q_over_vday": amounts/scale.volume, "participation": amounts/window,
        "date": scale.date, "day": scale.day, "sigma_range": scale.sigma_range,
    })
    frame["child_source_rows"] = [day.source_row.to_numpy()[p].tolist() for p in kept]
    frame["child_order_positions"] = [day.order_pos.to_numpy()[p].tolist() for p in kept]
    # Whole parent, equal weight; post-child mid paired with executed volume.
    paths = []
    for k, p in enumerate(kept):
        if len(p) < 5 or not valid[k]:
            continue
        child_times = day.ts_ns.to_numpy()[p]
        after = np.searchsorted(order_times, child_times, side="right")
        if np.any(after >= len(orders)):
            continue
        f = np.cumsum(qty[p]) / amounts[k]
        y = side[k]*(orders.mid.to_numpy()[after]-m0[k])/m0[k]*1e4
        paths.append(np.interp(PHI_GRID, np.r_[0, f], np.r_[0, y]))
    return frame, paths


def _decay(frame, qt, qm, max_age_ns):
    """Common complete cohort across ALL horizons within each clock; no extrapolation."""
    results = []
    d = frame.loc[frame.impact_bps.notna() & (frame.duration_s > 0)].copy()
    # Size's endpoint is the next order's PRE-mid. Relaxation samples the quote
    # clock itself, so normalize it at quote-asof(t_end), not a later price.
    # Export BOTH ratios rather than manufacture a z=1 discontinuity silently.
    at_end = asof_prices(qt, qm, d.end_ns.to_numpy(), max_age_ns)
    d["quote_completion_bps"] = d.sign * (at_end-d.mid_start) / d.mid_start * 1e4
    for clock, grid in [("seconds", SECONDS), ("z", Z_GRID)]:
        if not len(d):
            continue
        start, end = d.start_ns.to_numpy(), d.end_ns.to_numpy()
        queries = (end[:, None] + (grid*1e9).astype(np.int64)[None, :] if clock == "seconds"
                   else start[:, None] + ((end-start)[:, None]*grid).astype(np.int64))
        prices = asof_prices(qt, qm, queries, max_age_ns)
        prices[(queries // DAY_NS) != d.day.to_numpy()[:, None]] = np.nan
        complete = np.isfinite(prices).all(axis=1) & np.isfinite(d.quote_completion_bps)
        good = d.loc[complete]
        if not len(good):
            continue
        values = good.sign.to_numpy()[:, None]*(prices[complete]-good.mid_start.to_numpy()[:, None])
        values = values/good.mid_start.to_numpy()[:, None]*1e4
        reference = good.quote_completion_bps.mean()
        size_endpoint = good.impact_bps.mean()
        for j,h in enumerate(grid):
            results.append(dict(clock=clock, horizon=h, mean_impact_bps=values[:,j].mean(),
                                fraction_of_completion=values[:,j].mean()/reference if abs(reference)>1e-12 else np.nan,
                                fraction_of_size_endpoint=values[:,j].mean()/size_endpoint if abs(size_endpoint)>1e-12 else np.nan,
                                reference_quote_completion_bps=reference,
                                reference_size_endpoint_bps=size_endpoint,
                                sem_iid=values[:,j].std(ddof=1)/np.sqrt(len(good)) if len(good)>1 else np.nan,
                                n=len(good), n_before_complete_filter=len(d)))
    return pd.DataFrame(results)


def _specs(cfg):
    settings = cfg.METAORDER_COMPARISON
    specs = []
    for n in settings["trader_counts"]:
        specs.append((f"random_n{n}", "random_traders", dict(n_traders=n), n==settings["trader_counts"][0]))
    for wait in settings["mean_wait_seconds"]:
        specs.append((f"threshold_wait{wait:g}s", "time_threshold",
                      dict(mean_wait_s=wait, mu=settings["mu"], max_children=settings["max_children"],
                           max_target_delay_s=settings["max_target_delay_s"]), wait==settings["mean_wait_seconds"][0]))
    specs.append(("local_run_1s", "local_runs", dict(gap_s=cfg.PSEUDO_METAORDER_GAP_S), True))
    return specs


def _md_table(frame):
    # Avoid adding tabulate as a dependency to an otherwise plain pandas path.
    def cell(x):
        if isinstance(x, (float, np.floating)):
            return f"{x:.5g}" if np.isfinite(x) else "—"
        return str(x).replace("|", "/")
    return "| " + " | ".join(frame.columns) + " |\n| " + " | ".join(["---"]*len(frame.columns)) + " |\n" + "\n".join(
        "| " + " | ".join(cell(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None))


def _plots(root, curves, trajectories, decays, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for regime in summary.regime.unique():
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        primary = summary.loc[(summary.regime==regime)&summary.primary & (summary.granularity=="aggressive")]
        for variant in primary.variant.unique():
            label = variant
            select = lambda df: df.loc[(df.regime==regime)&(df.variant==variant)&(df.granularity=="aggressive")]
            c = select(curves)
            c = c.loc[(c.selection=="all_signed") & (c.n>=20)].groupby("bin").agg(x=("x","mean"),y=("y","mean"))
            axes[0,0].plot(c.x,c.y,"o-",label=label,ms=3)
            p = select(trajectories).groupby("phi").mean(numeric_only=True)
            if len(p): axes[0,1].plot(p.index,p.mean_impact_bps,label=label)
            d = select(decays)
            for clock,ax in [("z",axes[0,2]),("seconds",axes[1,0])]:
                z = d.loc[d.clock==clock].groupby("horizon").mean(numeric_only=True)
                if len(z): ax.plot(z.index,z.fraction_of_completion,label=label)
        # Primary methods: signed vs positive-only selection and child granularity.
        sub = summary.loc[(summary.regime==regime)&summary.primary]
        for gran, marker in [("aggressive","o"),("fill","s")]:
            g = sub.loc[sub.granularity==gran]
            for j,var in enumerate(primary.variant.unique()):
                values=g.loc[g.variant==var]
                if not len(values): continue
                axes[1,1].scatter(np.full(len(values),j)+(0.12 if gran=="fill" else -0.12),values.delta,
                                  marker=marker,label=gran if j==0 else None)
        names=list(primary.variant.unique())
        axes[1,1].set_xticks(range(len(names)),names,rotation=15)
        vals=primary.groupby("variant").agg(signed=("delta","mean"),positive=("positive_only_delta","mean"))
        vals.plot.bar(ax=axes[1,2],rot=15)
        axes[0,0].set(xscale="log",title="Size: all signed observations",xlabel="Q / daily volume",ylabel="Mean I / daily range")
        axes[0,1].set(title="Equal-parent execution trajectory",xlabel="Executed volume fraction",ylabel="Mean impact (bps)")
        axes[0,2].set(title="Duration-rescaled response",xlabel="z = (t-start) / T",ylabel="Response / quote response at end")
        axes[1,0].set(xscale="log",title="Clock-time response",xlabel="Seconds after completion",ylabel="Response / quote response at end")
        axes[1,1].set(title="Granularity & seed sensitivity",ylabel="Descriptive size exponent")
        axes[1,2].set(title="Effect of positive-only selection",ylabel="Descriptive size exponent")
        for ax in axes.flat:
            ax.grid(alpha=.25)
            if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=8)
        fig.suptitle(f"{regime}: identical daily scales / signed outcomes / complete horizons\nSeed replicates are not independent market samples",fontsize=13)
        fig.tight_layout(rect=(0,0,1,.94))
        fig.savefig(root/f"comparison_{regime}.png",dpi=140)
        plt.close(fig)


def run_comparison(cfg):
    settings = cfg.METAORDER_COMPARISON
    root = Path(cfg.RESULTS_DIR)/"metaorder_comparison"
    root.mkdir(parents=True,exist_ok=True)
    summaries, all_curves, all_paths, all_decay, audits, coverage, daily_results = [],[],[],[],[],[],[]
    fingerprints = []
    max_age_ns = int(settings["max_quote_age_s"]*1e9)
    for regime,spec in cfg.REGIMES.items():
        cache = Path(cfg.CACHE_DIR)/regime
        for filename in ("trades.parquet","fills.parquet","quotes.parquet"):
            p=cache/filename
            fingerprints.append(dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns))
        orders,fills,quotes,audit = _load(cache,spec)
        audit["regime"]=regime
        audits.append(audit)
        scales = _daily_scales(orders)
        scales.to_csv(root/f"daily_scales_{regime}.csv",index=False)
        qt=quotes.exch_ns.to_numpy(np.int64)
        qm=(quotes.bid_tick.to_numpy()+quotes.ask_tick.to_numpy())*cfg.TICK_SIZE/2
        # Expose the simple row contract for notebook users; no book imports needed.
        orders[["ts_ns","sign","qty","mid","vwap","order_pos","source_order_id","day"]].to_parquet(root/f"analysis_rows_{regime}.parquet",index=False)
        for gran,data in [("aggressive",orders),("fill",fills)]:
            data=data.copy()
            data["source_row"]=np.arange(len(data))
            byday={day:d.reset_index(drop=True) for day,d in data.groupby("day",sort=True)}
            for variant,method,params,primary in _specs(cfg):
                if gran=="fill" and (not primary or method=="local_runs"):
                    continue
                seeds=settings["seeds"] if method!="local_runs" else [settings["seeds"][0]]
                for seed in seeds:
                    start_time=time.perf_counter()
                    frames, paths = [],[]
                    tags=dict(regime=regime,granularity=gran,variant=variant,method=method,seed=seed,primary=primary)
                    for scale in scales.itertuples(index=False):
                        day=byday.get(scale.day)
                        if day is None or scale.sigma_range<=0: continue
                        rng=np.random.default_rng(np.random.SeedSequence([seed,int(scale.day)]))
                        parents=build_parents(day,method,rng,**params)
                        flat=np.concatenate(parents) if parents else np.array([],dtype=int)
                        if len(flat)!=len(np.unique(flat)):
                            raise AssertionError("A child assigned twice")
                        measured,daypaths=_measure(day,parents,orders,qt,qm,scale,max_age_ns)
                        valid=measured.impact_bps.notna() if len(measured) else pd.Series([],dtype=bool)
                        coverage.append(dict(**tags,date=scale.date,input_rows=len(day),assigned_rows=len(flat),
                                             all_groups=len(parents),groups_ge2=len(measured),
                                             measured_parents=int(valid.sum()),assigned_share=len(flat)/len(day)))
                        if not len(measured): continue
                        measured["parent_id"]=np.arange(len(measured))
                        frames.append(measured.assign(**tags))
                        paths.extend(daypaths)
                        dc=_curve(measured,"q_over_vday","impact_over_sigma",SIZE_EDGES)
                        daily_results.append(dict(**tags,date=scale.date,n=int(valid.sum()),
                                                  mean_impact_bps=measured.impact_bps.mean(),**fit_size(dc)))
                    if not frames: continue
                    frame=pd.concat(frames,ignore_index=True)
                    # Membership and endpoints are exportable/reviewable, including invalid end prices.
                    frame.to_parquet(root/f"parents_{regime}_{gran}_{variant}_seed{seed}.parquet",index=False)
                    good=frame.loc[frame.impact_bps.notna()]
                    curve=_curve(good,"q_over_vday","impact_over_sigma",SIZE_EDGES)
                    fit=fit_size(curve)
                    positive=_curve(good.loc[good.impact_bps>0],"q_over_vday","impact_over_sigma",SIZE_EDGES)
                    posfit=fit_size(positive)
                    all_curves.extend([curve.assign(**tags,selection="all_signed"),positive.assign(**tags,selection="positive_only")])
                    decay=_decay(frame,qt,qm,max_age_ns)
                    if len(decay): all_decay.append(decay.assign(**tags))
                    if paths:
                        matrix=np.vstack(paths)
                        path=pd.DataFrame(dict(phi=PHI_GRID,mean_impact_bps=matrix.mean(axis=0),
                                               sem_iid=matrix.std(axis=0,ddof=1)/np.sqrt(len(matrix)),n=len(matrix)))
                        all_paths.append(path.assign(**tags))
                        trajectory_half=matrix[:,10].mean()/matrix[:,-1].mean() if abs(matrix[:,-1].mean())>1e-12 else np.nan
                    else: trajectory_half=np.nan
                    # Joint binned signed conditional means; no log of individual impact.
                    joint=good.loc[good.duration_s>0].copy()
                    joint["qbin"]=pd.qcut(joint.q_over_vday,5,duplicates="drop",labels=False)
                    joint["tbin"]=pd.qcut(joint.duration_s,5,duplicates="drop",labels=False)
                    surface=joint.groupby(["qbin","tbin"],observed=True).agg(
                        q=("q_over_vday","mean"),duration_s=("duration_s","mean"),
                        impact_over_sigma=("impact_over_sigma","mean"),n=("impact_bps","count")).reset_index()
                    surface.assign(**tags).to_csv(root/f"surface_{regime}_{gran}_{variant}_seed{seed}.csv",index=False)
                    summaries.append(dict(**tags,n_parents=len(good),median_children=good.n_children.median(),
                                          median_duration_s=good.duration_s.median(),median_q_btc=good.qty_btc.median(),
                                          median_participation=good.participation.median(),positive_share=(good.impact_bps>0).mean(),
                                          mean_impact_bps=good.impact_bps.mean(),mean_100ms_bps=good.impact_100ms_bps.mean(),
                                          median_end_price_delay_s=good.end_price_delay_s.median(),
                                          trajectory_half=trajectory_half,positive_only_delta=posfit["delta"],**fit))
                    print(f"{regime:6} {gran:10} {variant:20} seed={seed}: n={len(good):,}, delta={fit['delta']:.3f}, {time.perf_counter()-start_time:.1f}s",flush=True)
    summary=pd.DataFrame(summaries)
    if summary.empty: raise ValueError("No comparison results")
    curves=pd.concat(all_curves,ignore_index=True)
    trajectories=pd.concat(all_paths,ignore_index=True)
    decays=pd.concat(all_decay,ignore_index=True)
    outputs={"summary":summary,"size_curves":curves,"trajectories":trajectories,"decay":decays,
             "input_audit":pd.DataFrame(audits),"coverage":pd.DataFrame(coverage),"daily_metrics":pd.DataFrame(daily_results)}
    for name,table in outputs.items(): table.to_csv(root/f"{name}.csv",index=False)
    _plots(root,curves,trajectories,decays,summary)
    grouped=summary.groupby(["regime","granularity","variant"],sort=False).agg(
        seeds=("seed","size"),parents=("n_parents","mean"),T_median_s=("median_duration_s","mean"),
        participation=("median_participation","mean"),delta_mean=("delta","mean"),delta_seed_sd=("delta","std"),
        delta_positive_only=("positive_only_delta","mean"),trajectory_half=("trajectory_half","mean")).reset_index()
    grouped.to_csv(root/"comparison.csv",index=False)
    code_hashes={}
    for p in [Path(__file__),Path(__file__).parents[1]/"sfcore/metaorder_proxies.py",Path(cfg.__file__)]:
        code_hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=dict(settings=settings,regimes=cfg.REGIMES,inputs=fingerprints,code_sha256=code_hashes,
                  endpoint="pre-mid at next strictly later aggressive-order timestamp; also export +100ms",
                  normalization="matched daily order volume; daily pre-mid range / first mid",
                  uncertainty="sem_iid is descriptive only; seed SD is construction sensitivity, not market CI",
                  threshold_semantics="forward scan, jumped rows unassigned, P(s) proportional s^(-1-mu), phi=1/mean_wait_s, target overshoot cutoff in seconds")
    (root/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    report="""# 两种论文代理与 run 基线：统一口径比较

完整配置、输入文件时间/大小、代码哈希见 manifest.json；使用当前缓存重算，不能与旧 SUMMARY.md 数值直接拼接。

- random_n6 / n20：2025-03 论文随机 trader，再按各 trader 内同向串分组。
- threshold_wait5s / 20s：2025-09 论文 Algorithm 2；分买卖、截断离散幂律子单数量、指数目标等待、4 秒目标超时截止。forward scan 跳过的成交保持未分配（coverage.csv）。这是明确秒单位的真实市场适配，不声称论文已校准这些参数。
- local_run_1s：旧 run 分组逻辑，使用统一末价、逐日归一化和同侧末笔完成点；不是原模块的旧数值。
- aggressive 与 fill 使用完全相同的已验证订单集合；fill 的交换时间与 pre-mid 从所属订单继承，不能视作独立 quote 或独立决策。fill 粒度仅作敏感性。
- 主结果保留正负/零 signed impact；positive_only 仅展示筛正造成的差异。delta 是同尺度分箱均值上的描述拟合，边界命中另有标记，不能把它当成已验证的幂律。
- 每天独立构造，3 个 seed 是同一市场数据的重复分组，seed SD 不是统计显著性。各方法 parent 人群不同，当前是无条件对比，不是匹配 Q/T/参与率后的因果比较。
- 价格轨迹每个 parent 等权，至少 5 child；明确包含 phi=0.5 和 1。衰减使用各时钟内共同完整样本，剔除越界、跨日和超过 30 秒的陈旧报价。零 T 不进入 z 衰减。
- 衰减图的分母是相同 quote 时钟在最后 child 时刻的响应，因此 z=1 为 1；另输出 fraction_of_size_endpoint，使用规模检验的下一笔订单前价格作分母。两种端点不可混用；初期上升可能包含报价发布/采样时间效应，不能自动解释为经济上的峰值滞后。
- 当前 raw 响应含市场漂移和后续流，未拟合 propagator 或推断永久冲击。10min 与 z=2 不代表同一时间窗口。

## 结果（跨 seed 均值；seed SD 仅为构造敏感性）

"""+_md_table(grouped)+"\n\n## 数据核验\n\n"+_md_table(pd.DataFrame(audits))+"\n\n## 输出\n\n- comparison_normal.png / comparison_stress.png：六面板对照。\n- parents_*.parquet：逐母单含 child 行索引、价格、规模、时长及状态。\n- size_curves.csv / trajectories.csv / decay.csv / surface_*.csv：底层曲线和 Q–T 面。\n- daily_metrics.csv：逐日指标；coverage.csv：单笔过滤及被算法跳过的行。\n- analysis_rows_*.parquet：可直接逐行分析的干净主动订单表。\n\n所有 SEM 列标为 sem_iid，仅描述横截面散布，未处理时间重叠；本报告不作正式显著性宣称。\n"
    (root/"README.md").write_text(report,encoding="utf-8")
    return summary,root


def run(ctx):
    """Full-framework hook, without overwriting its legacy fact01_03 outputs."""
    summary,root=run_comparison(ctx.cfg)
    ctx.out.manifest.append(dict(kind="table",fact="fact_meta_compare",slug="comparison",regime="comparison",path=str(root/"comparison.csv"),caption="Two published proxy constructions and harmonized run baseline"))
    metrics=[]
    for (regime,variant),g in summary.loc[summary.primary & (summary.granularity=="aggressive")].groupby(["regime","variant"]):
        metrics.append(dict(regime=regime,metric=f"{variant}: mean descriptive delta",value=g.delta.mean(),detail="all signed; seed sensitivity in metaorder_comparison"))
    return dict(fact="fact_meta_compare",id=4,priority="P0-V proxy",title="Published metaorder proxy comparison",metrics=metrics,literature="Maitrier et al. arXiv:2503.18199 and arXiv:2509.05065; explicit real-time adaptation")
