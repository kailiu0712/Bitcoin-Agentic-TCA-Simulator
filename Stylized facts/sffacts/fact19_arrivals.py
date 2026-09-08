"""
Stylized fact 19 (P0-C) - event arrivals and the ZJ summary-statistics table.

Source
    ``Stylied facts ZJ.pdf`` p3, which reports for each pair: average share
    price, average quote spread in dollars and in bps, mean total volume at the
    best quote, mean size of the best-quote orders, and the **mean and median
    interarrival time in milliseconds**.  The framework had every one of those
    except the interarrival statistics, which were absent entirely.

What is added beyond the table
    A mean interarrival time on its own says nothing about whether trading is
    clustered.  So the module also reports, for each event stream:

    * the interarrival survival function on log-log axes, and the share of
      exactly-zero gaps (same-nanosecond events, which a mean cannot show);
    * the Fano factor Var(N)/E(N) of event counts in fixed windows.  A Poisson
      stream gives 1 at every window length; a clustered one gives more, and
      increasingly more as the window grows.

Interpretation limits
    Over-dispersion is *not* by itself evidence of self-excitation.  A
    non-stationary arrival intensity - the market simply being busier at some
    hours than others - produces exactly the same signature over a one-week
    sample.  The Fano curve is therefore reported as a description; no Hawkes
    branching ratio is estimated from it.

Which stream is "the" arrival stream
    ZJ do not say, and the three candidates differ by more than an order of
    magnitude, so all three are reported side by side:

    * **aggressive orders** - one tick per marketable order, the framework's
      decision-level event;
    * **fills** - one tick per printed trade report; one aggressive order that
      sweeps several maker orders prints several of these, so its gaps are
      mechanically clustered and it is not a stream of independent decisions;
    * **top-of-book events** - one tick per change in the best quote or its
      size.  This is the fastest of the three and is closest in magnitude to
      ZJ's reported BTC figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfcore import plotting as pl
from sfcore import stats as st

FACT = "fact19"
FACT_ID = 19
PRIORITY = "P0-C"
TITLE = "Event arrivals, clustering, and market summary statistics"

#: Window lengths (seconds) for the count-dispersion (Fano) curve.
COUNT_WINDOWS_S: tuple[float, ...] = (0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0)

#: A dwell time longer than this is treated as a feed gap rather than a genuine
#: quiescent period, and is excluded from the time-weighted book averages.  Both
#: regimes are contiguous calendar days, so in practice this fires rarely; the
#: excluded fraction is reported so the choice stays visible.
MAX_QUOTE_DWELL_S = 60.0

MAX_ACF_LAG = 200


def _interarrival_stats(ts_ns: np.ndarray, label: str) -> dict:
    """Mean/median gap in ms, plus the share of exactly-simultaneous events."""
    ts = np.sort(np.asarray(ts_ns, dtype=np.int64))
    gaps = np.diff(ts)
    if gaps.size == 0:
        return {"stream": label, "events": int(ts.size)}
    ms = gaps / 1e6
    return {
        "stream": label,
        "events": int(ts.size),
        "mean_interarrival_ms": float(ms.mean()),
        "median_interarrival_ms": float(np.median(ms)),
        "p05_interarrival_ms": float(np.percentile(ms, 5)),
        "p95_interarrival_ms": float(np.percentile(ms, 95)),
        "share_zero_gap": float((gaps == 0).mean()),
        "events_per_second": float(1000.0 / ms.mean()) if ms.mean() > 0 else np.nan,
    }


def _survival(gaps_ms: np.ndarray, n_points: int = 220) -> pd.DataFrame:
    """P(gap > x) on a log grid.  Zero gaps are carried as the value at x -> 0."""
    positive = gaps_ms[gaps_ms > 0]
    if positive.size < 100:
        return pd.DataFrame()
    grid = np.logspace(np.log10(np.percentile(positive, 0.1)),
                       np.log10(np.percentile(positive, 99.99)), n_points)
    ordered = np.sort(gaps_ms)
    exceed = gaps_ms.size - np.searchsorted(ordered, grid, side="right")
    return pd.DataFrame({"gap_ms": grid, "survival": exceed / gaps_ms.size})


def _fano(ts_ns: np.ndarray, windows) -> pd.DataFrame:
    """Var/mean of the event count per fixed window (1 for a Poisson stream)."""
    ts = np.sort(np.asarray(ts_ns, dtype=np.int64))
    if ts.size < 100:
        return pd.DataFrame()
    rows = []
    for seconds in windows:
        step = int(seconds * 1e9)
        bucket = (ts - ts[0]) // step
        n_buckets = int(bucket[-1]) + 1
        if n_buckets < 50:
            continue
        counts = np.bincount(bucket, minlength=n_buckets).astype(np.float64)
        mean = counts.mean()
        rows.append({"window_s": seconds, "mean_count": mean,
                     "var_count": float(counts.var()),
                     "fano": float(counts.var() / mean) if mean > 0 else np.nan,
                     "n_windows": n_buckets})
    return pd.DataFrame(rows)


def _book_averages(rd, tick_size: float) -> dict:
    """Time-weighted top-of-book averages, using every quote-tape dwell period.

    The quote tape emits a row whenever the best price *or* its size changes,
    so the top of book is exactly piecewise-constant between rows: weighting
    each row by the time until the next one is an exact time average of the
    observed state, not a sample of it.
    """
    quotes = rd.quotes
    ts = quotes["recv_ns"].to_numpy(np.int64)
    dwell = np.diff(ts).astype(np.float64)
    keep = (dwell >= 0) & (dwell <= MAX_QUOTE_DWELL_S * 1e9)
    weight = np.where(keep, dwell, 0.0)
    total = weight.sum()
    if total <= 0:
        return {}
    head = slice(0, len(ts) - 1)
    bid = quotes["bid_tick"].to_numpy(np.float64)[head] * tick_size
    ask = quotes["ask_tick"].to_numpy(np.float64)[head] * tick_size
    mid = 0.5 * (bid + ask)
    bid_qty = quotes["bid_qty"].to_numpy(np.float64)[head]
    ask_qty = quotes["ask_qty"].to_numpy(np.float64)[head]

    def average(values):
        return float(np.sum(values * weight) / total)

    spread = ask - bid
    return {
        "average_price_usd": average(mid),
        "average_spread_usd": average(spread),
        "average_spread_bps": average(spread / mid * 1e4),
        "average_spread_ticks": average(spread / tick_size),
        "mean_total_volume_at_best_btc": average(bid_qty + ask_qty),
        "mean_size_of_best_quote_orders_btc": average(0.5 * (bid_qty + ask_qty)),
        "mean_bid_qty_btc": average(bid_qty),
        "mean_ask_qty_btc": average(ask_qty),
        "weighted_hours": total / 3.6e12,
        "share_of_span_excluded_as_gap": float(1.0 - total / max(ts[-1] - ts[0], 1)),
    }


def run(ctx) -> dict:
    pl.apply_style()
    cfg = ctx.cfg
    metrics: list[dict] = []
    summary_rows, arrival_rows, fano_frames, survival_frames = [], [], {}, {}

    for name, rd in ctx.each():
        streams = {
            "aggressive orders": rd.trades["recv_ns"].to_numpy(np.int64),
            "fills": rd.fills["recv_ns"].to_numpy(np.int64),
            "top-of-book events": rd.quotes["recv_ns"].to_numpy(np.int64),
        }
        book = _book_averages(rd, cfg.TICK_SIZE)
        summary_rows.append({"regime": name, "label": rd.label, **book})

        fanos, survivals = [], []
        for label, ts in streams.items():
            arrival_rows.append({"regime": name, **_interarrival_stats(ts, label)})
            gaps_ms = np.diff(np.sort(ts)) / 1e6
            surv = _survival(gaps_ms)
            if len(surv):
                survivals.append(surv.assign(stream=label))
            fano = _fano(ts, COUNT_WINDOWS_S)
            if len(fano):
                fanos.append(fano.assign(stream=label))
        fano_frames[name] = pd.concat(fanos, ignore_index=True) if fanos else pd.DataFrame()
        survival_frames[name] = (pd.concat(survivals, ignore_index=True)
                                 if survivals else pd.DataFrame())

        order_ts = np.sort(streams["aggressive orders"])
        per_second = np.bincount(((order_ts - order_ts[0]) // 1_000_000_000).astype(np.int64))
        count_acf = st.acf(per_second.astype(np.float64), MAX_ACF_LAG)

        fig, axes = pl.new_figure(1, 3, width=5.2, height=4.2)
        ax = axes[0]
        frame = survival_frames[name]
        for label, sub in frame.groupby("stream", sort=False):
            ax.plot(sub["gap_ms"], sub["survival"], lw=1.7, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        pl.finish(ax, "Interarrival survival function", "gap (ms)", "P(gap > x)",
                  legend=True,
                  note="A curve that starts below 1 has exactly-zero gaps: one "
                       "aggressive order prints several fills inside the same "
                       "nanosecond. The long flat plateau before the cliff is not "
                       "a straight line, so these gaps are not a simple power "
                       "law either.")

        ax = axes[1]
        frame = fano_frames[name]
        for label, sub in frame.groupby("stream", sort=False):
            ax.plot(sub["window_s"], sub["fano"], "o-", ms=4.5, lw=1.6, label=label)
        ax.axhline(1.0, **pl.REFERENCE_KW)
        ax.set_xscale("log")
        ax.set_yscale("log")
        pl.finish(ax, "Count dispersion (Fano factor)", "window length (s)",
                  "Var(count) / E(count)", legend=True,
                  note="1 is Poisson. Rising with the window is activity "
                       "clustering - which a non-stationary intensity also "
                       "produces, so this is description, not a Hawkes fit.")

        ax = axes[2]
        lags = np.arange(1, len(count_acf) + 1)
        ax.plot(lags, count_acf, lw=1.5, color=rd.color)
        ax.axhline(0.0, **pl.REFERENCE_KW)
        band = 1.96 / np.sqrt(max(per_second.size, 1))
        ax.fill_between(lags, -band, band, color="#9CA3AF", alpha=0.25,
                        label="white-noise band")
        ax.set_xscale("log")
        pl.finish(ax, "Autocorrelation of orders per second", "lag (seconds)",
                  "autocorrelation", legend=True,
                  note="Slow decay is the activity-clustering counterpart of "
                       "volatility clustering (fact 14).")
        fig.suptitle(f"Fact 19 - event arrivals | {rd.label}", x=0.02, ha="left",
                     fontsize=12.5, fontweight="semibold")
        pl.layout(fig)
        ctx.out.save_figure(fig, FACT, "arrivals", name)

        orders = [r for r in arrival_rows
                  if r["regime"] == name and r["stream"] == "aggressive orders"][0]
        metrics += [
            {"metric": "mean interarrival, aggressive orders (ms)", "regime": name,
             "value": orders.get("mean_interarrival_ms", np.nan),
             "detail": f"median {orders.get('median_interarrival_ms', float('nan')):.1f} ms; "
                       "ZJ p3 reports 185.1 / 100.9 ms for BTC on 2026-08-25 "
                       "(different window and event definition)"},
            {"metric": "mean quoted spread (bps), time-weighted", "regime": name,
             "value": book.get("average_spread_bps", np.nan),
             "detail": f"{book.get('average_spread_usd', float('nan')):.4f} USD; "
                       "ZJ p3 BTC: 0.1 bps / 0.777 USD"},
            {"metric": "mean total volume at best quote (BTC)", "regime": name,
             "value": book.get("mean_total_volume_at_best_btc", np.nan),
             "detail": "ZJ p3 BTC: 0.2868 BTC"},
        ]

    summary = pd.DataFrame(summary_rows)
    arrivals = pd.DataFrame(arrival_rows)
    if len(summary):
        ctx.out.save_table(summary, FACT, "market_summary_statistics", "comparison",
                           caption="ZJ p3 summary statistics, time-weighted over all packets")
    if len(arrivals):
        ctx.out.save_table(arrivals, FACT, "interarrival_statistics", "comparison",
                           caption="Interarrival statistics by event stream")
    for name in fano_frames:
        if len(fano_frames[name]):
            ctx.out.save_table(fano_frames[name], FACT, "count_dispersion", name,
                               caption="Fano factor of event counts by window length")

    if any(len(f) for f in fano_frames.values()):
        fig, axes = pl.new_figure(1, 2, width=5.8, height=4.4)
        for name, rd in ctx.each():
            sub = arrivals.loc[(arrivals.regime == name)]
            if len(sub):
                axes[0].bar(np.arange(len(sub)) + (0.2 if name == "stress" else -0.2),
                            sub["mean_interarrival_ms"], width=0.38, color=rd.color,
                            label=rd.label)
                axes[0].set_xticks(np.arange(len(sub)), sub["stream"], rotation=12)
            frame = fano_frames.get(name)
            if frame is not None and len(frame):
                one = frame.loc[frame.stream == "aggressive orders"]
                axes[1].plot(one["window_s"], one["fano"], "o-", ms=4.5, lw=1.7,
                             color=rd.color, label=rd.label)
        axes[0].set_yscale("log")
        pl.finish(axes[0], "Mean interarrival time by stream", "",
                  "milliseconds", legend=True)
        axes[1].axhline(1.0, **pl.REFERENCE_KW)
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        pl.finish(axes[1], "Aggressive-order count dispersion",
                  "window length (s)", "Var / mean", legend=True)
        fig.suptitle("Fact 19 - arrivals by regime", x=0.02, ha="left",
                     fontsize=12.5, fontweight="semibold")
        pl.layout(fig, rect=(0, 0, 1, 0.94))
        ctx.out.save_figure(fig, FACT, "arrivals", "comparison")

    return {
        "fact": FACT, "id": FACT_ID, "priority": PRIORITY, "title": TITLE,
        "metrics": metrics,
        "literature": "ZJ p3 summary statistics (BTC, 2026-08-25): 0.1 bps spread, "
                      "0.2868 BTC at the best quote, 185.1 ms mean interarrival. "
                      "Different venue window and event definition, so these are "
                      "reference points, not targets.",
    }
