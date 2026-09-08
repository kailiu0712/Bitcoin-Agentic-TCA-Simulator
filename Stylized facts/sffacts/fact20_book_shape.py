"""
Stylized fact 20 (P0-C) - book shape: tick spread, L1 depth, and the
depth profile away from the touch.

Source
    ``Stylied facts ZJ.pdf`` p5, p6 and p7.  ZJ present the spread as a
    histogram **in ticks** (1, 2, ... 9, >=10) with the one-tick share called
    out; the L1 depth as a log-x histogram of the quantity resting at the best
    quote; and a "volume profile" of mean level depth against that level's
    distance to the opposite best.

    ``fact06_spread`` already covers the spread, but only in bps on log-log
    axes, which hides exactly the quantity ZJ emphasise - how often the book is
    one tick wide.  ``fact05_liquidity_cost`` covers depth, but only as
    cumulative quantity inside bps bands, and never the top of book alone.

Time weighting
    Every distribution here is **time-weighted**: the quote tape carries a row
    whenever the best price *or* either best size changes, so the top of book is
    exactly piecewise-constant between rows and weighting a row by the time
    until the next one is an exact time average, not a sample.  The
    event-weighted version is reported alongside, because the two answer
    different questions - "what does the book look like at a random instant"
    versus "what did it look like at a random update" - and for a market whose
    update rate rises with activity they differ materially.

A stated substitution for ZJ p7
    ZJ's x-axis is a *level index* (L0...L9) placed at that level's distance to
    the opposite best.  Reconstructing it needs the quantity resting at each of
    the 100 published price levels.  The cached tapes carry cumulative depth in
    six bps bands and the top-of-book sizes, not per-level quantities, so the
    per-level curve cannot be recovered without extending the ingest kernel and
    re-reading 177M raw rows.

    What is plotted instead is the same question on the data that exists: the
    time-weighted mean incremental depth in each bps shell - [0,1], (1,2],
    (2,5], (5,10], (10,20], (20,50] bps from the mid - **divided by the shell's
    width in ticks**, against that shell's distance to the opposite best, built
    from the all-packet ``depth_sum`` accumulator.  Dividing by the width is
    what makes the curve readable: the shells are 1, 1, 3, 5, 10 and 30 bps
    wide, so raw shell totals rise and then fall mostly because of bandwidth.
    Depth per tick is depth per price level, which is the quantity ZJ's curve
    plots.

    It remains *not* a per-level curve.  Each point is an average over a band
    of levels rather than one level, the bands are geometrically rather than
    evenly spaced (hence log axes where ZJ use linear), and the innermost band
    mixes the touch with everything within 1 bp of it.  Shape and side
    asymmetry are comparable; a point-by-point reading against ZJ is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfcore import plotting as pl
from sffacts.fact19_arrivals import MAX_QUOTE_DWELL_S, _book_averages

FACT = "fact20"
FACT_ID = 20
PRIORITY = "P0-C"
TITLE = "Book shape: tick spread, L1 depth, depth profile"

#: Spread histogram support, in ticks, with everything above folded into ">=10".
MAX_SPREAD_TICKS = 10

#: Log-spaced bins for the L1 depth histogram.
DEPTH_HIST_BINS = 60

#: ZJ quote BTC depth in "lots" of 0.0001 BTC; kept so the axis is comparable.
LOT_BTC = 1e-4


def _weights(rd):
    """Dwell time of each quote-tape row, with feed gaps zeroed out."""
    ts = rd.quotes["recv_ns"].to_numpy(np.int64)
    dwell = np.diff(ts).astype(np.float64)
    weight = np.where((dwell >= 0) & (dwell <= MAX_QUOTE_DWELL_S * 1e9), dwell, 0.0)
    return weight, slice(0, len(ts) - 1)


def _tick_spread_table(rd, tick_size: float) -> pd.DataFrame:
    """P(spread = k ticks) for k = 1..9 and k >= 10, time- and event-weighted."""
    weight, head = _weights(rd)
    quotes = rd.quotes
    ticks = (quotes["ask_tick"].to_numpy(np.int64)
             - quotes["bid_tick"].to_numpy(np.int64))[head]
    valid = ticks > 0
    folded = np.clip(ticks, 1, MAX_SPREAD_TICKS)
    time_hist = np.bincount(folded[valid], weights=weight[valid],
                            minlength=MAX_SPREAD_TICKS + 1)[1:]
    event_hist = np.bincount(folded[valid], minlength=MAX_SPREAD_TICKS + 1)[1:]
    labels = [str(k) for k in range(1, MAX_SPREAD_TICKS)] + [f">={MAX_SPREAD_TICKS}"]
    mean_ticks = (float(np.sum(ticks[valid] * weight[valid]) / weight[valid].sum())
                  if weight[valid].sum() > 0 else np.nan)
    return pd.DataFrame({
        "spread_ticks": labels,
        "probability_time_weighted_pct": time_hist / time_hist.sum() * 100,
        "probability_event_weighted_pct": event_hist / event_hist.sum() * 100,
        "mean_spread_ticks_time_weighted": mean_ticks,
        # The last row folds everything at or above MAX_SPREAD_TICKS, so its
        # dollar figure is the lower edge of that bucket rather than its value.
        "spread_usd_lower_edge": [k * tick_size
                                  for k in range(1, MAX_SPREAD_TICKS + 1)],
    })


def _l1_depth_table(rd) -> pd.DataFrame:
    """Time-weighted log-binned distribution of the quantity at each best quote."""
    weight, head = _weights(rd)
    quotes = rd.quotes
    both = {"bid": quotes["bid_qty"].to_numpy(np.float64)[head],
            "ask": quotes["ask_qty"].to_numpy(np.float64)[head]}
    pooled = np.concatenate([v[v > 0] for v in both.values()])
    if pooled.size < 100:
        return pd.DataFrame()
    edges = np.logspace(np.log10(np.percentile(pooled, 0.05)),
                        np.log10(np.percentile(pooled, 99.95)), DEPTH_HIST_BINS + 1)
    frames = []
    for side, values in both.items():
        ok = (values > 0) & (weight > 0)
        counts, _ = np.histogram(values[ok], bins=edges, weights=weight[ok])
        share = counts / counts.sum() * 100 if counts.sum() > 0 else counts
        order = np.argsort(values[ok])
        sorted_v, sorted_w = values[ok][order], weight[ok][order]
        cumulative = np.cumsum(sorted_w) / sorted_w.sum()
        quantiles = np.interp([0.5, 0.9], cumulative, sorted_v)
        frames.append(pd.DataFrame({
            "side": side,
            "depth_btc_low": edges[:-1], "depth_btc_high": edges[1:],
            "depth_btc_centre": np.sqrt(edges[:-1] * edges[1:]),
            "probability_pct": share,
            "mean_btc": float(np.sum(sorted_v * sorted_w) / sorted_w.sum()),
            "median_btc": quantiles[0], "p90_btc": quantiles[1],
        }))
    return pd.concat(frames, ignore_index=True)


def _profile_table(rd, cfg) -> pd.DataFrame:
    """Time-weighted incremental depth per bps shell, placed in tick distance.

    Built from the all-packet ``depth_sum`` accumulator, so it is unconditional
    over the whole regime rather than conditioned on trade times.  A shell's
    representative offset is the geometric centre of its bps band; its distance
    to the *opposite* best adds the time-weighted half-spread.
    """
    acc = rd.accumulators
    total = float(acc["total_weight_ns"][0])
    if total <= 0:
        return pd.DataFrame()
    bands = np.asarray(cfg.DEPTH_BANDS_BPS, dtype=np.float64)
    cumulative = acc["depth_sum"] / total          # [band, side], side 0 = bid
    book = _book_averages(rd, cfg.TICK_SIZE)
    mid = book.get("average_price_usd", np.nan)
    half_spread_usd = 0.5 * book.get("average_spread_usd", np.nan)
    inner = np.concatenate([[0.0], bands[:-1]])
    # Geometric centre is the right summary on a band grid that is itself
    # geometric (1, 2, 5, 10, 20, 50 bps); the innermost shell starts at the
    # touch, so its centre is taken as half the outer edge.
    centre_bps = np.where(inner > 0, np.sqrt(np.maximum(inner, 1e-9) * bands),
                          bands / 2.0)
    offset_usd = centre_bps / 1e4 * mid
    rows = []
    for j, band in enumerate(bands):
        shell = cumulative[j] - (cumulative[j - 1] if j else 0.0)
        # The bands are 1, 1, 3, 5, 10 and 30 bps wide, so raw shell totals are
        # not comparable across shells and their shape is mostly bandwidth.
        # Dividing by the shell width in ticks gives depth per price level,
        # which is the quantity ZJ's per-level curve actually plots.
        width_ticks = (band - inner[j]) / 1e4 * mid / cfg.TICK_SIZE
        rows.append({
            "shell": f"{inner[j]:g}-{band:g} bps",
            "band_inner_bps": inner[j], "band_outer_bps": band,
            "centre_bps": centre_bps[j], "shell_width_ticks": width_ticks,
            "distance_to_opposite_best_ticks":
                (offset_usd[j] + half_spread_usd) / cfg.TICK_SIZE,
            "band_inner_distance_ticks":
                (inner[j] / 1e4 * mid + half_spread_usd) / cfg.TICK_SIZE,
            "shell_depth_bid_btc": shell[0], "shell_depth_ask_btc": shell[1],
            "depth_per_tick_bid_btc": shell[0] / width_ticks,
            "depth_per_tick_ask_btc": shell[1] / width_ticks,
            "cumulative_depth_bid_btc": cumulative[j, 0],
            "cumulative_depth_ask_btc": cumulative[j, 1],
        })
    table = pd.DataFrame(rows)
    # A shell whose INNER edge already holds >=95% of all visible depth is
    # measuring the end of the feed's 100-level publication window rather than
    # the book, so it is flagged instead of being silently plotted as liquidity.
    total = (table["cumulative_depth_bid_btc"].iloc[-1]
             + table["cumulative_depth_ask_btc"].iloc[-1])
    inner_cumulative = np.concatenate([[0.0], (table["cumulative_depth_bid_btc"]
                                               + table["cumulative_depth_ask_btc"])[:-1]])
    table["at_window_edge"] = inner_cumulative >= 0.95 * total if total > 0 else False
    return table


def _zj_l1_pooled(rd) -> tuple[np.ndarray, np.ndarray, dict]:
    """L1 depth pooled over both sides, time-weighted, as ZJ p6 present it.

    ZJ's p6 histogram is per *side*: their BTC mean of 1,433.8 lots is exactly
    the "Mean Size of Best Quote Orders" of their p3 table, i.e. half the total
    at the touch.  So bid and ask observations are pooled into one sample here
    rather than drawn as two curves, which is what ``_l1_depth_table`` does.
    """
    weight, head = _weights(rd)
    quotes = rd.quotes
    values = np.concatenate([quotes["bid_qty"].to_numpy(np.float64)[head],
                             quotes["ask_qty"].to_numpy(np.float64)[head]])
    weights = np.concatenate([weight, weight])
    ok = (values > 0) & (weights > 0)
    values, weights = values[ok], weights[ok]
    if values.size < 100:
        return np.array([]), np.array([]), {}
    edges = np.logspace(np.log10(np.percentile(values, 0.05)),
                        np.log10(np.percentile(values, 99.95)), DEPTH_HIST_BINS + 1)
    counts, _ = np.histogram(values, bins=edges, weights=weights)
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    median, p90 = np.interp([0.5, 0.9], cumulative, values[order])
    stats = {"mean_btc": float(np.sum(values * weights) / weights.sum()),
             "median_btc": float(median), "p90_btc": float(p90)}
    return edges, counts / counts.sum() * 100.0, stats


def _zj_style_figure(ctx, name, rd, profile) -> None:
    """A two-panel figure laid out like ZJ p6 and p7, on this dataset.

    Left  - L1 depth distribution: log-x histogram in lots with a mean/median/P90
            box, pooled across sides (ZJ p6).
    Right - depth per price level against distance to the opposite best, in
            lots, on linear axes, bid blue / ask red (ZJ p7).

    The right panel plots depth **per tick**, which is depth per price level and
    therefore the same quantity as ZJ's y-axis, but each point is an average
    over a bps band rather than one level - the tapes do not carry per-level
    quantities.  Bands whose inner edge already sits past 95% of the visible
    book are drawn hollow: those are the 100-level publication window running
    out, not the market thinning.
    """
    edges, share, stats = _zj_l1_pooled(rd)
    fig, axes = pl.new_figure(1, 2, width=6.2, height=4.6)

    ax = axes[0]
    if len(edges):
        # stairs() consumes the bin edges directly; bar() with a linear width
        # would misplace every bar on a log axis.
        ax.stairs(share, edges / LOT_BTC, fill=True, color=rd.color)
        ax.set_xscale("log")
        box = (f"Mean: {stats['mean_btc'] / LOT_BTC:,.1f} lots\n"
               f"Median: {stats['median_btc'] / LOT_BTC:,.1f} lots\n"
               f"P90: {stats['p90_btc'] / LOT_BTC:,.1f} lots")
        ax.text(0.03, 0.97, box, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                                      edgecolor="#D8DEE6"))
    pl.finish(ax, f"L1 depth distribution  (1 lot = {LOT_BTC:g} BTC)",
              "volume at best quote (lots) [log scale]", "probability of time (%)",
              note="Time-weighted and pooled across both sides, matching ZJ p6's "
                   "per-side convention. ZJ BTC (2026-08-25, tick 0.01): mean "
                   "1,433.8, median 578.2, P90 3,359.5 lots.")

    ax = axes[1]
    if len(profile):
        x = profile["distance_to_opposite_best_ticks"].to_numpy()
        edge = profile["at_window_edge"].to_numpy()
        tops = np.maximum(profile["depth_per_tick_bid_btc"].to_numpy(),
                          profile["depth_per_tick_ask_btc"].to_numpy()) / LOT_BTC
        if edge.any():
            ax.axvspan(profile.loc[edge, "band_inner_distance_ticks"].min(),
                       x.max() * 1.03, color="#9CA3AF", alpha=0.12, zorder=0)
        # Headroom so the staggered labels clear the panel title.
        ax.set_ylim(0, tops.max() * 1.22)
        for side, colour, label in (("bid", "#2C6FBB", "Bid depth"),
                                    ("ask", "#C0392B", "Ask depth")):
            y = profile[f"depth_per_tick_{side}_btc"].to_numpy() / LOT_BTC
            ax.plot(x, y, "-", lw=1.6, color=colour, label=label, zorder=2)
            ax.scatter(x[~edge], y[~edge], s=34, color=colour, zorder=3)
            if edge.any():
                ax.scatter(x[edge], y[edge], s=34, facecolors="none",
                           edgecolors=colour, zorder=3)
        # One grey label per shell, placed above the higher of the two sides and
        # staggered, so the tightly-spaced inner bands stay readable.
        for k, (xi, yi, tag) in enumerate(zip(x, tops, profile["shell"])):
            ax.annotate(tag.replace(" bps", ""), (xi, yi),
                        textcoords="offset points",
                        xytext=(0, 10 if k % 2 == 0 else 22),
                        fontsize=7.5, color="#4B5563", ha="center")
    pl.finish(ax, "Volume profile vs distance to opposite best",
              "distance to opposite best (ticks)",
              "mean depth per price level (lots)", legend=True,
              note="Hollow markers sit past 95% of the visible book - that is the "
                   "feed's 100-level window ending, not liquidity. Each point "
                   "averages a bps band, not a single level, so this is the "
                   "nearest measurable analogue of ZJ p7, not a reproduction.")
    fig.suptitle(f"Fact 20 - ZJ-style depth views | {rd.label}", x=0.02, ha="left",
                 fontsize=12.5, fontweight="semibold")
    pl.layout(fig)
    ctx.out.save_figure(fig, FACT, "zj_style_depth", name)


def run(ctx) -> dict:
    pl.apply_style()
    cfg = ctx.cfg
    metrics: list[dict] = []
    spreads, depths, profiles = {}, {}, {}

    for name, rd in ctx.each():
        spread = _tick_spread_table(rd, cfg.TICK_SIZE)
        depth = _l1_depth_table(rd)
        profile = _profile_table(rd, cfg)
        spreads[name], depths[name], profiles[name] = spread, depth, profile

        ctx.out.save_table(spread, FACT, "spread_in_ticks", name,
                           caption=f"Spread distribution in ticks - {rd.label}")
        if len(depth):
            ctx.out.save_table(depth, FACT, "l1_depth_distribution", name,
                               caption=f"Time-weighted L1 depth distribution - {rd.label}")
        if len(profile):
            ctx.out.save_table(profile, FACT, "depth_profile", name,
                               caption=f"Incremental depth per bps shell - {rd.label}")

        fig, axes = pl.new_figure(1, 3, width=5.2, height=4.2)
        ax = axes[0]
        positions = np.arange(len(spread))
        ax.bar(positions - 0.19, spread["probability_time_weighted_pct"], width=0.36,
               color=rd.color, label="time-weighted")
        ax.bar(positions + 0.19, spread["probability_event_weighted_pct"], width=0.36,
               color="#9CA3AF", label="event-weighted")
        ax.set_xticks(positions, spread["spread_ticks"])
        one_tick = spread["probability_time_weighted_pct"].iloc[0]
        mean_ticks = spread["mean_spread_ticks_time_weighted"].iloc[0]
        pl.finish(ax, "Spread distribution (ticks)",
                  f"spread (ticks of {cfg.TICK_SIZE:g} USD)", "probability (%)",
                  legend=True,
                  note=f"Mean {mean_ticks:.2f} ticks, 1-tick share "
                       f"{one_tick:.1f}% of time. ZJ p5 report 77.71 ticks / 81.4% "
                       "for BTC on a different venue-day, where the 1-tick mode "
                       "coexists with a fat >=10 tail.")

        ax = axes[1]
        if len(depth):
            for side, sub in depth.groupby("side", sort=False):
                ax.step(sub["depth_btc_centre"] / LOT_BTC, sub["probability_pct"],
                        where="mid", lw=1.6,
                        label=f"{side} (median {sub['median_btc'].iloc[0] / LOT_BTC:,.0f} lots)")
            head = depth.iloc[0]
            ax.set_xscale("log")
            pl.finish(ax, "L1 depth distribution",
                      f"volume at best quote (lots of {LOT_BTC:g} BTC)",
                      "probability (%)", legend=True,
                      note=f"Mean {head['mean_btc'] / LOT_BTC:,.0f} lots, "
                           f"P90 {head['p90_btc'] / LOT_BTC:,.0f} lots (bid side). "
                           "ZJ p6 BTC: mean 1,434, median 578, P90 3,360 lots.")

        ax = axes[2]
        if len(profile):
            ax.plot(profile["distance_to_opposite_best_ticks"],
                    profile["depth_per_tick_bid_btc"], "o-", ms=5, lw=1.7,
                    color="#2C6FBB", label="bid side")
            ax.plot(profile["distance_to_opposite_best_ticks"],
                    profile["depth_per_tick_ask_btc"], "s-", ms=5, lw=1.7,
                    color="#C0392B", label="ask side")
            for _, row in profile.iterrows():
                ax.annotate(row["shell"].replace(" bps", ""),
                            (row["distance_to_opposite_best_ticks"],
                             max(row["depth_per_tick_ask_btc"],
                                 row["depth_per_tick_bid_btc"])),
                            textcoords="offset points", xytext=(0, 8),
                            fontsize=7, color="#6B7280", ha="center")
            ax.set_xscale("log")
            ax.set_yscale("log")
            pl.finish(ax, "Depth profile away from the touch",
                      "distance to opposite best (ticks)",
                      "mean depth per tick (BTC)", legend=True,
                      note="Depth per tick, because the shells are bps bands of "
                           "unequal width and are NOT the L0-L9 levels of ZJ p7 "
                           "- per-level quantities are not carried on the tapes. "
                           "Log axes because the bands are geometrically spaced, "
                           "unlike ZJ's evenly spaced levels.")
        fig.suptitle(f"Fact 20 - book shape | {rd.label}", x=0.02, ha="left",
                     fontsize=12.5, fontweight="semibold")
        pl.layout(fig)
        ctx.out.save_figure(fig, FACT, "book_shape", name)

        # Same measurements, laid out the way ZJ p6/p7 present them.
        _zj_style_figure(ctx, name, rd, profile)

        metrics += [
            {"metric": "share of time the spread is one tick", "regime": name,
             "value": float(one_tick) / 100.0,
             "detail": f"mean {mean_ticks:.2f} ticks; ZJ p5 BTC: 81.4%"},
            {"metric": "median L1 depth, bid side (BTC)", "regime": name,
             "value": float(depth.loc[depth.side == "bid", "median_btc"].iloc[0])
                      if len(depth) else np.nan,
             "detail": "time-weighted over every top-of-book state"},
            {"metric": "ask/bid depth ratio in the 0-1 bps shell",
             "regime": name,
             "value": float(profile["shell_depth_ask_btc"].iloc[0]
                            / profile["shell_depth_bid_btc"].iloc[0])
                      if len(profile) else np.nan,
             "detail": ">1 means the offer side is the thicker one at the touch"},
        ]

    fig, axes = pl.new_figure(1, 2, width=5.8, height=4.4)
    width = 0.36
    for offset, (name, rd) in zip((-0.19, 0.19), ctx.each()):
        spread = spreads.get(name)
        if spread is not None and len(spread):
            positions = np.arange(len(spread))
            axes[0].bar(positions + offset, spread["probability_time_weighted_pct"],
                        width=width, color=rd.color, label=rd.label)
            axes[0].set_xticks(positions, spread["spread_ticks"])
        profile = profiles.get(name)
        if profile is not None and len(profile):
            axes[1].plot(profile["distance_to_opposite_best_ticks"],
                         profile["depth_per_tick_bid_btc"]
                         + profile["depth_per_tick_ask_btc"],
                         "o-", ms=5, lw=1.7, color=rd.color, label=rd.label)
            axes[1].set_xscale("log")
            axes[1].set_yscale("log")
    pl.finish(axes[0], "Spread in ticks by regime", "spread (ticks)",
              "probability of time (%)", legend=True)
    pl.finish(axes[1], "Total depth per tick by regime",
              "distance to opposite best (ticks)",
              "mean depth per tick, both sides (BTC)", legend=True)
    fig.suptitle("Fact 20 - book shape by regime", x=0.02, ha="left",
                 fontsize=12.5, fontweight="semibold")
    pl.layout(fig, rect=(0, 0, 1, 0.94))
    ctx.out.save_figure(fig, FACT, "book_shape", "comparison")

    return {
        "fact": FACT, "id": FACT_ID, "priority": PRIORITY, "title": TITLE,
        "metrics": metrics,
        "literature": "ZJ p5-p7 (BTC, 2026-08-25): 81.4% one-tick spread, L1 depth "
                      "median 578 lots, depth rising away from the touch after a "
                      "dip at L1. Different venue-day; shape, not level, compares.",
    }
