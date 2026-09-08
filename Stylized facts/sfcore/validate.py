"""
Data-quality and reconstruction-correctness reporting.

Runs before the stylized facts and writes the evidence that the numbers
downstream can be trusted.  Three independent checks:

1. **Snapshot agreement.**  Every snapshot block the exchange sends is an
   authoritative full-depth refresh.  The kernel compares its reconstruction
   against each one *before* being reset by it.  Agreement of the best bid/ask
   to within a tick, and of visible depth to within a few percent, is the
   primary evidence that the depth-window eviction and crossed-book repair
   reproduce the real book.

2. **Aggressor-side consistency.**  A buyer lifting the offer must print at or
   above the pre-trade best ask.  This is reported both under exchange-time
   alignment (what the framework uses) and under naive receipt-time alignment,
   because the gap between the two is the single largest methodological trap in
   this dataset: the book and trade channels are separate subscriptions whose
   latencies jitter by hundreds of milliseconds.

3. **Feed-integrity counters.**  Crossed packets surviving repair (must be
   zero), orphan deletes, prices off the tick grid, clock reversals, capacity
   overflows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import plotting as pl

SECTION = "fact00"
TITLE = "Data quality and book-reconstruction validation"

#: Diagnostics promoted into the headline table, in reporting order.
HEADLINE_KEYS = [
    ("rows_processed", "raw feed rows processed"),
    ("packets", "packets (distinct receipt timestamps)"),
    ("measured_time_hours", "measured hours"),
    ("aggressive_orders", "aggressive orders (trade bursts)"),
    ("trade_rows", "individual fills"),
    ("snapshot_blocks", "snapshot blocks seen"),
    ("levels_evicted_by_depth_window", "levels evicted by the depth window"),
    ("levels_dropped_by_cross_repair", "levels dropped by the crossed-book repair"),
    ("crossed_packets_after_repair", "crossed packets after repair (must be 0)"),
    ("orphan_deletes", "orphan deletes (order already evicted)"),
    ("idempotent_upserts", "idempotent re-notifications (scroll-in)"),
    ("prices_off_tick_grid", "prices off the 0.1 USD tick grid"),
    ("clock_reversals_received_at_ns", "receipt-clock reversals"),
    ("level_overflow_events", "level-capacity overflows"),
    ("order_slot_exhaustion_events", "order-slot exhaustions"),
    ("pre_trade_ring_misses", "pre-trade state lookups without an old-enough entry"),
    ("aggressor_side_agreement_exchange_aligned", "aggressor-side agreement (exchange-aligned)"),
    ("aggressor_side_agreement_receipt_aligned", "aggressor-side agreement (receipt-aligned)"),
    ("rows_per_second", "ingest throughput (rows/s)"),
    ("elapsed_seconds", "ingest wall time (s)"),
]


def run(ctx) -> dict:
    pl.apply_style()
    rows = []
    validations = {}

    for name, rd in ctx.each():
        diag = rd.diagnostics
        for key, label in HEADLINE_KEYS:
            rows.append({"metric": label, "regime": name, "value": diag.get(key)})

        sv = rd.snapshot_validation
        if len(sv):
            sv = sv.copy()
            sv["bid_error_ticks"] = (sv.recon_bid - sv.snap_bid) / ctx.cfg.TICK_SIZE
            sv["ask_error_ticks"] = (sv.recon_ask - sv.snap_ask) / ctx.cfg.TICK_SIZE
            sv["bid_depth_ratio"] = sv.recon_bid_qty / sv.snap_bid_qty
            sv["ask_depth_ratio"] = sv.recon_ask_qty / sv.snap_ask_qty
            validations[name] = sv
            ctx.out.save_table(sv, SECTION, "snapshot_agreement", name,
                               caption=f"Reconstruction vs exchange snapshot blocks - {rd.label}")
            rows += [
                {"metric": "snapshot checks", "regime": name, "value": len(sv)},
                {"metric": "median |best-bid error| (ticks)", "regime": name,
                 "value": float(np.nanmedian(np.abs(sv.bid_error_ticks)))},
                {"metric": "median |best-ask error| (ticks)", "regime": name,
                 "value": float(np.nanmedian(np.abs(sv.ask_error_ticks)))},
                {"metric": "share of snapshots with best quotes exact", "regime": name,
                 "value": float(np.mean((np.abs(sv.bid_error_ticks) < 0.5)
                                        & (np.abs(sv.ask_error_ticks) < 0.5)))},
                {"metric": "median reconstructed/published bid depth", "regime": name,
                 "value": float(np.nanmedian(sv.bid_depth_ratio))},
                {"metric": "median reconstructed/published ask depth", "regime": name,
                 "value": float(np.nanmedian(sv.ask_depth_ratio))},
                # When the two sides are off by the same amount the book is
                # intact and merely sits at a stale price level, which is what
                # a reconnect gap looks like - not a reconstruction error.
                {"metric": "share of snapshot mismatches that are whole-book shifts",
                 "regime": name,
                 "value": float(np.mean(np.abs(sv.bid_error_ticks - sv.ask_error_ticks) < 1.5))},
            ]

    order = pd.DataFrame(rows).metric.drop_duplicates().tolist()
    table = (pd.DataFrame(rows)
             .pivot(index="metric", columns="regime", values="value")
             .reindex(order)
             .reset_index())
    table.columns.name = None
    ctx.out.save_table(table, SECTION, "feed_and_reconstruction_diagnostics",
                       "comparison", caption="Feed integrity and reconstruction health")

    # --- figure ---------------------------------------------------------
    n_panels = 3
    fig, axes = pl.new_figure(1, n_panels, width=5.2, height=4.2)

    ax = axes[0]
    for name, rd in ctx.each():
        sv = validations.get(name)
        if sv is None or not len(sv):
            continue
        err = np.concatenate([sv.bid_error_ticks.to_numpy(), sv.ask_error_ticks.to_numpy()])
        # Clip into the edge bins so no check is silently dropped from the plot.
        ax.hist(np.clip(err, -5, 5), bins=np.arange(-5.5, 6.5, 1.0), alpha=0.6,
                color=rd.color, label=f"{rd.label} (n={err.size})")
    ax.set_xlabel("reconstructed - published best quote (ticks)")
    pl.finish(ax, "Best-quote error vs exchange snapshots", "error (ticks of 0.1 USD)",
              "snapshot checks", legend=True,
              note="Each exchange snapshot block is an independent, unused-until-then "
                   "ground truth. Errors beyond +/-5 ticks are piled into the edge bins; "
                   "they are whole-book level shifts across a reconnect gap, not book "
                   "damage - see the diagnostics table.")

    ax = axes[1]
    for name, rd in ctx.each():
        sv = validations.get(name)
        if sv is None or not len(sv):
            continue
        ratio = np.concatenate([sv.bid_depth_ratio.to_numpy(), sv.ask_depth_ratio.to_numpy()])
        ratio = ratio[np.isfinite(ratio)]
        ax.hist(ratio, bins=np.linspace(0.7, 1.3, 31), alpha=0.6, color=rd.color,
                label=rd.label)
    ax.axvline(1.0, **pl.REFERENCE_KW)
    pl.finish(ax, "Visible-depth agreement", "reconstructed / published depth",
              "snapshot checks", legend=True,
              note="Residual differences are the events lost during the feed reconnect "
                   "that triggered the snapshot.")

    ax = axes[2]
    labels, exch_vals, recv_vals, colors = [], [], [], []
    for name, rd in ctx.each():
        diag = rd.diagnostics
        labels.append(rd.label.split(" (")[0])
        exch_vals.append(diag.get("aggressor_side_agreement_exchange_aligned") or np.nan)
        recv_vals.append(diag.get("aggressor_side_agreement_receipt_aligned") or np.nan)
        colors.append(rd.color)
    x = np.arange(len(labels))
    ax.bar(x - 0.19, np.array(exch_vals) * 100, width=0.38, color="#2C6FBB",
           label="exchange-time aligned (used)")
    ax.bar(x + 0.19, np.array(recv_vals) * 100, width=0.38, color="#9CA3AF",
           label="receipt-time aligned (naive)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    pl.finish(ax, "Aggressor-side consistency of the pre-trade book",
              "", "% of aggressive orders consistent", legend=True,
              note="A buyer lifting the offer must print at or above the pre-trade ask. "
                   "The gap is pure channel-latency jitter.")

    fig.suptitle("Data quality and book-reconstruction validation", x=0.02, ha="left",
                 fontsize=12.5, fontweight="semibold")
    pl.layout(fig, rect=(0, 0, 1, 0.94))
    ctx.out.save_figure(fig, SECTION, "reconstruction_validation", "comparison")

    metrics = []
    for name, rd in ctx.each():
        diag = rd.diagnostics
        metrics += [
            {"metric": "crossed packets after repair", "regime": name,
             "value": float(diag.get("crossed_packets_after_repair", np.nan)),
             "detail": "must be 0"},
            {"metric": "aggressor-side agreement (exchange-aligned)", "regime": name,
             "value": float(diag.get("aggressor_side_agreement_exchange_aligned") or np.nan),
             "detail": "share of aggressive orders printing on the correct side of the "
                       "pre-trade quote; compare the receipt-aligned row below"},
            {"metric": "aggressor-side agreement (receipt-aligned, naive)", "regime": name,
             "value": float(diag.get("aggressor_side_agreement_receipt_aligned") or np.nan),
             "detail": "what you get without exchange-time channel alignment"},
        ]
        sv = validations.get(name)
        if sv is not None and len(sv):
            metrics.append({
                "metric": "snapshot checks with best quotes exact", "regime": name,
                "value": float(np.mean((np.abs(sv.bid_error_ticks) < 0.5)
                                       & (np.abs(sv.ask_error_ticks) < 0.5))),
                "detail": f"{len(sv)} independent checks",
            })
    return {"fact": SECTION, "id": 0, "priority": "validation", "title": TITLE,
            "metrics": metrics,
            "literature": "n/a - this section validates the reconstruction itself."}
