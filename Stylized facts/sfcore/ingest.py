"""
Stage 1: one streaming pass over the raw feed per regime.

Reconstructs the L3 book (see ``book.py`` for why that is non-trivial for this
feed) and writes the trade / quote / fill tapes plus the streaming book-state
accumulators and the feed diagnostics.

The pass is deliberately *complete*: every coherent event in the regime is
processed, and the unconditional book-state distributions are built from
time-weighted histograms over every packet rather than from a sample of the
book.  Nothing is sub-sampled to save time.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import book as bk
from . import reader as rd
from . import tapes as tp


def _ns_of_date(date: str) -> int:
    import pandas as pd

    return int(pd.Timestamp(date, tz="UTC").value)


def ingest_regime(cfg, regime: str) -> tp.TapePaths:
    """Run the full reconstruction for one regime and write its tapes."""
    spec = cfg.REGIMES[regime]
    paths = tp.TapePaths.for_regime(cfg.CACHE_DIR, regime)
    paths.trades.parent.mkdir(parents=True, exist_ok=True)

    files, first_measured = rd.regime_files(
        cfg.DATA_DIR, spec["start_date"], spec["end_date"], cfg.WARMUP_DAYS
    )
    emit_from_ns = _ns_of_date(first_measured)
    n_rows_total = rd.total_rows(files, cfg.SMOKE_TEST_ROWS_PER_DAY)

    sizes = tuple(cfg.VIRTUAL_ORDER_SIZES_BTC)
    bands = tuple(cfg.DEPTH_BANDS_BPS)
    n_sizes, n_bands = len(sizes), len(bands)

    state = bk.BookState(cfg)

    trade_cols = tp.trade_columns(bands, sizes)
    trade_dtypes = {"recv_ns": "int64", "exch_ns": "int64"}
    quote_dtypes = {"recv_ns": "int64", "exch_ns": "int64",
                    "bid_tick": "int32", "ask_tick": "int32",
                    "bid_qty": "float32", "ask_qty": "float32",
                    "bid_depth": "float32", "ask_depth": "float32"}
    fill_dtypes = {"recv_ns": "int64", "sign": "int8", "price": "float64",
                   "qty": "float64", "burst_idx": "int64"}

    w_trades = tp.TapeWriter(paths.trades, trade_cols, trade_dtypes)
    w_quotes = tp.TapeWriter(paths.quotes, tp.QUOTE_COLUMNS, quote_dtypes)
    w_fills = tp.TapeWriter(paths.fills, tp.FILL_COLUMNS, fill_dtypes)

    sv_rows: list[np.ndarray] = []

    # Log-histogram binning parameters.
    sp_lo, sp_hi = cfg.SPREAD_HIST_RANGE_BPS
    ct_lo, ct_hi = cfg.COST_HIST_RANGE_BPS
    sp_loglo = float(np.log(sp_lo))
    sp_invrange = 1.0 / (np.log(sp_hi) - sp_loglo)
    ct_loglo = float(np.log(ct_lo))
    ct_invrange = 1.0 / (np.log(ct_hi) - ct_loglo)

    ref_size_idx = int(np.argmin(np.abs(np.asarray(sizes) - 1.0)))

    started = time.perf_counter()
    bar = tqdm(
        total=n_rows_total,
        unit="ev",
        unit_scale=True,
        desc=f"[1/2] book+tapes {regime:<6}",
        dynamic_ncols=True,
        smoothing=0.05,
    )

    q_count = np.zeros(1, dtype=np.int64)
    b_count = np.zeros(1, dtype=np.int64)
    f_count = np.zeros(1, dtype=np.int64)
    sv_count = np.zeros(1, dtype=np.int64)

    for batch, path in rd.iter_batches(files, cfg.READ_BATCH_ROWS, cfg.TICK_SIZE,
                                       cfg.SMOKE_TEST_ROWS_PER_DAY):
        n = batch.n
        n_trade_rows = int(np.count_nonzero(batch.etype == rd.ET_TRADE))

        # Output buffers: one quote per packet at most, one burst/fill per
        # trade row at most.
        q_ts = np.zeros((n, 2), dtype=np.int64)
        q_val = np.zeros((n, 6), dtype=np.float32)
        cap_b = n_trade_rows + 8
        b_ts = np.zeros((cap_b, 2), dtype=np.int64)
        b_val = np.zeros((cap_b, tp.burst_matrix_columns(n_bands, n_sizes)), dtype=np.float64)
        f_ts = np.zeros(cap_b, dtype=np.int64)
        f_val = np.zeros((cap_b, 4), dtype=np.float64)
        sv = np.zeros((64, len(tp.SNAPSHOT_VALIDATION_COLUMNS)), dtype=np.float64)

        q_count[0] = 0
        b_count[0] = 0
        f_count[0] = 0
        sv_count[0] = 0

        bk.process_chunk(
            batch.recv_ns, batch.exch_ns, batch.etype, batch.side, batch.tick,
            batch.qty, batch.oid, n,
            state.lq, state.lt, state.lh, state.lv, state.ln,
            state.o_tick, state.o_qty, state.o_side, state.o_next, state.o_prev,
            state.o_key, state.h_key, state.h_val,
            state.S, state.F, state.prev_cost, state.prev_depth,
            state.cur_cost, state.cur_depth,
            state.rb_ts, state.rb_val,
            state.sizes, state.bands,
            q_ts, q_val, q_count,
            b_ts, b_val, b_count,
            f_ts, f_val, f_count,
            sv, sv_count,
            state.acc_spread_tw, state.acc_spread_ew, state.acc_cost_tw,
            state.acc_cost_stats, state.acc_depth_sum, state.acc_qi, state.acc_hour,
            emit_from_ns, cfg.BOOK_DEPTH_LEVELS, 1 if cfg.REPAIR_CROSSED_BOOK else 0,
            cfg.MIN_LEVELS_FOR_VALID_BOOK, cfg.MAX_LEVELS_PER_SIDE,
            cfg.TICK_SIZE, state.n_ticks, state.hmask, state.n_slots,
            sp_loglo, sp_invrange, ct_loglo, ct_invrange, cfg.HIST_BINS,
            cfg.QI_HIST_BINS, ref_size_idx, cfg.QUOTE_TAPE_DEPTH_BAND_INDEX,
            int(cfg.PRE_TRADE_GUARD_MS * 1e6),
        )

        nq, nb_, nf, nsv = (int(q_count[0]), int(b_count[0]), int(f_count[0]),
                            int(sv_count[0]))

        if nq:
            w_quotes.write({
                "recv_ns": q_ts[:nq, 0],
                "exch_ns": q_ts[:nq, 1],
                "bid_tick": q_val[:nq, 0].astype(np.int32),
                "ask_tick": q_val[:nq, 1].astype(np.int32),
                "bid_qty": q_val[:nq, 2],
                "ask_qty": q_val[:nq, 3],
                "bid_depth": q_val[:nq, 4],
                "ask_depth": q_val[:nq, 5],
            })
        if nb_:
            block = {"recv_ns": b_ts[:nb_, 0], "exch_ns": b_ts[:nb_, 1]}
            for j, name in enumerate(trade_cols[2:]):
                block[name] = b_val[:nb_, j]
            w_trades.write(block)
        if nf:
            w_fills.write({
                "recv_ns": f_ts[:nf],
                "sign": f_val[:nf, 0].astype(np.int8),
                "price": f_val[:nf, 1],
                "qty": f_val[:nf, 2],
                "burst_idx": f_val[:nf, 3].astype(np.int64),
            })
        if nsv:
            sv_rows.append(sv[:nsv].copy())

        bar.update(n)
        bar.set_postfix_str(
            f"{path.stem} | quotes {w_quotes.rows/1e6:.1f}M trades {w_trades.rows/1e3:.0f}k",
            refresh=False,
        )

    bar.close()
    w_trades.close()
    w_quotes.close()
    w_fills.close()

    if sv_rows:
        import pandas as pd

        sv_df = pd.DataFrame(np.vstack(sv_rows), columns=tp.SNAPSHOT_VALIDATION_COLUMNS)
        sv_df.to_parquet(paths.snapshot_validation, index=False)

    elapsed = time.perf_counter() - started
    diagnostics = _collect_diagnostics(cfg, state, regime, spec, files, elapsed,
                                       w_trades.rows, w_quotes.rows, w_fills.rows)
    with open(paths.diagnostics, "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)

    np.savez_compressed(
        paths.accumulators,
        spread_tw=state.acc_spread_tw,
        spread_ew=state.acc_spread_ew,
        cost_tw=state.acc_cost_tw,
        cost_stats=state.acc_cost_stats,
        depth_sum=state.acc_depth_sum,
        qi=state.acc_qi,
        hour=state.acc_hour,
        sizes=np.asarray(sizes),
        bands=np.asarray(bands),
        total_weight_ns=np.asarray([state.F[bk.F_TOTAL_W]]),
        spread_hist_range=np.asarray(cfg.SPREAD_HIST_RANGE_BPS),
        cost_hist_range=np.asarray(cfg.COST_HIST_RANGE_BPS),
        hist_bins=np.asarray([cfg.HIST_BINS]),
        qi_bins=np.asarray([cfg.QI_HIST_BINS]),
        ref_size_idx=np.asarray([ref_size_idx]),
    )
    return paths


def _collect_diagnostics(cfg, state, regime, spec, files, elapsed, n_trades,
                         n_quotes, n_fills) -> dict:
    """Feed-integrity and reconstruction-health counters for the report."""
    S = state.S
    n_rows = int(S[bk.S_N_ROWS])
    n_upserts = max(int(S[bk.S_IDEMPOTENT]), 0)
    tested = int(S[bk.S_SIDE_TESTED])
    return {
        "regime": regime,
        "label": spec["label"],
        "start_date": spec["start_date"],
        "end_date": spec["end_date"],
        "warmup_days": cfg.WARMUP_DAYS,
        "files": [f.name for f in files],
        "elapsed_seconds": round(elapsed, 1),
        "rows_processed": n_rows,
        "rows_per_second": round(n_rows / elapsed, 0) if elapsed > 0 else None,
        "packets": int(S[bk.S_N_PACKETS]),
        "packets_measured": int(S[bk.S_EMITTED_PACKETS]),
        "trade_rows": int(S[bk.S_TRADE_ROWS]),
        "aggressive_orders": int(S[bk.S_BURST]),
        "tape_rows": {"trades": n_trades, "quotes": n_quotes, "fills": n_fills},
        "snapshot_blocks": int(S[bk.S_SNAP_BLOCKS]),
        "snapshot_rows": int(S[bk.S_SNAP_ROWS]),
        "levels_evicted_by_depth_window": int(S[bk.S_EVICTED]),
        "levels_dropped_by_cross_repair": int(S[bk.S_REPAIRS]),
        "crossed_packets_after_repair": int(S[bk.S_CROSSED]),
        "orphan_deletes": int(S[bk.S_ORPHAN_DELETE]),
        "idempotent_upserts": n_upserts,
        "bad_rows": int(S[bk.S_BAD_ROWS]),
        "prices_off_tick_grid": int(S[bk.S_OFF_GRID]),
        "clock_reversals_received_at_ns": int(S[bk.S_RECV_REVERSALS]),
        "level_overflow_events": int(S[bk.S_LEVEL_OVERFLOW]),
        "order_slot_exhaustion_events": int(S[bk.S_HASH_FULL]),
        "aggressor_side_checks": tested,
        "aggressor_side_agreement_exchange_aligned": (
            round(S[bk.S_SIDE_AGREE] / tested, 5) if tested else None),
        "aggressor_side_agreement_receipt_aligned": (
            round(S[bk.S_SIDE_AGREE_RECV] / tested, 5) if tested else None),
        "pre_trade_ring_misses": int(S[bk.S_RING_MISS]),
        "exchange_clock_reversals": int(S[bk.S_EXCH_REVERSALS]),
        "measured_time_hours": round(float(state.F[bk.F_TOTAL_W]) / 3.6e12, 3),
        "config": {
            "book_depth_levels": cfg.BOOK_DEPTH_LEVELS,
            "repair_crossed_book": cfg.REPAIR_CROSSED_BOOK,
            "apply_implied_fills": cfg.APPLY_IMPLIED_FILLS,
            "tick_size": cfg.TICK_SIZE,
            "min_levels_for_valid_book": cfg.MIN_LEVELS_FOR_VALID_BOOK,
        },
    }
