"""
L3 order-book reconstruction for the Kraken BTC/USD feed, and the single
streaming pass that turns 160M raw events into compact analysis tapes.

Why this is not a naive "apply add/delete" book
-----------------------------------------------
The feed is a **depth-limited L3 book: the top 100 price levels per side**
(every snapshot block contains exactly 100 distinct levels on each side).  Three
consequences drive the whole design, and each was established empirically
against the raw data:

1.  **Silent scroll-out.**  When an order falls out of the 100-level window the
    feed stops reporting it and sends no ``delete``.  Never-deleted orders sit
    at a median of -15.5 bps from the last trade price, i.e. exactly at the
    window edge, while deleted orders sit at -2.4 bps.  A book that only
    applies explicit deletes therefore grows without bound - it reaches >3,000
    levels per side and $500 of crossing within 1M rows.  The reconstruction
    must evict the worst levels itself (``BOOK_DEPTH_LEVELS``).

2.  **Intra-packet crossing is normal.**  A marketable order is published as an
    aggressive ``add`` *followed* by the ``delete``s of the resting orders it
    swept, all inside one ``received_at_ns`` packet.  Book state is therefore
    only ever read at packet boundaries; reading mid-packet reports a crossed
    book ~80% of the time.

3.  **Residual ghosts still occur.**  An order can be cancelled or filled while
    outside the window (so no ``delete`` is ever sent) and the price can later
    move back through it.  One such ghost pins the window and corrupts the book
    for hours.  The repair is to drop the least-recently-updated touch level
    whenever the book is crossed or locked at a packet boundary; it fires only
    a handful of times per million packets but is essential.

Fills need no special handling: a fully consumed maker order produces a
``delete`` and a partially consumed one a ``modify`` with reduced quantity, so
subtracting traded quantity as well would double-count (``APPLY_IMPLIED_FILLS``).

Correctness evidence
--------------------
Every snapshot block in the stream is a free out-of-sample test: the
reconstruction is compared against it *before* being reset by it, and the
comparison is written to ``book_validation__<regime>.csv``.  Reconstructed and
published best bid/ask agree to within a tick or two and visible depth to within
a few percent, the residual being the events lost during the feed reconnect that
triggered the snapshot.

Emitted tapes
-------------
``quotes``  one row per packet in which the top of book changed - the backbone
            for order-flow imbalance, queue imbalance, price diffusion and
            volatility clustering.
``trades``  one row per *aggressive order* (a burst of fills sharing one packet
            and one side), carrying the full pre-trade book state: best quotes,
            cumulative depth in price bands, and the virtual-market-order cost
            curve.
``fills``   one row per individual fill, for trade-size distributions.

Alongside the tapes the kernel maintains streaming, time-weighted histograms of
spread, depth and liquidity cost over **every** packet, so the unconditional
book-state distributions use all coherent events rather than a sample of them.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# --------------------------------------------------------------------------
# Scalar-state slot indices (int64 array ``S``)
# --------------------------------------------------------------------------
S_PK = 0                 # packet counter
S_LAST_RECV = 1          # recv_ns of the packet currently being filled
S_IN_SNAPSHOT = 2        # inside a contiguous snapshot block
S_N_ORDERS = 3           # live orders
S_FREE_TOP = 4           # head of the order-slot free list
S_BURST = 5              # global aggressive-order counter
S_N_PACKETS = 6
S_N_ROWS = 7
S_ORPHAN_DELETE = 8      # delete for an order we do not hold (evicted earlier)
S_IDEMPOTENT = 9         # re-notification of an unchanged order (scroll-in)
S_EVICTED = 10           # levels dropped by the depth-window rule
S_REPAIRS = 11           # levels dropped by the crossed-book repair
S_SNAP_BLOCKS = 12
S_BAD_ROWS = 13
S_PK_LAST_SNAP = 14
S_MIXED_PACKETS = 15     # packets carrying both trade and book rows
S_TRADE_ROWS = 16
S_CROSSED = 17           # crossed/locked packets surviving repair (want 0)
S_OFF_GRID = 18          # prices outside the pre-allocated tick space
S_SIDE_AGREE = 19        # aggressor-side checks passed
S_SIDE_TESTED = 20
S_RECV_REVERSALS = 21
S_EXCH_REVERSALS = 22
S_LAST_EXCH = 23
S_LEVEL_OVERFLOW = 24
S_SNAP_ROWS = 25
S_HASH_FULL = 26
S_PREV_BT = 27           # previous packet-close best bid tick
S_PREV_AT = 28
S_PREV_LVLB = 29
S_PREV_LVLA = 30
S_EMITTED_PACKETS = 31
S_PKT_EXCH = 32          # running-max exchange time of applied book rows
S_RB_HEAD = 33           # next write slot in the pre-trade state ring
S_RB_COUNT = 34
S_SIDE_AGREE_RECV = 35   # aggressor check under naive receipt-time alignment
S_RING_MISS = 36         # bursts with no ring entry old enough
S_LEN = 40

# --------------------------------------------------------------------------
# Scalar-state slot indices (float64 array ``F``)
# --------------------------------------------------------------------------
F_PREV_SPREAD = 0        # bps
F_PREV_MID = 1           # USD
F_PREV_QI = 2            # top-of-book queue imbalance
F_PREV_VALID = 3
F_PREV_BQ = 4            # best bid size
F_PREV_AQ = 5
F_TOTAL_W = 6            # total accumulated time weight (ns)
F_LAST_TS = 7            # packet time at which the state in ``prev_*`` began
F_LEN = 16

# --------------------------------------------------------------------------
# Burst-tape column layout (float64 matrix ``b_val``)
# --------------------------------------------------------------------------
B_SIGN = 0
B_NFILL = 1
B_QTY = 2
B_NOTIONAL = 3
B_VWAP = 4
B_PFIRST = 5
B_PLAST = 6
B_BID = 7                # pre-trade best bid (USD)
B_ASK = 8
B_BIDQ = 9
B_ASKQ = 10
B_SPREAD = 11            # bps
B_MID = 12
B_LVLB = 13              # levels held on the bid side
B_LVLA = 14
B_PKSNAP = 15            # packets since the last snapshot reset
B_FIXED = 16             # depth block starts here, then the cost block

# --------------------------------------------------------------------------
# Pre-trade state ring buffer.
#
# The book and trade channels are separate exchange subscriptions whose
# collector latencies jitter between ~50 ms and ~400 ms, so a trade row
# routinely arrives *after* the book updates it caused.  Reading the book at
# the trade's own receipt time therefore gives a post-trade book 21% of the
# time (and 37% of the time for the largest orders).  The fix is to align the
# two channels on *exchange* time: every packet's closing state is pushed into
# this ring keyed by exchange timestamp, and an aggressive order takes the
# newest state whose exchange time precedes its own.  That lifts the
# aggressor-side consistency check from 0.79 to ~0.96, the residual being
# genuine hidden-liquidity and mid-point executions.
# --------------------------------------------------------------------------
RING_SIZE = 4096
R_BID = 0
R_ASK = 1
R_BIDQ = 2
R_ASKQ = 3
R_SPREAD = 4
R_MID = 5
R_LVLB = 6
R_LVLA = 7
R_PKSNAP = 8
R_FIXED = 9              # depth block, then cost block

# Event-type codes (mirror of reader.py, repeated so the kernel is standalone)
ET_ADD = 0
ET_MODIFY = 1
ET_DELETE = 2
ET_TRADE = 3
ET_SNAPSHOT = 4

BID = 0
ASK = 1


# ==========================================================================
# Hash table: order-id hash -> order slot.  Open addressing with linear
# probing and backward-shift deletion, so no tombstones accumulate across the
# tens of millions of insert/erase pairs in a week of feed.
# ==========================================================================


@njit(cache=True, inline="always")
def _h_find(h_key, key, mask):
    i = key & mask
    while True:
        k = h_key[i]
        if k == 0:
            return -1
        if k == key:
            return i
        i = (i + 1) & mask


@njit(cache=True, inline="always")
def _h_insert(h_key, h_val, key, val, mask):
    i = key & mask
    while h_key[i] != 0:
        if h_key[i] == key:
            h_val[i] = val
            return i
        i = (i + 1) & mask
    h_key[i] = key
    h_val[i] = val
    return i


@njit(cache=True, inline="always")
def _h_erase(h_key, h_val, i, mask):
    """Backward-shift deletion (Knuth 6.4 algorithm R)."""
    h_key[i] = 0
    j = i
    while True:
        j = (j + 1) & mask
        if h_key[j] == 0:
            break
        k = h_key[j] & mask
        # Skip entries that are still reachable from their ideal slot.
        if i <= j:
            if i < k and k <= j:
                continue
        else:
            if k > i or k <= j:
                continue
        h_key[i] = h_key[j]
        h_val[i] = h_val[j]
        h_key[j] = 0
        i = j


# ==========================================================================
# Sorted level index.  Each side keeps its occupied ticks in one ascending
# int32 array; with <=100 levels per side the array fits in L1 and a binary
# search plus memmove beats any tree or heap.
# ==========================================================================


@njit(cache=True, inline="always")
def _lv_search(lv, ln, sd, t):
    """Index of ``t`` in the sorted level array, or the insertion point."""
    lo = 0
    hi = ln[sd]
    while lo < hi:
        mid = (lo + hi) >> 1
        if lv[sd, mid] < t:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True, inline="always")
def _lv_insert(lv, ln, sd, t, S, max_levels):
    pos = _lv_search(lv, ln, sd, t)
    n = ln[sd]
    if n >= max_levels:
        S[S_LEVEL_OVERFLOW] += 1
        return
    for k in range(n, pos, -1):
        lv[sd, k] = lv[sd, k - 1]
    lv[sd, pos] = t
    ln[sd] = n + 1


@njit(cache=True, inline="always")
def _lv_remove(lv, ln, sd, t):
    pos = _lv_search(lv, ln, sd, t)
    n = ln[sd]
    if pos >= n or lv[sd, pos] != t:
        return
    for k in range(pos, n - 1):
        lv[sd, k] = lv[sd, k + 1]
    ln[sd] = n - 1


# ==========================================================================
# Order lifecycle
# ==========================================================================


@njit(cache=True, inline="always")
def _order_insert(lq, lt, lh, lv, ln, o_tick, o_qty, o_side, o_next, o_prev, o_key,
                  h_key, h_val, hmask, S, key, sd, t, q, pk, max_levels):
    slot = S[S_FREE_TOP]
    if slot < 0:
        S[S_HASH_FULL] += 1
        return
    S[S_FREE_TOP] = o_next[slot]

    o_side[slot] = sd
    o_tick[slot] = t
    o_qty[slot] = q
    o_key[slot] = key

    head = lh[sd, t]
    o_next[slot] = head
    o_prev[slot] = -1
    if head >= 0:
        o_prev[head] = slot
    else:
        _lv_insert(lv, ln, sd, t, S, max_levels)
        lq[sd, t] = 0.0
    lh[sd, t] = slot
    lq[sd, t] += q
    lt[sd, t] = pk

    _h_insert(h_key, h_val, key, slot, hmask)
    S[S_N_ORDERS] += 1


@njit(cache=True, inline="always")
def _order_unlink(lq, lh, lv, ln, o_tick, o_qty, o_side, o_next, o_prev, slot):
    """Detach ``slot`` from its price level (does not touch the hash table)."""
    sd = o_side[slot]
    t = o_tick[slot]
    nxt = o_next[slot]
    prv = o_prev[slot]
    if prv >= 0:
        o_next[prv] = nxt
    else:
        lh[sd, t] = nxt
    if nxt >= 0:
        o_prev[nxt] = prv
    lq[sd, t] -= o_qty[slot]
    if lh[sd, t] < 0:
        lq[sd, t] = 0.0
        _lv_remove(lv, ln, sd, t)
    elif lq[sd, t] < 0.0:
        lq[sd, t] = 0.0


@njit(cache=True, inline="always")
def _order_remove(lq, lh, lv, ln, o_tick, o_qty, o_side, o_next, o_prev, o_key,
                  h_key, h_val, hmask, S, slot, hidx):
    _order_unlink(lq, lh, lv, ln, o_tick, o_qty, o_side, o_next, o_prev, slot)
    _h_erase(h_key, h_val, hidx, hmask)
    o_next[slot] = S[S_FREE_TOP]
    S[S_FREE_TOP] = slot
    S[S_N_ORDERS] -= 1


@njit(cache=True, inline="always")
def _drop_level(lq, lt, lh, lv, ln, o_tick, o_qty, o_side, o_next, o_prev, o_key,
                h_key, h_val, hmask, S, sd, t):
    """Remove a whole price level and all orders resting on it."""
    slot = lh[sd, t]
    while slot >= 0:
        nxt = o_next[slot]
        hidx = _h_find(h_key, o_key[slot], hmask)
        if hidx >= 0:
            _h_erase(h_key, h_val, hidx, hmask)
        o_next[slot] = S[S_FREE_TOP]
        S[S_FREE_TOP] = slot
        S[S_N_ORDERS] -= 1
        slot = nxt
    lh[sd, t] = -1
    lq[sd, t] = 0.0
    lt[sd, t] = 0
    _lv_remove(lv, ln, sd, t)


@njit(cache=True)
def _reset_book(lq, lt, lh, lv, ln, o_tick, o_qty, o_side, o_next, o_prev, o_key,
                h_key, h_val, hmask, S, n_slots):
    """Clear the book (used when an authoritative snapshot block arrives)."""
    for sd in range(2):
        for k in range(ln[sd]):
            t = lv[sd, k]
            lh[sd, t] = -1
            lq[sd, t] = 0.0
            lt[sd, t] = 0
        ln[sd] = 0
    for i in range(h_key.shape[0]):
        h_key[i] = 0
    for i in range(n_slots - 1):
        o_next[i] = i + 1
    o_next[n_slots - 1] = -1
    S[S_FREE_TOP] = 0
    S[S_N_ORDERS] = 0


# ==========================================================================
# Book-state measurement
# ==========================================================================


@njit(cache=True, inline="always")
def _walk_side(lq, lv, ln, sd, mid, tick_size, sizes, bands, cost_out, depth_out, col):
    """Walk one side outward from the touch, filling the cost and depth grids.

    ``cost_out[i, col]``  signed cost in bps of a virtual market order of
                          ``sizes[i]`` BTC, relative to the mid, or NaN when the
                          visible book cannot absorb it.
    ``depth_out[j, col]`` cumulative visible quantity within ``bands[j]`` bps
                          of the mid.

    ``sizes`` and ``bands`` must both be ascending; the walk visits levels in
    increasing distance from the mid so both grids fill with a single pointer.
    """
    n = ln[sd]
    ns = sizes.shape[0]
    nb = bands.shape[0]
    cum_q = 0.0
    cum_n = 0.0
    si = 0
    bi = 0
    for k in range(n):
        idx = k if sd == ASK else n - 1 - k
        t = lv[sd, idx]
        q = lq[sd, t]
        if q <= 0.0:
            continue
        px = t * tick_size
        if sd == ASK:
            off_bps = (px - mid) / mid * 1.0e4
        else:
            off_bps = (mid - px) / mid * 1.0e4
        while bi < nb and off_bps > bands[bi]:
            depth_out[bi, col] = cum_q
            bi += 1
        while si < ns and cum_q + q >= sizes[si]:
            need = sizes[si] - cum_q
            vwap = (cum_n + need * px) / sizes[si]
            if sd == ASK:
                cost_out[si, col] = (vwap - mid) / mid * 1.0e4
            else:
                cost_out[si, col] = (mid - vwap) / mid * 1.0e4
            si += 1
        cum_q += q
        cum_n += q * px
        if si >= ns and bi >= nb:
            break
    while bi < nb:
        depth_out[bi, col] = cum_q
        bi += 1
    while si < ns:
        cost_out[si, col] = np.nan
        si += 1


@njit(cache=True, inline="always")
def _log_bin(x, log_lo, inv_range, nbins):
    if not (x > 0.0):
        return 0
    b = int((np.log(x) - log_lo) * inv_range * nbins)
    if b < 0:
        return 0
    if b >= nbins:
        return nbins - 1
    return b


# ==========================================================================
# The streaming pass
# ==========================================================================


@njit(cache=True, nogil=True)
def process_chunk(
    # ---- input rows -------------------------------------------------------
    recv, exch, etype, side, tick, qty, oid, nrows,
    # ---- book state -------------------------------------------------------
    lq, lt, lh, lv, ln,
    o_tick, o_qty, o_side, o_next, o_prev, o_key,
    h_key, h_val,
    S, F, prev_cost, prev_depth, cur_cost, cur_depth,
    # ---- pre-trade state ring (exchange-time keyed) ------------------------
    rb_ts, rb_val,
    # ---- measurement grids ------------------------------------------------
    sizes, bands,
    # ---- quote tape -------------------------------------------------------
    q_ts, q_val, q_count,
    # ---- burst tape -------------------------------------------------------
    b_ts, b_val, b_count,
    # ---- fill tape --------------------------------------------------------
    f_ts, f_val, f_count,
    # ---- snapshot validation ---------------------------------------------
    sv, sv_count,
    # ---- streaming accumulators ------------------------------------------
    acc_spread_tw, acc_spread_ew, acc_cost_tw, acc_cost_stats,
    acc_depth_sum, acc_qi, acc_hour,
    # ---- parameters -------------------------------------------------------
    emit_from_ns, depth_levels, repair_enabled, min_levels_valid, max_levels,
    tick_size, n_ticks, hmask, n_slots,
    sp_loglo, sp_invrange, ct_loglo, ct_invrange, nbins, qbins, ref_size_idx,
    qdepth_idx, guard_ns,
):
    """Consume one decoded batch, updating the book and appending tape rows.

    Returns the number of quote / burst / fill / snapshot-validation rows
    appended (via the ``*_count`` single-element arrays, so the caller can slice
    the output buffers).
    """
    ns = sizes.shape[0]
    nb = bands.shape[0]
    nq = q_count[0]
    nbst = b_count[0]
    nf = f_count[0]
    nsv = sv_count[0]

    i = 0
    while i < nrows:
        r = recv[i]

        # ---------------- packet boundary --------------------------------
        if r != S[S_LAST_RECV]:
            if S[S_LAST_RECV] >= 0 and S[S_IN_SNAPSHOT] == 0:
                # --- close the packet that just ended --------------------
                if repair_enabled == 1:
                    while ln[BID] > 0 and ln[ASK] > 0:
                        bt = lv[BID, ln[BID] - 1]
                        at = lv[ASK, 0]
                        if bt < at:
                            break
                        if lt[BID, bt] <= lt[ASK, at]:
                            _drop_level(lq, lt, lh, lv, ln, o_tick, o_qty, o_side,
                                        o_next, o_prev, o_key, h_key, h_val, hmask,
                                        S, BID, bt)
                        else:
                            _drop_level(lq, lt, lh, lv, ln, o_tick, o_qty, o_side,
                                        o_next, o_prev, o_key, h_key, h_val, hmask,
                                        S, ASK, at)
                        S[S_REPAIRS] += 1

                while ln[BID] > depth_levels:
                    _drop_level(lq, lt, lh, lv, ln, o_tick, o_qty, o_side, o_next,
                                o_prev, o_key, h_key, h_val, hmask, S, BID, lv[BID, 0])
                    S[S_EVICTED] += 1
                while ln[ASK] > depth_levels:
                    _drop_level(lq, lt, lh, lv, ln, o_tick, o_qty, o_side, o_next,
                                o_prev, o_key, h_key, h_val, hmask, S, ASK,
                                lv[ASK, ln[ASK] - 1])
                    S[S_EVICTED] += 1

                S[S_N_PACKETS] += 1

                # --- accumulate the previous state over the time it held ---
                # ``prev_*`` was computed at the close of the packet before this
                # one, i.e. it was the book from F_LAST_TS until S_LAST_RECV.
                emit = 1 if (F[F_LAST_TS] >= emit_from_ns) else 0
                if F[F_PREV_VALID] > 0.5 and emit == 1:
                    dt = float(S[S_LAST_RECV]) - F[F_LAST_TS]
                    if dt < 0.0:
                        dt = 0.0
                    F[F_TOTAL_W] += dt
                    hour = int((int(F[F_LAST_TS]) // 3600000000000) % 24)

                    sb = _log_bin(F[F_PREV_SPREAD], sp_loglo, sp_invrange, nbins)
                    acc_spread_tw[sb] += dt
                    acc_spread_ew[sb] += 1.0

                    qb = int((F[F_PREV_QI] + 1.0) * 0.5 * qbins)
                    if qb < 0:
                        qb = 0
                    if qb >= qbins:
                        qb = qbins - 1
                    acc_qi[qb] += dt

                    for a in range(ns):
                        for c in range(2):
                            v = prev_cost[a, c]
                            acc_cost_stats[3, a, c] += dt
                            if v == v:  # not NaN -> book can absorb the size
                                cb = _log_bin(v, ct_loglo, ct_invrange, nbins)
                                acc_cost_tw[a, c, cb] += dt
                                acc_cost_stats[0, a, c] += v * dt
                                acc_cost_stats[1, a, c] += dt
                                acc_cost_stats[2, a, c] += v * v * dt
                    for a in range(nb):
                        for c in range(2):
                            acc_depth_sum[a, c] += prev_depth[a, c] * dt

                    acc_hour[0, hour] += dt
                    acc_hour[1, hour] += F[F_PREV_SPREAD] * dt
                    acc_hour[2, hour] += prev_depth[qdepth_idx, 0] * dt
                    acc_hour[3, hour] += prev_depth[qdepth_idx, 1] * dt
                    rc = prev_cost[ref_size_idx, 1]
                    if rc == rc:
                        acc_hour[4, hour] += rc * dt
                        acc_hour[5, hour] += dt
                    acc_hour[6, hour] += 1.0
                    S[S_EMITTED_PACKETS] += 1

                # --- recompute state at the close of this packet ----------
                valid = 0
                bid_t = -1
                ask_t = -1
                mid = 0.0
                spread_bps = 0.0
                qi = 0.0
                bq = 0.0
                aq = 0.0
                if ln[BID] >= min_levels_valid and ln[ASK] >= min_levels_valid:
                    bid_t = lv[BID, ln[BID] - 1]
                    ask_t = lv[ASK, 0]
                    if bid_t < ask_t:
                        valid = 1
                        bq = lq[BID, bid_t]
                        aq = lq[ASK, ask_t]
                        bp = bid_t * tick_size
                        ap = ask_t * tick_size
                        mid = 0.5 * (bp + ap)
                        spread_bps = (ap - bp) / mid * 1.0e4
                        den = bq + aq
                        qi = (bq - aq) / den if den > 0.0 else 0.0
                    else:
                        S[S_CROSSED] += 1

                if valid == 1:
                    _walk_side(lq, lv, ln, BID, mid, tick_size, sizes, bands,
                               cur_cost, cur_depth, 0)
                    _walk_side(lq, lv, ln, ASK, mid, tick_size, sizes, bands,
                               cur_cost, cur_depth, 1)
                else:
                    for a in range(ns):
                        cur_cost[a, 0] = np.nan
                        cur_cost[a, 1] = np.nan
                    for a in range(nb):
                        cur_depth[a, 0] = np.nan
                        cur_depth[a, 1] = np.nan

                # --- quote tape: emit when the top of book changed --------
                if valid == 1 and S[S_LAST_RECV] >= emit_from_ns:
                    changed = (
                        bid_t != S[S_PREV_BT]
                        or ask_t != S[S_PREV_AT]
                        or bq != F[F_PREV_BQ]
                        or aq != F[F_PREV_AQ]
                    )
                    if changed and nq < q_ts.shape[0]:
                        q_ts[nq, 0] = S[S_LAST_RECV]
                        q_ts[nq, 1] = S[S_PKT_EXCH]
                        q_val[nq, 0] = bid_t
                        q_val[nq, 1] = ask_t
                        q_val[nq, 2] = bq
                        q_val[nq, 3] = aq
                        q_val[nq, 4] = cur_depth[qdepth_idx, 0]
                        q_val[nq, 5] = cur_depth[qdepth_idx, 1]
                        nq += 1

                # --- push the closing state into the exchange-time ring ----
                if valid == 1:
                    slot = S[S_RB_HEAD]
                    rb_ts[slot] = S[S_PKT_EXCH]
                    rb_val[slot, R_BID] = bid_t * tick_size
                    rb_val[slot, R_ASK] = ask_t * tick_size
                    rb_val[slot, R_BIDQ] = bq
                    rb_val[slot, R_ASKQ] = aq
                    rb_val[slot, R_SPREAD] = spread_bps
                    rb_val[slot, R_MID] = mid
                    rb_val[slot, R_LVLB] = float(ln[BID])
                    rb_val[slot, R_LVLA] = float(ln[ASK])
                    rb_val[slot, R_PKSNAP] = float(S[S_PK] - S[S_PK_LAST_SNAP])
                    for a in range(nb):
                        rb_val[slot, R_FIXED + 2 * a] = cur_depth[a, 0]
                        rb_val[slot, R_FIXED + 2 * a + 1] = cur_depth[a, 1]
                    rbase = R_FIXED + 2 * nb
                    for a in range(ns):
                        rb_val[slot, rbase + 2 * a] = cur_cost[a, 0]
                        rb_val[slot, rbase + 2 * a + 1] = cur_cost[a, 1]
                    slot += 1
                    if slot >= rb_ts.shape[0]:
                        slot = 0
                    S[S_RB_HEAD] = slot
                    if S[S_RB_COUNT] < rb_ts.shape[0]:
                        S[S_RB_COUNT] += 1

                for a in range(ns):
                    prev_cost[a, 0] = cur_cost[a, 0]
                    prev_cost[a, 1] = cur_cost[a, 1]
                for a in range(nb):
                    prev_depth[a, 0] = cur_depth[a, 0]
                    prev_depth[a, 1] = cur_depth[a, 1]
                S[S_PREV_BT] = bid_t
                S[S_PREV_AT] = ask_t
                S[S_PREV_LVLB] = ln[BID]
                S[S_PREV_LVLA] = ln[ASK]
                F[F_PREV_BQ] = bq
                F[F_PREV_AQ] = aq
                F[F_PREV_SPREAD] = spread_bps
                F[F_PREV_MID] = mid
                F[F_PREV_QI] = qi
                F[F_PREV_VALID] = float(valid)
                F[F_LAST_TS] = float(S[S_LAST_RECV])

            if S[S_LAST_RECV] >= 0 and r < S[S_LAST_RECV]:
                S[S_RECV_REVERSALS] += 1
            S[S_LAST_RECV] = r
            S[S_PK] += 1

        # ---------------- row dispatch ------------------------------------
        S[S_N_ROWS] += 1
        et = etype[i]
        pk = S[S_PK]

        if et == ET_SNAPSHOT:
            if S[S_IN_SNAPSHOT] == 0:
                # Validate the reconstruction against the authoritative block
                # before the block overwrites it.
                if F[F_PREV_VALID] > 0.5 and nsv < sv.shape[0]:
                    j = i
                    sbb = -1.0
                    sba = 1.0e18
                    sbq = 0.0
                    saq = 0.0
                    while j < nrows and etype[j] == ET_SNAPSHOT:
                        if side[j] == BID:
                            p = tick[j] * tick_size
                            if p > sbb:
                                sbb = p
                            sbq += qty[j]
                        elif side[j] == ASK:
                            p = tick[j] * tick_size
                            if p < sba:
                                sba = p
                            saq += qty[j]
                        j += 1
                    if sbb > 0.0 and sba < 1.0e17:
                        rb = 0.0
                        ra = 0.0
                        for k in range(ln[BID]):
                            rb += lq[BID, lv[BID, k]]
                        for k in range(ln[ASK]):
                            ra += lq[ASK, lv[ASK, k]]
                        sv[nsv, 0] = float(S[S_LAST_RECV])
                        sv[nsv, 1] = S[S_PREV_BT] * tick_size
                        sv[nsv, 2] = sbb
                        sv[nsv, 3] = S[S_PREV_AT] * tick_size
                        sv[nsv, 4] = sba
                        sv[nsv, 5] = rb
                        sv[nsv, 6] = sbq
                        sv[nsv, 7] = ra
                        sv[nsv, 8] = saq
                        sv[nsv, 9] = float(ln[BID])
                        sv[nsv, 10] = float(ln[ASK])
                        nsv += 1
                _reset_book(lq, lt, lh, lv, ln, o_tick, o_qty, o_side, o_next,
                            o_prev, o_key, h_key, h_val, hmask, S, n_slots)
                S[S_IN_SNAPSHOT] = 1
                S[S_SNAP_BLOCKS] += 1
                S[S_PK_LAST_SNAP] = pk
                F[F_PREV_VALID] = 0.0
            S[S_SNAP_ROWS] += 1
            e_ts = exch[i]
            if e_ts > S[S_PKT_EXCH]:
                S[S_PKT_EXCH] = e_ts
            t = tick[i]
            sd = side[i]
            if sd >= 0 and 0 <= t < n_ticks and oid[i] != 0:
                _order_insert(lq, lt, lh, lv, ln, o_tick, o_qty, o_side, o_next,
                              o_prev, o_key, h_key, h_val, hmask, S, oid[i], sd,
                              t, qty[i], pk, max_levels)
            else:
                S[S_BAD_ROWS] += 1
            i += 1
            continue

        if S[S_IN_SNAPSHOT] == 1:
            S[S_IN_SNAPSHOT] = 0

        if et == ET_TRADE:
            # One aggressive order = a maximal run of trade rows sharing this
            # packet and this side.  The book state carried is the state at the
            # close of the previous packet, i.e. genuinely pre-trade: the fills'
            # own book updates arrive in later packets.
            j = i
            sd = side[i]
            n_fill = 0
            tot_q = 0.0
            tot_n = 0.0
            p_first = tick[i] * tick_size
            p_last = p_first
            while j < nrows and etype[j] == ET_TRADE and recv[j] == r and side[j] == sd:
                px = tick[j] * tick_size
                qv = qty[j]
                tot_q += qv
                tot_n += px * qv
                p_last = px
                n_fill += 1
                if nf < f_ts.shape[0]:
                    f_ts[nf] = recv[j]
                    f_val[nf, 0] = 1.0 if sd == BID else -1.0
                    f_val[nf, 1] = px
                    f_val[nf, 2] = qv
                    f_val[nf, 3] = float(S[S_BURST])
                    nf += 1
                j += 1
            S[S_TRADE_ROWS] += n_fill

            if F[F_PREV_VALID] > 0.5 and r >= emit_from_ns and tot_q > 0.0:
                sign = 1.0 if sd == BID else -1.0

                # Locate the pre-trade book on the *exchange* clock: the newest
                # ring entry whose exchange time precedes this order's own.
                target = exch[i] - guard_ns
                cnt = S[S_RB_COUNT]
                head = S[S_RB_HEAD]
                found = -1
                oldest = -1
                for k in range(cnt):
                    idx = head - 1 - k
                    if idx < 0:
                        idx += rb_ts.shape[0]
                    oldest = idx
                    if rb_ts[idx] <= target:
                        found = idx
                        break
                if found < 0:
                    found = oldest
                    S[S_RING_MISS] += 1

                if found >= 0:
                    rbid = rb_val[found, R_BID]
                    rask = rb_val[found, R_ASK]
                    # Aggressor-side consistency: a buyer lifting the offer must
                    # trade at or above the pre-trade best ask.  Both the
                    # exchange-aligned and the naive receipt-aligned versions are
                    # counted so the diagnostics table can show the difference.
                    S[S_SIDE_TESTED] += 1
                    if sd == BID:
                        if p_first >= rask - 1e-9:
                            S[S_SIDE_AGREE] += 1
                        if p_first >= S[S_PREV_AT] * tick_size - 1e-9:
                            S[S_SIDE_AGREE_RECV] += 1
                    else:
                        if p_first <= rbid + 1e-9:
                            S[S_SIDE_AGREE] += 1
                        if p_first <= S[S_PREV_BT] * tick_size + 1e-9:
                            S[S_SIDE_AGREE_RECV] += 1

                    if nbst < b_ts.shape[0]:
                        b_ts[nbst, 0] = r
                        b_ts[nbst, 1] = exch[i]
                        b_val[nbst, B_SIGN] = sign
                        b_val[nbst, B_NFILL] = float(n_fill)
                        b_val[nbst, B_QTY] = tot_q
                        b_val[nbst, B_NOTIONAL] = tot_n
                        b_val[nbst, B_VWAP] = tot_n / tot_q
                        b_val[nbst, B_PFIRST] = p_first
                        b_val[nbst, B_PLAST] = p_last
                        b_val[nbst, B_BID] = rbid
                        b_val[nbst, B_ASK] = rask
                        b_val[nbst, B_BIDQ] = rb_val[found, R_BIDQ]
                        b_val[nbst, B_ASKQ] = rb_val[found, R_ASKQ]
                        b_val[nbst, B_SPREAD] = rb_val[found, R_SPREAD]
                        b_val[nbst, B_MID] = rb_val[found, R_MID]
                        b_val[nbst, B_LVLB] = rb_val[found, R_LVLB]
                        b_val[nbst, B_LVLA] = rb_val[found, R_LVLA]
                        b_val[nbst, B_PKSNAP] = rb_val[found, R_PKSNAP]
                        for a in range(2 * nb + 2 * ns):
                            b_val[nbst, B_FIXED + a] = rb_val[found, R_FIXED + a]
                        nbst += 1
            S[S_BURST] += 1
            i = j
            continue

        # ---- book-modifying rows -----------------------------------------
        key = oid[i]
        sd = side[i]
        t = tick[i]
        if key == 0 or sd < 0:
            S[S_BAD_ROWS] += 1
            i += 1
            continue
        if t < 0 or t >= n_ticks:
            S[S_OFF_GRID] += 1
            i += 1
            continue

        # Running-max exchange clock over applied book rows.  Exchange
        # timestamps reverse on ~0.1% of rows, so the max keeps the ring's key
        # monotone without discarding any event.
        e_ts = exch[i]
        if e_ts > 0:
            if e_ts > S[S_PKT_EXCH]:
                S[S_PKT_EXCH] = e_ts
            elif e_ts < S[S_PKT_EXCH]:
                S[S_EXCH_REVERSALS] += 1

        hidx = _h_find(h_key, key, hmask)

        if et == ET_ADD or et == ET_MODIFY:
            q = qty[i]
            if hidx >= 0:
                slot = h_val[hidx]
                if o_side[slot] == sd and o_tick[slot] == t and o_qty[slot] == q:
                    # Re-notification of an unchanged order: this is how the
                    # feed announces an order scrolling back into the visible
                    # depth window.  Refresh the level's recency, change nothing
                    # else.
                    lt[sd, t] = pk
                    S[S_IDEMPOTENT] += 1
                    i += 1
                    continue
                _order_remove(lq, lh, lv, ln, o_tick, o_qty, o_side, o_next,
                              o_prev, o_key, h_key, h_val, hmask, S, slot, hidx)
            _order_insert(lq, lt, lh, lv, ln, o_tick, o_qty, o_side, o_next,
                          o_prev, o_key, h_key, h_val, hmask, S, key, sd, t, q,
                          pk, max_levels)
        elif et == ET_DELETE:
            if hidx < 0:
                # The order was evicted with its level when it scrolled out of
                # the visible window; the exchange is now reporting its cancel.
                S[S_ORPHAN_DELETE] += 1
            else:
                slot = h_val[hidx]
                st = o_tick[slot]
                ssd = o_side[slot]
                _order_remove(lq, lh, lv, ln, o_tick, o_qty, o_side, o_next,
                              o_prev, o_key, h_key, h_val, hmask, S, slot, hidx)
                lt[ssd, st] = pk
        else:
            S[S_BAD_ROWS] += 1
        i += 1

    q_count[0] = nq
    b_count[0] = nbst
    f_count[0] = nf
    sv_count[0] = nsv


class BookState:
    """Owns every persistent array the kernel mutates across batches."""

    def __init__(self, cfg) -> None:
        n_ticks = 1 << cfg.TICK_SPACE_BITS
        n_slots = 1 << cfg.MAX_LIVE_ORDERS_BITS
        # The live-order population is ~250 (100 levels a side); the hash table
        # is sized far above that so probe chains stay at length ~1.
        h_bits = 16
        h_size = 1 << h_bits

        self.n_ticks = n_ticks
        self.n_slots = n_slots
        self.hmask = np.int64(h_size - 1)

        self.lq = np.zeros((2, n_ticks), dtype=np.float64)
        self.lt = np.zeros((2, n_ticks), dtype=np.int64)
        self.lh = np.full((2, n_ticks), -1, dtype=np.int32)
        self.lv = np.zeros((2, cfg.MAX_LEVELS_PER_SIDE), dtype=np.int32)
        self.ln = np.zeros(2, dtype=np.int64)

        self.o_tick = np.zeros(n_slots, dtype=np.int32)
        self.o_qty = np.zeros(n_slots, dtype=np.float64)
        self.o_side = np.zeros(n_slots, dtype=np.int8)
        self.o_next = np.arange(1, n_slots + 1, dtype=np.int32)
        self.o_next[-1] = -1
        self.o_prev = np.full(n_slots, -1, dtype=np.int32)
        self.o_key = np.zeros(n_slots, dtype=np.int64)

        self.h_key = np.zeros(h_size, dtype=np.int64)
        self.h_val = np.zeros(h_size, dtype=np.int32)

        self.S = np.zeros(S_LEN, dtype=np.int64)
        self.S[S_LAST_RECV] = -1
        self.S[S_PREV_BT] = -1
        self.S[S_PREV_AT] = -1
        self.F = np.zeros(F_LEN, dtype=np.float64)

        n_sizes = len(cfg.VIRTUAL_ORDER_SIZES_BTC)
        n_bands = len(cfg.DEPTH_BANDS_BPS)
        self.prev_cost = np.full((n_sizes, 2), np.nan)
        self.prev_depth = np.full((n_bands, 2), np.nan)
        self.cur_cost = np.full((n_sizes, 2), np.nan)
        self.cur_depth = np.full((n_bands, 2), np.nan)

        self.sizes = np.asarray(cfg.VIRTUAL_ORDER_SIZES_BTC, dtype=np.float64)
        self.bands = np.asarray(cfg.DEPTH_BANDS_BPS, dtype=np.float64)

        ring_width = R_FIXED + 2 * n_bands + 2 * n_sizes
        self.rb_ts = np.zeros(RING_SIZE, dtype=np.int64)
        self.rb_val = np.zeros((RING_SIZE, ring_width), dtype=np.float64)

        # Streaming (all-packet) accumulators.
        nb = cfg.HIST_BINS
        self.acc_spread_tw = np.zeros(nb)
        self.acc_spread_ew = np.zeros(nb)
        self.acc_cost_tw = np.zeros((n_sizes, 2, nb))
        self.acc_cost_stats = np.zeros((4, n_sizes, 2))
        self.acc_depth_sum = np.zeros((n_bands, 2))
        self.acc_qi = np.zeros(cfg.QI_HIST_BINS)
        self.acc_hour = np.zeros((7, 24))
