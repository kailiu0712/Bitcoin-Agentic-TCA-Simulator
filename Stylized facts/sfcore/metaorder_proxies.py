"""Readable metaorder grouping on an already-clean, chronological trade table.

No order-book reconstruction here. A child is an integer row position; each
method returns disjoint lists of positions. Call separately for each UTC day.
The September paper's forward scan can leave jumped-over rows unassigned:
we preserve that feature, report coverage, and do not silently fill the holes.
"""
from __future__ import annotations

import numpy as np


def random_traders(sign, rng, n_traders=6, weight_exponent=0.0):
    """March paper, arXiv:2503.18199 Algorithms 1–2.

    Assign identities independently; a trader's own sign flip ends its run.
    Other traders' intervening trades do not end it. Nonzero weight_exponent
    uses deterministic rank weights, an explicit alternative to iid weights.
    Singletons are returned too, so filtering and coverage remain auditable.
    """
    sign = np.asarray(sign)
    if n_traders < 1:
        raise ValueError("n_traders must be positive")
    weights = np.arange(1, n_traders + 1, dtype=float) ** (-weight_exponent)
    owners = rng.choice(n_traders, size=len(sign), p=weights / weights.sum())
    parents = []
    for owner in range(n_traders):
        rows = np.flatnonzero(owners == owner)
        boundaries = np.flatnonzero(np.diff(sign[rows]) != 0) + 1
        parents.extend(part for part in np.split(rows, boundaries) if len(part))
    return parents


def time_threshold(ts_ns, sign, rng, mean_wait_s=5.0, mu=1.5,
                   max_children=1000, max_target_delay_s=4.0):
    """September paper, arXiv:2509.05065 p17 Algorithm 2.

    Split buy/sell. Draw s with discrete P(s) proportional to s^(-1-mu)
    on 1..max_children. Draw Exp waiting TIMES with mean_wait_s (phi=1/mean).
    Pick first same-sign trade at/after the target; stop if its overshoot is
    > max_target_delay_s = C/phi. This is NOT an adjacent-trade gap threshold.

    Units are seconds, explicitly adapted from the simulation time units.
    The paper text sets C=4*phi: its numeric C/phi=4 becomes 4 seconds here.
    The paper does not specify s_max or a real-market phi: these are settings,
    not inferred market facts. Move forward even at tied nanosecond timestamps.
    """
    ts_ns = np.asarray(ts_ns, dtype=np.int64)
    sign = np.asarray(sign)
    if mean_wait_s <= 0 or mu <= 0 or max_children < 1 or max_target_delay_s < 0:
        raise ValueError("Invalid time-threshold parameters")
    if len(ts_ns) != len(sign) or np.any(np.diff(ts_ns) < 0):
        raise ValueError("Input must be chronological and arrays must match")
    sizes = np.arange(1, max_children + 1)
    cdf = np.cumsum(sizes.astype(float) ** (-1.0 - mu))
    cdf /= cdf[-1]
    parents = []
    for direction in (-1, 1):
        positions = np.flatnonzero(sign == direction)
        times = ts_ns[positions]
        i = 0
        while i < len(times):
            wanted = int(np.searchsorted(cdf, rng.random(), side="right")) + 1
            children = []
            for _ in range(wanted):
                children.append(int(positions[i]))
                wait_ns = max(1, int(rng.exponential(mean_wait_s) * 1e9))
                target = int(times[i]) + wait_ns
                j = int(np.searchsorted(times, target, side="left"))
                if j >= len(times) or int(times[j]) - target > max_target_delay_s * 1e9:
                    # Paper lines 15–16: break while i still points at assigned row.
                    i += 1
                    break
                # Paper lines 18–19 happen even on the last count iteration.
                # That next, still-unassigned row starts the following parent.
                i = j
            parents.append(np.asarray(children, dtype=np.int64))
    return parents


def local_runs(ts_ns, sign, qty, gap_s=1.0, opposite_tolerance=0.5):
    """Existing framework's local-flow baseline; only same-side rows are children.

    Unlike the legacy end_idx, completion is the last SAME-SIGN child rather
    than a tolerated trailing opposite trade. Measurement is shared across all
    methods, so this is a harmonized baseline, not a rerun of old summary.csv.
    """
    ts_ns, sign, qty = map(np.asarray, (ts_ns, sign, qty))
    parents = []
    i = 0
    while i < len(sign):
        direction, same, opposite = sign[i], float(qty[i]), 0.0
        children = [i]
        j = i + 1
        while j < len(sign) and 0 <= ts_ns[j] - ts_ns[j - 1] <= gap_s * 1e9:
            if sign[j] == direction:
                same += qty[j]
                children.append(j)
            elif opposite + qty[j] <= opposite_tolerance * same:
                opposite += qty[j]
            else:
                break
            j += 1
        parents.append(np.asarray(children, dtype=np.int64))
        i = j
    return parents


def build_parents(day, method, rng, **params):
    """Public interface: day has ts_ns, sign, qty; result indexes day.iloc."""
    if method == "random_traders":
        return random_traders(day.sign.to_numpy(), rng, **params)
    if method == "time_threshold":
        return time_threshold(day.ts_ns.to_numpy(), day.sign.to_numpy(), rng, **params)
    if method == "local_runs":
        return local_runs(day.ts_ns.to_numpy(), day.sign.to_numpy(), day.qty.to_numpy(), **params)
    raise ValueError(f"Unknown method: {method}")
