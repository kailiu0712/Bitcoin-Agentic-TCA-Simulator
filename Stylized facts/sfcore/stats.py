"""
Shared estimators.

Kept deliberately small and explicit: every stylized fact in ``sffacts`` is
built from these primitives, so a reader can check one implementation of an
autocorrelation or a power-law fit rather than thirteen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation of ``x`` for lags 1..max_lag, via FFT.

    The FFT route is what makes lag-2000 autocorrelations on multi-million
    point series affordable; the direct estimator would be O(N * max_lag).
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < max_lag + 2:
        max_lag = max(1, n - 2)
    if n < 4:
        return np.full(max_lag, np.nan)
    x = x - x.mean()
    size = 1 << int(np.ceil(np.log2(2 * n - 1)))
    f = np.fft.rfft(x, size)
    acov = np.fft.irfft(f * np.conjugate(f), size)[: max_lag + 1].real
    if acov[0] <= 0:
        return np.full(max_lag, np.nan)
    return (acov[1:] / acov[0])[:max_lag]


def acf_standard_error(n: int, max_lag: int) -> np.ndarray:
    """Bartlett white-noise standard error, used only to draw a null band."""
    return np.full(max_lag, 1.0 / np.sqrt(max(n, 1)))


# ---------------------------------------------------------------------------
# Power-law / concavity fits
# ---------------------------------------------------------------------------


def fit_loglog(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """OLS of log(y) on log(x).  Returns exponent, prefactor and fit quality.

    Used for (a) the decay exponent of the order-sign autocorrelation, (b) the
    concavity exponent of the impact curve, and (c) the slope of the
    liquidity-cost curve.  Points with non-positive or non-finite values are
    dropped rather than clipped, so a bad point cannot silently distort a fit.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 3:
        return {"exponent": np.nan, "prefactor": np.nan, "r2": np.nan,
                "se": np.nan, "n": int(ok.sum())}
    lx = np.log(x[ok])
    ly = np.log(y[ok])
    n = lx.size
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    resid = ly - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    sxx = float(np.sum((lx - lx.mean()) ** 2))
    se = float(np.sqrt(ss_res / max(n - 2, 1) / sxx)) if sxx > 0 else np.nan
    return {"exponent": float(slope), "prefactor": float(np.exp(intercept)),
            "r2": float(r2), "se": se, "n": int(n)}


def fit_log_model(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """OLS of y on log(x) - the logarithmic impact alternative to a power law.

    Zarinelli et al. note that a logarithmic form fits metaorder impact over a
    wider size range than the square root, so both are reported and compared by
    R-squared rather than one being assumed.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if ok.sum() < 3:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "n": 0}
    lx = np.log(x[ok])
    yy = y[ok]
    slope, intercept = np.polyfit(lx, yy, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    return {"slope": float(slope), "intercept": float(intercept),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
            "n": int(ok.sum())}


def fit_exponential_recovery(t: np.ndarray, y: np.ndarray, y_inf: float) -> dict[str, float]:
    """Fit y(t) = y_inf + A exp(-t / tau) by OLS on log(y - y_inf).

    This is the form Schnaubelt, Rende & Krauss use for post-trade spread
    recovery on Coinbase BTC/USD, where they report tau ~ 6.3 s.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    excess = y - y_inf
    ok = np.isfinite(t) & np.isfinite(excess) & (excess > 0) & (t > 0)
    if ok.sum() < 3:
        return {"tau_s": np.nan, "amplitude": np.nan, "r2": np.nan, "n": int(ok.sum())}
    lt = t[ok]
    ly = np.log(excess[ok])
    slope, intercept = np.polyfit(lt, ly, 1)
    if slope >= 0:
        return {"tau_s": np.nan, "amplitude": float(np.exp(intercept)),
                "r2": np.nan, "n": int(ok.sum())}
    pred = slope * lt + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    return {"tau_s": float(-1.0 / slope), "amplitude": float(np.exp(intercept)),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
            "n": int(ok.sum())}


# ---------------------------------------------------------------------------
# Histogram utilities (for the streaming, all-events accumulators)
# ---------------------------------------------------------------------------


def log_bin_edges(lo: float, hi: float, nbins: int) -> np.ndarray:
    return np.exp(np.linspace(np.log(lo), np.log(hi), nbins + 1))


def hist_quantiles(counts: np.ndarray, edges: np.ndarray, qs) -> np.ndarray:
    """Quantiles of a weighted histogram, interpolating inside the hit bin."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return np.full(len(qs), np.nan)
    cum = np.concatenate([[0.0], np.cumsum(counts)]) / total
    out = np.empty(len(qs))
    for i, q in enumerate(qs):
        j = int(np.searchsorted(cum, q, side="left"))
        j = min(max(j, 1), len(counts))
        lo_c, hi_c = cum[j - 1], cum[j]
        frac = 0.0 if hi_c <= lo_c else (q - lo_c) / (hi_c - lo_c)
        out[i] = edges[j - 1] * (edges[j] / edges[j - 1]) ** frac
    return out


def hist_mean(counts: np.ndarray, edges: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return np.nan
    centres = np.sqrt(edges[:-1] * edges[1:])
    return float((counts * centres).sum() / total)


# ---------------------------------------------------------------------------
# Conditioning and joins
# ---------------------------------------------------------------------------


def quantile_bins(x: np.ndarray, nbins: int) -> np.ndarray:
    """Quantile bin index, robust to heavy ties (crypto trade sizes cluster)."""
    x = np.asarray(x, dtype=np.float64)
    ranks = pd.Series(x).rank(method="average", pct=True).to_numpy()
    idx = np.clip((ranks * nbins).astype(int), 0, nbins - 1)
    return idx


def asof_left(sorted_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """Index of the last element of ``sorted_ts`` at or before each query.

    Returns -1 where no such element exists.  ``sorted_ts`` must be sorted.
    """
    idx = np.searchsorted(sorted_ts, query_ts, side="right") - 1
    return idx


def group_mean(values: np.ndarray, groups: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and count of ``values`` per integer group, ignoring NaNs."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    ok = np.isfinite(values) & (groups >= 0) & (groups < n_groups)
    sums = np.bincount(groups[ok], weights=values[ok], minlength=n_groups)
    counts = np.bincount(groups[ok], minlength=n_groups).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / counts, np.nan)
    return means, counts


def group_sem(values: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    """Standard error of the group mean, for error bars on binned curves."""
    values = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(values) & (groups >= 0) & (groups < n_groups)
    counts = np.bincount(groups[ok], minlength=n_groups).astype(np.float64)
    sums = np.bincount(groups[ok], weights=values[ok], minlength=n_groups)
    sq = np.bincount(groups[ok], weights=values[ok] ** 2, minlength=n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sums / counts
        var = np.maximum(sq / counts - mean ** 2, 0.0)
        sem = np.sqrt(var / np.maximum(counts - 1, 1))
    return np.where(counts > 1, sem, np.nan)


# ---------------------------------------------------------------------------
# Price-process diagnostics
# ---------------------------------------------------------------------------


def clock_bars(ts_ns: np.ndarray, values: np.ndarray, seconds: float) -> np.ndarray:
    """Last observation in each fixed clock bucket, carried forward.

    Used only where a literature target is itself a clock-time statement (the
    volatility-clustering exponents, the minutes-to-days return
    autocorrelation).  No event is discarded: every observation is assigned to
    a bucket and the last one in each bucket wins, exactly as a last-trade bar
    would be built.
    """
    ts_ns = np.asarray(ts_ns, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if ts_ns.size < 2:
        return np.array([])
    step = int(seconds * 1e9)
    bucket = (ts_ns - ts_ns[0]) // step
    n_bars = int(bucket[-1]) + 1
    if n_bars < 20:
        return np.array([])
    last = np.full(n_bars, np.nan)
    last[bucket] = values
    idx = np.where(np.isfinite(last), np.arange(n_bars), 0)
    np.maximum.accumulate(idx, out=idx)
    return last[idx]


def variance_ratio(returns: np.ndarray, ks) -> np.ndarray:
    """Lo-MacKinlay variance ratio Var(sum of k returns) / (k Var(return)).

    A value of 1 means diffusive; <1 mean-reverting (bid-ask bounce); >1
    trending.  This is the joint constraint on persistent order flow and
    adaptive liquidity in stylized fact 11.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    base = np.var(r)
    out = np.empty(len(ks))
    for i, k in enumerate(ks):
        k = int(k)
        if k < 1 or r.size < 5 * k or base <= 0:
            out[i] = np.nan
            continue
        m = (r.size // k) * k
        agg = r[:m].reshape(-1, k).sum(axis=1)
        out[i] = float(np.var(agg) / (k * base))
    return out


def hill_tail_index(x: np.ndarray, tail_fraction: float = 0.05) -> dict[str, float]:
    """Hill estimator of the upper-tail index of a positive sample.

    Reported for the trade-size distribution (stylized fact 16), where the
    literature describes a heavy upper tail.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 100:
        return {"alpha": np.nan, "k": 0, "threshold": np.nan}
    x = np.sort(x)
    k = max(int(x.size * tail_fraction), 10)
    tail = x[-k:]
    threshold = tail[0]
    alpha = k / np.sum(np.log(tail / threshold)) if threshold > 0 else np.nan
    return {"alpha": float(alpha), "k": int(k), "threshold": float(threshold)}


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    aa, bb = a[ok], b[ok]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def ols(y: np.ndarray, X: np.ndarray, add_const: bool = True) -> dict:
    """Plain OLS with heteroskedasticity-robust (HC0) standard errors."""
    y = np.asarray(y, dtype=np.float64)
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if X.shape[0] != y.shape[0]:
        X = X.T
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[ok], X[ok]
    if y.size < X.shape[1] + 2:
        return {"beta": np.full(X.shape[1] + int(add_const), np.nan),
                "se": np.full(X.shape[1] + int(add_const), np.nan),
                "r2": np.nan, "n": int(y.size)}
    if add_const:
        X = np.column_stack([np.ones(len(X)), X])
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    meat = (X * (resid ** 2)[:, None]).T @ X
    cov = xtx_inv @ meat @ xtx_inv
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan
    return {"beta": beta, "se": np.sqrt(np.clip(np.diag(cov), 0, None)),
            "r2": r2, "n": int(y.size)}
