"""
Estimators required by the proxy.ipynb / ZJ tests, added 2026-09-05.

Kept beside ``stats.py`` rather than inside it because these are specifically
the fits the two source documents use, and a reader comparing this framework's
numbers against those documents should be able to check them in one place.

Everything here is numpy-only: no scipy, no statsmodels.  That is not
minimalism for its own sake.  Two of the notebook's fits are non-linear in a
*single* parameter, and conditioning on that parameter makes the remainder an
exact weighted least-squares solve.  Scanning a dense grid of the awkward
parameter therefore reaches the same optimum as a generic optimiser, with no
starting values, no bounds, and no ``maxfev`` failure mode.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _wls(y: np.ndarray, X: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    """Weighted least squares.  Returns (coefficients, weighted SSR)."""
    rw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * rw[:, None], y * rw, rcond=None)
    resid = y - X @ beta
    return beta, float(np.sum(w * resid ** 2))


def wls_loglog(x, y, sem=None, min_rel_error: float = 1e-2) -> dict:
    """WLS of log(y) on log(x), weighted by the delta-method log standard error.

    This is the notebook's duration and volume-clock regression (cells 3, 4,
    11).  The points are *bin means*, so the weight is 1 / (sem / y)**2.

    A bin whose mean is non-positive cannot enter a log fit.  Such bins are
    dropped and counted, so that keeping signed observations upstream (rather
    than pre-filtering individual metaorders to positive impact) stays
    auditable instead of silently changing the estimand.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if sem is None:
        rel = np.full(x.shape, min_rel_error)
    else:
        # A bin mean of exactly zero has no relative error; it is also dropped
        # below, so guard the division rather than let it emit a warning.
        magnitude = np.where(np.abs(y) > 0, np.abs(y), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.maximum(np.asarray(sem, dtype=np.float64) / magnitude,
                             min_rel_error)
    usable = np.isfinite(x) & np.isfinite(y) & (x > 0)
    ok = usable & (y > 0) & np.isfinite(rel)
    dropped = int(usable.sum() - ok.sum())
    if ok.sum() < 3:
        return {"slope": np.nan, "intercept": np.nan, "se": np.nan, "r2": np.nan,
                "n": int(ok.sum()), "n_dropped_nonpositive": dropped}
    lx, ly, w = np.log(x[ok]), np.log(y[ok]), 1.0 / rel[ok] ** 2
    X = np.column_stack([np.ones(lx.size), lx])
    beta, ssr = _wls(ly, X, w)
    cov = np.linalg.pinv(X.T @ (X * w[:, None])) * (ssr / max(lx.size - 2, 1))
    centred = ly - np.average(ly, weights=w)
    sst = float(np.sum(w * centred ** 2))
    return {"slope": float(beta[1]), "intercept": float(beta[0]),
            "se": float(np.sqrt(max(cov[1, 1], 0.0))),
            "r2": float(1.0 - ssr / sst) if sst > 0 else np.nan,
            "n": int(lx.size), "n_dropped_nonpositive": dropped}


def ols_hc3(y: np.ndarray, X: np.ndarray, names=None) -> dict:
    """OLS with HC3 (leverage-corrected) robust covariance, as the notebook uses.

    HC3 divides each squared residual by ``(1 - h_ii)**2``.  It is the
    conservative choice for the individual-metaorder regressions, where a
    handful of very large parents would otherwise dominate an HC0 covariance.

    It still assumes independent observations.  Overlapping parents and serial
    dependence are *not* handled, so these standard errors remain descriptive
    and no formal significance claim is made from them.
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if X.shape[0] != y.shape[0]:
        X = X.T
    labels = ["const"] + (list(names) if names else
                          [f"x{i}" for i in range(1, X.shape[1] + 1)])
    X = np.column_stack([np.ones(len(X)), X])
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[ok], X[ok]
    k = X.shape[1]
    if y.size < k + 2:
        blank = np.full(k, np.nan)
        return {"params": blank, "se": blank, "t": blank, "r2": np.nan,
                "n": int(y.size), "names": labels}
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    leverage = np.clip(np.einsum("ij,jk,ik->i", X, xtx_inv, X), 0.0, 1.0 - 1e-10)
    adjusted = (resid / (1.0 - leverage)) ** 2
    cov = xtx_inv @ ((X * adjusted[:, None]).T @ X) @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    sst = float(np.sum((y - y.mean()) ** 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        t = beta / se
    return {"params": beta, "se": se, "t": t, "n": int(y.size), "names": labels,
            "r2": float(1.0 - np.sum(resid ** 2) / sst) if sst > 0 else np.nan}


def fit_broken_power_law(x, y, sem=None, n_break: int = 400,
                         min_side: int = 4, min_side_fraction: float = 0.15) -> dict:
    """Continuous broken power law in log-log space (proxy.ipynb cell 2).

        log y = c + d1 * (log x - u_c)   where log x <= u_c
              = c + d2 * (log x - u_c)   where log x >  u_c

    The two branches meet at ``(u_c, c)`` by construction, which is what makes
    the notebook's version "continuous".  For a *fixed* ``u_c`` the model is
    linear in ``(c, d1, d2)``, so the breakpoint is found by scanning a dense
    grid and solving the rest exactly at each candidate.

    ``x`` and ``y`` are bin means.  Non-positive means are dropped and counted
    (see ``wls_loglog``).

    The breakpoint is confined to the interior: each branch must keep at least
    ``min_side`` bins and ``min_side_fraction`` of them.  Without that guard the
    scan can park the break among the handful of sparse bins at one end and fit
    ``d2`` to three noise points, which silently swaps the meaning of the two
    branches - ``d1`` becomes the impact exponent and ``d2`` the noise one.
    ``break_at_boundary`` flags a fit that still lands against the guard.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y) & (x > 0)
    ok = finite & (y > 0)
    empty = {"x_break": np.nan, "y_break": np.nan, "delta_1": np.nan,
             "delta_2": np.nan, "delta_1_se": np.nan, "delta_2_se": np.nan,
             "x_break_resolution": np.nan, "prefactor_Y": np.nan,
             "r2_log": np.nan, "break_at_boundary": False, "n": int(ok.sum()),
             "n_dropped_nonpositive": int(finite.sum() - ok.sum())}
    guard = max(min_side, int(np.ceil(min_side_fraction * ok.sum())))
    if ok.sum() < 2 * guard + 1:
        return empty
    lx, ly = np.log(x[ok]), np.log(y[ok])
    rel = (np.full(lx.shape, 1e-2) if sem is None else
           np.maximum(np.asarray(sem, dtype=np.float64)[ok] / y[ok], 1e-4))
    w = 1.0 / rel ** 2
    order = np.argsort(lx)
    lx, ly, w = lx[order], ly[order], w[order]
    # A candidate breakpoint has to leave enough points on both sides for two
    # slopes to be identified, so the scan stays strictly inside that interior.
    lo, hi = lx[guard - 1], lx[-guard]
    if not hi > lo:
        return empty
    grid = np.linspace(lo, hi, n_break)
    best = None
    for u_c in grid:
        left = lx <= u_c
        if left.sum() < guard or (~left).sum() < guard:
            continue
        gap = lx - u_c
        X = np.column_stack([np.ones(lx.size), np.where(left, gap, 0.0),
                             np.where(left, 0.0, gap)])
        beta, ssr = _wls(ly, X, w)
        if best is None or ssr < best[0]:
            best = (ssr, u_c, beta, X)
    if best is None:
        return empty
    ssr, u_c, beta, X = best
    cov = np.linalg.pinv(X.T @ (X * w[:, None])) * (ssr / max(lx.size - 4, 1))
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    centred = ly - np.average(ly, weights=w)
    sst = float(np.sum(w * centred ** 2))
    x_break, y_break = float(np.exp(u_c)), float(np.exp(beta[0]))
    step = float(grid[1] - grid[0]) if len(grid) > 1 else np.nan
    return {"x_break": x_break, "y_break": y_break,
            "delta_1": float(beta[1]), "delta_2": float(beta[2]),
            "delta_1_se": float(se[1]), "delta_2_se": float(se[2]),
            # The breakpoint comes from a grid search conditioned on the same
            # data, so its grid resolution - not an asymptotic standard error -
            # is the honest uncertainty statement.
            "x_break_resolution": float(x_break * step),
            "prefactor_Y": float(y_break / x_break ** beta[2]),
            "r2_log": float(1.0 - ssr / sst) if sst > 0 else np.nan,
            "break_at_boundary": bool(u_c <= grid[0] + step or u_c >= grid[-1] - step),
            "n": int(lx.size),
            "n_dropped_nonpositive": int(finite.sum() - ok.sum())}


def fit_propagator_decay(z, y, sem=None, z_peak=None, n_beta: int = 800) -> dict:
    """Propagator decay measured from the observed peak (proxy.ipynb cell 5).

        I(z) = I_0 * (z'**(1-beta) - (z'-1)**(1-beta)),   z' = 1 + (z - z_peak)

    Linear in ``I_0`` once ``beta`` is fixed, so beta is scanned and ``I_0``
    solved exactly at each candidate.

    ``z_peak`` defaults to the argmax of the supplied curve.  That is the
    notebook's choice, and it means the peak is selected from the same sample
    that is then fitted: the returned interval does not price that selection
    step in, so it is descriptive only.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(z) & np.isfinite(y)
    blank = {"beta": np.nan, "beta_resolution": np.nan, "I_0": np.nan,
             "z_peak": np.nan, "r2": np.nan, "n": 0}
    if ok.sum() < 4:
        return blank
    z, y = z[ok], y[ok]
    w = (np.ones(z.size) if sem is None else
         1.0 / np.maximum(np.asarray(sem, dtype=np.float64)[ok], 1e-12) ** 2)
    if z_peak is None:
        z_peak = float(z[int(np.argmax(y))])
    keep = z >= z_peak
    if keep.sum() < 4:
        return dict(blank, z_peak=float(z_peak), n=int(keep.sum()))
    zz, yy, ww = z[keep], y[keep], w[keep]
    shifted = 1.0 + (zz - z_peak)
    betas = np.linspace(0.01, 0.95, n_beta)
    best = None
    for beta in betas:
        past = np.maximum(shifted - 1.0, 0.0)
        kernel = shifted ** (1.0 - beta) - np.where(shifted > 1.0,
                                                    past ** (1.0 - beta), 0.0)
        denom = float(np.sum(ww * kernel ** 2))
        if denom <= 0:
            continue
        amplitude = float(np.sum(ww * kernel * yy) / denom)
        ssr = float(np.sum(ww * (yy - amplitude * kernel) ** 2))
        if best is None or ssr < best[0]:
            best = (ssr, beta, amplitude)
    if best is None:
        return dict(blank, z_peak=float(z_peak), n=int(zz.size))
    ssr, beta, amplitude = best
    centred = yy - np.average(yy, weights=ww)
    sst = float(np.sum(ww * centred ** 2))
    return {"beta": float(beta), "beta_resolution": float(betas[1] - betas[0]),
            "I_0": float(amplitude), "z_peak": float(z_peak), "n": int(zz.size),
            "r2": float(1.0 - ssr / sst) if sst > 0 else np.nan}


def volatility_scaling(log_price, lags, segment_starts=None) -> pd.DataFrame:
    """sigma(tau) = std of the log-price increment over ``lag`` grid steps.

    ``log_price`` is a regular-grid series; proxy.ipynb cell 7 uses a 100 ms
    grid.  ``segment_starts`` holds the first index of each contiguous segment,
    so an increment is never taken across a gap between segments - the notebook
    covers a single day and does not need this, a multi-day regime does.

    The asymptotic standard error of a sample standard deviation, sd/sqrt(2N),
    is returned as ``sem_iid`` (named so it cannot shadow ``DataFrame.sem``) so the log-log fit can be weighted the way the notebook weights
    it.  N counts overlapping increments, so it overstates the independent
    sample size at long lags; the error bars are indicative, not inferential.
    """
    log_price = np.asarray(log_price, dtype=np.float64)
    n = log_price.size
    bounds = (np.array([0], dtype=np.int64) if segment_starts is None else
              np.asarray(segment_starts, dtype=np.int64))
    segment_id = np.searchsorted(bounds, np.arange(n), side="right")
    rows = []
    for k in np.unique(np.asarray(lags, dtype=np.int64)):
        if k < 1 or k >= n:
            continue
        same = segment_id[k:] == segment_id[:-k]
        diff = (log_price[k:] - log_price[:-k])[same]
        diff = diff[np.isfinite(diff)]
        if diff.size < 30:
            continue
        sd = float(np.std(diff))
        rows.append({"lag_steps": int(k), "sigma": sd, "n": int(diff.size),
                     "sem_iid": sd / np.sqrt(2 * diff.size)})
    return pd.DataFrame(rows)
