import numpy as np


def fit_power_law(points, bootstrap=500, seed=7):
    x = np.array([p["size"] for p in points], float); y = np.array([p["impact"] for p in points], float)
    ok = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y); x, y = np.log(x[ok]), np.log(y[ok])
    if len(x) < 3:
        return {"exponent": None, "ci95": [None, None], "r2": None, "n_buckets": int(len(x))}
    coef = np.polyfit(x, y, 1); pred = np.polyval(coef, x)
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    rng = np.random.default_rng(seed); slopes = []
    for _ in range(bootstrap):
        ix = rng.integers(0, len(x), len(x))
        if np.ptp(x[ix]) > 0: slopes.append(np.polyfit(x[ix], y[ix], 1)[0])
    return {"exponent": float(coef[0]), "intercept": float(coef[1]),
            "ci95": [float(v) for v in np.quantile(slopes, [.025, .975])],
            "r2": float(r2), "n_buckets": int(len(x))}

