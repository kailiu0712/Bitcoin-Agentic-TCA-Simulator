import numpy as np
import pandas as pd


LAGS = np.array([1, 2, 5, 10, 20, 50])
QUANTILE_GRID = np.array([.01,.05,.10,.25,.50,.75,.90,.95,.99])


def _q(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return {"mean": float(np.mean(x)), "median": float(np.median(x)),
            "q05": float(np.quantile(x, .05)), "q25": float(np.quantile(x, .25)),
            "q75": float(np.quantile(x, .75)), "q95": float(np.quantile(x, .95))} if len(x) else {}


def _quantiles(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    return {"probabilities":QUANTILE_GRID.tolist(),"values":np.quantile(x,QUANTILE_GRID).tolist()} if len(x) else {"probabilities":QUANTILE_GRID.tolist(),"values":[]}


def prepare(df):
    df = df.sort_values("timestamp_ns").copy()
    stable=df.event_type.eq("QUOTE_UPDATE")
    quote_mask=stable if stable.any() else df.event_type.ne("TRADE")
    quotes = df[quote_mask & df.best_bid.notna() & df.best_ask.notna()][["timestamp_ns", "best_bid", "best_ask", "bid_size", "ask_size", "mid", "spread"]]
    trades = df[df.event_type == "TRADE"].copy()
    if len(quotes) and ("mid" not in trades or trades.mid.isna().any()):
        base = trades.drop(columns=[c for c in quotes.columns if c != "timestamp_ns" and c in trades], errors="ignore")
        trades = pd.merge_asof(base.sort_values("timestamp_ns"), quotes.sort_values("timestamp_ns"), on="timestamp_ns", direction="backward")
    trades["sign"] = trades.side.str.upper().map({"BUY": 1, "SELL": -1})
    return trades.dropna(subset=["price", "size", "sign"]), quotes


def compute(df, lags=LAGS, flow_windows=(1, 5, 30, 60)):
    trades, quotes = prepare(df)
    sign = trades.sign.to_numpy(float); mid = trades.mid.to_numpy(float); size = trades["size"].to_numpy(float)
    ts = trades.timestamp_ns.to_numpy(np.int64); valid_mid = np.isfinite(mid)
    acf, response = {}, {}
    for lag in lags:
        if len(sign) > lag:
            acf[str(lag)] = float(np.mean(sign[:-lag] * sign[lag:]))
            ok = valid_mid[:-lag] & valid_mid[lag:]
            response[str(lag)] = float(np.mean(sign[:-lag][ok] * (mid[lag:][ok] - mid[:-lag][ok]))) if ok.any() else None
    impact = []
    if len(size) > 10:
        bins = pd.qcut(size, min(10, max(2, len(size) // 100)), duplicates="drop")
        temp = pd.DataFrame({"size": size[:-1], "impact": sign[:-1] * np.diff(mid), "bucket": bins[:-1]})
        for _, g in temp.replace([np.inf, -np.inf], np.nan).dropna().groupby("bucket", observed=True):
            impact.append({"size": float(g["size"].median()), "impact": float(g.impact.mean()), "n": int(len(g))})
    clock_size_impact, clock_response = {}, {}
    if len(trades) > 10 and len(quotes) > 10:
        qts = quotes.timestamp_ns.to_numpy(np.int64); qmid = quotes.mid.to_numpy(float)
        for seconds in flow_windows:
            target = ts + int(seconds * 1e9)
            ix = np.searchsorted(qts, target, side="left")
            ok = (ix < len(qts)) & valid_mid
            safe_ix = np.minimum(ix, len(qts) - 1)
            tolerance = max(1.0, min(10.0, seconds * .25)) * 1e9
            ok &= (qts[safe_ix] - target) <= tolerance
            impact_t = sign[ok] * (qmid[safe_ix[ok]] - mid[ok])
            clock_response[str(seconds)] = float(np.mean(impact_t)) if len(impact_t) else None
            size_t = size[ok]
            points = []
            if len(size_t) > 100:
                bins = pd.qcut(size_t, min(10, max(2, len(size_t) // 100)), duplicates="drop")
                temp = pd.DataFrame({"size": size_t, "impact": impact_t, "bucket": bins}).replace([np.inf, -np.inf], np.nan).dropna()
                for _, g in temp.groupby("bucket", observed=True):
                    points.append({"size": float(g["size"].median()), "impact": float(g.impact.mean()), "n": int(len(g))})
            clock_size_impact[str(seconds)] = points
    flows = {}
    if len(trades):
        t = pd.to_datetime(ts, unit="ns", utc=True)
        tmp = pd.DataFrame({"timestamp": t, "flow": sign * size, "mid": mid}).dropna().set_index("timestamp")
        for seconds in flow_windows:
            g = tmp.resample(f"{seconds}s").agg(flow=("flow", "sum"), first=("mid", "first"), last=("mid", "last")).dropna()
            g["change"] = g["last"] - g["first"]
            if len(g):
                qbin = pd.qcut(g.flow, min(10, max(2, len(g) // 20)), duplicates="drop")
                curve = g.groupby(qbin, observed=True).agg(flow=("flow", "mean"), change=("change", "mean"), n=("flow", "size"))
                flows[str(seconds)] = curve.reset_index(drop=True).to_dict("records")
    duration = (ts[-1] - ts[0]) / 1e9 if len(ts) > 1 else np.nan
    quote_spread = quotes.spread.dropna().to_numpy(float) if len(quotes) else np.array([])
    norm_spread = quote_spread / quotes.loc[quotes.spread.notna(), "mid"].to_numpy(float) if len(quote_spread) else np.array([])
    inter = np.diff(ts) / 1e9
    # One-second log returns in basis points. Event-time midpoint differences
    # are not comparable when real and simulated quote intensities differ.
    if len(quotes):
        clock_mid = pd.Series(
            quotes.mid.to_numpy(float),
            index=pd.to_datetime(quotes.timestamp_ns.to_numpy(np.int64), unit="ns", utc=True),
        ).groupby(level=0).last().resample("1s").last().ffill()
        ret = np.diff(np.log(clock_mid.to_numpy(float))) * 1e4
    else:
        ret = np.array([])
    distributions={"spread":_quantiles(quote_spread),"relative_spread":_quantiles(norm_spread),"trade_size":_quantiles(size),
        "interarrival":_quantiles(inter),"bid_depth":_quantiles(quotes.bid_size),"ask_depth":_quantiles(quotes.ask_size)}
    return {"n_trades": int(len(trades)), "duration_seconds": float(duration),
            "trades_per_second": float(len(trades) / duration) if duration > 0 else None,
            "spread": _q(quote_spread), "normalized_spread": _q(norm_spread),
            "trade_size": _q(size), "trade_interarrival_seconds": _q(inter),
            "sign_acf": acf, "response": response, "size_impact": impact,
            "clock_time_size_impact": clock_size_impact, "clock_response":clock_response, "flow_impact": flows,
            "return_volatility": float(np.nanstd(ret)) if len(ret) else None,
            "best_bid_depth": _q(quotes.bid_size), "best_ask_depth": _q(quotes.ask_size), "distributions":distributions,
            "price_change_frequency": float(np.mean(ret != 0)) if len(ret) else None}


def scalar_metrics(result):
    return {"median_spread": result.get("spread", {}).get("median"),
            "mean_trade_size": result.get("trade_size", {}).get("mean"),
            "trades_per_second": result.get("trades_per_second"),
            "sign_acf_1": result.get("sign_acf", {}).get("1"),
            "sign_acf_10": result.get("sign_acf", {}).get("10"),
            "response_1": result.get("response", {}).get("1"),
            "response_10": result.get("response", {}).get("10"),
            "return_volatility": result.get("return_volatility")}
