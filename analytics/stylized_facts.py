import numpy as np
import pandas as pd


LAGS = np.unique(np.r_[np.arange(1, 11), [20, 50, 100, 200, 500, 1000], np.round(np.logspace(np.log10(12), np.log10(1000), 28)).astype(int)])
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
    quote_columns = ["timestamp_ns", "best_bid", "best_ask", "bid_size", "ask_size", "mid", "spread"]
    quote_columns += [c for c in ("bid_depth_profile", "ask_depth_profile") if c in df]
    quotes = df[quote_mask & df.best_bid.notna() & df.best_ask.notna()][quote_columns]
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
    memory_fit = None
    # Fit over the PDF's identified range (approximately lags 10--100).
    # Longer-lag ACF estimates in a one-hour synthetic run are noisy and can
    # reverse the slope even when C(1) and C(100) have the right magnitudes.
    fit_lags=np.asarray([lag for lag in lags if 10 <= lag <= 100 and str(lag) in acf and acf[str(lag)] > 0], float)
    fit_acf=np.asarray([acf[str(int(lag))] for lag in fit_lags], float)
    if len(fit_lags) >= 3:
        # A single least-squares slope is very sensitive to one noisy ACF
        # point at the edge of a short synthetic sample.  The median of the
        # adjacent log-slopes preserves the power-law diagnostic while being
        # much more stable across random seeds.
        slopes = -np.diff(np.log(fit_acf)) / np.diff(np.log(fit_lags))
        slopes = slopes[np.isfinite(slopes)]
        memory_fit=float(np.clip(np.median(slopes), 0.0, 2.0)) if len(slopes) else None
    response_values=np.asarray([response[str(int(lag))] for lag in lags if str(lag) in response and response[str(lag)] is not None], float)
    response_monotone=float(np.mean(np.diff(response_values) >= -1e-12)) if len(response_values)>1 else None
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
    impact_surface={}
    for horizon,points in clock_size_impact.items():
        pos=[p for p in points if p.get("size",0)>0 and p.get("impact") is not None and p["impact"]>0]
        if len(pos)>=3:
            sx=np.asarray([p["size"] for p in pos],float); sy=np.asarray([p["impact"] for p in pos],float)
            slope=float(np.polyfit(np.log(sx),np.log(sy),1)[0])
            impact_surface[horizon]={"size_slope":slope,"size_monotone_fraction":float(np.mean(np.diff(sy)>=-1e-12)),"n":len(pos)}
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
    spread_one_tick_share = float(np.mean(np.isclose(quote_spread, 0.1, atol=1e-8))) if len(quote_spread) else None
    norm_spread = quote_spread / quotes.loc[quotes.spread.notna(), "mid"].to_numpy(float) if len(quote_spread) else np.array([])
    inter = np.diff(ts) / 1e9
    # Count dispersion is the directly measurable P0 signature that rejects
    # a Poisson taker process.  Report it at several wall-clock horizons.
    arrival_fano = {}
    if len(ts) > 10:
        tsec = ts / 1e9
        for horizon in (1, 10, 60, 300):
            edges = np.arange(tsec[0], tsec[-1] + horizon, horizon)
            counts, _ = np.histogram(tsec, edges)
            arrival_fano[str(horizon)] = float(np.var(counts) / np.mean(counts)) if len(counts) and np.mean(counts) else None
    profiles = {}
    for name in ("bid_depth_profile", "ask_depth_profile"):
        if name in quotes:
            rows = [x for x in quotes[name].tolist() if isinstance(x, (list, tuple)) and len(x)]
            if rows:
                profiles[name] = np.nanmedian(np.asarray(rows, float), axis=0).tolist()
    depth_shape = {}
    if profiles.get("bid_depth_profile") and profiles.get("ask_depth_profile"):
        for side in ("bid", "ask"):
            v=np.asarray(profiles[side + "_depth_profile"], float)
            depth_shape[side + "_monotone_fraction"] = float(np.mean(np.diff(v) >= -1e-12)) if len(v)>1 else None
            depth_shape[side + "_peak_level"] = int(np.argmax(v)) if len(v) else None
    liquidity_cost_curve={}
    if profiles and len(quotes):
        virtual_sizes=(.05,.1,.25,.5,1.,2.,5.)
        mid_level=float(np.nanmedian(quotes.mid.to_numpy(float)))
        for side in ("bid","ask"):
            depth=np.asarray(profiles.get(side+"_depth_profile",[]),float)
            if len(depth) and mid_level>0:
                cumulative=np.cumsum(depth); costs=[]
                for quantity in virtual_sizes:
                    ix=np.searchsorted(cumulative,quantity,side="left")
                    if ix>=len(depth): costs.append({"quantity":quantity,"cost_bps":None}); continue
                    used=np.diff(np.r_[0,cumulative[:ix]])
                    remainder=quantity-used.sum()
                    qty=np.r_[used,remainder]
                    distances=(np.arange(ix+1)+.5)*0.1
                    costs.append({"quantity":quantity,"cost_bps":float(np.sum(qty*distances)/quantity/mid_level*1e4)})
                valid=[x for x in costs if x["cost_bps"] is not None]
                if len(valid)>=2: liquidity_cost_curve[side]={"points":costs,"monotone_fraction":float(np.mean(np.diff([x["cost_bps"] for x in valid])>=-1e-12))}
    size_tail = {}
    if len(size):
        median_size = float(np.median(size))
        size_tail = {
            "p99_to_median": float(np.quantile(size, .99) / max(median_size, 1e-12)),
            "top_1pct_volume_share": float(np.sort(size)[-max(1, len(size) // 100):].sum() / max(size.sum(), 1e-12)),
        }
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
    diffusivity={}
    if len(ret)>100:
        base_var=float(np.var(ret));
        if base_var>0:
            for horizon in (1,5,10,60):
                n=len(ret)//horizon; grouped=ret[:n*horizon].reshape(n,horizon).sum(axis=1)
                diffusivity[str(horizon)] = float(np.var(grouped)/(horizon*base_var)) if len(grouped)>5 else None
    # Cont-Kukanov-Stoikov order-flow imbalance on quote updates. This P1
    # diagnostic tests whether book events explain price changes beyond trades.
    ofi_response = {}
    if len(quotes) > 3:
        q = quotes.dropna(subset=["best_bid", "best_ask", "bid_size", "ask_size", "mid"])
        bid, ask = q.best_bid.to_numpy(float), q.best_ask.to_numpy(float)
        bs, a_s = q.bid_size.to_numpy(float), q.ask_size.to_numpy(float)
        ofi = np.zeros(len(q))
        if len(q) > 1:
            ofi[1:] = (np.where(bid[1:] > bid[:-1], bs[1:], np.where(bid[1:] == bid[:-1], bs[1:] - bs[:-1], -bs[:-1]))
                       + np.where(ask[1:] < ask[:-1], -a_s[1:], np.where(ask[1:] == ask[:-1], -(a_s[1:] - a_s[:-1]), a_s[:-1])))
        block = 50; nblocks = len(q) // block
        if nblocks > 3:
            x = np.add.reduceat(ofi[:nblocks * block], np.arange(0, nblocks * block, block))[:-1]
            mids = q.mid.to_numpy(float)
            y = mids[block:nblocks * block:block] - mids[:(nblocks - 1) * block:block]
            slope, intercept = np.polyfit(x, y, 1); fitted = intercept + slope * x
            ss_res = float(np.sum((y - fitted) ** 2)); ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            ofi_response = {"block_quote_events": block, "slope": float(slope),
                            "r2": float(1.0 - ss_res / ss_tot) if ss_tot else None,
                            "n_blocks": int(nblocks)}
    # P0 resiliency probe: after the largest aggressive prints, measure how
    # quickly spread and L1 depth return toward their pre-trade state.
    resiliency = {}
    if len(trades) > 20 and len(quotes) > 20:
        qts = quotes.timestamp_ns.to_numpy(np.int64)
        qspread = quotes.spread.to_numpy(float)
        qdepth = (quotes.bid_size.to_numpy(float) + quotes.ask_size.to_numpy(float)) / 2.0
        cutoff = np.quantile(size, .99)
        selected = np.flatnonzero(size >= cutoff)
        for horizon in (1, 5, 30):
            values = []
            for i in selected[:5000]:
                qi = np.searchsorted(qts, ts[i], side="right") - 1
                after = np.searchsorted(qts, ts[i] + horizon * 1_000_000_000, side="left")
                if qi >= 0 and after < len(qts) and np.isfinite(qspread[qi]) and np.isfinite(qspread[after]):
                    values.append({"spread_change": float(qspread[after] - qspread[qi]),
                                   "depth_change": float(qdepth[after] - qdepth[qi])})
            if values:
                resiliency[str(horizon)] = {k: float(np.mean([v[k] for v in values])) for k in values[0]}
    distributions={"spread":_quantiles(quote_spread),"relative_spread":_quantiles(norm_spread),"trade_size":_quantiles(size),
        "interarrival":_quantiles(inter),"bid_depth":_quantiles(quotes.bid_size),"ask_depth":_quantiles(quotes.ask_size)}
    return {"n_trades": int(len(trades)), "duration_seconds": float(duration),
            "trades_per_second": float(len(trades) / duration) if duration > 0 else None,
            "spread": _q(quote_spread), "normalized_spread": _q(norm_spread),
            "trade_size": _q(size), "trade_interarrival_seconds": _q(inter),
            "sign_acf": acf, "response": response, "size_impact": impact,
            "sign_memory_exponent": memory_fit, "response_monotone_fraction": response_monotone,
            "clock_time_size_impact": clock_size_impact, "clock_response":clock_response, "flow_impact": flows,
            "return_volatility": float(np.nanstd(ret)) if len(ret) else None,
            "impact_surface": impact_surface, "diffusivity": diffusivity,
            "liquidity_cost_curve": liquidity_cost_curve,
            "arrival_fano": arrival_fano, "depth_profile": profiles,
            "spread_one_tick_share": spread_one_tick_share,
            "ofi_response": ofi_response,
            "resiliency": resiliency,
            "trade_size_tail": size_tail,
            "depth_shape": depth_shape,
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
