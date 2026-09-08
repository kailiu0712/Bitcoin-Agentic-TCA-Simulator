import numpy as np
from analytics.stylized_facts import scalar_metrics


EPS=1e-8


def _nrmse(a,b,scale=None):
    a=np.asarray(a,float);b=np.asarray(b,float);ok=np.isfinite(a)&np.isfinite(b)
    if not ok.any():return None
    if scale is None:scale=max(np.nanstd(b[ok]),np.nanmean(np.abs(b[ok])),EPS)
    return float(np.sqrt(np.mean((a[ok]-b[ok])**2))/max(scale,EPS))


def _curve_values(curve,key):
    return [x[key] for x in curve if x.get(key) is not None]


def _aligned_curve_nrmse(sim_curve, target_curve, x_key, y_key, log_x=False):
    """Compare curves at common x coordinates instead of matching bin indices."""
    sx=np.asarray(_curve_values(sim_curve,x_key),float);sy=np.asarray(_curve_values(sim_curve,y_key),float)
    tx=np.asarray(_curve_values(target_curve,x_key),float);ty=np.asarray(_curve_values(target_curve,y_key),float)
    sok=np.isfinite(sx)&np.isfinite(sy);tok=np.isfinite(tx)&np.isfinite(ty);sx,sy,tx,ty=sx[sok],sy[sok],tx[tok],ty[tok]
    if log_x:
        sp=sx>0;tp=tx>0;sx,sy,tx,ty=np.log(sx[sp]),sy[sp],np.log(tx[tp]),ty[tp]
    if len(sx)<2 or len(tx)<2:return None
    si=np.argsort(sx);ti=np.argsort(tx);sx,sy,tx,ty=sx[si],sy[si],tx[ti],ty[ti]
    lo=max(sx[0],tx[0]);hi=min(sx[-1],tx[-1])
    if not hi>lo:return None
    grid=np.linspace(lo,hi,21)
    return _nrmse(np.interp(grid,sx,sy),np.interp(grid,tx,ty))


def calibration_loss(sim,target,weights):
    rows=[]
    def add(name,real,value,distance):
        if distance is None or not np.isfinite(distance):return
        weight=float(weights.get(name,1.0));rows.append({"metric":name,"real_target":real,"simulated_target":value,
            "absolute_error":None if not np.isscalar(real) else float(abs(value-real)),
            "relative_error":None if not np.isscalar(real) else float(abs(value-real)/(abs(real)+EPS)),
            "normalized_distance":float(distance),"weight":weight,"contribution":weight*float(distance)})
    sm,tm=scalar_metrics(sim),scalar_metrics(target)
    scalar_names=["median_spread","mean_trade_size","trades_per_second","return_volatility"]
    for name in scalar_names:
        sv,tv=sm.get(name),tm.get(name)
        if sv is not None and tv is not None:add(name,tv,sv,abs(sv-tv)/(abs(tv)+EPS))
    dist_map={"spread_distribution":"spread","relative_spread_distribution":"relative_spread","trade_size_distribution":"trade_size",
        "interarrival_distribution":"interarrival","bid_depth_distribution":"bid_depth","ask_depth_distribution":"ask_depth"}
    for name,key in dist_map.items():
        a=sim.get("distributions",{}).get(key,{}).get("values",[]);b=target.get("distributions",{}).get(key,{}).get("values",[])
        if a and b:add(name,b,a,_nrmse(a,b))
    for name,key in (("sign_acf_curve","sign_acf"),("response_curve","response")):
        common=sorted(set(sim.get(key,{}))&set(target.get(key,{})),key=int)
        if common:
            a=[sim[key][x] for x in common];b=[target[key][x] for x in common];add(name,b,a,_nrmse(a,b))
    # Optional P0/P1 extensions.  Older target files remain valid because
    # these terms are only scored when the evidence target contains them.
    for name, key in (("arrival_clustering", "arrival_fano"),):
        common=sorted(set(sim.get(key, {})) & set(target.get(key, {})), key=int)
        if common:
            a=[sim[key][x] for x in common]; b=[target[key][x] for x in common]
            add(name, b, a, _nrmse(a, b))
    for name, key in (("book_shape_bid", "bid_depth_profile"), ("book_shape_ask", "ask_depth_profile")):
        a=sim.get("depth_profile", {}).get(key, []); b=target.get("depth_profile", {}).get(key, [])
        if a and b: add(name, b, a, _nrmse(a, b))
    for metric in ("p99_to_median", "top_1pct_volume_share"):
        sv=sim.get("trade_size_tail", {}).get(metric); tv=target.get("trade_size_tail", {}).get(metric)
        if sv is not None and tv is not None: add("trade_size_tail_" + metric, tv, sv, abs(sv-tv)/(abs(tv)+EPS))
    sv=sim.get("spread_one_tick_share"); tv=target.get("spread_one_tick_share")
    if sv is not None and tv is not None: add("spread_one_tick_share", tv, sv, abs(sv-tv)/(abs(tv)+EPS))
    for metric in ("slope", "r2"):
        sv=sim.get("ofi_response", {}).get(metric); tv=target.get("ofi_response", {}).get(metric)
        if sv is not None and tv is not None: add("ofi_" + metric, tv, sv, abs(sv-tv)/(abs(tv)+EPS))
    for metric in ("sign_memory_exponent", "response_monotone_fraction"):
        sv=sim.get(metric); tv=target.get(metric)
        if sv is not None and tv is not None: add(metric, tv, sv, abs(sv-tv)/(abs(tv)+EPS))
    for side in ("bid", "ask"):
        sv=sim.get("depth_shape", {}).get(side + "_monotone_fraction")
        tv=target.get("depth_shape_" + side)
        if sv is not None and tv is not None: add("depth_shape_" + side, tv, sv, abs(sv-tv)/(abs(tv)+EPS))
    common=sorted(set(sim.get("clock_response",{}))&set(target.get("clock_response",{})),key=int)
    if common:
        a=[sim["clock_response"][x] for x in common];b=[target["clock_response"][x] for x in common]
        add("clock_response_curve",b,a,_nrmse(a,b))
    for name,key in (("size_impact_curve","size_impact"),):
        distance=_aligned_curve_nrmse(sim.get(key,[]),target.get(key,[]),"size","impact",log_x=True)
        if distance is not None:add(name,target.get(key,[]),sim.get(key,[]),distance)
    flow=[]
    for horizon in sorted(set(sim.get("flow_impact",{}))&set(target.get("flow_impact",{})),key=int):
        distance=_aligned_curve_nrmse(sim["flow_impact"][horizon],target["flow_impact"][horizon],"flow","change")
        if distance is not None:flow.append(distance)
    if flow:add("flow_impact_curve",flow,flow,float(np.mean(flow)))
    return float(sum(x["contribution"] for x in rows)),rows
