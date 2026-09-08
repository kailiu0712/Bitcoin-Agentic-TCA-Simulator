from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _style(ax, xlabel, ylabel, logx=False, logy=False):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.grid(True, which="both", alpha=.22, linewidth=.7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def _dense_curve(x, y, logx=False, points=240):
    x=np.asarray(x,float); y=np.asarray(y,float); ok=np.isfinite(x)&np.isfinite(y)
    x,y=x[ok],y[ok]; order=np.argsort(x); x,y=x[order],y[order]
    x,idx=np.unique(x,return_index=True); y=y[idx]
    if len(x)<2: return x,y
    grid=np.linspace(np.log(x.min()),np.log(x.max()),points) if logx and np.all(x>0) else np.linspace(x.min(),x.max(),points)
    gx=np.exp(grid) if logx and np.all(x>0) else grid
    return gx,np.interp(grid,np.log(x),y) if logx and np.all(x>0) else np.interp(gx,x,y)


def _isotonic_increasing(y):
    """Pool-adjacent-violators fit used only as a transparent trend overlay."""
    values=[float(v) for v in y]; weights=[1.0]*len(values); i=0
    while i < len(values)-1:
        if values[i] <= values[i+1]: i+=1; continue
        total=values[i]*weights[i]+values[i+1]*weights[i+1]; weight=weights[i]+weights[i+1]
        values[i]=total/weight; weights[i]=weight; del values[i+1]; del weights[i+1]
        i=max(0,i-1)
    fitted=[]
    for value,weight in zip(values,weights): fitted.extend([value]*int(weight))
    return np.asarray(fitted[:len(y)],float)


def _save(out, name):
    plt.tight_layout(); plt.savefig(Path(out) / name, dpi=180, facecolor="white"); plt.close()


def make_plots(df, metrics, out, prefix="real"):
    from .stylized_facts import prepare
    out = Path(out); out.mkdir(parents=True, exist_ok=True); trades, quotes = prepare(df)
    # The reference report uses a time-weighted CDF for spread. A histogram
    # makes this tick-quantized simulator look like it has arbitrary modes.
    spread=np.asarray(quotes.spread.dropna(),float)
    if len(spread):
        values,counts=np.unique(np.round(spread,10),return_counts=True); cdf=np.cumsum(counts)/counts.sum()
        fig,ax=plt.subplots(figsize=(7.2,4.4)); ax.step(values,cdf,where="post",color="#2563eb",lw=2); ax.scatter(values,cdf,color="#0f172a",s=24,zorder=3)
        _style(ax,"Quoted spread (USD)","P(spread ≤ x)",logx=True); _save(out,f"{prefix}_spread_distribution.png")
        fig,ax=plt.subplots(figsize=(7.2,4.4)); ax.bar(values,counts/counts.sum(),width=max(np.min(np.diff(values))/2 if len(values)>1 else .01,.005),color="#2563eb",alpha=.85)
        _style(ax,"Quoted spread (USD)","Probability",logx=True); _save(out,f"{prefix}_spread_pmf.png")
    specs = [(trades["size"], "Trade size (BTC)", "trade_size_distribution.png", True),
             (np.diff(trades.timestamp_ns) / 1e9, "Inter-arrival (seconds)", "trade_interarrival.png", True),
             (np.diff(quotes.mid.dropna()), "Midpoint change (USD)", "return_distribution.png", False)]
    for x, label, fn, log in specs:
        x = np.asarray(x); x = x[np.isfinite(x) & (x > 0 if log else np.ones(len(x), bool))]
        plt.figure(); plt.hist(x, bins=80, log=True); plt.xlabel(label); plt.ylabel("Count")
        if log: plt.xscale("log")
        _save(out, f"{prefix}_{fn}")
    for key, label, fn in (("sign_acf", "Trade-sign autocorrelation", "sign_acf.png"), ("response", "Response (USD)", "response.png")):
        curve = metrics[key]; x=np.array(list(map(int, curve))); y=np.array(list(curve.values()), float)
        dx,dy=_dense_curve(x,y,logx=True)
        fig,ax=plt.subplots(figsize=(7.2,4.4)); ax.plot(dx,dy,color="#2563eb",lw=2); ax.scatter(x,y,color="#0f172a",s=24,zorder=3)
        _style(ax,"Trade-time lag",label,logx=True); _save(out, f"{prefix}_{fn}")
    pts=metrics["size_impact"]
    if pts:
        x=[p["size"] for p in pts]; y=[p["impact"] for p in pts]
        trend=_isotonic_increasing(y); dx,dy=_dense_curve(x,trend,logx=True)
        fig,ax=plt.subplots(figsize=(7.2,4.4)); ax.scatter(x,y,color="#64748b",s=24,label="raw bucket means"); ax.plot(dx,dy,color="#dc2626",lw=2,label="monotone trend")
        _style(ax,"Trade size (BTC)","R(v,1), USD",logx=True); ax.legend(frameon=False); _save(out,f"{prefix}_size_impact.png")
        pos=[(a,b) for a,b in zip(x,y) if a>0 and b>0]
        if pos:
            px,py=zip(*pos); trend=_isotonic_increasing(py); dx,dy=_dense_curve(px,trend,logx=True)
            fig,ax=plt.subplots(figsize=(7.2,4.4)); ax.loglog(px,py,"o",color="#64748b",label="raw positive buckets"); ax.loglog(dx,dy,color="#dc2626",lw=2,label="monotone trend")
            _style(ax,"Trade size (BTC)","Positive impact",logx=True,logy=True); ax.legend(frameon=False); _save(out,f"{prefix}_scaling_loglog.png")
    if metrics["flow_impact"]:
        plt.figure()
        fig,ax=plt.subplots(figsize=(7.2,4.4))
        for w, pts in metrics["flow_impact"].items():
            xx=[p["flow"] for p in pts]; yy=[p["change"] for p in pts]; dx,dy=_dense_curve(xx,yy)
            ax.plot(dx,dy,lw=2,label=f"{w}s"); ax.scatter(xx,yy,s=18)
        _style(ax,"Signed traded volume (BTC)","Midpoint change (USD)"); ax.legend(frameon=False); _save(out,f"{prefix}_flow_impact.png")
    if metrics.get("impact_surface"):
        fig,ax=plt.subplots(figsize=(7.2,4.4)); rows=[]
        for horizon,value in metrics["impact_surface"].items(): rows.append((float(horizon),value.get("size_slope")))
        rows=[r for r in rows if np.isfinite(r[1])]; rows.sort()
        if rows:
            xx,yy=zip(*rows); ax.plot(xx,yy,"o-",color="#7c3aed",lw=2); _style(ax,"Forward horizon (seconds)","Log-log size-impact slope"); _save(out,f"{prefix}_impact_surface.png")
    if metrics.get("diffusivity"):
        rows=[(float(k),v) for k,v in metrics["diffusivity"].items() if v is not None and np.isfinite(v)]
        if rows:
            rows.sort(); xx,yy=zip(*rows); fig,ax=plt.subplots(figsize=(7.2,4.4)); ax.semilogx(xx,yy,"o-",color="#0891b2",lw=2); ax.axhline(1,color="#64748b",ls="--",lw=1); _style(ax,"Aggregation horizon (seconds)","Variance ratio",logx=True); _save(out,f"{prefix}_diffusivity.png")
    if metrics.get("liquidity_cost_curve"):
        fig,ax=plt.subplots(figsize=(7.2,4.4))
        for side,color in (("bid","#dc2626"),("ask","#2563eb")):
            points=[p for p in metrics["liquidity_cost_curve"].get(side,{}).get("points",[]) if p.get("cost_bps") is not None]
            if points: ax.plot([p["quantity"] for p in points],[p["cost_bps"] for p in points],"o-",lw=2,color=color,label=side)
        _style(ax,"Virtual order size (BTC)","Estimated VWAP cost (bps)",logx=True); ax.legend(frameon=False); _save(out,f"{prefix}_liquidity_cost_curve.png")
    if len(quotes):
        q=quotes.iloc[:min(5000,len(quotes))]; t=(q.timestamp_ns-q.timestamp_ns.iloc[0])/1e9
        fig,ax=plt.subplots(figsize=(9,4.4)); ax.plot(t,q.best_bid,label="bid",lw=1); ax.plot(t,q.best_ask,label="ask",lw=1); ax.plot(t,q.mid,label="mid",lw=1.4)
        _style(ax,"Seconds","Price (USD)"); ax.legend(frameon=False,ncol=3); _save(out,f"{prefix}_sample_l1_path.png")
