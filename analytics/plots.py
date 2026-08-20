from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(out, name):
    plt.tight_layout(); plt.savefig(Path(out) / name, dpi=150); plt.close()


def make_plots(df, metrics, out, prefix="real"):
    from .stylized_facts import prepare
    out = Path(out); out.mkdir(parents=True, exist_ok=True); trades, quotes = prepare(df)
    specs = [(quotes.spread.dropna(), "Spread (USD)", "spread_distribution.png", False),
             (trades["size"], "Trade size (BTC)", "trade_size_distribution.png", True),
             (np.diff(trades.timestamp_ns) / 1e9, "Inter-arrival (seconds)", "trade_interarrival.png", True),
             (np.diff(quotes.mid.dropna()), "Midpoint change (USD)", "return_distribution.png", False)]
    for x, label, fn, log in specs:
        x = np.asarray(x); x = x[np.isfinite(x) & (x > 0 if log else np.ones(len(x), bool))]
        plt.figure(); plt.hist(x, bins=80, log=True); plt.xlabel(label); plt.ylabel("Count")
        if log: plt.xscale("log")
        _save(out, f"{prefix}_{fn}")
    for key, label, fn in (("sign_acf", "Trade-sign autocorrelation", "sign_acf.png"), ("response", "Response (USD)", "response.png")):
        curve = metrics[key]; x=np.array(list(map(int, curve))); y=np.array(list(curve.values()), float)
        plt.figure(); plt.plot(x, y, marker="o"); plt.xlabel("Trade-time lag"); plt.ylabel(label); _save(out, f"{prefix}_{fn}")
    pts=metrics["size_impact"]
    if pts:
        x=[p["size"] for p in pts]; y=[p["impact"] for p in pts]
        plt.figure(); plt.plot(x,y,marker="o"); plt.xlabel("Trade size (BTC)"); plt.ylabel("R(v,1), USD"); _save(out,f"{prefix}_size_impact.png")
        pos=[(a,b) for a,b in zip(x,y) if a>0 and b>0]
        if pos:
            plt.figure(); plt.loglog(*zip(*pos),marker="o"); plt.xlabel("Trade size (BTC)"); plt.ylabel("Positive impact"); _save(out,f"{prefix}_scaling_loglog.png")
    if metrics["flow_impact"]:
        plt.figure()
        for w, pts in metrics["flow_impact"].items(): plt.plot([p["flow"] for p in pts],[p["change"] for p in pts],marker=".",label=f"{w}s")
        plt.xlabel("Signed traded volume (BTC)"); plt.ylabel("Midpoint change (USD)"); plt.legend(); _save(out,f"{prefix}_flow_impact.png")
    if len(quotes):
        q=quotes.iloc[:min(5000,len(quotes))]; t=(q.timestamp_ns-q.timestamp_ns.iloc[0])/1e9
        plt.figure(); plt.plot(t,q.best_bid,label="bid"); plt.plot(t,q.best_ask,label="ask"); plt.plot(t,q.mid,label="mid"); plt.xlabel("Seconds"); plt.legend(); _save(out,f"{prefix}_sample_l1_path.png")
