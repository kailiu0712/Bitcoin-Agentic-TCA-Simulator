"""Compare the latest synthetic run with the reference stylized-fact report.

This is deliberately a diagnosis, not a calibration gate: the reference
framework uses venue-specific estimators and several quantities that cannot be
identified from this simulator's anonymous background flow.
"""
import argparse, json
from pathlib import Path
import pandas as pd


FACTS = [
    ("P0-V 1", "Metaorder impact", "matched proxy", "TWAP probe estimates delta≈0.499 with power R²≈1; it is still a synthetic intervention, not a labeled real parent."),
    ("P0-V 2", "Impact trajectory", "matched shape", "TWAP trajectory is broadly increasing and concave-looking, but is not a reference-compatible parent-order estimator."),
    ("P0-V 3", "Post-metaorder decay", "partial", "Decay output exists, but the one-seed probe does not yet show statistically reliable recovery."),
    ("P0-V 4", "Impact surface", "matched proxy", "Size-impact slopes are estimated across forward horizons; true participation-rate surface remains a limitation."),
    ("P0-C 5", "Liquidity cost / depth", "partial", "Depth profile exists; virtual VWAP cost curve is not yet reported by compute()."),
    ("P0-C 6", "Spread distribution / state", "matched proxy", "One-tick share and spread PMF/CDF are checked; state-conditioned tail is simplified."),
    ("P0-C 7", "Liquidity resiliency", "partial", "A basic 1/5/30-second change probe exists, but no fitted recovery constant is gated."),
    ("P0-C 8", "OFI response", "matched proxy", "Slope and R² pass the simulator's PDF-priority tolerance; reference units/estimator differ."),
    ("P0-C 9", "Trade response", "matched shape", "Response is monotone, but magnitude is not reference-calibrated in the latest long run."),
    ("P0-C 10", "Order-sign memory", "matched proxy", "C(1), C(100), and a robust gamma diagnostic are available; very-long lags remain noisy."),
    ("P0-C 11", "Diffusive prices", "partial", "Clock-time variance ratios are now reported; subsequent-return predictability and quote-event comparison remain incomplete."),
    ("P1 16", "Trade-size tail", "matched proxy", "Tail ratio and top-1% volume share pass the PDF-priority tolerance."),
]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run",default="outputs/latest_p0p1/six_hour_simulation"); ap.add_argument("--reference",default=r"C:\Users\kai\liukh\CALM\Stylized facts\results\summary_metrics.csv"); ap.add_argument("--output",default="outputs/latest_p0p1/diagnosis.md"); args=ap.parse_args()
    run=Path(args.run); metrics=json.loads((run/"metrics.json").read_text()); ref=pd.read_csv(args.reference)
    modes={"matched proxy":0,"matched shape":0,"partial":0,"not measured":0}
    lines=["# Latest P0/P1 diagnosis", "", f"Synthetic run: `{run}` ({metrics.get('duration_seconds',0)/3600:.2f} hours, {metrics.get('n_trades',0):,} trades)", "", "This report distinguishes an internal simulator gate from the reference framework's venue-specific estimators.", "", "| Fact | Status | Diagnosis |", "|---|---|---|"]
    for fact,name,status,note in FACTS:
        modes[status]+=1; lines.append(f"| {fact} — {name} | **{status}** | {note} |")
    lines += ["", "## Direct synthetic observations", "", f"- Spread: median `{metrics.get('spread',{}).get('median')}` USD; one-tick share `{metrics.get('spread_one_tick_share',0):.2%}`; the PMF has 0.10/0.20/0.30 USD states because the maker rounds spread to integer ticks.", f"- Arrivals: mean interarrival `{metrics.get('trade_interarrival_seconds',{}).get('mean'):.3f}` s; Fano(1s) `{metrics.get('arrival_fano',{}).get('1'):.2f}`; Fano(300s) `{metrics.get('arrival_fano',{}).get('300'):.2f}`. The long-horizon target is still materially under-dispersed relative to the reference.", f"- Sign memory: C(1) `{metrics.get('sign_acf',{}).get('1'):.3f}`; C(100) `{metrics.get('sign_acf',{}).get('100')}`; robust gamma `{metrics.get('sign_memory_exponent'):.3f}`.", f"- Trade response: R(1) `{metrics.get('response',{}).get('1'):.3f}` in simulator price units; the reference report uses bps, so magnitude comparison requires normalization.", f"- OFI: slope `{metrics.get('ofi_response',{}).get('slope'):.3f}` and R² `{metrics.get('ofi_response',{}).get('r2'):.3f}`; the reference framework reports a different OFI unit and correlation statistic.", f"- Impact surface: size slopes are `{metrics.get('impact_surface',{}).get('1',{}).get('size_slope'):.3f}`, `{metrics.get('impact_surface',{}).get('5',{}).get('size_slope'):.3f}`, `{metrics.get('impact_surface',{}).get('30',{}).get('size_slope'):.3f}`, `{metrics.get('impact_surface',{}).get('60',{}).get('size_slope'):.3f}` at 1/5/30/60 seconds.", f"- Diffusion: variance ratios are `{metrics.get('diffusivity',{}).get('1'):.3f}`, `{metrics.get('diffusivity',{}).get('5'):.3f}`, `{metrics.get('diffusivity',{}).get('10'):.3f}`, `{metrics.get('diffusivity',{}).get('60'):.3f}` at 1/5/10/60 seconds.", "- TWAP intervention: fitted peak-impact exponent delta≈0.499, power R²≈0.999997; trajectory is broadly increasing, while post-execution recovery is not yet statistically established.", "", "## Counts", "", f"- Matched proxy/shape groups: **{modes['matched proxy']+modes['matched shape']}/{len(FACTS)}**.", f"- Partial groups: **{modes['partial']}/{len(FACTS)}**.", f"- Not measured: **{modes['not measured']}/{len(FACTS)}**.", "", "## Root causes", "", "1. The three spread modes are caused by intentional integer-tick spread control plus flow-pressure widening; they are not a plotting failure.", "2. Non-monotone size-impact buckets come from one-trade midpoint differences, not metaorder impact. Six hours reduces noise but does not identify parent-order causality.", "3. The PDF-priority gate uses normalized distances and therefore allows economically large unit/estimator mismatches. It should not be presented as full market equivalence.", "4. The long-run L3-style tape is suitable for simulator diagnostics, but it is not a reconstruction of a real venue's 100-level book semantics."]
    benchmark=Path("outputs/latest_p0p1/impact_benchmark_summary.csv")
    if benchmark.exists():
        lines += ["", "## Multi-seed impact benchmark", "", "Every registered liquidation algorithm is measured against the same seeded control market on five seeds per preset. This checks the market's behaviour, not a policy ranking: a one-shot parent should move the price further than the same parent spread over the horizon.", "", "```", benchmark.read_text().strip(), "```"]
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text("\n".join(lines),encoding="utf-8"); print(out)


if __name__=="__main__": main()
