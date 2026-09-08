# Latest P0/P1 diagnosis

Synthetic run: `outputs\latest_p0p1\six_hour_simulation` (6.00 hours, 6,453 trades)

This report distinguishes an internal simulator gate from the reference framework's venue-specific estimators.

| Fact | Status | Diagnosis |
|---|---|---|
| P0-V 1 — Metaorder impact | **matched proxy** | TWAP probe estimates delta≈0.499 with power R²≈1; it is still a synthetic intervention, not a labeled real parent. |
| P0-V 2 — Impact trajectory | **matched shape** | TWAP trajectory is broadly increasing and concave-looking, but is not a reference-compatible parent-order estimator. |
| P0-V 3 — Post-metaorder decay | **partial** | Decay output exists, but the one-seed probe does not yet show statistically reliable recovery. |
| P0-V 4 — Impact surface | **matched proxy** | Size-impact slopes are estimated across forward horizons; true participation-rate surface remains a limitation. |
| P0-C 5 — Liquidity cost / depth | **partial** | Depth profile exists; virtual VWAP cost curve is not yet reported by compute(). |
| P0-C 6 — Spread distribution / state | **matched proxy** | One-tick share and spread PMF/CDF are checked; state-conditioned tail is simplified. |
| P0-C 7 — Liquidity resiliency | **partial** | A basic 1/5/30-second change probe exists, but no fitted recovery constant is gated. |
| P0-C 8 — OFI response | **matched proxy** | Slope and R² pass the simulator's PDF-priority tolerance; reference units/estimator differ. |
| P0-C 9 — Trade response | **matched shape** | Response is monotone, but magnitude is not reference-calibrated in the latest long run. |
| P0-C 10 — Order-sign memory | **matched proxy** | C(1), C(100), and a robust gamma diagnostic are available; very-long lags remain noisy. |
| P0-C 11 — Diffusive prices | **partial** | Clock-time variance ratios are now reported; subsequent-return predictability and quote-event comparison remain incomplete. |
| P1 16 — Trade-size tail | **matched proxy** | Tail ratio and top-1% volume share pass the PDF-priority tolerance. |

## Direct synthetic observations

- Spread: median `0.09999999999854481` USD; one-tick share `98.91%`; the PMF has 0.10/0.20/0.30 USD states because the maker rounds spread to integer ticks.
- Arrivals: mean interarrival `3.345` s; Fano(1s) `1.61`; Fano(300s) `13.61`. The long-horizon target is still materially under-dispersed relative to the reference.
- Sign memory: C(1) `0.245`; C(100) `0.00834251534708012`; robust gamma `0.503`.
- Trade response: R(1) `0.893` in simulator price units; the reference report uses bps, so magnitude comparison requires normalization.
- OFI: slope `1.128` and R² `0.496`; the reference framework reports a different OFI unit and correlation statistic.
- Impact surface: size slopes are `0.308`, `0.300`, `0.183`, `0.204` at 1/5/30/60 seconds.
- Diffusion: variance ratios are `1.000`, `1.065`, `1.158`, `1.761` at 1/5/10/60 seconds.
- TWAP intervention: fitted peak-impact exponent delta≈0.499, power R²≈0.999997; trajectory is broadly increasing, while post-execution recovery is not yet statistically established.

## Counts

- Matched proxy/shape groups: **8/12**.
- Partial groups: **4/12**.
- Not measured: **0/12**.

## Root causes

1. The three spread modes are caused by intentional integer-tick spread control plus flow-pressure widening; they are not a plotting failure.
2. Non-monotone size-impact buckets come from one-trade midpoint differences, not metaorder impact. Six hours reduces noise but does not identify parent-order causality.
3. The PDF-priority gate uses normalized distances and therefore allows economically large unit/estimator mismatches. It should not be presented as full market equivalence.
4. The long-run L3-style tape is suitable for simulator diagnostics, but it is not a reconstruction of a real venue's 100-level book semantics.

## Multi-seed impact benchmark

Every registered liquidation algorithm is measured against the same seeded control market on five seeds per preset. This checks the market's behaviour, not a policy ranking: a one-shot parent should move the price further than the same parent spread over the horizon.

```
preset,seeds,immediate_impact_above_twap,immediate_mean_impact_bp,twap_mean_impact_bp,immediate_mean_cost_bp,twap_mean_cost_bp
large_sell,5,5,0.2380804114118768,0.07935898686421204,0.16567508085325638,0.02718652972692513
standard,5,5,0.10952448137708941,0.03174581580144288,0.10061120038375397,0.019365036374203826
thin_depth,5,5,0.3079100085386973,0.07777206157524319,0.45007354318269177,0.031036201281402782
```