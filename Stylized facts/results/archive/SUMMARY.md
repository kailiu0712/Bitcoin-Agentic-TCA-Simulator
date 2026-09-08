# BTC/USD stylized facts - normal vs stress week

Generated 2026-08-29 01:19 from the Kraken BTC/USD L3 feed.

## Regimes

| Regime | Dates | Measured hours | Raw rows | Aggressive orders | Quote updates |
|---|---|---|---|---|---|
| **normal** | 2026-07-18 to 2026-07-24 | 167.954 | 85,642,266 | 187,779 | 4,490,391 |
| **stress** | 2026-07-26 to 2026-08-02 | 191.956 | 91,820,638 | 210,208 | 6,199,245 |

Book reconstruction: 314 s (from cache this run); analysis: 136 s.

## Headline results

### fact00 - Data quality and book-reconstruction validation  
*Priority: validation*

> **Literature:** n/a - this section validates the reconstruction itself.

| Metric | normal | stress | Note |
|---|---|---|---|
| crossed packets after repair | 0 | 0 | must be 0 |
| aggressor-side agreement (exchange-aligned) | 0.9921 | 0.9875 | share of aggressive orders printing on the correct side of the pre-trade quote; compare the receipt-aligned row below |
| aggressor-side agreement (receipt-aligned, naive) | 0.7767 | 0.7334 | what you get without exchange-time channel alignment |
| snapshot checks with best quotes exact | 0.7812 | 0.7381 | 42 independent checks |

### fact05 - Instantaneous liquidity-cost curve / cumulative LOB depth  
*Priority: P0-C*

> **Literature:** Schnaubelt, Rende & Krauss (2019): BTC/USD books are shallow and VWAP cost rises steeply with size.

| Metric | normal | stress | Note |
|---|---|---|---|
| cost-curve log-log slope (sell into bids) | 0.3597 | 0.4597 | R2=0.993 |
| cost-curve log-log slope (buy from asks) | 0.3913 | 0.4577 | R2=0.984 |
| cost of 1 BTC (sell into bids), bps | 0.7966 | 0.6692 |  |
| cost of 1 BTC (buy from asks), bps | 0.6113 | 0.5740 |  |
| mean depth within 10 bps (BTC, both sides) | 158.998 | 156.473 |  |

### fact06 - Bid-ask spread distribution and state dependence  
*Priority: P0-C*

> **Literature:** Schnaubelt, Rende & Krauss (2019): substantial hourly variation, no universal U-shape across venues.

| Metric | normal | stress | Note |
|---|---|---|---|
| median quoted spread (bps, time-weighted) | 0.0154 | 0.0159 | 0.10 USD at 65k |
| 99th pct quoted spread (bps) | 0.5446 | 0.6214 | 39.1x the median |
| hourly spread max/min ratio | 2.382 | 3.110 | U-shape test |
| hourly spread coefficient of variation | 0.2645 | 0.3196 |  |

### fact07 - Liquidity resiliency after large aggressive trades  
*Priority: P0-C*

> **Literature:** Schnaubelt et al.: spread recovery tau ~ 6.3s on Coinbase BTC/USD; deep-book liquidity recovers more slowly.

| Metric | normal | stress | Note |
|---|---|---|---|
| large-order size threshold (BTC) | 0.9116 | 0.8535 | top 1% |
| spread at the next order / pre-event | 1.489 | 1.282 | event-time peak dislocation |
| spread recovery time constant (s) | 14.889 | 8.155 | literature 6.3s; fit R2=0.58 |
| orders until spread is back within 1% of pre-event | 13.000 | 5.000 | event-time spread resiliency |
| orders until consumed-side depth is back to 99% | 34.000 | 21.000 | compare with the spread row above: depth recovering later is the Schnaubelt et al. ordering |

### fact08 - Order-flow imbalance -> short-horizon price response  
*Priority: P0-C*

> **Literature:** Cont, Kukanov & Stoikov (2014): mid changes linear in OFI, coefficient inversely related to depth.

| Metric | normal | stress | Note |
|---|---|---|---|
| OFI impact coefficient (bps per BTC) | 0.1552 | 0.1640 | R2=0.394, n=123,984 |
| OFI / mid-change correlation | 0.6129 | 0.6277 | blocks of 50 quote events |
| impact coefficient, thinnest / thickest depth quartile | 1.913 | 1.952 | CKS predicts > 1 (inverse in depth) |

### fact09 - Trade-response function R(l)  
*Priority: P0-C*

> **Literature:** Bouchaud, Gefen, Potters & Wyart (2004): stable, slowly growing aggregate response despite persistent order flow.

| Metric | normal | stress | Note |
|---|---|---|---|
| R(1) immediate response (bps) | 0.2482 | 0.2691 |  |
| R(100) (bps) | 0.8266 | 0.8095 |  |
| R(1000) (bps) | 0.6392 | 0.9167 |  |
| R(100)/R(1) growth factor | 3.330 | 3.008 | slow growth = adaptive liquidity absorbing persistent flow |
| mean effective half-spread paid (bps) | 0.1255 | 0.1083 |  |

### fact10 - Long memory of order signs  
*Priority: P0-C*

> **Literature:** Lillo & Farmer (2004): C(l) ~ l^-gamma with gamma ~ 0.6 on the LSE; long memory if gamma < 1.

| Metric | normal | stress | Note |
|---|---|---|---|
| sign-ACF decay exponent gamma (aggressive orders) | 0.5433 | 0.4314 | R2=0.774, lags 10-1000; literature ~0.6 |
| sign-ACF decay exponent gamma (individual fills) | 0.6569 | 0.5000 | R2=0.927 |
| C(1) sign autocorrelation | 0.2315 | 0.2629 |  |
| C(100) sign autocorrelation | 0.0111 | 0.0273 | long memory if still well above the noise floor |
| long memory (gamma < 1) | 1.000 | 1.000 | 1 = yes |

### fact11 - Persistent order flow with approximately diffusive prices  
*Priority: P0-C*

> **Literature:** Lillo-Farmer / Bouchaud et al.: adaptive liquidity keeps prices near-diffusive despite persistent signs; Schnaubelt et al.: little return autocorrelation from minutes to days, negative first-lag autocorrelation in tick trade prices.

| Metric | normal | stress | Note |
|---|---|---|---|
| mid-return ACF lag 1 (quote-event time) | -0.0683 | -0.0957 | 95% band +/-0.0008 |
| mid-return ACF lag 1 (aggressive-order time) | 0.1117 | 0.1263 |  |
| trade-price change ACF lag 1, order level (raw) | 0.0738 | 0.0579 | sign persistence pushes this positive; see the split below |
| trade-price change ACF lag 1, after a sign flip (bid-ask bounce) | -0.0114 | -0.0443 | the Roll bounce, isolated from order-sign persistence |
| trade-price change ACF lag 1, after a repeated sign | 0.1327 | 0.1220 | continuation of a sweep in the same direction |
| trade-price change ACF lag 1, fill level | 0.0741 | 0.0422 | positive by construction: sweeps walk the book one way |
| variance ratio at k>=100 (quote-event time) | 2.658 | 2.586 | 1.0 = diffusive; event time compresses busy trending periods |
| aggressor sign / contemporaneous mid-return correlation | 0.3739 | 0.4101 | this is impact (R(1)), not predictability |
| aggressor sign / *subsequent* mid-return correlation | 0.1355 | 0.1607 | the predictability that must stay small even though the sign series itself is strongly persistent |
| mid-return ACF lag 1 (1s bars) | 0.0881 | 0.1185 | noise floor +/-0.0024, n=691,199 |
| variance ratio at k>=10 (1s bars) | 1.686 | 1.781 | the clock-time diffusivity the literature refers to |
| mid-return ACF lag 1 (60s bars) | 0.0382 | 0.0241 | noise floor +/-0.0183, n=11,519 |
| variance ratio at k>=10 (60s bars) | 0.9445 | 1.042 | the clock-time diffusivity the literature refers to |

### fact12 - Selective liquidity taking / timing of large trades  
*Priority: P0-C / P1*

> **Literature:** Schnaubelt, Rende & Krauss: liquidity improves ~2-3 minutes before large BTC trades.

| Metric | normal | stress | Note |
|---|---|---|---|
| depth seen by top size decile / unconditional | 0.7848 | 0.7837 | > 1 means large orders time the book |
| depth seen by bottom size decile / unconditional | 0.8806 | 0.8903 |  |
| depth seen by top size decile / depth seen by the average order | 0.9093 | 0.9110 | event-weighted control: > 1 means large orders pick better books than other orders do |
| depth 150 s before a large order / unconditional | 0.9825 | 0.9706 | Schnaubelt et al. report liquidity improving 2-3 min ahead |
| depth at the moment of a large order / 150 s before | 0.8192 | 0.7877 | > 1 means depth built up into the trade |

### fact13 - Queue imbalance predicts the near-term price direction  
*Priority: P1*

> **Literature:** Gould & Bonart (2016): significant predictive power, strongest for large-tick instruments (not Bitcoin-specific evidence).

| Metric | normal | stress | Note |
|---|---|---|---|
| directional accuracy at h=1 quote event | 0.6932 | 0.6804 | 0.5 = no information |
| best directional accuracy | 0.7701 | 0.7660 | at h=20 quote events |
| imbalance / forward-return correlation at h=10 | 0.3243 | 0.3393 |  |

### fact14 - Volatility clustering at short horizons  
*Priority: P1*

> **Literature:** Schnaubelt et al.: squared-return ACF decay ~0.16 (minute), ~0.24 (hourly).

| Metric | normal | stress | Note |
|---|---|---|---|
| |return| ACF decay exponent (quote-event time) | 0.6022 | 0.4924 | R2=0.877 |
| squared-return ACF decay exponent (1s bars) | 0.2717 | 0.3381 | lags 2-200, n=691,200 bars, R2=0.76; no literature target |
| squared-return ACF decay exponent (60s bars) | 0.4085 | 0.7571 | lags 2-200, n=11,520 bars, R2=0.64; literature 0.16 |
| |return| ACF at lag 100 (quote-event time) | 0.0192 | 0.0254 | positive and slowly decaying = clustering |

### fact16 - BTC trade-size distribution  
*Priority: P1*

> **Literature:** Schnaubelt et al.: many small trades, heavy upper tail, round-number clustering.

| Metric | normal | stress | Note |
|---|---|---|---|
| median aggressive-order size (BTC) | 0.0013 | 0.0014 |  |
| share of aggressive orders below 0.01 BTC | 0.7591 | 0.7670 | many small trades |
| Hill tail index alpha (top 5%, aggressive orders) | 1.064 | 0.9008 | threshold 0.1752 BTC; alpha < 3 = heavy tail |
| 99.9th pct / median size ratio | 3,281 | 2,961 |  |
| mean aggressive-order notional (USD) | 3,590 | 3,151 |  |
| share of orders on the 0.1 BTC grid | 0.1192 | 0.1208 | round-number clustering |

### fact17 - 24/7 time-of-day liquidity and activity structure  
*Priority: P2*

> **Literature:** Schnaubelt et al.: hourly variation exists but no universal U-shape; BTC has no session open/close.

| Metric | normal | stress | Note |
|---|---|---|---|
| hourly spread: max/min ratio | 2.382 | 3.110 |  |
| hourly depth: max/min ratio | 1.145 | 1.178 |  |
| hourly volume: max/min ratio | 5.630 | 5.739 |  |
| edge-vs-middle ratio of hourly volume (U-shape score) | 0.6211 | 0.8013 | ~1 = no equity-like U-shape |
| busiest UTC hour by volume | 13.000 | 14.000 | hour of day |

### fact01_03 - Metaorder impact proxy: concavity, trajectory and decay  
*Priority: P0-V (proxy)*

> **Literature:** Donier & Bonart (BTC square-root impact); Toth et al. (latent liquidity); Zarinelli et al. (limited range, log alternative, ~2/3 relaxation); Bucci et al. (slow decay).

> **Caveat:** Proxy only: pseudo-metaorders are runs of same-signed aggressive orders, not labelled parent orders, and their participation rate is mechanically high.

| Metric | normal | stress | Note |
|---|---|---|---|
| pseudo-metaorders identified | 20,846 | 28,140 | gap 1.0s, >= 2 children |
| impact concavity exponent delta (proxy) | 0.1825 | 0.1544 | se 0.027, R2=0.761; square root = 0.5 |
| power-law R2 minus logarithmic R2 | -0.1837 | -0.1022 | positive favours the power law (Zarinelli et al. caveat) |
| trajectory concavity: impact at 50% executed / impact at 100% | 0.5100 | 0.4904 | 0.5 = linear build-up, ~0.71 = square-root build-up |
| impact 10 min after completion / peak (raw) | 1.308 | 1.346 | includes market drift and the correlated flow that follows |
| impact 10 min after completion / peak (drift-adjusted) | 1.208 | 1.462 | random-time control removed; Zarinelli et al. ~2/3 in US equities, Donier-Bonart near-full decay for the uninformed part of BTC impact |
| median pseudo-metaorder size (BTC) | 0.0085 | 0.0076 |  |
| Q/ADV explored: 1st percentile | 1.31e-07 | 1.32e-07 | Donier-Bonart span ~4 decades of metaorder size |
| Q/ADV explored: 99th percentile | 0.0042 | 0.0033 | decades covered = log10(p99/p01) |
| decades of size covered by the proxy | 4.502 | 4.392 | a narrow range flattens any concavity estimate |
| median participation rate of the proxy | 1.000 | 1.000 | mechanically high: runs are defined by same-signed flow |
| average daily volume (BTC) | 1,273 | 1,208 |  |
| daily volatility (bps) | 142.990 | 149.208 | from 60 s mid bars |

## Outputs

40 figures and 60 tables, under `results/<regime>/`.

### normal

- `fact05_liquidity_cost_and_depth__normal.png`
- `fact06_spread_distribution__normal.png`
- `fact07_resiliency__normal.png`
- `fact08_ofi_response__normal.png`
- `fact09_trade_response__normal.png`
- `fact10_sign_memory__normal.png`
- `fact11_diffusion__normal.png`
- `fact12_selective_liquidity__normal.png`
- `fact13_queue_imbalance__normal.png`
- `fact14_volatility_clustering__normal.png`
- `fact16_trade_size__normal.png`
- `fact17_time_of_day__normal.png`
- `fact01_03_metaorder_impact_proxy__normal.png`

### stress

- `fact05_liquidity_cost_and_depth__stress.png`
- `fact06_spread_distribution__stress.png`
- `fact07_resiliency__stress.png`
- `fact08_ofi_response__stress.png`
- `fact09_trade_response__stress.png`
- `fact10_sign_memory__stress.png`
- `fact11_diffusion__stress.png`
- `fact12_selective_liquidity__stress.png`
- `fact13_queue_imbalance__stress.png`
- `fact14_volatility_clustering__stress.png`
- `fact16_trade_size__stress.png`
- `fact17_time_of_day__stress.png`
- `fact01_03_metaorder_impact_proxy__stress.png`

### comparison

- `fact00_reconstruction_validation__compare.png`
- `fact05_liquidity_cost_and_depth__compare.png`
- `fact06_spread_distribution__compare.png`
- `fact07_resiliency__compare.png`
- `fact08_ofi_response__compare.png`
- `fact09_trade_response__compare.png`
- `fact10_sign_memory__compare.png`
- `fact11_diffusion__compare.png`
- `fact12_selective_liquidity__compare.png`
- `fact13_queue_imbalance__compare.png`
- `fact14_volatility_clustering__compare.png`
- `fact16_trade_size__compare.png`
- `fact17_time_of_day__compare.png`
- `fact01_03_metaorder_impact_proxy__compare.png`
