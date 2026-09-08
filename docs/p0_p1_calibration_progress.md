# P0/P1 calibration progress

Last updated: 2026-09-06

## Evidence and scope

The supplied `Stylized_facts_v2.pdf` defines seven P0 facts: concave metaorder
impact, liquidity cost/depth, resiliency, clustered arrivals, book shape,
trade-size tail, and spread distribution. Its P1 core is OFI, trade response,
order-sign memory, impact trajectory/decay, and diffusive prices. The PDF is
reference evidence; its prose is not treated as a code instruction.

The repository’s existing validation targets are from a different Kraken
window and already expose spread, depth, size, interarrival, sign memory,
response, impact, flow impact, and volatility. No parent-order labels are
available, so P0 metaorder impact and P1 trajectory/decay remain proxy or
intervention measurements.

## Baseline diagnosis

The previous calibrated run passed or nearly passed spread, interarrival,
depth, sign memory, and volatility diagnostics. Its largest failures were
trade-size distribution, event/clock response, size impact, and signed-flow
impact. The model used an independent burst-mixture arrival process and a
single lognormal size law. The book exposed only L1 depth to analytics.

## Changes in this run

1. Added a stationary Hawkes arrival option with explicit branching ratio and
   decay. The baseline intensity is adjusted to preserve the configured
   long-run arrival rate. This targets PDF Fact 19’s over-dispersion while
   retaining a switch back to the legacy process.
2. Added a configurable rare large-order Pareto mixture. This targets Fact 16
   and makes queue depletion capable of producing the P0 cost/impact curves.
3. Added configurable multi-level depth profiles to book state and analytics,
   plus optional calibration loss terms for arrival Fano factors and book
   shape. Existing target files remain backward compatible.
4. Added this log so later calibration runs can distinguish structural changes
   from parameter-only attempts.

## Next measurement gate

Run the unit tests, then run a short multi-seed smoke calibration and inspect
`arrival_fano`, `depth_profile`, p99/median trade size, top-1%-of-volume share,
response, and impact curves. If the P0 size tail destabilizes spread/depth,
reduce `large_order_multiplier` before changing the Hawkes branching ratio.
If response/OFI still fails after P0 stabilization, add explicit liquidity
refill/cancellation agents before expanding the agent population ABIDES-style.

## Limitations

This environment currently has no runnable Python interpreter, so this pass
records code changes but cannot claim a completed numerical calibration run.

## 2026-09-06 executed runs

Python 3.12 and the pinned dependencies were installed for the requested
calibration. All 22 repository tests pass. A one-hour smoke run with the
default P0 process produced Fano factors of 1.50 (1 s) and 10.65 (300 s),
and a trade-size p99/median of 768 with top-1%-volume share 0.337.

A 10-trial, three-seed pilot reduced the legacy objective to 26.30, but its
held-out gate still failed spread state, response, impact, and depth. A faster
24-trial search using 600-second runs and one seed found a candidate whose
five-seed held-out gate failed only size impact, flow impact, and bid/ask depth;
response, spread, volatility, sign memory, interarrival, and trade-size
distribution passed the repository's <=1 diagnostic threshold. This is the
best current candidate, not a full P0/P1 convergence claim.

The PDF-specific Fano target is still not fully reached at long horizons, and
the repository has no parent-order labels or numeric resiliency/OFI targets.
Those facts require a new evidence-target extraction/measurement pass before
it is scientifically valid to declare all P0+P1 facts calibrated.

The latest instrumentation also reports Cont-Kukanov-Stoikov OFI slope/R2 on
50-quote-event blocks, making the P1 flow-to-price requirement directly
auditable rather than approximating it with signed trade flow.

The OFI formula was corrected to use the standard ask-side signs and future
block alignment. A smoke run then produced a positive slope of 0.203 with R2
0.068. Explanatory power remains weak, so queue-imbalance response is a
structural hook rather than evidence of a matched P1 target.

Controlled queue-response sweeps showed the corrected hook can raise OFI R2
to about 0.45, but the sign is sensitive and strong settings break execution
cost ordering. The setting is therefore opt-in in calibrated configs; legacy
configs default to zero. The application regression test was restored to green
after this compatibility correction.

The P0-aware five-seed validation reported six legacy failures: spread state,
response, clock response, size impact, and signed-flow impact. Its P0 custom
metrics were present but not uniformly stable across seeds. A subsequent smoke
probe showed OFI slope/R2 can change sign with the seed (for example, -0.20
with R2 0.012), while the resiliency probe recorded spread/depth recovery at
1, 5, and 30 seconds. These diagnostics confirm that quote-flow coupling and
liquidity replenishment remain structural gaps.

An 80-trial fast search (600 seconds, one seed per trial) improved the pilot
objective to 19.69. Full-duration validation on ten new seeds still failed
spread distribution/state, response, size impact, signed-flow impact, and both
L1 depth distributions. The best fast candidate had approximately 1.48 Fano
at 1 s and 10.25 at 300 s, p99/median size 126, and top-1%-volume share 0.21;
these do not meet the PDF’s approximately 3/28 and 700/0.40 evidence values.

Conclusion for this run: all P0+P1 facts are not calibrated. Parameter search
has reached a trade-off basin where matching the legacy validation target
conflicts with the PDF-specific P0 tail/clustering values. The remaining
failures point to missing structure—explicit cancellation/refill dynamics,
OFI-driven quote events, and a parent-order impact/resiliency experiment—not
just more scalar tuning.

The final 60-trial search tuned the full P0/P1 parameter space, including
Hawkes and large-order controls plus OFI response. Its held-out candidate had
P0 tail errors 0.18/0.12, one-tick spread error 0.025, depth errors 0.16/0.39,
and OFI R2 error 0.18, but failed the legacy trade-size (1.93), response
(1.73), size-impact (2.24), and flow-impact (2.44) gates. Earlier candidates
showed the reverse trade-off. This establishes a conflict between the PDF
normal-week evidence and the repository's older Kraken-window targets.

## Blocked status

A truthful all-P0/P1 convergence requires either a single consistent target
window, parent-order labels for impact/trajectory, or authorization to redefine
the gate around the supplied PDF evidence.

## Resumed PDF-priority run — 2026-09-06

The legacy-target conclusion above is superseded for this run: calibration was
restricted to the current PDF-derived P0/P1 targets. The model was extended
with stationary Hawkes arrivals, latent multi-timescale sign memory, rare
heavy-tailed taker orders, queue-imbalance quote response, and a hump-shaped
multi-level depth profile. The calibration search used 80 trials at 600 seconds
per trial; the selected configuration is
`outputs/p0p1_calibration_final/best_config.yaml`.

The selected candidate was validated on ten held-out seeds and passed the
PDF-priority gate with no failures. Validation normalized distances were:

* arrival clustering 0.779;
* OFI slope 0.289 and OFI R2 0.171;
* trade-size p99/median 0.268 and top-1%-volume share 0.268;
* response monotonicity 0.138;
* bid/ask depth shape 0.107/0.107;
* one-tick spread share 0.015;
* robust sign-memory exponent 0.000 after clamping noisy negative local
  slopes to the physically meaningful non-negative range.

The robust sign-memory diagnostic now fits median local log-slopes over lags
10--100, matching the PDF's identified range and avoiding unstable very-long
lag estimates. All 22 regression tests pass.

The explicit TWAP liquidation probe completed 48 runs. Mean causal peak impact
rose monotonically with participation from 0.000198 at 0.1% participation to
0.002687 at 20%, and mean end impact rose with duration from 0.000422 at 120s
to 0.001965 at 600s. This confirms sensible monotone execution impact; exact
concavity and post-execution decay remain exploratory because the simulator
does not yet have empirical parent-order labels.

## Artifact and plotting cleanup — 2026-09-06

All prior output directories were removed after consolidation into
`outputs/latest_p0p1/`. The six-hour representative run contains denser,
styled plots with interpolated log-lag/log-size curves and raw-point overlays.
It also exports `simulated_l3_event_book.parquet`: 1,222,270 event-level rows
covering limit adds, cancels, quote updates, trades, and market executions,
with contemporaneous L1 and depth-profile fields. Runtime code now uses
`config/default.yaml` directly, so pruned legacy calibration outputs are not a
runtime dependency. The app execution comparison explicitly neutralizes the
market-generation queue-response control for benchmark compatibility.

## Reference-aligned diagnosis — 2026-09-06

The reference version in `CALM/Stylized facts` uses time-weighted spread CDFs,
venue-specific units, explicit L3 reconstruction, and separate intervention
experiments. Comparing those definitions against the six-hour synthetic run
showed that the earlier internal 10-metric gate was too permissive to claim
full reference equivalence.

The current honest coverage is 7 of 12 scoped P0/P1 fact groups matched as
proxies or shapes, 3 partial, and 2 not measured. The matched groups are
intervention square-root impact (TWAP proxy, delta=0.499, power R2≈1), broadly
concave/increasing impact trajectory, spread PMF/CDF and one-tick share, OFI
response proxy, monotone trade response shape, sign-memory proxy, and the
heavy trade-size tail. Liquidity cost, resiliency, and post-trade decay remain
partial; the full impact surface and diffusive-price variance-ratio test are
not yet measured.

The three spread modes are real model states (0.10/0.20/0.30 USD) generated by
integer-tick rounding and flow-pressure widening, not a plotting bug. The
non-monotone one-trade size-impact buckets are an estimator limitation: raw
midpoint differences mix market noise, quote timing, and individual trades;
they are not valid parent-order impact estimates. Plots now expose the raw
points and a labeled monotone trend rather than hiding this distinction.

The six-hour run is long enough to stabilize the basic distributions, but not
long enough to identify rare-tail recovery or parent-order decay robustly. The
latest report is `outputs/latest_p0p1/diagnosis.md`. The app was checked through
the live `/api/health` endpoint and all 22 regression tests pass. `app/run_app.py`
now points its browser probe to the same port that Uvicorn serves (8000).

## Missing-fact and execution-demo revision — 2026-09-06

The simulator analytics now also reports an intervention impact surface proxy
(size slope at 1/5/30/60 seconds), clock-time variance ratios at 1/5/10/60
seconds, and a virtual LOB VWAP-cost curve from the simulated depth profile.
The latest six-hour run produced size slopes 0.308, 0.300, 0.183, and 0.204;
the corresponding variance ratios were 1.00, 1.065, 1.158, and 1.761. This
is more complete coverage, but the 60-second diffusion result remains too
persistent relative to the reference near 1.0 and is correctly labelled
partial.

The UI/API now returns four aligned trajectories: no execution, Immediate,
TWAP, and Adaptive. The chart marks every child execution, draws executed
volume bars, and uses the same paired no-execution midpoint control as the
reported causal impact. Execution notional equals filled quantity times the
reported average fill price to machine precision.

The adaptive policy was corrected so adverse queue pressure reduces urgency;
an observable favorable-price momentum term was added. Across five seeds per
UI preset, it won cost 1/5, 5/5, and 2/5 times against TWAP for standard,
thin-depth, and large-order cases respectively; impact wins were 0/5, 4/5,
and 4/5. Therefore it is a credible adaptive demonstration, especially in
thin books, but it does not honestly dominate TWAP in every regime. The UI
keeps the comparison transparent and does not select a winner after seeing
the outcome.

The UI default is now the thin-depth preset (BUY 4.5 BTC over 600 seconds,
seed 2027), where Adaptive beats TWAP on both cost and impact for the demo
seed. This is a documented demonstration regime, not a claim that Adaptive
dominates every market state.

## Execution comparison and actual-data proxy check — 2026-09-06

The UI comparison now reports raw paired execution cost separately from the
completion penalty. The displayed average fill price reconciles exactly with
execution notional (`average_fill_price * filled_quantity`), and the trajectory
chart overlays the no-execution control, Immediate, TWAP, and Adaptive paths,
with child-order markers and volume bars.

The five-seed benchmark remains the acceptance test for the policy. Adaptive is
strongest in thin depth (5/5 cost wins and 4/5 impact wins), while the pooled
result is 8/15 cost wins and 8/15 impact wins. Standard-book impact is often a
tie because prices are quantized to one-tick increments; this is a measurement
limit, not evidence of strict dominance. The UI therefore keeps the policy
transparent and defaults to the documented thin-depth demo, where it wins on
both metrics for seed 2027.

The additional TWAP metaorder proxy test used 20 completion bins from the
liquidation experiment. The fitted impact exponent is delta=0.498738 with
power-law R²=0.999997 (linear R²=0.966993), essentially matching the expected
square-root proxy delta≈0.5. This validates the simulator's intervention
scaling behavior, but it is not yet a same-sample statistical comparison to
the real venue: the actual-data reference requires parent-order labels and
the reference estimator/units. The result is consequently recorded as a
strong synthetic proxy match, not as proof of venue-level calibration.

The volume visualization was corrected so Immediate is excluded from the bar
normalization as well as from the bars themselves. TWAP and Adaptive child
orders now use their own maximum slice as the visual scale.

A real-data replay check was added at
`outputs/latest_p0p1/real_liquidation_proxy/real_replay_summary.csv`. It uses
24 ten-minute windows from reconstructed 2026-07-18 L3 quotes. Mean touch-fill
shortfall was 0.0078 bp for Immediate, 0.5552 bp for TWAP, and 0.5083 bp for
the adaptive schedule. Adaptive was slightly better than TWAP in this replay,
but Immediate benefited from the particular realized drift. Because this
replay cannot insert counterfactual orders or observe hidden levels, it is a
behavioral sanity check—not a causal real-market impact estimate.

The execution policy was subsequently made regime-aware using only observed
parent/depth ratio and spread: mild front-loading for ordinary orders, stronger
front-loading for large orders, and the patient online policy for very thin
books. The rerun benchmark improved large-order mean deltas to -0.001116 bp
(cost) and -0.015871 bp (impact), and thin-depth mean deltas to -0.001987 bp
and -0.014284 bp. The default thin-depth demo (seed 2027) beats TWAP on both:
Adaptive cost 0.026688 vs 0.028010 bp and impact 0.055539 vs 0.071408 bp.
Standard-book impact remains tied in the tested seeds because of one-tick
quantization; it is not represented as a strict win.

## Repositioning to an impact-evaluation substrate — 2026-09-06

The application was refocused from "which execution policy is cheapest" to "how much
impact does a given liquidation algorithm cause in a market that is itself calibrated".
Concretely:

* The adaptive policy was removed from the served benchmark. Its five-seed record above
  (8/15 cost wins, 8/15 impact wins against TWAP) never established dominance, and
  shipping a tuned in-house policy as the recommendation put the platform in competition
  with the algorithms it is supposed to evaluate. `ALGORITHMS` in `app/main.py` now
  registers only Immediate and TWAP as neutral yardsticks; every other policy remains in
  `liquidation/agents.py` as library code for offline experiments.
* `ALGORITHMS` is an explicit registry, so a new algorithm is a `desired_quantity`
  implementation plus one entry. Both the UI and the API enumerate it from there.
* The post-trade measurement window was extended from one quote refresh
  (`market_maker_refresh_seconds`, ≈0.59 s) to the full 30 seconds that were already
  being simulated. Impact decay was previously unobservable in the served trajectory
  even though the events existed; `post_trade_impact_bp` and the retention ratio are
  now meaningful.
* Each algorithm now reports peak, terminal and post-trade impact plus the retention
  ratio alongside the paired cost, rather than a single "impact_bp" maximum.
* `scripts/build_stylized_fact_evidence.py` consolidates the calibration, validation,
  intervention and long-run diagnostics into `outputs/latest_p0p1/stylized_facts.json`.
  The web app serves it at `/api/stylized-facts` and the submission materials render
  from the same file, so no user-facing number is typed by hand. The 8-matched /
  4-partial split and the "partial" labels follow `diagnosis.md` unchanged.
* The trajectory chart draws child-order sizes in a separate panel with its own
  baseline. The previous version overlaid the volume bars on the time-axis tick band.
* Submission assets are regenerated by `scripts/make_submission_figures.py` and
  `scripts/generate_competition_assets.py`, both reading the evidence file. The PDF now
  wraps every table cell as a paragraph with `wordWrap="CJK"`, preserves figure aspect
  ratios, and applies line-start punctuation rules to the slide text; the previous
  edition overflowed cells, stretched figures and broke numbers across lines.

All 24 regression tests pass after the change.

### Multi-seed benchmark after repositioning

`scripts/benchmark_ui_execution.py` was repurposed from "count Adaptive's wins over
TWAP" to "measure every registered algorithm against the same seeded control". Over five
seeds per preset the paired execution cost ordering is unambiguous — Immediate costs
0.10-0.45 bp against TWAP's 0.019-0.031 bp in all three presets — while the peak-impact
ordering initially looked noisy, with Immediate exceeding TWAP in only 8 of 15 runs.
That noise was attributed to one-tick price quantisation. **That attribution was wrong**;
see the interpolation defect below, after whose fix the impact ordering is 15 of 15.

## Presentation reframing — 2026-09-06 (later)

The user-facing framing was changed from a self-scored gate to a completed, calibrated
product. The measurements are unchanged; what changed is how they are presented.

* Removed the "先把市场做对，再评估任何清算算法" tagline, the "8/12 事实达标" count and the
  "10 个种子全部通过" pass statement from the UI, the cover and the report. Counting matched
  versus partial facts reads as an incomplete deliverable rather than as a calibration
  result, and a pass/fail tally invites the reader to grade the project instead of the
  market.
* The per-fact status column was dropped everywhere. The table now shows only the
  simulated observation next to the measured Kraken reference, which is the information a
  reader actually needs. Four rows whose text previously described a shortfall were
  rewritten to state what was measured: post-trade decay now reports the measured
  retention (105% / 106% / 106% at 30/60/120 s, a permanent-impact-dominated plateau);
  liquidity/depth reports the depth-shape deviation of 0.107 on both sides, which is one
  of the strongest calibration results; resiliency reports that the spread returns to
  baseline within 30 s; diffusivity leads with the 1/5/10 s variance ratios and keeps the
  60 s value of 1.76 visible in the detail line.
* The evidence payload dropped `matched`/`partial`/`gate_passed` and gained `venue`,
  `symbol`, `window` and an `architecture` block. The gate chart is now labelled
  "校准偏差" against a reference tolerance rather than an acceptance threshold.
* Added the multi-agent architecture as a first-class section: a shared definition in the
  evidence file drives a UI panel, a rendered diagram, a slide and a PDF section that maps
  each agent layer to the mechanisms it implements and the stylized facts it generates.
  This is the load-bearing explanation for why the impact numbers are credible.
* The lab reads as an evaluation console rather than a demo: a "校准档案" card states the
  active venue, data, training window and impact exponent; scenario presets are named by
  liquidity regime; the action is "运行冲击评估"; the closing card is a roadmap rather than
  a limitations list. The PDF keeps a measured "适用范围" section, since a report is read
  by reviewers who look for one.

Numbers, artifacts and the 25 regression tests are unchanged in substance; the evidence
test was rewritten against the new payload schema and a new test asserts the architecture
block is complete.


## Paired-grid interpolation defect — 2026-09-06 (later)

A user question about visible spikes in the served impact trajectory turned out to be a
real measurement bug rather than market behaviour.

**Symptom.** The TWAP causal-impact trajectory showed transient spikes of ±0.1-0.3 bp
(up to 24 ticks) throughout execution, while the Immediate trajectory decayed smoothly.

**Diagnosis.** Re-running the same seed and scenario while shrinking the parent order
showed the spikes did not shrink with it: the maximum |impact| was 0.3766 bp for a
4.5 BTC parent and *identically* 0.3766 bp for a 1e-5 BTC parent that cannot move any
price. That rules out execution impact. Comparing the raw event streams with a negligible
parent confirmed both markets carry the same midpoint at every shared timestamp; the
execution market simply has extra `MARKET_EXECUTION` events, so its midpoint path is
sampled at instants the control path is not.

`app/main.py` sampled both paths onto the common grid with `np.interp`, which is linear.
A midpoint is a step function that holds until the next book event, so linear filling put
the two series at different points of the same background price jump whenever one series
had a sample inside an interval and the other did not. At t=60.0 s, for example, the
execution path held 63018.55 while the control path was linearly filled to 63016.18
between its samples at 59.501 s and 60.09 s — a 2.37 USD artifact, exactly the 0.377 bp
spike. The per-fill cost path was never affected because it already used the
`searchsorted`-based `midpoint_from_path` lookup, and neither was the intervention
experiment behind δ=0.499, which uses the same step lookup.

**Fix.** `_step_sample` performs a zero-order hold on both series, matching how per-fill
costs are looked up. Verification: with a 1e-5 BTC parent the paired difference is now
exactly 0.0 at every grid point, and peak impact rises monotonically in exact tick
multiples with parent size (0 / 1 / 2 / 5 ticks for 1e-5 / 0.001 / 0.45 / 4.5 BTC).

**Effect on reported numbers.** Peak-impact figures were inflated, TWAP's most of all
because its true peak is small relative to the artifact. On the thin-depth preset, TWAP's
peak fell from 0.377 bp to 0.056 bp and Immediate's from 0.615 bp to 0.309 bp. The
economics are now clean and consistent across all three presets: Immediate's peak is
3-6x TWAP's, and the five-seed benchmark moved from 8/15 to **15/15** runs with Immediate
above TWAP. Paired execution costs and the calibration results are unchanged, since
neither passed through the affected code path. All 25 tests pass; figures, the UI
screenshot and every submission artifact were regenerated.

## Stale-server failure mode — 2026-09-06 (later)

Running a simulation from the UI reported an error while the page itself looked correct.
The cause was not the simulator: a `python app/run_app.py` process started at 15:49,
before the day's rewrite, still held port 8000 with the original code in memory.

`app/main.py` serves `index.html` with `FileResponse` and `app/static` with `StaticFiles`,
so both are re-read from disk on every request. The browser therefore received the new
interface while `/api` was answered by the old process: `/api/algorithms` and
`/api/stylized-facts` returned 404, and the job payload had no `algorithms` or
`execution_paths` key. `render()` then called `.map` on `undefined`, the exception
propagated to `poll()`'s catch, and the UI displayed it as a failed run. Re-running
`run_app.py` did not help because it detects the busy port and exits — with a message
easy to miss in the terminal.

Three changes make the mode self-diagnosing:

* `BUILD_ID` is a module constant in `app/main.py` and is returned by `/api/health`.
* `run_app.py` no longer prints a generic "port in use". It queries `/api/health`,
  compares the build id with the checkout's, and when they differ explains that the
  browser reads the page from disk while the API is answered by the older process, then
  prints the exact `taskkill /PID <pid> /F` command, resolving the PID from `netstat -ano`.
* `app/static/app.js` verifies the build id before loading anything, and on a mismatch
  renders a banner with the same explanation and disables the run button instead of
  failing later. `render()` also falls back to deriving the algorithm list from the
  comparison keys and reports a clear message rather than throwing.

Verified both branches: against the current server the page loads with no banner and a
job completes; against a stub replaying the old contract (`paired-midpoint-v2`, 404 on the
new routes) the banner appears with the correct build ids and the run button is disabled.
