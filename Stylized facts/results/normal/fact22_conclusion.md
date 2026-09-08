The conclusion: algorithms trade on the wall clock
There is a clock, and it is UTC round time. Three layers, all measured:

Layer 1 — a 1-second grid, offset by ~0.3 s
Trades pile up at 0.30–0.40 s past every round second, peaking at 1.45× normal at 0.32 s:


0.0-0.1s past the second : 1.071x
0.1-0.2s                 : 1.097x
0.2-0.3s                 : 1.047x
0.3-0.4s                 : 1.297x   <-- the bump
0.4-0.5s                 : 1.023x
0.5-0.6s                 : 0.905x
...
0.8-0.9s                 : 0.835x   <-- the trough
The bump is broad (~100 ms wide), not a sharp spike. That's the signature of a timer firing on the round second and the order taking ~0.3 s of variable latency to reach and match at the exchange — not of the exchange stamping on a grid.

Layer 2 — a 10-second envelope
The round :00, :10, :20, :30, :40, :50 second of each minute gets 28% more trades than the other nine:


second 0 of the 10 s cycle : 1.283x   <-- round 10s boundary
second 1                   : 1.068x
seconds 2-9                : 0.90 - 1.02x
This is why the spectral comb's fundamental is 0.1 Hz (10 s), not 0.2 Hz. All 15 strong lines are exact multiples of 0.1 Hz; only 14 are multiples of 0.2 Hz — the line at 1.5 Hz (0.667 s) breaks the 0.2 Hz series and forces the fundamental down to 10 s.

Layer 3 — weaker minute and multi-minute anchors
Rayleigh z at 60 s = 33, at 300 s = 23, at 600 s = 25 — against controls at 2–6. So there's also scheduling on the round minute, five minutes and ten minutes, just much weaker.

How much of the flow
From z ≈ (clocked orders)² / (total orders): z = 210 on ~23,000 orders/day implies ≥ 2,200 orders per day, roughly 9%, sit on the clock. A lower bound, since they're spread over ~100 ms of phase rather than perfectly aligned.

The crucial caveat
This is not one TWAP. The repeated-clip test found only 2 of 158 same-size groups more regular than random (1.6 expected by chance), and the median group was more clustered than random. So this is many participants' schedulers all anchored to the same wall clock — cron-style timers on round seconds — not a single parent order being sliced.

Practical upshot: it belongs in your simulator's arrival process (order flow is not time-homogeneous at sub-minute scale; it has a 1 s comb with a 10 s envelope). It does not give you a metaorder identifier.