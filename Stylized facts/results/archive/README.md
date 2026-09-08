# Archived outputs — the run that preceded the 2026-09-05 revision

Everything here was produced *before* the six new stylized-fact modules were
added (see `../../PROGRESS_RECORD.md`). It is kept so the earlier numbers stay
citable, not because it is superseded in every respect.

| Item | What it is | Produced |
|---|---|---|
| `normal/`, `stress/` | per-regime figures and tables for facts 5–14, 16–17 and the `fact01_03` metaorder proxy | 2026-08-29 |
| `comparison_v0/` | the normal-vs-stress comparison figures of that same run (the directory the framework then called `comparison`) | 2026-08-29 |
| `SUMMARY.md`, `summary_metrics.csv`, `manifest.json` | that run's headline table and output index | 2026-08-29 |
| `metaorder_comparison/` | the standalone three-proxy comparison entry point (`run_metaorder_comparison.py`) | 2026-09-05, before the revision |

## Reading these against the new run

Do not splice rows from here into the new `SUMMARY.md`. Three things changed:

1. **Two measurement fixes** land in the new modules rather than in the old
   ones, so where a quantity appears in both places the definitions differ.
   `fact08b` anchors the mid change one quote earlier than `fact08` does (the
   off-by-one described in the progress record), and `fact20` reports the
   spread and depth **time-weighted**, where `fact06`'s headline histogram is
   event-weighted. Both old and new numbers are correct measurements of
   different things; the new modules say which.
2. `metaorder_comparison/` here used a different endpoint convention from the
   new `fact_meta_notebook` — the pre-trade mid of the *next* aggressive order,
   rather than the notebook's last-child mid. The new module carries both, so
   the comparable column exists, but it is not the primary one.
3. Nothing in this directory was rebuilt from the raw feed. Both runs read the
   same cached tapes in `../../cache/`, so the reconstruction underneath is
   identical and any difference is analysis, not ingest.
