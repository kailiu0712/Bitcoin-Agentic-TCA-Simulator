"""
Assemble the run summary.

Every stylized-fact module returns a list of headline metrics; this turns them
into one wide normal-vs-stress table, a machine-readable CSV, and a README that
indexes every figure and table produced.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "-"
        if value == 0:
            return "0"
        magnitude = abs(value)
        if magnitude >= 1000:
            return f"{value:,.0f}"
        if magnitude >= 1:
            return f"{value:,.3f}"
        if magnitude >= 1e-3:
            return f"{value:.4f}"
        return f"{value:.3g}"
    return str(value)


def build_metric_table(results: list[dict], regime_order: list[str]) -> pd.DataFrame:
    """One row per (fact, metric), one column per regime."""
    rows = []
    for res in results:
        by_metric: dict[str, dict] = {}
        for entry in res.get("metrics", []):
            key = entry["metric"]
            slot = by_metric.setdefault(key, {"detail": entry.get("detail", "")})
            slot[entry["regime"]] = entry.get("value")
            if entry.get("detail"):
                slot["detail"] = entry["detail"]
        for metric, slot in by_metric.items():
            row = {
                "fact": res["fact"],
                "priority": res["priority"],
                "stylized_fact": res["title"],
                "metric": metric,
            }
            for regime in regime_order:
                row[regime] = slot.get(regime)
            row["note"] = slot.get("detail", "")
            rows.append(row)
    return pd.DataFrame(rows)


def write_summary(ctx, results: list[dict], runtime: dict) -> Path:
    """Write SUMMARY.md, summary_metrics.csv and manifest.json."""
    results_dir = Path(ctx.cfg.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    regime_order = ctx.names

    table = build_metric_table(results, regime_order)
    table.to_csv(results_dir / "summary_metrics.csv", index=False, float_format="%.6g")

    with open(results_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"generated": dt.datetime.now().isoformat(timespec="seconds"),
                   "runtime": runtime, "outputs": ctx.out.manifest}, handle, indent=2)

    lines: list[str] = []
    lines.append("# BTC/USD stylized facts - normal vs stress week")
    lines.append("")
    lines.append(f"Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} "
                 f"from the Kraken BTC/USD L3 feed.")
    lines.append("")
    lines.append("## Regimes")
    lines.append("")
    lines.append("| Regime | Dates | Measured hours | Raw rows | Aggressive orders | Quote updates |")
    lines.append("|---|---|---|---|---|---|")
    for name, rd in ctx.each():
        d = rd.diagnostics
        lines.append(
            f"| **{name}** | {d['start_date']} to {d['end_date']} | "
            f"{_fmt(d.get('measured_time_hours'))} | {_fmt(d.get('rows_processed'))} | "
            f"{_fmt(d.get('aggressive_orders'))} | "
            f"{_fmt(d.get('tape_rows', {}).get('quotes'))} |"
        )
    lines.append("")
    recorded_ingest = sum(float(rd.diagnostics.get("elapsed_seconds") or 0.0)
                          for _, rd in ctx.each())
    lines.append(f"Book reconstruction: {recorded_ingest:.0f} s "
                 f"({'from cache this run' if runtime.get('ingest_seconds', 0) < 1 else 'this run'}); "
                 f"analysis: {runtime.get('analysis_seconds', 0):.0f} s.")
    lines.append("")

    lines.append("## Headline results")
    lines.append("")
    for res in results:
        lines.append(f"### {res['fact']} - {res['title']}  \n*Priority: {res['priority']}*")
        lines.append("")
        if res.get("literature"):
            lines.append(f"> **Literature:** {res['literature']}")
            lines.append("")
        if res.get("caveat"):
            lines.append(f"> **Caveat:** {res['caveat']}")
            lines.append("")
        sub = table[table.fact == res["fact"]]
        if len(sub):
            header = "| Metric | " + " | ".join(regime_order) + " | Note |"
            lines.append(header)
            lines.append("|---" * (len(regime_order) + 2) + "|")
            for _, row in sub.iterrows():
                cells = " | ".join(_fmt(row[r]) for r in regime_order)
                lines.append(f"| {row.metric} | {cells} | {row.note} |")
        lines.append("")

    lines.append("## Outputs")
    lines.append("")
    figures = [m for m in ctx.out.manifest if m["kind"] == "figure"]
    tables = [m for m in ctx.out.manifest if m["kind"] == "table"]
    lines.append(f"{len(figures)} figures and {len(tables)} tables, under "
                 f"`{results_dir.name}/<regime>/`.")
    lines.append("")
    for regime in regime_order + ["comparison"]:
        subset = [m for m in figures if m["regime"] == regime]
        if not subset:
            continue
        lines.append(f"### {regime}")
        lines.append("")
        for m in subset:
            lines.append(f"- `{Path(m['path']).name}`")
        lines.append("")

    path = results_dir / "SUMMARY.md"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path
