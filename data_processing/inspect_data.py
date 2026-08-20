import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq


def inspect_file(path):
    pf = pq.ParquetFile(path)
    categorical = {c: Counter() for c in ("symbol", "event_type", "side", "ord_type")}
    nulls = Counter()
    for batch in pf.iter_batches(batch_size=1_000_000, columns=list(categorical)):
        for i, col in enumerate(categorical):
            arr = batch.column(i)
            nulls[col] += arr.null_count
            for item in pc.value_counts(arr).to_pylist():
                if item["values"] is not None:
                    categorical[col][str(item["values"])] += item["counts"]
    return {"file": str(path.resolve()), "rows": pf.metadata.num_rows,
            "row_groups": pf.metadata.num_row_groups, "schema": str(pf.schema_arrow),
            "counts": {k: dict(v) for k, v in categorical.items()}, "nulls": dict(nulls)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/raw/btc/*.parquet")
    ap.add_argument("--output", default="data/metadata/inspection.json")
    args = ap.parse_args()
    paths = sorted(Path().glob(args.glob))
    report = {"files": [inspect_file(p) for p in paths]}
    report["total_rows"] = sum(x["rows"] for x in report["files"])
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(paths), "total_rows": report["total_rows"], "output": str(out)}))


if __name__ == "__main__":
    main()
