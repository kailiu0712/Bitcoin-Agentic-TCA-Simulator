"""Build one canonical event file while preserving raw source files."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from data_processing.preprocess_events import preprocess


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config/default.yaml"); ap.add_argument("--force",action="store_true")
    args=ap.parse_args(); cfg=yaml.safe_load(Path(args.config).read_text())["data"]
    raw=sorted(Path().glob(cfg["raw_glob"])); staging=Path("data/processed/sessions"); staging.mkdir(parents=True,exist_ok=True)
    legacy=Path("outputs/archive_v1/processed"); quality={}
    for source in raw:
        session=staging/source.name
        if session.exists() and not args.force: continue
        old=legacy/source.name
        if old.exists() and not args.force:
            shutil.copy2(old,session); print(f"reused verified legacy session {source.stem}",flush=True)
        else:
            print(f"reconstructing {source.stem}",flush=True); quality[source.stem]=preprocess(source,session,cfg["quote_sample_ms"])
    output=Path(cfg["processed_events"]); output.parent.mkdir(parents=True,exist_ok=True)
    canonical=pa.schema([("timestamp_ns",pa.int64()),("source_timestamp",pa.timestamp("ns",tz="UTC")),
        ("event_type",pa.string()),("side",pa.string()),("price",pa.float64()),("size",pa.float64()),
        ("order_id",pa.string()),("trade_id",pa.int64()),("order_type",pa.string()),
        ("best_bid",pa.float64()),("best_ask",pa.float64()),("bid_size",pa.float64()),("ask_size",pa.float64()),
        ("mid",pa.float64()),("spread",pa.float64()),("sign_source",pa.string()),("date",pa.string())])
    writer=None; sessions=[]
    for path in sorted(staging.glob("*.parquet")):
        frame=pd.read_parquet(path); date=path.stem; frame["date"]=date
        frame["trade_id"]=pd.to_numeric(frame["trade_id"],errors="coerce").astype("Int64")
        table=pa.Table.from_pandas(frame,schema=canonical,preserve_index=False)
        if writer is None: writer=pq.ParquetWriter(output,table.schema,compression="zstd")
        writer.write_table(table); sessions.append({"date":date,"rows":table.num_rows,
            "trades":int(table.column("event_type").to_pandas().eq("TRADE").sum()),
            "quotes":int(table.column("event_type").to_pandas().eq("QUOTE_UPDATE").sum())})
    if writer: writer.close()
    calibration=cfg["calibration_days"]; validation=cfg["validation_days"]
    summary={"schema":str(pq.ParquetFile(output).schema_arrow),"sessions":sessions,
        "usable_dates":[x["date"] for x in sessions],"calibration_dates":calibration,"validation_dates":validation,
        "quarantined":{"2026-07-14":"invalid Parquet footer/magic bytes"},"quality_new_sessions":quality}
    meta=Path(cfg["metadata"]); meta.parent.mkdir(parents=True,exist_ok=True); meta.write_text(json.dumps(summary,indent=2))
    print(json.dumps({"output":str(output),"rows":sum(x["rows"] for x in sessions),"sessions":len(sessions)},indent=2))


if __name__=="__main__": main()
