"""Stream Kraken-style L3 Parquet into compact canonical trade/quote observations.

Capture order (`received_at_ns`) is authoritative because source creation timestamps can
move backward during reconnect snapshots. Snapshot groups reset the reconstructed L3 book.
"""
import argparse
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class L3Book:
    def __init__(self):
        self.orders = {}
        self.depth = {"buy": defaultdict(float), "sell": defaultdict(float)}
        self.heaps = {"buy": [], "sell": []}

    def clear(self):
        self.orders.clear(); self.depth["buy"].clear(); self.depth["sell"].clear()
        self.heaps["buy"].clear(); self.heaps["sell"].clear()

    def upsert(self, oid, side, price, qty):
        old = self.orders.pop(oid, None)
        if old:
            self.depth[old[0]][old[1]] -= old[2]
        else:
            heapq.heappush(self.heaps[side], -price if side == "buy" else price)
        self.orders[oid] = (side, price, qty)
        self.depth[side][price] += qty
        return old is not None

    def delete(self, oid):
        old = self.orders.pop(oid, None)
        if old:
            self.depth[old[0]][old[1]] -= old[2]
        return old is not None

    def best(self, side):
        h = self.heaps[side]
        while h:
            p = -h[0] if side == "buy" else h[0]
            if self.depth[side].get(p, 0.0) > 1e-12:
                return p, self.depth[side][p]
            heapq.heappop(h)
        return None, None

    def state(self):
        bid, bs = self.best("buy"); ask, az = self.best("sell")
        valid = bid is not None and ask is not None and bid < ask
        return (bid, ask, bs, az, (bid + ask) / 2 if valid else None, ask - bid if valid else None)


SCHEMA = pa.schema([
    ("timestamp_ns", pa.int64()), ("source_timestamp", pa.timestamp("ns", tz="UTC")),
    ("event_type", pa.string()), ("side", pa.string()), ("price", pa.float64()),
    ("size", pa.float64()), ("order_id", pa.string()), ("trade_id", pa.int64()),
    ("order_type", pa.string()), ("best_bid", pa.float64()), ("best_ask", pa.float64()),
    ("bid_size", pa.float64()), ("ask_size", pa.float64()), ("mid", pa.float64()),
    ("spread", pa.float64()), ("sign_source", pa.string())])


def preprocess(path, output, sample_ms=100):
    pf, book = pq.ParquetFile(path), L3Book()
    writer = pq.ParquetWriter(output, SCHEMA, compression="zstd")
    quality, rows, last_ts, next_quote, snapshot_mode = Counter(), [], -1, -1, False
    seen_trade_ids = set()

    def emit(ts, source_ts, kind, side=None, price=None, qty=None, oid=None, tid=None, otype=None, sign_source=None):
        nonlocal rows
        bid, ask, bs, az, mid, spread = book.state()
        rows.append((ts, source_ts, kind, side, price, qty, oid, tid, otype, bid, ask, bs, az, mid, spread, sign_source))
        if len(rows) >= 100_000:
            writer.write_table(pa.Table.from_pylist([dict(zip(SCHEMA.names, x)) for x in rows], schema=SCHEMA)); rows = []

    cols = ["received_at_ns", "exchange_timestamp", "event_type", "side", "price", "qty", "order_id", "trade_id", "ord_type"]
    for batch in pf.iter_batches(batch_size=250_000, columns=cols):
        d = batch.to_pydict()
        for ts, source_ts, kind, side, price, qty, oid, tid, otype in zip(*(d[c] for c in cols)):
            quality["source_rows"] += 1
            if ts is None or ts < last_ts:
                quality["nonmonotone_or_null_capture_time"] += 1
                continue
            last_ts = ts
            if price is None or qty is None or price <= 0 or qty < 0:
                quality["invalid_price_or_quantity"] += 1
                continue
            if kind == "snapshot":
                if not snapshot_mode:
                    book.clear(); quality["snapshot_resets"] += 1
                snapshot_mode = True
                if oid: book.upsert(oid, side, price, qty)
                quality["snapshot_rows"] += 1
                continue
            snapshot_mode = False
            if kind in ("add", "modify"):
                if oid is None:
                    quality["missing_order_id"] += 1
                elif book.upsert(oid, side, price, qty):
                    quality["upserts_existing_order"] += 1
            elif kind == "delete":
                if not oid or not book.delete(oid): quality["delete_unknown_order"] += 1
            elif kind == "trade":
                if tid is not None and tid in seen_trade_ids:
                    quality["duplicate_trade_id"] += 1
                    continue
                if tid is not None: seen_trade_ids.add(tid)
                emit(ts, source_ts, "TRADE", side.upper(), price, qty, None, tid, otype,
                     "DIRECT_AGGRESSOR_SIDE")
                quality["trades"] += 1
            else:
                quality["unknown_event_type"] += 1
            if next_quote < 0: next_quote = ts
            if ts >= next_quote:
                bid, ask, *_ = book.state()
                if bid is not None and ask is not None:
                    if bid < ask: emit(ts, source_ts, "QUOTE_UPDATE")
                    else: quality["crossed_or_locked_samples"] += 1
                next_quote = ts + int(sample_ms * 1_000_000)
    if rows:
        writer.write_table(pa.Table.from_pylist([dict(zip(SCHEMA.names, x)) for x in rows], schema=SCHEMA))
    writer.close()
    quality["output_rows"] = pq.ParquetFile(output).metadata.num_rows
    return dict(quality)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/raw/btc/*.parquet")
    ap.add_argument("--output-dir", default="data/processed/sessions")
    ap.add_argument("--quote-sample-ms", type=int, default=100)
    args = ap.parse_args(); outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    report = {}
    for path in sorted(Path().glob(args.glob)):
        out = outdir / path.name
        print(f"processing {path} -> {out}", flush=True)
        report[path.stem] = preprocess(path, out, args.quote_sample_ms)
    qpath = Path("data/metadata/preprocessing.json"); qpath.parent.mkdir(parents=True, exist_ok=True)
    qpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(qpath)


if __name__ == "__main__": main()
