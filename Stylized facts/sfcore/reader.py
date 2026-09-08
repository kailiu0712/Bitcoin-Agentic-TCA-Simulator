"""
Streaming reader for the raw Kraken L3 parquet files.

Turns each Arrow record batch into a bundle of plain numpy arrays that the
numba book kernel can consume with zero Python-object overhead.  In particular
the 19-character order ids are hashed straight out of the Arrow string buffer
(never materialised as Python ``str``), which is what keeps the ~160M-row pass
inside a few minutes.

Column contract of the raw feed
-------------------------------
================== ============================================================
symbol             dictionary<string>  - always "BTC_USD" here
received_at_ns     int64               - collector receipt time; the canonical
                                        clock (monotone, shared by the book and
                                        trade channels)
exchange_timestamp timestamp[ns, UTC]  - exchange publish time
event_timestamp    timestamp[ns, UTC]  - *original* order time; for an ``add``
                                        this can be hours old, so it is NOT a
                                        usable clock and is not read
event_type         dictionary<string>  - add | modify | delete | trade | snapshot
side               dictionary<string>  - buy | sell (aggressor side on trades)
price              double              - exact multiple of 0.1 USD
qty                double              - 0.0 on ``delete`` rows
order_id           large_string        - null on trade rows
trade_id           int64               - null except on trade rows
ord_type           dictionary<string>  - taker order type, trades only
================== ============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit

# Event-type codes used throughout the kernel.
ET_ADD = 0
ET_MODIFY = 1
ET_DELETE = 2
ET_TRADE = 3
ET_SNAPSHOT = 4
ET_OTHER = 5

_EVENT_CODES = {
    "add": ET_ADD,
    "modify": ET_MODIFY,
    "delete": ET_DELETE,
    "trade": ET_TRADE,
    "snapshot": ET_SNAPSHOT,
}

# Side codes.  Bids are side 0, asks side 1, consistent with the book kernel.
SIDE_BID = 0
SIDE_ASK = 1
SIDE_NONE = -1

_SIDE_CODES = {"buy": SIDE_BID, "sell": SIDE_ASK}

READ_COLUMNS = [
    "received_at_ns",
    "exchange_timestamp",
    "event_type",
    "side",
    "price",
    "qty",
    "order_id",
    "trade_id",
]


@dataclass(slots=True)
class Batch:
    """One record batch, decoded into kernel-ready numpy arrays."""

    recv_ns: np.ndarray      # int64   collector receipt time
    exch_ns: np.ndarray      # int64   exchange timestamp
    etype: np.ndarray        # int8    ET_* code
    side: np.ndarray         # int8    SIDE_* code
    tick: np.ndarray         # int32   price / TICK_SIZE, rounded
    qty: np.ndarray          # float64
    oid: np.ndarray          # int64   63-bit FNV-1a hash of order_id, 0 = none
    n: int

    def __len__(self) -> int:
        return self.n


@njit(cache=True, nogil=True)
def _hash_offsets(offsets: np.ndarray, data: np.ndarray, out: np.ndarray, n: int) -> None:
    """FNV-1a over each string slice; 0 is reserved for "no order id".

    Order ids are 19 printable characters ("OXD55G-KMG53-ZM7TEV"), so a 63-bit
    hash over at most a few hundred simultaneously-live orders has a collision
    probability far below any other error source in the pipeline.
    """
    for i in range(n):
        a = offsets[i]
        b = offsets[i + 1]
        if b <= a:
            out[i] = 0
            continue
        h = np.uint64(14695981039346656037)
        for j in range(a, b):
            h = h ^ np.uint64(data[j])
            h = h * np.uint64(1099511628211)
        # Mask to 63 bits so the key stays positive in int64 space, and map the
        # (astronomically unlikely) 0 to 1 so 0 can mean "empty slot".
        v = np.int64(h & np.uint64(0x7FFFFFFFFFFFFFFF))
        out[i] = v if v != 0 else np.int64(1)


def _dictionary_codes(col: pa.Array, mapping: dict[str, int], default: int) -> np.ndarray:
    """Map a dictionary-encoded column to int8 codes without materialising strings."""
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    if not pa.types.is_dictionary(col.type):
        values = col.to_pandas().astype("string").str.lower()
        return values.map(mapping).fillna(default).to_numpy(dtype=np.int8)
    dictionary = [str(v).lower() for v in col.dictionary.to_pylist()]
    lookup = np.full(len(dictionary) + 1, default, dtype=np.int8)
    for i, name in enumerate(dictionary):
        lookup[i] = mapping.get(name, default)
    indices = col.indices.to_numpy(zero_copy_only=False)
    indices = np.where(np.isnan(indices.astype(np.float64)), len(dictionary), indices)
    return lookup[indices.astype(np.int64)]


def _hash_order_ids(col: pa.Array) -> np.ndarray:
    """Hash a (large_)string column straight from its Arrow buffers."""
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    n = len(col)
    out = np.zeros(n, dtype=np.int64)
    if n == 0:
        return out
    buffers = col.buffers()
    offset_buf, data_buf = buffers[1], buffers[2]
    if offset_buf is None or data_buf is None:
        return out
    offset_dtype = np.int64 if pa.types.is_large_string(col.type) else np.int32
    offsets = np.frombuffer(offset_buf, dtype=offset_dtype)
    # Arrow arrays can be zero-copy slices of a larger buffer.
    offsets = offsets[col.offset: col.offset + n + 1].astype(np.int64)
    data = np.frombuffer(data_buf, dtype=np.uint8)
    _hash_offsets(offsets, data, out, n)
    return out


def decode_batch(batch: pa.RecordBatch, tick_size: float) -> Batch:
    """Decode one Arrow record batch into kernel-ready numpy arrays."""
    n = batch.num_rows
    recv = batch.column("received_at_ns").to_numpy(zero_copy_only=False).astype(np.int64)

    exch_col = batch.column("exchange_timestamp")
    exch = exch_col.cast(pa.int64()).to_numpy(zero_copy_only=False).astype(np.int64)

    etype = _dictionary_codes(batch.column("event_type"), _EVENT_CODES, ET_OTHER)
    side = _dictionary_codes(batch.column("side"), _SIDE_CODES, SIDE_NONE)

    price = batch.column("price").to_numpy(zero_copy_only=False).astype(np.float64)
    qty = batch.column("qty").to_numpy(zero_copy_only=False).astype(np.float64)
    qty = np.nan_to_num(qty, nan=0.0)

    with np.errstate(invalid="ignore"):
        ticks = np.rint(np.nan_to_num(price, nan=-1.0) / tick_size)
    ticks = np.where(np.isfinite(ticks), ticks, -1.0)
    tick = ticks.astype(np.int64)
    tick[~np.isfinite(price)] = -1
    tick = tick.astype(np.int32)

    oid = _hash_order_ids(batch.column("order_id"))
    return Batch(recv, exch, etype, side, tick, qty, oid, n)


def regime_files(data_dir: Path, start_date: str, end_date: str, warmup_days: int) -> tuple[list[Path], str]:
    """Daily files covering ``start_date``..``end_date`` plus warm-up days.

    Returns the ordered file list and the first *measured* date; files before it
    are fed through the book but emit no tape rows.
    """
    import datetime as _dt

    start = _dt.date.fromisoformat(start_date)
    end = _dt.date.fromisoformat(end_date)
    first = start - _dt.timedelta(days=warmup_days)

    paths: list[Path] = []
    day = first
    while day <= end:
        candidate = data_dir / f"{day.isoformat()}.parquet"
        if candidate.exists():
            paths.append(candidate)
        day += _dt.timedelta(days=1)
    if not paths:
        raise FileNotFoundError(f"No daily parquet files found in {data_dir} for {start_date}..{end_date}")
    return paths, start_date


def iter_batches(
    paths: list[Path],
    batch_rows: int,
    tick_size: float,
    max_rows_per_file: int | None = None,
) -> Iterator[tuple[Batch, Path]]:
    """Yield decoded batches across the daily files, in feed order."""
    for path in paths:
        parquet = pq.ParquetFile(path)
        taken = 0
        for record_batch in parquet.iter_batches(batch_size=batch_rows, columns=READ_COLUMNS):
            if max_rows_per_file is not None:
                if taken >= max_rows_per_file:
                    break
                if taken + record_batch.num_rows > max_rows_per_file:
                    record_batch = record_batch.slice(0, max_rows_per_file - taken)
            taken += record_batch.num_rows
            yield decode_batch(record_batch, tick_size), path


def total_rows(paths: list[Path], max_rows_per_file: int | None = None) -> int:
    """Total rows across ``paths`` - used to size the progress bar."""
    total = 0
    for path in paths:
        rows = pq.ParquetFile(path).metadata.num_rows
        total += rows if max_rows_per_file is None else min(rows, max_rows_per_file)
    return total
