"""
Schema and streaming I/O for the intermediate event tapes.

Stage 1 (``ingest``) walks the raw feed once and writes three parquet tapes per
regime; stage 2 (the ``sffacts`` modules) reads them back.  Splitting the run
this way means the expensive 160M-row book reconstruction happens once, while
re-plotting or adding a stylized fact costs seconds.

Tape sizes for a one-week regime are roughly
    trades  ~0.5M rows    quotes  ~10-20M rows    fills  ~1M rows
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import book as bk

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------

TRADE_FIXED_COLUMNS = [
    "sign",          # +1 buyer-initiated, -1 seller-initiated
    "n_fills",       # maker orders consumed by this aggressive order
    "qty",           # BTC
    "notional",      # USD
    "vwap",          # USD, execution VWAP of the aggressive order
    "p_first",       # USD, first fill price
    "p_last",        # USD, last fill price
    "bid",           # USD, pre-trade best bid
    "ask",           # USD, pre-trade best ask
    "bid_qty",       # BTC at the pre-trade best bid
    "ask_qty",       # BTC at the pre-trade best ask
    "spread_bps",    # pre-trade quoted spread
    "mid",           # USD, pre-trade mid
    "n_lvl_bid",     # price levels held on the bid side
    "n_lvl_ask",
    "pk_since_snap",  # packets since the last snapshot reset (book warmth)
]

QUOTE_COLUMNS = ["recv_ns", "exch_ns", "bid_tick", "ask_tick", "bid_qty",
                 "ask_qty", "bid_depth", "ask_depth"]

FILL_COLUMNS = ["recv_ns", "sign", "price", "qty", "burst_idx"]

SNAPSHOT_VALIDATION_COLUMNS = [
    "recv_ns", "recon_bid", "snap_bid", "recon_ask", "snap_ask",
    "recon_bid_qty", "snap_bid_qty", "recon_ask_qty", "snap_ask_qty",
    "recon_n_lvl_bid", "recon_n_lvl_ask",
]


def depth_columns(bands: tuple[float, ...]) -> list[str]:
    """Cumulative visible quantity within ``band`` bps of the mid, per side."""
    cols: list[str] = []
    for band in bands:
        cols.append(f"depth_bid_{band:g}bps")
        cols.append(f"depth_ask_{band:g}bps")
    return cols


def cost_columns(sizes: tuple[float, ...]) -> list[str]:
    """Signed cost in bps of a virtual market order, by direction and size.

    ``cost_sell_*`` walks the bid side (a liquidation), ``cost_buy_*`` walks the
    ask side.  Both are positive costs relative to the mid.
    """
    cols: list[str] = []
    for size in sizes:
        cols.append(f"cost_sell_{size:g}btc")
        cols.append(f"cost_buy_{size:g}btc")
    return cols


def trade_columns(bands: tuple[float, ...], sizes: tuple[float, ...]) -> list[str]:
    return ["recv_ns", "exch_ns"] + TRADE_FIXED_COLUMNS + depth_columns(bands) + cost_columns(sizes)


# ---------------------------------------------------------------------------
# Streaming writers
# ---------------------------------------------------------------------------


class TapeWriter:
    """Append-only parquet writer that takes one numpy block at a time."""

    def __init__(self, path: Path, columns: list[str], dtypes: dict[str, str]) -> None:
        self.path = Path(path)
        self.columns = columns
        self.dtypes = dtypes
        self.schema = pa.schema([(c, _arrow_type(dtypes.get(c, "float64"))) for c in columns])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer: pq.ParquetWriter | None = None
        self.rows = 0

    def write(self, arrays: dict[str, np.ndarray]) -> None:
        n = len(next(iter(arrays.values())))
        if n == 0:
            return
        table = pa.table({c: pa.array(arrays[c]) for c in self.columns}, schema=self.schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        self._writer.write_table(table)
        self.rows += n

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        elif not self.path.exists():
            # No rows at all: still emit a valid empty tape so readers work.
            pq.write_table(pa.table({c: pa.array([], type=f.type)
                                     for c, f in zip(self.columns, self.schema)},
                                    schema=self.schema), self.path)


def _arrow_type(name: str):
    return {
        "int64": pa.int64(),
        "int32": pa.int32(),
        "float64": pa.float64(),
        "float32": pa.float32(),
        "int8": pa.int8(),
    }[name]


@dataclass(slots=True)
class TapePaths:
    """Where a regime's tapes live."""

    trades: Path
    quotes: Path
    fills: Path
    snapshot_validation: Path
    diagnostics: Path
    accumulators: Path

    @classmethod
    def for_regime(cls, cache_dir: Path, regime: str) -> "TapePaths":
        root = Path(cache_dir) / regime
        return cls(
            trades=root / "trades.parquet",
            quotes=root / "quotes.parquet",
            fills=root / "fills.parquet",
            snapshot_validation=root / "snapshot_validation.parquet",
            diagnostics=root / "diagnostics.json",
            accumulators=root / "accumulators.npz",
        )

    def all_exist(self) -> bool:
        return all(p.exists() for p in (self.trades, self.quotes, self.fills,
                                        self.diagnostics, self.accumulators))


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def read_trades(paths: TapePaths, columns: list[str] | None = None) -> pd.DataFrame:
    """Load the aggressive-order tape (small enough to hold in memory)."""
    return pq.read_table(paths.trades, columns=columns).to_pandas()


def read_quotes(paths: TapePaths, tick_size: float, columns: list[str] | None = None) -> pd.DataFrame:
    """Load the top-of-book tape and derive prices from integer ticks."""
    df = pq.read_table(paths.quotes, columns=columns).to_pandas()
    if "bid_tick" in df.columns:
        df["bid"] = df["bid_tick"].to_numpy(dtype=np.float64) * tick_size
    if "ask_tick" in df.columns:
        df["ask"] = df["ask_tick"].to_numpy(dtype=np.float64) * tick_size
    if "bid" in df.columns and "ask" in df.columns:
        df["mid"] = 0.5 * (df["bid"] + df["ask"])
        df["spread_bps"] = (df["ask"] - df["bid"]) / df["mid"] * 1e4
    return df


def read_fills(paths: TapePaths) -> pd.DataFrame:
    return pq.read_table(paths.fills).to_pandas()


def read_snapshot_validation(paths: TapePaths) -> pd.DataFrame:
    if not paths.snapshot_validation.exists():
        return pd.DataFrame(columns=SNAPSHOT_VALIDATION_COLUMNS)
    return pq.read_table(paths.snapshot_validation).to_pandas()


def read_accumulators(paths: TapePaths) -> dict[str, np.ndarray]:
    with np.load(paths.accumulators) as data:
        return {k: data[k] for k in data.files}


def read_diagnostics(paths: TapePaths) -> dict:
    import json

    with open(paths.diagnostics, "r", encoding="utf-8") as handle:
        return json.load(handle)


def burst_matrix_columns(n_bands: int, n_sizes: int) -> int:
    """Width of the kernel's burst value matrix."""
    return bk.B_FIXED + 2 * n_bands + 2 * n_sizes
