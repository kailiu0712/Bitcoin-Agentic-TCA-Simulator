"""
Analysis context: lazily-loaded tapes per regime, plus output plumbing.

A single ``Context`` is built once by ``run_stylized_facts.py`` and handed to
every module in ``sffacts``.  Tapes are loaded on first touch and cached, so
thirteen stylized facts share one copy of the ~4M-row quote tape rather than
re-reading it thirteen times.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from . import tapes as tp


class RegimeData:
    """Everything stage 2 knows about one regime."""

    def __init__(self, cfg, name: str) -> None:
        self.cfg = cfg
        self.name = name
        self.spec = cfg.REGIMES[name]
        self.label = self.spec["label"]
        self.color = self.spec["color"]
        self.paths = tp.TapePaths.for_regime(cfg.CACHE_DIR, name)

    # -- raw tapes ---------------------------------------------------------

    @cached_property
    def trades(self) -> pd.DataFrame:
        """Aggressive orders, with the pre-trade book state attached."""
        df = tp.read_trades(self.paths)
        return df.sort_values("recv_ns", kind="stable").reset_index(drop=True)

    @cached_property
    def quotes(self) -> pd.DataFrame:
        df = tp.read_quotes(self.paths, self.cfg.TICK_SIZE,
                            columns=list(tp.QUOTE_COLUMNS))
        return df.sort_values("recv_ns", kind="stable").reset_index(drop=True)

    @cached_property
    def fills(self) -> pd.DataFrame:
        return tp.read_fills(self.paths)

    @cached_property
    def accumulators(self) -> dict[str, np.ndarray]:
        return tp.read_accumulators(self.paths)

    @cached_property
    def diagnostics(self) -> dict:
        return tp.read_diagnostics(self.paths)

    @cached_property
    def snapshot_validation(self) -> pd.DataFrame:
        return tp.read_snapshot_validation(self.paths)

    # -- derived arrays (cached; used by several facts) --------------------

    @cached_property
    def q_recv(self) -> np.ndarray:
        return self.quotes["recv_ns"].to_numpy(np.int64)

    @cached_property
    def q_exch(self) -> np.ndarray:
        return np.maximum.accumulate(self.quotes["exch_ns"].to_numpy(np.int64))

    @cached_property
    def q_mid(self) -> np.ndarray:
        ts = self.cfg.TICK_SIZE
        bid = self.quotes["bid_tick"].to_numpy(np.float64) * ts
        ask = self.quotes["ask_tick"].to_numpy(np.float64) * ts
        return 0.5 * (bid + ask)

    @cached_property
    def q_spread_bps(self) -> np.ndarray:
        ts = self.cfg.TICK_SIZE
        bid = self.quotes["bid_tick"].to_numpy(np.float64) * ts
        ask = self.quotes["ask_tick"].to_numpy(np.float64) * ts
        return (ask - bid) / (0.5 * (bid + ask)) * 1e4

    @cached_property
    def q_imbalance(self) -> np.ndarray:
        bq = self.quotes["bid_qty"].to_numpy(np.float64)
        aq = self.quotes["ask_qty"].to_numpy(np.float64)
        den = bq + aq
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, (bq - aq) / den, np.nan)

    @property
    def hours(self) -> float:
        return float(self.diagnostics.get("measured_time_hours", np.nan))


@dataclass
class Output:
    """Path construction and table writing, with a manifest of everything made."""

    results_dir: Path
    dpi: int = 150
    fmt: str = "png"
    manifest: list[dict] = field(default_factory=list)

    def _dir(self, regime: str) -> Path:
        path = Path(self.results_dir) / regime
        path.mkdir(parents=True, exist_ok=True)
        return path

    def figure_path(self, fact: str, slug: str, regime: str) -> Path:
        suffix = regime if regime != "comparison" else "compare"
        return self._dir(regime) / f"{fact}_{slug}__{suffix}.{self.fmt}"

    def save_figure(self, fig, fact: str, slug: str, regime: str, caption: str = "") -> Path:
        from . import plotting as pl

        path = self.figure_path(fact, slug, regime)
        pl.save(fig, path, self.dpi)
        self.manifest.append({"kind": "figure", "fact": fact, "slug": slug,
                              "regime": regime, "path": str(path), "caption": caption})
        return path

    def save_table(self, df: pd.DataFrame, fact: str, slug: str, regime: str,
                   caption: str = "", float_format: str = "%.6g") -> Path:
        suffix = regime if regime != "comparison" else "compare"
        base = self._dir(regime) / f"{fact}_{slug}__{suffix}"
        csv_path = base.with_suffix(".csv")
        df.to_csv(csv_path, index=False, float_format=float_format)
        with open(base.with_suffix(".md"), "w", encoding="utf-8") as handle:
            if caption:
                handle.write(f"**{caption}**\n\n")
            handle.write(df.to_markdown(index=False, floatfmt=".6g"))
            handle.write("\n")
        self.manifest.append({"kind": "table", "fact": fact, "slug": slug,
                              "regime": regime, "path": str(csv_path), "caption": caption})
        return csv_path


@dataclass
class Context:
    """What every stylized-fact module receives."""

    cfg: object
    regimes: dict[str, RegimeData]
    out: Output

    @property
    def names(self) -> list[str]:
        return list(self.regimes.keys())

    def each(self):
        """Iterate (name, RegimeData) in configured order."""
        return self.regimes.items()

    def colors(self) -> dict[str, str]:
        return {name: rd.color for name, rd in self.regimes.items()}


def build_context(cfg) -> Context:
    regimes = {name: RegimeData(cfg, name) for name in cfg.REGIMES}
    out = Output(Path(cfg.RESULTS_DIR), dpi=cfg.FIG_DPI, fmt=cfg.FIG_FORMAT)
    return Context(cfg=cfg, regimes=regimes, out=out)
