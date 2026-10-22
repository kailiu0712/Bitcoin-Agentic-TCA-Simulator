"""
The impact-law tests' market constants, seeding, clock and reference prices (Coinbase SOL-USD),
copied from liquidation_analysis/impact_analysis/protocol.py, so that a dashboard run with seed
index k is set up exactly as the tests' runs with seed index k.
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

INPUTS = Path(__file__).resolve().parent / "inputs"

# the market of every model (Coinbase SOL/USD)
TICK_SIZE = 0.01
LOT_SIZE = 0.01
ORDER_SIZE = 26.06                                 # q0, SOL
TRADE_RATE = 0.412219                              # mu, q0 market orders per second per side
V_D = 2 * TRADE_RATE * 86400 * ORDER_SIZE          # expected daily background traded volume, SOL

# seeding and clock
N_SEEDS = 30                                       # seed indices 0..29 were tested
SEED_BASE = 1000                                   # seed index k -> seed 1000 + k
BASE_START = dt.datetime(2026, 8, 24, 1, 15, 0)    # seed index 0 starts here (clock of the real data)
SEED_SPACING_S = 7200                              # seed index k starts k * 2 h later
REF_T0 = dt.datetime(2026, 8, 24, 0, 0, 0)         # t_s = 0 of inputs/ref_mid_coinbase_sol.csv
BM_STEP_S = 0.1                                    # grid of the pre-drawn Brownian reference path
MAX_SIM_S = 11400                                  # the tests' longest run; the Brownian path is drawn this long


def params() -> pd.Series:
    return pd.read_csv(INPUTS / "params.csv").set_index("pair").loc["coinbase_sol"]


class Run:
    """Seed, random streams, clock and reference prices of one seed index (protocol.Run)."""

    def __init__(self, seed_index: int):
        self.seed_index = seed_index
        self.seed = SEED_BASE + seed_index
        self.rng_mm, self.rng_nt, self.rng_ref = (
            np.random.default_rng(s) for s in np.random.SeedSequence(self.seed).spawn(3))
        self.start = BASE_START + dt.timedelta(seconds=seed_index * SEED_SPACING_S)
        self.init_price = float(params()["init_price"])

    def historical_path(self) -> tuple[np.ndarray, np.ndarray]:
        """Real Coinbase SOL mid from this seed's start clock time: (t since start, mid)."""
        ref = pd.read_csv(INPUTS / "ref_mid_coinbase_sol.csv")
        off = (self.start - REF_T0).total_seconds()
        return ref.t_s.to_numpy() - off, ref.mid.to_numpy()

    def bm_path(self) -> tuple[np.ndarray, np.ndarray]:
        """Arithmetic Brownian motion with the calibrated daily volatility, drawn in advance on
        a fixed BM_STEP_S grid from the seed's own stream (identical with and without the liquidation)."""
        daily_vol = float(params()["daily_vol"])
        sigma = daily_vol * self.init_price / np.sqrt(86400)        # USD per sqrt(s)
        n = int(MAX_SIM_S / BM_STEP_S) + 1
        steps = self.rng_ref.standard_normal(n - 1) * sigma * np.sqrt(BM_STEP_S)
        path = self.init_price + np.r_[0.0, np.cumsum(steps)]
        return np.arange(n) * BM_STEP_S, path
