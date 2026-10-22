"""
The dashboard's six markets (Coinbase SOL-USD), each built exactly as its impact-law test built it
(liquidation_analysis/<folder>/run_liquidation*.py, named below). The keys are the tests' model names.

A market is a function market(kernel, book, run) -> (market_maker, noise_trader, others): it builds
the agents with ids 1 (market maker), 2 (noise trader) and 4, 5, ... (others), and sets their compute
delays. scenario.py adds the exchange (id 0) and the TWAP liquidation bot (id 3), as the tests did.

The simulator classes are imported inside each function: a run first puts its model's code
(simulators/<code>) on sys.path, and the two code bases use the same module names.
"""

import datetime as dt
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

import protocol as P

EXCHANGE_ID, MARKET_MAKER_ID, NOISE_TRADER_ID, LIQUIDATION_BOT_ID, TAKER_ID = 0, 1, 2, 3, 4


def santafe(reference: str) -> Callable:
    """Extended Santa Fe: MarketMaker with the calibrated quote / cancel rates per level, NoiseTrader.
    reference "none": the market maker quotes relative to the opposite best quote; "bm": a Brownian
    reference price; "historical": the real Coinbase SOL mid.
    (extended-santafe/{no reference price, Brownian motion reference price, Historical reference price})"""
    def market(kernel, book, run):
        from agent.market_maker import MarketMaker
        from agent.noise_trader import NoiseTrader
        from oracle.historical import HistoricalOracle

        rates = pd.read_csv(P.INPUTS / "rates_coinbase_sol.csv")
        oracle = {"none": lambda: None,
                  "bm": lambda: HistoricalOracle(*run.bm_path()),           # a pre-drawn path, served by time
                  "historical": lambda: HistoricalOracle(*run.historical_path())}[reference]()
        market_maker = MarketMaker(
            my_id=MARKET_MAKER_ID, exchange_id=EXCHANGE_ID, kernel=kernel, book=book, rng=run.rng_mm,
            quote_rates=rates["quote_rates"].tolist(), cancel_rates=rates["cancel_rates"].tolist(),
            quote_size=P.ORDER_SIZE, tick_size=P.TICK_SIZE, initial_mid=run.init_price, oracle=oracle)
        noise_trader = NoiseTrader(
            my_id=NOISE_TRADER_ID, exchange_id=EXCHANGE_ID, kernel=kernel, rng=run.rng_nt,
            trade_rate=P.TRADE_RATE, trade_size=P.ORDER_SIZE)
        return market_maker, noise_trader, []
    return market


def load_rates(name: str, value_col: str, keys: tuple[str, ...]) -> np.ndarray:
    """(s, i[, l], value) CSV -> array indexed by the sorted unique keys (NaN -> 0)."""
    df = pd.read_csv(P.INPUTS / name).fillna(0.0)
    axes = [np.sort(df[k].unique()) for k in keys]
    arr = np.zeros(tuple(len(a) for a in axes))
    arr[tuple(np.searchsorted(a, df[k].to_numpy()) for a, k in zip(axes, keys))] = df[value_col].to_numpy(float)
    return arr


def state_dependent_santafe(kernel, book, run):
    """Santa Fe whose rates depend on the spread s and the order-book imbalance i: lambda_l(s, i),
    nu_l(s, i), mu(s, i) from the Coinbase SOL calibration, no reference price.
    (state-dependent-santafe; its own kernel and agents: simulators/state_dependent)"""
    from agent.market_maker import MarketMaker
    from agent.noise_trader import NoiseTrader

    market_maker = MarketMaker(
        my_id=MARKET_MAKER_ID, exchange_id=EXCHANGE_ID, kernel=kernel, book=book, rng=run.rng_mm,
        quote_rates=load_rates("lambda.csv", "lambda", ("s", "i", "l")),
        cancel_rates=load_rates("nu.csv", "nu", ("s", "i", "l")),
        quote_size=P.ORDER_SIZE, tick_size=P.TICK_SIZE, initial_mid=run.init_price)
    noise_trader = NoiseTrader(
        my_id=NOISE_TRADER_ID, exchange_id=EXCHANGE_ID, kernel=kernel, rng=run.rng_nt,
        trade_rate=load_rates("mu.csv", "mu", ("s", "i")), trade_size=P.ORDER_SIZE, book=book)
    return market_maker, noise_trader, []


def smart_mm_market(kernel, book, run, mm_delay_ms: int, mm: dict, taker: dict):
    """SmartMM: a ladder market maker quoting around the real Coinbase SOL mid seen with a lag, with
    an inventory skew; a SmartTaker arbitraging the book against the real mid; the NoiseTrader.
    Only the noise trader draws random numbers."""
    from agent.noise_trader import NoiseTrader
    from agent.smart_mm import SmartMM
    from agent.smart_taker import SmartTaker
    from oracle.historical import HistoricalOracle

    oracle = HistoricalOracle(*run.historical_path())
    oracle.initialize(run.start)
    kernel.set_compute_delay(MARKET_MAKER_ID, dt.timedelta(milliseconds=mm_delay_ms))
    kernel.set_compute_delay(TAKER_ID, dt.timedelta(milliseconds=1))   # the taker is the faster one
    market_maker = SmartMM(my_id=MARKET_MAKER_ID, exchange_id=EXCHANGE_ID, kernel=kernel, oracle=oracle,
                           tick_size=P.TICK_SIZE, **mm)
    noise_trader = NoiseTrader(my_id=NOISE_TRADER_ID, exchange_id=EXCHANGE_ID, kernel=kernel, rng=run.rng_nt,
                               trade_rate=P.TRADE_RATE, trade_size=P.ORDER_SIZE)
    smart_taker = SmartTaker(my_id=TAKER_ID, exchange_id=EXCHANGE_ID, kernel=kernel, book=book, oracle=oracle,
                             **taker)
    return market_maker, noise_trader, [smart_taker]


def smart_mm(kernel, book, run):
    """Aggressive taker: 5 levels of q0 quotes 2 ticks apart, max inventory 10 q0 (one side pulled
    beyond it); the taker trades up to 5 q0 every 100 ms once the book is 3 ticks off the real mid.
    (smart-mm/run_liquidation.py, the parameters of simulator-SmartMM/main.py)"""
    q0 = P.ORDER_SIZE
    return smart_mm_market(
        kernel, book, run, mm_delay_ms=5,
        mm=dict(quote_size=q0, max_inventory=10 * q0, skew_ticks=5, n_levels=5, half_spread_ticks=1,
                level_gap_ticks=2, ref_lag=dt.timedelta(seconds=1), refresh_interval=dt.timedelta(seconds=1)),
        taker=dict(threshold_ticks=3, max_order_size=5 * q0, max_inventory=20 * q0,
                   poll_interval=dt.timedelta(milliseconds=100)))


def smart_mm_deep(kernel, book, run):
    """Mild taker: a deep book that is never pulled (20 levels 1 tick apart, sized so the tests'
    largest child order takes ~60% of a side; max inventory 3x their largest metaorder), a linear
    skew of 0.003 bp per SOL; the taker trades 0.1 SOL per second at most, 2 ticks off the real mid.
    (smart-mm/run_liquidation_deep.py, the parameters of smart-mm/main_deep.py)"""
    largest_q = 0.01 * P.V_D                       # the tests' largest metaorder, Q/V_D = 1%
    n_levels = 20
    quote_size = round(largest_q / 60 / 0.6 / n_levels, 2)                   # 25.78 SOL
    max_inventory = math.ceil(3.0 * largest_q / 1000.0) * 1000.0             # 56,000 SOL
    skew_ticks = 0.003 * 1e-4 * run.init_price / P.TICK_SIZE * max_inventory
    return smart_mm_market(
        kernel, book, run, mm_delay_ms=2,
        mm=dict(quote_size=quote_size, max_inventory=max_inventory, skew_ticks=skew_ticks, n_levels=n_levels,
                half_spread_ticks=1, level_gap_ticks=1, ref_lag=dt.timedelta(milliseconds=500),
                refresh_interval=dt.timedelta(milliseconds=500)),
        taker=dict(threshold_ticks=2, max_order_size=0.1, max_inventory=1000.0,
                   poll_interval=dt.timedelta(seconds=1), book_depth=n_levels))


@dataclass(frozen=True)
class Model:
    label: str
    code: str                   # simulators/<code>
    market: Callable
    n_agents: int = 4           # exchange, market maker, noise trader, liquidation bot (+ others)


MODELS = {
    "extended_santafe_none": Model("Santa Fe · no reference price", "main", santafe("none")),
    "extended_santafe_bm": Model("Santa Fe · Brownian reference price", "main", santafe("bm")),
    "extended_santafe_historical": Model("Santa Fe · historical reference price", "main", santafe("historical")),
    "state_dependent_santafe": Model("State-dependent Santa Fe", "state_dependent", state_dependent_santafe),
    "smart_mm": Model("SmartMM · aggressive taker", "main", smart_mm, n_agents=5),
    "smart_mm_deep": Model("SmartMM · mild taker", "main", smart_mm_deep, n_agents=5),
}
