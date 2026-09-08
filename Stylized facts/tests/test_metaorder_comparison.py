"""Behavioral checks for grouping, price endpoints, and comparable estimators."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfcore.metaorder_proxies import random_traders, time_threshold, local_runs
from sffacts.fact_meta_compare import asof_prices, fit_size, _measure, _decay, PHI_GRID, DAY_NS


class FixedOwners:
    def choice(self, n, size, p):
        return np.array([0, 1, 0, 1, 0, 0])


def test_other_traders_opposite_orders_do_not_break_a_parent():
    parents = random_traders([1, -1, 1, -1, 1, -1], FixedOwners(), 2)
    assert [p.tolist() for p in parents] == [[0, 2, 4], [5], [1, 3]]


class FixedWait:
    def random(self):
        return 0.99  # size two when s_max=2

    def exponential(self, mean):
        return 2.0


def test_september_algorithm_jumps_and_retains_forward_scan_coverage():
    times = np.arange(7, dtype=np.int64)*1_000_000_000
    parents = time_threshold(times, np.ones(7), FixedWait(), max_children=2)
    assert [p.tolist() for p in parents] == [[0, 2], [4, 6]]
    # Not the unrequested variant that recycles skipped rows into more parents.
    assert set(np.concatenate(parents)) == {0, 2, 4, 6}


def test_threshold_is_target_overshoot_not_total_child_gap():
    # 3-second child gap, but only 1 second beyond the sampled 2-second target.
    p = time_threshold(np.array([0, 3])*10**9, [1, 1], FixedWait(),
                       max_children=2, max_target_delay_s=1.0)
    assert [x.tolist() for x in p] == [[0, 1]]
    p = time_threshold(np.array([0, 4])*10**9, [1, 1], FixedWait(),
                       max_children=2, max_target_delay_s=1.0)
    assert [x.tolist() for x in p] == [[0], [1]]


@pytest.mark.parametrize("seed", [1, 8, 99])
def test_same_timestamp_children_terminate_and_never_duplicate(seed):
    ts = np.repeat(np.arange(100), 3)*10**9
    signs = np.tile([1, -1, 1], 100)
    parents = time_threshold(ts, signs, np.random.default_rng(seed), mean_wait_s=.01)
    flat = np.concatenate(parents)
    assert len(flat) == len(np.unique(flat))
    for p in parents:
        assert np.all(signs[p] == signs[p[0]])
        assert np.all(np.diff(p) > 0)


def test_local_run_completion_does_not_include_trailing_opposite_child():
    p = local_runs(np.array([0, .1, .2, 2])*1e9, [1, 1, -1, -1], [1, 1, .1, 5])
    assert [x.tolist() for x in p] == [[0, 1], [3]]


def test_asof_rejects_beyond_end_before_start_and_stale():
    actual = asof_prices(np.array([10, 20, 40]),np.array([100., 101., 103.]),
                         np.array([9,10,20,35,40,41]),max_age_ns=5)
    np.testing.assert_allclose(actual,[np.nan,100,101,np.nan,103,np.nan],equal_nan=True)


def test_power_fit_preserves_signed_bins_and_scores_same_response():
    x = np.logspace(-5,-2,20)
    curve = pd.DataFrame(dict(x=x,y=.3*x**.4,n=50))
    fit = fit_size(curve)
    assert fit['delta'] == pytest.approx(.4, abs=.002)
    assert fit['power_r2'] == pytest.approx(1)
    curve.loc[0,'y']=-.001
    assert fit_size(curve)['n_fit_bins'] == 20


def test_shared_measurement_post_child_prices_and_signed_outcomes():
    ts = np.arange(7,dtype=np.int64)*10**9
    orders = pd.DataFrame(dict(ts_ns=ts,sign=np.ones(7),qty=np.ones(7),
                               mid=np.arange(100.,107.),day=np.zeros(7,dtype=int),
                               source_row=np.arange(7),order_pos=np.arange(7)))
    scale = type('Scale',(),dict(day=0,date='1970-01-01',sigma_range=.1,volume=7))()
    frame, paths = _measure(orders,[np.arange(5)],orders,ts,orders.mid.to_numpy(),scale,10**9)
    assert frame.mid_start.iloc[0]==100
    assert frame.mid_end.iloc[0]==105
    assert frame.impact_bps.iloc[0]==pytest.approx(500)
    assert paths[0][PHI_GRID==.2][0]==pytest.approx(100)
    assert paths[0][-1]==pytest.approx(500)
    orders['mid']=np.arange(107.,100.,-1)
    frame,_=_measure(orders,[np.arange(5)],orders,ts,orders.mid.to_numpy(),scale,10**9)
    assert frame.impact_bps.iloc[0]<0  # do not screen out valid adverse outcomes


def test_no_next_day_endpoint_leak():
    orders=pd.DataFrame(dict(ts_ns=[DAY_NS-2,DAY_NS-1,DAY_NS+1],sign=[1,1,1],qty=[1,1,1],
                             mid=[100.,101.,102.],day=[0,0,1],source_row=[0,1,2],order_pos=[0,1,2]))
    scale=type('Scale',(),dict(day=0,date='1970-01-01',sigma_range=.1,volume=2))()
    frame,_=_measure(orders.iloc[:2],[np.array([0,1])],orders,orders.ts_ns.to_numpy(),
                     orders.mid.to_numpy(),scale,10**9)
    assert np.isnan(frame.impact_bps.iloc[0])


def test_decay_uses_its_own_quote_clock_reference():
    qt=np.arange(0,1000,dtype=np.int64)*10**9
    qm=100+np.arange(1000)*.01
    frame=pd.DataFrame(dict(start_ns=[10**9],end_ns=[10*10**9],sign=[1],
                            mid_start=[100.01],impact_bps=[20.],duration_s=[9.],day=[0]))
    result=_decay(frame,qt,qm,30*10**9)
    at_end=result.loc[(result.clock=='z')&(result.horizon==1)].iloc[0]
    assert at_end.fraction_of_completion==pytest.approx(1.)
    assert at_end.fraction_of_size_endpoint!=pytest.approx(1.)
