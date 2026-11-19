"""Tests for the order book simulator and imbalance measures."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lob import (  # noqa: E402
    LOBParams,
    aggregate_ofi,
    depth_imbalance,
    imbalance_accuracy,
    imbalance_signal_table,
    next_mid_move,
    ofi_regression,
    order_flow_imbalance,
    queue_imbalance,
    simulate_lob,
    weighted_mid,
)


@pytest.fixture(scope="module")
def hist():
    """Simulate a book shared by the statistical tests."""
    return simulate_lob(n_events=60_000, seed=11)


def test_book_invariants(hist):
    """The book never crosses, sizes are non-negative and best queues are non-empty."""
    assert not hist.truncated
    assert np.all(hist.ask > hist.bid)
    assert np.all(hist.bid_depth >= 0) and np.all(hist.ask_depth >= 0)
    assert np.all(hist.bid_size > 0) and np.all(hist.ask_size > 0)
    assert np.all(np.diff(hist.time) > 0)


def test_initial_book():
    """Row 0 is the initial book with the requested spread and depth."""
    h = simulate_lob(LOBParams(initial_spread=3, initial_depth=4), n_events=1, seed=0)
    assert h.spread[0] == 3
    assert np.all(h.bid_depth[0] == 4) and np.all(h.ask_depth[0] == 4)
    assert h.event[0] == -1


def test_size_changes_by_one_unit(hist):
    """Every event moves exactly one unit at one price."""
    total = hist.bid_depth.sum(axis=1) + hist.ask_depth.sum(axis=1)
    # Changes in the visible top levels are bounded by one order per event,
    # except when the best quote moves and levels shift in or out of view.
    still = (np.diff(hist.bid) == 0) & (np.diff(hist.ask) == 0)
    assert np.all(np.abs(np.diff(total))[still] <= 1)


def test_seed_is_reproducible():
    """The same seed gives the same history."""
    a = simulate_lob(n_events=2000, seed=5)
    b = simulate_lob(n_events=2000, seed=5)
    np.testing.assert_array_equal(a.bid_depth, b.bid_depth)
    np.testing.assert_array_equal(a.time, b.time)


def test_queue_imbalance_values():
    """Queue imbalance is (Vb - Va) / (Vb + Va) and NaN for an empty book."""
    out = queue_imbalance([3, 0, 1, 0], [1, 2, 1, 0])
    np.testing.assert_allclose(out[:3], [0.5, -1.0, 0.0])
    assert np.isnan(out[3])


def test_depth_imbalance_weights():
    """Equal weights sum levels; a large decay reduces to the best level."""
    b = np.array([[1, 5, 5]])
    a = np.array([[3, 1, 1]])
    assert depth_imbalance(b, a)[0] == pytest.approx((11 - 5) / 16)
    assert depth_imbalance(b, a, decay=50)[0] == pytest.approx(-0.5)
    assert depth_imbalance(b, a, levels=1)[0] == pytest.approx(-0.5)


def test_weighted_mid():
    """The weighted mid leans toward the thinner side."""
    wm = weighted_mid([99, 99, 99], [101, 101, 101], [3, 1, 2], [1, 3, 2])
    np.testing.assert_allclose(wm, [100.5, 99.5, 100.0])


def test_order_flow_imbalance_cases():
    """Each OFI case from Cont, Kukanov and Stoikov (2014)."""
    bid = [100, 100, 101, 100, 100, 100]
    bsz = [5, 7, 2, 4, 4, 4]
    ask = [102, 102, 102, 102, 101, 103]
    asz = [5, 5, 5, 5, 3, 6]
    e = order_flow_imbalance(bid, bsz, ask, asz)
    assert e[0] == 0
    assert e[1] == 2  # bid queue grows by 2
    assert e[2] == 2  # bid improves: new queue counts in full
    assert e[3] == -2  # bid drops: old queue of 2 removed
    assert e[4] == -3  # ask improves to 101 with 3
    assert e[5] == 3  # ask retreats: old queue of 3 removed


def test_aggregate_ofi_conserves_totals():
    """Window sums add up to the total OFI and the total mid change."""
    t = np.arange(10.0)
    e = np.arange(10.0)
    m = np.cumsum(np.ones(10))
    _, o, d = aggregate_ofi(t, e, m, interval=3)
    assert o.sum() == pytest.approx(e[1:].sum())
    assert d.sum() == pytest.approx(m[-1] - m[0])


def test_ofi_regression_recovers_line():
    """OLS recovers a known slope and intercept."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = 0.3 * x + 0.1 + rng.normal(scale=0.01, size=500)
    fit = ofi_regression(x, y)
    assert fit["beta"] == pytest.approx(0.3, abs=0.01)
    assert fit["alpha"] == pytest.approx(0.1, abs=0.01)
    assert fit["r2"] > 0.99


def test_next_mid_move():
    """The next non-zero change is found for every position before it."""
    out = next_mid_move([1, 1, 2, 2, 2, 1.5, 1.5])
    np.testing.assert_allclose(out[:5], [1, 1, -0.5, -0.5, -0.5])
    assert np.isnan(out[5]) and np.isnan(out[6])


def test_imbalance_predicts_next_move(hist):
    """Queue imbalance predicts the direction of the next mid move."""
    imb = queue_imbalance(hist.bid_size, hist.ask_size)
    assert imbalance_accuracy(imb, hist.mid) > 0.6
    tab = imbalance_signal_table(imb, hist.mid, n_buckets=5)
    p = tab["p_up"].to_numpy()
    assert p[1] < 0.4
    assert p[3] > 0.6


def test_ofi_explains_price_changes(hist):
    """OFI over 10-second windows explains most mid changes, with beta near 1/(2 depth)."""
    e = order_flow_imbalance(hist.bid, hist.bid_size, hist.ask, hist.ask_size)
    _, o, d = aggregate_ofi(hist.time, e, hist.mid, interval=10.0)
    fit = ofi_regression(o, d)
    depth = (hist.bid_size.mean() + hist.ask_size.mean()) / 2
    assert fit["r2"] > 0.5
    assert fit["beta"] == pytest.approx(1 / (2 * depth), rel=0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lam": (1.0,), "theta": (1.0, 1.0)},
        {"mu": -1},
        {"initial_depth": 0},
        {"n_ticks": 10},
    ],
)
def test_invalid_params(kwargs):
    """Invalid parameters raise ValueError."""
    with pytest.raises(ValueError):
        LOBParams(**kwargs)
