"""Tests for the Avellaneda-Stoikov simulator."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator import (  # noqa: E402
    ASParams,
    compare,
    estimate_sigma,
    fill_probability,
    optimal_spread,
    quotes,
    reservation_price,
    simulate,
)


def test_reservation_price_formula():
    """Reservation price matches r = s - q*gamma*sigma^2*(T-t)."""
    p = ASParams(gamma=0.1, sigma=2.0, T=1.0)
    # r = s - q * gamma * sigma^2 * (T - t)
    assert reservation_price(100.0, 3, 0.25, p) == pytest.approx(
        100.0 - 3 * 0.1 * 4.0 * 0.75
    )
    assert reservation_price(100.0, 0, 0.0, p) == 100.0
    assert reservation_price(100.0, 5, 1.0, p) == 100.0


def test_optimal_spread_formula():
    """Spread matches gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k)."""
    p = ASParams(gamma=0.1, sigma=2.0, k=1.5, T=1.0)
    expected = 0.1 * 4.0 * 1.0 + 20.0 * math.log(1 + 0.1 / 1.5)
    assert optimal_spread(0.0, p) == pytest.approx(expected)
    # At expiry only the liquidity term remains.
    assert optimal_spread(1.0, p) == pytest.approx(20.0 * math.log(1 + 0.1 / 1.5))


def test_quotes_skew_against_inventory():
    """Inventory quotes skew away from the position; symmetric ones do not."""
    p = ASParams()
    b0, a0 = quotes(100.0, 0, 0.0, p)
    b_long, a_long = quotes(100.0, 5, 0.0, p)
    b_short, a_short = quotes(100.0, -5, 0.0, p)
    assert a0 - b0 == pytest.approx(optimal_spread(0.0, p))
    assert (b0 + a0) / 2 == pytest.approx(100.0)
    assert b_long < b0 and a_long < a0  # long: lower quotes to sell
    assert b_short > b0 and a_short > a0  # short: raise quotes to buy
    bs, as_ = quotes(100.0, 5, 0.0, p, strategy="symmetric")
    assert (bs, as_) == pytest.approx((b0, a0))


def test_fill_probability_bounds_and_models():
    """Fill probabilities lie in [0, 1] and fall with distance."""
    p = ASParams()
    d = np.array([-10.0, 0.0, 0.5, 5.0])
    prob = fill_probability(d, p)
    assert np.all((prob >= 0) & (prob <= 1))
    assert np.all(np.diff(prob) <= 0)
    lin = fill_probability(d, ASParams(fill_model="linear"))
    assert lin[0] == 1.0
    assert lin[2] == pytest.approx(140 * math.exp(-1.5 * 0.5) * 0.005)


def test_simulation_shapes_and_accounting():
    """Arrays have the right shape and P&L is cash plus marked inventory."""
    p = ASParams(T=0.5, dt=0.01)
    r = simulate(p, n_paths=7, seed=3)
    assert r.mid.shape == (7, p.n_steps + 1)
    assert np.all(r.mid[:, 0] == p.s0)
    np.testing.assert_allclose(r.pnl, r.cash + r.inventory * r.mid)
    # Each step changes inventory by at most one unit per side.
    assert np.all(np.abs(np.diff(r.inventory, axis=1)) <= 1)
    net = r.n_bid_fills - r.n_ask_fills
    np.testing.assert_array_equal(r.final_inventory, p.q0 + net)


def test_seed_is_reproducible():
    """The same seed gives the same paths."""
    a = simulate(seed=42, n_paths=3)
    b = simulate(seed=42, n_paths=3)
    np.testing.assert_array_equal(a.pnl, b.pnl)


def test_compare_uses_common_price_paths():
    """Both strategies see the same mid-price paths."""
    res = compare(ASParams(T=0.2), n_paths=5, seed=1)
    np.testing.assert_array_equal(res["inventory"].mid, res["symmetric"].mid)


def test_inventory_limit_is_respected():
    """Inventory never exceeds q_max."""
    p = ASParams(q_max=2, gamma=0.001)
    r = simulate(p, strategy="symmetric", n_paths=50, seed=0)
    assert np.abs(r.inventory).max() <= 2


def test_zero_volatility_strategies_coincide():
    """With sigma = 0 both strategies quote identically."""
    p = ASParams(sigma=0.0)
    res = compare(p, n_paths=20, seed=5)
    np.testing.assert_allclose(res["inventory"].pnl, res["symmetric"].pnl)


def test_inventory_strategy_reduces_risk():
    """Main result of the paper: lower P&L variance and tighter inventory."""
    res = compare(ASParams(), n_paths=500, seed=7)
    inv, sym = res["inventory"].summary(), res["symmetric"].summary()
    assert inv["std_pnl"] < 0.7 * sym["std_pnl"]
    assert inv["std_final_q"] < 0.5 * sym["std_final_q"]
    assert inv["mean_pnl"] > 0


def test_paper_table_gamma_01():
    """Close to the gamma = 0.1 row of the paper using its linear fill model."""
    res = compare(ASParams(fill_model="linear"), n_paths=1000, seed=1)
    inv = res["inventory"].summary()
    assert inv["mean_pnl"] == pytest.approx(62.94, rel=0.1)
    assert inv["std_pnl"] == pytest.approx(5.89, rel=0.25)
    assert inv["std_final_q"] == pytest.approx(2.80, rel=0.2)


def test_estimate_sigma():
    """Sigma estimate recovers the step volatility and scales with sqrt(time)."""
    rng = np.random.default_rng(0)
    prices = 100 + np.cumsum(rng.normal(0, 0.5, 20000))
    assert estimate_sigma(prices, periods_per_unit=1) == pytest.approx(0.5, rel=0.03)
    assert estimate_sigma(prices, 4) == pytest.approx(1.0, rel=0.03)


@pytest.mark.parametrize(
    "kwargs", [{"gamma": 0}, {"dt": 2.0}, {"sigma": -1}, {"fill_model": "x"}]
)
def test_invalid_params(kwargs):
    """Invalid parameters raise ValueError."""
    with pytest.raises(ValueError):
        ASParams(**kwargs)
