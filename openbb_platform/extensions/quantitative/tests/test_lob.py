"""Tests for the ``/quantitative/lob`` commands."""

import numpy as np
import pandas as pd
import pytest
from openbb_core.app.utils import df_to_basemodel

from openbb_quantitative import (
    _lob_helpers as helpers,
    lob,
)

TICK = 0.01


def simulate_book(n=6000, n_levels=3, seed=7, fill_prob=0.5):
    """Symmetric queue-depletion book in which imbalance predicts the next move.

    Best queues random-walk; when one empties that quote steps back a tick and
    the spread widens, and a one-tick gap then fills from either side.
    """
    rng = np.random.default_rng(seed)
    bid, ask = 10000, 10001  # ticks
    qb, qa = 50.0, 50.0
    rows = []
    for _ in range(n):
        qb += rng.normal(0, 8)
        qa += rng.normal(0, 8)
        wide = ask - bid > 1
        if qa <= 0:
            ask += 1
            bid += 1 if wide else 0
            qa = rng.uniform(20, 80)
            if wide:
                qb = rng.uniform(5, 30)
        elif qb <= 0:
            bid -= 1
            ask -= 1 if wide else 0
            qb = rng.uniform(20, 80)
            if wide:
                qa = rng.uniform(5, 30)
        elif wide and rng.random() < fill_prob:
            if rng.random() < 0.5:
                bid += 1
                qb = rng.uniform(5, 30)
            else:
                ask -= 1
                qa = rng.uniform(5, 30)
        row = {}
        for lvl in range(1, n_levels + 1):
            row[f"bid_price_{lvl}"] = round((bid - lvl + 1) * TICK, 2)
            row[f"bid_size_{lvl}"] = qb if lvl == 1 else 60.0 + 10 * lvl
            row[f"ask_price_{lvl}"] = round((ask + lvl - 1) * TICK, 2)
            row[f"ask_size_{lvl}"] = qa if lvl == 1 else 60.0 + 10 * lvl
        rows.append(row)
    df = pd.DataFrame(rows)
    df.insert(0, "date", pd.date_range("2024-01-02 09:30", periods=n, freq="100ms"))
    return df


@pytest.fixture(scope="module")
def book_df():
    return simulate_book()


@pytest.fixture(scope="module")
def book_data(book_df):
    return df_to_basemodel(book_df)


@pytest.fixture
def tiny_data():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=5, freq="s"),
            "bid": [10.00, 10.00, 10.01, 10.01, 10.00],
            "ask": [10.02, 10.02, 10.02, 10.03, 10.02],
            "bid_size": [100.0, 120.0, 30.0, 30.0, 50.0],
            "ask_size": [100.0, 90.0, 90.0, 40.0, 60.0],
        }
    )
    return df_to_basemodel(frame)


def test_features_by_hand(tiny_data):
    out = lob.features(lob.LobFeaturesQueryParams(data=tiny_data)).results
    assert len(out) == 5
    assert out[0].mid_price == pytest.approx(10.01)
    assert out[0].ofi is None
    assert [r.ofi for r in out[1:]] == pytest.approx([30.0, 30.0, 90.0, -90.0])
    assert out[1].queue_imbalance == pytest.approx(30 / 210)
    assert out[2].weighted_mid == pytest.approx((10.01 * 90 + 10.02 * 30) / 120)
    assert out[3].spread == pytest.approx(0.02)


def test_features_multilevel(book_data, book_df):
    out = lob.features(
        lob.LobFeaturesQueryParams(data=book_data[:50], levels=2, decay=0.5)
    ).results
    frame = helpers.to_book(book_df.iloc[:50].set_index("date"))
    expected = helpers.depth_imbalance(frame, levels=2, decay=0.5)
    assert [r.depth_imbalance for r in out] == pytest.approx(expected.tolist())
    flows = helpers.order_flow_imbalance(frame, levels=2).sum(axis=1)
    assert out[5].ofi == pytest.approx(flows.iloc[5])


def test_ofi_windows(book_data):
    out = lob.ofi(lob.LobOfiQueryParams(data=book_data, freq="10s")).results
    assert {r.level for r in out} == {1, 2, 3}
    windows = {r.date for r in out}
    assert len(windows) == 60
    assert all(r.integrated_ofi is not None for r in out)
    count_based = lob.ofi(
        lob.LobOfiQueryParams(data=book_data, freq="100", levels=1)
    ).results
    assert len(count_based) == 60
    assert all(r.integrated_ofi is None for r in count_based)


def test_price_impact(book_data):
    res = lob.price_impact(lob.LobPriceImpactQueryParams(data=book_data, freq="10s"))
    out = res.results
    assert out.beta > 0
    assert out.beta_tstat > 3
    assert 0 < out.r2 <= 1
    assert out.nobs == 60


def test_imbalance_signal(book_data):
    out = lob.imbalance_signal(
        lob.LobImbalanceSignalQueryParams(data=book_data, bins=4)
    ).results
    assert [r.imbalance_low for r in out] == [-1.0, -0.5, 0.0, 0.5]
    p_up = [r.p_up for r in out]
    assert p_up == sorted(p_up)
    assert out[0].mean_move < 0 < out[-1].mean_move
    depth = lob.imbalance_signal(
        lob.LobImbalanceSignalQueryParams(data=book_data, bins=4, signal="depth")
    ).results
    assert len(depth) == 4


def test_microprice(book_data):
    out = lob.microprice(
        lob.LobMicropriceQueryParams(data=book_data, n_imbalance=4)
    ).results
    assert len(out) == 8
    one_tick = [r for r in out if r.spread_ticks == 1]
    adj = [r.adjustment_ticks for r in one_tick]
    assert adj == sorted(adj)
    assert adj[0] == pytest.approx(-adj[-1])
    assert all(abs(a) < 0.5 for a in adj)
    assert one_tick[0].adjustment == pytest.approx(adj[0] * TICK)


def test_examples_run():
    """Every documented example payload runs without error."""
    from openbb_core.app.model.example import APIEx

    for command, params_cls in [
        (lob.features, lob.LobFeaturesQueryParams),
        (lob.ofi, lob.LobOfiQueryParams),
        (lob.price_impact, lob.LobPriceImpactQueryParams),
        (lob.imbalance_signal, lob.LobImbalanceSignalQueryParams),
        (lob.microprice, lob.LobMicropriceQueryParams),
    ]:
        examples = lob.router.api_router.routes
        route = next(r for r in examples if r.endpoint.__name__ == command.__name__)
        example = route.openapi_extra["examples"][0]
        assert isinstance(example, APIEx)
        params = dict(example.parameters)
        params["data"] = df_to_basemodel(pd.DataFrame(params["data"]))
        assert command(params_cls(**params)).results


def test_errors(tiny_data):
    with pytest.raises(ValueError, match="two"):
        lob.features(lob.LobFeaturesQueryParams(data=tiny_data[:1]))
    with pytest.raises(ValueError):
        lob.features(lob.LobFeaturesQueryParams(data=tiny_data, levels=2))
