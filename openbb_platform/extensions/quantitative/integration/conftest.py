"""Integration-test fixtures for openbb-quantitative."""

import threading
import time

import numpy as np
import pandas as pd
import pytest

_SEED = 20240517
_N_ROWS = 500


@pytest.fixture(scope="session", autouse=True)
def api_server(pytestconfig):
    """Run the OpenBB REST API on 0.0.0.0:8000 for the integration session."""
    if pytestconfig.getoption("markexpr") == "not integration":
        yield None
        return

    import uvicorn
    from openbb_core.api.rest_api import app

    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")  # noqa: S104
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 60
    while time.time() < deadline:
        if server.started:
            break
        time.sleep(0.25)
    else:  # pragma: no cover
        raise RuntimeError("The OpenBB REST API did not start within 60s.")

    yield server

    server.should_exit = True
    thread.join(timeout=15)


def _timeseries_records() -> list[dict]:
    """Build a deterministic OHLCV time series as JSON-ready records."""
    rng = np.random.default_rng(_SEED)
    t = np.arange(_N_ROWS)
    base = 100 + 0.1 * t + rng.normal(0, 1, _N_ROWS).cumsum() * 0.3
    close = np.abs(base + rng.normal(0, 1, _N_ROWS)) + 1.0
    open_ = close + rng.normal(0, 0.5, _N_ROWS)
    high = np.maximum(open_, close) + np.abs(rng.normal(1, 0.5, _N_ROWS))
    low = np.minimum(open_, close) - np.abs(rng.normal(1, 0.5, _N_ROWS))
    volume = 1_000_000 + rng.normal(0, 100_000, _N_ROWS) + 5_000 * t
    dates = pd.date_range("2021-01-04", periods=_N_ROWS, freq="B")
    return [
        {
            "date": day.strftime("%Y-%m-%d"),
            "open": round(float(o), 4),
            "high": round(float(h), 4),
            "low": round(float(lo), 4),
            "close": round(float(c), 4),
            "volume": round(float(v), 2),
        }
        for day, o, h, lo, c, v in zip(dates, open_, high, low, close, volume)
    ]


@pytest.fixture(scope="session")
def prices_data() -> list[dict]:
    """Deterministic OHLCV records for the quantitative command tests."""
    return _timeseries_records()


def _book_records(n: int = 2000, tick: float = 0.01) -> list[dict]:
    """Two-level order book snapshots where the thinner queue empties first."""
    rng = np.random.default_rng(_SEED)
    bid, qb, qa = 10000, 50.0, 50.0
    start = pd.Timestamp("2024-01-02 09:30")
    rows = []
    for i in range(n):
        qb += rng.normal(0, 8)
        qa += rng.normal(0, 8)
        if qa <= 0:
            bid += 1
            qb, qa = rng.uniform(20, 80), rng.uniform(20, 80)
        elif qb <= 0:
            bid -= 1
            qb, qa = rng.uniform(20, 80), rng.uniform(20, 80)
        rows.append(
            {
                "date": (start + pd.Timedelta(milliseconds=100 * i)).isoformat(),
                "bid_price_1": round(bid * tick, 2),
                "bid_size_1": round(float(qb), 2),
                "ask_price_1": round((bid + 1) * tick, 2),
                "ask_size_1": round(float(qa), 2),
                "bid_price_2": round((bid - 1) * tick, 2),
                "bid_size_2": 70.0,
                "ask_price_2": round((bid + 2) * tick, 2),
                "ask_size_2": 70.0,
            }
        )
    return rows


@pytest.fixture(scope="session")
def book_data() -> list[dict]:
    """Order book snapshots as JSON-ready records."""
    return _book_records()
