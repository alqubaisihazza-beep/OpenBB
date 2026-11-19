"""Avellaneda-Stoikov market making simulator.

Implements the finite-horizon optimal quoting model from

    M. Avellaneda and S. Stoikov, "High-frequency trading in a limit order
    book", Quantitative Finance 8(3), 2008, pp. 217-224.

The mid price follows an arithmetic Brownian motion ``dS = sigma dW``. A market
maker with CARA utility (risk aversion ``gamma``) and inventory ``q`` quotes
around a *reservation price*

    r(s, q, t) = s - q * gamma * sigma**2 * (T - t)

with a total bid-ask spread of

    delta_a + delta_b = gamma * sigma**2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)

Market orders hit a quote at distance ``delta`` from the mid with Poisson
intensity ``lambda(delta) = A * exp(-k * delta)``.

The simulator is vectorised over Monte Carlo paths and uses common random
numbers, so the inventory-aware strategy and the symmetric benchmark (same
spread, centred on the mid) are compared on identical price paths and order
flow.

Only NumPy is required. ``SimulationResult.to_frame`` returns a pandas
DataFrame when pandas is installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Strategy = Literal["inventory", "symmetric"]


@dataclass(frozen=True)
class ASParams:
    """Model and simulation parameters.

    Defaults are the values used in section 4 of the paper.

    Parameters
    ----------
    s0 : float
        Initial mid price.
    T : float
        Terminal time (horizon length).
    sigma : float
        Volatility of the mid price, in price units per sqrt(time unit).
    dt : float
        Time step.
    gamma : float
        Risk aversion of the market maker.
    k : float
        Order book liquidity: decay of fill intensity with quote distance.
    A : float
        Order arrival intensity at zero distance from the mid.
    q0 : int
        Initial inventory.
    q_max : int | None
        Optional absolute inventory limit. At the limit the market maker stops
        quoting the side that would increase the position.
    fill_model : {"poisson", "linear"}
        How a fill probability is derived from the intensity over one step.
        ``"poisson"`` uses ``1 - exp(-lambda * dt)``. ``"linear"`` uses
        ``min(lambda * dt, 1)``, the approximation used in the paper.
    """

    s0: float = 100.0
    T: float = 1.0
    sigma: float = 2.0
    dt: float = 0.005
    gamma: float = 0.1
    k: float = 1.5
    A: float = 140.0
    q0: int = 0
    q_max: int | None = None
    fill_model: Literal["poisson", "linear"] = "poisson"

    def __post_init__(self) -> None:
        """Validate parameters."""
        for name in ("T", "dt", "gamma", "k", "A"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")
        if self.dt > self.T:
            raise ValueError("dt must not exceed T")
        if self.q_max is not None and self.q_max < 0:
            raise ValueError("q_max must be non-negative")
        if self.fill_model not in ("poisson", "linear"):
            raise ValueError("fill_model must be 'poisson' or 'linear'")

    @property
    def n_steps(self) -> int:
        """Number of time steps in the horizon."""
        return int(round(self.T / self.dt))


def reservation_price(
    s: np.ndarray | float, q: np.ndarray | float, t: float, p: ASParams
) -> np.ndarray | float:
    """Indifference (reservation) price r(s, q, t)."""
    return s - q * p.gamma * p.sigma**2 * (p.T - t)


def optimal_spread(t: float, p: ASParams) -> float:
    """Total optimal bid-ask spread delta_a + delta_b at time t."""
    return p.gamma * p.sigma**2 * (p.T - t) + (2.0 / p.gamma) * np.log(
        1.0 + p.gamma / p.k
    )


def quotes(
    s: np.ndarray | float,
    q: np.ndarray | float,
    t: float,
    p: ASParams,
    strategy: Strategy = "inventory",
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Bid and ask quotes for the given strategy.

    ``"inventory"`` centres the spread on the reservation price.
    ``"symmetric"`` centres the same spread on the mid price.
    """
    half = optimal_spread(t, p) / 2.0
    if strategy == "inventory":
        centre = reservation_price(s, q, t, p)
    elif strategy == "symmetric":
        centre = s
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return centre - half, centre + half


def fill_probability(delta: np.ndarray, p: ASParams) -> np.ndarray:
    """Probability that a quote at distance ``delta`` is hit within ``dt``.

    ``lambda = A * exp(-k * delta)``; see ``ASParams.fill_model``.
    """
    lam = p.A * np.exp(-p.k * delta)
    if p.fill_model == "linear":
        return np.minimum(lam * p.dt, 1.0)
    return -np.expm1(-lam * p.dt)


@dataclass
class SimulationResult:
    """Paths produced by :func:`simulate`.

    All path arrays have shape ``(n_paths, n_steps + 1)``. Quote arrays hold the
    quotes posted at the start of each step, so their last column is NaN.
    """

    strategy: str
    params: ASParams
    time: np.ndarray
    mid: np.ndarray
    reservation: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    inventory: np.ndarray
    cash: np.ndarray
    pnl: np.ndarray
    n_bid_fills: np.ndarray = field(repr=False)
    n_ask_fills: np.ndarray = field(repr=False)

    @property
    def final_pnl(self) -> np.ndarray:
        """Terminal mark-to-mid P&L per path."""
        return self.pnl[:, -1]

    @property
    def final_inventory(self) -> np.ndarray:
        """Terminal inventory per path."""
        return self.inventory[:, -1]

    def summary(self) -> dict[str, float]:
        """Summary statistics in the layout of the paper's tables."""
        pnl = self.final_pnl
        q = self.final_inventory
        std = float(pnl.std(ddof=1)) if pnl.size > 1 else 0.0
        return {
            "spread": float(optimal_spread(0.0, self.params)),
            "mean_pnl": float(pnl.mean()),
            "std_pnl": std,
            "pnl_to_std": float(pnl.mean() / std) if std > 0 else float("nan"),
            "mean_final_q": float(q.mean()),
            "std_final_q": float(q.std(ddof=1)) if q.size > 1 else 0.0,
            "mean_trades": float((self.n_bid_fills + self.n_ask_fills).mean()),
        }

    def to_frame(self, path: int = 0):
        """Return one path as a pandas DataFrame indexed by time."""
        import pandas as pd

        return pd.DataFrame(
            {
                "mid": self.mid[path],
                "reservation": self.reservation[path],
                "bid": self.bid[path],
                "ask": self.ask[path],
                "inventory": self.inventory[path],
                "cash": self.cash[path],
                "pnl": self.pnl[path],
            },
            index=pd.Index(self.time, name="t"),
        )


@dataclass(frozen=True)
class _Shocks:
    """Random draws shared between strategies (common random numbers)."""

    dW: np.ndarray
    u_bid: np.ndarray
    u_ask: np.ndarray


def _draw_shocks(p: ASParams, n_paths: int, rng: np.random.Generator) -> _Shocks:
    shape = (n_paths, p.n_steps)
    return _Shocks(
        dW=rng.standard_normal(shape) * np.sqrt(p.dt),
        u_bid=rng.random(shape),
        u_ask=rng.random(shape),
    )


def _run(p: ASParams, strategy: Strategy, shocks: _Shocks) -> SimulationResult:
    n_paths, n_steps = shocks.dW.shape
    cols = n_steps + 1
    time = np.linspace(0.0, n_steps * p.dt, cols)

    mid = np.empty((n_paths, cols))
    mid[:, 0] = p.s0
    mid[:, 1:] = p.s0 + p.sigma * np.cumsum(shocks.dW, axis=1)

    inv = np.empty((n_paths, cols))
    cash = np.empty((n_paths, cols))
    res = np.full((n_paths, cols), np.nan)
    bid = np.full((n_paths, cols), np.nan)
    ask = np.full((n_paths, cols), np.nan)
    inv[:, 0] = p.q0
    cash[:, 0] = 0.0
    n_bid = np.zeros(n_paths, dtype=np.int64)
    n_ask = np.zeros(n_paths, dtype=np.int64)

    for i in range(n_steps):
        t = time[i]
        s = mid[:, i]
        q = inv[:, i]
        b, a = quotes(s, q, t, p, strategy)
        res[:, i] = reservation_price(s, q, t, p)

        can_buy = np.ones(n_paths, dtype=bool)
        can_sell = np.ones(n_paths, dtype=bool)
        if p.q_max is not None:
            can_buy = q < p.q_max
            can_sell = q > -p.q_max

        bid[:, i] = np.where(can_buy, b, np.nan)
        ask[:, i] = np.where(can_sell, a, np.nan)

        # Quotes that cross the mid are filled at intensity > A, capped at 1.
        p_bid = fill_probability(s - b, p)
        p_ask = fill_probability(a - s, p)
        hit_bid = can_buy & (shocks.u_bid[:, i] < p_bid)
        hit_ask = can_sell & (shocks.u_ask[:, i] < p_ask)

        inv[:, i + 1] = q + hit_bid - hit_ask
        cash[:, i + 1] = (
            cash[:, i] - np.where(hit_bid, b, 0.0) + np.where(hit_ask, a, 0.0)
        )
        n_bid += hit_bid
        n_ask += hit_ask

    res[:, -1] = reservation_price(mid[:, -1], inv[:, -1], time[-1], p)
    pnl = cash + inv * mid

    return SimulationResult(
        strategy=strategy,
        params=p,
        time=time,
        mid=mid,
        reservation=res,
        bid=bid,
        ask=ask,
        inventory=inv,
        cash=cash,
        pnl=pnl,
        n_bid_fills=n_bid,
        n_ask_fills=n_ask,
    )


def simulate(
    params: ASParams | None = None,
    strategy: Strategy = "inventory",
    n_paths: int = 1,
    seed: int | None = None,
) -> SimulationResult:
    """Simulate one strategy over ``n_paths`` Monte Carlo paths."""
    p = params or ASParams()
    if n_paths < 1:
        raise ValueError("n_paths must be at least 1")
    shocks = _draw_shocks(p, n_paths, np.random.default_rng(seed))
    return _run(p, strategy, shocks)


def compare(
    params: ASParams | None = None,
    n_paths: int = 1000,
    seed: int | None = None,
) -> dict[str, SimulationResult]:
    """Run both strategies on the same price paths and order flow."""
    p = params or ASParams()
    if n_paths < 1:
        raise ValueError("n_paths must be at least 1")
    shocks = _draw_shocks(p, n_paths, np.random.default_rng(seed))
    return {s: _run(p, s, shocks) for s in ("inventory", "symmetric")}


def summary_table(results: dict[str, SimulationResult]):
    """Summaries of :func:`compare` output as a pandas DataFrame."""
    import pandas as pd

    return pd.DataFrame({name: r.summary() for name, r in results.items()}).T


def estimate_sigma(prices, periods_per_unit: float = 252.0) -> float:
    """Estimate the arithmetic volatility ``sigma`` from a price series.

    ``sigma`` in this model is in price units, so it is the standard deviation
    of price *changes*, scaled to one time unit. With daily closes and
    ``periods_per_unit=252`` the result is per year; use ``periods_per_unit=1``
    to keep a per-day sigma.
    """
    x = np.asarray(prices, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        raise ValueError("need at least 3 prices")
    return float(np.diff(x).std(ddof=1) * np.sqrt(periods_per_unit))
