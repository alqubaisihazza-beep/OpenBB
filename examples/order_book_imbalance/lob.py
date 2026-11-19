"""Limit order book simulation and order book imbalance measures.

The simulator implements the stochastic order book model of

    R. Cont, S. Stoikov and R. Talreja, "A stochastic model for order book
    dynamics", Operations Research 58(3), 2010, pp. 549-563.

Prices live on an integer tick grid and every order has unit size. Within
``K`` ticks of the opposite best quote:

* limit buy orders arrive at ``best_ask - i`` with rate ``lam[i-1]``, and limit
  sell orders at ``best_bid + i`` with the same rates;
* market buy and market sell orders each arrive with rate ``mu`` and take one
  unit from the opposite best queue;
* each resting order at distance ``i`` is cancelled with rate ``theta[i-1]``.

The imbalance functions work on any arrays of quotes, so they apply equally to
simulated output and to real top-of-book data:

* ``queue_imbalance``: (Vb - Va) / (Vb + Va) at the best quotes.
* ``depth_imbalance``: the same over several levels, optionally decayed.
* ``weighted_mid``: the size-weighted mid (a simple micro-price).
* ``order_flow_imbalance``: per-event order flow imbalance (OFI) from
  R. Cont, A. Kukanov and S. Stoikov, "The price impact of order book events",
  Journal of Financial Econometrics 12(1), 2014.
* ``ofi_regression``: the linear price impact regression dP = beta * OFI.
* ``imbalance_signal_table``: how well queue imbalance predicts the direction
  of the next mid-price move.

Only NumPy is required. Functions that return tables need pandas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Rates estimated by Cont, Stoikov and Talreja (2010), table 2, per second.
CST_LAMBDA = (1.85, 1.51, 1.09, 0.88, 0.77)
CST_THETA = (0.71, 0.81, 0.68, 0.56, 0.47)
CST_MU = 0.94

EVENT_NAMES = (
    "limit_buy",
    "limit_sell",
    "market_buy",
    "market_sell",
    "cancel_buy",
    "cancel_sell",
)


@dataclass(frozen=True)
class LOBParams:
    """Order book model parameters.

    Parameters
    ----------
    lam : tuple[float, ...]
        Limit order arrival rates at distances 1..K ticks from the opposite
        best quote, per side.
    theta : tuple[float, ...]
        Cancellation rate per resting order at distances 1..K.
    mu : float
        Market order arrival rate per side.
    n_ticks : int
        Width of the price grid.
    initial_depth : int
        Orders placed at every level when the book is initialised.
    initial_spread : int
        Spread in ticks at the start.
    levels : int
        Number of levels per side recorded in the history.
    tick_size : float
        Price of one tick, used only when converting to prices.
    """

    lam: tuple[float, ...] = CST_LAMBDA
    theta: tuple[float, ...] = CST_THETA
    mu: float = CST_MU
    n_ticks: int = 2000
    initial_depth: int = 5
    initial_spread: int = 1
    levels: int = 5
    tick_size: float = 0.01

    def __post_init__(self) -> None:
        """Validate parameters."""
        if len(self.lam) != len(self.theta) or not self.lam:
            raise ValueError("lam and theta must be non-empty and the same length")
        if min(self.lam) < 0 or min(self.theta) < 0 or self.mu < 0:
            raise ValueError("rates must be non-negative")
        if self.initial_depth < 1 or self.initial_spread < 1 or self.levels < 1:
            raise ValueError("initial_depth, initial_spread and levels must be >= 1")
        if self.n_ticks < 4 * (len(self.lam) + self.levels + self.initial_spread):
            raise ValueError("n_ticks is too small for the number of levels")

    @property
    def K(self) -> int:
        """Number of active levels."""
        return len(self.lam)


@dataclass
class BookHistory:
    """State of the book after each event.

    Prices are in ticks. ``bid_depth[n, j]`` is the size ``j`` ticks below the
    best bid after event ``n`` (``ask_depth`` likewise above the best ask); row
    0 is the initial book.
    """

    params: LOBParams
    time: np.ndarray
    event: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    bid_depth: np.ndarray
    ask_depth: np.ndarray
    truncated: bool = field(default=False)

    @property
    def bid_size(self) -> np.ndarray:
        """Size at the best bid."""
        return self.bid_depth[:, 0]

    @property
    def ask_size(self) -> np.ndarray:
        """Size at the best ask."""
        return self.ask_depth[:, 0]

    @property
    def mid(self) -> np.ndarray:
        """Mid price in ticks."""
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> np.ndarray:
        """Spread in ticks."""
        return self.ask - self.bid

    def to_frame(self):
        """Top of book as a pandas DataFrame indexed by event time."""
        import pandas as pd

        ts = self.params.tick_size
        return pd.DataFrame(
            {
                "event": np.asarray(["init", *EVENT_NAMES])[self.event + 1],
                "bid": self.bid * ts,
                "ask": self.ask * ts,
                "bid_size": self.bid_size,
                "ask_size": self.ask_size,
                "mid": self.mid * ts,
                "imbalance": queue_imbalance(self.bid_size, self.ask_size),
            },
            index=pd.Index(self.time, name="time"),
        )


def simulate_lob(
    params: LOBParams | None = None,
    n_events: int = 100_000,
    seed: int | None = None,
) -> BookHistory:
    """Simulate the Cont-Stoikov-Talreja order book for ``n_events`` events.

    The simulation stops early, with ``truncated=True``, if the best quotes come
    within a few ticks of the edge of the price grid.
    """
    p = params or LOBParams()
    if n_events < 1:
        raise ValueError("n_events must be at least 1")
    rng = np.random.default_rng(seed)
    K, L, N = p.K, p.levels, p.n_ticks
    lam = np.asarray(p.lam, dtype=float)
    theta = np.asarray(p.theta, dtype=float)
    lam_total = lam.sum()

    # Signed book: positive sizes are asks, negative sizes are bids.
    x = np.zeros(N, dtype=np.int64)
    centre = N // 2
    best_bid = centre - (p.initial_spread + 1) // 2
    best_ask = best_bid + p.initial_spread
    x[: best_bid + 1] = -p.initial_depth
    x[best_ask:] = p.initial_depth

    time = np.empty(n_events + 1)
    event = np.empty(n_events + 1, dtype=np.int8)
    bid = np.empty(n_events + 1, dtype=np.int64)
    ask = np.empty(n_events + 1, dtype=np.int64)
    bdep = np.empty((n_events + 1, L), dtype=np.int64)
    adep = np.empty((n_events + 1, L), dtype=np.int64)

    def record(n: int, t: float, ev: int) -> None:
        time[n] = t
        event[n] = ev
        bid[n] = best_bid
        ask[n] = best_ask
        bdep[n] = -x[best_bid - L + 1 : best_bid + 1][::-1]
        adep[n] = x[best_ask : best_ask + L]

    record(0, 0.0, -1)
    t = 0.0
    margin = K + L + 2
    last = n_events
    truncated = False

    for n in range(1, n_events + 1):
        # Resting orders at distance 1..K from the opposite best quote.
        qb = -x[best_ask - K : best_ask][::-1]  # bids at best_ask - i
        qa = x[best_bid + 1 : best_bid + K + 1]  # asks at best_bid + i
        qb = np.maximum(qb, 0)
        qa = np.maximum(qa, 0)
        cb = theta * qb
        ca = theta * qa
        rates = np.array([lam_total, lam_total, p.mu, p.mu, cb.sum(), ca.sum()])
        total = rates.sum()
        t += rng.exponential(1.0 / total)
        ev = int(np.searchsorted(np.cumsum(rates), rng.random() * total, side="right"))
        ev = min(ev, 5)

        if ev == 0:  # limit buy at best_ask - i
            i = 1 + _pick(rng, lam)
            px = best_ask - i
            x[px] -= 1
            best_bid = max(best_bid, px)
        elif ev == 1:  # limit sell at best_bid + i
            i = 1 + _pick(rng, lam)
            px = best_bid + i
            x[px] += 1
            best_ask = min(best_ask, px)
        elif ev == 2:  # market buy lifts the best ask
            x[best_ask] -= 1
            while x[best_ask] <= 0:
                best_ask += 1
        elif ev == 3:  # market sell hits the best bid
            x[best_bid] += 1
            while x[best_bid] >= 0:
                best_bid -= 1
        elif ev == 4:  # cancel a bid at best_ask - i
            px = best_ask - 1 - _pick(rng, cb)
            x[px] += 1
            while x[best_bid] >= 0:
                best_bid -= 1
        else:  # cancel an ask at best_bid + i
            px = best_bid + 1 + _pick(rng, ca)
            x[px] -= 1
            while x[best_ask] <= 0:
                best_ask += 1

        record(n, t, ev)
        if best_bid < margin or best_ask > N - margin:
            last, truncated = n, True
            break

    s = slice(0, last + 1)
    return BookHistory(
        params=p,
        time=time[s],
        event=event[s],
        bid=bid[s],
        ask=ask[s],
        bid_depth=bdep[s],
        ask_depth=adep[s],
        truncated=truncated,
    )


def _pick(rng: np.random.Generator, weights: np.ndarray) -> int:
    c = np.cumsum(weights)
    return min(int(np.searchsorted(c, rng.random() * c[-1], side="right")), len(c) - 1)


# ---------------------------------------------------------------------------
# Imbalance measures
# ---------------------------------------------------------------------------


def queue_imbalance(bid_size, ask_size) -> np.ndarray:
    """Best-level queue imbalance (Vb - Va) / (Vb + Va), in [-1, 1].

    Positive values mean more size on the bid. Returns NaN where both queues
    are empty.
    """
    b = np.asarray(bid_size, dtype=float)
    a = np.asarray(ask_size, dtype=float)
    tot = b + a
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, (b - a) / tot, np.nan)


def depth_imbalance(
    bid_depth, ask_depth, levels: int | None = None, decay: float = 0.0
) -> np.ndarray:
    """Imbalance over several levels with weights ``exp(-decay * level)``.

    ``bid_depth`` and ``ask_depth`` have shape ``(n, L)`` with the best level
    first. ``decay=0`` weights all levels equally.
    """
    b = np.asarray(bid_depth, dtype=float)
    a = np.asarray(ask_depth, dtype=float)
    if levels is not None:
        b, a = b[:, :levels], a[:, :levels]
    w = np.exp(-decay * np.arange(b.shape[1]))
    return queue_imbalance(b @ w, a @ w)


def weighted_mid(bid, ask, bid_size, ask_size) -> np.ndarray:
    """Size-weighted mid price, ``(ask * Vb + bid * Va) / (Vb + Va)``.

    This leans toward the side with less size, the side more likely to be
    depleted next. Falls back to the plain mid where both queues are empty.
    """
    b, a = np.asarray(bid, dtype=float), np.asarray(ask, dtype=float)
    imb = queue_imbalance(bid_size, ask_size)
    imb = np.where(np.isnan(imb), 0.0, imb)
    return (a + b) / 2.0 + imb * (a - b) / 2.0


def order_flow_imbalance(bid, bid_size, ask, ask_size) -> np.ndarray:
    """Per-update order flow imbalance e_n of Cont, Kukanov and Stoikov.

    e_n = 1{Pb_n >= Pb_n-1} qb_n - 1{Pb_n <= Pb_n-1} qb_n-1
        - 1{Pa_n <= Pa_n-1} qa_n + 1{Pa_n >= Pa_n-1} qa_n-1

    Positive values are net buying pressure. The first element is 0.
    """
    pb, qb = np.asarray(bid, dtype=float), np.asarray(bid_size, dtype=float)
    pa, qa = np.asarray(ask, dtype=float), np.asarray(ask_size, dtype=float)
    e = np.zeros(pb.shape)
    e[1:] = (
        (pb[1:] >= pb[:-1]) * qb[1:]
        - (pb[1:] <= pb[:-1]) * qb[:-1]
        - (pa[1:] <= pa[:-1]) * qa[1:]
        + (pa[1:] >= pa[:-1]) * qa[:-1]
    )
    return e


def aggregate_ofi(
    time, ofi, mid, interval: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum OFI and take mid changes over consecutive windows of length ``interval``.

    Returns ``(window_start, ofi_sum, mid_change)``. The mid change is from the
    last state before the window opens to the last state inside it.
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(ofi, dtype=float)
    m = np.asarray(mid, dtype=float)
    if interval <= 0:
        raise ValueError("interval must be positive")
    edges = np.arange(t[0], t[-1] + interval, interval)
    if edges.size < 2:
        raise ValueError("interval is longer than the sample")
    # Index of the last observation at or before each edge.
    idx = np.searchsorted(t, edges, side="right") - 1
    csum = np.concatenate([[0.0], np.cumsum(e)])
    ofi_sum = csum[idx[1:] + 1] - csum[idx[:-1] + 1]
    dmid = m[idx[1:]] - m[idx[:-1]]
    return edges[:-1], ofi_sum, dmid


def ofi_regression(ofi, dmid) -> dict[str, float]:
    """Ordinary least squares fit ``dmid = alpha + beta * ofi``.

    Cont, Kukanov and Stoikov find beta close to ``1 / (2 * depth)``, where
    depth is the typical size at the best quotes.
    """
    xv = np.asarray(ofi, dtype=float)
    yv = np.asarray(dmid, dtype=float)
    ok = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[ok], yv[ok]
    if xv.size < 3 or np.var(xv) == 0:
        raise ValueError("need at least 3 observations with varying OFI")
    beta, alpha = np.polyfit(xv, yv, 1)
    resid = yv - (alpha + beta * xv)
    ss_tot = ((yv - yv.mean()) ** 2).sum()
    r2 = 1.0 - (resid**2).sum() / ss_tot if ss_tot > 0 else float("nan")
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "r2": float(r2),
        "n": int(xv.size),
    }


def next_mid_move(mid) -> np.ndarray:
    """For each observation, the size of the next non-zero mid-price change.

    NaN where the mid never changes again.
    """
    m = np.asarray(mid, dtype=float)
    d = np.diff(m)
    nz = np.flatnonzero(d)
    out = np.full(m.shape, np.nan)
    if nz.size == 0:
        return out
    # For position n, the next change happens at the first nz index >= n.
    pos = np.searchsorted(nz, np.arange(m.size), side="left")
    has = pos < nz.size
    out[has] = d[nz[pos[has]]]
    return out


def imbalance_signal_table(imbalance, mid, n_buckets: int = 11, spread=None):
    """Next mid move statistics by imbalance bucket.

    For each bucket of equal width on [-1, 1] (an odd count keeps zero
    imbalance in the middle of a bucket): the number of observations, the
    mean imbalance, the probability that the next mid change is upward, and
    the mean size of that next change (in the units of ``mid``). Pass
    ``spread`` to keep only observations at the minimum spread, where queue
    imbalance is most informative.
    """
    import pandas as pd

    imb = np.asarray(imbalance, dtype=float)
    nxt = next_mid_move(mid)
    ok = np.isfinite(imb) & np.isfinite(nxt)
    if spread is not None:
        sp = np.asarray(spread, dtype=float)
        ok &= sp == np.nanmin(sp)
    imb, nxt = imb[ok], nxt[ok]
    edges = np.linspace(-1.0, 1.0, n_buckets + 1)
    b = np.clip(np.digitize(imb, edges[1:-1]), 0, n_buckets - 1)
    rows = []
    for k in range(n_buckets):
        sel = b == k
        cnt = int(sel.sum())
        rows.append(
            {
                "bucket": f"[{edges[k]:+.2f}, {edges[k + 1]:+.2f}{']' if k == n_buckets - 1 else ')'}",
                "count": cnt,
                "mean_imbalance": float(imb[sel].mean()) if cnt else np.nan,
                "p_up": float((nxt[sel] > 0).mean()) if cnt else np.nan,
                "mean_next_move": float(nxt[sel].mean()) if cnt else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("bucket")


def imbalance_accuracy(imbalance, mid) -> float:
    """Share of observations where sign(imbalance) matches the next mid move.

    Observations with zero imbalance or no later move are excluded.
    """
    imb = np.asarray(imbalance, dtype=float)
    nxt = next_mid_move(mid)
    ok = np.isfinite(imb) & np.isfinite(nxt) & (imb != 0)
    if not ok.any():
        return float("nan")
    return float((np.sign(imb[ok]) == np.sign(nxt[ok])).mean())
