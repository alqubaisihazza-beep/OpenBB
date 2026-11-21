"""Quantitative limit order book (LOB) and imbalance commands.

Every command takes order book snapshots as ``data``: one record per snapshot,
in time order, with ``bid_price_1``, ``bid_size_1``, ``ask_price_1`` and
``ask_size_1`` (and ``_2``, ``_3``, ... for deeper levels). Top-of-book data
may use ``bid``, ``ask``, ``bid_size`` and ``ask_size`` instead.
"""

from datetime import (
    date as dateType,
    datetime,
)
from typing import TYPE_CHECKING, Literal

from openbb_core.app.model.example import APIEx
from openbb_core.app.model.obbject import OBBject
from openbb_core.app.router import Router
from openbb_core.provider.abstract.data import Data
from openbb_core.provider.abstract.query_params import QueryParams
from pydantic import Field, PositiveInt

if TYPE_CHECKING:
    from pandas import DataFrame

router = Router(
    prefix="/lob", description="Limit order book and order-flow imbalance commands."
)

_BOOK_SAMPLE = {
    "date": "2024-01-02",
    "bid": 100.0,
    "ask": 100.01,
    "bid_size": 50.0,
    "ask_size": 40.0,
}

# Twelve snapshots of a one-tick market whose mid moves by half ticks, so the
# micro-price example has transitions to estimate from.
_MICRO_SAMPLE = [
    {
        "date": f"2024-01-02T09:30:{i:02d}",
        "bid": bid,
        "ask": round(bid + 0.01 * spread, 2),
        "bid_size": qb,
        "ask_size": qa,
    }
    for i, (bid, spread, qb, qa) in enumerate(
        [
            (100.00, 1, 80, 20),
            (100.00, 2, 60, 10),
            (100.01, 1, 30, 70),
            (100.01, 1, 20, 80),
            (100.00, 2, 50, 50),
            (100.00, 1, 70, 30),
            (100.01, 1, 40, 60),
            (100.01, 1, 15, 85),
            (100.00, 2, 45, 55),
            (100.00, 1, 85, 15),
            (100.00, 2, 55, 12),
            (100.01, 1, 35, 65),
        ]
    )
]

BookIndex = Field(default="date", description="Name of the timestamp column.")


def _book(data: list[Data], index: str) -> "DataFrame":
    """Convert snapshot records to a canonical book frame indexed by time."""
    from contextlib import suppress

    from openbb_core.app.utils import basemodel_to_df
    from pandas import DatetimeIndex, to_datetime

    from openbb_quantitative._lob_helpers import to_book

    df = basemodel_to_df(data, index=index)
    # Dates arrive as strings or date objects; integer indexes stay as they are.
    if not isinstance(df.index, DatetimeIndex) and df.index.dtype == object:
        with suppress(TypeError, ValueError):
            df.index = to_datetime(df.index)
    if len(df) < 2:  # noqa: PLR2004
        raise ValueError("Need at least two order book snapshots.")
    return to_book(df)


def _freq(value: str | int) -> str | int:
    """Read a window that may be a count of snapshots sent as a string."""
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return value


# ======== FEATURES ========


class LobFeaturesQueryParams(QueryParams):
    """Query parameters for the lob features endpoint."""

    __category__ = "lob"
    __output_columns__ = (
        "date",
        "mid_price",
        "spread",
        "weighted_mid",
        "queue_imbalance",
        "depth_imbalance",
        "ofi",
    )

    data: list[Data] = Field(description="Order book snapshots, in time order.")
    index: str = BookIndex
    levels: PositiveInt | None = Field(
        default=None,
        description="Levels used for depth imbalance and OFI; all available if not set.",
    )
    decay: float | None = Field(
        default=None,
        ge=0.0,
        description="Exponential level-weight decay for depth imbalance,"
        " weights exp(-decay * (level - 1)). Equal weights if not set.",
    )


class LobFeaturesData(Data):
    """Order book features for one snapshot."""

    date: datetime | dateType | str = Field(description="Snapshot time.")
    mid_price: float = Field(description="(best bid + best ask) / 2.")
    spread: float = Field(description="Best ask - best bid.")
    weighted_mid: float = Field(
        description="Size-weighted mid: (P_bid * Q_ask + P_ask * Q_bid) / (Q_bid + Q_ask)."
    )
    queue_imbalance: float | None = Field(
        default=None,
        description="(Q_bid - Q_ask) / (Q_bid + Q_ask) at the best level, in [-1, 1].",
    )
    depth_imbalance: float | None = Field(
        default=None,
        description="Weighted bid vs ask size imbalance over the chosen levels, in [-1, 1].",
    )
    ofi: float | None = Field(
        default=None,
        description="Order flow imbalance since the previous snapshot, summed over"
        " the chosen levels (Cont, Kukanov & Stoikov, 2014). Empty on the first row.",
    )


@router.command(
    methods=["POST"],
    examples=[
        APIEx(parameters={"data": APIEx.mock_data("timeseries", 10, _BOOK_SAMPLE)})
    ],
)
def features(params: LobFeaturesQueryParams) -> OBBject[list[LobFeaturesData]]:
    """Compute per-snapshot order book features.

    Returns mid price, spread, size-weighted mid, best-level queue imbalance,
    multi-level depth imbalance and order flow imbalance (OFI) for each
    snapshot. Positive imbalance and OFI mean buying pressure.
    """
    from math import isfinite

    from pandas import Series

    from openbb_quantitative import _lob_helpers as lob

    book = _book(params.data, params.index)
    n = params.levels or lob.levels(book)
    flows = lob.order_flow_imbalance(book, levels=n)
    ofi = flows if isinstance(flows, Series) else flows.sum(axis=1, min_count=1)
    frame = {
        "mid_price": lob.mid_price(book),
        "spread": lob.spread(book),
        "weighted_mid": lob.weighted_mid(book),
        "queue_imbalance": lob.queue_imbalance(book),
        "depth_imbalance": lob.depth_imbalance(book, levels=n, decay=params.decay),
        "ofi": ofi,
    }

    def _num(value) -> float | None:
        value = float(value)
        return value if isfinite(value) else None

    out = [
        LobFeaturesData(
            date=ts,
            mid_price=float(frame["mid_price"].iloc[i]),
            spread=float(frame["spread"].iloc[i]),
            weighted_mid=float(frame["weighted_mid"].iloc[i]),
            queue_imbalance=_num(frame["queue_imbalance"].iloc[i]),
            depth_imbalance=_num(frame["depth_imbalance"].iloc[i]),
            ofi=_num(frame["ofi"].iloc[i]),
        )
        for i, ts in enumerate(book.index)
    ]
    return OBBject(results=out)


# ======== ORDER FLOW IMBALANCE ========


class LobOfiQueryParams(QueryParams):
    """Query parameters for the lob ofi endpoint."""

    __category__ = "lob"
    __output_columns__ = ("date", "level", "ofi", "integrated_ofi")

    data: list[Data] = Field(description="Order book snapshots, in time order.")
    index: str = BookIndex
    freq: str = Field(
        default="1min",
        description="Aggregation window: a pandas offset alias such as '1s' or"
        " '1min' (needs timestamps), or a whole number of snapshots such as '100'.",
    )
    levels: PositiveInt | None = Field(
        default=None, description="Levels to include; all available if not set."
    )


class LobOfiData(Data):
    """Order flow imbalance for one window and level."""

    date: datetime | dateType | str | int = Field(
        description="Window start (or window number for snapshot-count windows)."
    )
    level: int = Field(description="Book level, 1 = best quote.")
    ofi: float = Field(description="OFI summed over the window at this level.")
    integrated_ofi: float | None = Field(
        default=None,
        description="Integrated OFI for the window (Cont, Cucuringu & Zhang, 2023):"
        " depth-scaled multi-level OFI projected on its first principal component."
        " Repeated on every level row of a window.",
    )


@router.command(
    methods=["POST"],
    examples=[
        APIEx(
            parameters={
                "freq": "2",
                "data": APIEx.mock_data("timeseries", 10, _BOOK_SAMPLE),
            }
        )
    ],
)
def ofi(params: LobOfiQueryParams) -> OBBject[list[LobOfiData]]:
    """Aggregate order flow imbalance (OFI) per window and level.

    OFI counts net buying pressure at a price level: bid size added or bid
    price improved, ask size cancelled or ask price lifted. Summed over a
    window it explains much of that window's mid-price change. With two or
    more levels and windows, also returns integrated OFI.
    """
    from pandas import DataFrame

    from openbb_quantitative import _lob_helpers as lob

    book = _book(params.data, params.index)
    n = params.levels or lob.levels(book)
    freq = _freq(params.freq)
    flows = lob.order_flow_imbalance(book, levels=n)
    if not isinstance(flows, DataFrame):
        flows = flows.to_frame("ofi_1")
    windows = lob._windows(flows, freq).sum(min_count=1).dropna(how="all")
    integrated = None
    if n > 1 and len(windows) > 1:
        integrated = lob.integrated_ofi(book, freq=freq, levels=n).ofi
    out = []
    for ts, row in windows.iterrows():
        io = None
        if integrated is not None and ts in integrated.index:
            io = float(integrated.loc[ts])
        for level in range(1, n + 1):
            value = row[f"ofi_{level}"]
            if value == value:  # skip NaN  # noqa: PLR0124
                out.append(
                    LobOfiData(
                        date=ts, level=level, ofi=float(value), integrated_ofi=io
                    )
                )
    return OBBject(results=out)


# ======== PRICE IMPACT ========


class LobPriceImpactQueryParams(QueryParams):
    """Query parameters for the lob price_impact endpoint."""

    __category__ = "lob"
    __output_columns__ = ("beta", "alpha", "beta_tstat", "alpha_tstat", "r2", "nobs")

    data: list[Data] = Field(description="Order book snapshots, in time order.")
    index: str = BookIndex
    freq: str = Field(
        default="1min",
        description="Aggregation window: a pandas offset alias such as '10s'"
        " (needs timestamps), or a whole number of snapshots such as '100'.",
    )
    levels: PositiveInt = Field(
        default=1, description="Levels whose OFI is summed; 1 = best level only."
    )
    normalize_depth: bool = Field(
        default=False,
        description="Divide each window's OFI by its average depth, making beta"
        " comparable across instruments and days.",
    )


class LobPriceImpactData(Data):
    """Linear price-impact regression of mid changes on OFI."""

    beta: float = Field(description="Mid-price change per unit of OFI.")
    alpha: float = Field(description="Regression intercept.")
    beta_tstat: float = Field(description="t-statistic of beta.")
    alpha_tstat: float = Field(description="t-statistic of alpha.")
    r2: float = Field(description="R-squared of the regression.")
    nobs: int = Field(description="Number of windows in the regression.")


@router.command(
    methods=["POST"],
    examples=[
        APIEx(
            parameters={
                "freq": "2",
                "data": APIEx.mock_data("timeseries", 12, _BOOK_SAMPLE),
            }
        )
    ],
)
def price_impact(
    params: LobPriceImpactQueryParams,
) -> OBBject[LobPriceImpactData]:
    """Estimate linear price impact: dMid = alpha + beta * OFI per window.

    Cont, Kukanov & Stoikov (2014) find mid-price changes over short windows
    are close to linear in OFI, with a slope inversely proportional to depth.
    """
    from openbb_quantitative import _lob_helpers as lob

    book = _book(params.data, params.index)
    res = lob.price_impact(
        book,
        freq=_freq(params.freq),
        levels=params.levels,
        normalize_depth=params.normalize_depth,
    )
    return OBBject(
        results=LobPriceImpactData(
            beta=res.beta,
            alpha=res.alpha,
            beta_tstat=res.beta_tstat,
            alpha_tstat=res.alpha_tstat,
            r2=res.r2,
            nobs=res.nobs,
        )
    )


# ======== IMBALANCE SIGNAL ========


class LobImbalanceSignalQueryParams(QueryParams):
    """Query parameters for the lob imbalance_signal endpoint."""

    __category__ = "lob"
    __output_columns__ = (
        "imbalance_low",
        "imbalance_high",
        "count",
        "p_up",
        "p_down",
        "p_flat",
        "mean_move",
    )

    data: list[Data] = Field(description="Order book snapshots, in time order.")
    index: str = BookIndex
    signal: Literal["queue", "depth"] = Field(
        default="queue",
        description="Imbalance to bucket: 'queue' (best level) or 'depth' (all levels).",
    )
    bins: PositiveInt = Field(
        default=10, description="Number of equal-width imbalance buckets on [-1, 1]."
    )
    horizon: PositiveInt = Field(
        default=1,
        description="Steps ahead. With skip_unchanged, a step is a mid-price change.",
    )
    skip_unchanged: bool = Field(
        default=True,
        description="Look ahead to the next different mid rather than a fixed"
        " number of snapshots.",
    )


class LobImbalanceSignalData(Data):
    """Next mid-price move statistics for one imbalance bucket."""

    imbalance_low: float = Field(description="Lower bucket edge.")
    imbalance_high: float = Field(description="Upper bucket edge.")
    count: int = Field(description="Snapshots in the bucket.")
    p_up: float | None = Field(default=None, description="Share where the mid rose.")
    p_down: float | None = Field(default=None, description="Share where the mid fell.")
    p_flat: float | None = Field(
        default=None, description="Share where the mid was unchanged."
    )
    mean_move: float | None = Field(
        default=None, description="Average mid change, in price units."
    )


@router.command(
    methods=["POST"],
    examples=[
        APIEx(
            parameters={
                "bins": 4,
                "data": APIEx.mock_data("timeseries", 20, _BOOK_SAMPLE),
            }
        )
    ],
)
def imbalance_signal(
    params: LobImbalanceSignalQueryParams,
) -> OBBject[list[LobImbalanceSignalData]]:
    """Tabulate the next mid-price move by order book imbalance bucket.

    Shows how strongly imbalance predicts the direction of the next price
    change (Gould & Bonart, 2016): with a predictive imbalance, P(up) rises
    from near 0 in the most negative bucket to near 1 in the most positive.
    """
    from math import isfinite

    from openbb_quantitative import _lob_helpers as lob

    book = _book(params.data, params.index)
    imb = (
        lob.queue_imbalance(book)
        if params.signal == "queue"
        else lob.depth_imbalance(book)
    )
    table = lob.imbalance_signal(
        book,
        horizon=params.horizon,
        bins=params.bins,
        imbalance=imb,
        skip_unchanged=params.skip_unchanged,
    )

    def _num(value) -> float | None:
        value = float(value)
        return value if isfinite(value) else None

    edges = [-1.0 + 2.0 * i / params.bins for i in range(params.bins + 1)]
    out = [
        LobImbalanceSignalData(
            imbalance_low=edges[i],
            imbalance_high=edges[i + 1],
            count=int(row["count"]),
            p_up=_num(row["p_up"]),
            p_down=_num(row["p_down"]),
            p_flat=_num(row["p_flat"]),
            mean_move=_num(row["mean_move"]),
        )
        for i, (_, row) in enumerate(table.iterrows())
    ]
    return OBBject(results=out)


# ======== MICRO-PRICE ========


class LobMicropriceQueryParams(QueryParams):
    """Query parameters for the lob microprice endpoint."""

    __category__ = "lob"
    __output_columns__ = (
        "imbalance_low",
        "imbalance_high",
        "spread_ticks",
        "adjustment",
        "adjustment_ticks",
        "next_move_ticks",
    )

    data: list[Data] = Field(description="Order book snapshots, in time order.")
    index: str = BookIndex
    tick_size: float | None = Field(
        default=None,
        gt=0.0,
        description="Price tick. Inferred as the smallest positive spread if not set.",
    )
    n_imbalance: int = Field(
        default=10, ge=2, description="Number of imbalance buckets on [0, 1]."
    )
    n_spread: PositiveInt = Field(
        default=2, description="Largest spread, in ticks, kept as a model state."
    )
    max_iter: PositiveInt = Field(
        default=6, description="Terms of the micro-price series expansion."
    )


class LobMicropriceData(Data):
    """Micro-price adjustment for one (imbalance, spread) state."""

    imbalance_low: float = Field(
        description="Lower edge of Q_bid / (Q_bid + Q_ask) for the bucket."
    )
    imbalance_high: float = Field(description="Upper edge of the bucket.")
    spread_ticks: int = Field(description="Spread, in ticks.")
    adjustment: float | None = Field(
        default=None,
        description="Micro-price minus mid, in price units. Empty if the state"
        " never occurred in the data.",
    )
    adjustment_ticks: float | None = Field(
        default=None, description="The adjustment in ticks."
    )
    next_move_ticks: float | None = Field(
        default=None,
        description="First-order term g1: expected mid change at the next price"
        " move only, in ticks. A better forecast of the next change when moves"
        " are autocorrelated.",
    )


@router.command(
    methods=["POST"],
    examples=[
        APIEx(
            parameters={
                "n_imbalance": 4,
                "data": _MICRO_SAMPLE,
            }
        )
    ],
)
def microprice(params: LobMicropriceQueryParams) -> OBBject[list[LobMicropriceData]]:
    """Fit Stoikov's (2018) micro-price and return its adjustment table.

    The micro-price is the expected mid-price far in the future given today's
    best-level imbalance and spread, estimated from a Markov chain fitted on
    the snapshots. The micro-price of a snapshot is its mid plus the
    adjustment for its (imbalance bucket, spread) state. Use snapshots sampled
    finely enough that the mid moves at most one tick between them.
    """
    from math import isfinite

    from openbb_quantitative import _lob_helpers as lob

    book = _book(params.data, params.index)
    model = lob.fit_microprice(
        book,
        tick_size=params.tick_size,
        n_imbalance=params.n_imbalance,
        n_spread=params.n_spread,
        max_iter=params.max_iter,
    )
    edges = model.bin_edges
    out = []
    for spread_ticks in model.adjustment.columns:
        col = model.adjustment[spread_ticks].to_numpy()
        first = model.first_order[spread_ticks].to_numpy()
        for i, value in enumerate(col):
            ok = isfinite(value)
            g1 = first[i]
            out.append(
                LobMicropriceData(
                    imbalance_low=float(edges[i]),
                    imbalance_high=float(edges[i + 1]),
                    spread_ticks=int(spread_ticks),
                    adjustment=float(value) if ok else None,
                    adjustment_ticks=float(value / model.tick_size) if ok else None,
                    next_move_ticks=(
                        float(g1 / model.tick_size) if isfinite(g1) else None
                    ),
                )
            )
    return OBBject(results=out)
