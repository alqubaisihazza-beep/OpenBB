# Limit Order Book and Imbalance Modeling

A limit order book simulator and the standard order book imbalance measures, in NumPy.

- **Simulator:** the stochastic order book model of Cont, Stoikov and Talreja, *A stochastic model for order book dynamics* (Operations Research, 2010).
- **Imbalance measures:** queue imbalance, multi-level depth imbalance, the size-weighted mid, and order flow imbalance (OFI) from Cont, Kukanov and Stoikov, *The price impact of order book events* (Journal of Financial Econometrics, 2014).

The imbalance functions take plain arrays of bid and ask prices and sizes, so they work the same on simulated books and on real top-of-book data.

## The order book model

Prices sit on an integer tick grid and every order is one unit. Within `K` ticks of the opposite best quote:

| Event | Rate |
| --- | --- |
| Limit buy at `best_ask - i`, limit sell at `best_bid + i` | `lam[i-1]` per side |
| Market buy (hits best ask), market sell (hits best bid) | `mu` per side |
| Cancellation of one resting order at distance `i` | `theta[i-1]` per order |

The defaults are the rates the paper estimated from Tokyo Stock Exchange data (`K = 5`, rates per second). Events are simulated exactly, one at a time, with exponential waiting times.

## Imbalance measures

| Function | Definition |
| --- | --- |
| `queue_imbalance` | `I = (Vb - Va) / (Vb + Va)` at the best quotes, in [-1, 1] |
| `depth_imbalance` | the same over several levels, weighted by `exp(-decay * level)` |
| `weighted_mid` | `(ask * Vb + bid * Va) / (Vb + Va)`, a simple micro-price |
| `order_flow_imbalance` | per-update OFI `e_n`: net size added to the bid minus net size added to the ask, accounting for quote moves |
| `aggregate_ofi`, `ofi_regression` | OFI and mid change per time window, and the fit `dP = alpha + beta * OFI` |
| `imbalance_signal_table`, `imbalance_accuracy` | how well imbalance predicts the direction of the next mid-price move |

## Usage

```python
from lob import (
    LOBParams, simulate_lob, queue_imbalance, imbalance_signal_table,
    order_flow_imbalance, aggregate_ofi, ofi_regression,
)

book = simulate_lob(LOBParams(), n_events=200_000, seed=1)
book.to_frame().head()  # bid, ask, sizes, mid, imbalance per event

imb = queue_imbalance(book.bid_size, book.ask_size)
print(imbalance_signal_table(imb, book.mid, n_buckets=5))

ofi = order_flow_imbalance(book.bid, book.bid_size, book.ask, book.ask_size)
_, ofi_10s, dmid_10s = aggregate_ofi(book.time, ofi, book.mid, interval=10.0)
print(ofi_regression(ofi_10s, dmid_10s))
```

To use real data, pass arrays of best bid, best ask and their sizes from any quote feed in place of the simulated ones.

## Results on a simulated book

200,000 events (about 2.3 hours of simulated trading), default parameters, `seed=1`. Mid moves are in ticks.

**Queue imbalance predicts the next mid move.** The thinner queue is the one more likely to empty first:

| Imbalance bucket | Observations | P(next move up) | Mean next move |
| --- | --- | --- | --- |
| -1.0 to -0.6 | 3,926 | 0.20 | -0.38 |
| -0.6 to -0.2 | 64,826 | 0.33 | -0.21 |
| -0.2 to +0.2 | 75,110 | 0.51 | +0.01 |
| +0.2 to +0.6 | 52,457 | 0.67 | +0.21 |
| +0.6 to +1.0 | 3,662 | 0.79 | +0.35 |

The sign of best-level imbalance matches the direction of the next move 66% of the time. Adding deeper levels makes the prediction worse in this model (55% with all five levels weighted equally, 61% with `decay=2`), because only the best queues decide which side empties next.

**OFI explains price changes linearly:**

| Window | beta (ticks per unit OFI) | R² |
| --- | --- | --- |
| 1 s | 0.24 | 0.44 |
| 10 s | 0.29 | 0.69 |
| 60 s | 0.29 | 0.72 |

Cont, Kukanov and Stoikov predict `beta ≈ 1 / (2 * depth)`. The average best-queue size here is 2.14, which gives 0.23, in line with the 1-second estimate.

## Tests

```bash
pytest examples/order_book_imbalance/tests
```
