# Avellaneda-Stoikov Market Making Simulator

A NumPy implementation of the optimal market making model from Avellaneda and Stoikov, *High-frequency trading in a limit order book* (Quantitative Finance, 2008).

A market maker quotes a bid and an ask around the mid price, earns the spread when market orders hit its quotes, and carries inventory risk in between. The model gives closed-form quotes that trade off those two effects.

## The model

| Quantity | Formula |
| --- | --- |
| Mid price | `dS = sigma dW` (arithmetic Brownian motion) |
| Reservation price | `r = s - q * gamma * sigma^2 * (T - t)` |
| Total spread | `delta_a + delta_b = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)` |
| Quotes | `bid = r - spread / 2`, `ask = r + spread / 2` |
| Fill intensity | `lambda(delta) = A * exp(-k * delta)` |

`q` is inventory, `gamma` is risk aversion, `k` measures order book liquidity, and `A` is the order arrival rate at the mid. A long position lowers both quotes so the market maker is more likely to sell; a short position raises them.

The **symmetric** benchmark posts the same spread centred on the mid price, ignoring inventory. `compare()` runs both strategies on identical price paths and order flow (common random numbers), so differences come from the quoting rule alone.

## Usage

```python
from simulator import ASParams, compare, simulate, summary_table

params = ASParams(gamma=0.1, sigma=2.0, k=1.5, A=140.0, T=1.0, dt=0.005)

results = compare(params, n_paths=1000, seed=1)
print(summary_table(results).round(2))

one_path = simulate(params, seed=7).to_frame()  # mid, reservation, bid, ask, inventory, cash, pnl
```

`ASParams` also takes `q0` (starting inventory), `q_max` (an inventory limit at which the market maker stops quoting the side that would grow the position) and `fill_model`:

- `"poisson"` (default): a quote fills within a step with probability `1 - exp(-lambda * dt)`.
- `"linear"`: probability `min(lambda * dt, 1)`, as in the paper.

### Calibrating sigma from OpenBB data

`sigma` is in price units, so it is the standard deviation of price changes rather than returns.

```python
from openbb import obb
from simulator import ASParams, compare, estimate_sigma

px = obb.equity.price.historical("SPY", interval="1m", provider="yfinance").to_df()["close"]
sigma_per_day = estimate_sigma(px, periods_per_unit=390)  # 390 one-minute bars per session

params = ASParams(s0=float(px.iloc[-1]), sigma=sigma_per_day, T=1.0, dt=1 / 390)
```

`A` and `k` describe order flow and have to be fitted from trade and quote data, for example by regressing the log arrival rate of market orders on their distance from the mid.

## Results against the paper

1,000 paths, paper parameters, `fill_model="linear"`, `seed=1`:

| gamma | strategy | mean P&L | std P&L | std final q | paper mean P&L | paper std P&L |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1 | inventory | 65.1 | 6.5 | 2.8 | 62.9 | 5.9 |
| 0.1 | symmetric | 67.9 | 13.8 | 8.9 | 67.2 | 13.4 |
| 0.01 | inventory | 68.3 | 9.2 | 5.6 | 66.8 | 8.8 |
| 0.01 | symmetric | 68.4 | 14.2 | 9.2 | 67.4 | 13.4 |
| 0.5 | inventory | 48.8 | 6.0 | 1.9 | 33.9 | 4.7 |
| 0.5 | symmetric | 58.5 | 12.0 | 7.3 | 51.0 | 11.8 |

The rows for gamma = 0.1 and 0.01 agree closely. For gamma = 0.5 this simulator earns more than the paper reports. The spread column of the paper's tables equals `(2 / gamma) * ln(1 + gamma / k)` alone, which suggests the authors' simulation differed from the full formula in some way; this implementation uses the full spread from the paper's equations.

The paper's conclusion holds throughout: the inventory strategy gives up a little mean P&L in exchange for roughly half the P&L standard deviation and a much tighter inventory.

## Tests

```bash
pytest examples/avellaneda_stoikov/tests
```
