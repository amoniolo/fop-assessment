# Betterment Robo-Advisory Prototype

**Fundamentals of Programming (5K7V0027) — Level 7 portfolio, Part 2**

A menu-driven Python package that models the automated advisory pipeline
operated by [Betterment](https://www.betterment.com), a US robo-adviser.

---

## Purpose

Betterment automates a process that a human financial adviser used to perform
by hand: it profiles a client, allocates their money to a model portfolio,
projects that portfolio against a stated goal, and reports afterwards on what
happened. This prototype implements that pipeline end to end.

The four menu functions are the four stages of it:

| # | Function | Stage of the advisory pipeline |
|---|----------|-------------------------------|
| 1 | Risk profile and strategic allocation | Onboarding: questionnaire → risk tier → model portfolio |
| 2 | Optimise portfolio and allocate to shares | Execution: efficient frontier → weights → whole-share order |
| 3 | Project a goal with a de-risking glide path | Planning: glide path + Monte Carlo → probability of success |
| 4 | Performance and risk report vs benchmark | Reporting: portfolio vs benchmark, return and risk |

Each stage can consume the previous stage's output through a shared session
object, so the four read as one system. Each also runs standalone, so the menu
can be entered at any point.

---

## Install and run

Requires **Python 3.11 or later** (developed and tested on 3.13.9).

```bash
cd betterment_prototype

python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
python main.py
```

Type `q` at any prompt to abandon a function and return to the menu.
Select `0` from the menu to exit.

The virtual environment above is the reproducible path and is what
`requirements.txt` pins. The package also runs unmodified on an Anaconda base
environment with the same four finance libraries installed by `pip`; that
configuration was tested and produces identical figures, on the older
conda-managed stack (pandas 2.3.3, numpy 2.3.5) as well as the newer pinned one.

### Running the tests

```bash
python tests/test_prototype.py     # no pytest required
python -m pytest tests -q          # if pytest is installed
```

46 tests, all passing. They cover the scoring and suitability rules, the glide
path and simulation, the risk metrics, and the offline data fallback. They run
with the network disabled.

---

## No network? It still runs.

`data/price_snapshot.csv` holds ten years of real daily adjusted-close prices
for all ten model sleeves plus the benchmark, captured from Yahoo Finance.

Every retrieval tries the live source first and falls back to that snapshot on
any failure, and **every function prints which source produced its figures** —
either `live data from Yahoo Finance` or `cached snapshot dated YYYY-MM-DD`.
So the program never fails because of a network problem, and the reader always
knows how current the numbers are.

To refresh the snapshot (needs a connection):

```bash
python tools/build_snapshot.py
```

### A note on TLS-inspecting networks

On many university and corporate networks, HTTPS is intercepted and re-signed
by a proxy. `yfinance` retrieves through `curl_cffi`, which carries its own
certificate bundle and ignores the operating system's, so on such a network
every live retrieval fails certificate verification even though the same
machine's browser works. `utils/data.py` handles this by exporting the Windows
certificate store to a PEM bundle and pointing curl at it. **This does not
disable verification** — certificates are still checked, against exactly the
authorities the operating system already trusts. If it does not work, the
snapshot fallback takes over and the program runs anyway.

---

## The four functions

### 1 — Risk profiling and strategic asset allocation

Five weighted questions produce a score from 1.00 to 5.00, which maps to a risk
tier from 1 (Conservative, 20% equity) to 5 (Aggressive Growth, 90% equity).
The tier maps to a model portfolio of ten ETF sleeves.

A **suitability rule** then applies: a short horizon caps equity regardless of
stated risk tolerance. A client who answers at maximum risk tolerance but needs
the money in two years is capped at 30% equity, because the cost of being wrong
is borne on a fixed date.

The function reports **which input actually bound the outcome** — the horizon
cap when it binds, otherwise the highest-weighted contribution to the score.
That is the input the client would have to change to get a different portfolio.

### 2 — Portfolio optimisation and discrete allocation

Mean-variance optimisation over the chosen universe, then conversion into whole
shares with the residual cash stated.

Two design decisions are made deliberately and are visible in the output:

- **Ledoit-Wolf shrinkage** on the covariance matrix, not the sample
  covariance. Mean-variance optimisation is unstable on a raw sample
  covariance, and worst exactly where this prototype operates — a small
  universe over a short lookback.
- **Per-asset weight bounds**, defaulting to 35%. Without them the optimiser
  concentrates into whichever two sleeves the lookback window happened to
  flatter.

The function reports how hard those bounds bound — how many holdings sit on the
ceiling, and how many assets were dropped — rather than presenting a tidy set of
weights as though it were an unconstrained finding.

### 3 — Goal projection with a de-risking glide path

Builds a glide path that walks equity down as the target date approaches,
derives per-period return and volatility from it, and runs a seeded Monte Carlo
over 10,000 paths. `numpy-financial` provides the deterministic anchor and
solves for the required contribution; a bisection search over the simulation
finds the contribution that lifts the success probability above 80%.

The output is the **gap between the deterministic and the stochastic answer**.
The deterministic calculation says a contribution reaches the target; the
simulation says it reaches the target some percentage of the time. Both are
correct and they answer different questions, and the second is the one a client
needs.

Volatility is combined through the two-asset variance formula rather than by
weighting the two volatilities linearly — that difference *is* the
diversification benefit, and a linear approximation would overstate the risk of
the balanced middle of the path.

### 4 — Performance and risk report against a benchmark

Builds the weighted daily return series, then compares it with a benchmark over
the same aligned window: CAGR, annualised volatility, Sharpe, Sortino, maximum
drawdown with its start and recovery dates, 95% historical value-at-risk, and
tracking error.

The interpretation **attributes the difference to return or to risk** rather
than reporting both and leaving the reader to do it.

---

## Libraries

The brief requires at least two Python finance libraries. Four are used:

| Library | Import | Used for | Functions |
|---------|--------|----------|-----------|
| [yfinance](https://pypi.org/project/yfinance/) | `yfinance` | Price history for the sleeves and benchmark. No API key. | F1, F2, F4 |
| [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/) | `pypfopt` | Expected returns, shrinkage covariance, efficient frontier, discrete allocation | F2 |
| [numpy-financial](https://numpy.org/numpy-financial/) | `numpy_financial` | Time value of money: `fv`, `pmt` | F3 |
| [QuantStats](https://pypi.org/project/quantstats/) | `quantstats` | Performance and risk analytics vs a benchmark | F4 |

pandas, NumPy, SciPy and Matplotlib are supporting infrastructure and are **not**
offered as satisfying that requirement.

**On QuantStats robustness.** Every metric in F4 is attempted through QuantStats
first and falls back to an explicit pandas/NumPy implementation only if the
library call raises. The output states which engine produced the numbers —
`QuantStats`, or `internal fallback (QuantStats unavailable or incompatible)`.
QuantStats is the primary path and is what the pinned set uses; the fallback
exists so that a version problem on another machine costs a footnote rather than
20% of the prototype.

The risk this guards against is a real one, and it materialised during
development. QuantStats 0.0.64 — the version originally planned — declares
`numpy<2.0.0` in its metadata, while numpy below 2.1 publishes no wheel for
Python 3.13. The intersection is empty, so `pip` fails with
`ResolutionImpossible` before installing anything. Testing showed the pin is
conservative rather than real: 0.0.64's code runs correctly under numpy 2 when
the pin is bypassed, because its `_numpy_compat` shim already routes around the
one removed symbol it used. But a dependency resolver acts on *declared*
metadata, not on observed behaviour, so a stale pin blocks an install exactly as
firmly as a genuine incompatibility. Release 0.0.81 drops the ceiling.

---

## Package layout

```
betterment_prototype/
    main.py                    Banner, menu loop, dispatch table, error handling
    config.py                  Sleeves, risk tiers, questionnaire, all parameters
    functions/
        f1_risk_profile.py     Questionnaire -> risk tier -> strategic allocation
        f2_optimise.py         Efficient frontier -> weights -> whole shares
        f3_goal_projection.py  Glide path + Monte Carlo -> probability of success
        f4_performance.py      Portfolio vs benchmark, return and risk
    utils/
        validation.py          Bounded, re-prompting input helpers
        data.py                Cache-first price retrieval with offline fallback
        display.py             Headers, aligned tables, currency and percent
        session.py             State passed between functions
    data/                      Bundled price snapshot for offline running
    output/                    Generated charts
    tests/                     44 tests
    tools/build_snapshot.py    Rebuilds the offline snapshot (development only)
    requirements.txt
    README.md
```

### Structural notes

- **The menu is a dictionary of `MenuEntry` records**, not a chain of
  `if`/`elif`. The same dictionary renders the menu, validates the input and
  dispatches, so the displayed options and the callable options cannot drift
  apart.
- **Validation is centralised** in `utils/validation.py`. Every helper loops
  until the input is valid and explains what was wrong. There is no bare
  `int(input(...))` anywhere in the codebase. Ticker validation confirms that
  data actually comes back, not merely that the string looks plausible.
- **Every dispatch is wrapped** in a handler that catches the expected
  exception types, prints one readable sentence, and returns to the menu. No
  traceback reaches the user.
- **Records are frozen dataclasses**; only the `Session` itself is mutable,
  because session state changes by design as the client moves down the pipeline.

---

## Charts

Written to `output/` when the relevant function runs:

| File | Produced by | Content |
|------|-------------|---------|
| `f2_efficient_frontier.png` | F2 | Frontier, individual sleeves, selected portfolio |
| `f3_goal_funnel.png` | F3 | Percentile funnel of simulated balances against the target |
| `f4_drawdown.png` | F4 | Cumulative growth and underwater drawdown vs benchmark |

---

## Reproducibility and data caveats

- **The random seed is fixed** at `20260905` in `config.py` and is printed in
  the F3 output, so the simulated figures reproduce exactly.
- **Prototype outputs are illustrative.** They are based on historical data over
  a lookback you choose, and they are not advice and not a forecast.
- **The capital-market assumptions in `config.py` are the author's own
  illustrative estimates** — a 7.0% nominal equity return, 3.5% for bonds, with
  16% and 5.5% volatility. They are not Betterment's published assumptions.
  They are labelled as illustrative in the code, in the F3 output and here.
- **The sleeve list and tier ladder are modelled on Betterment's publicly
  disclosed Core portfolio construction**, not copied from a proprietary
  document.
- **Prices are real market data** from Yahoo Finance. Nothing in the price path
  is synthetic.
