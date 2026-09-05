"""Function 2: portfolio optimisation and discrete allocation.

Business process modelled
-------------------------
Converting a strategic allocation into an executable portfolio -- the step
between advice and trading. F1 says what the client should hold in percentage
terms; this function says how many shares of what to buy, and what cash is
left over because shares are not divisible.

Two design decisions are defended in the code below and are worth carrying into
the report. First, covariance is estimated by Ledoit-Wolf shrinkage rather than
the sample covariance matrix, because mean-variance optimisation is notoriously
unstable on a raw sample covariance and the instability is worst exactly where
this artefact operates: a small universe over a short lookback. Second,
per-asset weight bounds are imposed. Both choices constrain the optimiser, and
the function reports how hard they bound rather than hiding it -- an optimiser
that silently produces a two-asset corner solution is the failure mode a reader
should be shown, not protected from.
"""

from __future__ import annotations

from datetime import date, timedelta

import matplotlib

# Agg is selected before pyplot is imported: the artefact writes charts to
# output/ and never opens a window, and on a machine with no display backend
# the default would fail at import time.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - must follow matplotlib.use
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from utils import display  # noqa: E402
from utils.data import get_prices, latest_prices  # noqa: E402
from utils.session import OptimisationResult, Session  # noqa: E402
from utils.validation import (  # noqa: E402
    prompt_choice,
    prompt_float,
    prompt_int,
    prompt_tickers,
    prompt_yes_no,
)


class OptimisationError(Exception):
    """The optimiser could not produce a usable portfolio.

    Distinct from PriceDataError so that the menu loop can tell the user
    whether the problem was the data or the optimisation, which are fixed in
    completely different ways.
    """


def run(session: Session) -> None:
    """Optimise a universe, allocate to whole shares, and explain the result."""
    display.header("Function 2  -  Portfolio Optimisation and Share Allocation")
    display.paragraph(
        "Mean-variance optimisation over a chosen universe, with Ledoit-Wolf "
        "shrinkage on the covariance matrix and per-asset weight bounds. The "
        "resulting weights are then converted into whole-share holdings."
    )

    tickers, from_f1 = _choose_universe(session)
    lookback_years = prompt_int(
        "Lookback period in years", minimum=1, maximum=20,
        default=config.DEFAULT_LOOKBACK_YEARS,
    )
    objective = prompt_choice(
        "Optimisation objective",
        {"1": "Maximum Sharpe ratio", "2": "Minimum volatility"},
    )
    objective_name = ("maximum Sharpe" if objective == "1" else "minimum volatility")

    upper_bound = prompt_float(
        "Maximum weight in any single holding (0.10 to 1.00)",
        minimum=0.10, maximum=1.00,
        default=config.DEFAULT_WEIGHT_BOUNDS[1],
    )
    bounds = (config.DEFAULT_WEIGHT_BOUNDS[0], upper_bound)

    amount = prompt_float(
        "Amount to invest", minimum=1_000.0, maximum=100_000_000.0,
        default=session.suggested_amount() or 50_000.0,
    )

    end = date.today()
    start = end - timedelta(days=int(lookback_years * 365.25))
    prices, source = get_prices(tickers, start, end)
    session.data_source_note = source

    weights, performance, bound_report = optimise(prices, objective, bounds)
    share_counts, leftover = discrete_allocation(weights, prices, amount)

    _print_weights(weights, amount, bound_report)
    _print_performance(performance)
    _print_shares(share_counts, leftover, amount, prices)

    chart_path = plot_efficient_frontier(prices, weights, performance, bounds)
    if chart_path is not None:
        display.note(f"\n   Efficient-frontier chart saved to output/{chart_path.name}")

    display.insight(
        explain_optimisation(weights, performance, bound_report, bounds,
                             lookback_years, objective_name, leftover,
                             amount, session, from_f1)
    )

    session.optimisation = OptimisationResult(
        weights=weights,
        expected_return=performance[0],
        volatility=performance[1],
        sharpe=performance[2],
        share_counts=share_counts,
        leftover_cash=leftover,
        amount=amount,
        objective=objective_name,
        lookback_years=lookback_years,
        weight_bounds=bounds,
    )
    session.record_run("F2 Optimisation")
    display.note(
        "\n   Saved to the session. Function 4 will offer to report these "
        "weights against a benchmark."
    )


def _choose_universe(session: Session) -> tuple[list[str], bool]:
    """Return the asset universe, defaulting to F1's sleeve when available.

    Returns:
        (tickers, came_from_f1). The flag matters because the insight line
        compares the optimised result to F1's strategic weights only when
        there is an F1 result to compare against.
    """
    if session.risk_profile is not None:
        profile = session.risk_profile
        display.note(
            f"\n   Function 1 assigned tier {profile.tier_level} "
            f"({profile.tier_label}) over {len(profile.weights)} sleeves."
        )
        if prompt_yes_no("Optimise that sleeve?", default=True):
            return list(profile.weights), True

    display.note(
        "\n   Enter a universe, or press Enter for the default model sleeve."
    )
    tickers = prompt_tickers(
        "Tickers, comma separated", minimum_count=2,
        default=list(config.SLEEVES),
    )
    return tickers, False


def optimise(
    prices: pd.DataFrame,
    objective: str,
    bounds: tuple[float, float],
) -> tuple[dict[str, float], tuple[float, float, float], dict[str, object]]:
    """Run the efficient-frontier optimisation.

    Args:
        prices: Adjusted-close prices, one column per asset.
        objective: "1" for maximum Sharpe, "2" for minimum volatility.
        bounds: (lower, upper) weight bound applied to every asset.

    Returns:
        (weights, (expected_return, volatility, sharpe), bound_report) where
        bound_report describes how many assets landed on a bound -- which is
        the diagnostic the insight line needs.

    Raises:
        OptimisationError: The solver failed or produced an unusable result.
    """
    from pypfopt import EfficientFrontier, expected_returns, risk_models

    if prices.shape[1] < 2:
        raise OptimisationError(
            "At least two assets with usable history are required; "
            f"only {prices.shape[1]} survived data cleaning."
        )

    mu = expected_returns.mean_historical_return(prices)

    # Ledoit-Wolf shrinkage rather than risk_models.sample_cov(). The sample
    # covariance of N assets estimated from a few hundred observations is
    # badly conditioned, and mean-variance optimisation amplifies exactly the
    # estimation error it contains -- it treats the noisiest correlation
    # estimate as a real diversification opportunity and levers into it.
    # Shrinkage pulls the estimate towards a structured target and produces
    # weights that are stable enough to be worth trading.
    covariance = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    # Bounds must be feasible: with N assets, an upper bound below 1/N makes
    # the weights unable to sum to one. Caught here with a readable message
    # rather than letting the solver return an opaque infeasibility.
    minimum_feasible = 1.0 / prices.shape[1]
    if bounds[1] < minimum_feasible:
        raise OptimisationError(
            f"An upper bound of {display.percent(bounds[1])} is infeasible for "
            f"{prices.shape[1]} assets: the weights could reach at most "
            f"{display.percent(bounds[1] * prices.shape[1])}. Use at least "
            f"{display.percent(minimum_feasible)}."
        )

    frontier = EfficientFrontier(mu, covariance, weight_bounds=bounds)

    try:
        if objective == "1":
            frontier.max_sharpe(risk_free_rate=config.RISK_FREE_RATE)
        else:
            frontier.min_volatility()
    except Exception as exc:  # noqa: BLE001 - solver failures are not enumerated
        raise OptimisationError(
            f"The optimiser did not converge ({type(exc).__name__}). Try a "
            "longer lookback, a looser weight bound, or fewer assets."
        ) from exc

    weights = frontier.clean_weights()
    performance = frontier.portfolio_performance(
        risk_free_rate=config.RISK_FREE_RATE
    )

    held = {t: w for t, w in weights.items() if w > 1e-4}
    if not held:
        raise OptimisationError("The optimiser produced an empty portfolio.")

    return held, performance, _bound_report(held, bounds, prices.shape[1])


def _bound_report(weights: dict[str, float], bounds: tuple[float, float],
                  universe_size: int) -> dict[str, object]:
    """Measure how hard the weight bounds constrained the solution.

    An optimiser that puts three assets at the ceiling and drops the rest has
    told you that, left alone, it would have concentrated further. That is the
    instability the bounds exist to contain, and reporting it is more useful
    than reporting a clean set of weights as though it were an unconstrained
    finding.
    """
    at_ceiling = [t for t, w in weights.items() if w >= bounds[1] - 1e-4]
    return {
        "at_ceiling": at_ceiling,
        "dropped": universe_size - len(weights),
        "universe_size": universe_size,
        "largest": max(weights.values()) if weights else 0.0,
        "top_two_share": sum(sorted(weights.values(), reverse=True)[:2]),
    }


def discrete_allocation(
    weights: dict[str, float],
    prices: pd.DataFrame,
    amount: float,
) -> tuple[dict[str, int], float]:
    """Convert continuous weights into whole-share holdings.

    Uses the greedy allocator rather than the integer-programming one: greedy
    needs no additional solver, which matters for the requirement that the
    package installs and runs from requirements.txt with no manual setup, and
    on a ten-asset portfolio the difference in residual cash is immaterial.

    Returns:
        (share counts per ticker, residual cash).
    """
    from pypfopt.discrete_allocation import DiscreteAllocation

    prices_now = latest_prices(prices[list(weights)])
    allocator = DiscreteAllocation(
        weights, prices_now, total_portfolio_value=amount
    )
    counts, leftover = allocator.greedy_portfolio()
    return {t: int(n) for t, n in counts.items() if n > 0}, float(leftover)


def _print_weights(weights: dict[str, float], amount: float,
                   bound_report: dict[str, object]) -> None:
    """Print the optimised weights, marking any that sit on the bound."""
    display.subheader("Optimised weights")
    at_ceiling = set(bound_report["at_ceiling"])
    rows = []
    for ticker, weight in sorted(weights.items(), key=lambda kv: -kv[1]):
        sleeve = config.SLEEVES.get(ticker)
        name = sleeve.name if sleeve else "-"
        flag = "at bound" if ticker in at_ceiling else ""
        rows.append([ticker, name, display.percent(weight),
                     display.money(weight * amount), flag])

    display.table(
        ["Ticker", "Sleeve", "Weight", "Amount", ""],
        rows,
        alignments=["l", "l", "r", "r", "l"],
    )


def _print_performance(performance: tuple[float, float, float]) -> None:
    """Print the frontier's expected return, volatility and Sharpe."""
    expected_return, volatility, sharpe = performance
    display.subheader("Expected performance (annualised)")
    display.key_values([
        ("Expected return", display.percent(expected_return, 2)),
        ("Volatility", display.percent(volatility, 2)),
        ("Sharpe ratio", display.ratio(sharpe)),
        ("Risk-free rate used", display.percent(config.RISK_FREE_RATE, 2)),
    ])
    display.note(
        "\n   These are the optimiser's own estimates, derived from the "
        "lookback window. They are not a forecast: the expected return is the "
        "historical mean of the window you chose."
    )


def _print_shares(share_counts: dict[str, int], leftover: float,
                  amount: float, prices: pd.DataFrame) -> None:
    """Print the executable whole-share order and the residual cash."""
    display.subheader("Executable allocation (whole shares)")
    prices_now = latest_prices(prices)
    rows = []
    invested = 0.0
    for ticker, count in sorted(share_counts.items(),
                                key=lambda kv: -kv[1] * float(prices_now[kv[0]])):
        price = float(prices_now[ticker])
        value = price * count
        invested += value
        rows.append([ticker, f"{count:,}", display.money(price),
                     display.money(value), display.percent(value / amount)])

    display.table(
        ["Ticker", "Shares", "Price", "Value", "% of total"], rows
    )
    display.key_values([
        ("Invested", display.money(invested)),
        ("Residual cash", display.money(leftover)),
        ("Cash drag", display.percent(leftover / amount, 2)),
    ])


def plot_efficient_frontier(prices: pd.DataFrame, weights: dict[str, float],
                            performance: tuple[float, float, float],
                            bounds: tuple[float, float]):
    """Plot the frontier with the chosen portfolio marked, and save it.

    Returns the saved path, or None if the chart could not be produced. A
    charting failure must not cost the user their numerical result, so this is
    the one place in F2 where a broad except is appropriate.
    """
    from pypfopt import EfficientFrontier, expected_returns, risk_models

    try:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        mu = expected_returns.mean_historical_return(prices)
        covariance = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

        # Trace the frontier by re-solving for a spread of target returns
        # between the minimum-volatility portfolio and the highest single
        # asset return the bounds can actually reach.
        base = EfficientFrontier(mu, covariance, weight_bounds=bounds)
        base.min_volatility()
        minimum_return = base.portfolio_performance()[0]
        maximum_return = float(mu.max()) * min(1.0, bounds[1] * len(mu))

        targets = np.linspace(minimum_return, maximum_return * 0.999, 40)
        frontier_points: list[tuple[float, float]] = []
        for target in targets:
            try:
                candidate = EfficientFrontier(mu, covariance, weight_bounds=bounds)
                candidate.efficient_return(target_return=target)
                point_return, point_vol, _ = candidate.portfolio_performance()
                frontier_points.append((point_vol, point_return))
            except Exception:  # noqa: BLE001 - infeasible targets are expected
                continue

        figure, axes = plt.subplots(figsize=(9, 6))

        if frontier_points:
            xs, ys = zip(*frontier_points)
            axes.plot(xs, ys, linewidth=2, label="Efficient frontier")

        # Individual assets, so the reader can see what diversification bought.
        annual_vol = prices.pct_change().std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)
        axes.scatter(annual_vol, mu, s=35, alpha=0.65, label="Individual sleeves")
        for ticker in mu.index:
            axes.annotate(ticker, (annual_vol[ticker], mu[ticker]),
                          fontsize=8, xytext=(4, 3), textcoords="offset points")

        axes.scatter([performance[1]], [performance[0]], marker="*", s=320,
                     color="crimson", zorder=5, label="Selected portfolio")

        axes.set_xlabel("Annualised volatility")
        axes.set_ylabel("Expected annual return")
        axes.set_title(
            f"Efficient frontier, Ledoit-Wolf covariance, "
            f"weights bounded at {display.percent(bounds[1], 0)}"
        )
        axes.legend(loc="best", fontsize=9)
        axes.grid(alpha=0.3)
        figure.tight_layout()

        path = config.OUTPUT_DIR / "f2_efficient_frontier.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        return path
    except Exception as exc:  # noqa: BLE001 - see docstring
        display.warn(f"The frontier chart could not be produced ({exc}). "
                     "The figures above are unaffected.")
        return None


def explain_optimisation(weights: dict[str, float],
                         performance: tuple[float, float, float],
                         bound_report: dict[str, object],
                         bounds: tuple[float, float],
                         lookback_years: int, objective_name: str,
                         leftover: float, amount: float,
                         session: Session, from_f1: bool) -> str:
    """Compose the interpretation line.

    The useful sentence here is not the Sharpe ratio. It is the comparison
    between what the optimiser did and what the strategic allocation said, plus
    an honest statement of how much of that difference is an artefact of the
    lookback window rather than a finding about the assets.
    """
    lines: list[str] = []
    at_ceiling = list(bound_report["at_ceiling"])
    top_two = float(bound_report["top_two_share"])
    dropped = int(bound_report["dropped"])

    if at_ceiling:
        lines.append(
            f"The weight bound is doing real work. {len(at_ceiling)} of "
            f"{len(weights)} holdings sit exactly on the "
            f"{display.percent(bounds[1], 0)} ceiling "
            f"({', '.join(at_ceiling)}), which means that left unconstrained "
            f"the optimiser would have concentrated further into them. It is "
            f"reporting the {lookback_years}-year lookback, not a durable "
            f"property of those assets: run the same optimisation over a "
            f"different window and the names on the ceiling will change."
        )
    else:
        lines.append(
            f"No holding reached the {display.percent(bounds[1], 0)} ceiling, "
            f"so the {objective_name} solution here is genuinely interior. "
            f"That is the more trustworthy case -- the bound was available and "
            f"was not needed."
        )

    if dropped:
        lines.append(
            f"The optimiser dropped {dropped} of the "
            f"{bound_report['universe_size']} assets offered to it entirely, "
            f"and put {display.percent(top_two)} into its top two holdings. "
            f"Mean-variance optimisation does this routinely: it treats "
            f"estimation error in the return means as though it were signal. "
            f"Ledoit-Wolf shrinkage on the covariance matrix limits how far "
            f"the effect can run, but it does not remove it."
        )

    if from_f1 and session.risk_profile is not None:
        equity_optimised = sum(
            w for t, w in weights.items()
            if t in config.SLEEVES and config.SLEEVES[t].is_equity
        )
        equity_strategic = session.risk_profile.equity_weight
        drift = equity_optimised - equity_strategic
        direction = "more" if drift > 0 else "less"
        lines.append(
            f"Against Function 1's strategic allocation, the optimised "
            f"portfolio holds {display.percent(abs(drift))} {direction} equity "
            f"({display.percent(equity_optimised)} against a target of "
            f"{display.percent(equity_strategic)}). The strategic weight was "
            f"set by the client's risk profile; this one was set by the return "
            f"history of the last {lookback_years} years. Where they disagree, "
            f"the suitability judgement in Function 1 is the one with a client "
            f"behind it."
        )

    lines.append(
        f"Whole-share rounding leaves {display.money(leftover)} uninvested, a "
        f"cash drag of {display.percent(leftover / amount, 2)}. That is the "
        f"real cost of executability, and it is why fractional shares are a "
        f"competitive feature rather than a technicality."
    )

    return "\n\n".join(lines)
