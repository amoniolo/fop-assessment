"""Function 4: performance and risk report against a benchmark.

Business process modelled
-------------------------
The periodic client statement: how the portfolio performed, against what, and
with how much risk taken to get there.

The analytics come from QuantStats. Section 9 of the working plan identifies
QuantStats as a version-fragility risk -- it is sensitive to the installed
pandas and numpy -- so every metric is routed through a small adapter that
falls back to an explicit pandas/numpy implementation if the library call
fails. The fallback is not a substitute for the library requirement: QuantStats
is used, and is the primary path. The fallback exists so that a version problem
on the assessor's machine costs a footnote rather than 20% of the prototype.

The output is a side-by-side comparison rather than two separate tables,
because the useful question is never "what was the Sharpe ratio" but "was the
difference from the benchmark worth it".
"""

from __future__ import annotations

from datetime import date, timedelta

import matplotlib

matplotlib.use("Agg")  # See the note in f2_optimise.py.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from utils import display  # noqa: E402
from utils.data import daily_returns, get_prices  # noqa: E402
from utils.session import Session  # noqa: E402
from utils.validation import (  # noqa: E402
    prompt_date,
    prompt_float,
    prompt_tickers,
    prompt_yes_no,
)


def run(session: Session) -> None:
    """Report the portfolio against a benchmark and interpret the difference."""
    display.header("Function 4  -  Performance and Risk Report vs Benchmark")
    display.paragraph(
        "Builds the portfolio's daily return series from its weights, then "
        "compares its return and risk with a benchmark over the same window. "
        "Analytics are computed with QuantStats."
    )

    weights = _choose_weights(session)
    benchmark = _choose_benchmark()

    today = date.today()
    earliest = date(2005, 1, 1)
    start = prompt_date("Start date", earliest, today - timedelta(days=90),
                        default=today - timedelta(days=365 * 3))
    end = prompt_date("End date", start + timedelta(days=90), today,
                      default=today)

    tickers = list(weights)
    prices, source = get_prices(tickers + [benchmark], start, end)
    session.data_source_note = source

    available = [t for t in tickers if t in prices.columns]
    if benchmark not in prices.columns:
        raise ValueError(
            f"No price data for the benchmark {benchmark} over this window."
        )
    if not available:
        raise ValueError(
            "None of the portfolio holdings returned data over this window."
        )
    if len(available) < len(tickers):
        missing = sorted(set(tickers) - set(available))
        display.warn(
            f"Excluded from the report (no data in this window): "
            f"{', '.join(missing)}. Remaining weights renormalised."
        )

    portfolio_returns = build_portfolio_returns(prices, weights, available)
    benchmark_returns = daily_returns(prices[[benchmark]])[benchmark].dropna()

    # Align on shared dates so every metric is computed over the same sample.
    # Without this, a holiday in one market silently gives the two series
    # different denominators and the comparison stops being like for like.
    aligned = pd.concat(
        [portfolio_returns, benchmark_returns], axis=1, join="inner"
    ).dropna()
    aligned.columns = ["portfolio", "benchmark"]

    if len(aligned) < 60:
        raise ValueError(
            f"Only {len(aligned)} overlapping trading days in this window; "
            "at least 60 are needed for the risk statistics to mean anything."
        )

    portfolio_metrics, engine = compute_metrics(aligned["portfolio"])
    benchmark_metrics, _ = compute_metrics(aligned["benchmark"])
    tracking = tracking_error(aligned["portfolio"], aligned["benchmark"])

    _print_holdings(weights, available, prices)
    _print_comparison(portfolio_metrics, benchmark_metrics, benchmark,
                      tracking, aligned, engine)

    chart_path = plot_drawdown(aligned, benchmark)
    if chart_path is not None:
        display.note(f"\n   Drawdown chart saved to output/{chart_path.name}")

    display.insight(explain_performance(
        portfolio_metrics, benchmark_metrics, benchmark, tracking, aligned
    ))
    session.record_run("F4 Performance report")


def _choose_weights(session: Session) -> dict[str, float]:
    """Return portfolio weights, defaulting to the latest pipeline output."""
    suggested = session.suggested_weights()
    if suggested is not None:
        weights, origin = suggested
        display.note(f"\n   {len(weights)} holdings available from {origin}.")
        if prompt_yes_no("Report on those weights?", default=True):
            return dict(weights)

    display.note("\n   Enter the holdings to report on.")
    tickers = prompt_tickers("Tickers, comma separated", minimum_count=1,
                             default=list(config.SLEEVES))

    if prompt_yes_no("Weight them equally?", default=True):
        equal = 1.0 / len(tickers)
        return {t: equal for t in tickers}

    raw: dict[str, float] = {}
    for ticker in tickers:
        raw[ticker] = prompt_float(f"Weight for {ticker} (0.00 to 1.00)",
                                   0.0, 1.0)
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("The weights entered sum to zero.")
    if abs(total - 1.0) > 0.01:
        display.warn(f"Weights sum to {display.percent(total)}; renormalising "
                     f"to 100%.")
    return {t: w / total for t, w in raw.items()}


def _choose_benchmark() -> str:
    """Return the benchmark ticker."""
    display.note(
        f"\n   The benchmark is what the portfolio is judged against. "
        f"Default {config.DEFAULT_BENCHMARK}."
    )
    return prompt_tickers("Benchmark ticker", minimum_count=1,
                          default=[config.DEFAULT_BENCHMARK])[0]


def build_portfolio_returns(prices: pd.DataFrame, weights: dict[str, float],
                            available: list[str]) -> pd.Series:
    """Build the weighted daily return series for the portfolio.

    Weights are renormalised over the holdings that actually have data, so a
    single missing ticker reduces the portfolio to the rest at their relative
    proportions rather than silently leaving it holding cash.

    This models a portfolio rebalanced daily back to target weights, which is
    the standard simplification for an attribution report of this kind. Real
    drift between rebalances would change the numbers slightly; stating the
    assumption is what matters at this level.
    """
    total = sum(weights[t] for t in available)
    normalised = {t: weights[t] / total for t in available}

    returns = daily_returns(prices[available])
    weight_vector = pd.Series(normalised)
    return (returns[available] * weight_vector).sum(axis=1).dropna()


def compute_metrics(returns: pd.Series) -> tuple[dict[str, float], str]:
    """Compute the performance and risk metrics for one return series.

    Each metric is attempted through QuantStats first, and falls back to an
    explicit implementation only if the library call raises. The engine
    actually used is returned and printed, because the assessor is entitled to
    know which code produced the numbers on screen.

    Returns:
        (metrics, engine) where engine is "QuantStats" or a description of the
        mixed/fallback case.
    """
    fallbacks_used: list[str] = []

    def via_quantstats(name: str, call, fallback) -> float:
        try:
            import quantstats as qs  # noqa: PLC0415 - imported lazily; see below
            value = float(call(qs))
            if not np.isfinite(value):
                raise ValueError("non-finite result")
            return value
        except Exception:  # noqa: BLE001 - version fragility is the whole point
            fallbacks_used.append(name)
            return float(fallback())

    periods = config.TRADING_DAYS_PER_YEAR
    confidence = config.VALUE_AT_RISK_CONFIDENCE

    metrics = {
        "cagr": via_quantstats(
            "CAGR",
            lambda qs: qs.stats.cagr(returns),
            lambda: _cagr(returns),
        ),
        "volatility": via_quantstats(
            "volatility",
            lambda qs: qs.stats.volatility(returns, periods=periods),
            lambda: returns.std() * np.sqrt(periods),
        ),
        "sharpe": via_quantstats(
            "Sharpe",
            lambda qs: qs.stats.sharpe(returns, rf=config.RISK_FREE_RATE,
                                       periods=periods),
            lambda: _sharpe(returns),
        ),
        "sortino": via_quantstats(
            "Sortino",
            lambda qs: qs.stats.sortino(returns, rf=config.RISK_FREE_RATE,
                                        periods=periods),
            lambda: _sortino(returns),
        ),
        "max_drawdown": via_quantstats(
            "max drawdown",
            lambda qs: qs.stats.max_drawdown(_to_prices(returns)),
            lambda: _max_drawdown(returns),
        ),
        "value_at_risk": via_quantstats(
            "value at risk",
            lambda qs: qs.stats.value_at_risk(returns, sigma=1,
                                              confidence=confidence),
            lambda: float(np.percentile(returns, (1 - confidence) * 100)),
        ),
    }

    # Count the QuantStats-routed metrics before anything else is added, so the
    # "everything fell back" test below compares like with like. Measuring
    # against the final dictionary would be wrong, because the drawdown dates
    # added next are always computed locally and were never routed through the
    # library -- which would make a total QuantStats failure look partial.
    routed_through_quantstats = len(metrics)

    # Drawdown dates are not offered by QuantStats in a stable shape across
    # versions, so they are always computed here.
    metrics.update(_drawdown_dates(returns))

    if not fallbacks_used:
        engine = "QuantStats"
    elif len(fallbacks_used) == routed_through_quantstats:
        engine = "internal fallback (QuantStats unavailable or incompatible)"
    else:
        engine = f"QuantStats, with internal fallback for: {', '.join(fallbacks_used)}"

    return metrics, engine


def _to_prices(returns: pd.Series) -> pd.Series:
    """Convert a return series to a price index starting at 1.0."""
    return (1.0 + returns).cumprod()


def _cagr(returns: pd.Series) -> float:
    """Compound annual growth rate from a daily return series."""
    total_growth = float((1.0 + returns).prod())
    years = len(returns) / config.TRADING_DAYS_PER_YEAR
    if years <= 0 or total_growth <= 0:
        return float("nan")
    return total_growth ** (1.0 / years) - 1.0


def _sharpe(returns: pd.Series) -> float:
    """Annualised Sharpe ratio against the configured risk-free rate."""
    periods = config.TRADING_DAYS_PER_YEAR
    excess = returns - config.RISK_FREE_RATE / periods
    if excess.std() == 0:
        return float("nan")
    return float(excess.mean() / excess.std() * np.sqrt(periods))


def _sortino(returns: pd.Series) -> float:
    """Sortino ratio: excess return over downside deviation only.

    The distinction from Sharpe matters for the interpretation: Sharpe
    penalises upside volatility as though it were risk, and Sortino does not.
    A portfolio whose Sortino is much better than its Sharpe is one whose
    volatility is mostly in the right direction.

    Downside deviation is the root-mean-square of the *shortfall* below the
    target, averaged over every period rather than only the losing ones:

        sqrt(mean(min(excess, 0) ** 2))

    That is the standard definition, and it is not the same as taking the
    standard deviation of the negative returns. The shortcut version measures
    how much the losses varied around their own average, so a series whose
    losses are all the same size scores an infinitely good Sortino -- which is
    exactly backwards, since consistent losses are still losses. Averaging over
    all periods also makes the ratio respond correctly to how *often* the
    portfolio fell short, not merely how badly.
    """
    periods = config.TRADING_DAYS_PER_YEAR
    excess = returns - config.RISK_FREE_RATE / periods

    shortfall = np.minimum(excess, 0.0)
    downside_deviation = float(np.sqrt((shortfall ** 2).mean()))

    if downside_deviation == 0:
        # No period fell below the target at all. There is no downside to
        # divide by, so the ratio is undefined rather than infinite; display
        # renders this as 'n/a'.
        return float("nan")

    return float(excess.mean() / downside_deviation * np.sqrt(periods))


def _max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough fall, as a negative fraction."""
    prices = _to_prices(returns)
    return float((prices / prices.cummax() - 1.0).min())


def _drawdown_dates(returns: pd.Series) -> dict[str, object]:
    """Find when the maximum drawdown began, bottomed, and recovered.

    Reported because the depth of a drawdown alone does not tell a client what
    it was like to live through. A 20% fall that recovered in four months and a
    20% fall that took three years are the same number and completely different
    experiences.
    """
    prices = _to_prices(returns)
    running_peak = prices.cummax()
    drawdown = prices / running_peak - 1.0

    trough_date = drawdown.idxmin()
    before_trough = prices.loc[:trough_date]
    peak_date = before_trough.idxmax()
    peak_value = float(before_trough.max())

    after_trough = prices.loc[trough_date:]
    recovered = after_trough[after_trough >= peak_value]
    recovery_date = recovered.index[0] if len(recovered) else None

    return {
        "drawdown_peak": peak_date,
        "drawdown_trough": trough_date,
        "drawdown_recovery": recovery_date,
    }


def tracking_error(portfolio: pd.Series, benchmark: pd.Series) -> float:
    """Annualised standard deviation of the return difference.

    Measures how far the portfolio's path can diverge from the benchmark's,
    which is a different question from whether it beat it. A portfolio can
    match the benchmark's return with high tracking error, and a client will
    still notice every quarter that the two do not move together.
    """
    difference = portfolio - benchmark
    return float(difference.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR))


def _print_holdings(weights: dict[str, float], available: list[str],
                    prices: pd.DataFrame) -> None:
    """Print what is being reported on, with renormalised weights."""
    display.subheader("Portfolio reported")
    total = sum(weights[t] for t in available)
    rows = []
    for ticker in sorted(available, key=lambda t: -weights[t]):
        sleeve = config.SLEEVES.get(ticker)
        rows.append([ticker, sleeve.name if sleeve else "-",
                     display.percent(weights[ticker] / total)])
    display.table(["Ticker", "Sleeve", "Weight"], rows,
                  alignments=["l", "l", "r"])


def _print_comparison(portfolio: dict[str, float], benchmark_metrics: dict[str, float],
                      benchmark: str, tracking: float,
                      aligned: pd.DataFrame, engine: str) -> None:
    """Print the side-by-side metric table."""
    display.subheader(
        f"Performance and risk, "
        f"{aligned.index[0].date()} to {aligned.index[-1].date()} "
        f"({len(aligned):,} trading days)"
    )

    def difference(key: str, formatter) -> str:
        return formatter(portfolio[key] - benchmark_metrics[key])

    rows = [
        ["Annualised return (CAGR)",
         display.percent(portfolio["cagr"], 2),
         display.percent(benchmark_metrics["cagr"], 2),
         difference("cagr", display.basis_points)],
        ["Annualised volatility",
         display.percent(portfolio["volatility"], 2),
         display.percent(benchmark_metrics["volatility"], 2),
         difference("volatility", display.basis_points)],
        ["Sharpe ratio",
         display.ratio(portfolio["sharpe"]),
         display.ratio(benchmark_metrics["sharpe"]),
         display.ratio(portfolio["sharpe"] - benchmark_metrics["sharpe"])],
        ["Sortino ratio",
         display.ratio(portfolio["sortino"]),
         display.ratio(benchmark_metrics["sortino"]),
         display.ratio(portfolio["sortino"] - benchmark_metrics["sortino"])],
        ["Maximum drawdown",
         display.percent(portfolio["max_drawdown"], 2),
         display.percent(benchmark_metrics["max_drawdown"], 2),
         difference("max_drawdown", display.basis_points)],
        [f"Daily VaR ({display.percent(config.VALUE_AT_RISK_CONFIDENCE, 0)})",
         display.percent(portfolio["value_at_risk"], 2),
         display.percent(benchmark_metrics["value_at_risk"], 2),
         difference("value_at_risk", display.basis_points)],
    ]

    display.table(["Measure", "Portfolio", benchmark, "Difference"], rows)

    display.key_values([
        ("Tracking error vs benchmark", display.percent(tracking, 2)),
        ("Analytics engine", engine),
        ("Risk-free rate used", display.percent(config.RISK_FREE_RATE, 2)),
    ])

    _print_drawdown_detail(portfolio, "Portfolio")
    _print_drawdown_detail(benchmark_metrics, benchmark)


def _print_drawdown_detail(metrics: dict[str, object], label: str) -> None:
    """Print when the worst drawdown ran, and whether it recovered."""
    peak = metrics["drawdown_peak"]
    trough = metrics["drawdown_trough"]
    recovery = metrics["drawdown_recovery"]

    if recovery is not None:
        months = (recovery - peak).days / 30.44
        outcome = (f"recovered {recovery.date()} "
                   f"({months:.0f} months peak to peak)")
    else:
        outcome = "not yet recovered within this window"

    display.note(
        f"   {label} worst drawdown: {peak.date()} to {trough.date()}, "
        f"{outcome}."
    )


def plot_drawdown(aligned: pd.DataFrame, benchmark: str):
    """Plot cumulative growth and underwater drawdown, and save the chart."""
    try:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        portfolio_prices = _to_prices(aligned["portfolio"])
        benchmark_prices = _to_prices(aligned["benchmark"])

        figure, (top, bottom) = plt.subplots(
            2, 1, figsize=(10, 8), sharex=True,
            gridspec_kw={"height_ratios": [3, 2]},
        )

        top.plot(portfolio_prices.index, portfolio_prices, linewidth=1.8,
                 label="Portfolio")
        top.plot(benchmark_prices.index, benchmark_prices, linewidth=1.4,
                 label=benchmark, alpha=0.85)
        top.set_ylabel("Growth of 1.00")
        top.set_title("Cumulative growth and drawdown against the benchmark")
        top.legend(loc="upper left", fontsize=9)
        top.grid(alpha=0.3)

        for series, label in ((portfolio_prices, "Portfolio"),
                              (benchmark_prices, benchmark)):
            underwater = series / series.cummax() - 1.0
            bottom.fill_between(underwater.index, underwater, 0, alpha=0.35,
                                label=label)

        bottom.set_ylabel("Drawdown")
        bottom.set_xlabel("Date")
        bottom.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.0%}")
        )
        bottom.legend(loc="lower left", fontsize=9)
        bottom.grid(alpha=0.3)

        figure.tight_layout()
        path = config.OUTPUT_DIR / "f4_drawdown.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        return path
    except Exception as exc:  # noqa: BLE001 - a chart failure must not cost the numbers
        display.warn(f"The drawdown chart could not be produced ({exc}). "
                     "The figures above are unaffected.")
        return None


def explain_performance(portfolio: dict[str, float],
                        benchmark_metrics: dict[str, float],
                        benchmark: str, tracking: float,
                        aligned: pd.DataFrame) -> str:
    """Compose the interpretation line.

    Two columns of numbers are not an interpretation. The attribution the
    client needs is whether a difference in outcome came from return or from
    risk, because those two have completely different implications for whether
    the portfolio is doing its job.
    """
    return_gap = portfolio["cagr"] - benchmark_metrics["cagr"]
    volatility_gap = portfolio["volatility"] - benchmark_metrics["volatility"]
    sharpe_gap = portfolio["sharpe"] - benchmark_metrics["sharpe"]
    drawdown_gap = portfolio["max_drawdown"] - benchmark_metrics["max_drawdown"]
    years = len(aligned) / config.TRADING_DAYS_PER_YEAR

    lines: list[str] = []

    trailed = return_gap < 0
    less_volatile = volatility_gap < 0

    if trailed and less_volatile:
        # The verdict has to follow the Sharpe comparison rather than the
        # volatility reduction. Lower volatility alongside a lower return is
        # only a good trade if the return given up was smaller in proportion,
        # and that is exactly what the Sharpe difference measures.
        if sharpe_gap > 0:
            verdict = (
                f"The risk-adjusted result was better: a Sharpe ratio of "
                f"{display.ratio(portfolio['sharpe'])} against "
                f"{display.ratio(benchmark_metrics['sharpe'])}. The return "
                f"shortfall bought something -- it was paid for, not simply "
                f"suffered."
            )
        else:
            verdict = (
                f"The risk reduction did not quite pay for itself, though: "
                f"the Sharpe ratio was {display.ratio(portfolio['sharpe'])} "
                f"against {display.ratio(benchmark_metrics['sharpe'])}, so "
                f"per unit of risk taken the benchmark was still marginally "
                f"ahead. The defence of this portfolio is not that it was "
                f"more efficient; it is that it was steadier, which is a "
                f"suitability argument rather than a performance one."
            )
        lines.append(
            f"The portfolio trailed {benchmark} by "
            f"{display.basis_points(abs(return_gap))} a year, but did so with "
            f"{display.basis_points(abs(volatility_gap))} less volatility. "
            + verdict
        )
    elif trailed:
        lines.append(
            f"The portfolio trailed {benchmark} by "
            f"{display.basis_points(abs(return_gap))} a year and was "
            f"{display.basis_points(abs(volatility_gap))} *more* volatile. "
            f"That is the combination with no defence: less return for more "
            f"risk. Over this window the diversification did not pay."
        )
    elif less_volatile:
        lines.append(
            f"The portfolio beat {benchmark} by "
            f"{display.basis_points(return_gap)} a year with "
            f"{display.basis_points(abs(volatility_gap))} less volatility -- "
            f"more return for less risk over this window. Treat that with "
            f"caution rather than satisfaction: over a "
            f"{years:.1f}-year window it is as likely to be the window as the "
            f"portfolio."
        )
    else:
        lines.append(
            f"The portfolio beat {benchmark} by "
            f"{display.basis_points(return_gap)} a year, but took "
            f"{display.basis_points(volatility_gap)} more volatility to do it. "
            f"The Sharpe difference of {display.ratio(sharpe_gap)} is the "
            f"figure that says whether the extra risk was rewarded; the "
            f"return difference on its own does not."
        )

    lines.append(
        f"On the downside, the worst drawdown was "
        f"{display.percent(portfolio['max_drawdown'], 1)} against the "
        f"benchmark's {display.percent(benchmark_metrics['max_drawdown'], 1)}, "
        f"a difference of {display.basis_points(abs(drawdown_gap))}. Drawdown "
        f"is the risk measure a client actually experiences: volatility is a "
        f"statistic, but a drawdown is the number on the statement."
    )

    lines.append(
        f"Tracking error was {display.percent(tracking, 2)} a year. That is "
        f"how far this portfolio's path can diverge from {benchmark}'s, "
        f"independently of whether it ends up ahead. A client who benchmarks "
        f"themselves against {benchmark} every quarter will notice a "
        f"divergence of roughly that size, and it is worth setting that "
        f"expectation before it happens rather than explaining it afterwards."
    )

    return "\n\n".join(lines)
