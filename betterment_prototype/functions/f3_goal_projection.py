"""Function 3: goal projection with a de-risking glide path.

Business process modelled
-------------------------
Betterment's goal-based projection and its automatic de-risking as a target
date approaches. A goal-based robo-adviser does not hold one allocation for
twenty years: it starts equity-heavy and walks the allocation down so that the
portfolio is not exposed to a bear market in the year the money is needed.

The point of the function is the gap between two answers to the same question.
The deterministic time-value-of-money calculation says a given contribution
reaches the target. The simulation says it reaches the target some percentage
of the time. Both are correct; they answer different questions, and the second
is the one a client actually needs. Naming that gap, and stating what it would
take to close it, is the output of this function.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # See the note in f2_optimise.py.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy_financial as npf  # noqa: E402

import config  # noqa: E402
from utils import display  # noqa: E402
from utils.session import Session  # noqa: E402
from utils.validation import prompt_float, prompt_int, prompt_yes_no  # noqa: E402

MONTHS_PER_YEAR = 12


def run(session: Session) -> None:
    """Project a goal, simulate it, and report the deterministic/stochastic gap."""
    display.header("Function 3  -  Goal Projection with a De-risking Glide Path")
    display.paragraph(
        "Projects a savings goal under a glide path that reduces equity as the "
        "target date approaches. A deterministic calculation gives the "
        "headline answer; a seeded Monte Carlo simulation gives the "
        "probability that the answer actually holds."
    )

    current_balance = prompt_float(
        "Current balance", minimum=0.0, maximum=100_000_000.0,
        default=session.suggested_amount() or 50_000.0,
    )
    monthly_contribution = prompt_float(
        "Monthly contribution", minimum=0.0, maximum=1_000_000.0, default=1_000.0,
    )
    years = prompt_int("Years to the target date", minimum=1, maximum=50, default=20)
    target_amount = prompt_float(
        "Target amount", minimum=1_000.0, maximum=1_000_000_000.0,
        default=max(500_000.0, current_balance * 4),
    )

    use_glidepath = prompt_yes_no(
        "Use a de-risking glide path? (No holds a flat allocation)", default=True,
    )
    if use_glidepath:
        start_equity = prompt_float(
            "Starting equity weight (0.00 to 1.00)", 0.0, 1.0,
            default=_suggested_start_equity(session),
        )
        end_equity = prompt_float(
            "Equity weight at the target date (0.00 to 1.00)", 0.0, 1.0,
            default=config.DEFAULT_GLIDEPATH_END_EQUITY,
        )
    else:
        start_equity = end_equity = prompt_float(
            "Flat equity weight (0.00 to 1.00)", 0.0, 1.0,
            default=_suggested_start_equity(session),
        )

    months = years * MONTHS_PER_YEAR
    glidepath = build_glidepath(years, start_equity, end_equity)

    display.note(
        f"\n   Random seed fixed at {config.RANDOM_SEED}, so these figures "
        f"reproduce exactly on any machine."
    )
    display.note(
        "   Return and volatility assumptions are the illustrative estimates "
        "in config.py, not a forecast and not Betterment's own."
    )

    _print_glidepath(glidepath, years)

    # Deterministic anchor. Uses the mean expected return of the glide path so
    # that the two calculations are answering the same question about the same
    # portfolio, and any difference between them is attributable to volatility
    # rather than to different return assumptions.
    monthly_rate = _mean_monthly_return(glidepath)
    deterministic_fv = float(npf.fv(
        rate=monthly_rate, nper=months,
        pmt=-monthly_contribution, pv=-current_balance,
    ))
    required_pmt = float(npf.pmt(
        rate=monthly_rate, nper=months,
        pv=-current_balance, fv=target_amount,
    ))

    print("\n   Running Monte Carlo simulation...")
    terminal_wealth = simulate(
        paths=config.DEFAULT_MONTE_CARLO_PATHS,
        seed=config.RANDOM_SEED,
        glidepath=glidepath,
        current_balance=current_balance,
        monthly_contribution=monthly_contribution,
    )
    success_probability = float((terminal_wealth >= target_amount).mean())

    stochastic_pmt = solve_contribution_for_probability(
        glidepath=glidepath,
        current_balance=current_balance,
        target_amount=target_amount,
        probability_target=config.SUCCESS_PROBABILITY_TARGET,
    )

    _print_projection(terminal_wealth, target_amount, deterministic_fv,
                      success_probability, monthly_rate, months)
    _print_contributions(monthly_contribution, required_pmt, stochastic_pmt,
                         success_probability)

    chart_path = plot_funnel(glidepath, current_balance, monthly_contribution,
                             target_amount, years)
    if chart_path is not None:
        display.note(f"\n   Percentile funnel chart saved to output/{chart_path.name}")

    display.insight(explain_projection(
        deterministic_fv, terminal_wealth, target_amount, success_probability,
        monthly_contribution, required_pmt, stochastic_pmt, years,
        start_equity, end_equity, use_glidepath,
    ))
    session.record_run("F3 Goal projection")


def _suggested_start_equity(session: Session) -> float:
    """Default the starting equity weight to F1's tier when one exists."""
    if session.risk_profile is not None:
        return session.risk_profile.equity_weight
    return config.DEFAULT_GLIDEPATH_START_EQUITY


def build_glidepath(years: int, start_equity: float,
                    end_equity: float) -> np.ndarray:
    """Build the per-month equity weight from the start to the target date.

    A linear path is used rather than a stepped one. Stepped paths are what
    actually gets implemented -- a portfolio is not rebalanced monthly to three
    decimal places -- but a linear path makes the mechanism legible, and the
    difference in the projected outcome is immaterial next to the width of the
    simulated distribution.

    Args:
        years: Length of the path.
        start_equity: Equity weight in month one.
        end_equity: Equity weight in the final month.

    Returns:
        An array of length years * 12, each element an equity weight.
    """
    months = years * MONTHS_PER_YEAR
    return np.linspace(start_equity, end_equity, months)


def _portfolio_moments(equity_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derive monthly mean and standard deviation from equity weights.

    Volatility is combined through the two-asset variance formula rather than
    by weighting the two volatilities linearly. That distinction is the whole
    reason a glide path works: because equities and bonds are imperfectly
    correlated, a blended portfolio's volatility is lower than the weighted
    average of its parts, and a linear approximation would overstate the risk
    of the balanced middle of the path.

    Returns:
        (monthly_mean, monthly_std), each the same length as equity_weights.
    """
    equity = config.ASSET_CLASS_ASSUMPTIONS["equity"]
    bonds = config.ASSET_CLASS_ASSUMPTIONS["bonds"]
    rho = config.EQUITY_BOND_CORRELATION

    bond_weights = 1.0 - equity_weights

    annual_mean = (equity_weights * equity["expected_return"]
                   + bond_weights * bonds["expected_return"])

    annual_variance = (
        (equity_weights ** 2) * equity["volatility"] ** 2
        + (bond_weights ** 2) * bonds["volatility"] ** 2
        + 2 * equity_weights * bond_weights * rho
        * equity["volatility"] * bonds["volatility"]
    )
    annual_std = np.sqrt(annual_variance)

    monthly_mean = annual_mean / MONTHS_PER_YEAR
    monthly_std = annual_std / np.sqrt(MONTHS_PER_YEAR)
    return monthly_mean, monthly_std


def _mean_monthly_return(glidepath: np.ndarray) -> float:
    """Average monthly return across the glide path, for the deterministic case."""
    monthly_mean, _ = _portfolio_moments(glidepath)
    return float(monthly_mean.mean())


def simulate(paths: int, seed: int, glidepath: np.ndarray,
             current_balance: float, monthly_contribution: float) -> np.ndarray:
    """Simulate terminal wealth over the glide path.

    Vectorised over paths rather than looped: a 10,000-path, 240-month
    simulation is 2.4 million draws, and a Python loop over them would take
    long enough that the user would think the program had hung.

    The generator is seeded from config.RANDOM_SEED and the seed is printed in
    the output, so the assessor reproduces exactly the figures in the report.

    Args:
        paths: Number of simulated paths.
        seed: Seed for the random generator.
        glidepath: Per-month equity weights from build_glidepath().
        current_balance: Starting balance.
        monthly_contribution: Contribution added at the end of each month.

    Returns:
        An array of length `paths` holding terminal wealth for each path.
    """
    generator = np.random.default_rng(seed)
    monthly_mean, monthly_std = _portfolio_moments(glidepath)
    months = len(glidepath)

    # One normal draw per path per month, scaled to that month's moments.
    shocks = generator.standard_normal((paths, months))
    returns = monthly_mean[np.newaxis, :] + shocks * monthly_std[np.newaxis, :]

    balance = np.full(paths, current_balance, dtype=float)
    for month in range(months):
        # Contribution is added after the month's return, which treats it as an
        # end-of-period payment and matches the convention npf.fv uses by
        # default. Getting this wrong is worth about one month of growth, and
        # it would make the deterministic and stochastic answers differ for a
        # reason that has nothing to do with volatility.
        balance = balance * (1.0 + returns[:, month]) + monthly_contribution

    return balance


def solve_contribution_for_probability(
    glidepath: np.ndarray,
    current_balance: float,
    target_amount: float,
    probability_target: float,
    tolerance: float = 0.005,
    maximum_iterations: int = 40,
) -> float | None:
    """Bisect on the contribution needed to hit a success probability.

    Bisection is valid here because success probability is monotonically
    non-decreasing in the contribution: adding money can never reduce the
    chance of reaching a fixed target. Every evaluation re-seeds from the same
    seed, so the objective function is deterministic and the search cannot
    oscillate on simulation noise -- which it would if each call drew fresh
    randomness.

    A smaller path count is used inside the search than for the headline
    figure: 40 bisection steps at 10,000 paths would be noticeably slow, and
    2,000 paths is ample to locate the contribution to within the tolerance.

    Returns:
        The monthly contribution, or None if the target is unreachable within
        the search ceiling.
    """
    search_paths = 2_000

    def probability(contribution: float) -> float:
        wealth = simulate(search_paths, config.RANDOM_SEED, glidepath,
                          current_balance, contribution)
        return float((wealth >= target_amount).mean())

    low = 0.0
    # Upper bound: enough to reach the target on contributions alone, with no
    # investment return at all. Nothing above that can be necessary.
    high = max(target_amount / len(glidepath), 1.0)

    if probability(high) < probability_target:
        return None

    for _ in range(maximum_iterations):
        middle = (low + high) / 2.0
        if probability(middle) >= probability_target:
            high = middle
        else:
            low = middle
        if high - low < tolerance * max(high, 1.0):
            break

    return high


def _print_glidepath(glidepath: np.ndarray, years: int) -> None:
    """Print the equity weight at a few checkpoints along the path."""
    display.subheader("Glide path")
    checkpoints = sorted({0, years // 4, years // 2, (3 * years) // 4, years - 1})
    rows = []
    for year in checkpoints:
        month_index = min(year * MONTHS_PER_YEAR, len(glidepath) - 1)
        equity = float(glidepath[month_index])
        rows.append([f"Year {year + 1}", display.percent(equity, 0),
                     display.percent(1 - equity, 0)])
    display.table(["Point on path", "Equity", "Bonds"], rows)


def _print_projection(terminal_wealth: np.ndarray, target_amount: float,
                      deterministic_fv: float, success_probability: float,
                      monthly_rate: float, months: int) -> None:
    """Print the deterministic anchor beside the simulated distribution."""
    display.subheader("Projected outcome at the target date")
    p5, p50, p95 = np.percentile(terminal_wealth, [5, 50, 95])

    display.table(
        ["Measure", "Value"],
        [
            ["Deterministic projection", display.money(deterministic_fv, 0)],
            ["Simulated median (50th pct)", display.money(float(p50), 0)],
            ["Pessimistic case (5th pct)", display.money(float(p5), 0)],
            ["Optimistic case (95th pct)", display.money(float(p95), 0)],
            ["Target", display.money(target_amount, 0)],
            ["Probability of reaching target",
             display.percent(success_probability)],
        ],
    )
    display.note(
        f"   Deterministic case compounds at {display.percent(monthly_rate * 12, 2)} "
        f"a year over {months} months with no volatility at all."
    )


def _print_contributions(current: float, deterministic_required: float,
                         stochastic_required: float | None,
                         success_probability: float) -> None:
    """Print the three contribution figures side by side."""
    display.subheader("Required monthly contribution")
    rows = [
        ["Currently planned", display.money(current), "-"],
        ["To reach target deterministically",
         display.money(abs(deterministic_required)),
         "certainty assumed"],
    ]
    if stochastic_required is not None:
        rows.append([
            f"To reach target {display.percent(config.SUCCESS_PROBABILITY_TARGET, 0)} "
            f"of the time",
            display.money(stochastic_required),
            "simulated",
        ])
    else:
        rows.append([
            f"To reach target {display.percent(config.SUCCESS_PROBABILITY_TARGET, 0)} "
            f"of the time",
            "not reachable",
            "raise the target date or lower the target",
        ])
    display.table(["Basis", "Monthly", "Note"], rows,
                  alignments=["l", "r", "l"])


def plot_funnel(glidepath: np.ndarray, current_balance: float,
                monthly_contribution: float, target_amount: float,
                years: int):
    """Plot the percentile funnel of simulated balances over time.

    Returns the saved path, or None if the chart failed. As in F2, a charting
    failure must not cost the user their numbers.
    """
    try:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Re-run with the path history retained. Kept separate from simulate()
        # so that the headline simulation does not carry the memory cost of a
        # 10,000 by 240 matrix it has no use for.
        paths = 2_000
        generator = np.random.default_rng(config.RANDOM_SEED)
        monthly_mean, monthly_std = _portfolio_moments(glidepath)
        months = len(glidepath)

        shocks = generator.standard_normal((paths, months))
        returns = monthly_mean[np.newaxis, :] + shocks * monthly_std[np.newaxis, :]

        history = np.empty((paths, months + 1))
        history[:, 0] = current_balance
        for month in range(months):
            history[:, month + 1] = (
                history[:, month] * (1.0 + returns[:, month]) + monthly_contribution
            )

        timeline = np.arange(months + 1) / MONTHS_PER_YEAR
        p5 = np.percentile(history, 5, axis=0)
        p25 = np.percentile(history, 25, axis=0)
        p50 = np.percentile(history, 50, axis=0)
        p75 = np.percentile(history, 75, axis=0)
        p95 = np.percentile(history, 95, axis=0)

        figure, axes = plt.subplots(figsize=(10, 6))
        axes.fill_between(timeline, p5, p95, alpha=0.20,
                          label="5th to 95th percentile")
        axes.fill_between(timeline, p25, p75, alpha=0.35,
                          label="25th to 75th percentile")
        axes.plot(timeline, p50, linewidth=2, label="Median path")
        axes.axhline(target_amount, linestyle="--", color="crimson",
                     linewidth=1.5, label="Target")

        axes.set_xlabel("Years from today")
        axes.set_ylabel(f"Portfolio value ({config.CURRENCY_CODE})")
        axes.set_title(
            f"Goal projection over {years} years, "
            f"{paths:,} simulated paths, seed {config.RANDOM_SEED}"
        )
        axes.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{config.CURRENCY_SYMBOL}{v:,.0f}")
        )
        axes.legend(loc="upper left", fontsize=9)
        axes.grid(alpha=0.3)
        figure.tight_layout()

        path = config.OUTPUT_DIR / "f3_goal_funnel.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        return path
    except Exception as exc:  # noqa: BLE001 - see docstring
        display.warn(f"The funnel chart could not be produced ({exc}). "
                     "The figures above are unaffected.")
        return None


def explain_projection(deterministic_fv: float, terminal_wealth: np.ndarray,
                       target_amount: float, success_probability: float,
                       monthly_contribution: float, required_pmt: float,
                       stochastic_pmt: float | None, years: int,
                       start_equity: float, end_equity: float,
                       use_glidepath: bool) -> str:
    """Compose the interpretation line.

    The gap between the deterministic and the stochastic answer is the most
    genuinely insightful output in the artefact, so it goes first and it is
    quantified in both directions: the probability the headline number hides,
    and the contribution increase that would close it.
    """
    median = float(np.median(terminal_wealth))
    p5 = float(np.percentile(terminal_wealth, 5))
    lines: list[str] = []

    deterministic_reaches = deterministic_fv >= target_amount
    verb = "reaches" if deterministic_reaches else "falls short of"

    lines.append(
        f"The deterministic calculation {verb} the target: "
        f"{display.money(deterministic_fv, 0)} against "
        f"{display.money(target_amount, 0)}. The simulation says the same plan "
        f"reaches the target {display.percent(success_probability)} of the "
        f"time. That gap is the whole point of the function. The deterministic "
        f"figure is not wrong -- it is the answer to a different question, "
        f"namely what happens if returns arrive smoothly at their average, "
        f"which they never do."
    )

    if stochastic_pmt is not None:
        increase = stochastic_pmt - monthly_contribution
        if increase > 0:
            annual_increase = increase * MONTHS_PER_YEAR
            lines.append(
                f"Raising the success probability to "
                f"{display.percent(config.SUCCESS_PROBABILITY_TARGET, 0)} takes "
                f"{display.money(stochastic_pmt)} a month, an increase of "
                f"{display.money(increase)} -- about "
                f"{display.money(annual_increase, 0)} a year, or "
                f"{display.percent(increase / monthly_contribution) if monthly_contribution else 'n/a'} "
                f"more than currently planned. That is the price of the "
                f"certainty the deterministic number implied for free."
            )
        else:
            lines.append(
                f"The planned contribution already clears the "
                f"{display.percent(config.SUCCESS_PROBABILITY_TARGET, 0)} "
                f"threshold: {display.money(stochastic_pmt)} a month would "
                f"suffice, against the {display.money(monthly_contribution)} "
                f"planned. The plan has genuine headroom."
            )
    else:
        lines.append(
            f"No monthly contribution within the search range reaches the "
            f"{display.percent(config.SUCCESS_PROBABILITY_TARGET, 0)} "
            f"threshold. The target is not achievable on this horizon at this "
            f"risk level; the levers are a later target date or a smaller target."
        )

    lines.append(
        f"The pessimistic case matters more than the median. In the worst 5% "
        f"of paths this plan ends at {display.money(p5, 0)}, "
        f"{display.percent(1 - p5 / median)} below the median of "
        f"{display.money(median, 0)}. A client should be shown that figure "
        f"before they are shown the median, because it is the one that "
        f"determines whether the plan needs a contingency."
    )

    if use_glidepath:
        lines.append(
            f"The glide path walks equity from "
            f"{display.percent(start_equity, 0)} to "
            f"{display.percent(end_equity, 0)} over {years} years. It lowers "
            f"the median outcome, because equity is the source of the return, "
            f"and it narrows the distribution near the target date, which is "
            f"the trade being made: less expected wealth in exchange for less "
            f"chance of a bad year arriving in the final year, when there is "
            f"no time left to recover from it."
        )
    else:
        lines.append(
            f"A flat {display.percent(start_equity, 0)} equity weight was held "
            f"throughout. Re-running with a glide path will lower the median "
            f"outcome and narrow the 5th-to-95th range near the target date; "
            f"comparing the two is the clearest way to see what de-risking "
            f"actually costs and buys."
        )

    return "\n\n".join(lines)
