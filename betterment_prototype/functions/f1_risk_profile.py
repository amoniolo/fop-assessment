"""Function 1: risk profiling and strategic asset allocation.

Business process modelled
-------------------------
Betterment's onboarding questionnaire and its mapping to a model portfolio.
This is the stage where a human adviser's judgement is replaced by code, which
makes it the function most directly aligned to the learning outcome about
automating a business process.

The design decision worth defending is the horizon cap. A questionnaire score
on its own will happily put a client who needs their money in two years into an
80% equity portfolio because they said they would buy the dip. A suitability
rule does not work that way: a short horizon caps equity regardless of stated
tolerance. So the score and the cap are computed separately and the binding one
is reported by name, because a client is entitled to know which of their
answers actually determined the outcome.
"""

from __future__ import annotations

import config
from utils import display
from utils.data import PriceDataError, get_prices, latest_prices
from utils.session import RiskProfileResult, Session
from utils.validation import prompt_choice, prompt_float

# Horizon answers map to a representative number of years for the cap rule.
# The midpoint of each band, with the open-ended top band held at 25.
_HORIZON_YEARS: dict[str, int] = {"1": 2, "2": 4, "3": 8, "4": 15, "5": 25}


def run(session: Session) -> None:
    """Run the questionnaire, derive a tier, and print the allocation."""
    display.header("Function 1  -  Risk Profile and Strategic Allocation")
    display.paragraph(
        "Five scored questions determine a risk tier from 1 to 5. The tier is "
        "then mapped to a model portfolio of exchange-traded funds, and a "
        "suitability rule caps equity where the investment horizon is short."
    )

    responses = _collect_responses()
    horizon_years = _HORIZON_YEARS[responses["horizon"]]

    amount = prompt_float(
        "How much do you have to invest?",
        minimum=1_000.0, maximum=100_000_000.0, default=50_000.0,
    )

    raw_score = score_responses(responses)
    tier = map_score_to_tier(raw_score, horizon_years)
    unconstrained_tier = map_score_to_tier(raw_score, horizon_years=99)
    weights = strategic_allocation(tier)

    capped = tier.level != unconstrained_tier.level
    binding = _binding_factor(responses, raw_score, capped, horizon_years)

    _print_profile(raw_score, tier, unconstrained_tier, capped, horizon_years)
    _print_allocation(weights, amount)

    display.insight(
        explain_allocation(tier, unconstrained_tier, weights, capped,
                           horizon_years, binding, amount)
    )

    session.risk_profile = RiskProfileResult(
        tier_level=tier.level,
        tier_label=tier.label,
        raw_score=raw_score,
        equity_weight=tier.equity_weight,
        weights=weights,
        investable_amount=amount,
        horizon_years=horizon_years,
        horizon_capped=capped,
        binding_factor=binding,
    )
    session.record_run("F1 Risk profile")
    display.note(
        "\n   Saved to the session. Function 2 will offer to optimise this "
        "allocation, and Function 3 will project it against a goal."
    )


def _collect_responses() -> dict[str, str]:
    """Ask every questionnaire item and return the chosen option keys."""
    display.subheader("Onboarding questionnaire")
    return {
        question.key: prompt_choice(question.text, question.options)
        for question in config.QUESTIONNAIRE
    }


def score_responses(responses: dict[str, str]) -> float:
    """Convert option keys into a weighted score between 1.0 and 5.0.

    Weighted rather than a simple sum because the five questions are not
    equally informative: horizon and drawdown reaction between them carry 60%,
    while the stated objective carries 10%. What a client says they want is the
    least reliable of the five answers, and the weights say so.

    Args:
        responses: Question key to the option key the client chose.

    Returns:
        The weighted score, on the same 1-5 scale as the individual answers.
    """
    total = 0.0
    for question in config.QUESTIONNAIRE:
        chosen = responses[question.key]
        total += question.scores[chosen] * question.weight
    # Weights sum to 1.0, so the result is already on the 1-5 scale. Dividing
    # by the weight sum anyway keeps the function correct if a weight is ever
    # edited in config.py without the others being rebalanced.
    weight_sum = sum(q.weight for q in config.QUESTIONNAIRE)
    return total / weight_sum


def map_score_to_tier(score: float, horizon_years: int) -> config.RiskTier:
    """Map a score to a tier, then apply the horizon cap.

    The two steps are kept separate and in this order so that the caller can
    run the function twice -- once with the real horizon and once with the cap
    disabled -- and see exactly what the cap changed. That comparison is what
    F1 reports back to the client.

    Args:
        score: Weighted questionnaire score, 1.0 to 5.0.
        horizon_years: Years until the client expects to draw on the money.
            Pass a large value to compute the uncapped tier.

    Returns:
        The applicable RiskTier from config.RISK_TIERS.
    """
    tier_level = config.SCORE_BANDS[-1][1]
    for upper_bound, level in config.SCORE_BANDS:
        if score <= upper_bound:
            tier_level = level
            break

    tier = config.RISK_TIERS[tier_level]

    equity_ceiling = _equity_ceiling(horizon_years)
    if equity_ceiling is None or tier.equity_weight <= equity_ceiling:
        return tier

    # Step down to the highest tier that satisfies the ceiling. Stepping down
    # the ladder rather than simply truncating the equity weight keeps the
    # client in a real model portfolio rather than an off-ladder blend that no
    # rebalancing rule would recognise.
    eligible = [t for t in config.RISK_TIERS.values()
                if t.equity_weight <= equity_ceiling]
    return max(eligible, key=lambda t: t.equity_weight) if eligible else tier


def _equity_ceiling(horizon_years: int) -> float | None:
    """Return the equity ceiling this horizon imposes, or None if unconstrained."""
    for maximum_years, ceiling in config.HORIZON_EQUITY_CAPS:
        if horizon_years <= maximum_years:
            return ceiling
    return None


def strategic_allocation(tier: config.RiskTier) -> dict[str, float]:
    """Return sleeve weights for a tier, ordered equity first then bonds."""
    return config.model_portfolio(tier.equity_weight)


def _binding_factor(responses: dict[str, str], score: float,
                    capped: bool, horizon_years: int) -> str:
    """Name the input that actually determined the outcome.

    Answering "which of my answers mattered?" is the difference between a
    result and an insight. The horizon cap wins when it bound; otherwise the
    single highest weighted contribution to the score does.
    """
    if capped:
        return f"the {horizon_years}-year investment horizon"

    contributions = {
        q.key: q.scores[responses[q.key]] * q.weight
        for q in config.QUESTIONNAIRE
    }
    dominant = max(contributions, key=contributions.get)
    readable = {
        "horizon": "the investment horizon",
        "drawdown": "the stated reaction to a 20% drawdown",
        "income_stability": "income stability",
        "emergency_reserve": "the size of the emergency reserve",
        "objective": "the stated objective",
    }
    return readable[dominant]


def _print_profile(score: float, tier: config.RiskTier,
                   uncapped: config.RiskTier, capped: bool,
                   horizon_years: int) -> None:
    """Print the tier alongside the score that produced it."""
    display.subheader("Risk profile")
    pairs = [
        ("Weighted score", f"{score:.2f} out of 5.00"),
        ("Assigned tier", f"{tier.level} - {tier.label}"),
        ("Target equity", display.percent(tier.equity_weight, 0)),
        ("Investment horizon", f"{horizon_years} years"),
    ]
    if capped:
        pairs.append(
            ("Suitability cap",
             f"applied - score alone indicated tier {uncapped.level} "
             f"({uncapped.label}, {display.percent(uncapped.equity_weight, 0)} equity)")
        )
    else:
        pairs.append(("Suitability cap", "not binding"))
    display.key_values(pairs)
    print()
    display.paragraph(f"   {tier.description}")


def _print_allocation(weights: dict[str, float], amount: float) -> None:
    """Print the allocation by asset class, then by ticker with amounts.

    Two tables rather than one: the asset-class view is what a client reads and
    the ticker view is what gets traded, and collapsing them into a single
    ten-row table serves neither reader.
    """
    display.subheader("Target allocation by asset class")
    by_class: dict[str, float] = {}
    for ticker, weight in weights.items():
        asset_class = config.SLEEVES[ticker].asset_class
        by_class[asset_class] = by_class.get(asset_class, 0.0) + weight

    display.table(
        ["Asset class", "Weight", "Amount"],
        [[name, display.percent(w), display.money(w * amount)]
         for name, w in by_class.items()],
    )

    display.subheader("Target allocation by holding")
    prices, share_note = _try_latest_prices(list(weights))
    rows = []
    for ticker, weight in weights.items():
        sleeve = config.SLEEVES[ticker]
        cash = weight * amount
        if prices is not None and ticker in prices:
            shares = f"{cash / float(prices[ticker]):,.1f}"
        else:
            shares = "-"
        rows.append([ticker, sleeve.name, display.percent(weight),
                     display.money(cash), shares])

    display.table(
        ["Ticker", "Sleeve", "Weight", "Amount", "Indic. shares"],
        rows,
        alignments=["l", "l", "r", "r", "r"],
    )
    display.note(f"   {share_note}")
    display.note(
        "   Indicative share counts are fractional. Function 2 converts an "
        "allocation into whole shares with the residual cash stated."
    )


def _try_latest_prices(tickers: list[str]):
    """Price the sleeves, degrading to a weights-only table if pricing fails.

    F1's answer is the allocation; the share counts are a convenience. So a
    pricing failure must not stop the function from producing its result --
    it downgrades the table and says so.
    """
    from datetime import date, timedelta

    end = date.today()
    start = end - timedelta(days=30)
    try:
        prices, source = get_prices(tickers, start, end, quiet=True)
    except PriceDataError:
        return None, ("Share counts unavailable: no price data could be "
                      "retrieved. The allocation above is unaffected.")
    return latest_prices(prices), f"Share counts priced from {source}."


def explain_allocation(tier: config.RiskTier, uncapped: config.RiskTier,
                       weights: dict[str, float], capped: bool,
                       horizon_years: int, binding: str,
                       amount: float) -> str:
    """Compose the interpretation line that closes the function.

    The requirement this satisfies is that the function explains *why* it
    produced its answer. The most useful sentence it can produce is the one
    naming the input that bound the outcome, because that is the input the
    client would have to change to get a different portfolio.
    """
    equity_cash = amount * tier.equity_weight
    lines = []

    if capped:
        gap = uncapped.equity_weight - tier.equity_weight
        lines.append(
            f"The binding constraint was {binding}, not the risk score. The "
            f"questionnaire alone indicated tier {uncapped.level} "
            f"({uncapped.label}) at "
            f"{display.percent(uncapped.equity_weight, 0)} equity, but a "
            f"{horizon_years}-year horizon caps equity at "
            f"{display.percent(tier.equity_weight, 0)}. That is a "
            f"{display.percent(gap, 0)} reduction in equity, or about "
            f"{display.money(gap * amount, 0)} of this "
            f"{display.money(amount, 0)}, moved into bonds because the money "
            f"is needed too soon to ride out a drawdown."
        )
        lines.append(
            "This is the rule doing what a suitability rule is for: stated "
            "risk tolerance does not override a short horizon, because the "
            "cost of being wrong is borne at a fixed date."
        )
    else:
        lines.append(
            f"The outcome was driven by {binding}. The {horizon_years}-year "
            f"horizon is long enough that the suitability cap never bound, so "
            f"the questionnaire score alone set the tier at {tier.level} "
            f"({tier.label})."
        )
        lines.append(
            f"That places {display.money(equity_cash, 0)} of the "
            f"{display.money(amount, 0)} in equity. On the illustrative "
            f"assumptions in config.py, a portfolio at this equity weight "
            f"should be expected to fall by roughly "
            f"{display.percent(tier.equity_weight * 0.35, 0)} in a severe "
            f"equity bear market -- the figure to test tolerance against, "
            f"rather than the expected return."
        )

    return "\n\n".join(lines)
