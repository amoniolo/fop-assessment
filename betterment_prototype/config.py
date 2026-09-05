"""Central configuration for the Betterment robo-advisory prototype.

Every tunable number the four functions depend on lives here rather than being
buried in whichever function happens to use it first. Two reasons: the assessor
can see the whole parameter set on one screen, and changing a model portfolio
never means editing a function.

ILLUSTRATIVE-ASSUMPTION NOTICE
------------------------------
The sleeve list and the tier table are modelled on Betterment's publicly
disclosed Core portfolio construction. The capital-market assumptions in
ASSET_CLASS_ASSUMPTIONS are the author's own illustrative estimates. They are
not Betterment's, and they are not a forecast. They exist only to make the F3
projection run, and everything derived from them is labelled illustrative in
the output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolved from this file rather than from the working directory, so the package
# runs correctly whether it is launched from inside or outside its own folder.
PACKAGE_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PACKAGE_ROOT / "data"
OUTPUT_DIR: Path = PACKAGE_ROOT / "output"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# Fixed and printed in the F3 output so the assessor reproduces exactly the
# figures the report shows.
RANDOM_SEED: int = 20260905

# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
CURRENCY_SYMBOL: str = "$"
CURRENCY_CODE: str = "USD"
CONSOLE_WIDTH: int = 78

# ---------------------------------------------------------------------------
# Asset sleeves
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sleeve:
    """One ETF building block of a model portfolio.

    Frozen because a sleeve definition is reference data: nothing in a running
    session should mutate it. asset_class is what F1 aggregates its output by,
    so the client sees six asset classes rather than ten tickers.
    """

    ticker: str
    name: str
    asset_class: str
    is_equity: bool


# Modelled on Betterment's disclosed Core portfolio sleeves.
SLEEVES: dict[str, Sleeve] = {
    "VTI": Sleeve("VTI", "US Total Market", "US Equity", True),
    "VTV": Sleeve("VTV", "US Large-Cap Value", "US Equity", True),
    "VOE": Sleeve("VOE", "US Mid-Cap Value", "US Equity", True),
    "VBR": Sleeve("VBR", "US Small-Cap Value", "US Equity", True),
    "VEA": Sleeve("VEA", "Developed ex-US Equity", "International Equity", True),
    "VWO": Sleeve("VWO", "Emerging Market Equity", "Emerging Equity", True),
    "BND": Sleeve("BND", "US Aggregate Bond", "US Bonds", False),
    "BNDX": Sleeve("BNDX", "International Bond (hedged)", "International Bonds", False),
    "VTIP": Sleeve("VTIP", "Short-Term TIPS", "Inflation-Protected", False),
    "VWOB": Sleeve("VWOB", "Emerging Market Bond", "Emerging Bonds", False),
}

EQUITY_TICKERS: tuple[str, ...] = tuple(t for t, s in SLEEVES.items() if s.is_equity)
BOND_TICKERS: tuple[str, ...] = tuple(t for t, s in SLEEVES.items() if not s.is_equity)

# Proportions *within* each side of the equity/bond split. Held constant across
# tiers, so the only thing a risk tier changes is the split itself. That is how
# a real model-portfolio ladder is built, and it keeps the tier table readable.
EQUITY_MIX: dict[str, float] = {
    "VTI": 0.40,
    "VTV": 0.15,
    "VOE": 0.10,
    "VBR": 0.10,
    "VEA": 0.17,
    "VWO": 0.08,
}
BOND_MIX: dict[str, float] = {
    "BND": 0.50,
    "BNDX": 0.25,
    "VTIP": 0.15,
    "VWOB": 0.10,
}

# ---------------------------------------------------------------------------
# Risk tiers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskTier:
    """A rung on the model-portfolio ladder.

    The horizon cap is deliberately *not* stored here. It is a suitability rule
    that acts on a tier from outside, in F1, so that the rule stays visible at
    the point where it changes the client's answer.
    """

    level: int
    label: str
    equity_weight: float
    description: str


RISK_TIERS: dict[int, RiskTier] = {
    1: RiskTier(1, "Conservative", 0.20,
                "Capital preservation with a modest growth sleeve."),
    2: RiskTier(2, "Moderately Conservative", 0.40,
                "Income-led, with equity held back to limit drawdown."),
    3: RiskTier(3, "Balanced", 0.60,
                "Growth and income in balance; the standard core allocation."),
    4: RiskTier(4, "Growth", 0.80,
                "Growth-led, accepting meaningful interim drawdown."),
    5: RiskTier(5, "Aggressive Growth", 0.90,
                "Maximum long-horizon growth; bonds held only as ballast."),
}

# ---------------------------------------------------------------------------
# F1 questionnaire
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Question:
    """One scored onboarding question.

    weight is what turns five equally shaped questions into a weighted score,
    and key is what lets F1 report which answer actually bound the outcome.
    """

    key: str
    text: str
    options: dict[str, str]      # option number -> wording shown to the client
    scores: dict[str, int]       # option number -> 1..5 points
    weight: float


QUESTIONNAIRE: tuple[Question, ...] = (
    Question(
        key="horizon",
        text="How long before you expect to draw on this money?",
        options={"1": "Less than 3 years", "2": "3 to 5 years",
                 "3": "6 to 10 years", "4": "11 to 20 years",
                 "5": "More than 20 years"},
        scores={"1": 1, "2": 2, "3": 3, "4": 4, "5": 5},
        weight=0.30,
    ),
    Question(
        key="drawdown",
        text="Your portfolio falls 20% in six months. What would you do?",
        options={"1": "Sell everything and hold cash",
                 "2": "Sell some holdings to reduce the risk",
                 "3": "Do nothing and wait for the recovery",
                 "4": "Keep contributing exactly as planned",
                 "5": "Invest more to buy at the lower prices"},
        scores={"1": 1, "2": 2, "3": 3, "4": 4, "5": 5},
        weight=0.30,
    ),
    Question(
        key="income_stability",
        text="How stable is your income over the next five years?",
        options={"1": "Very unstable or uncertain",
                 "2": "Somewhat variable",
                 "3": "Broadly stable",
                 "4": "Stable and rising"},
        scores={"1": 1, "2": 2, "3": 4, "4": 5},
        weight=0.15,
    ),
    Question(
        key="emergency_reserve",
        text="How many months of expenses do you hold outside this portfolio?",
        options={"1": "None", "2": "Up to 3 months",
                 "3": "3 to 6 months", "4": "More than 6 months"},
        scores={"1": 1, "2": 2, "3": 4, "4": 5},
        weight=0.15,
    ),
    Question(
        key="objective",
        text="What is this money mainly for?",
        options={"1": "Preserving what I already have",
                 "2": "Generating income",
                 "3": "Steady long-term growth",
                 "4": "Maximum growth, risk accepted"},
        scores={"1": 1, "2": 2, "3": 4, "4": 5},
        weight=0.10,
    ),
)

# Weighted score (1.0 to 5.0) mapped to a tier. Upper bounds are inclusive and
# the first matching band wins, so the table is read top to bottom.
SCORE_BANDS: tuple[tuple[float, int], ...] = (
    (1.8, 1),
    (2.6, 2),
    (3.4, 3),
    (4.2, 4),
    (5.0, 5),
)

# Suitability rule: a short horizon caps equity regardless of stated tolerance.
# Each row is (maximum horizon in years, equity ceiling); the first match wins.
# This is the rule that can override the questionnaire score downwards in F1.
HORIZON_EQUITY_CAPS: tuple[tuple[int, float], ...] = (
    (3, 0.30),
    (5, 0.50),
    (10, 0.70),
)

# ---------------------------------------------------------------------------
# F2 optimisation defaults
# ---------------------------------------------------------------------------
DEFAULT_LOOKBACK_YEARS: int = 5
# Per-asset ceiling. Left unbounded, mean-variance optimisation on a small
# universe puts everything into the two sleeves the lookback happened to
# flatter; the bound is what keeps the result executable rather than a corner
# solution. F2 reports how hard the bound bound, rather than hiding it.
DEFAULT_WEIGHT_BOUNDS: tuple[float, float] = (0.0, 0.35)
RISK_FREE_RATE: float = 0.042      # illustrative; US 3-month bill, late 2025

# ---------------------------------------------------------------------------
# F3 projection assumptions  (ILLUSTRATIVE - see the module docstring)
# ---------------------------------------------------------------------------
ASSET_CLASS_ASSUMPTIONS: dict[str, dict[str, float]] = {
    # Nominal annual arithmetic mean return, and annual volatility.
    "equity": {"expected_return": 0.070, "volatility": 0.160},
    "bonds": {"expected_return": 0.035, "volatility": 0.055},
}
EQUITY_BOND_CORRELATION: float = 0.15
DEFAULT_MONTE_CARLO_PATHS: int = 10_000
SUCCESS_PROBABILITY_TARGET: float = 0.80
DEFAULT_GLIDEPATH_START_EQUITY: float = 0.90
DEFAULT_GLIDEPATH_END_EQUITY: float = 0.30

# ---------------------------------------------------------------------------
# F4 reporting defaults
# ---------------------------------------------------------------------------
DEFAULT_BENCHMARK: str = "SPY"
TRADING_DAYS_PER_YEAR: int = 252
VALUE_AT_RISK_CONFIDENCE: float = 0.95

# ---------------------------------------------------------------------------
# Data retrieval
# ---------------------------------------------------------------------------
# The bundled snapshot is what makes the artefact run with the network
# disabled, which is the mitigation for the largest run-time risk in the plan.
SNAPSHOT_FILE: Path = DATA_DIR / "price_snapshot.csv"
SNAPSHOT_METADATA_FILE: Path = DATA_DIR / "snapshot_metadata.json"


def model_portfolio(equity_weight: float) -> dict[str, float]:
    """Build sleeve weights for any equity/bond split.

    Lives in config rather than in F1 because F3's glide path needs the same
    mapping at every point along its path. Duplicating it would let the two
    functions drift apart, and a glide path that disagreed with the model
    portfolio ladder would make the whole pipeline incoherent.

    Args:
        equity_weight: Target equity proportion, 0.0 to 1.0.

    Returns:
        Ticker to weight, summing to 1.0, with zero-weight sleeves dropped.
    """
    bond_weight = 1.0 - equity_weight
    weights = {t: w * equity_weight for t, w in EQUITY_MIX.items()}
    weights.update({t: w * bond_weight for t, w in BOND_MIX.items()})
    return {t: w for t, w in weights.items() if w > 1e-9}
