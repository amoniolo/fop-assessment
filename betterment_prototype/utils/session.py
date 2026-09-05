"""In-memory state carried between the four functions.

This module is what makes the artefact a pipeline rather than four scripts.
F2 picks up F1's strategic allocation, F3 picks up F1's tier and amount, and
F4 picks up F2's optimised weights -- but each function also runs standalone,
because every field here is optional and every function falls back to asking
for what it needs.

A frozen dataclass would be wrong here: session state is mutated by design as
the client moves down the pipeline. What is frozen instead are the *records*
below, which are snapshots of a completed stage and should never be edited
after the stage that produced them has finished.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class RiskProfileResult:
    """The completed output of F1. Frozen: a finished profile is a record."""

    tier_level: int
    tier_label: str
    raw_score: float
    equity_weight: float
    weights: dict[str, float]
    investable_amount: float
    horizon_years: int
    horizon_capped: bool
    binding_factor: str
    completed_at: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class OptimisationResult:
    """The completed output of F2. Frozen for the same reason."""

    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe: float
    share_counts: dict[str, int]
    leftover_cash: float
    amount: float
    objective: str
    lookback_years: int
    weight_bounds: tuple[float, float]
    completed_at: datetime = field(default_factory=datetime.now)


@dataclass
class Session:
    """Mutable state passed to every function through the dispatch table.

    Each function reads what it can use and writes back what it produced. A
    function never reaches into another function's module, which is what keeps
    the four independently testable despite being chained.
    """

    risk_profile: RiskProfileResult | None = None
    optimisation: OptimisationResult | None = None
    data_source_note: str = ""
    functions_run: list[str] = field(default_factory=list)

    def record_run(self, name: str) -> None:
        """Note that a function completed, for the exit summary."""
        if name not in self.functions_run:
            self.functions_run.append(name)

    def suggested_weights(self) -> tuple[dict[str, float], str] | None:
        """Return the most recent weights available, and where they came from.

        Preference order is F2 then F1: an optimised, executable portfolio is a
        better default for a performance report than a strategic target that
        was never traded. Returns None when neither stage has run, which is the
        signal for the calling function to ask for weights directly.
        """
        if self.optimisation is not None:
            return self.optimisation.weights, "the optimised portfolio from Function 2"
        if self.risk_profile is not None:
            return self.risk_profile.weights, "the strategic allocation from Function 1"
        return None

    def suggested_amount(self) -> float | None:
        """Return the investable amount captured earlier in the pipeline."""
        if self.optimisation is not None:
            return self.optimisation.amount
        if self.risk_profile is not None:
            return self.risk_profile.investable_amount
        return None
