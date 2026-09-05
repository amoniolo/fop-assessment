"""Tests for the calculation logic behind the four functions.

Run from the package directory:

    python -m pytest tests -q          (if pytest is installed)
    python tests/test_prototype.py     (no pytest needed)

The tests deliberately cover the *pure* functions -- scoring, tier mapping,
glide-path construction, the simulation, and the metric calculations -- rather
than the interactive prompts. Those functions are where a wrong answer would be
silent, whereas a broken prompt is obvious the moment the program is run.

Every test that touches prices uses the bundled snapshot, so the suite runs
with the network disabled.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from functions import f1_risk_profile as f1  # noqa: E402
from functions import f3_goal_projection as f3  # noqa: E402
from functions import f4_performance as f4  # noqa: E402
from utils import data  # noqa: E402
from utils.session import Session  # noqa: E402


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_model_portfolio_weights_sum_to_one():
    for equity in (0.0, 0.2, 0.5, 0.9, 1.0):
        weights = config.model_portfolio(equity)
        assert abs(sum(weights.values()) - 1.0) < 1e-9, equity


def test_model_portfolio_respects_the_equity_split():
    weights = config.model_portfolio(0.60)
    equity = sum(w for t, w in weights.items() if config.SLEEVES[t].is_equity)
    assert abs(equity - 0.60) < 1e-9


def test_sleeve_mixes_are_internally_normalised():
    assert abs(sum(config.EQUITY_MIX.values()) - 1.0) < 1e-9
    assert abs(sum(config.BOND_MIX.values()) - 1.0) < 1e-9


def test_questionnaire_weights_sum_to_one():
    assert abs(sum(q.weight for q in config.QUESTIONNAIRE) - 1.0) < 1e-9


def test_every_question_scores_every_option():
    for question in config.QUESTIONNAIRE:
        assert set(question.options) == set(question.scores), question.key


# ---------------------------------------------------------------------------
# F1
# ---------------------------------------------------------------------------


def _responses(horizon: str, drawdown: str, income: str,
               reserve: str, objective: str) -> dict[str, str]:
    return {
        "horizon": horizon, "drawdown": drawdown,
        "income_stability": income, "emergency_reserve": reserve,
        "objective": objective,
    }


def test_scoring_is_bounded_by_the_answer_scale():
    lowest = f1.score_responses(_responses("1", "1", "1", "1", "1"))
    highest = f1.score_responses(_responses("5", "5", "4", "4", "4"))
    assert abs(lowest - 1.0) < 1e-9
    assert abs(highest - 5.0) < 1e-9


def test_scoring_is_monotonic_in_a_single_answer():
    """Raising one answer must never lower the score."""
    previous = -1.0
    for drawdown in ("1", "2", "3", "4", "5"):
        score = f1.score_responses(_responses("3", drawdown, "3", "3", "3"))
        assert score > previous
        previous = score


def test_long_horizon_leaves_the_cap_unbound():
    """A 25-year horizon should not trigger any equity ceiling."""
    score = f1.score_responses(_responses("5", "5", "4", "4", "4"))
    tier = f1.map_score_to_tier(score, horizon_years=25)
    uncapped = f1.map_score_to_tier(score, horizon_years=99)
    assert tier.level == uncapped.level == 5


def test_short_horizon_overrides_a_high_risk_score():
    """The suitability rule is the point of F1, so it gets its own test.

    A client answering at maximum risk tolerance but needing the money in two
    years must not be given the aggressive portfolio.
    """
    score = f1.score_responses(_responses("1", "5", "4", "4", "4"))
    tier = f1.map_score_to_tier(score, horizon_years=2)
    assert tier.equity_weight <= 0.30
    uncapped = f1.map_score_to_tier(score, horizon_years=99)
    assert tier.equity_weight < uncapped.equity_weight


def test_cap_steps_down_to_a_real_tier():
    """The capped result must be a tier on the ladder, not an off-ladder blend."""
    score = f1.score_responses(_responses("1", "5", "4", "4", "4"))
    tier = f1.map_score_to_tier(score, horizon_years=2)
    assert tier in config.RISK_TIERS.values()


def test_every_score_maps_to_a_tier():
    for step in range(100, 501):
        tier = f1.map_score_to_tier(step / 100.0, horizon_years=99)
        assert tier.level in config.RISK_TIERS


def test_strategic_allocation_matches_the_tier():
    for tier in config.RISK_TIERS.values():
        weights = f1.strategic_allocation(tier)
        equity = sum(w for t, w in weights.items()
                     if config.SLEEVES[t].is_equity)
        assert abs(equity - tier.equity_weight) < 1e-9


# ---------------------------------------------------------------------------
# F3
# ---------------------------------------------------------------------------


def test_glidepath_has_one_weight_per_month():
    path = f3.build_glidepath(years=20, start_equity=0.9, end_equity=0.3)
    assert len(path) == 240


def test_glidepath_runs_from_start_to_end_and_never_rises():
    path = f3.build_glidepath(years=10, start_equity=0.9, end_equity=0.3)
    assert abs(path[0] - 0.9) < 1e-9
    assert abs(path[-1] - 0.3) < 1e-9
    assert np.all(np.diff(path) <= 1e-12)


def test_flat_glidepath_is_constant():
    path = f3.build_glidepath(years=5, start_equity=0.6, end_equity=0.6)
    assert np.allclose(path, 0.6)


def test_blended_volatility_is_below_the_weighted_average():
    """The diversification benefit is the reason a glide path is worth having.

    If this test ever fails, _portfolio_moments has been changed to combine
    volatilities linearly, and every F3 projection would overstate risk in the
    balanced middle of the path.
    """
    equity_weights = np.array([0.5])
    _, std = f3._portfolio_moments(equity_weights)
    annual_std = float(std[0]) * np.sqrt(12)

    equity_vol = config.ASSET_CLASS_ASSUMPTIONS["equity"]["volatility"]
    bond_vol = config.ASSET_CLASS_ASSUMPTIONS["bonds"]["volatility"]
    linear = 0.5 * equity_vol + 0.5 * bond_vol

    assert annual_std < linear


def test_simulation_is_reproducible_from_the_seed():
    """The report quotes these figures, so the seed must pin them exactly."""
    path = f3.build_glidepath(10, 0.8, 0.4)
    first = f3.simulate(500, config.RANDOM_SEED, path, 10_000.0, 500.0)
    second = f3.simulate(500, config.RANDOM_SEED, path, 10_000.0, 500.0)
    assert np.array_equal(first, second)


def test_different_seeds_give_different_paths():
    path = f3.build_glidepath(10, 0.8, 0.4)
    first = f3.simulate(500, 1, path, 10_000.0, 500.0)
    second = f3.simulate(500, 2, path, 10_000.0, 500.0)
    assert not np.array_equal(first, second)


def test_simulation_returns_one_result_per_path():
    path = f3.build_glidepath(5, 0.7, 0.4)
    result = f3.simulate(321, config.RANDOM_SEED, path, 1_000.0, 100.0)
    assert result.shape == (321,)
    assert np.all(np.isfinite(result))


def test_zero_volatility_reproduces_the_deterministic_answer(monkeypatch=None):
    """With volatility removed, the simulation must equal npf.fv exactly.

    This is the strongest available check that the simulation's compounding and
    contribution timing agree with the deterministic anchor. If they disagree,
    the headline insight of F3 -- the gap between the two -- would be measuring
    a bookkeeping difference rather than the effect of volatility.
    """
    import numpy_financial as npf

    original = config.ASSET_CLASS_ASSUMPTIONS
    config.ASSET_CLASS_ASSUMPTIONS = {
        "equity": {"expected_return": 0.06, "volatility": 0.0},
        "bonds": {"expected_return": 0.06, "volatility": 0.0},
    }
    try:
        path = f3.build_glidepath(10, 0.6, 0.6)
        simulated = f3.simulate(10, 1, path, 25_000.0, 750.0)
        monthly_rate = f3._mean_monthly_return(path)
        expected = float(npf.fv(rate=monthly_rate, nper=len(path),
                                pmt=-750.0, pv=-25_000.0))
        assert abs(float(simulated[0]) - expected) < 0.01
    finally:
        config.ASSET_CLASS_ASSUMPTIONS = original


def test_more_contribution_never_lowers_success_probability():
    """Monotonicity is the precondition for the bisection search to be valid."""
    path = f3.build_glidepath(10, 0.7, 0.4)
    target = 300_000.0
    previous = -1.0
    for contribution in (0.0, 500.0, 1_000.0, 2_000.0, 4_000.0):
        wealth = f3.simulate(1_000, config.RANDOM_SEED, path, 20_000.0,
                             contribution)
        probability = float((wealth >= target).mean())
        assert probability >= previous - 1e-12
        previous = probability


def test_contribution_solver_reaches_the_probability_target():
    path = f3.build_glidepath(15, 0.8, 0.4)
    solved = f3.solve_contribution_for_probability(
        glidepath=path, current_balance=20_000.0,
        target_amount=400_000.0,
        probability_target=config.SUCCESS_PROBABILITY_TARGET,
    )
    assert solved is not None
    wealth = f3.simulate(2_000, config.RANDOM_SEED, path, 20_000.0, solved)
    probability = float((wealth >= 400_000.0).mean())
    assert probability >= config.SUCCESS_PROBABILITY_TARGET - 0.02


def test_unreachable_target_returns_none():
    path = f3.build_glidepath(1, 0.6, 0.6)
    solved = f3.solve_contribution_for_probability(
        glidepath=path, current_balance=0.0,
        target_amount=500_000_000.0, probability_target=0.80,
    )
    assert solved is None


# ---------------------------------------------------------------------------
# F4
# ---------------------------------------------------------------------------


def _known_returns() -> pd.Series:
    """A deterministic return series with a known shape."""
    index = pd.bdate_range("2020-01-01", periods=504)
    values = np.full(504, 0.0004)
    values[100:140] = -0.004          # an engineered drawdown
    return pd.Series(values, index=index)


def test_cagr_matches_a_hand_calculation():
    index = pd.bdate_range("2020-01-01", periods=config.TRADING_DAYS_PER_YEAR)
    returns = pd.Series(np.full(len(index), 0.0), index=index)
    assert abs(f4._cagr(returns)) < 1e-12


def test_max_drawdown_is_negative_and_finds_the_engineered_fall():
    returns = _known_returns()
    drawdown = f4._max_drawdown(returns)
    assert drawdown < 0
    # 40 days at -0.4% compounds to roughly -14.8%.
    assert -0.20 < drawdown < -0.10


def test_drawdown_dates_bracket_the_engineered_fall():
    returns = _known_returns()
    dates = f4._drawdown_dates(returns)
    assert dates["drawdown_peak"] < dates["drawdown_trough"]
    assert dates["drawdown_trough"] == returns.index[139]


def test_tracking_error_is_zero_against_itself():
    returns = _known_returns()
    assert abs(f4.tracking_error(returns, returns)) < 1e-12


def test_tracking_error_is_positive_for_different_series():
    a = _known_returns()
    b = a.copy()
    b.iloc[::2] += 0.001
    assert f4.tracking_error(a, b) > 0


def test_sortino_exceeds_sharpe_when_volatility_is_mostly_upside():
    """Sortino should be the more generous of the two on an upside-skewed series.

    Sharpe penalises upside volatility as though it were risk; Sortino does
    not. A series whose swings are mostly upward should therefore score better
    on Sortino, and if it ever does not, the downside filter is wrong.
    """
    index = pd.bdate_range("2020-01-01", periods=504)
    values = np.full(504, 0.0008)
    values[::10] = 0.02              # frequent large upside
    # Varied downside, so the test does not depend on a degenerate spread.
    rng = np.random.default_rng(7)
    downside_days = np.arange(5, 504, 25)
    values[downside_days] = -rng.uniform(0.001, 0.006, size=len(downside_days))
    returns = pd.Series(values, index=index)
    assert f4._sortino(returns) > f4._sharpe(returns)


def test_sortino_is_finite_when_every_loss_is_the_same_size():
    """Uniform losses must not produce an infinitely good Sortino.

    Taking the standard deviation *of the losing periods* would give a zero
    denominator here and score a consistently losing portfolio as perfect. The
    root-mean-square shortfall used in _sortino does not have that failure.
    """
    index = pd.bdate_range("2020-01-01", periods=504)
    values = np.full(504, 0.0008)
    values[5::50] = -0.002           # identical losses, zero spread
    returns = pd.Series(values, index=index)
    result = f4._sortino(returns)
    assert np.isfinite(result)


def test_sortino_penalises_more_frequent_losses():
    """Downside deviation must respond to how often the portfolio fell short."""
    index = pd.bdate_range("2020-01-01", periods=504)

    rare = np.full(504, 0.0008)
    rare[::100] = -0.004
    frequent = np.full(504, 0.0008)
    frequent[::20] = -0.004

    assert f4._sortino(pd.Series(rare, index=index)) > \
        f4._sortino(pd.Series(frequent, index=index))


def test_metrics_are_computed_and_named():
    returns = _known_returns()
    metrics, engine = f4.compute_metrics(returns)
    for key in ("cagr", "volatility", "sharpe", "sortino",
                "max_drawdown", "value_at_risk"):
        assert key in metrics
    assert isinstance(engine, str) and engine


def test_engine_reports_total_fallback_when_quantstats_is_absent():
    """A missing QuantStats must not be reported as a partial fallback.

    The engine line exists so the reader knows which code produced the figures
    on screen, so it has to be exactly right when the library is unavailable.
    The count it compares against must cover only the metrics actually routed
    through QuantStats -- the drawdown dates never are, and including them
    would make a total failure read as partial.
    """
    import builtins

    real_import = builtins.__import__

    def no_quantstats(name, *args, **kwargs):
        if name == "quantstats":
            raise ImportError("simulated: quantstats is not installed")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = no_quantstats
    try:
        _, engine = f4.compute_metrics(_known_returns())
    finally:
        builtins.__import__ = real_import

    assert "unavailable" in engine, engine
    assert "with internal fallback for" not in engine, engine


def test_engine_reports_quantstats_when_it_works():
    _, engine = f4.compute_metrics(_known_returns())
    assert engine == "QuantStats", engine


def test_portfolio_returns_renormalise_over_available_holdings():
    index = pd.bdate_range("2024-01-01", periods=30)
    prices = pd.DataFrame(
        {"AAA": np.linspace(100, 110, 30), "BBB": np.linspace(50, 55, 30)},
        index=index,
    )
    weights = {"AAA": 0.25, "BBB": 0.25, "CCC": 0.50}   # CCC has no data
    returns = f4.build_portfolio_returns(prices, weights, ["AAA", "BBB"])
    assert len(returns) == 29
    assert np.all(np.isfinite(returns))


# ---------------------------------------------------------------------------
# Data module and offline fallback
# ---------------------------------------------------------------------------


def test_snapshot_is_bundled():
    """Section 8.2 requires the artefact to run with the network disabled."""
    assert data.snapshot_available(), (
        "data/price_snapshot.csv is missing. Run tools/build_snapshot.py "
        "with a connection before submitting."
    )


def test_snapshot_covers_every_sleeve_and_the_benchmark():
    frame = pd.read_csv(config.SNAPSHOT_FILE, index_col=0, parse_dates=True)
    required = set(config.SLEEVES) | {config.DEFAULT_BENCHMARK}
    assert required.issubset(set(frame.columns)), required - set(frame.columns)


def test_snapshot_loads_and_slices_to_a_window():
    frame, source = data._load_snapshot(
        ["VTI", "BND"], date(2022, 1, 1), date(2023, 1, 1)
    )
    assert frame is not None
    assert list(frame.columns) == ["VTI", "BND"]
    assert len(frame) > 200
    assert "snapshot" in source


def test_clean_drops_a_column_with_almost_no_history():
    index = pd.bdate_range("2024-01-01", periods=100)
    frame = pd.DataFrame(
        {"GOOD": np.linspace(100, 110, 100),
         "SPARSE": [1.0] * 5 + [np.nan] * 95},
        index=index,
    )
    cleaned = data._clean(frame, ["GOOD", "SPARSE"])
    assert cleaned is not None
    assert list(cleaned.columns) == ["GOOD"]


def test_missing_ticker_in_snapshot_is_reported_not_crashed():
    frame, source = data._load_snapshot(
        ["NOT_A_REAL_TICKER"], date(2022, 1, 1), date(2023, 1, 1)
    )
    assert frame is None
    assert "does not cover" in source


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def test_empty_session_suggests_nothing():
    session = Session()
    assert session.suggested_weights() is None
    assert session.suggested_amount() is None


def test_session_prefers_the_optimised_portfolio_over_the_strategic_one():
    from utils.session import OptimisationResult, RiskProfileResult

    session = Session()
    session.risk_profile = RiskProfileResult(
        tier_level=3, tier_label="Balanced", raw_score=3.0,
        equity_weight=0.6, weights={"VTI": 1.0}, investable_amount=1000.0,
        horizon_years=10, horizon_capped=False, binding_factor="x",
    )
    assert session.suggested_weights()[0] == {"VTI": 1.0}

    session.optimisation = OptimisationResult(
        weights={"BND": 1.0}, expected_return=0.05, volatility=0.05,
        sharpe=0.2, share_counts={"BND": 10}, leftover_cash=5.0,
        amount=2000.0, objective="max sharpe", lookback_years=5,
        weight_bounds=(0.0, 0.35),
    )
    assert session.suggested_weights()[0] == {"BND": 1.0}
    assert session.suggested_amount() == 2000.0


def test_record_run_does_not_duplicate():
    session = Session()
    session.record_run("F1 Risk profile")
    session.record_run("F1 Risk profile")
    assert session.functions_run == ["F1 Risk profile"]


# ---------------------------------------------------------------------------
# Menu structure
# ---------------------------------------------------------------------------


def test_menu_has_four_functions_and_an_exit():
    import main

    handlers = [e for e in main.MENU.values() if e.handler is not None]
    assert len(handlers) == 4
    assert main.MENU["0"].handler is None


def test_every_menu_handler_is_callable():
    import main

    for key, entry in main.MENU.items():
        if entry.handler is not None:
            assert callable(entry.handler), key


def test_menu_labels_map_to_recorded_names():
    """The exit summary depends on this mapping staying in step with MENU."""
    import main

    for entry in main.MENU.values():
        if entry.handler is not None:
            assert main._label_of(entry).startswith("F"), entry.label


# ---------------------------------------------------------------------------
# Runner, so the suite works without pytest installed
# ---------------------------------------------------------------------------


def _run_all() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures: list[tuple[str, str]] = []

    for name, test in tests:
        try:
            test()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures.append((name, f"AssertionError: {exc}"))
            print(f"  FAIL  {name}")
        except Exception as exc:  # noqa: BLE001 - report, do not stop the suite
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  ERROR {name}")

    print(f"\n{len(tests) - len(failures)} passed, {len(failures)} failed, "
          f"{len(tests)} total")
    for name, reason in failures:
        print(f"\n  {name}\n    {reason}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
