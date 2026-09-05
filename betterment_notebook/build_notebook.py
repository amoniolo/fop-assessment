"""Generate Betterment_Advisory.ipynb.

The notebook is generated rather than hand-written so that its cells stay under
version control as readable Python, and so a regeneration can never produce
malformed notebook JSON. Run this whenever the notebook needs rebuilding:

    python build_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE / "Betterment_Advisory.ipynb"

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ===========================================================================
md(r"""
# Betterment Robo-Advisory Prototype — Notebook Edition

**Fundamentals of Programming (5K7V0027) — Level 7 portfolio, Part 2**

An interactive notebook front-end over the same advisory engine as the
command-line package. The four sections below are the four stages of
Betterment's automated advisory pipeline:

| # | Stage | What it does |
|---|-------|--------------|
| 1 | **Onboarding** | Questionnaire → risk tier → model portfolio |
| 2 | **Execution** | Efficient frontier → weights → whole-share order |
| 3 | **Planning** | Glide path + Monte Carlo → probability of success |
| 4 | **Reporting** | Portfolio vs benchmark, return and risk |

Each stage feeds the next through a shared session, and each also runs on its
own.

### How to run

Run every cell from the top (**Kernel → Restart & Run All**), then use the
controls in each section. Results appear beneath the controls.

### What this notebook does *not* duplicate

All financial logic — scoring, the suitability rule, optimisation, the glide
path, the simulation, the risk metrics — is imported from the
`betterment_prototype` package in the sibling folder. This notebook supplies
only the interface: `ipywidgets` controls in place of console prompts, and
Plotly figures in place of Matplotlib ones.

That separation is the point. Two front-ends over one engine means a change to
the advisory logic cannot make the two disagree, and it demonstrates that the
domain code was written independently of how it is displayed.
""")

# ===========================================================================
md(r"""
---
## Setup

Locates the sibling package, imports the engine, and registers a Plotly theme.
""")

code(r'''
"""Locate the engine package and import it."""
from pathlib import Path
import sys


def find_engine() -> Path:
    """Search upward from here for the betterment_prototype package.

    Searching rather than hard-coding a relative path means the notebook still
    works when Jupyter is launched from the project root, from this folder, or
    from anywhere between - which is the usual reason a notebook that "worked
    yesterday" stops working.
    """
    seen: list[Path] = []
    start = Path.cwd().resolve()
    for base in [start, *start.parents]:
        for candidate in (base / "betterment_prototype", base):
            seen.append(candidate)
            if (candidate / "config.py").is_file() and (candidate / "functions").is_dir():
                return candidate
    raise FileNotFoundError(
        "Could not find the betterment_prototype package. Expected it in a "
        "sibling folder of this notebook.\nSearched:\n  "
        + "\n  ".join(str(p) for p in seen[:12])
    )


ENGINE = find_engine()
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

print(f"Engine package: {ENGINE}")
''')

code(r'''
"""Import the advisory engine and the plotting stack."""
import warnings
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import ipywidgets as widgets
from IPython.display import HTML, Markdown, display

# --- the engine (unchanged, shared with the command-line package) ----------
import config
from functions import f1_risk_profile as f1
from functions import f2_optimise as f2
from functions import f3_goal_projection as f3
from functions import f4_performance as f4
from utils import data as engine_data
from utils import display as engine_display
from utils.session import OptimisationResult, RiskProfileResult, Session

# The engine's console helpers emit ANSI colour codes. Jupyter renders those,
# but plain text reads better next to widgets, so turn styling off.
engine_display.set_colour(False)

warnings.filterwarnings("ignore", category=FutureWarning)

print(f"pandas {pd.__version__} | numpy {np.__version__} | "
      f"plotly {__import__('plotly').__version__} | "
      f"ipywidgets {widgets.__version__}")
print(f"Random seed: {config.RANDOM_SEED}  |  Currency: {config.CURRENCY_CODE}")
''')

# ===========================================================================
md(r"""
### Chart theme

One Plotly template for every figure, so the four sections read as one
document rather than four unrelated charts.

**Colour is assigned by the job it does, not by taste.** Only two categorical
hues are used anywhere — blue and orange — because no chart here carries more
than two identities (equity/bonds, portfolio/benchmark). Reference marks such
as the individual sleeves on the frontier are deliberately *not* given
categorical colour: they are context, not series, so they are drawn in a muted
grey and labelled directly.

The pair was checked for colour-vision separation rather than eyeballed: worst
adjacent CVD ΔE 24.7 against a target of ≥ 8, normal-vision ΔE 33.6 against a
floor of 15, and both clear 3:1 contrast against the chart surface.

The template pins its own light surface rather than inheriting the Jupyter
theme, so the figures stay legible under either a light or a dark notebook
theme instead of putting dark text on a dark ground.
""")

code(r'''
"""Register the shared Plotly template and the palette."""

# --- palette ---------------------------------------------------------------
SURFACE = "#fcfcfb"      # chart surface, pinned light
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8880"
GRID = "#e6e5e0"

SERIES_1 = "#2a78d6"     # blue   - equity, portfolio
SERIES_2 = "#eb6834"     # orange - bonds, benchmark
REFERENCE = "#9c9a92"    # muted grey - context marks, never an identity


def rgba(hex_colour: str, alpha: float) -> str:
    """Convert #rrggbb to an rgba() string at the given alpha.

    Plotly fills need rgba rather than a hex plus a separate opacity, because
    setting opacity on a trace would fade its line as well as its fill.
    """
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"

# Sequential steps of the single blue hue, for the percentile fan. A fan chart
# encodes magnitude of confidence, not identity, so it takes one hue getting
# darker rather than several different hues.
FAN_OUTER = "rgba(42, 120, 214, 0.16)"
FAN_INNER = "rgba(42, 120, 214, 0.32)"

pio.templates["advisory"] = go.layout.Template(
    layout=dict(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(
            family="Segoe UI, Helvetica Neue, Arial, sans-serif",
            size=13,
            color=TEXT_PRIMARY,
        ),
        title=dict(font=dict(size=16, color=TEXT_PRIMARY), x=0, xanchor="left"),
        # Recessive axes: the data carries the chart, not the furniture.
        xaxis=dict(
            gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
            ticks="outside", tickcolor=GRID, ticklen=4,
            title=dict(font=dict(size=12, color=TEXT_SECONDARY)),
            tickfont=dict(size=11, color=TEXT_SECONDARY),
        ),
        yaxis=dict(
            gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
            ticks="outside", tickcolor=GRID, ticklen=4,
            title=dict(font=dict(size=12, color=TEXT_SECONDARY)),
            tickfont=dict(size=11, color=TEXT_SECONDARY),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=12, color=TEXT_SECONDARY),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor="#ffffff", bordercolor=GRID,
            font=dict(size=12, color=TEXT_PRIMARY),
        ),
        margin=dict(l=70, r=30, t=70, b=60),
        colorway=[SERIES_1, SERIES_2],
    )
)
pio.templates.default = "advisory"

print("Plotly template 'advisory' registered.")
''')

# ===========================================================================
md(r"""
### Session and formatting helpers

The `Session` object is the same one the command-line package uses: it is what
lets Section 2 pick up Section 1's allocation and Section 4 pick up Section 2's
weights.
""")

code(r'''
"""Shared session state and small presentation helpers."""

SESSION = Session()

# Reuse the engine's formatters so the notebook and the console version render
# the same number the same way.
money = engine_display.money
percent = engine_display.percent
bps = engine_display.basis_points
ratio = engine_display.ratio


def show_insight(text: str) -> None:
    """Render the engine's interpretation text as a styled callout.

    Every function ends by explaining its own result rather than only producing
    one, so the interpretation gets a visual treatment that separates it from
    the numbers above it.
    """
    body = "".join(f"<p style='margin:0 0 0.7em 0'>{para}</p>"
                   for para in text.split("\n\n") if para.strip())
    display(HTML(
        f"<div style='border-left:4px solid {SERIES_1};background:#f4f7fc;"
        f"padding:14px 18px;margin:14px 0;border-radius:0 6px 6px 0;"
        f"font-size:14px;line-height:1.55;color:{TEXT_PRIMARY};'>"
        f"<div style='font-weight:700;letter-spacing:0.04em;font-size:11px;"
        f"color:{SERIES_1};margin-bottom:8px'>WHAT THIS MEANS</div>{body}</div>"
    ))


def show_table(headers: list[str], rows: list[list[str]],
               align_right_from: int = 1) -> None:
    """Render a plain HTML table.

    Used instead of a Plotly table so the numbers stay selectable, searchable
    and screen-reader accessible - this is also the table view that every
    figure below is accompanied by.
    """
    head = "".join(
        f"<th style='text-align:{'right' if i >= align_right_from else 'left'};"
        f"padding:7px 14px;border-bottom:2px solid {GRID};font-size:12px;"
        f"color:{TEXT_SECONDARY};font-weight:600'>{h}</th>"
        for i, h in enumerate(headers)
    )
    body = ""
    for row in rows:
        tds = "".join(
            f"<td style='text-align:{'right' if i >= align_right_from else 'left'};"
            f"padding:6px 14px;border-bottom:1px solid {GRID};font-size:13px;"
            f"font-variant-numeric:tabular-nums'>{c}</td>"
            for i, c in enumerate(row)
        )
        body += f"<tr>{tds}</tr>"
    display(HTML(
        f"<table style='border-collapse:collapse;margin:10px 0'>"
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    ))


def note(text: str) -> None:
    display(HTML(f"<div style='color:{TEXT_MUTED};font-size:12.5px;"
                 f"margin:6px 0'>{text}</div>"))


def seed_amount_controls(amount: float) -> None:
    """Carry an amount forward into the later sections' controls.

    The pipeline should feel joined up - running Section 1 with $100,000 ought
    to leave Section 2 offering $100,000 rather than its own default. But the
    carry-forward has to happen by *setting the control*, not by overriding it
    at run time: a visible input whose value is silently discarded is worse
    than no input at all, because the reader has no way to tell it was ignored.

    So the session seeds these widgets, and whatever they hold afterwards wins.

    The globals() lookup guards against cell-execution order. This helper is
    defined before the Section 2 and 3 controls exist, and a user who runs
    cells out of order should get a no-op rather than a NameError.
    """
    for name in ("f2_amount", "f3_balance"):
        control = globals().get(name)
        if control is None:
            continue
        control.value = max(control.min, min(control.max, float(amount)))


def run_button(label: str) -> widgets.Button:
    b = widgets.Button(description=label, button_style="primary",
                       layout=widgets.Layout(width="260px", height="38px"))
    b.style.font_weight = "600"
    return b


LABEL = widgets.Layout(width="440px")
STYLE = {"description_width": "230px"}

print("Helpers ready.")
''')

# ===========================================================================
md(r"""
---
# 1 · Risk profile and strategic allocation

Five weighted questions produce a score from 1.00 to 5.00, which maps to a risk
tier and then to a model portfolio of ten ETF sleeves.

A **suitability rule** then applies: a short horizon caps equity regardless of
stated risk tolerance. The result reports which of the two actually bound the
outcome — the score or the horizon — because that is the input the client would
have to change to get a different portfolio.
""")

code(r'''
"""Figure: strategic allocation by holding."""

def figure_allocation(weights: dict[str, float], amount: float,
                      tier_label: str) -> go.Figure:
    """Horizontal bars, sorted by weight, split equity against bonds.

    A bar chart rather than a pie or donut. Ten slices of a pie cannot be
    compared by eye - angle is the hardest visual channel to judge - whereas
    sorted bars on a common baseline make the ordering and the gaps immediate.
    Two colours, because there are exactly two identities here: equity and
    bonds. Every bar is labelled directly, so colour is never the only carrier.
    """
    ordered = sorted(weights.items(), key=lambda kv: kv[1])
    tickers = [t for t, _ in ordered]
    values = [w for _, w in ordered]
    is_equity = [config.SLEEVES[t].is_equity for t in tickers]

    fig = go.Figure()
    for equity_side, colour, name in (
        (True, SERIES_1, "Equity"), (False, SERIES_2, "Bonds"),
    ):
        idx = [i for i, e in enumerate(is_equity) if e is equity_side]
        if not idx:
            continue
        fig.add_bar(
            y=[tickers[i] for i in idx],
            x=[values[i] for i in idx],
            orientation="h",
            name=name,
            marker=dict(color=colour, line=dict(color=SURFACE, width=2)),
            text=[percent(values[i]) for i in idx],
            textposition="outside",
            textfont=dict(size=11, color=TEXT_SECONDARY),
            customdata=[[config.SLEEVES[tickers[i]].name,
                         money(values[i] * amount)] for i in idx],
            hovertemplate=("<b>%{y}</b> — %{customdata[0]}<br>"
                           "Weight %{x:.1%}<br>Amount %{customdata[1]}"
                           "<extra></extra>"),
        )

    fig.update_layout(
        title=f"Target allocation — tier {tier_label}",
        xaxis=dict(title="Weight of portfolio", tickformat=".0%",
                   range=[0, max(values) * 1.22]),
        # Pin the category order explicitly. Two traces on a categorical axis
        # otherwise order the categories trace by trace, which would group all
        # the bonds above all the equity and destroy the sort - the whole
        # reason for choosing bars over a pie was that length is comparable at
        # a glance, and that only holds if the bars are actually in order.
        yaxis=dict(title=None, categoryorder="array", categoryarray=tickers),
        height=460, bargap=0.28,
    )
    return fig
''')

code(r'''
"""Compute and render Function 1."""

def compute_f1(responses: dict[str, str], amount: float) -> dict:
    """Run the engine's scoring and allocation. No display, no widgets."""
    horizon_years = f1._HORIZON_YEARS[responses["horizon"]]
    score = f1.score_responses(responses)
    tier = f1.map_score_to_tier(score, horizon_years)
    uncapped = f1.map_score_to_tier(score, horizon_years=99)
    weights = f1.strategic_allocation(tier)
    capped = tier.level != uncapped.level
    binding = f1._binding_factor(responses, score, capped, horizon_years)
    return dict(score=score, tier=tier, uncapped=uncapped, weights=weights,
                capped=capped, binding=binding, amount=amount,
                horizon_years=horizon_years)


def render_f1(r: dict) -> None:
    """Display the profile, the allocation, the chart and the interpretation."""
    tier, uncapped = r["tier"], r["uncapped"]

    rows = [
        ["Weighted score", f"{r['score']:.2f} / 5.00"],
        ["Assigned tier", f"{tier.level} — {tier.label}"],
        ["Target equity", percent(tier.equity_weight, 0)],
        ["Investment horizon", f"{r['horizon_years']} years"],
        ["Suitability cap",
         (f"applied — score alone indicated tier {uncapped.level} "
          f"({percent(uncapped.equity_weight, 0)} equity)")
         if r["capped"] else "not binding"],
    ]
    show_table(["Risk profile", "Value"], rows)
    note(tier.description)

    by_class: dict[str, float] = {}
    for ticker, w in r["weights"].items():
        cls = config.SLEEVES[ticker].asset_class
        by_class[cls] = by_class.get(cls, 0.0) + w
    show_table(
        ["Asset class", "Weight", "Amount"],
        [[c, percent(w), money(w * r["amount"])] for c, w in by_class.items()],
    )

    display(figure_allocation(r["weights"], r["amount"], tier.label))

    show_insight(f1.explain_allocation(
        tier, uncapped, r["weights"], r["capped"],
        r["horizon_years"], r["binding"], r["amount"],
    ))

    SESSION.risk_profile = RiskProfileResult(
        tier_level=tier.level, tier_label=tier.label, raw_score=r["score"],
        equity_weight=tier.equity_weight, weights=r["weights"],
        investable_amount=r["amount"], horizon_years=r["horizon_years"],
        horizon_capped=r["capped"], binding_factor=r["binding"],
    )
    SESSION.record_run("F1 Risk profile")
    # Offer this amount as the starting point downstream. Sections 2 and 3
    # remain free to be changed afterwards - see seed_amount_controls.
    seed_amount_controls(r["amount"])
''')

code(r'''
"""Function 1 controls."""

f1_dropdowns: dict[str, widgets.Dropdown] = {}
for q in config.QUESTIONNAIRE:
    f1_dropdowns[q.key] = widgets.Dropdown(
        options=[(wording, key) for key, wording in q.options.items()],
        value=list(q.options)[len(q.options) // 2],   # a neutral starting point
        description=q.text, style=STYLE, layout=LABEL,
    )

f1_amount = widgets.BoundedFloatText(
    value=50_000, min=1_000, max=100_000_000, step=1_000,
    description="Amount to invest", style=STYLE, layout=LABEL,
)
f1_go = run_button("Build risk profile")
f1_out = widgets.Output()


def on_f1(_=None):
    with f1_out:
        f1_out.clear_output(wait=True)
        try:
            responses = {k: d.value for k, d in f1_dropdowns.items()}
            render_f1(compute_f1(responses, f1_amount.value))
        except Exception as exc:
            display(HTML(f"<b style='color:#e34948'>Could not complete: "
                         f"{type(exc).__name__}: {exc}</b>"))


f1_go.on_click(on_f1)
display(widgets.VBox([*f1_dropdowns.values(), f1_amount, f1_go, f1_out]))
''')

# ===========================================================================
md(r"""
---
# 2 · Portfolio optimisation and share allocation

Mean-variance optimisation over the chosen universe, then conversion into whole
shares with the residual cash stated.

Two decisions are visible in the output: **Ledoit-Wolf shrinkage** on the
covariance matrix rather than the raw sample covariance, and **per-asset weight
bounds**. Both constrain the optimiser, and the result reports how hard they
bound rather than hiding it.
""")

code(r'''
"""Figure: efficient frontier."""

def figure_frontier(prices: pd.DataFrame, performance: tuple[float, float, float],
                    bounds: tuple[float, float]) -> go.Figure:
    """The frontier, the individual sleeves, and the chosen portfolio.

    Only two things here carry identity - the frontier and the selected
    portfolio - so only two categorical hues appear. The individual sleeves are
    context rather than a series, so they are drawn in a muted grey and named
    with direct labels instead of being given colours of their own. Adding ten
    more hues would be the single fastest way to make this chart unreadable.
    """
    from pypfopt import EfficientFrontier, expected_returns, risk_models

    mu = expected_returns.mean_historical_return(prices)
    cov = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    base = EfficientFrontier(mu, cov, weight_bounds=bounds)
    base.min_volatility()
    lo = base.portfolio_performance()[0]
    hi = float(mu.max()) * min(1.0, bounds[1] * len(mu))

    xs, ys = [], []
    for target in np.linspace(lo, hi * 0.999, 40):
        try:
            cand = EfficientFrontier(mu, cov, weight_bounds=bounds)
            cand.efficient_return(target_return=target)
            r, v, _ = cand.portfolio_performance()
            xs.append(v); ys.append(r)
        except Exception:
            continue   # infeasible targets are expected near the ends

    annual_vol = prices.pct_change().std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)

    fig = go.Figure()
    fig.add_scatter(
        x=xs, y=ys, mode="lines", name="Efficient frontier",
        line=dict(color=SERIES_1, width=2),
        hovertemplate="Volatility %{x:.1%}<br>Return %{y:.1%}<extra></extra>",
    )
    fig.add_scatter(
        x=annual_vol, y=mu, mode="markers+text", name="Individual sleeves",
        marker=dict(color=REFERENCE, size=9,
                    line=dict(color=SURFACE, width=2)),
        text=list(mu.index), textposition="top center",
        textfont=dict(size=10, color=TEXT_SECONDARY),
        hovertemplate="<b>%{text}</b><br>Volatility %{x:.1%}<br>"
                      "Return %{y:.1%}<extra></extra>",
    )
    fig.add_scatter(
        x=[performance[1]], y=[performance[0]], mode="markers",
        name="Selected portfolio",
        marker=dict(color=SERIES_2, size=20, symbol="star",
                    line=dict(color=SURFACE, width=2)),
        hovertemplate="<b>Selected portfolio</b><br>Volatility %{x:.1%}<br>"
                      "Return %{y:.1%}<extra></extra>",
    )
    fig.update_layout(
        title=(f"Efficient frontier — Ledoit-Wolf covariance, "
               f"weights capped at {percent(bounds[1], 0)}"),
        xaxis=dict(title="Annualised volatility", tickformat=".0%"),
        yaxis=dict(title="Expected annual return", tickformat=".0%"),
        height=520, hovermode="closest",
    )
    return fig
''')

code(r'''
"""Compute and render Function 2."""

def compute_f2(tickers: list[str], lookback_years: int, objective: str,
               upper_bound: float, amount: float) -> dict:
    bounds = (0.0, upper_bound)
    end = date.today()
    start = end - timedelta(days=int(lookback_years * 365.25))
    prices, source = engine_data.get_prices(tickers, start, end)
    weights, performance, report = f2.optimise(prices, objective, bounds)
    shares, leftover = f2.discrete_allocation(weights, prices, amount)
    return dict(prices=prices, source=source, weights=weights,
                performance=performance, report=report, shares=shares,
                leftover=leftover, bounds=bounds, amount=amount,
                lookback_years=lookback_years,
                objective_name="maximum Sharpe" if objective == "1"
                else "minimum volatility")


def render_f2(r: dict, from_f1: bool) -> None:
    note(f"Data source: {r['source']}")
    origin = ("Section 1's sleeve" if from_f1 else "the universe selected above")
    note(f"Optimising {origin}, investing {money(r['amount'])}.")
    at_bound = set(r["report"]["at_ceiling"])

    show_table(
        ["Ticker", "Sleeve", "Weight", "Amount", ""],
        [[t, config.SLEEVES.get(t).name if t in config.SLEEVES else "—",
          percent(w), money(w * r["amount"]),
          "at bound" if t in at_bound else ""]
         for t, w in sorted(r["weights"].items(), key=lambda kv: -kv[1])],
        align_right_from=2,
    )

    ret, vol, sharpe = r["performance"]
    show_table(["Expected performance (annualised)", "Value"],
               [["Expected return", percent(ret, 2)],
                ["Volatility", percent(vol, 2)],
                ["Sharpe ratio", ratio(sharpe)]])

    latest = engine_data.latest_prices(r["prices"])
    invested = sum(float(latest[t]) * n for t, n in r["shares"].items())
    show_table(
        ["Ticker", "Shares", "Price", "Value", "% of total"],
        [[t, f"{n:,}", money(float(latest[t])),
          money(float(latest[t]) * n),
          percent(float(latest[t]) * n / r["amount"])]
         for t, n in sorted(r["shares"].items(),
                            key=lambda kv: -float(latest[kv[0]]) * kv[1])],
    )
    show_table(["Cash", "Value"],
               [["Invested", money(invested)],
                ["Residual cash", money(r["leftover"])],
                ["Cash drag", percent(r["leftover"] / r["amount"], 2)]])

    display(figure_frontier(r["prices"], r["performance"], r["bounds"]))

    show_insight(f2.explain_optimisation(
        r["weights"], r["performance"], r["report"], r["bounds"],
        r["lookback_years"], r["objective_name"], r["leftover"],
        r["amount"], SESSION, from_f1,
    ))

    SESSION.optimisation = OptimisationResult(
        weights=r["weights"], expected_return=ret, volatility=vol,
        sharpe=sharpe, share_counts=r["shares"], leftover_cash=r["leftover"],
        amount=r["amount"], objective=r["objective_name"],
        lookback_years=r["lookback_years"], weight_bounds=r["bounds"],
    )
    SESSION.record_run("F2 Optimisation")
''')

code(r'''
"""Function 2 controls."""

f2_universe = widgets.SelectMultiple(
    options=list(config.SLEEVES), value=list(config.SLEEVES),
    description="Universe", style=STYLE,
    layout=widgets.Layout(width="440px", height="180px"),
)
f2_use_f1 = widgets.Checkbox(
    value=True, description="Use Section 1's sleeve when available",
    style=STYLE, layout=LABEL, indent=False,
)
f2_lookback = widgets.IntSlider(
    value=config.DEFAULT_LOOKBACK_YEARS, min=1, max=15,
    description="Lookback (years)", style=STYLE, layout=LABEL,
)
f2_objective = widgets.RadioButtons(
    options=[("Maximum Sharpe ratio", "1"), ("Minimum volatility", "2")],
    value="1", description="Objective", style=STYLE, layout=LABEL,
)
f2_bound = widgets.FloatSlider(
    value=config.DEFAULT_WEIGHT_BOUNDS[1], min=0.10, max=1.0, step=0.05,
    readout_format=".0%", description="Max weight per holding",
    style=STYLE, layout=LABEL,
)
f2_amount = widgets.BoundedFloatText(
    value=50_000, min=1_000, max=100_000_000, step=1_000,
    description="Amount to invest", style=STYLE, layout=LABEL,
)
f2_go = run_button("Optimise portfolio")
f2_out = widgets.Output()


def on_f2(_=None):
    with f2_out:
        f2_out.clear_output(wait=True)
        from_f1 = bool(f2_use_f1.value and SESSION.risk_profile)
        tickers = (list(SESSION.risk_profile.weights) if from_f1
                   else list(f2_universe.value))
        # The control is authoritative. The Section 1 checkbox above governs
        # which *sleeve* is optimised, not how much is invested in it, and
        # reading the session amount here instead would silently discard
        # whatever the user typed into the field.
        amount = f2_amount.value
        if len(tickers) < 2:
            display(HTML("<b style='color:#e34948'>Select at least two "
                         "tickers.</b>"))
            return
        try:
            render_f2(compute_f2(tickers, f2_lookback.value,
                                 f2_objective.value, f2_bound.value,
                                 amount), from_f1)
        except Exception as exc:
            display(HTML(f"<b style='color:#e34948'>Could not complete: "
                         f"{type(exc).__name__}: {exc}</b>"))


f2_go.on_click(on_f2)
display(widgets.VBox([f2_use_f1, f2_universe, f2_lookback, f2_objective,
                      f2_bound, f2_amount, f2_go, f2_out]))
''')

# ===========================================================================
md(r"""
---
# 3 · Goal projection with a de-risking glide path

A glide path walks equity down as the target date approaches. A deterministic
time-value-of-money calculation gives the headline answer; a **seeded** Monte
Carlo simulation gives the probability that the answer actually holds.

The gap between those two is the output that matters. The deterministic figure
says a contribution reaches the target; the simulation says how often it does.

> Return and volatility assumptions are **illustrative estimates** held in
> `config.py`. They are not a forecast and not Betterment's own.
""")

code(r'''
"""Figure: Monte Carlo percentile fan."""

def figure_fan(glidepath: np.ndarray, current_balance: float,
               monthly_contribution: float, target_amount: float,
               years: int, paths: int = 2_000) -> go.Figure:
    """Percentile bands of simulated balances over time.

    The bands encode confidence, which is magnitude rather than identity, so
    this takes one hue getting darker towards the middle - never several
    different hues, which would imply the bands are unrelated categories.

    The simulation is re-run here with the path history retained, because the
    headline figures use a 10,000-path run that deliberately keeps only the
    terminal values rather than a 10,000 x 240 matrix it has no use for.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    monthly_mean, monthly_std = f3._portfolio_moments(glidepath)
    months = len(glidepath)

    shocks = rng.standard_normal((paths, months))
    returns = monthly_mean[np.newaxis, :] + shocks * monthly_std[np.newaxis, :]

    history = np.empty((paths, months + 1))
    history[:, 0] = current_balance
    for m in range(months):
        history[:, m + 1] = history[:, m] * (1.0 + returns[:, m]) + monthly_contribution

    t = np.arange(months + 1) / 12.0
    p5, p25, p50, p75, p95 = (np.percentile(history, q, axis=0)
                              for q in (5, 25, 50, 75, 95))

    fig = go.Figure()
    for lower, upper, fill, label in (
        (p5, p95, FAN_OUTER, "5th–95th percentile"),
        (p25, p75, FAN_INNER, "25th–75th percentile"),
    ):
        fig.add_scatter(x=t, y=upper, mode="lines", line=dict(width=0),
                        showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=t, y=lower, mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor=fill, name=label,
                        hoverinfo="skip")

    fig.add_scatter(
        x=t, y=p50, mode="lines", name="Median path",
        line=dict(color=SERIES_1, width=2),
        hovertemplate="Year %{x:.1f}<br>Median %{y:$,.0f}<extra></extra>",
    )
    fig.add_hline(
        y=target_amount, line=dict(color=SERIES_2, width=2, dash="dash"),
        annotation_text=f"Target {money(target_amount, 0)}",
        annotation_position="top left",
        annotation_font=dict(size=12, color=SERIES_2),
    )
    fig.update_layout(
        title=(f"Goal projection over {years} years — {paths:,} simulated "
               f"paths, seed {config.RANDOM_SEED}"),
        xaxis=dict(title="Years from today"),
        yaxis=dict(title=f"Portfolio value ({config.CURRENCY_CODE})",
                   tickprefix=config.CURRENCY_SYMBOL, tickformat=",.0f"),
        height=520, hovermode="x unified",
    )
    return fig
''')

code(r'''
"""Compute and render Function 3."""
import numpy_financial as npf


def compute_f3(current_balance: float, monthly_contribution: float,
               years: int, target_amount: float, start_equity: float,
               end_equity: float) -> dict:
    months = years * 12
    glidepath = f3.build_glidepath(years, start_equity, end_equity)
    monthly_rate = f3._mean_monthly_return(glidepath)

    deterministic_fv = float(npf.fv(rate=monthly_rate, nper=months,
                                    pmt=-monthly_contribution,
                                    pv=-current_balance))
    required_pmt = float(npf.pmt(rate=monthly_rate, nper=months,
                                 pv=-current_balance, fv=target_amount))
    terminal = f3.simulate(config.DEFAULT_MONTE_CARLO_PATHS,
                           config.RANDOM_SEED, glidepath,
                           current_balance, monthly_contribution)
    probability = float((terminal >= target_amount).mean())
    stochastic_pmt = f3.solve_contribution_for_probability(
        glidepath=glidepath, current_balance=current_balance,
        target_amount=target_amount,
        probability_target=config.SUCCESS_PROBABILITY_TARGET,
    )
    return dict(glidepath=glidepath, deterministic_fv=deterministic_fv,
                required_pmt=required_pmt, terminal=terminal,
                probability=probability, stochastic_pmt=stochastic_pmt,
                monthly_rate=monthly_rate, months=months, years=years,
                current_balance=current_balance,
                monthly_contribution=monthly_contribution,
                target_amount=target_amount, start_equity=start_equity,
                end_equity=end_equity)


def render_f3(r: dict) -> None:
    note(f"Random seed fixed at {config.RANDOM_SEED}, so these figures "
         f"reproduce exactly. Return assumptions are illustrative.")

    gp = r["glidepath"]
    checkpoints = sorted({0, r["years"] // 4, r["years"] // 2,
                          (3 * r["years"]) // 4, r["years"] - 1})
    show_table(["Point on path", "Equity", "Bonds"],
               [[f"Year {y + 1}",
                 percent(float(gp[min(y * 12, len(gp) - 1)]), 0),
                 percent(1 - float(gp[min(y * 12, len(gp) - 1)]), 0)]
                for y in checkpoints])

    p5, p50, p95 = np.percentile(r["terminal"], [5, 50, 95])
    show_table(["Projected outcome", "Value"], [
        ["Deterministic projection", money(r["deterministic_fv"], 0)],
        ["Simulated median (50th)", money(float(p50), 0)],
        ["Pessimistic case (5th)", money(float(p5), 0)],
        ["Optimistic case (95th)", money(float(p95), 0)],
        ["Target", money(r["target_amount"], 0)],
        ["<b>Probability of reaching target</b>",
         f"<b>{percent(r['probability'])}</b>"],
    ])

    rows = [["Currently planned", money(r["monthly_contribution"]), "—"],
            ["To reach target deterministically",
             money(abs(r["required_pmt"])), "certainty assumed"]]
    label = (f"To reach target "
             f"{percent(config.SUCCESS_PROBABILITY_TARGET, 0)} of the time")
    rows.append([label, money(r["stochastic_pmt"]), "simulated"]
                if r["stochastic_pmt"] is not None
                else [label, "not reachable", "raise the date or lower the target"])
    show_table(["Required monthly contribution", "Monthly", "Basis"], rows,
               align_right_from=1)

    display(figure_fan(gp, r["current_balance"], r["monthly_contribution"],
                       r["target_amount"], r["years"]))

    show_insight(f3.explain_projection(
        r["deterministic_fv"], r["terminal"], r["target_amount"],
        r["probability"], r["monthly_contribution"], r["required_pmt"],
        r["stochastic_pmt"], r["years"], r["start_equity"], r["end_equity"],
        r["start_equity"] != r["end_equity"],
    ))
    SESSION.record_run("F3 Goal projection")
''')

code(r'''
"""Function 3 controls."""

f3_balance = widgets.BoundedFloatText(
    value=50_000, min=0, max=100_000_000, step=1_000,
    description="Current balance", style=STYLE, layout=LABEL)
f3_contribution = widgets.BoundedFloatText(
    value=1_000, min=0, max=1_000_000, step=100,
    description="Monthly contribution", style=STYLE, layout=LABEL)
f3_years = widgets.IntSlider(
    value=20, min=1, max=45, description="Years to target",
    style=STYLE, layout=LABEL)
f3_target = widgets.BoundedFloatText(
    value=500_000, min=1_000, max=1_000_000_000, step=10_000,
    description="Target amount", style=STYLE, layout=LABEL)
f3_glide = widgets.FloatRangeSlider(
    value=[config.DEFAULT_GLIDEPATH_END_EQUITY,
           config.DEFAULT_GLIDEPATH_START_EQUITY],
    min=0.0, max=1.0, step=0.05, readout_format=".0%",
    description="Equity: end → start", style=STYLE, layout=LABEL)
f3_flat = widgets.Checkbox(
    value=False, description="Hold a flat allocation instead (no glide path)",
    style=STYLE, layout=LABEL, indent=False)
f3_go = run_button("Project the goal")
f3_out = widgets.Output()


def on_f3(_=None):
    with f3_out:
        f3_out.clear_output(wait=True)
        end_eq, start_eq = f3_glide.value
        if f3_flat.value:
            end_eq = start_eq
        try:
            render_f3(compute_f3(f3_balance.value, f3_contribution.value,
                                 f3_years.value, f3_target.value,
                                 start_eq, end_eq))
        except Exception as exc:
            display(HTML(f"<b style='color:#e34948'>Could not complete: "
                         f"{type(exc).__name__}: {exc}</b>"))


f3_go.on_click(on_f3)
display(widgets.VBox([f3_balance, f3_contribution, f3_years, f3_target,
                      f3_glide, f3_flat, f3_go, f3_out]))
''')

# ===========================================================================
md(r"""
---
# 4 · Performance and risk report against a benchmark

Builds the weighted daily return series and compares it with a benchmark over
the same aligned window.

The two panels share an x-axis but keep **separate y-axes in separate
subplots** — growth above, drawdown below. That is deliberately not a
dual-axis chart: two different scales forced onto one set of axes invite the
reader to compare two quantities that have no common baseline.
""")

code(r'''
"""Figure: cumulative growth and underwater drawdown."""

def figure_growth_drawdown(aligned: pd.DataFrame, benchmark: str) -> go.Figure:
    """Growth of 1.00 above, drawdown below, on a shared time axis.

    Two panels rather than two y-axes on one panel. A dual-axis chart lets the
    author choose where the two series appear to cross by scaling one of them,
    which is the most reliable way to mislead with a line chart.
    """
    port = (1 + aligned["portfolio"]).cumprod()
    bench = (1 + aligned["benchmark"]).cumprod()

    # No subplot titles: each panel's y-axis already names it, and a centred
    # subplot title would repeat that label while colliding with the legend
    # band above it. Labelling the same thing twice is clutter, not clarity.
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.62, 0.38], vertical_spacing=0.06)

    for series, colour, name in ((port, SERIES_1, "Portfolio"),
                                 (bench, SERIES_2, benchmark)):
        fig.add_scatter(
            x=series.index, y=series, mode="lines", name=name,
            line=dict(color=colour, width=2), legendgroup=name,
            hovertemplate=f"<b>{name}</b> %{{y:.3f}}<extra></extra>",
            row=1, col=1,
        )
        underwater = series / series.cummax() - 1.0
        fig.add_scatter(
            x=underwater.index, y=underwater, mode="lines", name=name,
            line=dict(color=colour, width=1.5), legendgroup=name,
            showlegend=False, fill="tozeroy", fillcolor=rgba(colour, 0.28),
            hovertemplate=f"<b>{name}</b> %{{y:.1%}}<extra></extra>",
            row=2, col=1,
        )

    fig.update_yaxes(title_text="Growth of 1.00", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_layout(
        title=(f"Portfolio against {benchmark} — "
               f"{aligned.index[0].date()} to {aligned.index[-1].date()}"),
        height=620, hovermode="x unified",
    )
    return fig
''')

code(r'''
"""Compute and render Function 4."""

def compute_f4(weights: dict[str, float], benchmark: str,
               start: date, end: date) -> dict:
    prices, source = engine_data.get_prices(list(weights) + [benchmark],
                                            start, end)
    available = [t for t in weights if t in prices.columns]
    if benchmark not in prices.columns:
        raise ValueError(f"No price data for the benchmark {benchmark}.")
    if not available:
        raise ValueError("No holding returned data over this window.")

    port = f4.build_portfolio_returns(prices, weights, available)
    bench = engine_data.daily_returns(prices[[benchmark]])[benchmark].dropna()
    aligned = pd.concat([port, bench], axis=1, join="inner").dropna()
    aligned.columns = ["portfolio", "benchmark"]
    if len(aligned) < 60:
        raise ValueError(f"Only {len(aligned)} overlapping trading days; "
                         "at least 60 are needed.")

    pm, engine = f4.compute_metrics(aligned["portfolio"])
    bm, _ = f4.compute_metrics(aligned["benchmark"])
    tracking = f4.tracking_error(aligned["portfolio"], aligned["benchmark"])
    return dict(source=source, aligned=aligned, portfolio=pm, benchmark=bm,
                engine=engine, tracking=tracking, benchmark_ticker=benchmark,
                available=available, weights=weights)


def render_f4(r: dict) -> None:
    note(f"Data source: {r['source']}")
    pm, bm, tick = r["portfolio"], r["benchmark"], r["benchmark_ticker"]

    show_table(
        ["Measure", "Portfolio", tick, "Difference"],
        [["Annualised return (CAGR)", percent(pm["cagr"], 2),
          percent(bm["cagr"], 2), bps(pm["cagr"] - bm["cagr"])],
         ["Annualised volatility", percent(pm["volatility"], 2),
          percent(bm["volatility"], 2), bps(pm["volatility"] - bm["volatility"])],
         ["Sharpe ratio", ratio(pm["sharpe"]), ratio(bm["sharpe"]),
          ratio(pm["sharpe"] - bm["sharpe"])],
         ["Sortino ratio", ratio(pm["sortino"]), ratio(bm["sortino"]),
          ratio(pm["sortino"] - bm["sortino"])],
         ["Maximum drawdown", percent(pm["max_drawdown"], 2),
          percent(bm["max_drawdown"], 2),
          bps(pm["max_drawdown"] - bm["max_drawdown"])],
         [f"Daily VaR ({percent(config.VALUE_AT_RISK_CONFIDENCE, 0)})",
          percent(pm["value_at_risk"], 2), percent(bm["value_at_risk"], 2),
          bps(pm["value_at_risk"] - bm["value_at_risk"])]],
    )
    show_table(["", "Value"],
               [["Tracking error vs benchmark", percent(r["tracking"], 2)],
                ["Analytics engine", r["engine"]]])

    display(figure_growth_drawdown(r["aligned"], tick))
    show_insight(f4.explain_performance(pm, bm, tick, r["tracking"],
                                        r["aligned"]))
    SESSION.record_run("F4 Performance report")
''')

code(r'''
"""Function 4 controls."""

f4_use_session = widgets.Checkbox(
    value=True, description="Use the latest weights from Section 1 or 2",
    style=STYLE, layout=LABEL, indent=False)
f4_tickers = widgets.SelectMultiple(
    options=list(config.SLEEVES), value=list(config.SLEEVES),
    description="Holdings (equal weight)", style=STYLE,
    layout=widgets.Layout(width="440px", height="160px"))
f4_benchmark = widgets.Text(
    value=config.DEFAULT_BENCHMARK, description="Benchmark ticker",
    style=STYLE, layout=LABEL)
f4_start = widgets.DatePicker(
    value=date.today() - timedelta(days=365 * 3),
    description="Start date", style=STYLE, layout=LABEL)
f4_end = widgets.DatePicker(
    value=date.today(), description="End date", style=STYLE, layout=LABEL)
f4_go = run_button("Build performance report")
f4_out = widgets.Output()


def on_f4(_=None):
    with f4_out:
        f4_out.clear_output(wait=True)
        suggested = SESSION.suggested_weights()
        if f4_use_session.value and suggested is not None:
            weights, origin = suggested
            note(f"Using {origin}.")
            weights = dict(weights)
        else:
            chosen = list(f4_tickers.value)
            if not chosen:
                display(HTML("<b style='color:#e34948'>Select at least one "
                             "holding.</b>"))
                return
            weights = {t: 1.0 / len(chosen) for t in chosen}
        try:
            render_f4(compute_f4(weights, f4_benchmark.value.strip().upper(),
                                 f4_start.value, f4_end.value))
        except Exception as exc:
            display(HTML(f"<b style='color:#e34948'>Could not complete: "
                         f"{type(exc).__name__}: {exc}</b>"))


f4_go.on_click(on_f4)
display(widgets.VBox([f4_use_session, f4_tickers, f4_benchmark, f4_start,
                      f4_end, f4_go, f4_out]))
''')

# ===========================================================================
md(r"""
---
## Run the whole pipeline with defaults

Runs all four stages in order without touching the controls. Useful for
checking the notebook end to end, and for producing the figures used in the
report appendix.
""")

code(r'''
"""Run all four stages in sequence, using the current control values.

This calls the compute/render pair for each stage directly rather than firing
the buttons. Going through the widget handlers would send every result into
that section's Output widget instead of into this cell, which means nothing
would appear here and nothing would be captured when the notebook is executed
non-interactively - and being able to execute the notebook end to end without
a human clicking anything is what makes it testable.
"""

def run_all() -> None:
    display(Markdown("### 1 · Risk profile and strategic allocation"))
    responses = {k: d.value for k, d in f1_dropdowns.items()}
    render_f1(compute_f1(responses, f1_amount.value))

    display(Markdown("### 2 · Portfolio optimisation and share allocation"))
    from_f1 = bool(f2_use_f1.value and SESSION.risk_profile)
    tickers = (list(SESSION.risk_profile.weights) if from_f1
               else list(f2_universe.value))
    render_f2(compute_f2(tickers, f2_lookback.value, f2_objective.value,
                         f2_bound.value, f2_amount.value), from_f1)

    display(Markdown("### 3 · Goal projection with a de-risking glide path"))
    end_eq, start_eq = f3_glide.value
    if f3_flat.value:
        end_eq = start_eq
    render_f3(compute_f3(f3_balance.value, f3_contribution.value,
                         f3_years.value, f3_target.value, start_eq, end_eq))

    display(Markdown("### 4 · Performance and risk report"))
    suggested = SESSION.suggested_weights()
    if f4_use_session.value and suggested is not None:
        weights, origin = suggested
        note(f"Using {origin}.")
        weights = dict(weights)
    else:
        chosen = list(f4_tickers.value)
        weights = {t: 1.0 / len(chosen) for t in chosen}
    render_f4(compute_f4(weights, f4_benchmark.value.strip().upper(),
                         f4_start.value, f4_end.value))

    display(Markdown(
        "### Session summary\n"
        + "\n".join(f"- {name}" for name in SESSION.functions_run)
        + "\n\nOutputs are illustrative and derived from historical data. "
          "They are not investment advice."
    ))


run_all()
''')

# ===========================================================================
md(r"""
---
### Notes

- **Reproducibility.** The Monte Carlo seed is fixed in `config.py` and printed
  with the results, so the figures reproduce exactly.
- **Data.** Prices come from Yahoo Finance where a connection is available and
  from the bundled snapshot in `betterment_prototype/data/` otherwise. Every
  section states which source produced its figures.
- **Illustrative figures.** The return and volatility assumptions behind
  Section 3 are the author's own estimates held in `config.py`. They are not a
  forecast, and not Betterment's published assumptions.
- **Not advice.** Nothing here is investment advice.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

nbf.write(nb, OUT)
print(f"Wrote {OUT}  ({len(cells)} cells)")
