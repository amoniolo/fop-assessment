# Fundamentals of Programming (5K7V0027)

Level 7 portfolio, Part 2 — Python prototype.

A menu-driven Python application modelling the automated advisory pipeline
operated by [Betterment](https://www.betterment.com), a US robo-adviser,
delivered in two front-ends over one shared engine.

| Folder | What it is |
|--------|------------|
| [`betterment_prototype/`](betterment_prototype/) | The command-line application, and the advisory engine both front-ends use. Start here. |
| [`betterment_notebook/`](betterment_notebook/) | A Jupyter front-end over the same engine — `ipywidgets` controls, Plotly figures. |

## The four functions

The four menu options are the four stages of the advisory pipeline. Each stage
can consume the previous stage's output through a shared session object, and
each also runs standalone.

| # | Stage | What it does |
|---|-------|--------------|
| 1 | Onboarding | Questionnaire → risk tier → model portfolio, with a suitability rule that caps equity on a short horizon |
| 2 | Execution | Efficient frontier → weights → whole-share order, with residual cash stated |
| 3 | Planning | De-risking glide path + Monte Carlo → probability of reaching a goal |
| 4 | Reporting | Portfolio against a benchmark: return, risk, drawdown, tracking error |

Every function ends by interpreting its own result rather than only producing
one.

## Quick start

```bash
cd betterment_prototype
python -m venv .venv
.venv\Scripts\activate           # Windows
source .venv/bin/activate        # macOS / Linux
pip install -r requirements.txt

python main.py                   # the command-line application
python tests/test_prototype.py   # 46 tests
```

For the notebook edition, see [`betterment_notebook/README.md`](betterment_notebook/README.md).

## Design note

No financial logic is duplicated between the two front-ends. Scoring, the
suitability rule, optimisation, the glide path, the simulation and the risk
metrics all live in `betterment_prototype`; the notebook imports them and
supplies only the interface. A change to the advisory logic therefore cannot
make the two versions disagree.

## Runs without a network

`betterment_prototype/data/price_snapshot.csv` holds ten years of real daily
adjusted-close prices for every model sleeve and the benchmark. Retrieval tries
Yahoo Finance first and falls back to that snapshot on any failure, and every
function states which source produced its figures.

## Caveats

- Prototype outputs are illustrative, derived from historical data over a
  chosen lookback. They are not a forecast.
- The capital-market assumptions behind Function 3 are the author's own
  illustrative estimates held in `config.py`, not Betterment's published ones.
- The Monte Carlo seed is fixed and printed with the results, so the figures
  reproduce exactly.
- Nothing here is investment advice.
