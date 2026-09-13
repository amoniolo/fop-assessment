# Betterment Robo-Advisory Prototype — Notebook Edition

**Fundamentals of Programming (5K7V0027) — Level 7 portfolio, Part 2**

An interactive Jupyter front-end over the same advisory engine as the
command-line package in `../betterment_prototype`. `ipywidgets` replace the
console prompts; Plotly replaces Matplotlib.

---

## Run it

```bash
cd betterment_notebook
jupyter lab Betterment_Advisory.ipynb     # or: jupyter notebook
```

Then **Kernel → Restart & Run All**, and use the controls in each section.

The final cell runs all four stages end to end with whatever the controls are
currently set to, so a full run takes a minute or two — the Monte Carlo and the
contribution solver do real work. That cell is also what produces the figures
in `figures/` for the report appendix.

### Dependencies

Everything is already present in a standard Anaconda install except the four
finance libraries, which the sibling package needs anyway:

```bash
pip install -r ../betterment_prototype/requirements.txt
pip install -r requirements-notebook.txt
```

Verified on Anaconda Python 3.13.9 with plotly 6.3.0, ipywidgets 8.1.7,
notebook 7.4.5.

---

## Design: one engine, two front-ends

**No financial logic lives in this notebook.** Scoring, the suitability rule,
optimisation, the glide path, the simulation and the risk metrics are all
imported from `betterment_prototype`. The notebook supplies only the interface.

That is the point of the structure rather than a convenience. Two front-ends
over one engine means a change to the advisory logic cannot make the console
version and the notebook disagree, and it demonstrates that the domain code was
written independently of how it happens to be displayed. The console package's
`run()` functions are the only console-coupled code in the engine, and this
notebook uses none of them.

Each section is built as a `compute_*` / `render_*` pair:

- `compute_*` takes plain values and returns plain results. No widgets, no
  printing. This is what makes the notebook testable.
- `render_*` displays tables, a figure, and the engine's interpretation text.
- the widget handler is a thin wrapper that reads the controls and calls both.

The final `run_all()` calls the pairs directly rather than firing the buttons,
so the notebook can be executed with no human clicking anything:

```bash
python -c "import nbformat; from nbclient import NotebookClient; \
nb = nbformat.read('Betterment_Advisory.ipynb', as_version=4); \
NotebookClient(nb, timeout=900, kernel_name='python3').execute()"
```

That run completes with **zero errors** and produces four Plotly figures.

### Control-wiring tests

```bash
python test_notebook.py
```

14 checks. These execute the real notebook in a kernel, set control values,
fire the actual button handlers and assert the results move accordingly. They
exist because the engine's own test suite covers the financial logic but not
the wiring between a widget and the calculation - and a control whose value
never reaches the calculation is exactly the kind of fault that suite cannot
see.

---

## Chart design

Colour is assigned by function.

**Only two categorical hues appear anywhere** — blue `#2a78d6` and orange
`#eb6834` — because no chart here carries more than two identities: equity
against bonds, portfolio against benchmark. The two are far enough apart to
stay distinguishable under colour-vision deficiency, and both hold their
contrast against the chart surface.

| Figure | Form | Why that form |
|--------|------|---------------|
| Allocation | Sorted horizontal bars | Ten pie slices cannot be compared by eye — angle is the hardest visual channel to judge. Bars on a common baseline make the ordering immediate. Every bar is labelled directly, so colour is never the sole carrier. |
| Efficient frontier | Scatter + line | Only the frontier and the selected portfolio carry identity. The ten individual sleeves are *context*, so they are muted grey with direct labels rather than ten more hues. |
| Goal projection | Percentile fan | Confidence is magnitude, not identity, so the bands are one hue getting darker toward the median — never several different hues, which would imply unrelated categories. |
| Growth & drawdown | Two stacked panels | Deliberately **not** a dual-axis chart. Two scales on one set of axes let the author decide where the series appear to cross by rescaling one of them. |

Every figure is accompanied by an HTML table of the same numbers, so the data
is selectable, searchable and screen-reader accessible rather than locked
inside a canvas.

The Plotly template pins its own light surface instead of inheriting the
Jupyter theme, so figures stay legible under either a light or a dark notebook
theme rather than putting dark text on a dark ground.

---

## Files

```
betterment_notebook/
    Betterment_Advisory.ipynb    The notebook
    build_notebook.py            Regenerates the notebook from Python source
    requirements-notebook.txt    Notebook-only extras
    test_notebook.py             Regression tests for the control wiring
    figures/                     Exported PNGs for the report appendix
    README.md
```

### Why the notebook is generated

`build_notebook.py` writes the `.ipynb`. Notebook JSON is unpleasant to edit by
hand and easy to corrupt; keeping the cells as ordinary Python means they can be
diffed, reviewed and regenerated:

```bash
python build_notebook.py
```

Edit the notebook directly if you prefer — but then either stop using the
generator or fold your changes back into it, otherwise the next regeneration
overwrites them.

---

## Reproducibility and caveats

- **Seed fixed.** The Monte Carlo seed lives in `config.py` and is printed in
  the chart title, so Section 3 reproduces exactly.
- **Data source stated.** Prices come from Yahoo Finance where a connection is
  available and from the bundled snapshot in `betterment_prototype/data/`
  otherwise. Every section says which it used.
- **Illustrative assumptions.** The return and volatility estimates behind
  Section 3 are the author's own, held in `config.py`. They are not a forecast
  and not Betterment's published assumptions.
- **Not advice.** Nothing here is investment advice.
