"""Regression tests for the notebook's control wiring.

The notebook's financial logic is the engine's, and is covered by
`../betterment_prototype/tests/`. What is untested there, and what actually
broke, is the wiring between the widgets and the compute functions: a control
whose value never reaches the calculation.

These tests execute the real notebook in a kernel, set control values
programmatically, and assert that the results move accordingly. Running the
notebook rather than importing a copy of its code is the point - a test that
re-implements the wiring would not have caught the bug either.

    python test_notebook.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "Betterment_Advisory.ipynb"


def build_client() -> NotebookClient:
    """Execute every cell except the final `run_all()` call.

    The last code cell runs the whole pipeline, which is slow and unnecessary
    here: these tests drive the sections themselves.
    """
    nb = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
    nb.cells[code_cells[-1]].source = "pass  # run_all() suppressed for tests"
    client = NotebookClient(nb, timeout=900, kernel_name="python3")
    client.allow_errors = False
    return client


def run_checks() -> int:
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        if condition:
            print(f"  PASS  {name}")
        else:
            failures.append(f"{name}: {detail}")
            print(f"  FAIL  {name}  {detail}")

    client = build_client()
    with client.setup_kernel():
        client.execute(cleanup_kc=False)

        def evaluate(expression: str) -> str:
            """Run an expression in the notebook's kernel, return its stdout."""
            cell = nbformat.v4.new_code_cell(f"print({expression})")
            result = client.execute_cell(cell, len(client.nb.cells) - 1)
            for output in result.get("outputs", []):
                if output["output_type"] == "stream":
                    return output["text"].strip()
                if output["output_type"] == "error":
                    raise AssertionError(
                        f"{expression} -> {output['ename']}: {output['evalue']}"
                    )
            return ""

        def run(statement: str) -> None:
            cell = nbformat.v4.new_code_cell(statement)
            result = client.execute_cell(cell, len(client.nb.cells) - 1)
            for output in result.get("outputs", []):
                if output["output_type"] == "error":
                    raise AssertionError(
                        f"{statement} -> {output['ename']}: "
                        f"{output['evalue']}\n"
                        + "\n".join(output.get("traceback", [])[-6:])
                    )

        def errors_in(output_widget: str) -> str:
            """Return any error banner the handler wrote into its Output."""
            return evaluate(
                f"[o['data']['text/html'] for o in {output_widget}.outputs "
                f"if o.get('output_type') == 'display_data' "
                f"and 'Could not complete' in o.get('data', {{}})"
                f".get('text/html', '')]"
            )

        # ------------------------------------------------------------------
        # The bug: F2's amount control was ignored when Section 1 had run.
        #
        # These drive the real button handlers rather than re-implementing
        # what they do. An earlier version of this file inlined the handler's
        # own logic and passed against the broken notebook - the assertion has
        # to run through the code path that was wrong, or it proves nothing.
        # ------------------------------------------------------------------

        # Distinct values throughout. Using the same number for f1_amount and
        # f2_amount's default would make the seeding check pass whether or not
        # any seeding happened.
        run("f2_amount.value = 9_000")          # deliberately not the default
        run("f1_amount.value = 50_000")
        run("on_f1()")

        check("F1 handler completed without error",
              errors_in("f1_out") == "[]", errors_in("f1_out"))
        check("F1 amount reaches the session",
              evaluate("SESSION.risk_profile.investable_amount") == "50000.0",
              evaluate("SESSION.risk_profile.investable_amount"))
        check("F1 seeds the F2 amount control (9,000 -> 50,000)",
              evaluate("f2_amount.value") == "50000.0",
              evaluate("f2_amount.value"))

        # The regression itself. Edit the control, keep the Section 1 checkbox
        # ticked - the combination that used to discard the edit - and assert
        # on what the handler actually invested.
        run("f2_amount.value = 250_000")
        run("f2_use_f1.value = True")
        run("on_f2()")

        check("F2 handler completed without error",
              errors_in("f2_out") == "[]", errors_in("f2_out"))
        check("F2 honours a user-edited amount while using F1's sleeve",
              evaluate("SESSION.optimisation.amount") == "250000.0",
              evaluate("SESSION.optimisation.amount"))

        leftover = float(evaluate("SESSION.optimisation.leftover_cash"))
        check("F2 residual cash is a remainder of 250,000, not of 50,000",
              0 <= leftover < 250_000 * 0.02,
              f"leftover {leftover:,.2f}")

        # Sanity: the share counts must be far larger than a 50,000 order.
        run("_shares_250k = sum(SESSION.optimisation.share_counts.values())")
        shares_250k = int(evaluate("_shares_250k"))
        run("f2_amount.value = 50_000")
        run("on_f2()")
        shares_50k = int(evaluate("sum(SESSION.optimisation.share_counts.values())"))
        check("F2 share counts scale with the amount (250k > 50k)",
              shares_250k > shares_50k * 3,
              f"{shares_250k} shares at 250k vs {shares_50k} at 50k")

        # ------------------------------------------------------------------
        # The same class of bug in the other direction: with the checkbox off,
        # the chosen universe must be used rather than F1's sleeve.
        # ------------------------------------------------------------------
        run("f2_use_f1.value = False")
        run("f2_universe.value = ('VTI', 'BND', 'VEA')")
        run("on_f2()")
        check("F2 handler completed without error (custom universe)",
              errors_in("f2_out") == "[]", errors_in("f2_out"))
        check("F2 uses the selected universe when the F1 checkbox is off",
              int(evaluate("len(SESSION.optimisation.weights)")) <= 3,
              evaluate("list(SESSION.optimisation.weights)"))

        # ------------------------------------------------------------------
        # F3 and F4 controls, checked for the same failure mode.
        # ------------------------------------------------------------------
        run("f3_balance.value = 123_456")
        run("f3_contribution.value = 750")
        run("f3_years.value = 12")
        run("f3_target.value = 400_000")
        run("""
_end_eq, _start_eq = f3_glide.value
_r4 = compute_f3(f3_balance.value, f3_contribution.value, f3_years.value,
                 f3_target.value, _start_eq, _end_eq)
""")
        check("F3 honours the balance control",
              evaluate("_r4['current_balance']") == "123456.0",
              evaluate("_r4['current_balance']"))
        check("F3 honours the contribution control",
              evaluate("_r4['monthly_contribution']") == "750.0",
              evaluate("_r4['monthly_contribution']"))
        check("F3 honours the years control",
              evaluate("len(_r4['glidepath'])") == "144",
              evaluate("len(_r4['glidepath'])"))
        check("F3 honours the target control",
              evaluate("_r4['target_amount']") == "400000.0",
              evaluate("_r4['target_amount']"))

        run("f4_use_session.value = False")
        run("f4_tickers.value = ('VTI', 'BND')")
        run("""
_chosen = list(f4_tickers.value)
_w = {t: 1.0 / len(_chosen) for t in _chosen}
_r5 = compute_f4(_w, f4_benchmark.value.strip().upper(),
                 f4_start.value, f4_end.value)
""")
        check("F4 honours the holdings control",
              evaluate("sorted(_r5['available'])") == "['BND', 'VTI']",
              evaluate("sorted(_r5['available'])"))

    print(f"\n{len(failures)} failed")
    for f in failures:
        print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run_checks())
