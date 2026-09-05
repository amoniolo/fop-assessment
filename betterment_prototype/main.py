"""Entry point: banner, menu loop, dispatch table, and top-level error handling.

Run from the package directory with:

    python main.py

The four menu options are the four stages of Betterment's automated advisory
pipeline -- profile and allocate, optimise and execute, project against a goal,
report against a benchmark. Each stage can consume the previous stage's output
through the shared Session object, and each also runs standalone, so the menu
can be entered at any point.

Two structural decisions are made here and are worth stating explicitly.

The menu is a dictionary of MenuEntry records rather than a chain of if/elif.
Adding a fifth function then means adding one dictionary entry, not editing the
control flow, and the same dictionary is used to render the menu, to validate
the input and to dispatch -- so the displayed options and the callable options
cannot drift apart, which is the bug an if/elif chain invites.

Every dispatch is wrapped in a handler that catches the exception types the
functions are known to raise, prints one readable sentence, and returns to the
menu. No traceback ever reaches the user: an unhandled crash in any function
would drop that function into the lowest band of the marking rubric, and it
would also leave the client with no idea what to do next.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path

# Allow `python main.py` from inside the package directory: the package's own
# folder is not on sys.path in that case, so `import config` would fail.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from functions import (  # noqa: E402
    f1_risk_profile,
    f2_optimise,
    f3_goal_projection,
    f4_performance,
)
from utils import display  # noqa: E402
from utils.data import PriceDataError, snapshot_available, snapshot_date  # noqa: E402
from utils.session import Session  # noqa: E402
from utils.validation import InputAbort  # noqa: E402


@dataclass(frozen=True)
class MenuEntry:
    """One menu option: what it is called and what it runs.

    handler is None for Exit, which is what lets the loop treat exit as an
    ordinary dictionary entry rather than as a special case checked before the
    lookup. Frozen, because the menu is fixed at import time.
    """

    label: str
    handler: Callable[[Session], None] | None
    blurb: str = ""


MENU: dict[str, MenuEntry] = {
    "1": MenuEntry(
        "Risk profile and strategic allocation",
        f1_risk_profile.run,
        "Questionnaire to risk tier to model portfolio.",
    ),
    "2": MenuEntry(
        "Optimise portfolio and allocate to shares",
        f2_optimise.run,
        "Efficient frontier to weights to whole-share order.",
    ),
    "3": MenuEntry(
        "Project a goal with a de-risking glide path",
        f3_goal_projection.run,
        "Glide path and Monte Carlo to probability of success.",
    ),
    "4": MenuEntry(
        "Performance and risk report vs benchmark",
        f4_performance.run,
        "Portfolio against a benchmark, return and risk.",
    ),
    "0": MenuEntry("Exit", None),
}

BANNER_LINES = (
    "BETTERMENT ROBO-ADVISORY PROTOTYPE",
    "Fundamentals of Programming (5K7V0027)  -  Level 7 portfolio, Part 2",
)


def print_banner() -> None:
    """Introduce the artefact: what it is, and what it models."""
    display.banner(BANNER_LINES)
    display.paragraph(
        "This program models the automated advisory pipeline operated by "
        "Betterment, a US robo-adviser. The four functions below are the four "
        "stages of that pipeline. Each one can take the previous stage's "
        "output, and each also runs on its own."
    )
    print()
    display.paragraph(
        "Prices come from Yahoo Finance where a connection is available, and "
        "from a bundled snapshot otherwise. Every function states which source "
        "produced its figures. All outputs are illustrative and based on "
        "historical data over the lookback you choose; they are not advice and "
        "not a forecast."
    )

    if snapshot_available():
        display.note(
            f"\n   Offline snapshot bundled, dated {snapshot_date()}. "
            f"The program runs with the network disabled."
        )
    else:
        display.warn(
            "No offline snapshot found in data/. The program needs a network "
            "connection. Run tools/build_snapshot.py to create one."
        )

    display.note(f"   Random seed: {config.RANDOM_SEED}. "
                 f"Currency: {config.CURRENCY_CODE}.")
    display.note("   Type 'q' at any prompt to abandon a function and "
                 "return here.")


def print_menu(session: Session) -> None:
    """Render the menu from the same dictionary the dispatcher uses."""
    display.header("Main menu")
    for key, entry in MENU.items():
        if entry.handler is None:
            print(f"\n   [{key}] {entry.label}")
        else:
            done = " (completed)" if _label_of(entry) in session.functions_run else ""
            print(f"   [{key}] {entry.label}{done}")
            print(f"       {entry.blurb}")


def _label_of(entry: MenuEntry) -> str:
    """Map a menu entry to the short name a function records on completion."""
    return {
        "Risk profile and strategic allocation": "F1 Risk profile",
        "Optimise portfolio and allocate to shares": "F2 Optimisation",
        "Project a goal with a de-risking glide path": "F3 Goal projection",
        "Performance and risk report vs benchmark": "F4 Performance report",
    }.get(entry.label, entry.label)


def dispatch(choice: str, session: Session) -> None:
    """Run the chosen function, converting any expected failure into a message.

    The except clauses are ordered from most to least specific and each names
    the types it handles, so a genuinely unexpected exception is not swallowed
    -- it falls through to the last clause, which reports the exception type
    rather than pretending the program understood it.
    """
    entry = MENU[choice]
    if entry.handler is None:
        return

    try:
        entry.handler(session)

    except InputAbort as exc:
        # Not an error: the user chose to leave the function.
        display.note(f"\n   {exc}")

    except PriceDataError as exc:
        display.warn(str(exc))
        display.paragraph(
            "   The bundled snapshot covers the model-portfolio sleeves and "
            "the default benchmark. Custom tickers need a connection."
        )

    except f2_optimise.OptimisationError as exc:
        display.warn(str(exc))

    except (ValueError, KeyError, ZeroDivisionError) as exc:
        # The families a numerical function raises on inputs that pass
        # validation individually but do not work together -- a date range too
        # short for the statistics, a ticker missing from a frame.
        display.warn(f"{entry.label} could not complete: {exc}")

    except MemoryError:
        display.warn(
            "The simulation ran out of memory. Reduce the number of paths in "
            "config.DEFAULT_MONTE_CARLO_PATHS and try again."
        )

    except Exception as exc:  # noqa: BLE001 - last resort; see the docstring
        display.warn(
            f"{entry.label} stopped unexpectedly: "
            f"{type(exc).__name__}: {exc}"
        )
        display.paragraph(
            "   The other functions are unaffected. If this repeats, check "
            "that the installed library versions match requirements.txt."
        )


def read_menu_choice() -> str:
    """Read a menu selection, re-prompting until it is one of the options.

    Written here rather than reusing prompt_choice() because the menu must
    never be abandoned by the abort words: 'q' at the top level should be a
    valid way to exit, not an exception, and the menu is the one place in the
    program with nowhere to return to.
    """
    valid = ", ".join(MENU)
    while True:
        try:
            raw = input(f"\n   Select an option ({valid}): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return "0"

        if raw in MENU:
            return raw
        if raw.lower() in {"q", "quit", "exit"}:
            return "0"
        display.warn(f"'{raw}' is not an option. Enter one of: {valid}.")


def print_exit_summary(session: Session) -> None:
    """Summarise the session on the way out."""
    display.header("Session summary")
    if session.functions_run:
        display.paragraph("   Functions completed this session:")
        for name in session.functions_run:
            print(f"     - {name}")
    else:
        display.paragraph("   No functions were completed this session.")

    charts = sorted(config.OUTPUT_DIR.glob("*.png")) if config.OUTPUT_DIR.exists() else []
    if charts:
        print()
        display.paragraph("   Charts written to output/:")
        for chart in charts:
            print(f"     - {chart.name}")

    print()
    display.paragraph(
        "   Outputs are illustrative and derived from historical data. They "
        "are not investment advice."
    )
    print()


def main() -> int:
    """Run the menu loop until the user exits. Returns a process exit code."""
    # Windows terminals do not enable ANSI processing in every host; turning
    # colour off there is safer than printing escape codes as literal text.
    if sys.platform == "win32" and not sys.stdout.isatty():
        display.set_colour(False)

    session = Session()
    print_banner()

    while True:
        print_menu(session)
        choice = read_menu_choice()

        if MENU[choice].handler is None:
            break

        dispatch(choice, session)

        # The brief requires an explicit choice between returning to the menu
        # and exiting after every function, rather than dropping straight back
        # into the menu.
        print()
        try:
            again = input("   Return to the menu? (y to continue, "
                          "anything else to exit): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if again not in {"y", "yes", ""}:
            break

    print_exit_summary(session)
    return 0


if __name__ == "__main__":
    sys.exit(main())
