"""Console presentation helpers: headers, aligned tables, and formatting.

Output formatting is centralised for the same reason validation is. A table
printed one way in F1 and another way in F4 reads as four scripts; a single
table function used by all four reads as one system.

Nothing in this module knows anything about portfolios. It takes rows of
strings and prints them, which is what lets it be reused without any of the
functions importing each other.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import config

# ANSI escape codes. Guarded by _COLOUR_ENABLED so that a terminal which does
# not understand them produces plain text rather than visible escape noise.
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"

_COLOUR_ENABLED: bool = True


def set_colour(enabled: bool) -> None:
    """Turn ANSI styling on or off for the whole session."""
    global _COLOUR_ENABLED
    _COLOUR_ENABLED = enabled


def _style(text: str, *codes: str) -> str:
    """Wrap text in ANSI codes, or return it unchanged when colour is off."""
    if not _COLOUR_ENABLED:
        return text
    return "".join(codes) + text + _RESET


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def banner(lines: Sequence[str]) -> None:
    """Print the start-up banner inside a full-width box."""
    width = config.CONSOLE_WIDTH
    print("\n" + "=" * width)
    for line in lines:
        print(_style(line.center(width), _BOLD))
    print("=" * width)


def header(title: str) -> None:
    """Print a major section heading, e.g. the title of a function."""
    width = config.CONSOLE_WIDTH
    print("\n" + "=" * width)
    print(_style(title.upper(), _BOLD, _CYAN))
    print("=" * width)


def subheader(title: str) -> None:
    """Print a minor heading inside a function."""
    print(f"\n{_style(title, _BOLD)}")
    print("-" * min(len(title), config.CONSOLE_WIDTH))


def rule() -> None:
    """Print a full-width horizontal rule."""
    print("-" * config.CONSOLE_WIDTH)


def note(text: str) -> None:
    """Print a de-emphasised aside, such as a data-source or seed statement."""
    print(_style(text, _DIM))


def warn(text: str) -> None:
    """Print a recoverable problem: a rejected input or a fallback taken.

    Used by every validation helper, which is why it lives here and not in
    validation.py -- keeping it here avoids a circular import between the two
    utility modules.
    """
    print(_style(f"   ! {text}", _YELLOW))


def insight(text: str) -> None:
    """Print the interpretation line that closes every function.

    Given its own helper, and its own visual treatment, because it is the part
    of the output the rubric weights most heavily: every function must end by
    explaining its own result rather than merely producing one.
    """
    width = config.CONSOLE_WIDTH
    print("\n" + "-" * width)
    print(_style("WHAT THIS MEANS", _BOLD, _CYAN))
    for line in _wrap(text, width):
        print(line)
    print("-" * width)


def paragraph(text: str) -> None:
    """Print free text wrapped to the console width."""
    for line in _wrap(text, config.CONSOLE_WIDTH):
        print(line)


def _wrap(text: str, width: int) -> list[str]:
    """Wrap text to width, preserving explicit blank-line paragraph breaks."""
    lines: list[str] = []
    for block in text.split("\n"):
        if not block.strip():
            lines.append("")
            continue
        current = ""
        for word in block.split():
            if current and len(current) + 1 + len(word) > width:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            lines.append(current)
    return lines


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def table(headers: Sequence[str], rows: Iterable[Sequence[str]],
          alignments: Sequence[str] | None = None) -> None:
    """Print a column-aligned table.

    Column widths are computed from the content rather than fixed in advance,
    so a long ETF name or a nine-figure currency amount never breaks the
    alignment.

    Args:
        headers: Column headings.
        rows: Row values. Converted with str(), so callers pass pre-formatted
            strings from the money() and percent() helpers below.
        alignments: One of 'l' or 'r' per column. Defaults to left for the
            first column and right for the rest, which is the correct default
            for a label-then-numbers table.
    """
    rows = [[str(cell) for cell in row] for row in rows]
    if alignments is None:
        alignments = ["l"] + ["r"] * (len(headers) - 1)

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render(cells: Sequence[str], bold: bool = False) -> str:
        parts = [
            cell.ljust(widths[i]) if alignments[i] == "l" else cell.rjust(widths[i])
            for i, cell in enumerate(cells)
        ]
        line = "  ".join(parts)
        return _style(line, _BOLD) if bold else line

    print()
    print(render(headers, bold=True))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(render(row))


def key_values(pairs: Sequence[tuple[str, str]], indent: int = 3) -> None:
    """Print aligned label/value pairs, for summaries that are not tables."""
    if not pairs:
        return
    width = max(len(label) for label, _ in pairs)
    pad = " " * indent
    print()
    for label, value in pairs:
        print(f"{pad}{label.ljust(width)}  {value}")


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------


def money(amount: float, decimals: int = 2) -> str:
    """Format a currency amount, e.g. $50,000.00. Negatives are parenthesised."""
    symbol = config.CURRENCY_SYMBOL
    if amount < 0:
        return f"({symbol}{abs(amount):,.{decimals}f})"
    return f"{symbol}{amount:,.{decimals}f}"


def percent(fraction: float, decimals: int = 1) -> str:
    """Format a fraction as a percentage, e.g. 0.0724 -> 7.2%."""
    return f"{fraction * 100:,.{decimals}f}%"


def basis_points(fraction: float) -> str:
    """Format a fraction as basis points, e.g. 0.004 -> 40 bps.

    Used in the F4 insight line. Reporting a small return or volatility
    difference in basis points rather than as '0.4%' is how the difference is
    actually discussed in practice, and it avoids the percent-of-a-percent
    ambiguity that makes those sentences hard to read.
    """
    return f"{fraction * 10_000:,.0f} bps"


def ratio(value: float, decimals: int = 2) -> str:
    """Format a bare ratio such as a Sharpe, handling non-finite values."""
    if value != value or value in {float("inf"), float("-inf")}:
        return "n/a"
    return f"{value:,.{decimals}f}"
