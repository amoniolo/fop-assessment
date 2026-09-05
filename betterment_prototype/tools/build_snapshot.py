"""Build the bundled offline price snapshot.

Run once, with a network connection, before submitting the package:

    python tools/build_snapshot.py

It downloads the full history for every model-portfolio sleeve plus the default
benchmark and writes data/price_snapshot.csv, which is what utils/data.py falls
back to when yfinance is unavailable. The point of the snapshot is that the
assessor can run the artefact with the network disabled and still get real
figures rather than an error.

This is a development tool, not part of the running artefact. It is kept
separate so that nothing in the four functions can accidentally depend on it,
and so the package still runs if the tools/ folder is missing.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

# Import the package's own config rather than duplicating the ticker list.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

# Ten years is enough for any lookback the menu offers, and keeps the CSV to a
# size that is reasonable to include in a submitted archive.
HISTORY_YEARS = 10


def main() -> int:
    """Download and write the snapshot. Returns a process exit code."""
    try:
        import pandas as pd
        import yfinance as yf
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Install requirements.txt first.")
        return 1

    tickers = sorted(set(config.SLEEVES) | {config.DEFAULT_BENCHMARK})
    end = date.today()
    start = end - timedelta(days=int(HISTORY_YEARS * 365.25))

    print(f"Downloading {len(tickers)} tickers from {start} to {end}...")
    raw = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if raw is None or raw.empty:
        print("No data returned. Check the connection and try again.")
        return 1

    if isinstance(raw.columns, pd.MultiIndex):
        frame = raw.xs("Close", axis=1, level=0)
    else:
        frame = raw[["Close"]].copy()
        frame.columns = tickers[:1]

    frame = frame.ffill().dropna(how="all")
    frame.index.name = "Date"

    missing = [t for t in tickers if t not in frame.columns]
    if missing:
        print(f"Warning: no data for {', '.join(missing)}. "
              f"Offline running will not cover them.")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.SNAPSHOT_FILE)

    metadata = {
        "captured": date.today().isoformat(),
        "tickers": list(frame.columns),
        "first_date": str(frame.index.min().date()),
        "last_date": str(frame.index.max().date()),
        "rows": int(len(frame)),
        "source": "Yahoo Finance via yfinance, adjusted close",
        "note": ("Bundled so the artefact runs with the network disabled. "
                 "Figures produced from this snapshot are as of the last "
                 "date above, not today."),
    }
    config.SNAPSHOT_METADATA_FILE.write_text(json.dumps(metadata, indent=2))

    print(f"Wrote {len(frame):,} rows for {len(frame.columns)} tickers to "
          f"{config.SNAPSHOT_FILE}")
    print(f"Coverage: {metadata['first_date']} to {metadata['last_date']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
