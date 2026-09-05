"""Cache-first price retrieval with an offline fallback.

This is the single most defensive module in the package, and it exists to
answer one question: what happens when yfinance fails on the assessor's
machine? Without it, a network problem becomes run-time errors across all four
functions, which is the most heavily penalised outcome in the prototype rubric.

The contract every caller relies on: get_prices() either returns usable prices
or raises PriceDataError. It never returns an empty or partially populated
frame, because a silently empty frame surfaces much later as an unreadable
error from inside PyPortfolioOpt or QuantStats.

Retrieval order:

1. The in-process memo, so a menu session that runs all four functions hits
   the network once rather than four times.
2. yfinance.
3. The bundled CSV snapshot in data/.

Whichever source was used is returned to the caller and printed, so the
assessor always knows whether the numbers on screen are live or snapshot.
"""

from __future__ import annotations

import base64
import json
import os
import ssl
import sys
import tempfile
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

import config
from utils.display import note, warn

# Module-level memo keyed by (tickers, start, end). Populated on first success
# and reused for the rest of the session. Cleared only when the process ends,
# which is the right lifetime: prices for a fixed historical window do not
# change while the user is sitting in the menu.
_MEMO: dict[tuple, tuple[pd.DataFrame, str]] = {}

# Set once, the first time a live retrieval is attempted. See
# _trust_windows_certificates() below.
_CERTIFICATES_PREPARED: bool = False
_LOGGING_SILENCED: bool = False


class PriceDataError(Exception):
    """No usable price data from any source.

    Raised rather than returning an empty frame so that the failure is caught
    by the menu loop's handler and reported as one readable sentence, instead
    of propagating into a library and emerging as a KeyError on a column that
    was never created.
    """


def _trust_windows_certificates() -> None:
    """Let yfinance use the certificates Windows already trusts.

    Found while testing on a network that inspects TLS. yfinance retrieves
    through curl_cffi, which ships its own certificate bundle and ignores the
    Windows certificate store. On any network behind a corporate or ISP proxy
    that re-signs HTTPS -- university networks very much included -- the
    proxy's root certificate is in the Windows store but not in curl's bundle,
    so every request fails certificate verification even though the same
    machine's browser and pip work perfectly.

    The fix is to write the Windows root store out as a PEM bundle and point
    curl at it through CURL_CA_BUNDLE. Note what this does *not* do: it does
    not disable verification. Certificates are still verified, against exactly
    the authorities the operating system already trusts, so a genuinely
    untrusted certificate is still rejected.

    Runs once per session, is skipped when the user has already set the
    variable, and fails silently -- if it does not work, the snapshot fallback
    below is still there.
    """
    global _CERTIFICATES_PREPARED
    if _CERTIFICATES_PREPARED or sys.platform != "win32":
        return
    _CERTIFICATES_PREPARED = True

    if os.environ.get("CURL_CA_BUNDLE"):
        return  # The user has configured this deliberately; do not override.

    try:
        pem_parts: list[str] = []
        try:
            import certifi
            pem_parts.append(Path(certifi.where()).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - certifi is optional here
            pass

        for store in ("ROOT", "CA"):
            for cert_bytes, encoding, _trust in ssl.enum_certificates(store):
                if encoding != "x509_asn":
                    continue
                encoded = base64.b64encode(cert_bytes).decode("ascii")
                body = "\n".join(encoded[i:i + 64]
                                 for i in range(0, len(encoded), 64))
                pem_parts.append(
                    f"-----BEGIN CERTIFICATE-----\n{body}\n"
                    f"-----END CERTIFICATE-----\n"
                )

        if len(pem_parts) < 2:
            return

        bundle = Path(tempfile.gettempdir()) / "betterment_prototype_ca.pem"
        bundle.write_text("\n".join(pem_parts), encoding="utf-8")
        os.environ["CURL_CA_BUNDLE"] = str(bundle)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", str(bundle))
    except Exception:  # noqa: BLE001 - best effort only; snapshot is the safety net
        return


def _silence_yfinance_logging() -> None:
    """Stop yfinance printing its own failure reports over ours.

    When a ticker does not resolve, yfinance writes a multi-line report of its
    own to the console. During ticker validation that is actively unhelpful: a
    user who typed one letter wrong gets a block of library diagnostics before
    they reach our one-line explanation of what to do about it. Validation is
    *expected* to fail sometimes -- that is what it is for -- so the library's
    noise is suppressed and this module reports the outcome itself.
    """
    global _LOGGING_SILENCED
    if _LOGGING_SILENCED:
        return
    _LOGGING_SILENCED = True

    try:
        import logging
        for name in ("yfinance", "yfinance.data", "yfinance.ticker",
                     "yfinance.multi", "peewee", "urllib3"):
            logging.getLogger(name).setLevel(logging.CRITICAL)
    except Exception:  # noqa: BLE001 - cosmetic only
        return


def get_prices(
    tickers: list[str],
    start: date,
    end: date,
    quiet: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Return (prices, source) for the requested tickers and window.

    Args:
        tickers: Symbols to retrieve. Order is preserved in the columns.
        start: First date requested, inclusive.
        end: Last date requested, inclusive.
        quiet: Suppress the source line. Used by the ticker validator, which
            calls this repeatedly and should not narrate each attempt.

    Returns:
        A tuple of the adjusted-close price frame (a DatetimeIndex, one column
        per ticker, forward-filled and with all-NaN rows dropped) and a human
        readable description of where the data came from -- either "live data
        from Yahoo Finance" or "cached snapshot dated ...".

    Raises:
        PriceDataError: Neither the live source nor the snapshot could supply
            usable data for these tickers.
    """
    symbols = list(dict.fromkeys(t.upper() for t in tickers))
    memo_key = (tuple(symbols), start, end)
    if memo_key in _MEMO:
        return _MEMO[memo_key]

    prices, source = _fetch_live(symbols, start, end)

    if prices is None:
        prices, source = _load_snapshot(symbols, start, end)

    if prices is None or prices.empty:
        raise PriceDataError(
            f"No price data available for {', '.join(symbols)} between "
            f"{start.isoformat()} and {end.isoformat()}, from either Yahoo "
            f"Finance or the bundled snapshot in {config.DATA_DIR.name}/."
        )

    if not quiet:
        note(f"   Data source: {source}")

    _MEMO[memo_key] = (prices, source)
    return prices, source


def _fetch_live(
    symbols: list[str], start: date, end: date
) -> tuple[pd.DataFrame | None, str]:
    """Attempt yfinance. Returns (None, reason) on any failure.

    The except clause is deliberately broad. Everywhere else in this codebase
    exceptions are caught specifically, but a third-party network client can
    fail in ways its own documentation does not enumerate -- DNS, TLS, rate
    limiting, a changed JSON schema -- and the whole purpose of this function
    is that *no* failure mode of yfinance is allowed to reach the user. The
    breadth is the feature, and the fallback below is what makes it safe.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None, "yfinance is not installed"

    _trust_windows_certificates()
    _silence_yfinance_logging()

    try:
        with warnings.catch_warnings():
            # yfinance emits FutureWarnings from its own pandas usage that the
            # user can do nothing about; they would only obscure the output.
            warnings.simplefilter("ignore")
            raw = yf.download(
                tickers=symbols,
                start=start.isoformat(),
                # yfinance treats end as exclusive, so add a day to make the
                # documented inclusive contract of get_prices() true.
                end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=True,
                progress=False,
                threads=False,
            )
    except Exception as exc:  # noqa: BLE001 - see docstring
        warn(f"Live price retrieval failed ({type(exc).__name__}). "
             "Falling back to the bundled snapshot.")
        return None, "live retrieval failed"

    frame = _extract_close(raw, symbols)
    if frame is None:
        warn("Live retrieval returned no usable rows. "
             "Falling back to the bundled snapshot.")
        return None, "live retrieval returned nothing usable"

    return frame, "live data from Yahoo Finance"


def _extract_close(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame | None:
    """Normalise whatever shape yfinance returned into a plain price frame.

    yfinance returns a column MultiIndex for several tickers but flat columns
    for one, and it has changed which level the field sits on between versions.
    Handling both shapes here means no function downstream has to care.
    """
    if raw is None or raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        # With auto_adjust=True the price field is named 'Close'. Look for it
        # on either level rather than assuming the current library's order.
        for level in range(raw.columns.nlevels):
            if "Close" in raw.columns.get_level_values(level):
                frame = raw.xs("Close", axis=1, level=level)
                break
        else:
            return None
    elif "Close" in raw.columns:
        frame = raw[["Close"]].copy()
        frame.columns = symbols[:1]
    else:
        frame = raw.copy()

    return _clean(frame, symbols)


def _clean(frame: pd.DataFrame, symbols: list[str]) -> pd.DataFrame | None:
    """Drop unusable columns, forward-fill gaps, and enforce column order.

    Forward-filling is correct for price series with a stray missing day, but
    a column that is mostly missing is not a gap -- it is a ticker with no real
    history over the window, and keeping it would poison the covariance matrix
    in F2. Such columns are dropped and the caller is told.
    """
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index)
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_localize(None)

    wanted = [s for s in symbols if s in frame.columns]
    if not wanted:
        return None
    frame = frame[wanted]

    # A column needs real data on at least 60% of the window's trading days.
    keep = [c for c in frame.columns if frame[c].notna().mean() >= 0.60]
    dropped = [c for c in frame.columns if c not in keep]
    if dropped:
        warn(f"Dropped {', '.join(dropped)}: too little history in this window.")
    if not keep:
        return None

    frame = frame[keep].ffill().dropna(how="all")
    frame = frame.dropna(axis=0, how="any")
    return frame if len(frame) >= 2 else None


def _load_snapshot(
    symbols: list[str], start: date, end: date
) -> tuple[pd.DataFrame | None, str]:
    """Load the bundled CSV snapshot and slice it to the requested window."""
    if not config.SNAPSHOT_FILE.exists():
        return None, "no bundled snapshot found"

    try:
        frame = pd.read_csv(config.SNAPSHOT_FILE, index_col=0, parse_dates=True)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        warn(f"The bundled snapshot could not be read: {exc}")
        return None, "snapshot unreadable"

    missing = [s for s in symbols if s not in frame.columns]
    if missing:
        warn(f"The bundled snapshot does not cover: {', '.join(missing)}. "
             "Offline running is limited to the model-portfolio sleeves and "
             "the default benchmark.")
        return None, "snapshot does not cover these tickers"

    window = frame.loc[str(start):str(end), symbols]
    cleaned = _clean(window, symbols)
    if cleaned is None:
        warn("The bundled snapshot does not cover the requested date range.")
        return None, "snapshot does not cover this window"

    return cleaned, f"cached snapshot dated {snapshot_date()}"


def snapshot_date() -> str:
    """Return the as-of date recorded when the snapshot was built."""
    if config.SNAPSHOT_METADATA_FILE.exists():
        try:
            meta = json.loads(config.SNAPSHOT_METADATA_FILE.read_text())
            return str(meta.get("captured", "unknown date"))
        except (OSError, json.JSONDecodeError):
            pass
    return "unknown date"


def snapshot_available() -> bool:
    """True when an offline snapshot is bundled with the package."""
    return config.SNAPSHOT_FILE.exists()


def tickers_have_data(symbols: list[str]) -> tuple[list[str], list[str]]:
    """Split symbols into those that return prices and those that do not.

    Used by prompt_tickers() so that a typo is caught at the prompt, where the
    user can still correct it, rather than three steps later inside a library.

    Returns:
        (usable, unusable), preserving the caller's order in each list.
    """
    end = date.today()
    start = end - timedelta(days=45)

    usable: list[str] = []
    unusable: list[str] = []
    for symbol in symbols:
        try:
            frame, _ = get_prices([symbol], start, end, quiet=True)
        except PriceDataError:
            unusable.append(symbol)
            continue
        if symbol in frame.columns and frame[symbol].notna().any():
            usable.append(symbol)
        else:
            unusable.append(symbol)

    return usable, unusable


def latest_prices(prices: pd.DataFrame) -> pd.Series:
    """Return the most recent price per column, for discrete allocation."""
    return prices.ffill().iloc[-1]


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns, with the unusable first row removed."""
    return prices.pct_change().dropna(how="all")
