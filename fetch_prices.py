"""
Price Data Fetcher (Tiingo API)
================================
Downloads adjusted daily OHLCV data for all tickers in the universe.

Tiingo free tier: 500 requests/hour, 50 requests/day for EOD.
Paid tier (starter $10/mo): 5000 requests/hour — recommended.

The script is designed to be **resumable**: it skips tickers that already have
data on disk, so you can re-run after interruptions.

Usage:
    python -m src.data.fetch_prices
"""

import time
from pathlib import Path

import polars as pl
import requests

from src.data.universe import load_universe
from src.utils.config import DATA_DIR, get_config, get_logger, get_tiingo_api_key

logger = get_logger("data.fetch_prices")


def fetch_ticker_prices(
    ticker: str,
    api_key: str,
    start_date: str,
    end_date: str,
) -> pl.DataFrame | None:
    """
    Fetch adjusted daily prices for a single ticker from Tiingo.

    Returns None if the request fails (e.g., ticker not found).
    """
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {
        "startDate": start_date,
        "endDate": end_date,
        "format": "json",
        "resampleFreq": "daily",
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Token {api_key}"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)

        if resp.status_code == 404:
            logger.warning(f"  {ticker}: not found on Tiingo (404)")
            return None
        if resp.status_code == 429:
            logger.warning(f"  {ticker}: rate limited (429), sleeping 60s...")
            time.sleep(60)
            return fetch_ticker_prices(ticker, api_key, start_date, end_date)

        resp.raise_for_status()
        data = resp.json()

        if not data:
            logger.warning(f"  {ticker}: empty response")
            return None

        df = pl.DataFrame(data)

        # Tiingo returns ISO datetime strings; parse to date
        df = df.with_columns(
            pl.col("date").str.slice(0, 10).str.to_date("%Y-%m-%d").alias("date")
        )

        # Select and rename columns we need
        # adjClose, adjHigh, adjLow, adjOpen, adjVolume are split/dividend-adjusted
        df = df.select(
            pl.lit(ticker).alias("ticker"),
            pl.col("date"),
            pl.col("adjOpen").alias("open"),
            pl.col("adjHigh").alias("high"),
            pl.col("adjLow").alias("low"),
            pl.col("adjClose").alias("close"),
            pl.col("adjVolume").alias("volume"),
            # Keep raw close for reference
            pl.col("close").alias("close_raw"),
        )

        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"  {ticker}: request failed — {e}")
        return None


def fetch_all_prices(
    rate_limit_sleep: float = 0.25,
) -> None:
    """
    Fetch prices for all tickers in the universe. Saves per-ticker parquet files
    for resumability, then combines into a single panel.
    """
    config = get_config()
    api_key = get_tiingo_api_key()
    universe = load_universe()
    tickers = universe["ticker"].to_list()

    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]

    raw_dir = DATA_DIR / "raw" / "prices"
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Fetching daily prices for {len(tickers)} tickers "
        f"({start_date} to {end_date})"
    )

    success = 0
    skipped = 0
    failed = 0

    for i, ticker in enumerate(tickers):
        ticker_path = raw_dir / f"{ticker}.parquet"

        # Skip if already downloaded (resumable)
        if ticker_path.exists():
            skipped += 1
            continue

        logger.info(f"  [{i+1}/{len(tickers)}] Fetching {ticker}...")
        df = fetch_ticker_prices(ticker, api_key, start_date, end_date)

        if df is not None and len(df) > 0:
            df.write_parquet(ticker_path)
            success += 1
        else:
            failed += 1

        # Respect rate limits
        time.sleep(rate_limit_sleep)

    logger.info(
        f"Done: {success} downloaded, {skipped} already cached, {failed} failed"
    )

    # Combine into single panel
    _combine_to_panel(raw_dir)


def _combine_to_panel(raw_dir: Path) -> None:
    """Combine per-ticker parquet files into a single panel dataset."""
    logger.info("Combining per-ticker files into panel dataset...")

    parquet_files = sorted(raw_dir.glob("*.parquet"))
    if not parquet_files:
        logger.error("No parquet files found to combine!")
        return

    dfs = []
    for f in parquet_files:
        try:
            df = pl.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"  Could not read {f.name}: {e}")

    panel = pl.concat(dfs)
    panel = panel.sort(["ticker", "date"])

    out_path = DATA_DIR / "processed" / "price_panel.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(out_path)

    n_tickers = panel["ticker"].n_unique()
    n_rows = len(panel)
    date_range = f"{panel['date'].min()} to {panel['date'].max()}"

    logger.info(
        f"Panel saved: {n_tickers} tickers, {n_rows:,} rows, {date_range}"
    )


if __name__ == "__main__":
    fetch_all_prices()
