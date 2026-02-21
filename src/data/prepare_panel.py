"""
Panel Data Preparation
=======================
Cleans the raw price panel and prepares it for factor computation.

Steps:
1. Remove tickers with insufficient history
2. Filter out penny stocks
3. Compute daily returns
4. Add month-end indicators for rebalancing
5. Basic data quality checks

Usage:
    python -m src.data.prepare_panel
"""

import polars as pl

from src.utils.config import DATA_DIR, get_config, get_logger

logger = get_logger("data.prepare_panel")


def load_raw_panel() -> pl.DataFrame:
    """Load the combined price panel."""
    path = DATA_DIR / "processed" / "price_panel.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Price panel not found at {path}. "
            "Run `python -m src.data.fetch_prices` first."
        )
    return pl.read_parquet(path)


def clean_panel(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """
    Apply data quality filters.

    Filters:
    - Remove tickers with fewer than `min_history_days` observations
    - Remove dates where close < `min_price` (penny stocks)
    - Remove rows with zero or null volume
    """
    min_days = config["data"]["min_history_days"]
    min_price = config["data"]["min_price"]
    n_before = df["ticker"].n_unique()

    # Count observations per ticker
    ticker_counts = df.group_by("ticker").agg(pl.len().alias("n_obs"))
    valid_tickers = ticker_counts.filter(pl.col("n_obs") >= min_days)["ticker"]
    df = df.filter(pl.col("ticker").is_in(valid_tickers))

    # Remove penny stocks
    df = df.filter(pl.col("close") >= min_price)

    # Remove zero/null volume days
    df = df.filter(
        pl.col("volume").is_not_null() & (pl.col("volume") > 0)
    )

    n_after = df["ticker"].n_unique()
    logger.info(
        f"Cleaning: {n_before} → {n_after} tickers "
        f"(removed {n_before - n_after} with <{min_days} days or quality issues)"
    )

    return df


def add_returns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute daily simple and log returns.

    Uses adjusted close prices (already split/dividend adjusted by Tiingo).
    """
    df = df.sort(["ticker", "date"])

    df = df.with_columns(
        # Daily simple return
        (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1).alias(
            "ret_daily"
        ),
        # Daily log return
        (pl.col("close") / pl.col("close").shift(1).over("ticker"))
        .log()
        .alias("logret_daily"),
    )

    # First observation per ticker has null return — that's correct
    return df


def add_month_end_flag(df: pl.DataFrame) -> pl.DataFrame:
    """
    Flag the last trading day of each month for rebalancing.

    We identify month-ends as the last observed date within each (year, month)
    group, which naturally handles holidays and weekends.
    """
    df = df.with_columns(
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
    )

    # Last trading date per month (across all tickers)
    month_ends = (
        df.select("date", "year", "month")
        .unique()
        .group_by(["year", "month"])
        .agg(pl.col("date").max().alias("month_end_date"))
    )

    df = df.join(month_ends, on=["year", "month"], how="left")
    df = df.with_columns(
        (pl.col("date") == pl.col("month_end_date")).alias("is_month_end")
    )
    df = df.drop(["year", "month", "month_end_date"])

    n_rebal = df.filter(pl.col("is_month_end"))["date"].n_unique()
    logger.info(f"Identified {n_rebal} month-end rebalancing dates")

    return df


def run_data_quality_report(df: pl.DataFrame) -> None:
    """Print summary statistics for sanity checking."""
    logger.info("=" * 60)
    logger.info("DATA QUALITY REPORT")
    logger.info("=" * 60)

    n_tickers = df["ticker"].n_unique()
    n_rows = len(df)
    date_min = df["date"].min()
    date_max = df["date"].max()

    logger.info(f"Tickers:    {n_tickers}")
    logger.info(f"Rows:       {n_rows:,}")
    logger.info(f"Date range: {date_min} to {date_max}")

    # Check for extreme returns (potential data errors)
    ret_stats = df.select(
        pl.col("ret_daily").mean().alias("mean"),
        pl.col("ret_daily").std().alias("std"),
        pl.col("ret_daily").min().alias("min"),
        pl.col("ret_daily").max().alias("max"),
        pl.col("ret_daily").is_null().sum().alias("n_null"),
    )
    logger.info(f"Daily returns: {ret_stats.to_dicts()[0]}")

    # Flag suspicious returns (>50% daily move is almost certainly data error)
    extreme = df.filter(pl.col("ret_daily").abs() > 0.5)
    if len(extreme) > 0:
        logger.warning(
            f"⚠️  {len(extreme)} rows with |daily return| > 50% — "
            "inspect for data errors"
        )
        for row in extreme.head(5).iter_rows(named=True):
            logger.warning(
                f"   {row['ticker']} on {row['date']}: "
                f"ret={row['ret_daily']:.2%}, close={row['close']:.2f}"
            )

    logger.info("=" * 60)


def prepare_panel() -> pl.DataFrame:
    """Full pipeline: load → clean → returns → month flags → save."""
    config = get_config()

    logger.info("Loading raw price panel...")
    df = load_raw_panel()

    logger.info("Cleaning data...")
    df = clean_panel(df, config)

    logger.info("Computing returns...")
    df = add_returns(df)

    logger.info("Adding month-end flags...")
    df = add_month_end_flag(df)

    # Data quality check
    run_data_quality_report(df)

    # Save cleaned panel
    out_path = DATA_DIR / "processed" / "clean_panel.parquet"
    df.write_parquet(out_path)
    logger.info(f"Clean panel saved to {out_path}")

    return df


if __name__ == "__main__":
    prepare_panel()
