"""
Factor Computation
===================
Computes cross-sectional equity factors from the cleaned price panel.

Factors implemented:
1. **Momentum (MOM)**: Classic 12-1 month momentum (Jegadeesh & Titman, 1993)
2. **Volatility (VOL)**: Realized volatility over past 21 trading days
3. **Size (SIZE)**: Log market capitalization (or proxy)
4. **Short-Term Reversal (STR)**: Past 1-month return (Jegadeesh, 1990)

Target:
- Forward 1-month excess return (cross-sectionally demeaned)

All factors are cross-sectionally z-scored at each rebalancing date to ensure
comparability across time.

Usage:
    python -m src.factors.compute_factors
"""

import polars as pl
import numpy as np

from src.utils.config import DATA_DIR, get_config, get_logger

logger = get_logger("factors.compute")


def load_clean_panel() -> pl.DataFrame:
    """Load the cleaned price panel."""
    path = DATA_DIR / "processed" / "clean_panel.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Clean panel not found at {path}. "
            "Run `python -m src.data.prepare_panel` first."
        )
    return pl.read_parquet(path)


def compute_momentum(df: pl.DataFrame, lookback: int, skip: int) -> pl.DataFrame:
    """
    12-1 Month Momentum: cumulative return from t-lookback to t-skip.

    This is the classic momentum signal: buy past winners, sell past losers,
    but skip the most recent month to avoid short-term reversal contamination.
    """
    logger.info(f"Computing momentum (lookback={lookback}, skip={skip})...")

    df = df.sort(["ticker", "date"])

    df = df.with_columns(
        # Cumulative return over full lookback window
        (
            pl.col("close")
            / pl.col("close").shift(lookback).over("ticker")
            - 1
        ).alias("ret_12m"),
        # Return over skip period (most recent month)
        (
            pl.col("close")
            / pl.col("close").shift(skip).over("ticker")
            - 1
        ).alias("ret_1m"),
    )

    # Momentum = 12m return minus 1m return
    df = df.with_columns(
        (pl.col("ret_12m") - pl.col("ret_1m")).alias("factor_mom")
    )

    return df


def compute_volatility(df: pl.DataFrame, window: int) -> pl.DataFrame:
    """
    Realized volatility: rolling standard deviation of daily returns.

    Low-volatility anomaly: stocks with lower volatility tend to outperform
    on a risk-adjusted basis (Ang et al., 2006).
    """
    logger.info(f"Computing realized volatility (window={window})...")

    df = df.sort(["ticker", "date"])

    df = df.with_columns(
        pl.col("ret_daily")
        .rolling_std(window_size=window)
        .over("ticker")
        .alias("factor_vol")
    )

    # Annualize for interpretability (but z-scoring makes this cosmetic)
    df = df.with_columns(
        (pl.col("factor_vol") * np.sqrt(252)).alias("factor_vol_ann")
    )

    return df


def compute_size(df: pl.DataFrame) -> pl.DataFrame:
    """
    Size factor: log of market cap proxy.

    Ideally we'd use shares outstanding × price. As a proxy with price-only
    data, we use a rolling average of dollar volume as a liquidity/size proxy,
    or market cap if available from the data source.

    We use log transform since market cap is highly right-skewed.
    """
    logger.info("Computing size factor...")

    df = df.sort(["ticker", "date"])

    # Dollar volume as size proxy: close × volume, smoothed over 21 days
    df = df.with_columns(
        (pl.col("close") * pl.col("volume")).alias("dollar_volume")
    )

    df = df.with_columns(
        pl.col("dollar_volume")
        .rolling_mean(window_size=21)
        .over("ticker")
        .log()
        .alias("factor_size")
    )

    return df


def compute_short_term_reversal(df: pl.DataFrame) -> pl.DataFrame:
    """
    Short-term reversal: past 1-month return.

    Included separately from momentum as it captures a distinct effect
    (Jegadeesh, 1990). Expected sign: negative (past losers outperform).
    """
    logger.info("Computing short-term reversal...")

    # ret_1m already computed in momentum step
    df = df.with_columns(pl.col("ret_1m").alias("factor_str"))

    return df


def compute_forward_returns(df: pl.DataFrame, forward_days: int) -> pl.DataFrame:
    """
    Compute forward returns as the prediction target.

    Target = forward N-day return, cross-sectionally demeaned (excess of
    cross-sectional median) to remove market-level movements.
    """
    logger.info(f"Computing forward {forward_days}-day returns...")

    df = df.sort(["ticker", "date"])

    # Forward return
    df = df.with_columns(
        (
            pl.col("close").shift(-forward_days).over("ticker")
            / pl.col("close")
            - 1
        ).alias("fwd_ret")
    )

    # Cross-sectional excess return (subtract daily cross-sectional median)
    df = df.with_columns(
        (
            pl.col("fwd_ret")
            - pl.col("fwd_ret").median().over("date")
        ).alias("fwd_ret_xs")
    )

    return df


def zscore_factors(df: pl.DataFrame, factor_cols: list[str]) -> pl.DataFrame:
    """
    Cross-sectionally z-score all factors at each date.

    Z-scoring ensures factors are comparable in magnitude and prevents
    any single factor from dominating in the regression purely due to scale.
    Uses median and MAD for robustness to outliers.
    """
    logger.info("Z-scoring factors cross-sectionally...")

    for col in factor_cols:
        # Robust z-score: (x - median) / MAD
        # MAD = median absolute deviation, scaled by 1.4826 for normal consistency
        zscore_col = f"{col}_z"
        df = df.with_columns(
            (
                (pl.col(col) - pl.col(col).median().over("date"))
                / (
                    (pl.col(col) - pl.col(col).median().over("date"))
                    .abs()
                    .median()
                    .over("date")
                    * 1.4826
                )
            ).alias(zscore_col)
        )

        # Winsorize at ±3 to limit outlier influence
        df = df.with_columns(
            pl.col(zscore_col).clip(-3.0, 3.0).alias(zscore_col)
        )

    return df


def compute_all_factors() -> pl.DataFrame:
    """Full factor computation pipeline."""
    config = get_config()
    fconfig = config["factors"]

    df = load_clean_panel()
    logger.info(f"Loaded panel: {df['ticker'].n_unique()} tickers, {len(df):,} rows")

    # Compute individual factors
    df = compute_momentum(df, fconfig["momentum_lookback"], fconfig["momentum_skip"])
    df = compute_volatility(df, fconfig["volatility_window"])
    df = compute_size(df)
    df = compute_short_term_reversal(df)
    df = compute_forward_returns(df, fconfig["forward_return_days"])

    # Z-score factors
    factor_cols = ["factor_mom", "factor_vol", "factor_size", "factor_str"]
    df = zscore_factors(df, factor_cols)

    # Report coverage
    month_end_data = df.filter(pl.col("is_month_end"))
    z_cols = [f"{c}_z" for c in factor_cols]

    for col in z_cols + ["fwd_ret_xs"]:
        n_valid = month_end_data.select(pl.col(col).is_not_null().sum()).item()
        n_total = len(month_end_data)
        logger.info(f"  {col}: {n_valid}/{n_total} valid ({n_valid/n_total:.1%})")

    # Save
    out_path = DATA_DIR / "processed" / "factor_panel.parquet"
    df.write_parquet(out_path)
    logger.info(f"Factor panel saved to {out_path} ({len(df):,} rows)")

    # Also save a month-end only snapshot for regression
    month_end_path = DATA_DIR / "processed" / "factor_monthly.parquet"
    month_end_df = df.filter(pl.col("is_month_end"))
    month_end_df.write_parquet(month_end_path)
    logger.info(
        f"Monthly snapshot saved to {month_end_path} "
        f"({len(month_end_df):,} rows, {month_end_df['date'].n_unique()} months)"
    )

    return df


if __name__ == "__main__":
    compute_all_factors()
