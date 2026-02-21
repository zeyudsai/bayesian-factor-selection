"""
Fama-MacBeth Cross-Sectional Regression (Baseline)
====================================================
Implements the two-pass Fama-MacBeth (1973) regression procedure as a
baseline model before applying Bayesian methods.

Procedure:
1. At each month-end t, run cross-sectional OLS:
       r_{i,t→t+1} = α_t + β_{1,t} * factor1_i,t + ... + ε_{i,t}
2. Collect the time series of estimated coefficients {β̂_t}.
3. Test whether the mean coefficient is significantly different from zero
   using Newey-West standard errors (to handle serial correlation).

This serves as a sanity check: if classic factors (momentum, size, etc.)
don't show expected signs and significance, the data pipeline has issues.

Usage:
    python -m src.models.fama_macbeth
"""

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from src.utils.config import DATA_DIR, get_logger

logger = get_logger("models.fama_macbeth")


def load_monthly_factors() -> pl.DataFrame:
    """Load month-end factor panel."""
    path = DATA_DIR / "processed" / "factor_monthly.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Monthly factor panel not found at {path}. "
            "Run `python -m src.factors.compute_factors` first."
        )
    return pl.read_parquet(path)


def newey_west_se(x: np.ndarray, max_lag: int | None = None) -> float:
    """
    Newey-West standard error for the mean of a time series.

    Accounts for serial correlation in the Fama-MacBeth coefficient
    time series, which is critical for valid inference.

    Parameters
    ----------
    x : array-like
        Time series of estimated coefficients.
    max_lag : int, optional
        Maximum lag for autocovariance. Default: floor(4 * (T/100)^(2/9)).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    T = len(x)

    if T < 5:
        return np.nan

    if max_lag is None:
        max_lag = int(np.floor(4 * (T / 100) ** (2 / 9)))
        max_lag = max(max_lag, 1)

    x_demeaned = x - np.mean(x)

    # Gamma_0 (variance)
    gamma_0 = np.sum(x_demeaned**2) / T

    # Add autocovariance terms with Bartlett kernel
    nw_var = gamma_0
    for lag in range(1, max_lag + 1):
        weight = 1 - lag / (max_lag + 1)
        gamma_lag = np.sum(x_demeaned[lag:] * x_demeaned[:-lag]) / T
        nw_var += 2 * weight * gamma_lag

    # Standard error of the mean
    se = np.sqrt(nw_var / T)
    return se


def run_fama_macbeth(
    df: pl.DataFrame,
    factor_cols: list[str],
    target_col: str = "fwd_ret_xs",
) -> pd.DataFrame:
    """
    Run Fama-MacBeth two-pass regression.

    Returns a DataFrame with one row per factor, showing:
    - mean_coef: time-series average of cross-sectional slopes
    - nw_se: Newey-West standard error
    - t_stat: mean_coef / nw_se
    - p_value: two-sided p-value
    - n_months: number of cross-sections used
    """
    logger.info(f"Running Fama-MacBeth with factors: {factor_cols}")
    logger.info(f"Target: {target_col}")

    # Get unique rebalancing dates
    dates = sorted(df.filter(pl.col(target_col).is_not_null())["date"].unique().to_list())
    logger.info(f"Number of cross-sections (months): {len(dates)}")

    # First pass: cross-sectional regressions
    all_coefs = {col: [] for col in ["intercept"] + factor_cols}
    all_dates = []
    n_stocks_per_month = []

    for date in dates:
        # Get cross-section for this date
        cs = df.filter(pl.col("date") == date)

        # Drop rows with any missing factor or target
        cols_needed = factor_cols + [target_col]
        cs = cs.drop_nulls(subset=cols_needed)

        n_stocks = len(cs)
        if n_stocks < 30:  # Need minimum stocks for meaningful regression
            continue

        # Extract arrays for OLS
        y = cs[target_col].to_numpy()
        X = np.column_stack([cs[col].to_numpy() for col in factor_cols])
        X = np.column_stack([np.ones(n_stocks), X])  # Add intercept

        # OLS: β = (X'X)^{-1} X'y
        try:
            coefs = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        all_dates.append(date)
        all_coefs["intercept"].append(coefs[0])
        for i, col in enumerate(factor_cols):
            all_coefs[col].append(coefs[i + 1])
        n_stocks_per_month.append(n_stocks)

    # Second pass: time-series analysis of coefficients
    results = []
    for col in ["intercept"] + factor_cols:
        coef_series = np.array(all_coefs[col])
        mean_coef = np.nanmean(coef_series)
        nw_se = newey_west_se(coef_series)
        t_stat = mean_coef / nw_se if nw_se > 0 else np.nan
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(coef_series) - 1))

        results.append(
            {
                "factor": col,
                "mean_coef": mean_coef,
                "nw_se": nw_se,
                "t_stat": t_stat,
                "p_value": p_value,
                "n_months": len(coef_series),
                "pct_positive": np.mean(coef_series > 0),
            }
        )

    results_df = pd.DataFrame(results)

    # Also save the coefficient time series for later analysis
    coef_ts = pd.DataFrame(all_coefs)
    coef_ts["date"] = all_dates
    coef_ts["n_stocks"] = n_stocks_per_month
    coef_ts_path = DATA_DIR / "processed" / "fama_macbeth_coef_ts.csv"
    coef_ts.to_csv(coef_ts_path, index=False)
    logger.info(f"Coefficient time series saved to {coef_ts_path}")

    return results_df


def print_results(results: pd.DataFrame) -> None:
    """Pretty-print Fama-MacBeth results."""
    logger.info("=" * 70)
    logger.info("FAMA-MACBETH REGRESSION RESULTS")
    logger.info("=" * 70)
    logger.info(
        f"{'Factor':<20} {'Mean β':>10} {'NW SE':>10} "
        f"{'t-stat':>10} {'p-value':>10} {'% pos':>8}"
    )
    logger.info("-" * 70)

    for _, row in results.iterrows():
        sig = ""
        if row["p_value"] < 0.01:
            sig = "***"
        elif row["p_value"] < 0.05:
            sig = "**"
        elif row["p_value"] < 0.10:
            sig = "*"

        logger.info(
            f"{row['factor']:<20} {row['mean_coef']:>10.5f} {row['nw_se']:>10.5f} "
            f"{row['t_stat']:>9.2f}{sig:<3} {row['p_value']:>10.4f} "
            f"{row['pct_positive']:>7.1%}"
        )

    logger.info("-" * 70)
    logger.info("Significance: *** p<0.01, ** p<0.05, * p<0.10")
    logger.info("Standard errors: Newey-West adjusted for serial correlation")
    logger.info("")
    logger.info("EXPECTED SIGNS (sanity check):")
    logger.info("  Momentum (MOM):  POSITIVE  (past winners continue to win)")
    logger.info("  Volatility (VOL): NEGATIVE  (low-vol anomaly)")
    logger.info("  Size (SIZE):     NEGATIVE  (small firms outperform)")
    logger.info("  ST Reversal (STR): NEGATIVE (short-term mean reversion)")
    logger.info("=" * 70)


def main():
    """Run the full Fama-MacBeth analysis."""
    df = load_monthly_factors()

    # Use z-scored factors
    factor_cols = [
        "factor_mom_z",
        "factor_vol_z",
        "factor_size_z",
        "factor_str_z",
    ]

    results = run_fama_macbeth(df, factor_cols, target_col="fwd_ret_xs")
    print_results(results)

    # Save results
    out_path = DATA_DIR / "processed" / "fama_macbeth_results.csv"
    results.to_csv(out_path, index=False)
    logger.info(f"Results saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
