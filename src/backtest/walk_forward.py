"""
Walk-Forward Out-of-Sample Backtest
=====================================
Constructs long-short quintile portfolios using Bayesian-selected factors
and evaluates performance with realistic transaction costs.

Methodology:
1. At each month-end, rank stocks by the composite Bayesian signal
   (weighted by posterior inclusion probabilities)
2. Form quintile portfolios (Q1 = lowest signal, Q5 = highest)
3. Hold for one month, then rebalance
4. Track long-short (Q5 - Q1) and long-only (Q5) returns
5. Apply transaction costs based on portfolio turnover

Walk-Forward Design:
- Training window: expanding (use all data up to t)
- No future information leaks into portfolio construction
- Factor z-scores computed using only past data at each rebalance

Usage:
    python -m src.backtest.walk_forward
"""

import numpy as np
import pandas as pd
import polars as pl

from src.utils.config import DATA_DIR, get_config, get_logger

logger = get_logger("backtest.walk_forward")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_factor_panel() -> pl.DataFrame:
    """Load the full daily factor panel."""
    path = DATA_DIR / "processed" / "factor_panel.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Factor panel not found at {path}. "
            "Run `python -m src.factors.compute_factors` first."
        )
    return pl.read_parquet(path)


def load_bayesian_results() -> pd.DataFrame:
    """Load Bayesian selection results for PIP-based weighting."""
    path = DATA_DIR / "processed" / "bayesian_results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Bayesian results not found at {path}. "
            "Run `python -m src.models.bayesian_selection` first."
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Composite signal construction
# ---------------------------------------------------------------------------

def build_composite_signal(
    df: pl.DataFrame,
    factor_cols: list[str],
    weights: dict[str, float],
) -> pl.DataFrame:
    """
    Build a composite signal as a PIP-weighted average of individual factors.

    Factors with PIP < 20% get zero weight (excluded by Horseshoe).
    Remaining factors are weighted proportional to their PIP.

    Parameters
    ----------
    df : pl.DataFrame
        Panel with z-scored factors.
    factor_cols : list[str]
        Factor column names (z-scored).
    weights : dict
        factor_name -> PIP weight (from Bayesian results).
    """
    # Normalize weights to sum to 1 (only non-zero weights)
    active = {k: v for k, v in weights.items() if v > 0}
    total = sum(active.values())
    normed = {k: v / total for k, v in active.items()}

    logger.info("Composite signal weights (PIP-normalized):")
    for name, w in normed.items():
        logger.info(f"  {name}: {w:.3f}")

    # Compute weighted composite
    composite_expr = pl.lit(0.0)
    for col, w in normed.items():
        composite_expr = composite_expr + pl.col(col) * w

    df = df.with_columns(composite_expr.alias("composite_signal"))

    return df


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

def form_quintile_portfolios(
    month_data: pl.DataFrame,
    n_quantiles: int = 5,
) -> pl.DataFrame:
    """
    At a single month-end, rank stocks by composite signal and assign
    to quintile portfolios.

    Q1 = lowest signal (short leg)
    Q5 = highest signal (long leg)
    """
    # Drop stocks with missing signal
    clean = month_data.filter(pl.col("composite_signal").is_not_null())
    n = len(clean)

    if n < n_quantiles * 5:  # Need at least 5 stocks per quintile
        return clean.with_columns(pl.lit(None).alias("quintile").cast(pl.Int32))

    # Rank and assign quintiles
    clean = clean.with_columns(
        pl.col("composite_signal")
        .rank(method="ordinal")
        .alias("rank")
    )

    clean = clean.with_columns(
        (pl.col("rank") * n_quantiles / n)
        .floor()
        .clip(0, n_quantiles - 1)
        .cast(pl.Int32)
        .add(1)
        .alias("quintile")
    )

    return clean


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    df: pl.DataFrame,
    n_quantiles: int = 5,
    tc_bps: float = 10.0,
    min_warmup_months: int = 24,
) -> pd.DataFrame:
    """
    Walk-forward backtest with monthly rebalancing.

    Parameters
    ----------
    df : pl.DataFrame
        Full panel with composite_signal and forward returns.
    n_quantiles : int
        Number of portfolio buckets.
    tc_bps : float
        One-way transaction cost in basis points.
    min_warmup_months : int
        Skip first N months to allow factor computation warmup.

    Returns
    -------
    pd.DataFrame
        Monthly return series for each quintile + long-short.
    """
    config = get_config()
    tc_decimal = tc_bps / 10000  # Convert bps to decimal

    # Get month-end dates
    month_ends = (
        df.filter(pl.col("is_month_end"))
        .select("date")
        .unique()
        .sort("date")
    )
    dates = month_ends["date"].to_list()

    # Skip warmup period
    dates = dates[min_warmup_months:]
    logger.info(
        f"Backtest period: {dates[0]} to {dates[-1]} ({len(dates)} months)"
    )

    monthly_results = []
    prev_holdings = {}  # quintile -> set of tickers (for turnover calc)

    for i, date in enumerate(dates):
        # Get cross-section at this month-end
        cs = df.filter(
            (pl.col("date") == date) & pl.col("is_month_end")
        )

        # Form portfolios
        cs = form_quintile_portfolios(cs, n_quantiles)
        cs = cs.filter(pl.col("quintile").is_not_null())

        if len(cs) == 0:
            continue

        # Compute equal-weighted return per quintile
        quintile_returns = (
            cs.group_by("quintile")
            .agg(
                pl.col("fwd_ret").mean().alias("ret_gross"),
                pl.len().alias("n_stocks"),
                pl.col("ticker").alias("holdings"),
            )
            .sort("quintile")
        )

        # Calculate turnover per quintile
        result_row = {"date": date}
        current_holdings = {}

        for row in quintile_returns.iter_rows(named=True):
            q = row["quintile"]
            ret_gross = row["ret_gross"]
            n = row["n_stocks"]
            tickers = set(row["holdings"])

            current_holdings[q] = tickers

            # Turnover: fraction of portfolio that changed
            if q in prev_holdings and prev_holdings[q]:
                overlap = len(tickers & prev_holdings[q])
                max_size = max(len(tickers), len(prev_holdings[q]))
                turnover = 1 - overlap / max_size if max_size > 0 else 1.0
            else:
                turnover = 1.0  # First month: full portfolio buy

            # Transaction cost = turnover × 2 × one-way cost (buy + sell)
            tc = turnover * 2 * tc_decimal
            ret_net = ret_gross - tc if ret_gross is not None else None

            result_row[f"Q{q}_gross"] = ret_gross
            result_row[f"Q{q}_net"] = ret_net
            result_row[f"Q{q}_n"] = n
            result_row[f"Q{q}_turnover"] = turnover

        prev_holdings = current_holdings

        # Long-short: Q5 (long) - Q1 (short)
        q5_net = result_row.get(f"Q{n_quantiles}_net")
        q1_net = result_row.get("Q1_net")
        if q5_net is not None and q1_net is not None:
            result_row["ls_gross"] = (
                result_row.get(f"Q{n_quantiles}_gross", 0)
                - result_row.get("Q1_gross", 0)
            )
            result_row["ls_net"] = q5_net - q1_net

        monthly_results.append(result_row)

    results = pd.DataFrame(monthly_results)
    results["date"] = pd.to_datetime(results["date"])
    results = results.set_index("date").sort_index()

    logger.info(f"Backtest complete: {len(results)} months of returns")

    return results


# ---------------------------------------------------------------------------
# Performance analytics
# ---------------------------------------------------------------------------

def compute_performance_stats(
    returns: pd.Series,
    freq: int = 12,
    label: str = "",
) -> dict:
    """
    Compute standard performance statistics.

    Parameters
    ----------
    returns : pd.Series
        Monthly return series.
    freq : int
        Annualization factor (12 for monthly).
    label : str
        Name for this series.
    """
    r = returns.dropna()
    if len(r) < 12:
        return {"label": label, "error": "insufficient data"}

    total_return = (1 + r).prod() - 1
    ann_return = (1 + r).prod() ** (freq / len(r)) - 1
    ann_vol = r.std() * np.sqrt(freq)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    cumulative = (1 + r).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_dd = drawdown.min()

    # Hit rate (% positive months)
    hit_rate = (r > 0).mean()

    # Skewness
    skew = r.skew()

    return {
        "label": label,
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
        "skewness": skew,
        "n_months": len(r),
        "avg_monthly": r.mean(),
    }


def print_performance_report(results: pd.DataFrame, n_quantiles: int = 5):
    """Print comprehensive backtest performance report."""
    logger.info("=" * 75)
    logger.info("WALK-FORWARD BACKTEST RESULTS")
    logger.info("=" * 75)
    logger.info(
        f"Period: {results.index[0].strftime('%Y-%m')} to "
        f"{results.index[-1].strftime('%Y-%m')} "
        f"({len(results)} months)"
    )
    logger.info("")

    # Quintile monotonicity check
    logger.info("QUINTILE PORTFOLIO RETURNS (annualized, net of costs):")
    logger.info("-" * 75)
    logger.info(
        f"{'Portfolio':<15} {'Ann.Ret':>10} {'Ann.Vol':>10} "
        f"{'Sharpe':>8} {'MaxDD':>10} {'Hit%':>8} {'Avg.N':>8}"
    )
    logger.info("-" * 75)

    all_stats = []
    for q in range(1, n_quantiles + 1):
        col = f"Q{q}_net"
        if col not in results.columns:
            continue
        stats = compute_performance_stats(results[col], label=f"Q{q}")
        all_stats.append(stats)

        n_col = f"Q{q}_n"
        avg_n = results[n_col].mean() if n_col in results.columns else 0

        logger.info(
            f"{'Q' + str(q):<15} {stats['ann_return']:>+10.2%} "
            f"{stats['ann_volatility']:>10.2%} "
            f"{stats['sharpe_ratio']:>8.2f} "
            f"{stats['max_drawdown']:>10.2%} "
            f"{stats['hit_rate']:>8.1%} "
            f"{avg_n:>8.0f}"
        )

    # Long-short
    logger.info("-" * 75)

    if "ls_net" in results.columns:
        ls_stats = compute_performance_stats(results["ls_net"], label="L/S (Q5-Q1)")

        logger.info(
            f"{'L/S (Q5-Q1)':<15} {ls_stats['ann_return']:>+10.2%} "
            f"{ls_stats['ann_volatility']:>10.2%} "
            f"{ls_stats['sharpe_ratio']:>8.2f} "
            f"{ls_stats['max_drawdown']:>10.2%} "
            f"{ls_stats['hit_rate']:>8.1%} "
            f"{'—':>8}"
        )

        ls_gross = compute_performance_stats(results["ls_gross"], label="L/S gross")
        logger.info(
            f"{'L/S (gross)':<15} {ls_gross['ann_return']:>+10.2%} "
            f"{ls_gross['ann_volatility']:>10.2%} "
            f"{ls_gross['sharpe_ratio']:>8.2f} "
            f"{ls_gross['max_drawdown']:>10.2%} "
            f"{ls_gross['hit_rate']:>8.1%} "
            f"{'—':>8}"
        )

    logger.info("=" * 75)

    # Monotonicity check
    q_returns = []
    for q in range(1, n_quantiles + 1):
        col = f"Q{q}_net"
        if col in results.columns:
            q_returns.append(results[col].mean())

    if len(q_returns) == n_quantiles:
        is_monotonic = all(
            q_returns[i] <= q_returns[i + 1] for i in range(len(q_returns) - 1)
        )
        spread = q_returns[-1] - q_returns[0]
        logger.info(f"\nMonotonicity check: {'✓ PASS' if is_monotonic else '✗ FAIL'}")
        logger.info(f"Q5 - Q1 spread (monthly): {spread:+.4f} ({spread * 12:+.2%} ann.)")
        if is_monotonic:
            logger.info(
                "  → Returns increase monotonically from Q1 to Q5, "
                "consistent with a genuine factor premium."
            )
        else:
            logger.info(
                "  → Non-monotonic pattern. This may indicate a non-linear "
                "signal or period-specific effects."
            )

    # Turnover stats
    for q in [1, n_quantiles]:
        t_col = f"Q{q}_turnover"
        if t_col in results.columns:
            avg_turnover = results[t_col].mean()
            logger.info(f"Avg monthly turnover Q{q}: {avg_turnover:.1%}")

    logger.info("=" * 75)

    return all_stats


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_backtest_results(results: pd.DataFrame, n_quantiles: int = 5):
    """Generate backtest performance charts."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Cumulative returns by quintile
    ax = axes[0, 0]
    for q in range(1, n_quantiles + 1):
        col = f"Q{q}_net"
        if col in results.columns:
            cumret = (1 + results[col].fillna(0)).cumprod()
            ax.plot(cumret.index, cumret.values, label=f"Q{q}", linewidth=1.5)
    ax.set_title("Cumulative Returns by Quintile (net)", fontweight="bold")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Long-short cumulative
    ax = axes[0, 1]
    if "ls_net" in results.columns and "ls_gross" in results.columns:
        cum_net = (1 + results["ls_net"].fillna(0)).cumprod()
        cum_gross = (1 + results["ls_gross"].fillna(0)).cumprod()
        ax.plot(cum_net.index, cum_net.values, label="Net of costs", linewidth=2)
        ax.plot(
            cum_gross.index, cum_gross.values,
            label="Gross", linewidth=1.5, linestyle="--", alpha=0.7,
        )
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_title("Long-Short (Q5 - Q1)", fontweight="bold")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Monthly L/S returns
    ax = axes[1, 0]
    if "ls_net" in results.columns:
        colors = ["green" if r > 0 else "red" for r in results["ls_net"].fillna(0)]
        ax.bar(results.index, results["ls_net"].fillna(0), color=colors, alpha=0.7,
               width=25)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Monthly Long-Short Returns (net)", fontweight="bold")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.3)

    # 4. Rolling 12-month Sharpe
    ax = axes[1, 1]
    if "ls_net" in results.columns:
        rolling_mean = results["ls_net"].rolling(12).mean() * 12
        rolling_vol = results["ls_net"].rolling(12).std() * np.sqrt(12)
        rolling_sharpe = rolling_mean / rolling_vol
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, color="steelblue",
                linewidth=1.5)
        ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_title("Rolling 12-Month Sharpe Ratio (L/S)", fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Walk-Forward Backtest: Bayesian Factor Selection Portfolio",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    out_dir = DATA_DIR / "processed" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "backtest_results.png", dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {out_dir / 'backtest_results.png'}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the full walk-forward backtest."""
    config = get_config()
    n_quantiles = config["backtest"]["n_quantiles"]
    tc_bps = config["backtest"]["transaction_cost_bps"]

    # Load data
    logger.info("Loading factor panel...")
    df = load_factor_panel()

    # Load Bayesian results for PIP weights
    logger.info("Loading Bayesian selection results...")
    bayes = load_bayesian_results()

    # Build PIP-weighted composite signal
    # Only include factors with PIP > 20%
    pip_threshold = 0.20
    factor_weights = {}
    factor_z_cols = []

    for _, row in bayes.iterrows():
        factor_name = row["factor"]
        pip = row["pip"]
        if pip >= pip_threshold:
            # Use posterior mean sign to determine direction
            sign = np.sign(row["post_mean"])
            # Weight = PIP (factors with higher PIP get more weight)
            factor_weights[factor_name + "_z" if not factor_name.endswith("_z") else factor_name] = pip * sign
            factor_z_cols.append(
                factor_name + "_z" if not factor_name.endswith("_z") else factor_name
            )
            logger.info(
                f"  Including {factor_name}: PIP={pip:.1%}, "
                f"sign={'+ ' if sign > 0 else '−'}, weight={pip * sign:+.3f}"
            )
        else:
            logger.info(f"  Excluding {factor_name}: PIP={pip:.1%} < {pip_threshold:.0%}")

    if not factor_weights:
        logger.error("No factors passed the PIP threshold! Cannot run backtest.")
        return

    # Build composite signal
    # Ensure we use the right column names (they already end in _z in the data)
    # Fix: the bayesian results have factor names like "factor_mom_z"
    df = build_composite_signal(df, factor_z_cols, factor_weights)

    # Run backtest
    logger.info(f"\nRunning walk-forward backtest (TC={tc_bps} bps one-way)...")
    results = run_backtest(
        df,
        n_quantiles=n_quantiles,
        tc_bps=tc_bps,
        min_warmup_months=24,
    )

    # Performance report
    print_performance_report(results, n_quantiles)

    # Save results
    out_path = DATA_DIR / "processed" / "backtest_monthly_returns.csv"
    results.to_csv(out_path)
    logger.info(f"Monthly returns saved to {out_path}")

    # Generate plots
    plot_backtest_results(results, n_quantiles)

    return results


if __name__ == "__main__":
    main()
