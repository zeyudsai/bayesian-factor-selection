"""
Bayesian Factor Selection via Horseshoe Prior
===============================================
The core contribution of this project: using Bayesian shrinkage to perform
principled factor selection, addressing the "factor zoo" multiple testing problem.

Why Horseshoe?
- Unlike Lasso (L1), the Horseshoe prior has heavy tails that preserve truly
  large signals while aggressively shrinking noise toward zero.
- Unlike simple t-tests, we get posterior inclusion probabilities — a continuous
  measure of each factor's importance rather than binary significance.
- The regularized (Finnish) Horseshoe adds a slab component to prevent
  unshrunk coefficients from exploding.

Reference:
- Carvalho, Polson & Scott (2010). The Horseshoe Estimator for Sparse Signals.
- Piironen & Vehtari (2017). Sparsity Information and Regularization in the
  Horseshoe and Other Shrinkage Priors.

Usage:
    python -m src.models.bayesian_selection
"""

import arviz as az
import numpy as np
import pandas as pd
import polars as pl
import pymc as pm

from src.utils.config import DATA_DIR, get_logger

logger = get_logger("models.bayesian")


def load_monthly_factors() -> pl.DataFrame:
    """Load month-end factor panel."""
    path = DATA_DIR / "processed" / "factor_monthly.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Monthly factor panel not found at {path}. "
            "Run `python -m src.factors.compute_factors` first."
        )
    return pl.read_parquet(path)


def prepare_pooled_data(
    df: pl.DataFrame,
    factor_cols: list[str],
    target_col: str = "fwd_ret_xs",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Prepare pooled cross-sectional data for Bayesian regression.

    We pool all month-end cross-sections into a single dataset.
    This is a simplification vs. running per-month regressions, but it
    gives us a single posterior over factor loadings — which is what we
    want for factor selection.
    """
    cols_needed = factor_cols + [target_col]
    df_clean = df.drop_nulls(subset=cols_needed)

    n_before = len(df)
    n_after = len(df_clean)
    logger.info(
        f"Pooled data: {n_after:,} obs ({n_before - n_after:,} dropped due to NaN)"
    )

    y = df_clean[target_col].to_numpy().astype(np.float64)
    X = np.column_stack(
        [df_clean[col].to_numpy().astype(np.float64) for col in factor_cols]
    )

    return X, y, factor_cols


def build_horseshoe_model(
    X: np.ndarray,
    y: np.ndarray,
    p0: float = 2.0,
    slab_scale: float = 2.0,
    slab_df: float = 4.0,
) -> pm.Model:
    """
    Regularized (Finnish) Horseshoe regression model.

    Uses non-centered parameterization for better NUTS sampling geometry.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Factor matrix (already z-scored).
    y : ndarray of shape (n,)
        Forward excess returns.
    p0 : float
        Prior guess for the number of relevant factors.
    slab_scale : float
        Scale of the slab component.
    slab_df : float
        Degrees of freedom for the slab.
    """
    n, p = X.shape
    logger.info(f"Building Horseshoe model: n={n:,}, p={p} factors")
    logger.info(f"Prior expected sparsity: {p0:.0f} out of {p} factors relevant")

    # Global shrinkage scale (Piironen & Vehtari, 2017)
    tau0 = p0 / (p - p0) / np.sqrt(n)
    logger.info(f"Global shrinkage scale τ₀ = {tau0:.6f}")

    with pm.Model() as model:
        # Observation noise
        sigma = pm.HalfNormal("sigma", sigma=0.1)

        # Intercept
        alpha = pm.Normal("alpha", mu=0, sigma=0.01)

        # Global shrinkage
        tau = pm.HalfCauchy("tau", beta=tau0)

        # Local shrinkage (one per factor)
        lam = pm.HalfCauchy("lam", beta=1.0, shape=p)

        # Slab component (regularized horseshoe)
        c2 = pm.InverseGamma(
            "c2", alpha=slab_df / 2, beta=slab_df / 2 * slab_scale**2
        )

        # Effective shrunk scale per factor
        lam_tilde_sq = c2 * lam**2 / (c2 + tau**2 * lam**2)

        # Factor coefficients (non-centered parameterization)
        beta_raw = pm.Normal("beta_raw", mu=0, sigma=1, shape=p)
        beta = pm.Deterministic(
            "beta", beta_raw * tau * pm.math.sqrt(lam_tilde_sq)
        )

        # Shrinkage factors: kappa near 1 = fully shrunk, near 0 = retained
        pm.Deterministic("kappa", 1.0 / (1.0 + tau**2 * lam**2))

        # Likelihood
        mu = alpha + pm.math.dot(X, beta)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

    return model


def fit_horseshoe(
    X: np.ndarray,
    y: np.ndarray,
    factor_names: list[str],
    n_samples: int = 2000,
    n_tune: int = 2000,
    n_chains: int = 4,
    target_accept: float = 0.95,
    p0: float = 2.0,
) -> tuple[pm.Model, az.InferenceData]:
    """Build and sample the Horseshoe model."""
    model = build_horseshoe_model(X, y, p0=p0)

    logger.info(
        f"Sampling: {n_chains} chains × {n_samples} samples "
        f"(+ {n_tune} tuning), target_accept={target_accept}"
    )

    with model:
        idata = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            target_accept=target_accept,
            random_seed=42,
            return_inferencedata=True,
        )

    return model, idata


def analyze_results(
    idata: az.InferenceData,
    factor_names: list[str],
) -> pd.DataFrame:
    """
    Extract and summarize posterior results.

    Key outputs:
    - Posterior mean and 95% HDI for each factor coefficient
    - Shrinkage factor κ: 1 = fully shrunk, 0 = retained
    - Posterior inclusion probability: P(|β| > threshold)
    - Convergence diagnostics: R-hat and ESS
    """
    logger.info("=" * 70)
    logger.info("BAYESIAN HORSESHOE FACTOR SELECTION RESULTS")
    logger.info("=" * 70)

    # Extract beta posteriors: shape (chains, draws, p)
    beta_samples = idata.posterior["beta"].values
    beta_flat = beta_samples.reshape(-1, beta_samples.shape[-1])

    # Extract kappa (shrinkage)
    kappa_samples = idata.posterior["kappa"].values
    kappa_flat = kappa_samples.reshape(-1, kappa_samples.shape[-1])

    # Compute R-hat and ESS for the full beta variable
    rhat_ds = az.rhat(idata, var_names=["beta"])
    ess_ds = az.ess(idata, var_names=["beta"])
    rhat_values = rhat_ds["beta"].values.flatten()
    ess_values = ess_ds["beta"].values.flatten()

    results = []
    for i, name in enumerate(factor_names):
        samples = beta_flat[:, i]
        kappa_i = kappa_flat[:, i]

        mean = float(np.mean(samples))
        std = float(np.std(samples))
        hdi = az.hdi(samples, hdi_prob=0.95)

        # Posterior inclusion probability:
        # P(|β| > 0.0005) i.e. 0.05% monthly return per 1-SD factor exposure
        pip = float(np.mean(np.abs(samples) > 0.0005))

        # Mean shrinkage
        mean_kappa = float(np.mean(kappa_i))

        results.append(
            {
                "factor": name,
                "post_mean": mean,
                "post_std": std,
                "hdi_2.5%": float(hdi[0]),
                "hdi_97.5%": float(hdi[1]),
                "pip": pip,
                "shrinkage_kappa": mean_kappa,
                "signal_strength": 1 - mean_kappa,
                "rhat": float(rhat_values[i]),
                "ess_bulk": float(ess_values[i]),
            }
        )

    results_df = pd.DataFrame(results)

    # Print results
    logger.info(
        f"{'Factor':<20} {'Post.Mean':>10} {'95% HDI':>22} "
        f"{'PIP':>8} {'κ (shrink)':>10} {'R-hat':>7}"
    )
    logger.info("-" * 80)

    for _, row in results_df.iterrows():
        hdi_str = f"[{row['hdi_2.5%']:+.5f}, {row['hdi_97.5%']:+.5f}]"
        logger.info(
            f"{row['factor']:<20} {row['post_mean']:>+10.5f} {hdi_str:>22} "
            f"{row['pip']:>8.1%} {row['shrinkage_kappa']:>10.3f} "
            f"{row['rhat']:>7.3f}"
        )

    logger.info("-" * 80)
    logger.info("")
    logger.info("INTERPRETATION GUIDE:")
    logger.info("  PIP (Posterior Inclusion Prob): >80% = strong signal, <20% = noise")
    logger.info("  κ (shrinkage): 1.0 = fully shrunk to zero, 0.0 = unshrunk")
    logger.info("  R-hat: should be <1.01 for convergence")
    logger.info("  95% HDI: if it excludes zero, the factor is credibly nonzero")
    logger.info("=" * 70)

    return results_df


def compare_with_frequentist(
    bayes_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load Fama-MacBeth results and compare side-by-side with Bayesian results.
    """
    fm_path = DATA_DIR / "processed" / "fama_macbeth_results.csv"
    if not fm_path.exists():
        logger.warning("Fama-MacBeth results not found, skipping comparison")
        return bayes_results

    fm = pd.read_csv(fm_path)
    fm = fm[fm["factor"] != "intercept"].copy()

    comparison = bayes_results.merge(
        fm[["factor", "mean_coef", "t_stat", "p_value"]],
        on="factor",
        how="left",
        suffixes=("_bayes", "_fm"),
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("BAYESIAN vs FREQUENTIST COMPARISON")
    logger.info("=" * 70)
    logger.info(
        f"{'Factor':<20} {'FM t-stat':>10} {'FM p-val':>10} "
        f"{'Bayes PIP':>10} {'κ':>8} {'Agreement':>12}"
    )
    logger.info("-" * 70)

    for _, row in comparison.iterrows():
        fm_sig = row["p_value"] < 0.05 if pd.notna(row["p_value"]) else None
        bayes_sig = row["pip"] > 0.80

        if fm_sig is not None:
            agree = "✓ AGREE" if fm_sig == bayes_sig else "✗ DISAGREE"
        else:
            agree = "N/A"

        logger.info(
            f"{row['factor']:<20} {row.get('t_stat', np.nan):>10.2f} "
            f"{row.get('p_value', np.nan):>10.4f} "
            f"{row['pip']:>10.1%} {row['shrinkage_kappa']:>8.3f} "
            f"{agree:>12}"
        )

    logger.info("=" * 70)
    return comparison


def main():
    """Run the full Bayesian factor selection pipeline."""
    df = load_monthly_factors()

    factor_cols = [
        "factor_mom_z",
        "factor_vol_z",
        "factor_size_z",
        "factor_str_z",
    ]

    # Prepare data
    X, y, names = prepare_pooled_data(df, factor_cols)

    # Subsample for faster MCMC if dataset is very large
    n = len(y)
    if n > 50000:
        logger.info(f"Subsampling from {n:,} to 50,000 for faster MCMC...")
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=50000, replace=False)
        X = X[idx]
        y = y[idx]

    # Fit Horseshoe model
    model, idata = fit_horseshoe(
        X, y, names,
        n_samples=2000,
        n_tune=2000,
        n_chains=4,
        target_accept=0.95,
        p0=2.0,
    )

    # Analyze results
    results = analyze_results(idata, names)

    # Compare with Fama-MacBeth
    comparison = compare_with_frequentist(results)

    # Save results
    out_dir = DATA_DIR / "processed"
    results.to_csv(out_dir / "bayesian_results.csv", index=False)
    comparison.to_csv(out_dir / "bayesian_vs_frequentist.csv", index=False)

    # Save ArviZ InferenceData for later plotting
    idata.to_netcdf(str(out_dir / "horseshoe_trace.nc"))

    logger.info(f"\nResults saved to {out_dir}")
    logger.info("Trace saved as horseshoe_trace.nc (load with az.from_netcdf)")

    return model, idata, results


if __name__ == "__main__":
    main()
