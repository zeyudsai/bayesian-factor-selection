"""
Bayesian Results Visualization
================================
Publication-quality plots for the Bayesian factor selection results.

Generates:
1. Posterior density plots for each factor coefficient
2. Shrinkage profile (κ) visualization
3. Bayesian vs Frequentist comparison chart
4. Posterior inclusion probability summary

Usage:
    python -m src.models.plot_results
"""

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.config import DATA_DIR, get_logger

logger = get_logger("models.plot_results")

# Style
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    }
)

FACTOR_LABELS = {
    "factor_mom_z": "Momentum (12-1m)",
    "factor_vol_z": "Realized Volatility",
    "factor_size_z": "Size (log $Vol)",
    "factor_str_z": "Short-Term Reversal",
}

OUTPUT_DIR = DATA_DIR / "processed" / "figures"


def load_results():
    """Load all saved results."""
    trace_path = DATA_DIR / "processed" / "horseshoe_trace.nc"
    results_path = DATA_DIR / "processed" / "bayesian_results.csv"
    comparison_path = DATA_DIR / "processed" / "bayesian_vs_frequentist.csv"

    idata = az.from_netcdf(str(trace_path))
    results = pd.read_csv(results_path)

    comparison = None
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)

    return idata, results, comparison


def plot_posterior_densities(idata: az.InferenceData, factor_names: list[str]):
    """
    Plot posterior density for each factor coefficient with HDI.

    This is the money plot for interviews — shows which factors the model
    believes are nonzero vs shrunk to zero.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    beta_samples = idata.posterior["beta"].values
    beta_flat = beta_samples.reshape(-1, beta_samples.shape[-1])

    for i, (name, ax) in enumerate(zip(factor_names, axes)):
        samples = beta_flat[:, i]
        label = FACTOR_LABELS.get(name, name)

        # Density plot
        ax.hist(
            samples, bins=80, density=True, alpha=0.6, color="steelblue",
            edgecolor="none",
        )

        # HDI
        hdi = az.hdi(samples, hdi_prob=0.95)
        ax.axvspan(hdi[0], hdi[1], alpha=0.15, color="steelblue", label="95% HDI")

        # Zero reference
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
                    label="Zero")

        # Posterior mean
        mean = np.mean(samples)
        ax.axvline(mean, color="darkblue", linewidth=2, label=f"Mean: {mean:+.5f}")

        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Coefficient (β)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    fig.suptitle(
        "Posterior Distributions of Factor Coefficients\n(Regularized Horseshoe Prior)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "posterior_densities.png", dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {OUTPUT_DIR / 'posterior_densities.png'}")
    plt.close()


def plot_shrinkage_profile(results: pd.DataFrame):
    """
    Visualize the shrinkage factor κ for each factor.

    κ = 1 → fully shrunk (noise)
    κ = 0 → unshrunk (signal)
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = [FACTOR_LABELS.get(f, f) for f in results["factor"]]
    kappas = results["shrinkage_kappa"].values
    signals = 1 - kappas

    colors = ["#2ecc71" if s > 0.5 else "#e74c3c" if s < 0.2 else "#f39c12"
              for s in signals]

    bars = ax.barh(labels, signals, color=colors, edgecolor="white", height=0.6)

    # Add value labels
    for bar, s, k in zip(bars, signals, kappas):
        ax.text(
            bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            f"κ={k:.3f}",
            va="center", fontsize=10, color="gray",
        )

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Signal Strength (1 - κ)", fontsize=12)
    ax.set_title(
        "Horseshoe Shrinkage Profile\n"
        "Green = retained signal, Red = shrunk to noise",
        fontweight="bold",
    )
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "shrinkage_profile.png", dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {OUTPUT_DIR / 'shrinkage_profile.png'}")
    plt.close()


def plot_pip_summary(results: pd.DataFrame):
    """
    Posterior Inclusion Probability bar chart.

    PIP > 80% → strong evidence for inclusion
    PIP < 20% → strong evidence for exclusion
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = [FACTOR_LABELS.get(f, f) for f in results["factor"]]
    pips = results["pip"].values

    colors = ["#2ecc71" if p > 0.8 else "#e74c3c" if p < 0.2 else "#f39c12"
              for p in pips]

    bars = ax.barh(labels, pips, color=colors, edgecolor="white", height=0.6)

    for bar, p in zip(bars, pips):
        ax.text(
            bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{p:.1%}", va="center", fontsize=11,
        )

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Posterior Inclusion Probability", fontsize=12)
    ax.set_title(
        "Factor Selection: Posterior Inclusion Probabilities\n"
        "P(|β| > 0.05% monthly return)",
        fontweight="bold",
    )
    ax.axvline(0.8, color="green", linestyle="--", alpha=0.5, label="Strong inclusion")
    ax.axvline(0.2, color="red", linestyle="--", alpha=0.5, label="Strong exclusion")
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "pip_summary.png", dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {OUTPUT_DIR / 'pip_summary.png'}")
    plt.close()


def plot_bayesian_vs_frequentist(comparison: pd.DataFrame):
    """
    Side-by-side comparison of Bayesian PIP vs Frequentist p-value.
    """
    if comparison is None or "p_value" not in comparison.columns:
        logger.info("No frequentist results available, skipping comparison plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    labels = [FACTOR_LABELS.get(f, f) for f in comparison["factor"]]
    y_pos = range(len(labels))

    # Left: Frequentist |t-stat|
    t_stats = comparison["t_stat"].abs().values
    colors_t = ["#2ecc71" if t > 1.96 else "#e74c3c" for t in t_stats]
    ax1.barh(y_pos, t_stats, color=colors_t, height=0.6)
    ax1.axvline(1.96, color="black", linestyle="--", label="5% significance")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("|t-statistic| (Newey-West)")
    ax1.set_title("Frequentist: Fama-MacBeth", fontweight="bold")
    ax1.legend()

    # Right: Bayesian PIP
    pips = comparison["pip"].values
    colors_p = ["#2ecc71" if p > 0.8 else "#e74c3c" if p < 0.2 else "#f39c12"
                for p in pips]
    ax2.barh(y_pos, pips, color=colors_p, height=0.6)
    ax2.axvline(0.8, color="green", linestyle="--", alpha=0.5)
    ax2.axvline(0.2, color="red", linestyle="--", alpha=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Posterior Inclusion Probability")
    ax2.set_title("Bayesian: Horseshoe Selection", fontweight="bold")
    ax2.set_xlim(0, 1.05)

    fig.suptitle(
        "Factor Selection: Frequentist vs Bayesian",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "bayesian_vs_frequentist.png", dpi=150, bbox_inches="tight"
    )
    logger.info(f"Saved: {OUTPUT_DIR / 'bayesian_vs_frequentist.png'}")
    plt.close()


def main():
    """Generate all plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    idata, results, comparison = load_results()

    factor_names = results["factor"].tolist()

    plot_posterior_densities(idata, factor_names)
    plot_shrinkage_profile(results)
    plot_pip_summary(results)
    plot_bayesian_vs_frequentist(comparison)

    logger.info(f"\nAll figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
