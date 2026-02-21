# Bayesian Factor Selection for Systematic Equity Alpha

A rigorous, end-to-end research framework for equity factor discovery and selection using Bayesian shrinkage and model averaging methods. This project addresses the "factor zoo" problem — the proliferation of spurious factors due to multiple testing and data-mining — by applying principled Bayesian inference to cross-sectional asset pricing.

## Motivation

The academic finance literature has documented hundreds of factors that purportedly predict stock returns (Harvey, Liu & Zhu, 2016). Many of these factors are likely false discoveries resulting from extensive data mining. Traditional frequentist approaches (e.g., single-factor t-tests) are poorly equipped to handle this multiplicity problem. This project applies **Bayesian Model Averaging (BMA)** and **Bayesian shrinkage priors** (e.g., Horseshoe, Finnish Horseshoe) to perform disciplined factor selection, providing posterior inclusion probabilities rather than binary significance decisions.

## Research Pipeline

```
Raw Market Data → Data Cleaning → Factor Construction → Fama-MacBeth Baseline
                                                              ↓
                              Bayesian Factor Selection ← Cross-Sectional Returns
                                       ↓
                          Walk-Forward OOS Validation → Performance w/ Transaction Costs
```

## Project Structure

```
bayesian-factor-selection/
├── src/
│   ├── data/           # Data ingestion, cleaning, universe construction
│   ├── factors/        # Factor definitions and computation
│   ├── models/         # Bayesian selection models (BMA, shrinkage priors)
│   ├── backtest/       # Walk-forward validation and performance analysis
│   └── utils/          # Shared utilities, config, logging
├── data/
│   ├── raw/            # Raw API downloads (git-ignored)
│   ├── processed/      # Cleaned panel data (git-ignored)
│   └── external/       # Ken French factors, FRED macro data
├── notebooks/          # Exploratory analysis and result visualization
├── tests/              # Unit and integration tests
├── configs/            # YAML configuration files
└── README.md
```

## Data & Known Limitations

- **Price data**: Tiingo API (adjusted OHLCV, 2010–2025)
- **Universe**: Current S&P 500 constituents only
- **⚠️ Survivorship bias**: Due to data vendor limitations, the backtest universe is restricted to stocks that are *currently* in the S&P 500. This introduces survivorship bias, as delisted or removed stocks are excluded. Results should be interpreted with this caveat. A production-grade version would use point-in-time constituent lists from CRSP or similar providers.
- **Fundamental data**: Market cap from Tiingo; accounting data not yet included
- **Factor benchmark**: Ken French Data Library (Fama-French 5 factors + Momentum)

## Tech Stack

- **Python 3.11+**
- **Polars** for high-performance data wrangling on panel data
- **Pandas** for statistical modeling interfaces (statsmodels, scipy)
- **NumPy / SciPy** for numerical computation
- **statsmodels** for Fama-MacBeth regression baseline
- **PyMC / NumPyro** for Bayesian inference (upcoming)
- **Matplotlib / Seaborn** for publication-quality figures

## Setup

```bash
pip install polars pandas numpy scipy statsmodels requests python-dotenv pyyaml
```

Set your Tiingo API key:
```bash
echo "TIINGO_API_KEY=your_key_here" > .env
```

## Usage

```bash
# Step 1: Download S&P 500 universe
python -m src.data.universe

# Step 2: Fetch price data from Tiingo
python -m src.data.fetch_prices

# Step 3: Clean and prepare panel data
python -m src.data.prepare_panel

# Step 4: Compute factors
python -m src.factors.compute_factors

# Step 5: Run Fama-MacBeth baseline
python -m src.models.fama_macbeth
```

## Author

Sai — Postdoctoral Researcher in Statistics, TU Dortmund / Lamarr Institute  
Research focus: Scalable Bayesian methods, computational statistics, coreset theory

## References

- Harvey, C. R., Liu, Y., & Zhu, H. (2016). *...and the Cross-Section of Expected Returns.* Review of Financial Studies.
- Fama, E. F., & MacBeth, J. D. (1973). *Risk, Return, and Equilibrium: Empirical Tests.* Journal of Political Economy.
- Piironen, J., & Vehtari, A. (2017). *Sparsity information and regularization in the horseshoe and other shrinkage priors.* Electronic Journal of Statistics.

---

*This is an independent research project. It is not investment advice.*
