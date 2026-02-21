"""
S&P 500 Universe Construction
==============================
Fetches current S&P 500 constituents and saves as the investable universe.

NOTE: This uses the *current* S&P 500 membership, which introduces survivorship
bias. See README for discussion of this limitation.

Usage:
    python -m src.data.universe
"""

import pandas as pd
import polars as pl

from src.utils.config import DATA_DIR, get_logger

logger = get_logger("data.universe")

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def fetch_sp500_tickers() -> pl.DataFrame:
    """
    Scrape current S&P 500 constituents from Wikipedia.

    Returns a Polars DataFrame with columns: [ticker, company, sector, sub_industry].
    """
    logger.info("Fetching S&P 500 constituents from Wikipedia...")

    # pandas read_html is the most reliable way to parse Wikipedia tables
    tables = pd.read_html(SP500_WIKI_URL)
    df_pd = tables[0]

    # Wikipedia uses "Symbol" column; some tickers have dots (BRK.B) which
    # Tiingo expects as hyphens (BRK-B)
    df = pl.DataFrame(
        {
            "ticker": df_pd["Symbol"]
            .str.replace(".", "-", regex=False)
            .tolist(),
            "company": df_pd["Security"].tolist(),
            "sector": df_pd["GICS Sector"].tolist(),
            "sub_industry": df_pd["GICS Sub-Industry"].tolist(),
        }
    )

    logger.info(f"Found {len(df)} S&P 500 constituents")
    return df


def save_universe(df: pl.DataFrame) -> None:
    """Save universe to parquet in data/external/."""
    out_dir = DATA_DIR / "external"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sp500_universe.parquet"
    df.write_parquet(out_path)
    logger.info(f"Universe saved to {out_path}")

    # Also save a human-readable CSV
    csv_path = out_dir / "sp500_universe.csv"
    df.write_csv(csv_path)
    logger.info(f"CSV copy saved to {csv_path}")


def load_universe() -> pl.DataFrame:
    """Load previously saved universe."""
    path = DATA_DIR / "external" / "sp500_universe.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Universe file not found at {path}. Run `python -m src.data.universe` first."
        )
    return pl.read_parquet(path)


if __name__ == "__main__":
    df = fetch_sp500_tickers()
    save_universe(df)
    print(f"\nSample tickers: {df['ticker'].head(10).to_list()}")
    print(f"Sectors:\n{df.group_by('sector').len().sort('len', descending=True)}")
