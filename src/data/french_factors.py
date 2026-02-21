"""
Ken French Data Library Loader
================================
Downloads Fama-French factor returns for benchmarking.

These serve as ground truth: if our self-constructed momentum factor
doesn't correlate with Ken French's UMD, something is wrong.

Usage:
    python -m src.data.french_factors
"""

import io

import pandas as pd
import polars as pl
import requests

from src.utils.config import DATA_DIR, get_logger

logger = get_logger("data.french_factors")

# Fama-French 5 Factors + Momentum (CSV zip)
FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Momentum_Factor_daily_CSV.zip"
)


def _download_french_csv(url: str, skip_header: int = 3) -> pd.DataFrame:
    """Download and parse a Ken French CSV zip file."""
    logger.info(f"Downloading {url}...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    # The CSV is inside a zip; pandas handles this directly
    from zipfile import ZipFile

    with ZipFile(io.BytesIO(resp.content)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            # Read raw text to find where daily data starts
            text = f.read().decode("utf-8")

    # French files have header rows before the data
    lines = text.strip().split("\n")

    # Find the first line that starts with a date (YYYYMMDD format, 8 digits)
    data_lines = []
    header_found = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if header_found:
                break  # End of daily section (monthly section follows)
            continue
        # Check if line starts with a date
        first_field = stripped.split(",")[0].strip()
        if first_field.isdigit() and len(first_field) == 8:
            header_found = True
            data_lines.append(stripped)
        elif header_found:
            # Non-date line after data started → section break
            if not first_field.isdigit():
                break
            data_lines.append(stripped)

    if not data_lines:
        raise ValueError(f"Could not parse data from {url}")

    # Parse the data lines
    df = pd.read_csv(io.StringIO("\n".join(data_lines)), header=None)

    return df


def fetch_french_factors() -> pl.DataFrame:
    """
    Download FF5 + Momentum daily factors.

    Returns a Polars DataFrame with columns:
    [date, mkt_rf, smb, hml, rmw, cma, rf, umd]

    All values are in decimal (e.g., 0.01 = 1%).
    """
    # Fama-French 5 Factors
    logger.info("Fetching Fama-French 5 factors...")
    ff5 = _download_french_csv(FF5_URL)
    ff5.columns = ["date", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"]

    # Momentum
    logger.info("Fetching Momentum factor (UMD)...")
    mom = _download_french_csv(MOM_URL)
    mom.columns = ["date", "umd"] if mom.shape[1] == 2 else ["date", "umd"] + [
        f"extra_{i}" for i in range(mom.shape[1] - 2)
    ]
    mom = mom[["date", "umd"]]

    # Merge
    merged = ff5.merge(mom, on="date", how="inner")

    # Parse dates
    merged["date"] = pd.to_datetime(merged["date"].astype(str), format="%Y%m%d")

    # Convert from percentage to decimal
    for col in ["mkt_rf", "smb", "hml", "rmw", "cma", "rf", "umd"]:
        merged[col] = merged[col].astype(float) / 100

    # Convert to Polars
    df = pl.from_pandas(merged)

    logger.info(
        f"French factors: {len(df)} days, "
        f"{df['date'].min()} to {df['date'].max()}"
    )

    return df


def save_french_factors(df: pl.DataFrame) -> None:
    """Save to data/external/."""
    out_dir = DATA_DIR / "external"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "french_factors_daily.parquet"
    df.write_parquet(out_path)
    logger.info(f"Saved to {out_path}")


def load_french_factors() -> pl.DataFrame:
    """Load previously saved French factors."""
    path = DATA_DIR / "external" / "french_factors_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"French factors not found at {path}. "
            "Run `python -m src.data.french_factors` first."
        )
    return pl.read_parquet(path)


if __name__ == "__main__":
    df = fetch_french_factors()
    save_french_factors(df)

    # Quick summary
    print("\nFactor summary statistics (daily, in %):")
    summary = df.select(
        pl.exclude("date").mean().name.suffix("_mean"),
    )
    print(summary)
