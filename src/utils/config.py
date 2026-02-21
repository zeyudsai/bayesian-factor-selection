"""Shared utilities: configuration loading, logging, and common helpers."""

import logging
import os
from pathlib import Path

import yaml

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "configs"


def get_config(config_name: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    config_path = CONFIG_DIR / config_name
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a consistent logger across modules."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def get_tiingo_api_key() -> str:
    """Read Tiingo API key from environment or .env file."""
    key = os.environ.get("TIINGO_API_KEY")
    if key:
        return key

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("TIINGO_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("\"'")

    raise ValueError(
        "TIINGO_API_KEY not found. Set it via environment variable or .env file."
    )
