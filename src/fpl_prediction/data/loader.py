"""Data loading utilities for FPL Prediction."""

import pandas as pd


def load_season(path: str) -> pd.DataFrame:
    """Load a season CSV file and normalize columns.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with GW as numeric and season as string.
    """
    df = pd.read_csv(path)
    df["GW"] = pd.to_numeric(df["GW"], errors="coerce")
    df["season"] = df["season"].astype(str)
    return df
