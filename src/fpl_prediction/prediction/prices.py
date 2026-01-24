"""Attach player prices to prediction outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PriceJoinConfig:
    """Configuration for joining prices onto predictions."""

    input_file: str
    players_file: str
    output_file: str
    pred_name_col: str = "name"
    player_first_col: str = "first_name"
    player_last_col: str = "second_name"
    price_col: str = "now_cost"


def add_prices_to_predictions(config: PriceJoinConfig) -> pd.DataFrame:
    """Attach player prices to a predictions CSV.

    Args:
        config: PriceJoinConfig with input/output paths and column names.

    Returns:
        DataFrame with price column added.
    """
    pred_df = pd.read_csv(config.input_file)
    if config.price_col in pred_df.columns:
        pred_df = pred_df.drop(columns=[config.price_col])
    players_df = pd.read_csv(config.players_file)

    if config.pred_name_col not in pred_df.columns:
        raise ValueError(f"Missing column in predictions: {config.pred_name_col}")

    missing_player_cols = [
        col
        for col in (config.player_first_col, config.player_last_col, config.price_col)
        if col not in players_df.columns
    ]
    if missing_player_cols:
        raise ValueError(
            "Missing columns in players file: " + ", ".join(missing_player_cols)
        )

    players_df = players_df.copy()
    players_df["name"] = (
        players_df[config.player_first_col].fillna("")
        + " "
        + players_df[config.player_last_col].fillna("")
    ).str.strip()

    prices = players_df[["name", config.price_col]].drop_duplicates()
    prices[config.price_col] = (
        pd.to_numeric(prices[config.price_col], errors="coerce") / 10
    )

    if config.pred_name_col == "name":
        merged = pred_df.merge(prices, on="name", how="left")
    else:
        merged = pred_df.merge(
            prices, left_on=config.pred_name_col, right_on="name", how="left"
        )
        merged = merged.drop(columns=["name"])

    merged.to_csv(config.output_file, index=False)
    return merged


def summarize_price_join(df: pd.DataFrame, price_col: str) -> tuple[int, int]:
    """Return row and missing price counts."""
    rows = len(df)
    missing = int(df[price_col].isna().sum()) if price_col in df.columns else rows
    return rows, missing


def resolve_players_file(
    input_file: str,
    players_file: str | None,
    season_col: str = "season",
) -> str:
    """Resolve players file path using predictions season when needed."""
    if players_file:
        return players_file

    pred_df = pd.read_csv(input_file)
    if season_col not in pred_df.columns:
        raise ValueError(
            f"Missing '{season_col}' column in predictions; pass --players explicitly."
        )

    seasons = pred_df[season_col].dropna().astype(str).unique()
    if len(seasons) == 0:
        raise ValueError(
            f"No season value found in '{season_col}'; pass --players explicitly."
        )
    if len(seasons) > 1:
        raise ValueError(
            "Multiple seasons found in predictions; pass --players explicitly."
        )

    season = seasons[0]
    players_path = (
        Path("Fantasy-Premier-League")
        / "data"
        / season
        / "cleaned_players.csv"
    )
    if not players_path.is_file():
        raise FileNotFoundError(
            f"Players file not found at {players_path}; pass --players explicitly."
        )

    return str(players_path)
