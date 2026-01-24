"""Feature preprocessing for FPL Prediction models."""

from typing import cast

import numpy as np
import pandas as pd

from fpl_prediction.config.features import (
    BASE_COLS,
    FIXTURE_COLS,
    PER90_COLS,
    ROLL_HINT_COLS,
    ROLLING_COLS,
)


def add_rolling_features_lstm(df: pd.DataFrame, roll_window: int) -> pd.DataFrame:
    """Add rolling features for LSTM model.

    Uses ROLL_HINT_COLS and creates shifted rolling means.

    Args:
        df: Input DataFrame sorted by season, player_id, GW.
        roll_window: Size of the rolling window.

    Returns:
        DataFrame with additional roll_{window}_{col} columns.
    """
    df = df.sort_values(["season", "player_id", "GW"]).copy()
    grouped = df.groupby(["season", "player_id"], sort=False)
    for col in ROLL_HINT_COLS:
        shifted = grouped[col].shift(1)
        roll_name = f"roll_{roll_window}_{col}"
        df[roll_name] = (
            shifted.groupby([df["season"], df["player_id"]])
            .rolling(window=roll_window, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
    return df


def add_rolling_features_xgboost(
    df: pd.DataFrame, roll_windows: tuple[int, ...]
) -> pd.DataFrame:
    """Add rolling features for XGBoost model.

    Uses ROLLING_COLS and creates multiple window sizes with per90 variants.

    Args:
        df: Input DataFrame sorted by season, player_id, GW.
        roll_windows: Tuple of window sizes (e.g., (3, 5, 8)).

    Returns:
        DataFrame with additional rolling feature columns.
    """
    df = df.sort_values(["season", "player_id", "GW"]).copy()
    grouped = df.groupby(["season", "player_id"], sort=False)
    shifted_minutes = grouped["minutes"].shift(1)

    # Add history count features
    for window in roll_windows:
        count_name = f"roll_hist_count_{window}"
        df[count_name] = (
            shifted_minutes.groupby([df["season"], df["player_id"]])
            .rolling(window=window, min_periods=1)
            .count()
            .reset_index(level=[0, 1], drop=True)
        )

    # Add rolling mean features
    for col in ROLLING_COLS:
        shifted = grouped[col].shift(1)
        for window in roll_windows:
            roll_name = f"roll_{window}_{col}"
            df[roll_name] = (
                shifted.groupby([df["season"], df["player_id"]])
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )

    # Add per90 variants of rolling features
    for window in roll_windows:
        minutes_col = f"roll_{window}_minutes"
        minutes_vals = df[minutes_col]
        for col in PER90_COLS:
            base_col = f"roll_{window}_{col}"
            per90_col = f"roll_{window}_{col}_per90"
            df[per90_col] = np.where(
                minutes_vals > 0,
                df[base_col] / minutes_vals * 90.0,
                np.nan,
            )

    return df


def add_per90_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-90-minute features for LSTM model.

    Args:
        df: Input DataFrame with minutes and PER90_COLS.

    Returns:
        DataFrame with additional {col}_per90 columns.
    """
    minutes = df["minutes"].replace(0, np.nan)
    for col in PER90_COLS:
        per90_col = f"{col}_per90"
        df[per90_col] = df[col] / minutes * 90.0
    return df


def build_features_lstm(
    df: pd.DataFrame, position: str, roll_window: int
) -> pd.DataFrame:
    """Build features for LSTM model.

    Filters by position and adds rolling and per90 features.

    Args:
        df: Input DataFrame with all positions.
        position: Position to filter (GK, DEF, MID, FWD).
        roll_window: Size of the rolling window.

    Returns:
        DataFrame with position-filtered and engineered features.
    """
    df = pd.DataFrame(df)
    df = cast(pd.DataFrame, df.loc[df["position"] == position].copy())
    df = add_rolling_features_lstm(df, roll_window)
    df = add_per90_features(df)
    return df


def build_features_xgboost(
    df: pd.DataFrame, position: str, roll_windows: tuple[int, ...]
) -> pd.DataFrame:
    """Build features for XGBoost model.

    Filters by position and adds rolling features with multiple windows.

    Args:
        df: Input DataFrame with all positions.
        position: Position to filter (GK, DEF, MID, FWD).
        roll_windows: Tuple of window sizes.

    Returns:
        DataFrame with position-filtered and engineered features.
    """
    df = df[df["position"] == position].copy()
    df = add_rolling_features_xgboost(df, roll_windows)
    return df


def get_feature_columns_lstm(df: pd.DataFrame, roll_window: int) -> list[str]:
    """Get feature columns for LSTM model.

    Args:
        df: DataFrame to check for available columns.
        roll_window: Rolling window size used in preprocessing.

    Returns:
        List of feature column names that exist in the DataFrame.
    """
    feature_cols: list[str] = []
    feature_cols.extend(BASE_COLS)
    feature_cols.extend([f"{col}_per90" for col in PER90_COLS])
    feature_cols.extend([f"roll_{roll_window}_{col}" for col in ROLL_HINT_COLS])
    feature_cols.extend(FIXTURE_COLS)
    return [col for col in feature_cols if col in df.columns]


def get_feature_columns_xgboost(
    df: pd.DataFrame, roll_windows: tuple[int, ...]
) -> list[str]:
    """Get feature columns for XGBoost model.

    Args:
        df: DataFrame to check for available columns.
        roll_windows: Tuple of rolling window sizes.

    Returns:
        List of feature column names that exist in the DataFrame.
    """
    feature_cols: list[str] = []
    for window in roll_windows:
        for col in ROLLING_COLS:
            feature_cols.append(f"roll_{window}_{col}")
        for col in PER90_COLS:
            feature_cols.append(f"roll_{window}_{col}_per90")
        feature_cols.append(f"roll_hist_count_{window}")

    feature_cols.extend(FIXTURE_COLS)
    return [col for col in feature_cols if col in df.columns]
