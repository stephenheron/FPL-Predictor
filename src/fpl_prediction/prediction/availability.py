"""Player availability calculations for predictions."""

import numpy as np
import pandas as pd

from fpl_prediction.config.settings import (
    AVAILABILITY_MULTIPLIERS,
    AVAILABILITY_THRESHOLDS,
)


def compute_minutes_last_3(
    df: pd.DataFrame, predict_gw: int | None = None
) -> pd.Series:
    """Compute total minutes played in the last 3 gameweeks.

    Args:
        df: DataFrame with player gameweek data.
        predict_gw: If provided, compute from history before this GW.

    Returns:
        Series with total minutes in last 3 GWs for each row.
    """
    df = df.sort_values(["season", "player_id", "GW"]).copy()

    if predict_gw is not None:
        # Use historical data before the prediction GW
        history = df.loc[
            df["GW"] < predict_gw, ["season", "player_id", "GW", "minutes"]
        ].copy()
        history = pd.DataFrame(history)
        history["GW"] = pd.to_numeric(history["GW"], errors="coerce")

        minutes_lookup = history.groupby(
            ["season", "player_id", "GW"], sort=False
        )["minutes"].sum()
        minutes_lookup = minutes_lookup.to_dict()

        keys = list(zip(df["season"], df["player_id"]))
        minutes_last_3 = [
            sum(
                minutes_lookup.get((season, player_id, gw), 0)
                for gw in range(predict_gw - 3, predict_gw)
            )
            for season, player_id in keys
        ]
        return pd.Series(minutes_last_3, index=df.index, dtype=float)

    # Compute rolling sum for all GWs
    grouped = df.groupby(["season", "player_id"], sort=False)
    shifted = grouped["minutes"].shift(1)
    shifted = pd.Series(shifted, index=df.index, dtype=float)
    minutes_last_3 = (
        shifted.groupby([df["season"], df["player_id"]])
        .rolling(window=3, min_periods=1)
        .sum()
        .reset_index(level=[0, 1], drop=True)
    )
    minutes_last_3 = minutes_last_3.reindex(df.index)
    return pd.Series(minutes_last_3.fillna(0), index=df.index, dtype=float)


def compute_availability_multipliers(minutes_last_3: np.ndarray) -> np.ndarray:
    """Compute availability multipliers based on recent minutes.

    Players with more recent playing time get higher multipliers:
    - >= 180 minutes: 1.0 (full weight)
    - >= 90 minutes: 0.7
    - > 0 minutes: 0.4
    - 0 minutes: 0.2 (minimal weight)

    Args:
        minutes_last_3: Array of total minutes in last 3 GWs.

    Returns:
        Array of multipliers with same shape.
    """
    return np.select(
        [
            minutes_last_3 >= AVAILABILITY_THRESHOLDS["high"],
            minutes_last_3 >= AVAILABILITY_THRESHOLDS["medium"],
            minutes_last_3 > AVAILABILITY_THRESHOLDS["low"],
        ],
        [
            AVAILABILITY_MULTIPLIERS["high"],
            AVAILABILITY_MULTIPLIERS["medium"],
            AVAILABILITY_MULTIPLIERS["low"],
        ],
        default=AVAILABILITY_MULTIPLIERS["none"],
    )


def compute_minutes_last_3_xgboost(
    df: pd.DataFrame, predict_gw: int | None = None
) -> pd.Series:
    """Compute minutes for XGBoost predictions.

    Handles both specific GW prediction and general scoring.

    Args:
        df: DataFrame with player data.
        predict_gw: Optional specific GW to predict.

    Returns:
        Series with minutes in last 3 GWs.
    """
    if predict_gw is not None:
        history = df[df["GW"] < predict_gw][["player_id", "GW", "minutes"]].copy()
        history["GW"] = pd.to_numeric(history["GW"], errors="coerce")
        minutes_lookup = history.groupby(["player_id", "GW"], sort=False)[
            "minutes"
        ].sum()
        minutes_lookup = minutes_lookup.to_dict()

        def last_three_minutes(player_id: float) -> float:
            return sum(
                minutes_lookup.get((player_id, gw), 0)
                for gw in range(predict_gw - 3, predict_gw)
            )

        return df["player_id"].map(last_three_minutes)

    if "roll_3_minutes" in df.columns:
        return df["roll_3_minutes"].fillna(0) * 3

    return pd.Series(np.zeros(len(df)), index=df.index)
