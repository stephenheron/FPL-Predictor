"""Player availability calculations for predictions."""

import numpy as np
import pandas as pd


def compute_minutes_last_3(
    df: pd.DataFrame, predict_gw: int | None = None
) -> pd.Series:
    """Compute weighted average minutes per game over last 3 gameweeks.

    Uses recency weighting: 50% last GW, 30% GW-1, 20% GW-2.
    This better captures rotation risk and recent playing time trends.

    Args:
        df: DataFrame with player gameweek data.
        predict_gw: If provided, compute from history before this GW.

    Returns:
        Series with weighted average minutes per game for each row.
    """
    # Weights: most recent GW gets highest weight
    weights = [0.5, 0.3, 0.2]  # GW-1, GW-2, GW-3

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
        weighted_minutes = []
        for season, player_id in keys:
            # Get minutes for last 3 GWs (most recent first)
            gw_minutes = [
                minutes_lookup.get((season, player_id, predict_gw - i - 1), 0)
                for i in range(3)
            ]
            # Weighted average
            weighted_avg = sum(m * w for m, w in zip(gw_minutes, weights))
            weighted_minutes.append(weighted_avg)

        return pd.Series(weighted_minutes, index=df.index, dtype=float)

    # Compute weighted rolling average for all GWs
    grouped = df.groupby(["season", "player_id"], sort=False)

    def weighted_rolling(series: pd.Series) -> pd.Series:
        """Compute weighted rolling average with recency bias."""
        result = []
        values = series.values
        for i in range(len(values)):
            if i == 0:
                result.append(0.0)  # No history for first GW
            else:
                # Get up to 3 previous values
                prev = values[max(0, i - 3) : i][::-1]  # Most recent first
                if len(prev) == 0:
                    result.append(0.0)
                else:
                    # Apply weights (truncate if fewer than 3 games)
                    w = weights[: len(prev)]
                    w_sum = sum(w)
                    weighted_avg = sum(m * wt for m, wt in zip(prev, w)) / w_sum * sum(weights)
                    result.append(weighted_avg)
        return pd.Series(result, index=series.index)

    weighted_mins = grouped["minutes"].transform(weighted_rolling)
    return pd.Series(weighted_mins.fillna(0), index=df.index, dtype=float)


def compute_availability_multipliers(
    weighted_minutes: np.ndarray,
    min_multiplier: float = 0.1,
) -> np.ndarray:
    """Compute availability multipliers based on weighted recent minutes.

    Uses a continuous scale based on weighted average minutes per game,
    with more recent games weighted higher (50% last GW, 30% GW-1, 20% GW-2).

    Args:
        weighted_minutes: Array of weighted average minutes per game.
        min_multiplier: Floor multiplier for players with 0 minutes.

    Returns:
        Array of multipliers (0.1 to 1.0) with same shape.
    """
    # Scale by 90 minutes (full game), cap at 1.0, floor at min_multiplier
    multipliers = weighted_minutes / 90.0
    multipliers = np.clip(multipliers, min_multiplier, 1.0)
    return multipliers


def compute_minutes_last_3_xgboost(
    df: pd.DataFrame, predict_gw: int | None = None
) -> pd.Series:
    """Compute weighted average minutes for XGBoost predictions.

    Uses recency weighting: 50% last GW, 30% GW-1, 20% GW-2.

    Args:
        df: DataFrame with player data.
        predict_gw: Optional specific GW to predict.

    Returns:
        Series with weighted average minutes per game.
    """
    weights = [0.5, 0.3, 0.2]  # GW-1, GW-2, GW-3

    if predict_gw is not None:
        history = df[df["GW"] < predict_gw][["player_id", "GW", "minutes"]].copy()
        history["GW"] = pd.to_numeric(history["GW"], errors="coerce")
        minutes_lookup = history.groupby(["player_id", "GW"], sort=False)[
            "minutes"
        ].sum()
        minutes_lookup = minutes_lookup.to_dict()

        def weighted_minutes(player_id: float) -> float:
            gw_minutes = [
                minutes_lookup.get((player_id, predict_gw - i - 1), 0)
                for i in range(3)
            ]
            return sum(m * w for m, w in zip(gw_minutes, weights))

        return df["player_id"].map(weighted_minutes)

    if "roll_3_minutes" in df.columns:
        # Approximate: use roll_3 average as the weighted value
        # This is less accurate but works for non-specific GW predictions
        return df["roll_3_minutes"].fillna(0)

    return pd.Series(np.zeros(len(df)), index=df.index)
