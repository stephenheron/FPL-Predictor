"""Ensemble logic for combining model predictions."""

import json
from pathlib import Path
from typing import cast

import pandas as pd


def load_meta_weights(position: str) -> dict | None:
    """Load learned meta weights for a position.

    Args:
        position: Position code (GK, DEF, MID, FWD).

    Returns:
        Dictionary with learned weights, or None if not found.
    """
    weights_file = Path(f"models/meta_weights_{position.lower()}.json")
    if weights_file.exists():
        with weights_file.open() as f:
            return json.load(f)
    return None


def combine_predictions(
    xgb_file: str,
    lstm_file: str,
    output_file: str,
    weight_xgb: float = 0.5,
    normalize_scores: bool = True,
) -> pd.DataFrame:
    """Combine XGBoost and LSTM predictions with weighted average.

    Args:
        xgb_file: Path to XGBoost predictions CSV.
        lstm_file: Path to LSTM predictions CSV.
        output_file: Path to save combined predictions.
        weight_xgb: Weight for XGBoost predictions (0-1), LSTM gets 1-weight_xgb.

    Returns:
        DataFrame with combined predictions.

    Raises:
        ValueError: If weight_xgb is not between 0 and 1.
    """
    if not 0.0 <= weight_xgb <= 1.0:
        raise ValueError("weight_xgb must be between 0 and 1.")

    xgb_df = pd.read_csv(xgb_file)
    lstm_df = pd.read_csv(lstm_file)

    # Rename prediction columns if needed
    if "predicted_points" in xgb_df.columns:
        xgb_df = xgb_df.rename(columns={"predicted_points": "predicted_points_xgb"})
    if "predicted_points" in lstm_df.columns:
        lstm_df = lstm_df.rename(columns={"predicted_points": "predicted_points_lstm"})

    # Determine join keys
    join_keys = ["player_id", "GW"]
    if "season" in xgb_df.columns and "season" in lstm_df.columns:
        join_keys.append("season")

    if "fixture" in xgb_df.columns and "fixture" in lstm_df.columns:
        join_keys.append("fixture")
    elif (
        "opponent_team" in xgb_df.columns
        and "opponent_team" in lstm_df.columns
        and "was_home" in xgb_df.columns
        and "was_home" in lstm_df.columns
    ):
        join_keys.extend(["opponent_team", "was_home"])
    elif (
        "opponent_name" in xgb_df.columns
        and "opponent_name" in lstm_df.columns
        and "was_home" in xgb_df.columns
        and "was_home" in lstm_df.columns
    ):
        join_keys.extend(["opponent_name", "was_home"])

    # Merge predictions
    merged = pd.merge(
        xgb_df,
        lstm_df,
        on=join_keys,
        how="inner",
        suffixes=("_xgb", "_lstm"),
    )

    if "predicted_points_xgb" not in merged.columns:
        raise ValueError(f"Missing predicted_points in {xgb_file}")
    if "predicted_points_lstm" not in merged.columns:
        raise ValueError(f"Missing predicted_points in {lstm_file}")

    group_keys = ["GW"]
    if "season" in merged.columns:
        group_keys.append("season")

    if normalize_scores:
        grouped = merged.groupby(group_keys, dropna=False)
        xgb_mean = grouped["predicted_points_xgb"].transform("mean")
        xgb_std = grouped["predicted_points_xgb"].transform("std").fillna(0.0)
        xgb_std = xgb_std.mask(xgb_std == 0, 1.0)
        lstm_mean = grouped["predicted_points_lstm"].transform("mean")
        lstm_std = grouped["predicted_points_lstm"].transform("std").fillna(0.0)
        lstm_std = lstm_std.mask(lstm_std == 0, 1.0)

        xgb_scores = (merged["predicted_points_xgb"] - xgb_mean) / xgb_std
        lstm_scores = (merged["predicted_points_lstm"] - lstm_mean) / lstm_std
    else:
        xgb_scores = merged["predicted_points_xgb"]
        lstm_scores = merged["predicted_points_lstm"]
        xgb_mean = merged["predicted_points_xgb"]
        lstm_mean = merged["predicted_points_lstm"]
        xgb_std = 1.0
        lstm_std = 1.0

    combined_scores = xgb_scores * weight_xgb + lstm_scores * (1.0 - weight_xgb)

    if normalize_scores:
        combined_mean = xgb_mean * weight_xgb + lstm_mean * (1.0 - weight_xgb)
        combined_std = xgb_std * weight_xgb + lstm_std * (1.0 - weight_xgb)
        merged["combined_points"] = combined_scores * combined_std + combined_mean
    else:
        merged["combined_points"] = combined_scores

    # Helper to pick best column value
    def pick_column(df: pd.DataFrame, col: str) -> pd.Series | None:
        xgb_col = f"{col}_xgb" if f"{col}_xgb" in df.columns else None
        lstm_col = f"{col}_lstm" if f"{col}_lstm" in df.columns else None
        xgb_series: pd.Series | None = cast(pd.Series, df[xgb_col]) if xgb_col else None
        lstm_series: pd.Series | None = cast(pd.Series, df[lstm_col]) if lstm_col else None
        if xgb_series is not None and lstm_series is not None:
            return xgb_series.where(xgb_series.notna(), lstm_series)
        if xgb_series is not None:
            return xgb_series
        if lstm_series is not None:
            return lstm_series
        if col in df.columns:
            return cast(pd.Series, df[col])
        return None

    # Build output DataFrame
    output = pd.DataFrame()
    output["player_id"] = merged["player_id"]
    for col in [
        "fpl_id",
        "name",
        "team",
        "season",
        "GW",
        "was_home",
        "opponent_name",
        "total_points",
        "now_cost",
    ]:
        values = pick_column(merged, col)
        if values is not None:
            output[col] = values

    output["predicted_points_xgb"] = merged["predicted_points_xgb"]
    output["predicted_points_lstm"] = merged["predicted_points_lstm"]
    output["combined_points"] = merged["combined_points"]

    output.to_csv(output_file, index=False)
    print(f"Saved combined predictions to: {output_file}")
    return output


def combine_position_predictions(
    position: str,
    xgb_file: str | None = None,
    lstm_file: str | None = None,
    output_file: str | None = None,
    weight_xgb: float | None = None,
    normalize_scores: bool = True,
    use_meta_weights: bool = True,
) -> pd.DataFrame:
    """Combine predictions for a position with default file paths.

    Args:
        position: Position code (GK, DEF, MID, FWD).
        xgb_file: Optional XGBoost file (defaults to {pos}_predictions.csv).
        lstm_file: Optional LSTM file (defaults to {pos}_predictions_lstm.csv).
        output_file: Optional output file (defaults to {pos}_predictions_combined.csv).
        weight_xgb: Weight for XGBoost predictions. If None and use_meta_weights
            is True, attempts to load learned weights.
        normalize_scores: Whether to normalize scores before combining.
        use_meta_weights: If True and weight_xgb is None, load learned meta weights.

    Returns:
        DataFrame with combined predictions.
    """
    pos_lower = position.lower()
    xgb_file = xgb_file or f"reports/predictions/{pos_lower}_predictions.csv"
    lstm_file = lstm_file or f"reports/predictions/{pos_lower}_predictions_lstm.csv"
    output_file = output_file or f"reports/predictions/{pos_lower}_predictions_combined.csv"

    # Auto-load meta weights if available
    if weight_xgb is None and use_meta_weights:
        meta = load_meta_weights(position)
        if meta:
            loaded_weight = meta.get("weight_xgb")
            weight_xgb = float(loaded_weight) if loaded_weight is not None else 0.5
            print(f"Using learned meta-weights for {position}: XGB={weight_xgb:.3f}")
        else:
            weight_xgb = 0.5  # Fallback to default
    elif weight_xgb is None:
        weight_xgb = 0.5

    final_weight_xgb = float(weight_xgb) if weight_xgb is not None else 0.5

    return combine_predictions(
        xgb_file,
        lstm_file,
        output_file,
        final_weight_xgb,
        normalize_scores,
    )
