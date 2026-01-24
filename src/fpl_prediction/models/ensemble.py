"""Ensemble logic for combining model predictions."""

import pandas as pd


def combine_predictions(
    xgb_file: str,
    lstm_file: str,
    output_file: str,
    weight_xgb: float = 0.5,
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

    # Compute weighted average
    merged["combined_points"] = (
        merged["predicted_points_xgb"] * weight_xgb
        + merged["predicted_points_lstm"] * (1.0 - weight_xgb)
    )

    # Helper to pick best column value
    def pick_column(df: pd.DataFrame, col: str) -> pd.Series | None:
        xgb_col = f"{col}_xgb" if f"{col}_xgb" in df.columns else None
        lstm_col = f"{col}_lstm" if f"{col}_lstm" in df.columns else None
        if xgb_col and lstm_col:
            return df[xgb_col].combine_first(df[lstm_col])
        if xgb_col:
            return df[xgb_col]
        if lstm_col:
            return df[lstm_col]
        if col in df.columns:
            return df[col]
        return None

    # Build output DataFrame
    output = pd.DataFrame()
    output["player_id"] = merged["player_id"]
    for col in ["name", "team", "season", "GW", "was_home", "opponent_name", "total_points"]:
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
    weight_xgb: float = 0.5,
) -> pd.DataFrame:
    """Combine predictions for a position with default file paths.

    Args:
        position: Position code (GK, DEF, MID, FWD).
        xgb_file: Optional XGBoost file (defaults to {pos}_predictions.csv).
        lstm_file: Optional LSTM file (defaults to {pos}_predictions_lstm.csv).
        output_file: Optional output file (defaults to {pos}_predictions_combined.csv).
        weight_xgb: Weight for XGBoost predictions.

    Returns:
        DataFrame with combined predictions.
    """
    pos_lower = position.lower()
    xgb_file = xgb_file or f"{pos_lower}_predictions.csv"
    lstm_file = lstm_file or f"{pos_lower}_predictions_lstm.csv"
    output_file = output_file or f"{pos_lower}_predictions_combined.csv"

    return combine_predictions(xgb_file, lstm_file, output_file, weight_xgb)
