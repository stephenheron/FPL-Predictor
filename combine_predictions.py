import pandas as pd


def combine_predictions(
    xgb_file: str,
    lstm_file: str,
    output_file: str,
    weight_xgb: float = 0.5,
) -> pd.DataFrame:
    if not 0.0 <= weight_xgb <= 1.0:
        raise ValueError("weight_xgb must be between 0 and 1.")

    xgb_df = pd.read_csv(xgb_file)
    lstm_df = pd.read_csv(lstm_file)

    if "predicted_points" in xgb_df.columns:
        xgb_df = xgb_df.rename(columns={"predicted_points": "predicted_points_xgb"})
    if "predicted_points" in lstm_df.columns:
        lstm_df = lstm_df.rename(columns={"predicted_points": "predicted_points_lstm"})

    join_keys = ["player_id", "GW"]
    if "season" in xgb_df.columns and "season" in lstm_df.columns:
        join_keys.append("season")

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

    merged["combined_points"] = (
        merged["predicted_points_xgb"] * weight_xgb
        + merged["predicted_points_lstm"] * (1.0 - weight_xgb)
    )

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
    return output
