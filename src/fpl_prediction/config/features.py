"""Feature column definitions for FPL Prediction models."""

# Base columns used in LSTM model (per-gameweek stats)
BASE_COLS: tuple[str, ...] = (
    "minutes",
    "starts",
    "total_points",
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "ict_index",
    "bps",
    "bonus",
    "threat",
    "influence",
    "creativity",
    "us_key_passes",
    "us_xGChain",
    "us_xGBuildup",
)

# Columns to compute per-90-minute stats
PER90_COLS: tuple[str, ...] = (
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "us_key_passes",
    "us_xGChain",
    "us_xGBuildup",
    "total_points",
)

# Columns for rolling hint features (LSTM)
ROLL_HINT_COLS: tuple[str, ...] = (
    "minutes",
    "total_points",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "ict_index",
)

# Extended rolling columns for XGBoost
ROLLING_COLS: tuple[str, ...] = (
    "minutes",
    "starts",
    "total_points",
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "ict_index",
    "bps",
    "bonus",
    "threat",
    "influence",
    "creativity",
    "us_key_passes",
    "us_xGChain",
    "us_xGBuildup",
)

# Fixture context columns
FIXTURE_COLS: tuple[str, ...] = (
    "was_home",
    "opp_dyn_attack",
    "opp_dyn_defence",
    "opp_dyn_overall",
)

# Output columns for predictions
OUTPUT_COLS: tuple[str, ...] = (
    "name",
    "player_id",
    "team",
    "season",
    "GW",
    "was_home",
    "opponent_name",
    "total_points",
)


def get_lstm_feature_columns(roll_window: int, available_cols: list[str]) -> list[str]:
    """Get feature columns for LSTM model."""
    feature_cols: list[str] = []
    feature_cols.extend(BASE_COLS)
    feature_cols.extend([f"{col}_per90" for col in PER90_COLS])
    feature_cols.extend([f"roll_{roll_window}_{col}" for col in ROLL_HINT_COLS])
    feature_cols.extend(FIXTURE_COLS)
    return [col for col in feature_cols if col in available_cols]


def get_xgboost_feature_columns(
    roll_windows: tuple[int, ...], available_cols: list[str]
) -> list[str]:
    """Get feature columns for XGBoost model."""
    feature_cols: list[str] = []
    for window in roll_windows:
        for col in ROLLING_COLS:
            feature_cols.append(f"roll_{window}_{col}")
        for col in PER90_COLS:
            feature_cols.append(f"roll_{window}_{col}_per90")
        feature_cols.append(f"roll_hist_count_{window}")
    feature_cols.extend(FIXTURE_COLS)
    return [col for col in feature_cols if col in available_cols]
