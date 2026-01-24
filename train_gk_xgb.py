import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


@dataclass
class Config:
    position: str
    roll_windows: tuple
    train_windows: tuple
    train_files: tuple
    holdout_file: str
    importance_out: str
    model_out: str
    fixed_window_config: str
    train_full: bool
    random_state: int = 42


ROLLING_COLS = [
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
    "saves",
    "clean_sheets",
    "expected_goals_conceded",
]

PER90_COLS = [
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "saves",
    "expected_goals_conceded",
    "total_points",
]

FIXTURE_COLS = [
    "was_home",
    "opp_dyn_attack",
    "opp_dyn_defence",
    "opp_dyn_overall",
]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train GK XGBoost model with walk-forward CV.")
    parser.add_argument(
        "--train-files",
        nargs="+",
        default=[
            "merged_fpl_understat_2022-23.csv",
            "merged_fpl_understat_2023-24.csv",
            "merged_fpl_understat_2024-25.csv",
        ],
    )
    parser.add_argument("--holdout-file", default="merged_fpl_understat_2025-26.csv")
    parser.add_argument("--position", default="GK")
    parser.add_argument("--roll-windows", nargs="+", type=int, default=[3, 5, 8])
    parser.add_argument("--train-windows", nargs="+", type=int, default=[15, 25, 35])
    parser.add_argument("--importance-out", default="feature_importance_gk.csv")
    parser.add_argument("--model-out", default="xgb_gk_model.json")
    parser.add_argument(
        "--fixed-window-config",
        default="best_windows.json",
        help="Optional JSON file with fixed training windows by position.",
    )
    parser.add_argument(
        "--train-full",
        action="store_true",
        help="Train on all seasons (includes holdout) and skip holdout eval.",
    )
    args = parser.parse_args()

    return Config(
        position=args.position,
        roll_windows=tuple(args.roll_windows),
        train_windows=tuple(args.train_windows),
        train_files=tuple(args.train_files),
        holdout_file=args.holdout_file,
        importance_out=args.importance_out,
        model_out=args.model_out,
        fixed_window_config=args.fixed_window_config,
        train_full=args.train_full,
    )


def load_season(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["GW"] = pd.to_numeric(df["GW"], errors="coerce")
    df["season"] = df["season"].astype(str)
    return df


def add_rolling_features(df: pd.DataFrame, roll_windows: tuple) -> pd.DataFrame:
    df = df.sort_values(["season", "player_id", "GW"]).copy()
    grouped = df.groupby(["season", "player_id"], sort=False)
    shifted_minutes = grouped["minutes"].shift(1)

    for window in roll_windows:
        count_name = f"roll_hist_count_{window}"
        df[count_name] = (
            shifted_minutes.groupby([df["season"], df["player_id"]])
            .rolling(window=window, min_periods=1)
            .count()
            .reset_index(level=[0, 1], drop=True)
        )

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


def build_features(df: pd.DataFrame, position: str, roll_windows: tuple) -> pd.DataFrame:
    df = df[df["position"] == position].copy()
    df = add_rolling_features(df, roll_windows)
    return df


def get_feature_columns(df: pd.DataFrame, roll_windows: tuple) -> list:
    feature_cols = []
    for window in roll_windows:
        for col in ROLLING_COLS:
            feature_cols.append(f"roll_{window}_{col}")
        for col in PER90_COLS:
            feature_cols.append(f"roll_{window}_{col}_per90")
        feature_cols.append(f"roll_hist_count_{window}")

    feature_cols.extend(FIXTURE_COLS)
    existing = [col for col in feature_cols if col in df.columns]
    return existing


def train_model(random_state: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )


def get_fold_pairs(df: pd.DataFrame, season: str, window: int) -> list:
    season_df = df[df["season"] == season]
    max_gw = int(season_df["GW"].max())
    pairs = []
    for gw in range(window + 1, max_gw + 1):
        train_mask = (season_df["GW"] >= gw - window) & (season_df["GW"] <= gw - 1)
        val_mask = season_df["GW"] == gw
        train_idx = season_df[train_mask].index
        val_idx = season_df[val_mask].index
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        pairs.append((train_idx, val_idx))
    return pairs


def evaluate_cv(
    df: pd.DataFrame, feature_cols: list, train_windows: tuple, random_state: int
) -> pd.DataFrame:
    results = []
    seasons = sorted(df["season"].unique())
    for window in train_windows:
        fold_mae = []
        fold_rmse = []
        for season in seasons:
            for train_idx, val_idx in get_fold_pairs(df, season, window):
                X_train = df.loc[train_idx, feature_cols]
                y_train = df.loc[train_idx, "total_points"]
                X_val = df.loc[val_idx, feature_cols]
                y_val = df.loc[val_idx, "total_points"]

                model = train_model(random_state)
                model.fit(X_train, y_train)
                preds = model.predict(X_val)

                fold_mae.append(mean_absolute_error(y_val, preds))
                fold_rmse.append(np.sqrt(mean_squared_error(y_val, preds)))

        results.append(
            {
                "train_window": window,
                "mae": float(np.mean(fold_mae)) if fold_mae else np.nan,
                "rmse": float(np.mean(fold_rmse)) if fold_rmse else np.nan,
                "folds": len(fold_mae),
            }
        )
    return pd.DataFrame(results)


def restrict_to_window(df: pd.DataFrame, window: int) -> pd.DataFrame:
    result = []
    for season in sorted(df["season"].unique()):
        season_df = df[df["season"] == season]
        max_gw = int(season_df["GW"].max())
        result.append(season_df[season_df["GW"] > max_gw - window])
    return pd.concat(result, ignore_index=False)


def main() -> None:
    config = parse_args()

    train_frames = [load_season(path) for path in config.train_files]
    holdout_df = load_season(config.holdout_file)

    train_df = pd.concat(train_frames, ignore_index=True)

    if config.train_full:
        train_df = pd.concat([train_df, holdout_df], ignore_index=True)

    train_df = build_features(train_df, config.position, config.roll_windows)
    holdout_df = build_features(holdout_df, config.position, config.roll_windows)

    feature_cols = get_feature_columns(train_df, config.roll_windows)

    best_window = None
    config_path = Path(config.fixed_window_config)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            window_map = json.load(handle)
        best_window = window_map.get(config.position)
        if best_window is not None:
            print(
                f"Using fixed window for {config.position} from {config.fixed_window_config}: {best_window}"
            )

    if best_window is None:
        cv_results = evaluate_cv(
            train_df, feature_cols, config.train_windows, config.random_state
        )
        cv_results = cv_results.sort_values("mae")

        best_window = int(cv_results.iloc[0]["train_window"])
        print("CV results (sorted by MAE):")
        print(cv_results.to_string(index=False))
        print(f"Best window: {best_window}")

    final_train = restrict_to_window(train_df, best_window)
    model = train_model(config.random_state)
    model.fit(final_train[feature_cols], final_train["total_points"])
    model.save_model(config.model_out)
    print(f"Saved model to: {config.model_out}")

    importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print("Top 25 feature importances:")
    print(importance_df.head(25).to_string(index=False))
    importance_df.to_csv(config.importance_out, index=False)
    print(f"Saved full feature importances to: {config.importance_out}")

    if config.train_full:
        print("Skipped holdout evaluation (train-full enabled).")
    else:
        holdout_preds = model.predict(holdout_df[feature_cols])
        holdout_mae = mean_absolute_error(holdout_df["total_points"], holdout_preds)
        holdout_rmse = np.sqrt(mean_squared_error(holdout_df["total_points"], holdout_preds))
        print("Holdout evaluation (2025-26):")
        print(f"MAE: {holdout_mae:.4f}")
        print(f"RMSE: {holdout_rmse:.4f}")


if __name__ == "__main__":
    main()
