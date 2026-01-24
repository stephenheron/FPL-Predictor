import argparse

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from train_mid_xgb import build_features, get_feature_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MID predictions using saved model.")
    parser.add_argument(
        "--input-file",
        default="merged_fpl_understat_2025-26.csv",
        help="Season CSV to score.",
    )
    parser.add_argument(
        "--model-file",
        default="xgb_mid_model.json",
        help="Saved XGBoost model file.",
    )
    parser.add_argument(
        "--position",
        default="MID",
        help="Position to score.",
    )
    parser.add_argument(
        "--roll-windows",
        nargs="+",
        type=int,
        default=[3, 5, 8],
        help="Rolling windows used in training.",
    )
    parser.add_argument(
        "--predict-gw",
        type=int,
        default=None,
        help="Optional GW filter to score only a single GW.",
    )
    parser.add_argument(
        "--fixtures-file",
        default="Fantasy-Premier-League/data/2025-26/fixtures.csv",
        help="Fixtures CSV for upcoming GWs.",
    )
    parser.add_argument(
        "--teams-file",
        default="Fantasy-Premier-League/data/2025-26/teams.csv",
        help="Teams CSV for opponent strengths.",
    )
    parser.add_argument(
        "--output-file",
        default="mid_predictions.csv",
        help="Where to write predictions.",
    )
    return parser.parse_args()


def build_future_rows(
    df: pd.DataFrame,
    fixtures_path: str,
    teams_path: str,
    predict_gw: int,
) -> pd.DataFrame:
    fixtures = pd.read_csv(fixtures_path)
    fixtures = fixtures[fixtures["event"] == predict_gw]
    if fixtures.empty:
        raise ValueError(f"No fixtures found for GW {predict_gw} in {fixtures_path}")

    teams = pd.read_csv(teams_path)
    team_id_to_name = teams.set_index("id")["name"].to_dict()
    team_id_to_short = teams.set_index("id")["short_name"].to_dict()
    team_id_to_code = teams.set_index("id")["code"].to_dict()
    team_name_to_id = teams.set_index("name")["id"].to_dict()

    strength_cols = [
        "strength_overall_home",
        "strength_overall_away",
        "strength_attack_home",
        "strength_attack_away",
        "strength_defence_home",
        "strength_defence_away",
    ]
    strength_map = teams.set_index("id")[strength_cols].to_dict(orient="index")

    contexts = []
    for _, fixture in fixtures.iterrows():
        home_id = int(fixture["team_h"])
        away_id = int(fixture["team_a"])
        contexts.append(
            {
                "team_id": home_id,
                "opponent_id": away_id,
                "was_home": True,
                "opponent_strength": fixture["team_a_difficulty"],
                "fixture_id": fixture["id"],
            }
        )
        contexts.append(
            {
                "team_id": away_id,
                "opponent_id": home_id,
                "was_home": False,
                "opponent_strength": fixture["team_h_difficulty"],
                "fixture_id": fixture["id"],
            }
        )

    latest_rows = (
        df.sort_values(["season", "GW"])
        .groupby("player_id", as_index=False)
        .tail(1)
        .copy()
    )
    latest_rows["team_id"] = latest_rows["team"].map(team_name_to_id)
    if latest_rows["team_id"].isna().any():
        missing = latest_rows[latest_rows["team_id"].isna()]["team"].unique()
        raise ValueError(f"Missing team ids for: {missing}")

    future_rows = []
    for context in contexts:
        team_rows = latest_rows[latest_rows["team_id"] == context["team_id"]].copy()
        if team_rows.empty:
            continue

        opponent_id = context["opponent_id"]
        opponent_name = team_id_to_name.get(opponent_id)
        opponent_short = team_id_to_short.get(opponent_id)
        opponent_code = team_id_to_code.get(opponent_id)
        opponent_strengths = strength_map.get(opponent_id, {})

        team_rows["GW"] = predict_gw
        team_rows["round"] = predict_gw
        team_rows["fixture"] = context["fixture_id"]
        team_rows["was_home"] = context["was_home"]
        team_rows["opponent_team"] = opponent_id
        team_rows["opponent_name"] = opponent_name
        team_rows["opponent_short_name"] = opponent_short
        team_rows["opponent_code"] = opponent_code
        team_rows["opponent_strength"] = context["opponent_strength"]

        for col, value in opponent_strengths.items():
            team_rows[f"opponent_{col}"] = value

        team_rows["team_h_score"] = np.nan
        team_rows["team_a_score"] = np.nan
        team_rows["total_points"] = np.nan
        team_rows["is_future"] = True
        future_rows.append(team_rows)

    if not future_rows:
        raise ValueError(f"No future rows created for GW {predict_gw}")

    return pd.concat(future_rows, ignore_index=True)


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input_file)
    df["is_future"] = False
    if args.predict_gw is not None and (df["GW"] == args.predict_gw).sum() == 0:
        future_rows = build_future_rows(
            df,
            args.fixtures_file,
            args.teams_file,
            args.predict_gw,
        )
        df = pd.concat([df, future_rows], ignore_index=True)
    df = build_features(df, args.position, tuple(args.roll_windows))

    feature_cols = get_feature_columns(df, tuple(args.roll_windows))
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    model = XGBRegressor()
    model.load_model(args.model_file)

    if args.predict_gw is not None:
        df = df[df["GW"] == args.predict_gw]

    preds = model.predict(df[feature_cols])

    output_cols = [
        "name",
        "player_id",
        "team",
        "season",
        "GW",
        "was_home",
        "opponent_name",
        "opponent_strength",
        "total_points",
    ]
    existing_cols = [col for col in output_cols if col in df.columns]
    result = df[existing_cols].copy()
    result["predicted_points"] = preds
    result.to_csv(args.output_file, index=False)

    print(f"Saved predictions to: {args.output_file}")


if __name__ == "__main__":
    main()
