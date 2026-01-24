import argparse

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from train_def_xgb import build_features, get_feature_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DEF predictions using saved model.")
    parser.add_argument(
        "--input-file",
        default="merged_fpl_understat_2025-26.csv",
        help="Season CSV to score.",
    )
    parser.add_argument(
        "--model-file",
        default="xgb_def_model.json",
        help="Saved XGBoost model file.",
    )
    parser.add_argument(
        "--position",
        default="DEF",
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
        help="Teams CSV for opponent lookup.",
    )
    parser.add_argument(
        "--output-file",
        default="def_predictions.csv",
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

    contexts = []
    for _, fixture in fixtures.iterrows():
        home_id = int(fixture["team_h"])
        away_id = int(fixture["team_a"])
        contexts.append(
            {
                "team_id": home_id,
                "opponent_id": away_id,
                "was_home": True,
                "fixture_id": fixture["id"],
            }
        )
        contexts.append(
            {
                "team_id": away_id,
                "opponent_id": home_id,
                "was_home": False,
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

        team_rows["GW"] = predict_gw
        team_rows["round"] = predict_gw
        team_rows["fixture"] = context["fixture_id"]
        team_rows["was_home"] = context["was_home"]
        team_rows["opponent_team"] = opponent_id
        team_rows["opponent_name"] = opponent_name
        team_rows["opponent_short_name"] = opponent_short
        team_rows["opponent_code"] = opponent_code

        for col in ["opp_dyn_attack", "opp_dyn_defence", "opp_dyn_overall"]:
            team_rows[col] = np.nan

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

    minutes_last_3 = None
    if args.predict_gw is not None:
        history = df[df["GW"] < args.predict_gw][["player_id", "GW", "minutes"]].copy()
        history["GW"] = pd.to_numeric(history["GW"], errors="coerce")
        minutes_lookup = history.groupby(["player_id", "GW"], sort=False)[
            "minutes"
        ].sum()
        minutes_lookup = minutes_lookup.to_dict()

        df = df[df["GW"] == args.predict_gw].copy()

        def last_three_minutes(player_id: float) -> float:
            return sum(
                minutes_lookup.get((player_id, gw), 0)
                for gw in range(args.predict_gw - 3, args.predict_gw)
            )

        minutes_last_3 = df["player_id"].map(last_three_minutes)
    elif "roll_3_minutes" in df.columns:
        minutes_last_3 = df["roll_3_minutes"].fillna(0) * 3
    else:
        minutes_last_3 = pd.Series(np.zeros(len(df)), index=df.index)

    preds = model.predict(df[feature_cols])

    multipliers = np.select(
        [
            minutes_last_3 >= 180,
            minutes_last_3 >= 90,
            minutes_last_3 > 0,
        ],
        [1.0, 0.7, 0.4],
        default=0.2,
    )
    preds = preds * multipliers

    output_cols = [
        "name",
        "player_id",
        "team",
        "season",
        "GW",
        "was_home",
        "opponent_name",
        "total_points",
    ]
    existing_cols = [col for col in output_cols if col in df.columns]
    result = df[existing_cols].copy()
    result["availability_multiplier"] = multipliers
    result["predicted_points"] = preds
    result.to_csv(args.output_file, index=False)

    print(f"Saved predictions to: {args.output_file}")


if __name__ == "__main__":
    main()
