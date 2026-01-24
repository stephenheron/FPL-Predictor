"""Fixture data utilities for building future prediction rows."""

import numpy as np
import pandas as pd


def build_future_rows(
    df: pd.DataFrame,
    fixtures_path: str,
    teams_path: str,
    predict_gw: int,
) -> pd.DataFrame:
    """Build future rows for prediction when GW data doesn't exist.

    Takes the latest row for each player and creates projected rows
    for the upcoming gameweek based on fixture data.

    Args:
        df: Current season DataFrame with player data.
        fixtures_path: Path to fixtures CSV.
        teams_path: Path to teams CSV.
        predict_gw: Gameweek to generate predictions for.

    Returns:
        DataFrame with future rows for all players in upcoming fixtures.

    Raises:
        ValueError: If no fixtures found for the specified GW.
    """
    fixtures = pd.read_csv(fixtures_path)
    fixtures = fixtures[fixtures["event"] == predict_gw]
    if fixtures.empty:
        raise ValueError(f"No fixtures found for GW {predict_gw} in {fixtures_path}")

    teams = pd.read_csv(teams_path)
    team_id_to_name = teams.set_index("id")["name"].to_dict()
    team_id_to_short = teams.set_index("id")["short_name"].to_dict()
    team_id_to_code = teams.set_index("id")["code"].to_dict()
    team_name_to_id = teams.set_index("name")["id"].to_dict()

    # Build fixture contexts (home and away for each match)
    contexts: list[dict] = []
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

    # Get latest row for each player
    latest_rows = (
        df.sort_values(["season", "GW"])
        .groupby("player_id", as_index=False)
        .tail(1)
        .copy()
    )
    latest_rows["team_id"] = latest_rows["team"].map(team_name_to_id.get)

    missing_mask = latest_rows["team_id"].isna()
    if bool(missing_mask.any()):
        missing = latest_rows.loc[missing_mask, "team"].dropna().unique().tolist()
        raise ValueError(f"Missing team ids for: {missing}")

    # Create future rows for each fixture context
    future_rows: list[pd.DataFrame] = []
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

        # Clear unknown future values
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
