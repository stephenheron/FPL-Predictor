"""Data joining pipeline for FPL and Understat data."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from fpl_prediction.config.player_mappings import MANUAL_NAME_TO_ID
from fpl_prediction.data.name_matching import create_name_variants, normalize_name


def load_understat_data(understat_dir: Path) -> pd.DataFrame:
    """Load all understat data into a single DataFrame.

    Args:
        understat_dir: Directory containing player CSV files.

    Returns:
        DataFrame with all understat data and normalized names.
    """
    all_data = []

    for csv_file in understat_dir.glob("*.csv"):
        name_with_id = csv_file.stem
        parts = name_with_id.rsplit("_", 1)

        if len(parts) == 2 and parts[1].isdigit():
            name_part = parts[0]
            player_id = int(parts[1])
        else:
            name_part = name_with_id
            player_id = None

        df = pd.read_csv(csv_file)
        df["understat_player_name"] = name_part.replace("_", " ")
        df["understat_player_id"] = player_id
        df["normalized_name"] = normalize_name(name_part)
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def add_dynamic_opponent_strengths(
    merged_df: pd.DataFrame,
    ratings_state: Optional[dict] = None,
    k_factor: float = 0.2,
) -> tuple[pd.DataFrame, dict]:
    """Add dynamic opponent strength features.

    Computes team attack/defence ratings based on xG performance.

    Args:
        merged_df: DataFrame with fixture data.
        ratings_state: Optional existing ratings state.
        k_factor: Learning rate for rating updates.

    Returns:
        Tuple of (updated DataFrame, ratings state).
    """
    required_cols = [
        "season",
        "fixture",
        "team",
        "opponent_name",
        "was_home",
        "kickoff_time",
        "expected_goals",
        "expected_goals_conceded",
    ]
    missing_cols = [col for col in required_cols if col not in merged_df.columns]

    rating_slots = ["attack_home", "attack_away", "defence_home", "defence_away"]

    if missing_cols:
        print(
            "Skipping dynamic opponent strengths - missing columns: "
            + ", ".join(missing_cols)
        )
        if ratings_state is None:
            ratings_state = {
                "teams": {},
                "league_mean": {slot: 0.0 for slot in rating_slots},
            }
        return merged_df, ratings_state

    if ratings_state is None:
        ratings_state = {
            "teams": {},
            "league_mean": {slot: 0.0 for slot in rating_slots},
        }

    fixture_cols = [
        "season",
        "fixture",
        "team",
        "opponent_name",
        "was_home",
        "kickoff_time",
    ]

    fixture_df = merged_df[
        fixture_cols + ["expected_goals", "expected_goals_conceded"]
    ].copy()
    fixture_df[["expected_goals", "expected_goals_conceded"]] = fixture_df[
        ["expected_goals", "expected_goals_conceded"]
    ].fillna(0)

    fixture_df["kickoff_time_dt"] = pd.to_datetime(
        fixture_df["kickoff_time"], errors="coerce"
    )

    fixture_agg = (
        fixture_df.groupby(fixture_cols, dropna=False, as_index=False)
        .agg(
            {
                "expected_goals": "sum",
                "expected_goals_conceded": "sum",
                "kickoff_time_dt": "max",
            }
        )
        .sort_values(["kickoff_time_dt", "fixture", "team"], kind="mergesort")
        .reset_index(drop=True)
    )

    home_mask = fixture_agg["was_home"] == True
    away_mask = fixture_agg["was_home"] == False

    overall_xg = fixture_agg["expected_goals"].mean()
    overall_xga = fixture_agg["expected_goals_conceded"].mean()
    if pd.isna(overall_xg):
        overall_xg = 0.0
    if pd.isna(overall_xga):
        overall_xga = 0.0

    home_xg = fixture_agg.loc[home_mask, "expected_goals"].mean()
    home_xga = fixture_agg.loc[home_mask, "expected_goals_conceded"].mean()
    away_xg = fixture_agg.loc[away_mask, "expected_goals"].mean()
    away_xga = fixture_agg.loc[away_mask, "expected_goals_conceded"].mean()

    if pd.isna(home_xg):
        home_xg = overall_xg
    if pd.isna(home_xga):
        home_xga = overall_xga
    if pd.isna(away_xg):
        away_xg = overall_xg
    if pd.isna(away_xga):
        away_xga = overall_xga

    def ensure_team(team_name: str) -> None:
        if team_name not in ratings_state["teams"]:
            ratings_state["teams"][team_name] = {
                slot: ratings_state["league_mean"][slot] for slot in rating_slots
            }

    def clamp_rating(value: float) -> float:
        return max(-1.5, min(1.5, value))

    records = []
    decay = 0.005
    eps = 1e-6

    grouped = fixture_agg.groupby(["fixture", "kickoff_time_dt"], sort=False)
    for _, group in grouped:
        teams_in_group = set(group["team"]).union(set(group["opponent_name"]))
        for team_name in teams_in_group:
            ensure_team(team_name)

        pre_ratings = {
            team_name: ratings_state["teams"][team_name].copy()
            for team_name in teams_in_group
        }
        updates = {
            team_name: {slot: 0.0 for slot in rating_slots}
            for team_name in teams_in_group
        }

        for row in group.itertuples(index=False):
            team_name = row.team
            opponent_name = row.opponent_name
            was_home = bool(row.was_home)

            if was_home:
                team_attack_key = "attack_home"
                team_defence_key = "defence_home"
                opp_attack_key = "attack_away"
                opp_defence_key = "defence_away"
                baseline_xg = home_xg
                baseline_xga = home_xga
            else:
                team_attack_key = "attack_away"
                team_defence_key = "defence_away"
                opp_attack_key = "attack_home"
                opp_defence_key = "defence_home"
                baseline_xg = away_xg
                baseline_xga = away_xga

            team_attack = pre_ratings[team_name][team_attack_key]
            team_defence = pre_ratings[team_name][team_defence_key]
            opp_attack = pre_ratings[opponent_name][opp_attack_key]
            opp_defence = pre_ratings[opponent_name][opp_defence_key]

            expected_xg = baseline_xg * np.exp(team_attack - opp_defence)
            expected_xga = baseline_xga * np.exp(opp_attack - team_defence)

            actual_xg = row.expected_goals
            actual_xga = row.expected_goals_conceded

            attack_delta = k_factor * (actual_xg - expected_xg) / max(baseline_xg, eps)
            defence_delta = k_factor * (expected_xga - actual_xga) / max(
                baseline_xga, eps
            )

            updates[team_name][team_attack_key] += attack_delta
            updates[team_name][team_defence_key] += defence_delta

            opp_attack_display = 1000.0 * np.exp(opp_attack)
            opp_defence_display = 1000.0 * np.exp(opp_defence)
            records.append(
                {
                    "season": row.season,
                    "fixture": row.fixture,
                    "team": team_name,
                    "opponent_name": opponent_name,
                    "was_home": row.was_home,
                    "kickoff_time": row.kickoff_time,
                    "opp_dyn_attack": opp_attack_display,
                    "opp_dyn_defence": opp_defence_display,
                    "opp_dyn_overall": (opp_attack_display + opp_defence_display) / 2.0,
                }
            )

        for team_name, deltas in updates.items():
            for slot, delta in deltas.items():
                updated = ratings_state["teams"][team_name][slot] * (1.0 - decay)
                updated += delta
                ratings_state["teams"][team_name][slot] = clamp_rating(updated)

    season_teams = set(fixture_agg["team"]).union(set(fixture_agg["opponent_name"]))
    if season_teams:
        ratings_state["league_mean"] = {
            slot: sum(ratings_state["teams"][team][slot] for team in season_teams)
            / len(season_teams)
            for slot in rating_slots
        }

    fixture_strengths = pd.DataFrame(records)
    merged_df = merged_df.merge(fixture_strengths, on=fixture_cols, how="left")

    for col in ["opp_dyn_attack", "opp_dyn_defence", "opp_dyn_overall"]:
        if col in merged_df.columns:
            stats = merged_df[col].describe()[["min", "mean", "max"]]
            print(
                f"{col} stats: min={stats['min']:.2f}, mean={stats['mean']:.2f}, "
                f"max={stats['max']:.2f}"
            )

    return merged_df, ratings_state


def process_season(
    season: str,
    data_base_dir: Path,
    output_dir: Path,
    ratings_state: Optional[dict] = None,
    k_factor: float = 0.2,
) -> tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Process a single season and return the merged DataFrame.

    Args:
        season: Season string (e.g., "2024-25").
        data_base_dir: Base directory with season data.
        output_dir: Directory to save output files.
        ratings_state: Optional existing ratings state.
        k_factor: Learning rate for opponent strength updates.

    Returns:
        Tuple of (merged DataFrame, updated ratings state).
    """
    print(f"\n{'=' * 50}")
    print(f"Processing season: {season}")
    print(f"{'=' * 50}")

    base_dir = data_base_dir / season
    merged_gw_path = base_dir / "gws" / "merged_gw.csv"
    understat_dir = base_dir / "understat"
    teams_path = base_dir / "teams.csv"

    for path, name in [
        (merged_gw_path, "merged_gw.csv"),
        (understat_dir, "understat"),
        (teams_path, "teams.csv"),
    ]:
        if not path.exists():
            print(f"  Skipping: {path} not found")
            return None, ratings_state

    # Load FPL data
    print("Loading FPL data...")
    fpl_df = pd.read_csv(merged_gw_path)
    fpl_df["match_date"] = pd.to_datetime(fpl_df["kickoff_time"]).dt.date.astype(str)
    fpl_df["season"] = season

    # Drop unused columns
    drop_fpl_cols = {
        "modified",
        "clearances_blocks_interceptions",
        "defensive_contribution",
        "recoveries",
        "tackles",
    }
    fpl_df = fpl_df.drop(columns=[c for c in drop_fpl_cols if c in fpl_df.columns])
    fpl_df = fpl_df.drop(columns=[c for c in fpl_df.columns if c.startswith("mng_")])

    # Add opponent team info
    print("Loading teams data...")
    teams_df = pd.read_csv(teams_path)
    opponent_columns = [
        "id",
        "name",
        "short_name",
        "code",
        "strength",
        "strength_overall_home",
        "strength_overall_away",
        "strength_attack_home",
        "strength_attack_away",
        "strength_defence_home",
        "strength_defence_away",
    ]
    opponent_df = teams_df[opponent_columns].rename(
        columns=lambda col: f"opponent_{col}"
    )
    fpl_df = fpl_df.merge(
        opponent_df, left_on="opponent_team", right_on="opponent_id", how="left"
    )
    fpl_df = fpl_df.drop(columns=["opponent_id"])
    print(f"FPL data: {len(fpl_df)} rows")

    # Load understat data
    print("Loading understat data...")
    understat_df = load_understat_data(understat_dir)
    if len(understat_df) == 0:
        print("  Skipping: No understat data found")
        return None, ratings_state
    understat_df["match_date"] = pd.to_datetime(
        understat_df["date"], format="mixed"
    ).dt.date.astype(str)
    print(f"Understat data: {len(understat_df)} rows")

    # Build understat lookups
    understat_primary: dict[tuple, pd.Series] = {}
    single_name_players: set[str] = set()

    for _, row in understat_df.iterrows():
        name = row["understat_player_name"]
        normalized = normalize_name(name)
        parts = normalized.split("_")

        if len(parts) == 1:
            single_name_players.add(normalized)

        key = (normalized, row["match_date"])
        if key not in understat_primary:
            understat_primary[key] = row

    # Build ID lookup
    understat_by_id: dict[int, dict[str, pd.Series]] = {}
    for _, row in understat_df.iterrows():
        player_id = row["understat_player_id"]
        if player_id is not None:
            date = row["match_date"]
            if player_id not in understat_by_id:
                understat_by_id[player_id] = {}
            understat_by_id[player_id][date] = row

    # Columns to extract from understat
    understat_cols = [
        c
        for c in understat_df.columns
        if c
        not in [
            "normalized_name",
            "match_date",
            "understat_player_name",
            "understat_player_id",
        ]
    ]
    drop_us_cols = {
        "a_goals",
        "a_team",
        "assists",
        "date",
        "goals",
        "h_a",
        "h_goals",
        "h_team",
        "position",
        "time",
        "xA",
        "xG",
        "xGA",
        "npxGA",
        "ppda",
        "ppda_allowed",
        "deep",
        "deep_allowed",
        "scored",
        "missed",
        "xpts",
        "result",
        "wins",
        "draws",
        "loses",
        "pts",
        "npxGD",
        "player_name",
        "games",
        "yellow_cards",
        "red_cards",
        "team_title",
        "isResult",
        "side",
        "h",
        "a",
        "datetime",
        "forecast",
    }
    understat_cols = [c for c in understat_cols if c not in drop_us_cols]
    print(f"Found {len(single_name_players)} single-name players in understat")

    # Match FPL rows to understat data
    print("Matching data...")
    manual_matches = 0
    matches = []

    for _, fpl_row in fpl_df.iterrows():
        match_date = fpl_row["match_date"]
        fpl_name = fpl_row["name"]
        matched_row = None

        has_manual_mapping = fpl_name in MANUAL_NAME_TO_ID
        if has_manual_mapping:
            player_id = MANUAL_NAME_TO_ID[fpl_name]
            if player_id in understat_by_id and match_date in understat_by_id[player_id]:
                matched_row = understat_by_id[player_id][match_date]
                manual_matches += 1

        if matched_row is None and not has_manual_mapping:
            fpl_normalized = normalize_name(fpl_name)
            key = (fpl_normalized, match_date)
            if key in understat_primary:
                matched_row = understat_primary[key]

        if matched_row is None and not has_manual_mapping:
            name_variants = create_name_variants(
                fpl_name, include_single_names=False, conservative=False
            )
            for variant in name_variants:
                key = (variant, match_date)
                if key in understat_primary:
                    matched_row = understat_primary[key]
                    break

        if matched_row is None and not has_manual_mapping:
            fpl_normalized = normalize_name(fpl_name)
            fpl_parts = fpl_normalized.split("_")
            if len(fpl_parts) <= 2:
                for part in fpl_parts:
                    if part in single_name_players:
                        key = (part, match_date)
                        if key in understat_primary:
                            matched_row = understat_primary[key]
                            break

        if matched_row is not None:
            match_data = {f"us_{c}": matched_row[c] for c in understat_cols}
            match_data["player_id"] = matched_row["understat_player_id"]
        else:
            if has_manual_mapping:
                match_data = {f"us_{c}": 0 for c in understat_cols}
                match_data["player_id"] = MANUAL_NAME_TO_ID[fpl_name]
            else:
                match_data = {f"us_{c}": None for c in understat_cols}
                match_data["player_id"] = None

        matches.append(match_data)

    # Add understat columns to FPL data
    understat_matched_df = pd.DataFrame(matches)
    merged_df = pd.concat(
        [fpl_df.reset_index(drop=True), understat_matched_df], axis=1
    )
    understat_prefixed_cols = [f"us_{c}" for c in understat_cols]
    merged_df.loc[merged_df["minutes"] == 0, understat_prefixed_cols] = 0

    # Check match statistics
    metric_col = None
    if "us_xG" in merged_df.columns:
        metric_col = "us_xG"
    elif "us_npxG" in merged_df.columns:
        metric_col = "us_npxG"

    total = len(merged_df)
    if metric_col:
        matched = merged_df[metric_col].notna().sum()
        print(
            f"Matched {matched}/{total} rows ({100 * matched / total:.1f}%) "
            f"[{manual_matches} via manual mapping]"
        )

        played = merged_df[merged_df["minutes"] > 0]
        if len(played) > 0:
            played_matched = played[metric_col].notna().sum()
            played_total = len(played)
            print(
                f"Matched (players with minutes>0): {played_matched}/{played_total} "
                f"({100 * played_matched / played_total:.1f}%)"
            )

            unmatched = merged_df[
                (merged_df[metric_col].isna()) & (merged_df["minutes"] > 0)
            ][["name", "match_date", "team"]].drop_duplicates()
            if len(unmatched) > 0:
                print(
                    f"\nSample unmatched players who played "
                    f"({min(5, len(unmatched))} of {len(unmatched)}):"
                )
                print(unmatched.head(5).to_string(index=False))

        if season == "2022-23":
            unmatched_played_mask = (merged_df["minutes"] > 0) & (
                merged_df[metric_col].isna()
            )
            removed = int(unmatched_played_mask.sum())
            if removed:
                merged_df = merged_df[~unmatched_played_mask].copy()
                print(f"Removed {removed} unmatched rows from 2022-23")

    # Drop rows without player_id
    missing_player_id = merged_df["player_id"].isna()
    removed_missing_id = int(missing_player_id.sum())
    if removed_missing_id:
        merged_df = merged_df[~missing_player_id].copy()
        print(f"Removed {removed_missing_id} rows without player_id")

    # Add dynamic opponent strengths
    merged_df, ratings_state = add_dynamic_opponent_strengths(
        merged_df, ratings_state=ratings_state, k_factor=k_factor
    )

    # Drop helper columns
    merged_df = merged_df.drop(columns=["match_date"])

    # Save output
    output_path = output_dir / f"merged_fpl_understat_{season}.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    return merged_df, ratings_state


def run_pipeline(
    data_base_dir: Path,
    output_dir: Path,
    seasons: list[str] | None = None,
) -> None:
    """Run the full data joining pipeline.

    Args:
        data_base_dir: Base directory with season data.
        output_dir: Directory to save output files.
        seasons: List of seasons to process (default: 2022-23 to 2025-26).
    """
    if seasons is None:
        seasons = ["2022-23", "2023-24", "2024-25", "2025-26"]

    all_dfs = []
    ratings_state = None

    for season in seasons:
        df, ratings_state = process_season(
            season,
            data_base_dir,
            output_dir,
            ratings_state=ratings_state,
        k_factor=0.2,
        )
        if df is not None:
            all_dfs.append(df)

    if all_dfs:
        print(f"\n{'=' * 50}")
        print("Combining all seasons...")
        print(f"{'=' * 50}")
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_path = output_dir / "merged_fpl_understat_all.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"Combined {len(all_dfs)} seasons: {len(combined_df)} total rows")
        print(f"Saved to {combined_path}")


def main() -> None:
    """Main entry point for join_data pipeline."""
    data_base_dir = Path("Fantasy-Premier-League/data")
    output_dir = Path(".")
    run_pipeline(data_base_dir, output_dir)


if __name__ == "__main__":
    main()
