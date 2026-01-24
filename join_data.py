import numpy as np
import pandas as pd
from pathlib import Path
import unicodedata
import re
from typing import Optional


def normalize_name(name: str) -> str:
    """Normalize a player name for matching."""
    # Handle Turkish/special characters that don't decompose with NFD
    turkish_map = {
        "ı": "i",
        "İ": "i",  # Turkish dotless i
        "ğ": "g",
        "Ğ": "g",  # Turkish soft g
        "ş": "s",
        "Ş": "s",  # Turkish s with cedilla
        "ü": "u",
        "Ü": "u",  # Turkish u with umlaut
        "ö": "o",
        "Ö": "o",  # Turkish o with umlaut
        "ç": "c",
        "Ç": "c",  # C with cedilla
        "ñ": "n",
        "Ñ": "n",  # Spanish n with tilde
        "ø": "o",
        "Ø": "o",  # Nordic o
        "æ": "ae",
        "Æ": "ae",  # Nordic ae
        "ß": "ss",  # German sharp s
    }
    for char, replacement in turkish_map.items():
        name = name.replace(char, replacement)

    # Remove accents
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    # Replace spaces and hyphens with underscores
    name = re.sub(r"[\s-]+", "_", name)
    # Remove any other special characters
    name = re.sub(r"[^\w]", "", name)
    return name.lower()


# Manual mapping: FPL name -> Understat player ID
# For players whose names differ significantly between FPL and Understat
MANUAL_NAME_TO_ID = {
    # Brazilian players known by nicknames
    "Carlos Henrique Casimiro": 2248,  # Casemiro
    "Joelinton Cássio Apolinário de Lira": 87,  # Joelinton
    "Norberto Bercique Gomes Betuncal": 9983,  # Beto
    "Richarlison de Andrade": 6026,  # Richarlison
    "Alisson Ramses Becker": 1257,  # Alisson
    "Ederson Santana de Moraes": 6054,  # Ederson
    "Lucas Tolentino Coelho de Lima": 7365,  # Lucas Paquetá
    "Francisco Evanilson de Lima Barbosa": 12963,  # Evanilson
    "Jorge Luiz Frello Filho": 1389,  # Jorginho
    "Willian Borges da Silva": 700,  # Willian
    "Danilo dos Santos de Oliveira": 11317,  # Danilo
    "Gabriel dos Santos Magalhães": 5613,  # Gabriel (Arsenal defender)
    "Gabriel Fernando de Jesus": 5543,  # Gabriel Jesus
    "Gabriel Martinelli Silva": 7752,  # Martinelli
    "Murillo Santiago Costa dos Santos": 12123,  # Murillo
    "Diogo Dalot Teixeira": 7281,  # Diogo Dalot (Man Utd)
    "Diogo Teixeira da Silva": 6854,  # Diogo Jota (Liverpool)
    "Antony Matheus dos Santos": 11094,  # Antony
    "Norberto Murara Neto": 1297,  # Neto (Bournemouth GK)
    "Vini de Souza Costa": 10872,  # Vinicius Souza
    "Felipe Rodrigues da Silva": 7921,  # Felipe (Forest)
    "João Maria Lobo Alves Palhares Costa Palhinha Gonçalves": 10715,  # Palhinha
    # Players with name format differences
    "Benjamin White": 7298,  # Ben White
    "Matty Cash": 8864,  # Matthew Cash
    "Tino Livramento": 9512,  # Valentino Livramento
    "Destiny Udogie": 8831,  # Iyenoma Destiny Udogie
    "Lesley Ugochukwu": 9451,  # Chimuanya Ugochukwu
    "Amad Diallo": 8127,  # Amad Diallo Traore
    "Joe Gomez": 987,  # Joseph Gomez
    "Hwang Hee-chan": 8845,  # Hee-Chan Hwang
    "Yehor Yarmoliuk": 11772,  # Yehor Yarmolyuk
    "João Victor Gomes da Silva": 12766,  # Jota Silva
    "Toti António Gomes": 10293,  # Toti
    "Joe Aribo": 10766,  # Joe Ayodele-Aribo
    "Cheick Doucouré": 8666,  # Cheick Oumar Doucoure
    "Jaden Philogene": 9415,  # Jaden Philogene-Bidace
    "Olu Aina": 725,  # Ola Aina
    "Benoît Badiashile": 7240,  # Benoit Badiashile Mukinayi
    "Alexandre Moreno Lopera": 4120,  # Álex Moreno
    "Maximilian Kilman": 7332,  # Max Kilman
    "Ryan Giles": 7277,  # Ryan John Giles
    "Sam Szmodics": 12752,  # Sammie Szmodics
    "Josh King": 12410,  # Joshua King
    "Nayef Aguerd": 6935,  # Naif Aguerd
    "Ollie Scarles": 12358,  # Oliver Scarles
    "Amari'i Bell": 11713,  # Amari'i Bell (HTML entity in understat)
    "João Pedro Ferreira Silva": 8272,  # Joao Pedro
    "Lewis Cook": 1789,  # Lewis Cook
    "Enes Ünal": 6219,  # Enes Unal
    "João Victor Gomes da Silva": 11384,  # Joao Gomes
    "Anis Slimane": 11730,  # Anis Ben Slimane
    "Hamed Traorè": 6986,  # Hamed Junior Traore
    "Ben Brereton": 11815,  # Ben Brereton Diaz
    "Carlos Roberto Forbs Borges": 13092,  # Carlos Forbs
    "Arnaud Kalimuendo": 8056,  # Arnaud Kalimuendo Muinga
    "Junior Kroupi": 11504,  # Eli Junior Kroupi
    "Yegor Yarmolyuk": 11772,  # Yehor Yarmolyuk
    "Yegor Yarmoliuk": 11772,  # Yehor Yarmolyuk
    "Odysseas Vlachodimos": 375,  # Odisseas Vlachodimos
    "Michale Olakigbe": 11487,  # Michael Olakigbe
    "Will Smallbone": 8224,  # William Smallbone
    "Łukasz Fabiański": 706,  # Lukasz Fabianski
    "Abdukodir Khusanov": 11763,  # Abduqodir Khusanov
    "Nico O'Reilly": 11592,  # Nico O'Reilly (HTML apostrophe)
    "Daniel Ballard": 13715,  # Dan Ballard
    "Luke O'Nien": 14099,  # Luke O'Nien (HTML apostrophe)
    "Trey Nyoni": 12203,  # Treymaurice Nyoni
    "Yéremy Pino Santos": 9024,  # Yeremi Pino
    "Álex Jiménez Sánchez": 12168,  # Alejandro Jimenez
    "Sávio Moreira de Oliveira": 11735,  # Savio (Savinho)
    "Murillo Costa dos Santos": 12123,  # Murillo
    "Jair Paula da Cunha Filho": 13779,  # Jair
    "André Trindade da Costa Neto": 13022,  # Andre
    "Felipe Rodrigues da Silva": 13068,  # Morato
    "Welington Damascena Santos": 13403,  # Welington
    "Estêvão Almeida de Oliveira Gonçalves": 13775,  # Estevao
    "Victor da Silva": 182,  # Victor da Silva
    "Jordan Beyer": 160,  # Jordan Beyer
    "Felipe Augusto de Almeida Monteiro": 445,  # Felipe Augusto
    "Jonathan Castro Otto": 560,  # Jonny Otto
    "Igor Thiago Nascimento Rodrigues": 13222,  # Thiago
    "João Pedro Ferreira Silva": 12766,  # Jota Silva
    "Francisco Jorge Tomás Oliveira": 10327,  # Francisco Oliveira
    "Kevin Santos Lopes de Macedo": 14030,  # Kevin
    "Rayan Cherki": 8094,  # Mathis Cherki
    "Fer López González": 13200,  # Fernando Lopez
    "Pablo Felipe Pereira de Jesus": 14290,  # Pablo Felipe
    "Enes Ünal": 6219,  # Enes Unal
    "Lewis Cook": 1789,  # Lewis Cook
    "Mathias Jorgensen": 999999,  # Manual placeholder
    # Players with apostrophes in names
    "Dara O'Shea": 8756,  # Dara O'Shea
    "Jake O'Brien": 12014,  # Jake O'Brien
    "Matt O'Riley": 13206,  # Matt O'Riley
    # Goalkeeper/special cases
    "André Onana": 10913,  # André Onana (different from Amadou Onana)
    "Gabriel Osho": 12151,  # Gabriel Osho
    # Eastern European name differences
    "Đorđe Petrović": 12032,  # Djordje Petrovic
    # Players with nicknames in FPL name
    "Rodrigo 'Rodri' Hernandez": 2496,  # Rodri (Man City)
    "Rodrigo 'Rodri' Hernandez Cascante": 2496,  # Rodri alternate name
    "Rodrigo Hernandez": 2496,  # Rodri without nickname
    "Sávio 'Savinho' Moreira de Oliveira": 11735,  # Savinho (Sávio)
    # Collision fixes (players who share name parts)
    "Jonny Evans": 807,  # Jonny Evans (not Jonny from Wolves)
    "Kyle Walker-Peters": 885,  # Kyle Walker-Peters (not Kyle Walker)
    "Kevin Danso": 5261,  # Kevin Danso
    "Emerson Palmieri dos Santos": 1245,  # Emerson Palmieri (West Ham) - FIXED
    "Emerson Leite de Souza Junior": 7430,  # Emerson Royal (Spurs)
}

# Common name parts that shouldn't be matched alone (too many collisions)
COMMON_NAME_PARTS = {
    "gabriel",
    "andre",
    "rodrigo",
    "lucas",
    "pedro",
    "bruno",
    "matheus",
    "silva",
    "santos",
    "oliveira",
    "souza",
    "lima",
    "costa",
    "ferreira",
    "dos",
    "da",
    "de",
    "do",
    "neto",
    "junior",
    "emerson",
    "kevin",
    "kyle",
    "van",
    "der",
    "den",
    "mohamed",
    "mohammed",
    "jose",
    "carlos",
    "antonio",
}


def create_name_variants(
    name: str, include_single_names: bool = True, conservative: bool = False
) -> list:
    """Create multiple name variants for fuzzy matching.

    Variants are ordered from most specific to least specific.
    Single-name variants are only included if include_single_names=True.
    If conservative=True, skip variants that use common/ambiguous name parts.
    """
    normalized = normalize_name(name)
    parts = normalized.split("_")

    variants = [normalized]

    # Full reversal (for Asian name ordering)
    if len(parts) >= 2:
        variants.append("_".join(reversed(parts)))

    if len(parts) >= 2:
        # First + Last
        first_last = f"{parts[0]}_{parts[-1]}"
        first_second = f"{parts[0]}_{parts[1]}"

        # In conservative mode, only add if neither part is too common
        if conservative:
            if parts[0] not in COMMON_NAME_PARTS and parts[-1] not in COMMON_NAME_PARTS:
                variants.append(first_last)
            if parts[0] not in COMMON_NAME_PARTS and parts[1] not in COMMON_NAME_PARTS:
                variants.append(first_second)
            # Reversed variants for Asian names
            if parts[-1] not in COMMON_NAME_PARTS and parts[0] not in COMMON_NAME_PARTS:
                variants.append(f"{parts[-1]}_{parts[0]}")
        else:
            variants.append(first_last)
            variants.append(first_second)
            variants.append(f"{parts[-1]}_{parts[0]}")
            variants.append(f"{parts[1]}_{parts[0]}")

    # For longer names, try various two-part combinations
    if len(parts) >= 3:
        mid_last = f"{parts[1]}_{parts[2]}"
        second_last = f"{parts[-2]}_{parts[-1]}"
        if conservative:
            if parts[1] not in COMMON_NAME_PARTS and parts[2] not in COMMON_NAME_PARTS:
                variants.append(mid_last)
            if (
                parts[-2] not in COMMON_NAME_PARTS
                and parts[-1] not in COMMON_NAME_PARTS
            ):
                variants.append(second_last)
        else:
            variants.append(mid_last)
            variants.append(second_last)

    # Single-name variants (risky - only use for understat indexing, not FPL lookup)
    if include_single_names:
        variants.append(parts[0])  # First name
        if len(parts) >= 2:
            variants.append(parts[-1])  # Last name
        if len(parts) >= 3:
            for part in parts[1:-1]:  # Middle names
                if part not in variants:
                    variants.append(part)

    return variants


def load_understat_data(understat_dir: Path) -> pd.DataFrame:
    """Load all understat data into a single DataFrame with player name and date as keys."""
    all_data = []

    for csv_file in understat_dir.glob("*.csv"):
        # Extract player name and ID from filename (e.g., "Alex_Iwobi_500.csv")
        name_with_id = csv_file.stem
        parts = name_with_id.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            name_part = parts[0]
            player_id = int(parts[1])
        else:
            name_part = name_with_id
            player_id = None

        # Read the CSV
        df = pd.read_csv(csv_file)
        df["understat_player_name"] = name_part.replace("_", " ")
        df["understat_player_id"] = player_id
        df["normalized_name"] = normalize_name(name_part)
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


def add_dynamic_opponent_strengths(
    merged_df: pd.DataFrame,
    ratings_state: Optional[dict] = None,
    k_factor: float = 0.2,
) -> tuple[pd.DataFrame, dict]:
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
    if missing_cols:
        print(
            "Skipping dynamic opponent strengths - missing columns: "
            + ", ".join(missing_cols)
        )
        if ratings_state is None:
            ratings_state = {
                "teams": {},
                "league_mean": {
                    "attack_home": 0.0,
                    "attack_away": 0.0,
                    "defence_home": 0.0,
                    "defence_away": 0.0,
                },
            }
        return merged_df, ratings_state

    rating_slots = ["attack_home", "attack_away", "defence_home", "defence_away"]

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
                f"{col} stats: min={stats['min']:.2f}, mean={stats['mean']:.2f}, max={stats['max']:.2f}"
            )
    return merged_df, ratings_state


def process_season(
    season: str,
    data_base_dir: Path,
    output_dir: Path,
    ratings_state: Optional[dict] = None,
    k_factor: float = 0.2,
) -> tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Process a single season and return the merged DataFrame and ratings."""
    print(f"\n{'=' * 50}")
    print(f"Processing season: {season}")
    print(f"{'=' * 50}")

    base_dir = data_base_dir / season
    merged_gw_path = base_dir / "gws" / "merged_gw.csv"
    understat_dir = base_dir / "understat"
    teams_path = base_dir / "teams.csv"

    if not merged_gw_path.exists():
        print(f"  Skipping: {merged_gw_path} not found")
        return None, ratings_state
    if not understat_dir.exists():
        print(f"  Skipping: {understat_dir} not found")
        return None, ratings_state
    if not teams_path.exists():
        print(f"  Skipping: {teams_path} not found")
        return None, ratings_state

    # Load FPL data
    print("Loading FPL data...")
    fpl_df = pd.read_csv(merged_gw_path)
    fpl_df["match_date"] = pd.to_datetime(fpl_df["kickoff_time"]).dt.date.astype(str)
    fpl_df["season"] = season

    drop_fpl_cols = {
        "modified",
        "clearances_blocks_interceptions",
        "defensive_contribution",
        "recoveries",
        "tackles",
    }
    fpl_df = fpl_df.drop(columns=[c for c in drop_fpl_cols if c in fpl_df.columns])
    fpl_df = fpl_df.drop(columns=[c for c in fpl_df.columns if c.startswith("mng_")])

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
        print(f"  Skipping: No understat data found")
        return None, ratings_state
    understat_df["match_date"] = pd.to_datetime(
        understat_df["date"], format="mixed"
    ).dt.date.astype(str)

    print(f"Understat data: {len(understat_df)} rows")

    # Build understat lookup - two lookups:
    # 1. Primary: exact normalized name (no collisions)
    # 2. Secondary: name variants (may have collisions, used as fallback)
    understat_primary = {}  # (normalized_name, date) -> row
    understat_variants = {}  # (variant, date) -> row
    single_name_players = set()  # Players known by single name in understat

    for _, row in understat_df.iterrows():
        name = row["understat_player_name"]
        normalized = normalize_name(name)
        parts = normalized.split("_")

        # Track if this is a single-name player (e.g., "Alisson", "Joelinton")
        if len(parts) == 1:
            single_name_players.add(normalized)

        # Primary lookup: exact normalized name
        key = (normalized, row["match_date"])
        if key not in understat_primary:
            understat_primary[key] = row

        # Secondary lookup: all variants (for fallback matching)
        name_variants = create_name_variants(name, include_single_names=True)
        for variant in name_variants:
            key = (variant, row["match_date"])
            if key not in understat_variants:
                understat_variants[key] = row

    # Rename understat columns to avoid conflicts (prefix with 'us_')
    # Keep understat_player_id as 'player_id' (without prefix) since it's the consistent cross-season ID
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

    # Build a lookup from understat player ID to rows for that player
    understat_by_id = {}
    for _, row in understat_df.iterrows():
        player_id = row["understat_player_id"]
        if player_id is not None:
            date = row["match_date"]
            if player_id not in understat_by_id:
                understat_by_id[player_id] = {}
            understat_by_id[player_id][date] = row

    # Match FPL rows to understat data
    # Strategy: Manual mapping first, then exact match, then variants, then single-name
    print("Matching data...")
    manual_matches = 0
    matches = []
    for idx, fpl_row in fpl_df.iterrows():
        match_date = fpl_row["match_date"]
        fpl_name = fpl_row["name"]
        matched_row = None

        # 0. Check manual mapping first (highest priority)
        has_manual_mapping = fpl_name in MANUAL_NAME_TO_ID
        if has_manual_mapping:
            player_id = MANUAL_NAME_TO_ID[fpl_name]
            if (
                player_id in understat_by_id
                and match_date in understat_by_id[player_id]
            ):
                matched_row = understat_by_id[player_id][match_date]
                manual_matches += 1
            # If manual mapping exists but no data for this date, don't fall back to variant matching
            # (to avoid matching wrong player with similar name)

        # 1. Try exact normalized name match in primary lookup (safest)
        # Skip if player has manual mapping (to avoid collisions)
        if matched_row is None and not has_manual_mapping:
            fpl_normalized = normalize_name(fpl_name)
            key = (fpl_normalized, match_date)
            if key in understat_primary:
                matched_row = understat_primary[key]

        # 2. Try name variants (without single names) in primary lookup
        # Skip if player has manual mapping (to avoid collisions)
        if matched_row is None and not has_manual_mapping:
            name_variants = create_name_variants(
                fpl_name, include_single_names=False, conservative=False
            )
            for variant in name_variants:
                key = (variant, match_date)
                if key in understat_primary:
                    matched_row = understat_primary[key]
                    break

        # 3. Try single-name match ONLY for known single-name players
        # Skip if player has manual mapping (to avoid collisions)
        # Also skip if FPL name has multiple parts (to avoid Gabriel Jesus -> Gabriel defender collision)
        if matched_row is None and not has_manual_mapping:
            fpl_normalized = normalize_name(fpl_name)
            fpl_parts = fpl_normalized.split("_")
            # Only use single-name matching if FPL name is also short (1-2 parts)
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
    merged_df = pd.concat([fpl_df.reset_index(drop=True), understat_matched_df], axis=1)
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
            f"Matched {matched}/{total} rows ({100 * matched / total:.1f}%) [{manual_matches} via manual mapping]"
        )
    else:
        print(f"Matched 0/{total} rows (0.0%) [{manual_matches} via manual mapping]")

    # Check match rate for players who actually played
    played = merged_df[merged_df["minutes"] > 0]
    if len(played) > 0 and metric_col:
        played_matched = played[metric_col].notna().sum()
        played_total = len(played)
        print(
            f"Matched (players with minutes>0): {played_matched}/{played_total} ({100 * played_matched / played_total:.1f}%)"
        )

        # Show some unmatched examples (only for players who played)
        unmatched = merged_df[
            (merged_df[metric_col].isna()) & (merged_df["minutes"] > 0)
        ][["name", "match_date", "team"]].drop_duplicates()
        if len(unmatched) > 0:
            print(
                f"\nSample unmatched players who played ({min(5, len(unmatched))} of {len(unmatched)}):"
            )
            print(unmatched.head(5).to_string(index=False))

    if metric_col and season == "2022-23":
        unmatched_played_mask = (merged_df["minutes"] > 0) & (
            merged_df[metric_col].isna()
        )
        removed = int(unmatched_played_mask.sum())
        if removed:
            merged_df = merged_df[~unmatched_played_mask].copy()
            print(f"Removed {removed} unmatched rows from 2022-23")

    # Drop rows without an understat player_id
    missing_player_id = merged_df["player_id"].isna()
    removed_missing_id = int(missing_player_id.sum())
    if removed_missing_id:
        merged_df = merged_df[~missing_player_id].copy()
        print(f"Removed {removed_missing_id} rows without player_id")

    merged_df, ratings_state = add_dynamic_opponent_strengths(
        merged_df, ratings_state=ratings_state, k_factor=k_factor
    )

    # Drop helper columns
    merged_df = merged_df.drop(columns=["match_date"])


    # Save individual season file
    output_path = output_dir / f"merged_fpl_understat_{season}.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    return merged_df, ratings_state


def main():
    data_base_dir = Path(
        "/Users/stephenheron/Workspace/f/fpl-prediction/Fantasy-Premier-League/data"
    )
    output_dir = Path("/Users/stephenheron/Workspace/f/fpl-prediction")

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

    # Combine all seasons into one file
    if all_dfs:
        print(f"\n{'=' * 50}")
        print("Combining all seasons...")
        print(f"{'=' * 50}")
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_path = output_dir / "merged_fpl_understat_all.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"Combined {len(all_dfs)} seasons: {len(combined_df)} total rows")
        print(f"Saved to {combined_path}")


if __name__ == "__main__":
    main()
