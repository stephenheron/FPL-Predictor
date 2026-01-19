import pandas as pd
from pathlib import Path
import unicodedata
import re
from typing import Optional

def normalize_name(name: str) -> str:
    """Normalize a player name for matching."""
    # Handle Turkish/special characters that don't decompose with NFD
    turkish_map = {
        'ı': 'i', 'İ': 'i',  # Turkish dotless i
        'ğ': 'g', 'Ğ': 'g',  # Turkish soft g
        'ş': 's', 'Ş': 's',  # Turkish s with cedilla
        'ü': 'u', 'Ü': 'u',  # Turkish u with umlaut
        'ö': 'o', 'Ö': 'o',  # Turkish o with umlaut
        'ç': 'c', 'Ç': 'c',  # C with cedilla
        'ñ': 'n', 'Ñ': 'n',  # Spanish n with tilde
        'ø': 'o', 'Ø': 'o',  # Nordic o
        'æ': 'ae', 'Æ': 'ae',  # Nordic ae
        'ß': 'ss',  # German sharp s
    }
    for char, replacement in turkish_map.items():
        name = name.replace(char, replacement)

    # Remove accents
    name = unicodedata.normalize('NFD', name)
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
    # Replace spaces and hyphens with underscores
    name = re.sub(r'[\s-]+', '_', name)
    # Remove any other special characters
    name = re.sub(r'[^\w]', '', name)
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
    "Diogo Teixeira da Silva": 7281,  # Diogo Dalot

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
    "Dara O'Shea": 8756,  # Dara O'Shea

    # Goalkeeper/special cases
    "André Onana": 10913,  # André Onana (different from Amadou Onana)
    "Gabriel Osho": 12151,  # Gabriel Osho

    # Eastern European name differences
    "Đorđe Petrović": 12032,  # Djordje Petrovic

    # Collision fixes (players who share name parts)
    "Jonny Evans": 807,  # Jonny Evans (not Jonny from Wolves)
    "Kyle Walker-Peters": 885,  # Kyle Walker-Peters (not Kyle Walker)
    "Kevin Danso": 5261,  # Kevin Danso
    "Emerson Palmieri dos Santos": 7430,  # Emerson (West Ham)
    "Emerson Leite de Souza Junior": 7430,  # Emerson Royal? - needs verification
}

# Common name parts that shouldn't be matched alone (too many collisions)
COMMON_NAME_PARTS = {
    'gabriel', 'andre', 'rodrigo', 'lucas', 'pedro', 'bruno', 'matheus',
    'silva', 'santos', 'oliveira', 'souza', 'lima', 'costa', 'ferreira',
    'dos', 'da', 'de', 'do', 'neto', 'junior',
    'emerson', 'kevin', 'kyle', 'van', 'der', 'den',
    'mohamed', 'mohammed', 'jose', 'carlos', 'antonio',
}

def create_name_variants(name: str, include_single_names: bool = True, conservative: bool = False) -> list:
    """Create multiple name variants for fuzzy matching.

    Variants are ordered from most specific to least specific.
    Single-name variants are only included if include_single_names=True.
    If conservative=True, skip variants that use common/ambiguous name parts.
    """
    normalized = normalize_name(name)
    parts = normalized.split('_')

    variants = [normalized]

    # Full reversal (for Asian name ordering)
    if len(parts) >= 2:
        variants.append('_'.join(reversed(parts)))

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
            if parts[-2] not in COMMON_NAME_PARTS and parts[-1] not in COMMON_NAME_PARTS:
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
        parts = name_with_id.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            name_part = parts[0]
            player_id = int(parts[1])
        else:
            name_part = name_with_id
            player_id = None

        # Read the CSV
        df = pd.read_csv(csv_file)
        df['understat_player_name'] = name_part.replace('_', ' ')
        df['understat_player_id'] = player_id
        df['normalized_name'] = normalize_name(name_part)
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)

def process_season(season: str, data_base_dir: Path, output_dir: Path) -> Optional[pd.DataFrame]:
    """Process a single season and return the merged DataFrame."""
    print(f"\n{'='*50}")
    print(f"Processing season: {season}")
    print(f"{'='*50}")

    base_dir = data_base_dir / season
    merged_gw_path = base_dir / "gws" / "merged_gw.csv"
    understat_dir = base_dir / "understat"

    if not merged_gw_path.exists():
        print(f"  Skipping: {merged_gw_path} not found")
        return None
    if not understat_dir.exists():
        print(f"  Skipping: {understat_dir} not found")
        return None

    # Load FPL data
    print("Loading FPL data...")
    fpl_df = pd.read_csv(merged_gw_path)
    fpl_df['match_date'] = pd.to_datetime(fpl_df['kickoff_time']).dt.date.astype(str)
    fpl_df['season'] = season

    print(f"FPL data: {len(fpl_df)} rows")

    # Load understat data
    print("Loading understat data...")
    understat_df = load_understat_data(understat_dir)
    if len(understat_df) == 0:
        print(f"  Skipping: No understat data found")
        return None
    understat_df['match_date'] = pd.to_datetime(understat_df['date'], format='mixed').dt.date.astype(str)

    print(f"Understat data: {len(understat_df)} rows")

    # Build understat lookup - two lookups:
    # 1. Primary: exact normalized name (no collisions)
    # 2. Secondary: name variants (may have collisions, used as fallback)
    understat_primary = {}  # (normalized_name, date) -> row
    understat_variants = {}  # (variant, date) -> row
    single_name_players = set()  # Players known by single name in understat

    for _, row in understat_df.iterrows():
        name = row['understat_player_name']
        normalized = normalize_name(name)
        parts = normalized.split('_')

        # Track if this is a single-name player (e.g., "Alisson", "Joelinton")
        if len(parts) == 1:
            single_name_players.add(normalized)

        # Primary lookup: exact normalized name
        key = (normalized, row['match_date'])
        if key not in understat_primary:
            understat_primary[key] = row

        # Secondary lookup: all variants (for fallback matching)
        name_variants = create_name_variants(name, include_single_names=True)
        for variant in name_variants:
            key = (variant, row['match_date'])
            if key not in understat_variants:
                understat_variants[key] = row

    # Rename understat columns to avoid conflicts (prefix with 'us_')
    # Keep understat_player_id as 'player_id' (without prefix) since it's the consistent cross-season ID
    understat_cols = [c for c in understat_df.columns if c not in ['normalized_name', 'match_date', 'understat_player_name', 'understat_player_id']]

    print(f"Found {len(single_name_players)} single-name players in understat")

    # Build a lookup from understat player ID to rows for that player
    understat_by_id = {}
    for _, row in understat_df.iterrows():
        player_id = row['understat_player_id']
        if player_id is not None:
            date = row['match_date']
            if player_id not in understat_by_id:
                understat_by_id[player_id] = {}
            understat_by_id[player_id][date] = row

    # Match FPL rows to understat data
    # Strategy: Manual mapping first, then exact match, then variants, then single-name
    print("Matching data...")
    manual_matches = 0
    matches = []
    for idx, fpl_row in fpl_df.iterrows():
        match_date = fpl_row['match_date']
        fpl_name = fpl_row['name']
        matched_row = None

        # 0. Check manual mapping first (highest priority)
        has_manual_mapping = fpl_name in MANUAL_NAME_TO_ID
        if has_manual_mapping:
            player_id = MANUAL_NAME_TO_ID[fpl_name]
            if player_id in understat_by_id and match_date in understat_by_id[player_id]:
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
            name_variants = create_name_variants(fpl_name, include_single_names=False, conservative=False)
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
            fpl_parts = fpl_normalized.split('_')
            # Only use single-name matching if FPL name is also short (1-2 parts)
            if len(fpl_parts) <= 2:
                for part in fpl_parts:
                    if part in single_name_players:
                        key = (part, match_date)
                        if key in understat_primary:
                            matched_row = understat_primary[key]
                            break

        if matched_row is not None:
            match_data = {f'us_{c}': matched_row[c] for c in understat_cols}
            match_data['player_id'] = matched_row['understat_player_id']
        else:
            match_data = {f'us_{c}': None for c in understat_cols}
            match_data['player_id'] = None

        matches.append(match_data)

    # Add understat columns to FPL data
    understat_matched_df = pd.DataFrame(matches)
    merged_df = pd.concat([fpl_df.reset_index(drop=True), understat_matched_df], axis=1)

    # Check match statistics
    matched = merged_df['us_xG'].notna().sum()
    total = len(merged_df)
    print(f"Matched {matched}/{total} rows ({100*matched/total:.1f}%) [{manual_matches} via manual mapping]")

    # Check match rate for players who actually played
    played = merged_df[merged_df['minutes'] > 0]
    if len(played) > 0:
        played_matched = played['us_xG'].notna().sum()
        played_total = len(played)
        print(f"Matched (players with minutes>0): {played_matched}/{played_total} ({100*played_matched/played_total:.1f}%)")

        # Show some unmatched examples (only for players who played)
        unmatched = merged_df[(merged_df['us_xG'].isna()) & (merged_df['minutes'] > 0)][['name', 'match_date', 'team']].drop_duplicates()
        if len(unmatched) > 0:
            print(f"\nSample unmatched players who played ({min(5, len(unmatched))} of {len(unmatched)}):")
            print(unmatched.head(5).to_string(index=False))

    # Drop helper columns
    merged_df = merged_df.drop(columns=['match_date'])

    # Save individual season file
    output_path = output_dir / f"merged_fpl_understat_{season}.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    return merged_df


def main():
    data_base_dir = Path("/Users/stephenheron/Workspace/f/fpl-prediction/Fantasy-Premier-League/data")
    output_dir = Path("/Users/stephenheron/Workspace/f/fpl-prediction")

    seasons = ["2023-24", "2024-25", "2025-26"]

    all_dfs = []
    for season in seasons:
        df = process_season(season, data_base_dir, output_dir)
        if df is not None:
            all_dfs.append(df)

    # Combine all seasons into one file
    if all_dfs:
        print(f"\n{'='*50}")
        print("Combining all seasons...")
        print(f"{'='*50}")
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_path = output_dir / "merged_fpl_understat_all.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"Combined {len(all_dfs)} seasons: {len(combined_df)} total rows")
        print(f"Saved to {combined_path}")

if __name__ == "__main__":
    main()
