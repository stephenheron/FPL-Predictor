"""Player name normalization and matching utilities."""

import re
import unicodedata

from fpl_prediction.config.player_mappings import COMMON_NAME_PARTS


# Turkish and special character mappings that don't decompose with NFD
SPECIAL_CHAR_MAP: dict[str, str] = {
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


def normalize_name(name: str) -> str:
    """Normalize a player name for matching.

    Converts to lowercase, removes accents, replaces spaces/hyphens
    with underscores, and removes special characters.

    Args:
        name: Player name to normalize.

    Returns:
        Normalized name for matching.
    """
    # Handle special characters that don't decompose with NFD
    for char, replacement in SPECIAL_CHAR_MAP.items():
        name = name.replace(char, replacement)

    # Remove accents using NFD normalization
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")

    # Replace spaces and hyphens with underscores
    name = re.sub(r"[\s-]+", "_", name)

    # Remove any other special characters
    name = re.sub(r"[^\w]", "", name)

    return name.lower()


def create_name_variants(
    name: str, include_single_names: bool = True, conservative: bool = False
) -> list[str]:
    """Create multiple name variants for fuzzy matching.

    Variants are ordered from most specific to least specific.
    Single-name variants are only included if include_single_names=True.
    If conservative=True, skip variants that use common/ambiguous name parts.

    Args:
        name: Player name to generate variants for.
        include_single_names: Include single-name variants (risky).
        conservative: Skip common name parts to avoid collisions.

    Returns:
        List of name variants ordered by specificity.
    """
    normalized = normalize_name(name)
    parts = normalized.split("_")

    variants = [normalized]

    # Full reversal (for Asian name ordering)
    if len(parts) >= 2:
        variants.append("_".join(reversed(parts)))

    if len(parts) >= 2:
        first_last = f"{parts[0]}_{parts[-1]}"
        first_second = f"{parts[0]}_{parts[1]}"

        if conservative:
            if parts[0] not in COMMON_NAME_PARTS and parts[-1] not in COMMON_NAME_PARTS:
                variants.append(first_last)
            if parts[0] not in COMMON_NAME_PARTS and parts[1] not in COMMON_NAME_PARTS:
                variants.append(first_second)
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

    # Single-name variants (risky - only use for understat indexing)
    if include_single_names:
        variants.append(parts[0])  # First name
        if len(parts) >= 2:
            variants.append(parts[-1])  # Last name
        if len(parts) >= 3:
            for part in parts[1:-1]:  # Middle names
                if part not in variants:
                    variants.append(part)

    return variants
