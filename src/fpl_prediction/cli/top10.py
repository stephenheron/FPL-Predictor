"""CLI for listing top predicted players by position."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_PATHS: dict[str, Path] = {
    "GK": Path("reports/predictions/gk_predictions_combined.csv"),
    "DEF": Path("reports/predictions/def_predictions_combined.csv"),
    "MID": Path("reports/predictions/mid_predictions_combined.csv"),
    "FWD": Path("reports/predictions/fwd_predictions_combined.csv"),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Show top predicted players by position.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    fpl-top10
    fpl-top10 --gw 23 --limit 15
    fpl-top10 --metric predicted_points_xgb
        """,
    )
    parser.add_argument("--gw", type=int, default=None, help="Gameweek to report.")
    parser.add_argument("--limit", type=int, default=10, help="Players per position.")
    parser.add_argument(
        "--metric",
        type=str,
        default="combined_points",
        help="Prediction column to rank by (default: combined_points).",
    )
    parser.add_argument("--gk", type=Path, default=DEFAULT_PATHS["GK"], help="GK CSV.")
    parser.add_argument("--def", dest="def_path", type=Path, default=DEFAULT_PATHS["DEF"])
    parser.add_argument("--mid", type=Path, default=DEFAULT_PATHS["MID"], help="MID CSV.")
    parser.add_argument("--fwd", type=Path, default=DEFAULT_PATHS["FWD"], help="FWD CSV.")
    return parser.parse_args()


def _resolve_common_gw(paths: list[Path]) -> int | None:
    gw_sets: list[set[int]] = []
    for path in paths:
        gws = (
            pd.read_csv(path, usecols=lambda col: col == "GW")
            .dropna()
            .astype({"GW": int})
            .loc[:, "GW"]
            .unique()
            .tolist()
        )
        if gws:
            gw_sets.append(set(gws))

    if not gw_sets:
        return None

    common = set.intersection(*gw_sets)
    if common:
        return max(common)
    return None


def _format_table(df: pd.DataFrame, metric: str) -> str:
    view = df.copy()
    view.insert(0, "rank", pd.Series(range(1, len(view) + 1), index=view.index))
    if metric in view.columns:
        view[metric] = view[metric].map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
    if metric == "combined_points":
        view = view.rename(columns={"combined_points": "points"})
    return view.to_string(index=False)


def _load_top(path: Path, position: str, gw: int | None, limit: int, metric: str) -> tuple[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file for {position}: {path}")

    df = pd.read_csv(path)
    if df.empty:
        return position, "No rows found."

    if metric not in df.columns:
        raise ValueError(f"Column '{metric}' not found in {path}")

    resolved_gw = gw
    if "GW" in df.columns:
        if resolved_gw is None:
            resolved_gw = int(df["GW"].max())
        df = df.loc[df["GW"] == resolved_gw]

    top = df.sort_values(metric, ascending=False).head(limit)
    top = top.loc[:, ["name", "team", metric]]
    title = f"Top {limit} {position}"
    if resolved_gw is not None:
        title = f"{title} (GW {resolved_gw})"
    return title, _format_table(top, metric)


def main() -> None:
    """Main entry point for the top-10 CLI."""
    args = parse_args()
    paths = {
        "GK": args.gk,
        "DEF": args.def_path,
        "MID": args.mid,
        "FWD": args.fwd,
    }

    common_gw = None
    if args.gw is None:
        common_gw = _resolve_common_gw(list(paths.values()))

    if args.gw is None and common_gw is None:
        print("No common GW found across inputs; using latest GW per position.")

    gw = args.gw if args.gw is not None else common_gw
    for position, path in paths.items():
        title, table = _load_top(path, position, gw, args.limit, args.metric)
        print(f"\n{title}")
        print(table)


if __name__ == "__main__":
    main()
