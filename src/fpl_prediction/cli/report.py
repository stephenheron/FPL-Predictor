"""CLI for generating markdown reports from optimization output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


POSITION_ORDER = ["GK", "DEF", "MID", "FWD"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a markdown report for a gameweek.",
    )
    parser.add_argument("--gw", type=int, default=None, help="Gameweek label for the report.")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("reports/output/optimal_squad.json"),
        help="Path to optimal squad JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown path (defaults to predictions/<gw>.md).",
    )
    return parser.parse_args()


def _format_value(value: object, decimals: int | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if decimals is not None:
        try:
            return f"{float(value):.{decimals}f}"
        except (TypeError, ValueError):
            return ""
    return str(value)


def _render_table(df: pd.DataFrame, columns: list[tuple[str, str]]) -> list[str]:
    header = "| " + " | ".join([label for label, _ in columns]) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for _, row in df.iterrows():
        items = []
        for _, key in columns:
            items.append(_format_value(row.get(key)))
        lines.append("| " + " | ".join(items) + " |")
    return lines


def _render_section(title: str, df: pd.DataFrame) -> list[str]:
    if df.empty:
        return [f"## {title}", "", "No players found."]

    lines = [f"## {title}"]
    for position in POSITION_ORDER:
        position_df = df.loc[df["position"] == position].copy()
        if position_df.empty:
            continue
        position_df = position_df.sort_values("predicted_points", ascending=False)
        opponent_series = (
            position_df["opponent_name"]
            if "opponent_name" in position_df.columns
            else pd.Series([""] * len(position_df), index=position_df.index)
        )
        position_df = position_df.assign(
            now_cost=position_df["now_cost"].map(lambda value: _format_value(value, 1)),
            predicted_points=position_df["predicted_points"].map(
                lambda value: _format_value(value, 2)
            ),
            opponent_name=opponent_series.fillna(""),
        )
        lines.append("")
        lines.append(f"### {position}")
        columns = [
            ("Player", "name"),
            ("Team", "team"),
            ("Opponent", "opponent_name"),
            ("Cost", "now_cost"),
            ("Predicted", "predicted_points"),
        ]
        lines.extend(_render_table(position_df, columns))
    return lines


def main() -> None:
    """Main entry point for markdown report generation."""
    args = parse_args()

    if args.output is None and args.gw is None:
        raise ValueError("Provide --gw or --output.")

    if not args.input_json.exists():
        raise FileNotFoundError(f"Missing input JSON: {args.input_json}")

    with args.input_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    summary = payload.get("summary", {})
    players = payload.get("players", [])
    df = pd.DataFrame(players)

    if df.empty:
        raise ValueError("No players found in optimal squad JSON.")

    output_path = args.output
    if output_path is None:
        output_path = Path("predictions") / f"{args.gw}.md"

    starters = df.loc[df["is_starter"]].copy()
    bench = df.loc[~df["is_starter"]].copy()

    lines = [f"# FPL Predictions - GW {args.gw}" if args.gw is not None else "# FPL Predictions"]
    lines.append("")
    lines.append("## Summary")
    summary_rows = pd.DataFrame(
        [
            {
                "Metric": "Total cost",
                "Value": _format_value(summary.get("total_cost"), 1),
            },
            {
                "Metric": "Total predicted points",
                "Value": _format_value(summary.get("total_points"), 2),
            },
            {
                "Metric": "Starter points",
                "Value": _format_value(summary.get("starter_points"), 2),
            },
            {
                "Metric": "Bench points",
                "Value": _format_value(summary.get("bench_points"), 2),
            },
        ]
    )
    lines.extend(_render_table(summary_rows, [("Metric", "Metric"), ("Value", "Value")]))
    lines.append("")
    lines.extend(_render_section("Starting XI", starters))
    lines.append("")
    lines.extend(_render_section("Bench", bench))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved markdown report to {output_path}")


if __name__ == "__main__":
    main()
