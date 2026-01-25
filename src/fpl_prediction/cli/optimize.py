"""CLI for building an optimal FPL squad using ILP."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fpl_prediction.optimization.ilp import SquadConstraints, build_optimal_squad, summarize_squad


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize FPL squad using ILP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    fpl-optimize --gw 23
    fpl-optimize --budget 101.5 --max-per-team 3
    fpl-optimize --output output/optimal_squad.csv
        """,
    )

    parser.add_argument("--gw", type=int, default=None, help="Gameweek to optimize (default: latest common).")
    parser.add_argument("--budget", type=float, default=100.0, help="Budget limit (default: 100.0).")
    parser.add_argument(
        "--max-per-team", type=int, default=3, help="Max players per team (default: 3)."
    )
    parser.add_argument(
        "--bench-weight",
        type=float,
        default=0.03,
        help="Bench weight relative to starters (default: 0.03).",
    )
    parser.add_argument(
        "--bench-max-cost",
        type=float,
        default=5.5,
        help="Maximum cost per bench player (default: 5.5).",
    )
    parser.add_argument(
        "--bench-gk-max-cost",
        type=float,
        default=5.0,
        help="Maximum cost for the bench goalkeeper (default: 5.0).",
    )
    parser.add_argument(
        "--conflict-penalty-weight",
        type=float,
        default=0.2,
        help=(
            "Penalty weight for starter attacker vs opponent defender conflicts "
            "as a fraction of their average points (default: 0.2)."
        ),
    )
    parser.add_argument(
        "--min-total-spend",
        type=float,
        default=95.0,
        help="Minimum total spend for the squad (default: 95.0).",
    )
    parser.add_argument(
        "--gk",
        type=Path,
        default=Path("gk_predictions_combined.csv"),
        help="Goalkeeper predictions CSV.",
    )
    parser.add_argument(
        "--def",
        dest="def_path",
        type=Path,
        default=Path("def_predictions_combined.csv"),
        help="Defender predictions CSV.",
    )
    parser.add_argument(
        "--mid",
        type=Path,
        default=Path("mid_predictions_combined.csv"),
        help="Midfielder predictions CSV.",
    )
    parser.add_argument(
        "--fwd",
        type=Path,
        default=Path("fwd_predictions_combined.csv"),
        help="Forward predictions CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path for selected squad.",
    )
    return parser.parse_args()


def _print_squad(selected: pd.DataFrame, summary: dict[str, float], gw: int | None) -> None:
    if gw is not None:
        print(f"Gameweek: {gw}")
    starters: pd.DataFrame = selected.loc[selected["is_starter"]].copy()
    bench: pd.DataFrame = selected.loc[~selected["is_starter"]].copy()

    print("\nStarting XI:")
    for position in ["GK", "DEF", "MID", "FWD"]:
        position_df: pd.DataFrame = starters.loc[starters["position"] == position].copy()
        if position_df.empty:
            continue
        print(f"\n{position}:")
        for _, row in position_df.iterrows():
            print(
                f"- {row['name']} ({row['team']}) | "
                f"Cost: {row['now_cost']:.1f} | Points: {row['predicted_points']:.2f}"
            )

    print("\nBench:")
    for position in ["GK", "DEF", "MID", "FWD"]:
        position_df: pd.DataFrame = bench.loc[bench["position"] == position].copy()
        if position_df.empty:
            continue
        print(f"\n{position}:")
        for _, row in position_df.iterrows():
            print(
                f"- {row['name']} ({row['team']}) | "
                f"Cost: {row['now_cost']:.1f} | Points: {row['predicted_points']:.2f}"
            )

    print(
        f"\nTotal cost: {summary['total_cost']:.1f} | "
        f"Total predicted points: {summary['total_points']:.2f}"
    )
    print(
        f"Starter points: {summary['starter_points']:.2f} | "
        f"Bench points: {summary['bench_points']:.2f}"
    )


def main() -> None:
    """Main entry point for squad optimization CLI."""
    args = parse_args()

    constraints = SquadConstraints(budget=args.budget, max_per_team=args.max_per_team)
    selected, resolved_gw = build_optimal_squad(
        gk_path=args.gk,
        def_path=args.def_path,
        mid_path=args.mid,
        fwd_path=args.fwd,
        gw=args.gw,
        constraints=constraints,
        bench_weight=args.bench_weight,
        bench_max_cost=args.bench_max_cost,
        bench_gk_max_cost=args.bench_gk_max_cost,
        min_total_spend=args.min_total_spend,
        conflict_penalty_weight=args.conflict_penalty_weight,
    )

    summary = summarize_squad(selected)
    _print_squad(selected, summary, resolved_gw)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(args.output, index=False)
        print(f"\nSaved squad to {args.output}")


if __name__ == "__main__":
    main()
