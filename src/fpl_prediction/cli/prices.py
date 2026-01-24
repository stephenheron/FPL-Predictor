"""CLI for attaching player prices to predictions."""

from __future__ import annotations

import argparse

from fpl_prediction.prediction.prices import (
    PriceJoinConfig,
    add_prices_to_predictions,
    resolve_players_file,
    summarize_price_join,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Attach player prices to prediction outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Add prices in place:
        python -m fpl_prediction prices --input mid_predictions_gw23.csv

    Write to a new file:
        python -m fpl_prediction prices \
            --input mid_predictions_gw23.csv \
            --output mid_predictions_gw23_with_prices.csv
""",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Predictions CSV to update.",
    )
    parser.add_argument(
        "--players",
        default=None,
        help="Players CSV with prices (defaults to season-based path).",
    )
    parser.add_argument(
        "--season-col",
        default="season",
        help="Season column in predictions for auto path (default: season).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (defaults to --input).",
    )
    parser.add_argument(
        "--pred-name-col",
        default="name",
        help="Column in predictions for player name (default: name).",
    )
    parser.add_argument(
        "--player-first-col",
        default="first_name",
        help="Players first-name column (default: first_name).",
    )
    parser.add_argument(
        "--player-last-col",
        default="second_name",
        help="Players last-name column (default: second_name).",
    )
    parser.add_argument(
        "--price-col",
        default="now_cost",
        help="Price column to attach (default: now_cost).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for price-attachment CLI."""
    args = parse_args()
    output_file = args.output or args.input
    players_file = resolve_players_file(args.input, args.players, args.season_col)

    config = PriceJoinConfig(
        input_file=args.input,
        players_file=players_file,
        output_file=output_file,
        pred_name_col=args.pred_name_col,
        player_first_col=args.player_first_col,
        player_last_col=args.player_last_col,
        price_col=args.price_col,
    )

    merged = add_prices_to_predictions(config)
    rows, missing = summarize_price_join(merged, args.price_col)

    print(f"Saved prices to: {output_file}")
    print(f"Rows: {rows}")
    print(f"Missing prices: {missing}")


if __name__ == "__main__":
    main()
