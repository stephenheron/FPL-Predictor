"""CLI for generating FPL predictions.

Usage:
    python -m fpl_prediction predict --position FWD --model lstm
    python -m fpl_prediction predict --position all --model all --gw 23
    fpl-predict --position MID --model xgboost --gw 24
"""

import argparse

from fpl_prediction.config.settings import POSITIONS
from fpl_prediction.models.ensemble import combine_position_predictions
from fpl_prediction.prediction.predictor import predict_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate FPL predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Predict with LSTM for forwards:
        fpl-predict --position FWD --model lstm

    Predict for specific gameweek:
        fpl-predict --position FWD --model lstm --gw 23

    Predict all positions with all models:
        fpl-predict --position all --model all --gw 23

    Predict and combine results:
        fpl-predict --position FWD --model all --gw 23 --combine
        """,
    )
    parser.add_argument(
        "--position",
        type=str,
        default="all",
        choices=["GK", "DEF", "MID", "FWD", "all"],
        help="Position to predict for (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["lstm", "xgboost", "all"],
        help="Model type to use (default: all)",
    )
    parser.add_argument(
        "--gw",
        type=int,
        default=None,
        help="Gameweek to predict for (optional).",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine LSTM and XGBoost predictions (requires --model all).",
    )
    parser.add_argument(
        "--weight-xgb",
        type=float,
        default=None,
        help="Weight for XGBoost in combined predictions (auto-loads learned weights if not specified).",
    )
    parser.add_argument(
        "--normalize-scores",
        action="store_true",
        help="Normalize model scores before combining (default: True).",
    )
    parser.add_argument(
        "--no-normalize-scores",
        action="store_false",
        dest="normalize_scores",
        help="Disable score normalization when combining.",
    )
    parser.set_defaults(normalize_scores=True)
    return parser.parse_args()


def main() -> None:
    """Main entry point for prediction CLI."""
    args = parse_args()

    positions = list(POSITIONS) if args.position == "all" else [args.position]
    model_types = ["lstm", "xgboost"] if args.model == "all" else [args.model]

    for position in positions:
        for model_type in model_types:
            print(f"\n{'=' * 60}")
            print(f"Predicting with {model_type.upper()} model for {position}")
            if args.gw:
                print(f"Gameweek: {args.gw}")
            print(f"{'=' * 60}\n")
            predict_model(position, model_type, predict_gw=args.gw)

        # Combine predictions if requested
        if args.combine and args.model == "all":
            print(f"\n{'=' * 60}")
            print(f"Combining predictions for {position}")
            print(f"{'=' * 60}\n")
            combine_position_predictions(
                position,
                weight_xgb=args.weight_xgb,
                normalize_scores=args.normalize_scores,
            )


if __name__ == "__main__":
    main()
