"""CLI for training FPL prediction models.

Usage:
    python -m fpl_prediction train --position FWD --model lstm
    python -m fpl_prediction train --position all --model all
    fpl-train --position MID --model xgboost
"""

import argparse

from fpl_prediction.config.settings import POSITIONS
from fpl_prediction.training.trainer import train_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train FPL prediction models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Train LSTM for forwards:
        fpl-train --position FWD --model lstm

    Train XGBoost for all positions:
        fpl-train --position all --model xgboost

    Train all models for all positions:
        fpl-train --position all --model all

    Train with full data (no holdout):
        fpl-train --position FWD --model lstm --train-full
        """,
    )
    parser.add_argument(
        "--position",
        type=str,
        default="all",
        choices=["GK", "DEF", "MID", "FWD", "all"],
        help="Position to train model for (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["lstm", "xgboost", "all"],
        help="Model type to train (default: all)",
    )
    parser.add_argument(
        "--train-full",
        action="store_true",
        help="Train on all seasons including holdout (skip holdout eval).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for training CLI."""
    args = parse_args()

    positions = list(POSITIONS) if args.position == "all" else [args.position]
    model_types = ["lstm", "xgboost"] if args.model == "all" else [args.model]

    for position in positions:
        for model_type in model_types:
            print(f"\n{'=' * 60}")
            print(f"Training {model_type.upper()} model for {position}")
            print(f"{'=' * 60}\n")
            train_model(position, model_type, train_full=args.train_full)


if __name__ == "__main__":
    main()
