"""CLI for training meta-model ensemble weights.

Usage:
    fpl-train-meta --position FWD
    fpl-train-meta --position all --model lasso
    fpl-train-meta --position MID --regenerate-oof
"""

import argparse

from fpl_prediction.config.settings import POSITIONS
from fpl_prediction.training.meta_trainer import train_meta_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train meta-model to learn optimal ensemble weights.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Train Ridge meta-model for all positions:
        fpl-train-meta --position all

    Train Lasso meta-model for midfielders:
        fpl-train-meta --position MID --model lasso

    Regenerate OOF predictions:
        fpl-train-meta --position FWD --regenerate-oof

    Custom alpha values:
        fpl-train-meta --position all --alpha 0.01 0.1 1.0
        """,
    )
    parser.add_argument(
        "--position",
        type=str,
        default="all",
        choices=["GK", "DEF", "MID", "FWD", "all"],
        help="Position to train meta-model for (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ridge",
        choices=["ridge", "lasso"],
        help="Meta-model type (default: ridge)",
    )
    parser.add_argument(
        "--regenerate-oof",
        action="store_true",
        help="Force regeneration of out-of-fold predictions (slow)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=None,
        help="Alpha values to try (default: 0.001 0.01 0.1 1.0 10.0)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for meta-model training CLI."""
    args = parse_args()

    positions = list(POSITIONS) if args.position == "all" else [args.position]
    alpha_range = tuple(args.alpha) if args.alpha else None

    for position in positions:
        print(f"\n{'=' * 60}")
        print(f"Training {args.model.upper()} meta-model for {position}")
        print(f"{'=' * 60}\n")

        weights = train_meta_model(
            position,
            meta_model=args.model,
            regenerate_oof=args.regenerate_oof,
            alpha_range=alpha_range,
        )

        # Report results
        print(f"\nResults for {position}:")
        print(f"  Model: {weights.model_type} (alpha={weights.alpha:.4f})")
        print(f"  XGBoost weight: {weights.weight_xgb:.4f}")
        print(f"  LSTM weight: {weights.weight_lstm:.4f}")
        print(f"  Intercept: {weights.intercept:.4f}")
        print(f"  Dominant model: {weights.dominant_model} ({weights.trust_ratio:.2f}x)")
        print(f"  OOF MAE: {weights.mae:.4f}")
        print(f"  OOF RMSE: {weights.rmse:.4f}")


if __name__ == "__main__":
    main()
