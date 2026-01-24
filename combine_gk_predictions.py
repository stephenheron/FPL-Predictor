import argparse

from combine_predictions import combine_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine GK LSTM and XGBoost predictions.")
    parser.add_argument("--xgb-file", default="gk_predictions.csv")
    parser.add_argument("--lstm-file", default="gk_predictions_lstm.csv")
    parser.add_argument("--output-file", default="gk_predictions_combined.csv")
    parser.add_argument(
        "--weight-xgb",
        type=float,
        default=0.5,
        help="Weight for XGBoost predictions (0-1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combine_predictions(args.xgb_file, args.lstm_file, args.output_file, args.weight_xgb)
    print(f"Saved combined predictions to: {args.output_file}")


if __name__ == "__main__":
    main()
