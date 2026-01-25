"""Main entry point for fpl_prediction package.

Usage:
    python -m fpl_prediction train --position FWD --model lstm
    python -m fpl_prediction predict --position FWD --model lstm --gw 23
    python -m fpl_prediction prices --input mid_predictions_gw23.csv
    python -m fpl_prediction optimize --gw 23
"""

import sys


def main() -> None:
    """Main entry point for the CLI."""
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("""FPL Prediction CLI

Usage:
    python -m fpl_prediction <command> [options]

Commands:
    train     Train prediction models
    predict   Generate predictions
    prices    Attach player prices to predictions
    optimize  Build optimal squad using ILP

Examples:
    python -m fpl_prediction train --position FWD --model lstm
    python -m fpl_prediction predict --position all --model all --gw 23
    python -m fpl_prediction prices --input mid_predictions_gw23.csv
    python -m fpl_prediction optimize --gw 23

Run 'python -m fpl_prediction <command> --help' for command-specific help.
""")
        return

    command = sys.argv[1]
    # Remove the command from argv so subcommand parser sees the right args
    sys.argv = [f"fpl_prediction {command}"] + sys.argv[2:]

    if command == "train":
        from fpl_prediction.cli.train import main as train_main
        train_main()
    elif command == "predict":
        from fpl_prediction.cli.predict import main as predict_main
        predict_main()
    elif command == "prices":
        from fpl_prediction.cli.prices import main as prices_main
        prices_main()
    elif command == "optimize":
        from fpl_prediction.cli.optimize import main as optimize_main
        optimize_main()
    else:
        print(f"Unknown command: {command}")
        print("Valid commands: train, predict, prices, optimize")
        sys.exit(1)


if __name__ == "__main__":
    main()
