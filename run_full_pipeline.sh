#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <gw>"
  exit 1
fi

gw=$1

python -m fpl_prediction.pipeline.join_data
python -m fpl_prediction.cli.train --train-full
python -m fpl_prediction.cli.predict --position all --model all --gw "$gw" --combine
python -m fpl_prediction.cli.prices --input reports/predictions/gk_predictions_combined.csv
python -m fpl_prediction.cli.prices --input reports/predictions/def_predictions_combined.csv
python -m fpl_prediction.cli.prices --input reports/predictions/mid_predictions_combined.csv
python -m fpl_prediction.cli.prices --input reports/predictions/fwd_predictions_combined.csv
