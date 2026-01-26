#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <gw>"
  exit 1
fi

gw=$1

python -m fpl_prediction.cli.predict --position all --model all --gw "$gw" --combine
python -m fpl_prediction.cli.optimize --gw "$gw" --output-json reports/output/optimal_squad.json
