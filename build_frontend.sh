#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$REPO_ROOT/frontend"
DATA_SOURCE="$REPO_ROOT/reports/output/optimal_squad.json"
DATA_DEST="$FRONTEND_DIR/public/data/optimal_squad.json"

if [[ ! -f "$DATA_SOURCE" ]]; then
  echo "Data file not found: $DATA_SOURCE" >&2
  exit 1
fi

mkdir -p "$(dirname "$DATA_DEST")"
cp "$DATA_SOURCE" "$DATA_DEST"

cd "$FRONTEND_DIR"
npm install
npm run build

echo "Static site built in: $FRONTEND_DIR/dist"
