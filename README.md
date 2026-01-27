# FPL Prediction

Predict Fantasy Premier League points by position using rolling form features,
fixture strength, and sequence models. The package trains and scores XGBoost and
LSTM models for all positions (FWD, MID, DEF, GK), plus an optional ensemble blend.

## Installation

This project uses Python 3.11+.

```bash
pip install -e .
```

## Quick Start

```bash
# Get latest FPL data
cd Fantasy-Premier-League
python global_scraper.py
python understat.py

# Run the full pipeline for a gameweek
./full_pipeline.sh GW

# Top 10 by position
uv run fpl-top10
```

## Package Structure

```
src/fpl_prediction/
├── cli/           # Command-line interfaces
├── config/        # Centralized settings and feature definitions
├── data/          # Data loading, preprocessing, sequences
├── models/        # LSTM and ensemble model definitions
├── pipeline/      # Data joining pipeline (FPL + Understat)
├── prediction/    # Prediction and availability logic
└── training/      # Unified trainers for LSTM and XGBoost
```

## Data Preparation

The pipeline merges FPL gameweek data with Understat match stats:

```bash
python -c "from fpl_prediction.pipeline import run_pipeline; run_pipeline()"
```

This expects FPL + Understat data in `Fantasy-Premier-League/data/<season>/` folders
and outputs `data/raw/merged_fpl_understat_<season>.csv` files.

## Training Models

### Train a Single Model

```bash
# Train LSTM for forwards
python -m fpl_prediction train --position FWD --model lstm

# Train XGBoost for midfielders
python -m fpl_prediction train --position MID --model xgboost
```

### Train All Models

```bash
# Train both model types for all positions
python -m fpl_prediction train --position all --model all

# Train with full data (include holdout season, skip evaluation)
python -m fpl_prediction train --position all --model all --train-full
```

### Alternative CLI

```bash
fpl-train --position FWD --model lstm
fpl-train --position all --model xgboost
```

### Outputs

- XGBoost: `models/xgb/xgb_{pos}_model.json`, `reports/training/feature_importance_{pos}.csv`
- LSTM: `models/lstm/lstm_{pos}_model.pt`, `models/lstm/lstm_{pos}_scaler.pkl`,
  `reports/training/lstm_{pos}_training_report.csv`

## Generating Predictions

### Predict for a Specific Gameweek

```bash
# LSTM predictions for forwards, GW 23
python -m fpl_prediction predict --position FWD --model lstm --gw 23

# XGBoost predictions for all positions
python -m fpl_prediction predict --position all --model xgboost --gw 23
```

### Predict and Combine

```bash
# Generate both model predictions and combine them
python -m fpl_prediction predict --position FWD --model all --gw 23 --combine

# Adjust XGBoost weight in ensemble (default 0.5)
python -m fpl_prediction predict --position all --model all --gw 23 --combine --weight-xgb 0.6
```

### Alternative CLI

```bash
fpl-predict --position FWD --model lstm --gw 23
fpl-predict --position all --model all --gw 23 --combine
```

### Outputs

- XGBoost: `reports/predictions/{pos}_predictions.csv`
- LSTM: `reports/predictions/{pos}_predictions_lstm.csv`
- Combined: `reports/predictions/{pos}_predictions_combined.csv`

### Top 10 by Position

```bash
fpl-top10
fpl-top10 --gw 23 --limit 15
fpl-top10 --metric predicted_points_xgb
```

## Attaching Player Prices

Add FPL prices from the season-specific `Fantasy-Premier-League/data/<season>/cleaned_players.csv`
to a predictions CSV (updates in place by default):

```bash
python -m fpl_prediction prices --input reports/predictions/mid_predictions_gw23.csv
```

The tool reads the `season` column to pick the right players file.

Write to a new file instead:

```bash
python -m fpl_prediction prices \
  --input reports/predictions/mid_predictions_gw23.csv \
  --output reports/predictions/mid_predictions_gw23_with_prices.csv
```

Override the players file if needed:

```bash
python -m fpl_prediction prices \
  --input reports/predictions/mid_predictions_gw23.csv \
  --players Fantasy-Premier-League/data/2024-25/cleaned_players.csv
```

### Alternative CLI

```bash
fpl-prices --input reports/predictions/mid_predictions_gw23.csv
```

## Configuration

All hyperparameters are centralized in `src/fpl_prediction/config/`:

- `settings.py`: Model configs (LSTMConfig, XGBoostConfig, PredictionConfig)
- `features.py`: Feature column definitions (BASE_COLS, PER90_COLS, etc.)
- `player_mappings.py`: Manual FPL-to-Understat player ID mappings

### Key Settings

| Parameter | LSTM Default | XGBoost Default |
|-----------|--------------|-----------------|
| seq_len | 5 | - |
| roll_window | 8 | 3, 5, 8 |
| hidden_size | 64 | - |
| num_layers | 2 | - |
| dropout | 0.2 | - |
| batch_size | 128 | - |
| n_estimators | - | 300 |
| max_depth | - | 5 |

## Features

- **Rolling features**: Historical averages over configurable windows
- **Per-90 rates**: Normalized stats for key FPL and Understat metrics
- **Fixture context**: Home/away, dynamic opponent strength ratings
- **Availability**: Minutes-based multipliers for prediction adjustment
- **Walk-forward CV**: XGBoost training window selection per season

## Notes

- LSTM models use 5-game sequences with roll-8 hint features
- Availability multipliers: >=180min (1.0), >=90min (0.7), >0min (0.4), 0min (0.2)
- XGBoost uses walk-forward validation to select best training window (15/25/35 GWs)
- Dynamic opponent strengths are computed from xG performance history
