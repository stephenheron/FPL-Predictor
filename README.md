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

Train all models for all positions:

```bash
python -m fpl_prediction train --position all --model all
```

Generate predictions for gameweek 23:

```bash
python -m fpl_prediction predict --position all --model all --gw 23 --combine
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
and outputs `merged_fpl_understat_<season>.csv` files.

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

- XGBoost: `xgb_{pos}_model.json`, `feature_importance_{pos}.csv`
- LSTM: `lstm_{pos}_model.pt`, `lstm_{pos}_scaler.pkl`, `lstm_{pos}_training_report.csv`

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

- XGBoost: `{pos}_predictions.csv`
- LSTM: `{pos}_predictions_lstm.csv`
- Combined: `{pos}_predictions_combined.csv`

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
