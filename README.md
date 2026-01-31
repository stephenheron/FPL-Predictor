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

## Learning Ensemble Weights

Instead of using fixed 50/50 weights for combining XGBoost and LSTM predictions, you can
train a meta-model to learn optimal weights from historical data using Ridge or Lasso regression.

### Train Meta-Model

```bash
# Train Ridge meta-model for all positions (first run is slow - generates OOF predictions)
python -m fpl_prediction train-meta --position all --model ridge

# Train for a specific position
python -m fpl_prediction train-meta --position MID --model ridge

# Use Lasso to see if one model should be dropped entirely
python -m fpl_prediction train-meta --position FWD --model lasso
```

### How It Works

The meta-model uses **leave-one-season-out cross-validation** to generate out-of-fold predictions:

1. For each season, train base models (XGBoost + LSTM) on all other seasons
2. Generate predictions for the held-out season
3. Train Ridge/Lasso regression on these OOF predictions to learn optimal weights
4. Save weights to `models/meta_weights_{pos}.json`

This ensures the meta-model doesn't overfit by only seeing predictions the base models
made on data they weren't trained on.

### Learned Weights

After training, the weights are automatically used when combining predictions:

```bash
# Predictions now auto-load learned weights
fpl-predict --position MID --model all --gw 24 --combine
# Output: "Using learned meta-weights for MID: XGB=0.851"
```

Current learned weights:

| Position | XGB Weight | LSTM Weight | Dominant Model |
|----------|------------|-------------|----------------|
| GK | 0.58 | 0.66 | LSTM (1.1x) |
| DEF | 0.79 | 0.19 | XGB (4.1x) |
| MID | 0.85 | 0.20 | XGB (4.3x) |
| FWD | 0.77 | 0.23 | XGB (3.3x) |

**Key insight**: XGBoost dominates for outfield players, but LSTM is slightly preferred for goalkeepers.

### Outputs

- Weights: `models/meta_weights_{pos}.json`
- OOF predictions (cached): `models/meta_oof_{pos}.csv`

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
# Generate both model predictions and combine them (auto-loads learned weights if available)
python -m fpl_prediction predict --position FWD --model all --gw 23 --combine

# Override with manual XGBoost weight
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

- `settings.py`: Model configs (LSTMConfig, XGBoostConfig, MetaConfig, PredictionConfig)
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
