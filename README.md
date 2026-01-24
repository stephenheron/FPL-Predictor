# FPL Prediction

Predict Fantasy Premier League points by position using rolling form features,
fixture strength, and sequence models. The repo trains and scores XGBoost and
LSTM models for all positions (FWD, MID, DEF, GK), plus an optional blend.

## What is in here

- `join_data.py`: merges FPL gameweek data with Understat match stats by player.
- `train_*_xgb.py`: train XGBoost regressors per position.
- `predict_*.py`: score a season CSV and (optionally) build future fixtures for a
  target GW.
- `train_*_lstm.py`: train LSTM regressors per position.
- `predict_*_lstm.py`: score a season CSV with LSTM models.
- `combine_*_predictions.py`: blend XGBoost + LSTM outputs for each position.

## Setup

This project uses Python 3.9+.

Install dependencies (including model libraries):

```bash
pip install -e .
pip install xgboost scikit-learn torch
```

## Data prep (optional)

`join_data.py` expects FPL data + Understat data in the
`Fantasy-Premier-League/data/<season>/` folders. It outputs
`merged_fpl_understat_<season>.csv` files in the repo root.

```bash
python join_data.py
```

## Train models

XGBoost forward model (default uses 2022-23 to 2024-25, holds out 2025-26):

```bash
python train_fwd_xgb.py
```

XGBoost midfielder model:

```bash
python train_mid_xgb.py
```

Outputs:

- `xgb_fwd_model.json`, `xgb_mid_model.json`
- `feature_importance_fwd.csv`, `feature_importance_mid.csv`

LSTM models (default uses 2022-23 to 2024-25, holds out 2025-26):

```bash
python train_mid_lstm.py
python train_fwd_lstm.py
python train_def_lstm.py
python train_gk_lstm.py
```

Use `--train-full` to include the holdout season and skip holdout evaluation:

```bash
python train_mid_lstm.py --train-full
```

Outputs per position:

- `lstm_<pos>_model.pt`, `lstm_<pos>_scaler.pkl`
- `lstm_<pos>_training_report.csv`

## Run predictions

Forward predictions for a specific GW (builds future fixture rows if needed):

```bash
python predict_fwd.py --predict-gw 23
```

Midfielder predictions:

```bash
python predict_mid.py --predict-gw 23
```

Outputs default to `fwd_predictions.csv` / `mid_predictions.csv` and include
`predicted_points` plus fixture context.

LSTM predictions (example for GW 23):

```bash
python predict_mid_lstm.py --predict-gw 23
python predict_fwd_lstm.py --predict-gw 23
python predict_def_lstm.py --predict-gw 23
python predict_gk_lstm.py --predict-gw 23
```

LSTM outputs default to `*_predictions_lstm.csv` and include
`predicted_points` plus the availability multiplier.

## Combine model outputs

Blend XGBoost + LSTM predictions with an optional weight (default 0.5):

```bash
python combine_mid_predictions.py --xgb-file mid_predictions_gw23.csv --weight-xgb 0.6
python combine_fwd_predictions.py --xgb-file fwd_predictions_gw23.csv
python combine_def_predictions.py --xgb-file def_predictions.csv
python combine_gk_predictions.py --xgb-file gk_predictions.csv
```

Combined outputs default to `*_predictions_combined.csv` with
`predicted_points_xgb`, `predicted_points_lstm`, and `combined_points`.

## Useful flags

- `--input-file`: season CSV to score (default `merged_fpl_understat_2025-26.csv`)
- `--model-file`: path to a saved model
- `--roll-windows`: rolling windows used in training (default `3 5 8`)
- `--predict-gw`: score a single future GW (adds fixtures from FPL data)
- `--fixtures-file`, `--teams-file`: fixture + team strength data
- `--train-full`: train on all seasons (skip holdout evaluation)

## Notes

- Feature engineering uses rolling historical averages and per-90 rates for key
  FPL and Understat stats, plus opponent strength + home/away indicators.
- LSTM models use 5-game sequences with roll-8 hint features and per-90 rates.
- Availability multipliers are based on minutes in the last 3 GWs when scoring
  a future GW.
- Training uses walk-forward validation per season to choose the best training
  window (default candidates: 15/25/35 GWs).
