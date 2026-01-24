# FPL Prediction

Predict Fantasy Premier League points by position using rolling form features and
fixture strength. The repo currently trains and scores separate XGBoost models
for forwards (FWD) and midfielders (MID).

## What is in here

- `join_data.py`: merges FPL gameweek data with Understat match stats by player.
- `train_fwd_xgb.py` / `train_mid_xgb.py`: train XGBoost regressors for FWD/MID.
- `predict_fwd.py` / `predict_mid.py`: score a season CSV and (optionally) build
  future fixtures for a target GW.

## Setup

This project uses Python 3.9+.

Install dependencies (including model libraries):

```bash
pip install -e .
pip install xgboost scikit-learn
```

## Data prep (optional)

`join_data.py` expects FPL data + Understat data in the
`Fantasy-Premier-League/data/<season>/` folders. It outputs
`merged_fpl_understat_<season>.csv` files in the repo root.

```bash
python join_data.py
```

## Train models

Forward model (default uses 2022-23 to 2024-25, holds out 2025-26):

```bash
python train_fwd_xgb.py
```

Midfielder model:

```bash
python train_mid_xgb.py
```

Outputs:

- `xgb_fwd_model.json`, `xgb_mid_model.json`
- `feature_importance_fwd.csv`, `feature_importance_mid.csv`

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

## Useful flags

- `--input-file`: season CSV to score (default `merged_fpl_understat_2025-26.csv`)
- `--model-file`: path to a saved model
- `--roll-windows`: rolling windows used in training (default `3 5 8`)
- `--predict-gw`: score a single future GW (adds fixtures from FPL data)
- `--fixtures-file`, `--teams-file`: fixture + team strength data

## Notes

- Feature engineering uses rolling historical averages and per-90 rates for key
  FPL and Understat stats, plus opponent strength + home/away indicators.
- Training uses walk-forward validation per season to choose the best training
  window (default candidates: 15/25/35 GWs).
