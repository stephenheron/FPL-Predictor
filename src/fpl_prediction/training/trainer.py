"""Unified training logic for LSTM and XGBoost models."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from xgboost import XGBRegressor

from fpl_prediction.config.settings import LSTMConfig, XGBoostConfig
from fpl_prediction.data import (
    build_features_lstm,
    build_features_xgboost,
    build_sequences,
    get_feature_columns_lstm,
    get_feature_columns_xgboost,
    load_season,
    scale_sequences,
)
from fpl_prediction.models import LSTMRegressor, SequenceDataset, get_device
from fpl_prediction.training.evaluation import evaluate, evaluate_cv_xgboost, run_epoch


class LSTMTrainer:
    """Unified LSTM trainer for all positions.

    Args:
        config: LSTMConfig with training parameters.
    """

    def __init__(self, config: LSTMConfig) -> None:
        self.config = config
        self.scaler = StandardScaler()
        self.model: LSTMRegressor | None = None
        self.feature_cols: list[str] = []

    def train(self) -> None:
        """Train the LSTM model."""
        config = self.config
        torch.manual_seed(config.random_state)
        np.random.seed(config.random_state)

        # Load and prepare data
        train_frames = [load_season(path) for path in config.train_files]
        holdout_df = load_season(config.holdout_file)
        train_df = pd.concat(train_frames, ignore_index=True)

        if config.train_full:
            train_df = pd.concat([train_df, holdout_df], ignore_index=True)

        train_df = build_features_lstm(train_df, config.position, config.roll_window)
        holdout_df = build_features_lstm(
            holdout_df, config.position, config.roll_window
        )

        self.feature_cols = get_feature_columns_lstm(train_df, config.roll_window)

        # Determine validation season
        seasons = sorted(train_df["season"].unique())
        train_seasons = [
            str(frame["season"].astype(str).iloc[0])
            for frame in train_frames
            if not frame.empty
        ]

        if config.val_season:
            val_season = config.val_season
        elif config.train_full:
            val_season = train_seasons[-1] if train_seasons else None
        else:
            val_season = seasons[-1] if seasons else None

        if val_season is None:
            raise ValueError("No seasons available for training.")

        # Split data
        train_split = pd.DataFrame(train_df[train_df["season"] != val_season])
        val_split = pd.DataFrame(train_df[train_df["season"] == val_season])

        if train_split.empty or val_split.empty:
            raise ValueError(
                f"Validation split failed. Check val season {val_season} in train data."
            )

        # Build sequences
        X_train, y_train, _ = build_sequences(
            train_split, self.feature_cols, config.seq_len
        )
        X_val, y_val, _ = build_sequences(val_split, self.feature_cols, config.seq_len)
        X_holdout, y_holdout, _ = build_sequences(
            holdout_df, self.feature_cols, config.seq_len
        )

        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError("Not enough sequences to train the model.")

        # Scale sequences
        X_train = scale_sequences(self.scaler, X_train, fit=True)
        X_val = scale_sequences(self.scaler, X_val)
        if len(X_holdout):
            X_holdout = scale_sequences(self.scaler, X_holdout)

        # Create data loaders
        train_ds = SequenceDataset(X_train, y_train)
        val_ds = SequenceDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size)

        # Initialize model
        device = get_device()
        self.model = LSTMRegressor(
            input_size=X_train.shape[-1],
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        loss_fn = nn.MSELoss()

        # Training loop with early stopping
        best_mae = float("inf")
        best_state = None
        patience_counter = 0
        report_rows = []

        for epoch in range(1, config.epochs + 1):
            train_loss = run_epoch(
                self.model, train_loader, optimizer, loss_fn, device
            )
            val_mae, val_rmse = evaluate(self.model, val_loader, device)
            report_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_mae": val_mae,
                    "val_rmse": val_rmse,
                }
            )

            if val_mae < best_mae:
                best_mae = val_mae
                best_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    break

        # Save training report
        report_df = pd.DataFrame(report_rows)
        report_df.to_csv(config.report_out, index=False)
        print(f"Saved training report to: {config.report_out}")

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Save model
        model_payload = {
            "model_state": self.model.state_dict(),
            "input_size": X_train.shape[-1],
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "seq_len": config.seq_len,
            "roll_window": config.roll_window,
            "feature_cols": self.feature_cols,
        }
        torch.save(model_payload, config.model_out)
        print(f"Saved model to: {config.model_out}")

        # Save scaler
        with Path(config.scaler_out).open("wb") as handle:
            pickle.dump(self.scaler, handle)
        print(f"Saved scaler to: {config.scaler_out}")

        # Holdout evaluation
        if config.train_full:
            print("Holdout evaluation skipped (train-full enabled).")
        elif len(X_holdout):
            holdout_ds = SequenceDataset(X_holdout, y_holdout)
            holdout_loader = DataLoader(holdout_ds, batch_size=config.batch_size)
            holdout_mae, holdout_rmse = evaluate(self.model, holdout_loader, device)
            print("Holdout evaluation (2025-26):")
            print(f"MAE: {holdout_mae:.4f}")
            print(f"RMSE: {holdout_rmse:.4f}")
        else:
            print("Holdout evaluation skipped (no sequences available).")


class XGBoostTrainer:
    """Unified XGBoost trainer for all positions.

    Args:
        config: XGBoostConfig with training parameters.
    """

    def __init__(self, config: XGBoostConfig) -> None:
        self.config = config
        self.model: XGBRegressor | None = None
        self.feature_cols: list[str] = []

    def train(self) -> None:
        """Train the XGBoost model."""
        config = self.config

        # Load and prepare data
        train_frames = [load_season(path) for path in config.train_files]
        holdout_df = load_season(config.holdout_file)
        train_df = pd.concat(train_frames, ignore_index=True)

        if config.train_full:
            train_df = pd.concat([train_df, holdout_df], ignore_index=True)

        train_df = build_features_xgboost(
            train_df, config.position, config.roll_windows
        )
        holdout_df = build_features_xgboost(
            holdout_df, config.position, config.roll_windows
        )

        self.feature_cols = get_feature_columns_xgboost(
            train_df, config.roll_windows, position=config.position
        )

        # Determine best training window
        best_window = None
        config_path = Path(config.fixed_window_config)
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                window_map = json.load(handle)
            best_window = window_map.get(config.position)
            if best_window is not None:
                print(
                    f"Using fixed window for {config.position} "
                    f"from {config.fixed_window_config}: {best_window}"
                )

        if best_window is None:
            cv_results = evaluate_cv_xgboost(
                train_df,
                self.feature_cols,
                config.train_windows,
                config.random_state,
            )
            cv_results = cv_results.sort_values("mae")
            best_window = int(cv_results.iloc[0]["train_window"])
            print("CV results (sorted by MAE):")
            print(cv_results.to_string(index=False))
            print(f"Best window: {best_window}")

        # Train final model on best window
        final_train = self._restrict_to_window(train_df, best_window)
        self.model = XGBRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.xgb_learning_rate,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            objective="reg:squarederror",
            random_state=config.random_state,
            n_jobs=-1,
        )
        self.model.fit(final_train[self.feature_cols], final_train["total_points"])
        self.model.save_model(config.model_out)
        print(f"Saved model to: {config.model_out}")

        # Save feature importance
        importance_df = pd.DataFrame(
            {"feature": self.feature_cols, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)
        print("Top 25 feature importances:")
        print(importance_df.head(25).to_string(index=False))
        importance_df.to_csv(config.importance_out, index=False)
        print(f"Saved full feature importances to: {config.importance_out}")

        # Holdout evaluation
        if config.train_full:
            print("Skipped holdout evaluation (train-full enabled).")
        else:
            holdout_preds = self.model.predict(holdout_df[self.feature_cols])
            holdout_mae = mean_absolute_error(holdout_df["total_points"], holdout_preds)
            holdout_rmse = np.sqrt(
                mean_squared_error(holdout_df["total_points"], holdout_preds)
            )
            print("Holdout evaluation (2025-26):")
            print(f"MAE: {holdout_mae:.4f}")
            print(f"RMSE: {holdout_rmse:.4f}")

    def _restrict_to_window(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Restrict data to most recent N gameweeks per season."""
        result = []
        for season in sorted(df["season"].unique()):
            season_df = df[df["season"] == season]
            max_gw = int(season_df["GW"].max())
            result.append(season_df[season_df["GW"] > max_gw - window])
        return pd.concat(result, ignore_index=False)


def train_model(
    position: str,
    model_type: str,
    train_full: bool = False,
) -> None:
    """Train a model for a given position.

    Args:
        position: Player position (GK, DEF, MID, FWD).
        model_type: Model type (lstm or xgboost).
        train_full: If True, train on all data including holdout.
    """
    if model_type == "lstm":
        config = LSTMConfig(position=position, train_full=train_full)  # type: ignore
        trainer = LSTMTrainer(config)
    elif model_type == "xgboost":
        config = XGBoostConfig(position=position, train_full=train_full)  # type: ignore
        trainer = XGBoostTrainer(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    trainer.train()
