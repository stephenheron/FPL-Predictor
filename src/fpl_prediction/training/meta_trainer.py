"""Meta-model trainer for learning optimal ensemble weights."""

import json
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from fpl_prediction.config.settings import LSTMConfig, MetaConfig, XGBoostConfig
from fpl_prediction.data import (
    build_features_lstm,
    build_features_xgboost,
    build_sequences,
    get_feature_columns_lstm,
    get_feature_columns_xgboost,
    load_season,
    scale_sequences,
)
from fpl_prediction.models import LSTMRegressor, get_device
from fpl_prediction.training.trainer import LSTMTrainer, XGBoostTrainer


@dataclass
class MetaWeights:
    """Learned meta-model weights."""

    position: str
    weight_xgb: float
    weight_lstm: float
    intercept: float
    alpha: float
    model_type: str
    mae: float
    rmse: float
    dominant_model: str
    trust_ratio: float


class MetaTrainer:
    """Trainer for meta-model ensemble weights.

    Uses leave-one-season-out to generate out-of-fold predictions,
    then trains Ridge/Lasso to learn optimal blend weights.

    Args:
        config: MetaConfig with training parameters.
    """

    def __init__(self, config: MetaConfig) -> None:
        self.config = config
        self.oof_predictions: pd.DataFrame | None = None
        self.weights: MetaWeights | None = None

    def generate_oof_predictions(self) -> pd.DataFrame:
        """Generate out-of-fold predictions using leave-one-season-out.

        For each season, trains base models on all OTHER seasons,
        then generates predictions for the held-out season.

        Returns:
            DataFrame with columns: player_id, season, GW, pred_xgb, pred_lstm, actual
        """
        config = self.config
        all_oof = []

        # Extract season identifiers from file paths
        seasons = []
        for f in config.train_files:
            # Extract "2022-23" from "data/merged_fpl_understat_2022-23.csv"
            season = Path(f).stem.split("_")[-1]
            seasons.append((season, f))

        print(f"Generating OOF predictions for {config.position}")
        print(f"Seasons: {[s[0] for s in seasons]}")

        for holdout_season, holdout_file in seasons:
            print(f"\n--- Holdout season: {holdout_season} ---")

            # Get training files (exclude holdout)
            train_files = tuple(f for s, f in seasons if s != holdout_season)

            # Load holdout data
            holdout_df = load_season(holdout_file)
            holdout_df = holdout_df[holdout_df["position"] == config.position].copy()

            if holdout_df.empty:
                print(f"No data for {config.position} in {holdout_season}, skipping")
                continue

            # Generate XGBoost OOF predictions
            xgb_preds = self._generate_xgb_oof(train_files, holdout_file, holdout_df)

            # Generate LSTM OOF predictions
            lstm_preds = self._generate_lstm_oof(train_files, holdout_file, holdout_df)

            # Combine predictions (inner join on player_id + GW)
            oof_df = pd.merge(
                xgb_preds,
                lstm_preds,
                on=["player_id", "season", "GW"],
                how="inner",
                suffixes=("_xgb", "_lstm"),
            )

            # Keep actual points from XGBoost side (they should be the same)
            oof_df = oof_df.rename(columns={"actual_xgb": "actual"})
            oof_df = oof_df.drop(columns=["actual_lstm"], errors="ignore")

            all_oof.append(oof_df)
            print(f"Collected {len(oof_df)} OOF samples for {holdout_season}")

        if not all_oof:
            raise ValueError("No OOF predictions generated. Check data availability.")

        self.oof_predictions = pd.concat(all_oof, ignore_index=True)

        # Save OOF predictions
        self.oof_predictions.to_csv(config.oof_predictions_out, index=False)
        print(f"\nSaved OOF predictions to: {config.oof_predictions_out}")
        print(f"Total OOF samples: {len(self.oof_predictions)}")

        return self.oof_predictions

    def _generate_xgb_oof(
        self,
        train_files: tuple[str, ...],
        holdout_file: str,
        holdout_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Train XGBoost on train_files and predict on holdout."""
        config = self.config

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config for this fold
            xgb_config = XGBoostConfig(
                position=config.position,
                train_files=train_files,
                holdout_file=holdout_file,
                train_full=False,
                random_state=config.random_state,
            )
            # Override output paths to temp directory
            model_path = Path(tmpdir) / "xgb_model.json"

            # Train XGBoost
            trainer = XGBoostTrainer(xgb_config)

            # Load and prepare data
            train_frames = [load_season(path) for path in train_files]
            train_df = pd.concat(train_frames, ignore_index=True)
            train_df = build_features_xgboost(
                train_df, config.position, xgb_config.roll_windows
            )

            feature_cols = get_feature_columns_xgboost(
                train_df, xgb_config.roll_windows, position=config.position
            )

            # Filter to position
            train_df = train_df[train_df["position"] == config.position]

            # Train model
            model = XGBRegressor(
                n_estimators=xgb_config.n_estimators,
                max_depth=xgb_config.max_depth,
                learning_rate=xgb_config.xgb_learning_rate,
                subsample=xgb_config.subsample,
                colsample_bytree=xgb_config.colsample_bytree,
                objective="reg:squarederror",
                random_state=config.random_state,
                n_jobs=-1,
            )
            model.fit(train_df[feature_cols], train_df["total_points"])

            # Prepare holdout for prediction
            holdout_features = build_features_xgboost(
                holdout_df.copy(), config.position, xgb_config.roll_windows
            )

            # Generate predictions
            preds = model.predict(holdout_features[feature_cols])

            return pd.DataFrame(
                {
                    "player_id": holdout_features["player_id"],
                    "season": holdout_features["season"],
                    "GW": holdout_features["GW"],
                    "pred_xgb": preds,
                    "actual": holdout_features["total_points"],
                }
            )

    def _generate_lstm_oof(
        self,
        train_files: tuple[str, ...],
        holdout_file: str,
        holdout_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Train LSTM on train_files and predict on holdout."""
        config = self.config

        # Create config for this fold
        lstm_config = LSTMConfig(
            position=config.position,
            train_files=train_files,
            holdout_file=holdout_file,
            train_full=False,
            random_state=config.random_state,
        )

        # Load and prepare training data
        train_frames = [load_season(path) for path in train_files]
        train_df = pd.concat(train_frames, ignore_index=True)
        train_df = build_features_lstm(train_df, config.position, lstm_config.roll_window)

        feature_cols = get_feature_columns_lstm(train_df, lstm_config.roll_window)

        # Filter to position
        train_df = train_df[train_df["position"] == config.position]

        # Build sequences for training
        X_train, y_train, _ = build_sequences(train_df, feature_cols, lstm_config.seq_len)

        if len(X_train) == 0:
            # Return empty predictions if no sequences
            return pd.DataFrame(columns=["player_id", "season", "GW", "pred_lstm", "actual"])

        # Scale sequences
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train = scale_sequences(scaler, X_train, fit=True)

        # Train LSTM
        device = get_device()
        model = LSTMRegressor(
            input_size=X_train.shape[-1],
            hidden_size=lstm_config.hidden_size,
            num_layers=lstm_config.num_layers,
            dropout=lstm_config.dropout,
        ).to(device)

        from torch import nn
        from torch.utils.data import DataLoader

        from fpl_prediction.models import SequenceDataset

        train_ds = SequenceDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=lstm_config.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lstm_config.learning_rate)
        loss_fn = nn.MSELoss()

        # Quick training (fewer epochs for OOF generation)
        model.train()
        for epoch in range(min(lstm_config.epochs, 20)):
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = loss_fn(outputs, y_batch)
                loss.backward()
                optimizer.step()

        # Prepare holdout for prediction
        holdout_features = build_features_lstm(
            holdout_df.copy(), config.position, lstm_config.roll_window
        )
        holdout_features = holdout_features[holdout_features["position"] == config.position]

        X_holdout, y_holdout, meta_holdout = build_sequences(
            holdout_features, feature_cols, lstm_config.seq_len
        )

        if len(X_holdout) == 0:
            return pd.DataFrame(columns=["player_id", "season", "GW", "pred_lstm", "actual"])

        X_holdout = scale_sequences(scaler, X_holdout)

        # Generate predictions
        model.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(X_holdout.astype(np.float32)).to(device)
            preds = model(inputs).cpu().numpy()

        return pd.DataFrame(
            {
                "player_id": meta_holdout["player_id"].values,
                "season": meta_holdout["season"].values,
                "GW": meta_holdout["GW"].values,
                "pred_lstm": preds.flatten(),
                "actual": y_holdout,
            }
        )

    def train(self) -> MetaWeights:
        """Train meta-model on OOF predictions.

        Returns:
            MetaWeights with learned weights.
        """
        config = self.config

        # Load or generate OOF predictions
        if self.oof_predictions is None:
            oof_path = Path(config.oof_predictions_out)
            if oof_path.exists():
                print(f"Loading cached OOF predictions from: {oof_path}")
                self.oof_predictions = pd.read_csv(oof_path)
            else:
                self.generate_oof_predictions()

        if self.oof_predictions is None or len(self.oof_predictions) == 0:
            raise ValueError("No OOF predictions available for training.")

        # Prepare features and target
        X = self.oof_predictions[["pred_xgb", "pred_lstm"]].values
        y = self.oof_predictions["actual"].values

        print(f"\nTraining {config.meta_model} meta-model for {config.position}")
        print(f"Samples: {len(X)}, Alpha range: {config.alpha_range}")

        # Train with cross-validation to find best alpha
        if config.meta_model == "ridge":
            model = RidgeCV(alphas=config.alpha_range, cv=5)
        else:  # lasso
            model = LassoCV(alphas=config.alpha_range, cv=5, random_state=config.random_state)

        model.fit(X, y)

        # Extract weights
        weight_xgb, weight_lstm = model.coef_
        intercept = model.intercept_
        alpha = model.alpha_

        # Evaluate on OOF data
        preds = model.predict(X)
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))

        # Determine dominant model
        if abs(weight_xgb) > abs(weight_lstm):
            dominant = "xgb"
            trust_ratio = abs(weight_xgb) / abs(weight_lstm) if weight_lstm != 0 else float("inf")
        else:
            dominant = "lstm"
            trust_ratio = abs(weight_lstm) / abs(weight_xgb) if weight_xgb != 0 else float("inf")

        self.weights = MetaWeights(
            position=config.position,
            weight_xgb=float(weight_xgb),
            weight_lstm=float(weight_lstm),
            intercept=float(intercept),
            alpha=float(alpha),
            model_type=config.meta_model,
            mae=float(mae),
            rmse=float(rmse),
            dominant_model=dominant,
            trust_ratio=float(trust_ratio),
        )

        return self.weights

    def save_weights(self) -> None:
        """Save learned weights to JSON."""
        if self.weights is None:
            raise ValueError("No weights to save. Call train() first.")

        config = self.config
        weights_dict = {
            "position": self.weights.position,
            "weight_xgb": self.weights.weight_xgb,
            "weight_lstm": self.weights.weight_lstm,
            "intercept": self.weights.intercept,
            "alpha": self.weights.alpha,
            "model_type": self.weights.model_type,
            "mae": self.weights.mae,
            "rmse": self.weights.rmse,
            "dominant_model": self.weights.dominant_model,
            "trust_ratio": self.weights.trust_ratio,
        }

        Path(config.weights_out).parent.mkdir(parents=True, exist_ok=True)
        with open(config.weights_out, "w") as f:
            json.dump(weights_dict, f, indent=2)

        print(f"Saved weights to: {config.weights_out}")


def train_meta_model(
    position: str,
    meta_model: str = "ridge",
    regenerate_oof: bool = False,
    alpha_range: tuple[float, ...] | None = None,
) -> MetaWeights:
    """Train a meta-model for a given position.

    Args:
        position: Player position (GK, DEF, MID, FWD).
        meta_model: Model type (ridge or lasso).
        regenerate_oof: If True, regenerate OOF predictions even if cached.
        alpha_range: Optional custom alpha values to try.

    Returns:
        MetaWeights with learned weights.
    """
    config = MetaConfig(
        position=position,  # type: ignore
        meta_model=meta_model,  # type: ignore
        alpha_range=alpha_range or (0.001, 0.01, 0.1, 1.0, 10.0),
    )

    trainer = MetaTrainer(config)

    if regenerate_oof:
        trainer.generate_oof_predictions()

    weights = trainer.train()
    trainer.save_weights()

    return weights
