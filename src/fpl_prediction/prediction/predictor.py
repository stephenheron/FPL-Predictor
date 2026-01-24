"""Unified prediction logic for LSTM and XGBoost models."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor

from fpl_prediction.config.features import OUTPUT_COLS
from fpl_prediction.config.settings import PredictionConfig
from fpl_prediction.data import (
    build_features_lstm,
    build_features_xgboost,
    build_future_rows,
    build_prediction_sequences,
    get_feature_columns_lstm,
    get_feature_columns_xgboost,
)
from fpl_prediction.models import LSTMRegressor, get_device
from fpl_prediction.prediction.availability import (
    compute_availability_multipliers,
    compute_minutes_last_3,
    compute_minutes_last_3_xgboost,
)


class LSTMPredictor:
    """Unified LSTM predictor for all positions.

    Args:
        config: PredictionConfig with prediction parameters.
    """

    def __init__(self, config: PredictionConfig) -> None:
        self.config = config
        self.model: LSTMRegressor | None = None
        self.checkpoint: dict = {}
        self.scaler = None

    def predict(self) -> pd.DataFrame:
        """Generate predictions using the LSTM model.

        Returns:
            DataFrame with predictions.
        """
        config = self.config

        # Load data
        df = pd.read_csv(config.input_file)
        df["is_future"] = False

        # Build future rows if needed
        if config.predict_gw is not None and (df["GW"] == config.predict_gw).sum() == 0:
            future_rows = build_future_rows(
                df,
                config.fixtures_file,
                config.teams_file,
                config.predict_gw,
            )
            df = pd.concat([df, future_rows], ignore_index=True)

        # Load model checkpoint
        self.checkpoint = torch.load(config.model_file, map_location="cpu")
        roll_window = self.checkpoint.get("roll_window", config.roll_window)

        # Build features
        df = build_features_lstm(df, config.position, roll_window)

        # Get feature columns
        feature_cols = self.checkpoint.get("feature_cols") or get_feature_columns_lstm(
            df, roll_window
        )
        seq_len = config.seq_len or self.checkpoint.get("seq_len")
        if seq_len is None:
            raise ValueError("Sequence length is missing. Provide --seq-len.")

        # Build sequences
        sequences, meta_rows = build_prediction_sequences(
            df, feature_cols, seq_len, config.predict_gw
        )

        if len(sequences) == 0:
            raise ValueError("No sequences available for prediction.")

        # Load scaler and scale sequences
        with Path(config.scaler_file).open("rb") as handle:
            self.scaler = pickle.load(handle)

        flat = sequences.reshape(-1, sequences.shape[-1])
        scaled = self.scaler.transform(flat).reshape(sequences.shape).astype(np.float32)

        # Initialize and load model
        device = get_device()
        self.model = LSTMRegressor(
            input_size=scaled.shape[-1],
            hidden_size=self.checkpoint["hidden_size"],
            num_layers=self.checkpoint["num_layers"],
            dropout=self.checkpoint["dropout"],
        )
        self.model.load_state_dict(self.checkpoint["model_state"])
        self.model.to(device)
        self.model.eval()

        # Generate predictions
        with torch.no_grad():
            inputs = torch.from_numpy(scaled).to(device)
            preds = self.model(inputs).cpu().numpy()

        # Apply availability multipliers
        minutes_last_3 = compute_minutes_last_3(df, config.predict_gw)
        minutes_last_3 = (
            minutes_last_3.reindex(meta_rows["row_index"]).fillna(0).to_numpy()
        )
        multipliers = compute_availability_multipliers(minutes_last_3)
        preds = preds * multipliers

        # Build output DataFrame
        existing_cols = [col for col in OUTPUT_COLS if col in meta_rows.columns]
        result = meta_rows[existing_cols].copy()
        result["availability_multiplier"] = multipliers
        result["predicted_points"] = preds

        # Save predictions
        result.to_csv(config.output_file, index=False)
        print(f"Saved predictions to: {config.output_file}")

        return result


class XGBoostPredictor:
    """Unified XGBoost predictor for all positions.

    Args:
        config: PredictionConfig with prediction parameters.
    """

    def __init__(self, config: PredictionConfig) -> None:
        self.config = config
        self.model: XGBRegressor | None = None

    def predict(self) -> pd.DataFrame:
        """Generate predictions using the XGBoost model.

        Returns:
            DataFrame with predictions.
        """
        config = self.config

        # Load data
        df = pd.read_csv(config.input_file)
        df["is_future"] = False

        # Build future rows if needed
        if config.predict_gw is not None and (df["GW"] == config.predict_gw).sum() == 0:
            future_rows = build_future_rows(
                df,
                config.fixtures_file,
                config.teams_file,
                config.predict_gw,
            )
            df = pd.concat([df, future_rows], ignore_index=True)

        # Build features
        df = build_features_xgboost(df, config.position, config.roll_windows)

        # Get feature columns
        feature_cols = get_feature_columns_xgboost(df, config.roll_windows)
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        # Load model
        self.model = XGBRegressor()
        self.model.load_model(config.model_file)

        # Compute availability BEFORE filtering (need access to historical data)
        if config.predict_gw is not None:
            minutes_last_3 = compute_minutes_last_3_xgboost(df, config.predict_gw)
            # Filter to prediction GW
            gw_mask = df["GW"] == config.predict_gw
            df = df[gw_mask].copy()
            minutes_last_3 = minutes_last_3[gw_mask]
        elif "roll_3_minutes" in df.columns:
            minutes_last_3 = df["roll_3_minutes"].fillna(0) * 3
        else:
            minutes_last_3 = pd.Series(np.zeros(len(df)), index=df.index)

        # Generate predictions
        preds = self.model.predict(df[feature_cols])

        multipliers = compute_availability_multipliers(minutes_last_3.to_numpy())
        preds = preds * multipliers

        # Build output DataFrame
        existing_cols = [col for col in OUTPUT_COLS if col in df.columns]
        result = df[existing_cols].copy()
        result["availability_multiplier"] = multipliers
        result["predicted_points"] = preds

        # Save predictions
        result.to_csv(config.output_file, index=False)
        print(f"Saved predictions to: {config.output_file}")

        return result


def predict_model(
    position: str,
    model_type: str,
    predict_gw: int | None = None,
    output_file: str | None = None,
) -> pd.DataFrame:
    """Generate predictions for a given position.

    Args:
        position: Player position (GK, DEF, MID, FWD).
        model_type: Model type (lstm or xgboost).
        predict_gw: Optional gameweek to predict.
        output_file: Optional custom output file path.

    Returns:
        DataFrame with predictions.
    """
    config = PredictionConfig(
        position=position,  # type: ignore
        model_type=model_type,  # type: ignore
        predict_gw=predict_gw,
    )

    if output_file:
        # Override the default output file
        config = PredictionConfig(
            position=position,  # type: ignore
            model_type=model_type,  # type: ignore
            predict_gw=predict_gw,
        )
        # Use a mutable approach since dataclass is frozen-like
        object.__setattr__(config, "_output_file_override", output_file)

    if model_type == "lstm":
        predictor = LSTMPredictor(config)
    elif model_type == "xgboost":
        predictor = XGBoostPredictor(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return predictor.predict()
