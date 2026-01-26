"""Centralized configuration settings for FPL Prediction models."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Position = Literal["GK", "DEF", "MID", "FWD"]
ModelType = Literal["lstm", "xgboost"]


@dataclass
class LSTMConfig:
    """Configuration for LSTM model training and prediction."""

    position: Position
    train_files: tuple[str, ...] = (
        "merged_fpl_understat_2022-23.csv",
        "merged_fpl_understat_2023-24.csv",
        "merged_fpl_understat_2024-25.csv",
    )
    holdout_file: str = "merged_fpl_understat_2025-26.csv"
    seq_len: int = 5
    roll_window: int = 8
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 128
    learning_rate: float = 1e-3
    epochs: int = 50
    patience: int = 6
    random_state: int = 42
    train_full: bool = False
    val_season: str | None = None

    @property
    def model_out(self) -> str:
        return f"lstm_{self.position.lower()}_model.pt"

    @property
    def scaler_out(self) -> str:
        return f"lstm_{self.position.lower()}_scaler.pkl"

    @property
    def report_out(self) -> str:
        return f"lstm_{self.position.lower()}_training_report.csv"


@dataclass
class XGBoostConfig:
    """Configuration for XGBoost model training and prediction."""

    position: Position
    train_files: tuple[str, ...] = (
        "merged_fpl_understat_2022-23.csv",
        "merged_fpl_understat_2023-24.csv",
        "merged_fpl_understat_2024-25.csv",
    )
    holdout_file: str = "merged_fpl_understat_2025-26.csv"
    roll_windows: tuple[int, ...] = (3, 5, 8)
    train_windows: tuple[int, ...] = (15, 25, 35)
    fixed_window_config: str = "best_windows.json"
    random_state: int = 42
    train_full: bool = False
    n_estimators: int = 300
    max_depth: int = 5
    xgb_learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8

    @property
    def model_out(self) -> str:
        return f"xgb_{self.position.lower()}_model.json"

    @property
    def importance_out(self) -> str:
        return f"feature_importance_{self.position.lower()}.csv"


@dataclass
class PredictionConfig:
    """Configuration for generating predictions."""

    position: Position
    model_type: ModelType
    input_file: str = "merged_fpl_understat_2025-26.csv"
    fixtures_file: str = "Fantasy-Premier-League/data/2025-26/fixtures.csv"
    teams_file: str = "Fantasy-Premier-League/data/2025-26/teams.csv"
    predict_gw: int | None = None
    roll_windows: tuple[int, ...] = (3, 5, 8)
    roll_window: int = 8
    seq_len: int | None = None

    @property
    def model_file(self) -> str:
        if self.model_type == "lstm":
            return f"lstm_{self.position.lower()}_model.pt"
        return f"xgb_{self.position.lower()}_model.json"

    @property
    def scaler_file(self) -> str:
        return f"lstm_{self.position.lower()}_scaler.pkl"

    @property
    def output_file(self) -> str:
        suffix = "_lstm" if self.model_type == "lstm" else ""
        return f"reports/predictions/{self.position.lower()}_predictions{suffix}.csv"


# Availability multiplier thresholds
AVAILABILITY_THRESHOLDS = {
    "high": 180,  # >= 180 minutes in last 3 GWs -> multiplier 1.0
    "medium": 90,  # >= 90 minutes -> multiplier 0.7
    "low": 0,  # > 0 minutes -> multiplier 0.4
}

AVAILABILITY_MULTIPLIERS = {
    "high": 1.0,
    "medium": 0.7,
    "low": 0.4,
    "none": 0.2,
}


# All valid positions
POSITIONS: tuple[Position, ...] = ("GK", "DEF", "MID", "FWD")
