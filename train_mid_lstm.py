import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sized, cast

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULTS = {
    "train_files": [
        "merged_fpl_understat_2022-23.csv",
        "merged_fpl_understat_2023-24.csv",
        "merged_fpl_understat_2024-25.csv",
    ],
    "holdout_file": "merged_fpl_understat_2025-26.csv",
    "position": "MID",
    "seq_len": 5,
    "roll_window": 8,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "epochs": 50,
    "patience": 6,
    "random_state": 42,
    "model_out": "lstm_mid_model.pt",
    "scaler_out": "lstm_mid_scaler.pkl",
    "report_out": "lstm_mid_training_report.csv",
    "train_full": False,
}

BASE_COLS = [
    "minutes",
    "starts",
    "total_points",
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "ict_index",
    "bps",
    "bonus",
    "threat",
    "influence",
    "creativity",
    "us_key_passes",
    "us_xGChain",
    "us_xGBuildup",
]

PER90_COLS = [
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "us_key_passes",
    "us_xGChain",
    "us_xGBuildup",
    "total_points",
]

ROLL_HINT_COLS = [
    "minutes",
    "total_points",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "ict_index",
]

FIXTURE_COLS = [
    "was_home",
    "opp_dyn_attack",
    "opp_dyn_defence",
    "opp_dyn_overall",
]


@dataclass
class Config:
    train_files: tuple
    holdout_file: str
    position: str
    seq_len: int
    roll_window: int
    hidden_size: int
    num_layers: int
    dropout: float
    batch_size: int
    learning_rate: float
    epochs: int
    patience: int
    random_state: int
    model_out: str
    scaler_out: str
    report_out: str
    val_season: str | None
    train_full: bool


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray | None = None) -> None:
        self.sequences = torch.from_numpy(sequences)
        self.targets = None if targets is None else torch.from_numpy(targets)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        if self.targets is None:
            return self.sequences[idx]
        return self.sequences[idx], self.targets[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        return self.fc(self.dropout(last_output)).squeeze(-1)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train MID LSTM model.")
    parser.add_argument("--train-files", nargs="+", default=DEFAULTS["train_files"])
    parser.add_argument("--holdout-file", default=DEFAULTS["holdout_file"])
    parser.add_argument("--position", default=DEFAULTS["position"])
    parser.add_argument("--seq-len", type=int, default=DEFAULTS["seq_len"])
    parser.add_argument("--roll-window", type=int, default=DEFAULTS["roll_window"])
    parser.add_argument("--hidden-size", type=int, default=DEFAULTS["hidden_size"])
    parser.add_argument("--num-layers", type=int, default=DEFAULTS["num_layers"])
    parser.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    parser.add_argument("--random-state", type=int, default=DEFAULTS["random_state"])
    parser.add_argument("--model-out", default=DEFAULTS["model_out"])
    parser.add_argument("--scaler-out", default=DEFAULTS["scaler_out"])
    parser.add_argument("--report-out", default=DEFAULTS["report_out"])
    parser.add_argument("--val-season", default=None)
    parser.add_argument(
        "--train-full",
        action="store_true",
        help="Train on all seasons (includes holdout) and skip holdout eval.",
    )
    args = parser.parse_args()

    return Config(
        train_files=tuple(args.train_files),
        holdout_file=args.holdout_file,
        position=args.position,
        seq_len=args.seq_len,
        roll_window=args.roll_window,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        patience=args.patience,
        random_state=args.random_state,
        model_out=args.model_out,
        scaler_out=args.scaler_out,
        report_out=args.report_out,
        val_season=args.val_season,
        train_full=args.train_full,
    )


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_season(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["GW"] = pd.to_numeric(df["GW"], errors="coerce")
    df["season"] = df["season"].astype(str)
    return df


def add_rolling_features(df: pd.DataFrame, roll_window: int) -> pd.DataFrame:
    df = df.sort_values(["season", "player_id", "GW"]).copy()
    grouped = df.groupby(["season", "player_id"], sort=False)
    for col in ROLL_HINT_COLS:
        shifted = grouped[col].shift(1)
        roll_name = f"roll_{roll_window}_{col}"
        df[roll_name] = (
            shifted.groupby([df["season"], df["player_id"]])
            .rolling(window=roll_window, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
    return df


def add_per90_features(df: pd.DataFrame) -> pd.DataFrame:
    minutes = df["minutes"].replace(0, np.nan)
    for col in PER90_COLS:
        per90_col = f"{col}_per90"
        df[per90_col] = df[col] / minutes * 90.0
    return df


def build_features(df: pd.DataFrame, position: str, roll_window: int) -> pd.DataFrame:
    df = pd.DataFrame(df)
    df = cast(pd.DataFrame, df.loc[df["position"] == position].copy())
    df = add_rolling_features(df, roll_window)
    df = add_per90_features(df)
    return df


def get_feature_columns(df: pd.DataFrame, roll_window: int) -> list:
    feature_cols = []
    feature_cols.extend(BASE_COLS)
    feature_cols.extend([f"{col}_per90" for col in PER90_COLS])
    feature_cols.extend([f"roll_{roll_window}_{col}" for col in ROLL_HINT_COLS])
    feature_cols.extend(FIXTURE_COLS)
    return [col for col in feature_cols if col in df.columns]


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    seq_len: int,
    target_col: str = "total_points",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    sequences = []
    targets = []
    meta_rows = []
    df = df.sort_values(["season", "player_id", "GW"])
    for _, group in df.groupby(["season", "player_id"], sort=False):
        group = group.sort_values("GW")
        values = group[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        target_vals = group[target_col].to_numpy(dtype=np.float32)
        for idx in range(seq_len, len(group)):
            sequences.append(values[idx - seq_len : idx])
            targets.append(target_vals[idx])
            meta_rows.append(group.iloc[idx])
    if not sequences:
        return np.empty((0, seq_len, len(feature_cols)), dtype=np.float32), np.array([]), pd.DataFrame()
    return (
        np.stack(sequences),
        np.array(targets, dtype=np.float32),
        pd.DataFrame(meta_rows),
    )


def scale_sequences(
    scaler: StandardScaler,
    sequences: np.ndarray,
    fit: bool = False,
) -> np.ndarray:
    sequences = np.asarray(sequences, dtype=np.float32)
    original_shape = sequences.shape
    flat = sequences.reshape(-1, original_shape[-1])
    if fit:
        scaler.fit(flat)
    scaled = np.asarray(scaler.transform(flat), dtype=np.float32)
    return scaled.reshape(original_shape).astype(np.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    dataset_size = len(cast(Sized, loader.dataset))
    for batch, targets in loader:
        batch = batch.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        preds = model(batch)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.size(0)
    return total_loss / dataset_size


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch, batch_targets in loader:
            batch = batch.to(device)
            outputs = model(batch).cpu().numpy()
            preds.append(outputs)
            targets.append(batch_targets.numpy())
    preds_arr = np.concatenate(preds) if preds else np.array([])
    targets_arr = np.concatenate(targets) if targets else np.array([])
    mae = mean_absolute_error(targets_arr, preds_arr) if len(preds_arr) else float("nan")
    rmse = (
        np.sqrt(mean_squared_error(targets_arr, preds_arr))
        if len(preds_arr)
        else float("nan")
    )
    return mae, rmse


def main() -> None:
    config = parse_args()
    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)

    train_frames = [load_season(path) for path in config.train_files]
    holdout_df = load_season(config.holdout_file)
    train_df = pd.concat(train_frames, ignore_index=True)
    if config.train_full:
        train_df = pd.concat([train_df, holdout_df], ignore_index=True)
    train_df = build_features(train_df, config.position, config.roll_window)
    holdout_df = build_features(holdout_df, config.position, config.roll_window)

    feature_cols = get_feature_columns(train_df, config.roll_window)

    seasons = sorted(train_df["season"].unique())
    train_seasons = [
        str(frame["season"].astype(str).iloc[0])
        for frame in train_frames
        if not frame.empty
    ]
    default_val = None
    if config.val_season:
        default_val = config.val_season
    elif config.train_full:
        default_val = train_seasons[-1] if train_seasons else None
    else:
        default_val = seasons[-1] if seasons else None
    val_season = default_val
    if val_season is None:
        raise ValueError("No seasons available for training.")

    train_split = pd.DataFrame(train_df[train_df["season"] != val_season])
    val_split = pd.DataFrame(train_df[train_df["season"] == val_season])
    if train_split.empty or val_split.empty:
        raise ValueError(
            f"Validation split failed. Check val season {val_season} in train data."
        )

    X_train, y_train, _ = build_sequences(train_split, feature_cols, config.seq_len)
    X_val, y_val, _ = build_sequences(val_split, feature_cols, config.seq_len)
    X_holdout, y_holdout, _ = build_sequences(holdout_df, feature_cols, config.seq_len)

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Not enough sequences to train the model.")

    scaler = StandardScaler()
    X_train = scale_sequences(scaler, X_train, fit=True)
    X_val = scale_sequences(scaler, X_val)
    if len(X_holdout):
        X_holdout = scale_sequences(scaler, X_holdout)

    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    device = get_device()
    model = LSTMRegressor(
        input_size=X_train.shape[-1],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    best_mae = float("inf")
    best_state = None
    patience_counter = 0
    report_rows = []

    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, loss_fn, device)
        val_mae, val_rmse = evaluate(model, val_loader, device)
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
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(config.report_out, index=False)
    print(f"Saved training report to: {config.report_out}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model_payload = {
        "model_state": model.state_dict(),
        "input_size": X_train.shape[-1],
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "seq_len": config.seq_len,
        "roll_window": config.roll_window,
        "feature_cols": feature_cols,
    }
    torch.save(model_payload, config.model_out)
    print(f"Saved model to: {config.model_out}")

    with Path(config.scaler_out).open("wb") as handle:
        pickle.dump(scaler, handle)
    print(f"Saved scaler to: {config.scaler_out}")

    if config.train_full:
        print("Holdout evaluation skipped (train-full enabled).")
    elif len(X_holdout):
        holdout_ds = SequenceDataset(X_holdout, y_holdout)
        holdout_loader = DataLoader(holdout_ds, batch_size=config.batch_size)
        holdout_mae, holdout_rmse = evaluate(model, holdout_loader, device)
        print("Holdout evaluation (2025-26):")
        print(f"MAE: {holdout_mae:.4f}")
        print(f"RMSE: {holdout_rmse:.4f}")
    else:
        print("Holdout evaluation skipped (no sequences available).")


if __name__ == "__main__":
    main()
