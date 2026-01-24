"""Evaluation metrics and utilities for model training."""

from typing import Sized, cast

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Run a single training epoch.

    Args:
        model: PyTorch model to train.
        loader: DataLoader with training data.
        optimizer: Optimizer for parameter updates.
        loss_fn: Loss function.
        device: Device to run on.

    Returns:
        Average training loss for the epoch.
    """
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


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Evaluate model on a dataset.

    Args:
        model: PyTorch model to evaluate.
        loader: DataLoader with evaluation data.
        device: Device to run on.

    Returns:
        Tuple of (MAE, RMSE) scores.
    """
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []

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


def get_fold_pairs(
    df, season: str, window: int
) -> list[tuple[list[int], list[int]]]:
    """Get train/validation index pairs for walk-forward CV.

    Args:
        df: DataFrame with season and GW columns.
        season: Season to create folds for.
        window: Training window size.

    Returns:
        List of (train_indices, val_indices) tuples.
    """
    import pandas as pd

    season_df = df[df["season"] == season]
    max_gw = int(season_df["GW"].max())
    pairs = []

    for gw in range(window + 1, max_gw + 1):
        train_mask = (season_df["GW"] >= gw - window) & (season_df["GW"] <= gw - 1)
        val_mask = season_df["GW"] == gw
        train_idx = season_df[train_mask].index.tolist()
        val_idx = season_df[val_mask].index.tolist()
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        pairs.append((train_idx, val_idx))

    return pairs


def evaluate_cv_xgboost(
    df,
    feature_cols: list[str],
    train_windows: tuple[int, ...],
    random_state: int,
):
    """Evaluate XGBoost with walk-forward cross-validation.

    Args:
        df: DataFrame with features and target.
        feature_cols: List of feature column names.
        train_windows: Tuple of window sizes to evaluate.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with CV results for each window size.
    """
    import pandas as pd
    from xgboost import XGBRegressor

    results = []
    seasons = sorted(df["season"].unique())

    for window in train_windows:
        fold_mae = []
        fold_rmse = []

        for season in seasons:
            for train_idx, val_idx in get_fold_pairs(df, season, window):
                X_train = df.loc[train_idx, feature_cols]
                y_train = df.loc[train_idx, "total_points"]
                X_val = df.loc[val_idx, feature_cols]
                y_val = df.loc[val_idx, "total_points"]

                model = XGBRegressor(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=random_state,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_val)

                fold_mae.append(mean_absolute_error(y_val, preds))
                fold_rmse.append(np.sqrt(mean_squared_error(y_val, preds)))

        results.append(
            {
                "train_window": window,
                "mae": float(np.mean(fold_mae)) if fold_mae else np.nan,
                "rmse": float(np.mean(fold_rmse)) if fold_rmse else np.nan,
                "folds": len(fold_mae),
            }
        )

    return pd.DataFrame(results)
