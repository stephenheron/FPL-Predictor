"""Sequence building utilities for LSTM models."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
    target_col: str = "total_points",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build sequences for LSTM training from player gameweek data.

    Creates sliding window sequences for each player within each season.

    Args:
        df: Input DataFrame with player gameweek data.
        feature_cols: List of feature column names.
        seq_len: Length of each sequence.
        target_col: Target column name for prediction.

    Returns:
        Tuple of (sequences array, targets array, metadata DataFrame).
    """
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    meta_rows: list[pd.Series] = []
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
        return (
            np.empty((0, seq_len, len(feature_cols)), dtype=np.float32),
            np.array([]),
            pd.DataFrame(),
        )

    return (
        np.stack(sequences),
        np.array(targets, dtype=np.float32),
        pd.DataFrame(meta_rows),
    )


def build_prediction_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
    predict_gw: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build sequences for prediction (without targets).

    Args:
        df: Input DataFrame with player gameweek data.
        feature_cols: List of feature column names.
        seq_len: Length of each sequence.
        predict_gw: Optional GW filter to score only a single GW.

    Returns:
        Tuple of (sequences array, metadata DataFrame with row indices).
    """
    sequences: list[np.ndarray] = []
    meta_rows: list[dict] = []
    df = df.sort_values(["season", "player_id", "GW"]).copy()

    for _, group in df.groupby(["season", "player_id"], sort=False):
        group = group.sort_values("GW")
        values = group[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)

        for idx in range(seq_len, len(group)):
            row = group.iloc[idx]
            if predict_gw is not None and int(row["GW"]) != predict_gw:
                continue
            sequences.append(values[idx - seq_len : idx])
            meta = row.to_dict()
            meta["row_index"] = row.name
            meta_rows.append(meta)

    if not sequences:
        return (
            np.empty((0, seq_len, len(feature_cols)), dtype=np.float32),
            pd.DataFrame(),
        )

    return np.stack(sequences), pd.DataFrame(meta_rows)


def scale_sequences(
    scaler: StandardScaler,
    sequences: np.ndarray,
    fit: bool = False,
) -> np.ndarray:
    """Scale sequences using StandardScaler.

    Reshapes 3D sequence data to 2D, applies scaling, and reshapes back.

    Args:
        scaler: StandardScaler instance.
        sequences: 3D array of shape (n_samples, seq_len, n_features).
        fit: If True, fit the scaler before transforming.

    Returns:
        Scaled sequences with same shape as input.
    """
    sequences = np.asarray(sequences, dtype=np.float32)
    original_shape = sequences.shape
    flat = sequences.reshape(-1, original_shape[-1])

    if fit:
        scaler.fit(flat)

    scaled = np.asarray(scaler.transform(flat), dtype=np.float32)
    return scaled.reshape(original_shape).astype(np.float32)
