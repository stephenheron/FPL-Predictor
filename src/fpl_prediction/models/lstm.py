"""LSTM model definition for FPL Prediction."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """PyTorch Dataset for sequence data.

    Args:
        sequences: Array of shape (n_samples, seq_len, n_features).
        targets: Optional array of shape (n_samples,) for training.
    """

    def __init__(
        self, sequences: np.ndarray, targets: np.ndarray | None = None
    ) -> None:
        self.sequences = torch.from_numpy(sequences)
        self.targets = None if targets is None else torch.from_numpy(targets)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        if self.targets is None:
            return self.sequences[idx]
        return self.sequences[idx], self.targets[idx]


class LSTMRegressor(nn.Module):
    """LSTM-based regression model for points prediction.

    Architecture:
        - LSTM layers with dropout between layers
        - Dropout after final LSTM output
        - Linear layer to single output

    Args:
        input_size: Number of input features.
        hidden_size: Size of LSTM hidden state.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
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
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, features).

        Returns:
            Predictions of shape (batch,).
        """
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        return self.fc(self.dropout(last_output)).squeeze(-1)


def get_device() -> torch.device:
    """Get the best available device for PyTorch.

    Returns:
        torch.device for MPS (Apple Silicon), CUDA, or CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
