"""Data loading and preprocessing modules."""

from .fixtures import build_future_rows
from .loader import load_season
from .name_matching import create_name_variants, normalize_name
from .preprocessor import (
    add_per90_features,
    add_rolling_features_lstm,
    add_rolling_features_xgboost,
    build_features_lstm,
    build_features_xgboost,
    get_feature_columns_lstm,
    get_feature_columns_xgboost,
)
from .sequences import build_prediction_sequences, build_sequences, scale_sequences

__all__ = [
    "add_per90_features",
    "add_rolling_features_lstm",
    "add_rolling_features_xgboost",
    "build_features_lstm",
    "build_features_xgboost",
    "build_future_rows",
    "build_prediction_sequences",
    "build_sequences",
    "create_name_variants",
    "get_feature_columns_lstm",
    "get_feature_columns_xgboost",
    "load_season",
    "normalize_name",
    "scale_sequences",
]
