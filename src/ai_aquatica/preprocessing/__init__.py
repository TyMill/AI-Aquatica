"""Preprocessing utilities for cleaning and transforming water-quality data."""

from .cleaning import handle_missing_values, normalize_data, remove_duplicates, standardize_data
from .columns import build_column_mapping, canonicalize_column_name, normalize_water_quality_columns
from .missing import (
    fill_missing_with_autoencoder,
    fill_missing_with_knn,
    fill_missing_with_mean,
    fill_missing_with_median,
    fill_missing_with_mode,
    fill_missing_with_regression,
)
from .transformations import boxcox_transform, log_transform, sqrt_transform

__all__ = [
    "normalize_water_quality_columns",
    "canonicalize_column_name",
    "build_column_mapping",
    "remove_duplicates",
    "handle_missing_values",
    "normalize_data",
    "standardize_data",
    "fill_missing_with_mean",
    "fill_missing_with_median",
    "fill_missing_with_mode",
    "fill_missing_with_knn",
    "fill_missing_with_regression",
    "fill_missing_with_autoencoder",
    "log_transform",
    "sqrt_transform",
    "boxcox_transform",
]
