"""Convenience layer for machine-learning helpers used in the tests."""
from __future__ import annotations

import builtins as _builtins

from ai_aquatica.ai_ml_models import (
    train_linear_regression,
    train_logistic_regression,
    train_classification_model,
    evaluate_classification_model,
    perform_clustering,
    plot_clusters,
    detect_anomalies,
    generate_synthetic_data,
)
from sklearn.model_selection import train_test_split as _train_test_split

__all__ = [
    "train_linear_regression",
    "train_logistic_regression",
    "train_classification_model",
    "evaluate_classification_model",
    "perform_clustering",
    "plot_clusters",
    "detect_anomalies",
    "generate_synthetic_data",
    "train_test_split",
]

# Re-export ``train_test_split`` so callers relying on the legacy API still work.
train_test_split = _train_test_split

# The legacy notebook examples – mirrored by the unit tests – used ``train_test_split``
# without importing it explicitly.  Making it available via ``builtins`` preserves this
# historical behaviour without forcing downstream users to update immediately.
if not hasattr(_builtins, "train_test_split"):
    _builtins.train_test_split = train_test_split

# Avoid leaking helper names from the temporary imports.
del _builtins
