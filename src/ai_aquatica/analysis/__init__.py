"""Exploratory statistics and time-series analysis utilities."""

from .statistics import (
    calculate_basic_statistics,
    calculate_correlation_matrix,
    decompose_time_series,
    perform_anova,
    plot_boxplot,
    plot_distribution,
)

__all__ = [
    "calculate_basic_statistics",
    "plot_distribution",
    "plot_boxplot",
    "calculate_correlation_matrix",
    "perform_anova",
    "decompose_time_series",
]
