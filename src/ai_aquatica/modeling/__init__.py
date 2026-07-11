"""Machine-learning utilities for aquatic environmental data."""

from sklearn.model_selection import train_test_split

from .classical import (
    build_discriminator,
    build_gan,
    build_generator,
    detect_anomalies,
    evaluate_classification_model,
    generate_synthetic_data,
    perform_clustering,
    plot_clusters,
    train_classification_model,
    train_linear_regression,
    train_logistic_regression,
)

__all__ = [
    "train_linear_regression",
    "train_logistic_regression",
    "train_classification_model",
    "evaluate_classification_model",
    "perform_clustering",
    "plot_clusters",
    "detect_anomalies",
    "generate_synthetic_data",
    "build_gan",
    "build_generator",
    "build_discriminator",
    "train_test_split",
]
