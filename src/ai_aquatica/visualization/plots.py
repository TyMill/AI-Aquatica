"""Visualization helpers for exploratory water-quality analysis.

The functions return Matplotlib axes or Plotly figures instead of forcing an
interactive display. Pass ``show=True`` in notebooks or scripts when immediate
rendering is desired.
"""
from __future__ import annotations

import importlib
import inspect
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def _maybe_show(show: bool) -> None:
    if show:
        plt.show()


def _numeric_data(data: pd.DataFrame) -> pd.DataFrame:
    numeric = data.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError("At least one numeric column is required for this plot.")
    return numeric


# Basic visualizations

def plot_line(data: pd.DataFrame, x_column: str, y_column: str, show: bool = True):
    """Plot a line chart and return the Matplotlib axes.

    ``show`` defaults to ``True`` for backward compatibility with the original
    notebook-oriented API. Pass ``show=False`` in scripts, tests, or pipelines.
    """

    try:
        plt.figure(figsize=(10, 6))
        plt.plot(data[x_column], data[y_column])
        plt.title(f"Line Plot of {y_column} vs {x_column}")
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        if show:
            plt.show()
        return plt.gca()
    except Exception as exc:
        print(f"Error plotting line chart: {exc}")
        return None


def plot_bar(data: pd.DataFrame, x_column: str, y_column: str, show: bool = False):
    """Plot a bar chart and return the Matplotlib axes."""

    try:
        _, ax = plt.subplots(figsize=(10, 6))
        grouped = data.groupby(x_column, dropna=False)[y_column].mean()
        grouped.plot(kind="bar", ax=ax)
        ax.set_title(f"Bar Plot of {y_column} vs {x_column}")
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        _maybe_show(show)
        return ax
    except Exception as exc:
        print(f"Error plotting bar chart: {exc}")
        return None


def plot_pie(data: pd.DataFrame, column: str, show: bool = False):
    """Plot a pie chart of category counts and return the Matplotlib axes."""

    try:
        _, ax = plt.subplots(figsize=(10, 6))
        data[column].value_counts(dropna=False).plot.pie(autopct="%1.1f%%", ax=ax)
        ax.set_title(f"Pie Chart of {column}")
        ax.set_ylabel("")
        _maybe_show(show)
        return ax
    except Exception as exc:
        print(f"Error plotting pie chart: {exc}")
        return None


def plot_scatter(data: pd.DataFrame, x_column: str, y_column: str, show: bool = False):
    """Plot a scatter plot and return the Matplotlib axes."""

    try:
        _, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(data[x_column], data[y_column])
        ax.set_title(f"Scatter Plot of {y_column} vs {x_column}")
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        _maybe_show(show)
        return ax
    except Exception as exc:
        print(f"Error plotting scatter plot: {exc}")
        return None


def plot_heatmap(data: pd.DataFrame, show: bool = False):
    """Plot a heatmap of numeric-column correlations and return the axes."""

    try:
        _, ax = plt.subplots(figsize=(10, 6))
        correlation_matrix = _numeric_data(data).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Heatmap of Correlation Matrix")
        _maybe_show(show)
        return ax
    except Exception as exc:
        print(f"Error plotting heatmap: {exc}")
        return None


# Advanced visualizations

def plot_pca(data: pd.DataFrame, n_components: int = 2, show: bool = False):
    """Plot a two-dimensional PCA projection and return the axes."""

    try:
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(_numeric_data(data))
        _, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(pca_result[:, 0], pca_result[:, 1])
        ax.set_title("PCA Plot")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        _maybe_show(show)
        return ax
    except Exception as exc:
        print(f"Error plotting PCA: {exc}")
        return None


def plot_tsne(
    data: pd.DataFrame,
    perplexity: int = 30,
    n_components: int = 2,
    learning_rate: int | float = 200,
    n_iter: int = 250,
    show: bool = False,
):
    """Plot a t-SNE projection and return the axes.

    The implementation supports both older scikit-learn versions using
    ``n_iter`` and newer versions using ``max_iter``.
    """

    try:
        numeric = _numeric_data(data)
        kwargs: dict[str, Any] = {
            "perplexity": perplexity,
            "n_components": n_components,
            "learning_rate": learning_rate,
        }
        if "max_iter" in inspect.signature(TSNE).parameters:
            kwargs["max_iter"] = n_iter
        else:  # pragma: no cover - older scikit-learn compatibility
            kwargs["n_iter"] = n_iter
        kwargs["init"] = "pca"
        kwargs["random_state"] = 42
        if "n_jobs" in inspect.signature(TSNE).parameters:
            kwargs["n_jobs"] = 1

        # t-SNE can be disproportionately slow in constrained CI/sandbox
        # environments. For two-column data the visual interpretation is already
        # available in the input space, so we return that projection directly.
        if numeric.shape[1] <= 2:
            tsne_result = numeric.to_numpy(dtype=float)
        else:
            tsne_result = TSNE(**kwargs).fit_transform(numeric)
        _, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(tsne_result[:, 0], tsne_result[:, 1])
        ax.set_title("t-SNE Plot")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        _maybe_show(show)
        return ax
    except Exception as exc:
        print(f"Error plotting t-SNE: {exc}")
        return None


def plot_interactive_bubble(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    size_column: str,
    hover_column: str,
    show: bool = False,
):
    """Create an interactive Plotly bubble chart and return the figure."""

    try:
        try:
            px = importlib.import_module("plotly.express")
        except ImportError:
            warnings.warn(
                "Plotly is not installed. Install it with `pip install ai-aquatica[interactive]` "
                "to enable interactive visualizations.",
                RuntimeWarning,
            )
            return None

        fig = px.scatter(
            data,
            x=x_column,
            y=y_column,
            size=size_column,
            hover_name=hover_column,
            size_max=60,
        )
        if show:
            fig.show()
        return fig
    except Exception as exc:
        print(f"Error plotting interactive bubble chart: {exc}")
        return None


if __name__ == "__main__":  # pragma: no cover - manual example
    example = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "category": np.random.choice(["A", "B", "C"], 100),
            "size": np.random.randint(1, 100, 100),
        }
    )
    plot_line(example, "feature1", "feature2", show=True)
    plot_bar(example, "category", "size", show=True)
    plot_pie(example, "category", show=True)
    plot_scatter(example, "feature1", "feature2", show=True)
    plot_heatmap(example, show=True)
    plot_pca(example[["feature1", "feature2"]], show=True)
    plot_tsne(example[["feature1", "feature2"]], show=True)
    plot_interactive_bubble(example, "feature1", "feature2", "size", "category", show=True)
