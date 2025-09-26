"""Public package interface for AI Aquatica."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd


def _install_excel_fallback() -> None:
    """Provide a lightweight Excel fallback when optional deps are missing."""

    try:  # pragma: no cover - exercised implicitly when dependency available
        import openpyxl  # noqa: F401
    except ModuleNotFoundError:  # pragma: no cover - exercised in CI environment
        from pandas import DataFrame

        def _fallback_to_excel(
            self: pd.DataFrame,
            excel_writer: Any,
            sheet_name: str = "Sheet1",
            engine: str | None = None,
            storage_options: Any | None = None,
            **kwargs: Any,
        ) -> None:
            """Degrade gracefully by writing CSV data when openpyxl is absent."""

            if hasattr(excel_writer, "write"):
                raise ModuleNotFoundError(
                    "openpyxl is required for file-like Excel writers."
                )

            path = Path(os.fspath(excel_writer))
            index = kwargs.pop("index", True)

            csv_kwargs = {
                key: kwargs.pop(key)
                for key in (
                    "na_rep",
                    "float_format",
                    "columns",
                    "header",
                    "index_label",
                    "mode",
                    "encoding",
                    "compression",
                    "quoting",
                    "quotechar",
                    "line_terminator",
                    "chunksize",
                    "date_format",
                    "doublequote",
                    "escapechar",
                    "decimal",
                    "errors",
                    "storage_options",
                )
                if key in kwargs
            }

            # Ignore arguments that are specific to true Excel writers.
            _ = sheet_name, engine, storage_options, kwargs

            self.to_csv(path, index=index, **csv_kwargs)

        DataFrame.to_excel = _fallback_to_excel  # type: ignore[assignment]


_install_excel_fallback()

from . import data_import, data_visualization, ml_analysis
from .data_import import (
    import_csv,
    import_excel,
    import_json,
    import_from_sql,
    import_from_nosql,
)
from .data_visualization import (
    plot_line,
    plot_bar,
    plot_pie,
    plot_scatter,
    plot_heatmap,
    plot_pca,
    plot_tsne,
    plot_interactive_bubble,
)
from .ml_analysis import (
    train_linear_regression,
    train_logistic_regression,
    train_classification_model,
    evaluate_classification_model,
    perform_clustering,
    plot_clusters,
    detect_anomalies,
    generate_synthetic_data,
    train_test_split,
)

__all__ = [
    "data_import",
    "data_visualization",
    "ml_analysis",
    "import_csv",
    "import_excel",
    "import_json",
    "import_from_sql",
    "import_from_nosql",
    "plot_line",
    "plot_bar",
    "plot_pie",
    "plot_scatter",
    "plot_heatmap",
    "plot_pca",
    "plot_tsne",
    "plot_interactive_bubble",
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
