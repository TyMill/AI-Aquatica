"""AI-Aquatica: reproducible AI-assisted analysis of water-quality data."""
from __future__ import annotations

from .analysis import calculate_basic_statistics, calculate_correlation_matrix
from .core import AnalysisResult, WaterQualityDataset, WaterQualityPipeline
from .datasets import load_example_dataset, make_synthetic_water_quality
from .hydrochemistry import (
    IonBalanceConfig,
    add_bicarbonate_from_alkalinity,
    assess_ion_balance_inputs,
    bicarbonate_from_alkalinity,
    calculate_charge_balance,
    calculate_charge_balance_from_alkalinity,
    summarize_ion_balance,
)
from .io import (
    detect_csv_format,
    import_csv,
    import_excel,
    import_from_nosql,
    import_from_sql,
    import_json,
    load_water_quality_csv,
)
from .modeling import (
    detect_anomalies,
    evaluate_classification_model,
    generate_synthetic_data,
    perform_clustering,
    plot_clusters,
    train_classification_model,
    train_linear_regression,
    train_logistic_regression,
    train_test_split,
)
from .preprocessing import (
    handle_missing_values,
    normalize_data,
    normalize_water_quality_columns,
    remove_duplicates,
    standardize_data,
)
from .reporting import HtmlReportConfig, generate_water_quality_report
from .visualization import (
    plot_bar,
    plot_heatmap,
    plot_interactive_bubble,
    plot_line,
    plot_pca,
    plot_pie,
    plot_scatter,
    plot_tsne,
)

__version__ = "2.2.0"

__all__ = [
    "__version__",
    "WaterQualityDataset",
    "WaterQualityPipeline",
    "AnalysisResult",
    "IonBalanceConfig",
    "assess_ion_balance_inputs",
    "add_bicarbonate_from_alkalinity",
    "bicarbonate_from_alkalinity",
    "calculate_charge_balance",
    "calculate_charge_balance_from_alkalinity",
    "summarize_ion_balance",
    "HtmlReportConfig",
    "generate_water_quality_report",
    "load_example_dataset",
    "make_synthetic_water_quality",
    "detect_csv_format",
    "load_water_quality_csv",
    "import_csv",
    "import_excel",
    "import_json",
    "import_from_sql",
    "import_from_nosql",
    "normalize_water_quality_columns",
    "remove_duplicates",
    "handle_missing_values",
    "normalize_data",
    "standardize_data",
    "calculate_basic_statistics",
    "calculate_correlation_matrix",
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
