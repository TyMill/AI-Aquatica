"""Reproduce all AI-Aquatica SoftwareX illustrative results.

Usage
-----
python examples/real_dataset_workflow.py
python examples/real_dataset_workflow.py examples/data/water_quality.csv --output outputs/real_dataset

The script performs:
1. regional CSV format detection and loading,
2. hydrochemical charge-balance diagnostics,
3. leakage-safe campaign-grouped station classification,
4. a QC-filtered classification sensitivity analysis,
5. leakage-safe campaign-grouped chlorophyll-a regression with a mean baseline,
6. export of metrics, fold-level results, predictions, figures, processed data,
   and a standalone HTML report.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ai_aquatica.core import WaterQualityPipeline
from ai_aquatica.hydrochemistry import NITROGEN_AS_N_MASS_PER_MEQ
from ai_aquatica.io import detect_csv_format, load_water_quality_csv

DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "water_quality.csv"


def _available(data: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in data.columns]


def _summary_table(pipeline: WaterQualityPipeline) -> pd.DataFrame:
    rows = []
    for result_name in [
        "station_classification_grouped",
        "station_classification_qc_filtered",
        "chlorophyll_regression_grouped",
    ]:
        result = pipeline.results.get(result_name)
        if result is None:
            continue
        metrics = result.metrics
        row = {
            "analysis": result_name,
            "validation": metrics.get("validation"),
            "n_samples": metrics.get("n_samples_evaluated", metrics.get("n_test")),
            "quality_policy": metrics.get("quality_policy"),
        }
        for key in [
            "accuracy",
            "balanced_accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "f1_weighted",
            "r2",
            "mae",
            "rmse",
            "baseline_r2",
            "baseline_mae",
            "baseline_rmse",
        ]:
            if key in metrics:
                row[key] = metrics[key]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-Aquatica reproducible real-dataset workflow")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(DEFAULT_DATASET),
        help="Path to a water-quality CSV file.",
    )
    parser.add_argument("--output", default="outputs/real_dataset", help="Output directory")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap iterations for 95% CIs")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    output_dir = Path(args.output)
    artifact_dir = output_dir / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    detected = detect_csv_format(csv_path)
    data = load_water_quality_csv(csv_path)

    pipeline = WaterQualityPipeline.from_dataframe(data).describe()

    cations = _available(data, ["Ca", "Mg", "NH4"])
    anions = _available(data, ["Cl", "SO4", "NO3", "NO2"])
    if set(["Ca", "Mg"]).issubset(cations) and "Alkalinity" in data.columns and anions:
        pipeline.ion_balance_from_alkalinity(
            cations=cations,
            anions=anions,
            alkalinity_col="Alkalinity",
            alkalinity_units="mg_CaCO3_L",
            units="mg/L",
            threshold=10.0,
            equivalent_weights=NITROGEN_AS_N_MASS_PER_MEQ,
        )

    candidate_features = [
        "Temp",
        "pH",
        "Eh",
        "COD_Mn",
        "COD_Cr",
        "BOD5",
        "O2_dissolved",
        "Secchi_depth",
        "NO3",
        "NO2",
        "NH4",
        "TN",
        "SRP",
        "TP",
        "TH",
        "Ca",
        "Mg",
        "Cl",
        "SO4",
        "Alkalinity",
        "Fe_total",
        "Mn_total",
        "Pb",
        "Zn",
        "Cd",
        "Cu",
    ]
    features = _available(pipeline.data, candidate_features)

    if "Station" in pipeline.data.columns and "month" in pipeline.data.columns and len(features) >= 4:
        (
            pipeline.select_features(features=features, target="Station")
            .impute(strategy="median")
            .scale()
            .pca(n_components=2, use_for_model=False)
            .train_random_forest(
                task="classification",
                validation="group_kfold",
                group_column="month",
                n_splits=5,
                random_state=42,
                n_estimators=300,
                quality_policy="warn",
                bootstrap_iterations=args.bootstrap,
                result_name="station_classification_grouped",
            )
            .train_random_forest(
                task="classification",
                validation="group_kfold",
                group_column="month",
                n_splits=4,
                random_state=42,
                n_estimators=300,
                quality_policy="filter",
                bootstrap_iterations=args.bootstrap,
                result_name="station_classification_qc_filtered",
            )
        )

    if "Chl_a" in pipeline.data.columns and "month" in pipeline.data.columns and len(features) >= 4:
        (
            pipeline.select_features(features=features, target="Chl_a")
            .impute(strategy="median")
            .scale()
            .train_random_forest(
                task="regression",
                validation="group_kfold",
                group_column="month",
                n_splits=5,
                random_state=42,
                n_estimators=300,
                quality_policy="warn",
                bootstrap_iterations=args.bootstrap,
                result_name="chlorophyll_regression_grouped",
            )
        )

    pipeline.data.to_csv(output_dir / "processed_water_quality.csv", index=False)
    _summary_table(pipeline).to_csv(output_dir / "validation_summary.csv", index=False)
    pipeline.export_artifacts(artifact_dir)
    report_path = pipeline.export_html_report(output_dir / "ai_aquatica_real_dataset_report.html")

    print("Detected CSV format:", detected)
    print("Input shape:", data.shape)
    print("Features used:", features)
    print("Report:", report_path)
    print("Validation summary:", output_dir / "validation_summary.csv")
    print("Artifacts:", artifact_dir)


if __name__ == "__main__":
    main()
