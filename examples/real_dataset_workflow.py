"""Run AI-Aquatica on a real water-quality CSV file.

Usage
-----
python examples/real_dataset_workflow.py
python examples/real_dataset_workflow.py examples/data/water_quality.csv --output outputs/real_dataset

The script is intentionally conservative: it auto-detects CSV conventions,
normalizes common water-quality column names, runs hydrochemical ion-balance QC
when the required columns are present, trains a Random Forest station classifier
when station labels are available, and exports a standalone HTML report.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow running this example directly from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_aquatica.core import WaterQualityPipeline
from ai_aquatica.io import detect_csv_format, load_water_quality_csv

DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "water_quality.csv"


def _available(data, columns):
    return [column for column in columns if column in data.columns]


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-Aquatica real-dataset workflow")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(DEFAULT_DATASET),
        help="Path to a water-quality CSV file. Defaults to examples/data/water_quality.csv.",
    )
    parser.add_argument("--output", default="outputs/real_dataset", help="Output directory")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {csv_path}. Provide a path or keep the bundled file at "
            "examples/data/water_quality.csv."
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    detected = detect_csv_format(csv_path)
    data = load_water_quality_csv(csv_path)

    pipeline = WaterQualityPipeline.from_dataframe(data).describe()

    cations = _available(data, ["Ca", "Mg", "Na", "K", "NH4"])
    anions = _available(data, ["Cl", "SO4", "NO3", "NO2"])
    if {"Ca", "Mg"}.issubset(cations) and "Alkalinity" in data.columns and anions:
        pipeline.ion_balance_from_alkalinity(
            cations=cations,
            anions=anions,
            alkalinity_col="Alkalinity",
            alkalinity_units="mg_CaCO3_L",
            threshold=10.0,
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
    features = _available(pipeline.dataset.data, candidate_features)
    if "Station" in pipeline.dataset.data.columns and len(features) >= 4:
        pipeline.select_features(features, target="Station").impute().scale().pca(n_components=2)
        pipeline.train_random_forest(task="classification", test_size=0.25, random_state=42)

    report_path = pipeline.export_html_report(output_dir / "ai_aquatica_real_dataset_report.html")
    pipeline.dataset.data.to_csv(output_dir / "processed_water_quality.csv", index=False)

    print("Detected CSV format:", detected)
    print("Input shape:", data.shape)
    print("Report:", report_path)
    print("Processed data:", output_dir / "processed_water_quality.csv")


if __name__ == "__main__":
    main()
