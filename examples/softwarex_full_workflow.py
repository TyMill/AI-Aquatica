"""SoftwareX reproducibility workflow for AI-Aquatica.

Run with:
    python examples/softwarex_full_workflow.py

Outputs are written to ``outputs/softwarex_example/``.
"""
from __future__ import annotations

from pathlib import Path

from ai_aquatica.core import WaterQualityPipeline
from ai_aquatica.datasets import load_example_dataset

OUTPUT_DIR = Path("outputs/softwarex_example")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    data = load_example_dataset()
    features = [
        "temperature",
        "pH",
        "conductivity",
        "dissolved_oxygen",
        "nitrate",
        "phosphate",
        "chlorophyll_a",
    ]

    pipeline = (
        WaterQualityPipeline.from_dataframe(data)
        .describe()
        .ion_balance(cations=["Ca", "Mg", "Na", "K"], anions=["HCO3", "Cl", "SO4"], units="mg/L")
        .select_features(features, target="water_quality_class")
        .impute(strategy="median")
        .scale()
        .pca(n_components=2)
        .train_random_forest(task="classification")
    )

    report_path = pipeline.export_html_report(OUTPUT_DIR / "ai_aquatica_report.html")
    pipeline.dataset.data.to_csv(OUTPUT_DIR / "processed_water_quality.csv", index=False)

    print(f"HTML report: {report_path}")
    print(f"Processed data: {OUTPUT_DIR / 'processed_water_quality.csv'}")
    print("Ion balance summary:", pipeline.results["ion_balance"].metrics)
    print("Random forest metrics:", pipeline.results["random_forest"].metrics)


if __name__ == "__main__":
    main()
