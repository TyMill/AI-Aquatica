"""Deterministic AI-Aquatica workflow for reviewers and new users."""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ai_aquatica.modeling.classical import (
    detect_anomalies,
    evaluate_classification_model,
    train_classification_model,
)
from ai_aquatica.preprocessing.cleaning import remove_duplicates
from ai_aquatica.preprocessing.transformations import standardize_data
from ai_aquatica.hydrochemistry.legacy import calculate_ion_balance, identify_potential_errors
from ai_aquatica.preprocessing.missing import fill_missing_with_median
from ai_aquatica.reporting.legacy_reports import generate_statistical_report


def build_demo_dataset(seed: int = 42, n_samples: int = 120) -> pd.DataFrame:
    """Create a compact synthetic water-quality dataset."""

    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {
            "pH": rng.normal(7.4, 0.35, n_samples),
            "temperature_c": rng.normal(17.0, 4.0, n_samples),
            "conductivity_us_cm": rng.normal(620, 80, n_samples),
            "nitrate_mg_l": rng.gamma(2.0, 1.1, n_samples),
            "phosphate_mg_l": rng.gamma(1.4, 0.08, n_samples),
            "calcium_meq_l": rng.normal(3.2, 0.4, n_samples),
            "magnesium_meq_l": rng.normal(1.4, 0.2, n_samples),
            "chloride_meq_l": rng.normal(2.2, 0.3, n_samples),
            "sulfate_meq_l": rng.normal(1.6, 0.25, n_samples),
        }
    )
    data.loc[rng.choice(n_samples, 8, replace=False), "nitrate_mg_l"] = np.nan
    data["quality_alert"] = (
        (data["nitrate_mg_l"].fillna(data["nitrate_mg_l"].median()) > 3.0)
        | (data["phosphate_mg_l"] > 0.18)
    ).astype(int)
    return data


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    data = remove_duplicates(build_demo_dataset())
    data = fill_missing_with_median(data)

    feature_columns = [
        "pH",
        "temperature_c",
        "conductivity_us_cm",
        "nitrate_mg_l",
        "phosphate_mg_l",
    ]
    X = standardize_data(data[feature_columns])
    y = data["quality_alert"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    model = train_classification_model(X_train, y_train, model_type="random_forest")
    metrics = evaluate_classification_model(model, X_test, y_test)

    anomaly_labels = detect_anomalies(X, method="isolation_forest")
    ion_balance = calculate_ion_balance(
        data,
        cations=["calcium_meq_l", "magnesium_meq_l"],
        anions=["chloride_meq_l", "sulfate_meq_l"],
    )
    ion_warnings = identify_potential_errors(ion_balance, threshold=10)

    generate_statistical_report(data, report_path=output_dir / "statistical_report.html")

    print("AI-Aquatica demo completed")
    print(f"F1 score: {metrics.get('f1_score'):.3f}")
    print(f"Detected anomalies: {(anomaly_labels == -1).sum()}")
    print(f"Ion-balance warnings: {len(ion_warnings)}")
    print(f"Report: {output_dir / 'statistical_report.html'}")


if __name__ == "__main__":
    main()
