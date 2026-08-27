from pathlib import Path

from ai_aquatica.core import WaterQualityPipeline
from ai_aquatica.datasets import load_example_dataset
from ai_aquatica.hydrochemistry import (
    IonBalanceConfig,
    calculate_charge_balance,
    summarize_ion_balance,
)
from ai_aquatica.reporting import generate_water_quality_report


def test_professional_pipeline_generates_ion_balance_and_html_report(tmp_path):
    data = load_example_dataset().head(80)
    features = [
        "temperature",
        "pH",
        "conductivity",
        "dissolved_oxygen",
        "nitrate",
        "phosphate",
    ]

    pipeline = (
        WaterQualityPipeline.from_dataframe(data)
        .describe()
        .ion_balance(cations=["Ca", "Mg", "Na", "K"], anions=["HCO3", "Cl", "SO4"], units="mg/L")
        .select_features(features, target="water_quality_class")
        .impute()
        .scale()
        .pca(n_components=2)
        .train_random_forest(task="classification", quality_policy="ignore")
    )

    assert "ion_balance" in pipeline.results
    assert "random_forest" in pipeline.results
    assert "Ion_Balance" in pipeline.dataset.data.columns

    report_path = pipeline.export_html_report(tmp_path / "report.html")
    assert Path(report_path).exists()
    html = Path(report_path).read_text(encoding="utf-8")
    assert "AI-Aquatica water quality report" in html
    assert "Correlation structure" in html


def test_hydrochemistry_ion_balance_supports_mg_l_units():
    data = load_example_dataset().head(10)
    config = IonBalanceConfig(
        cations=["Ca", "Mg", "Na", "K"],
        anions=["HCO3", "Cl", "SO4"],
        units="mg/L",
        threshold=10,
    )
    result = calculate_charge_balance(data, config)
    summary = summarize_ion_balance(result, threshold=10)

    assert "Charge_Balance_Error_pct" in result.columns
    assert summary.n_samples == 10
    assert summary.max_abs_error >= 0


def test_standalone_html_report(tmp_path):
    data = load_example_dataset().head(20)
    path = generate_water_quality_report(data, tmp_path / "standalone.html")
    assert path.exists()
    assert "Dataset preview" in path.read_text(encoding="utf-8")
