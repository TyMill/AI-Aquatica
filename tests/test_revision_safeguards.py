import numpy as np
import pandas as pd

from ai_aquatica.core import WaterQualityPipeline
from ai_aquatica.hydrochemistry import (
    NITROGEN_AS_N_MASS_PER_MEQ,
    IonBalanceConfig,
    calculate_charge_balance,
    summarize_ion_balance,
)


def test_all_missing_ions_are_indeterminate_not_zero_percent():
    data = pd.DataFrame(
        {
            "Ca": [20.039, np.nan],
            "Mg": [12.1525, np.nan],
            "Cl": [35.453, np.nan],
            "SO4": [48.03, np.nan],
        }
    )
    result = calculate_charge_balance(
        data,
        IonBalanceConfig(cations=["Ca", "Mg"], anions=["Cl", "SO4"], units="mg/L"),
    )

    assert result.loc[1, "Ion_Balance_Status"] == "indeterminate"
    assert np.isnan(result.loc[1, "Charge_Balance_Error_pct"])
    assert pd.isna(result.loc[1, "Potential_Error"])
    assert result.loc[1, "Ion_Balance_Diagnostic"] == "missing_selected_ion"

    summary = summarize_ion_balance(result)
    assert summary.n_samples == 2
    assert summary.n_evaluable == 1
    assert summary.n_indeterminate == 1


def test_negative_or_non_numeric_ion_values_are_indeterminate():
    data = pd.DataFrame(
        {
            "Ca": [20.039, -1.0, "not-a-number"],
            "Cl": [35.453, 35.453, 35.453],
        }
    )
    result = calculate_charge_balance(
        data,
        IonBalanceConfig(cations=["Ca"], anions=["Cl"], units="mg/L"),
    )

    assert result.loc[1, "Ion_Balance_Status"] == "indeterminate"
    assert result.loc[2, "Ion_Balance_Status"] == "indeterminate"
    assert result.loc[1, "Ion_Invalid_Cell_Count"] == 1
    assert result.loc[2, "Ion_Invalid_Cell_Count"] == 1


def test_preprocessing_is_deferred_and_grouped_validation_is_reproducible():
    data = pd.DataFrame(
        {
            "x1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
            "x2": [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "target": ["a", "a", "b", "b", "a", "a", "b", "b"],
            "campaign": [1, 1, 2, 2, 3, 3, 4, 4],
        }
    )
    pipeline = (
        WaterQualityPipeline.from_dataframe(data)
        .select_features(features=["x1", "x2"], target="target")
        .impute("median")
        .scale()
    )

    # Registration of preprocessing must not fit-transform the complete dataset.
    assert pipeline._X.isna().sum().sum() == 1

    pipeline.train_random_forest(
        task="classification",
        validation="group_kfold",
        group_column="campaign",
        n_splits=4,
        n_estimators=20,
        bootstrap_iterations=20,
        quality_policy="ignore",
        result_name="grouped_test",
    )

    result = pipeline.results["grouped_test"]
    assert result.metrics["validation"] == "group_kfold"
    assert result.metrics["confidence_interval_method"] == "cluster bootstrap by validation group"
    assert result.metrics["preprocessing"]["fit_scope"] == "training partition/fold only"
    assert len(result.tables["predictions"]) == len(data)
    assert len(result.tables["fold_metrics"]) == 4


def test_pipeline_data_property_and_features_alias():
    data = pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 0, 1, 1]})
    pipeline = WaterQualityPipeline.from_dataframe(data).select_features(features=["x"], target="y")
    assert pipeline.data.equals(data)


def test_group_kfold_rejects_missing_group_values():
    data = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "target": ["a", "b", "a", "b"],
            "campaign": [1, 1, np.nan, 2],
        }
    )
    pipeline = WaterQualityPipeline.from_dataframe(data).select_features(["x"], target="target")
    try:
        pipeline.train_random_forest(
            validation="group_kfold",
            group_column="campaign",
            n_splits=2,
            quality_policy="ignore",
            bootstrap_iterations=5,
        )
    except ValueError as exc:
        assert "group_column contains missing values" in str(exc)
    else:
        raise AssertionError("Expected missing validation groups to be rejected.")


def test_grouped_regression_exports_baseline_and_group_aware_intervals():
    data = pd.DataFrame(
        {
            "x1": [1.0, 1.5, 2.0, np.nan, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
            "x2": [6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0],
            "target": [2.0, 2.2, 2.8, 3.1, 3.5, 3.8, 4.2, 4.6, 5.1, 5.4, 5.9, 6.3],
            "campaign": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        }
    )
    pipeline = (
        WaterQualityPipeline.from_dataframe(data)
        .select_features(["x1", "x2"], target="target")
        .impute("median")
        .scale()
        .train_random_forest(
            task="regression",
            validation="group_kfold",
            group_column="campaign",
            n_splits=3,
            n_estimators=20,
            bootstrap_iterations=20,
            quality_policy="ignore",
            result_name="regression_test",
        )
    )

    metrics = pipeline.results["regression_test"].metrics
    assert metrics["validation"] == "group_kfold"
    assert metrics["confidence_interval_method"] == "cluster bootstrap by validation group"
    assert "rmse" in metrics
    assert "baseline_rmse" in metrics
    assert "r2_ci95_low" in metrics
    assert "r2_ci95_high" in metrics


def test_nitrogen_species_reported_as_n_use_elemental_reporting_basis():
    data = pd.DataFrame({"NH4": [14.0067], "NO3": [14.0067]})
    result = calculate_charge_balance(
        data,
        IonBalanceConfig(
            cations=["NH4"],
            anions=["NO3"],
            units="mg/L",
            equivalent_weights=NITROGEN_AS_N_MASS_PER_MEQ,
        ),
    )

    assert np.isclose(result.loc[0, "Cations_Sum_meq_L"], 1.0)
    assert np.isclose(result.loc[0, "Anions_Sum_meq_L"], 1.0)
    assert np.isclose(result.loc[0, "Charge_Balance_Error_pct"], 0.0)
    assert result.loc[0, "Ion_Balance_Status"] == "acceptable"


def test_pipeline_records_reporting_basis_conversion_overrides():
    data = pd.DataFrame({
        "NH4": [14.0067],
        "NO3": [14.0067],
        "Alkalinity": [0.0],
    })
    pipeline = WaterQualityPipeline.from_dataframe(data).ion_balance_from_alkalinity(
        cations=["NH4"],
        anions=["NO3"],
        alkalinity_col="Alkalinity",
        equivalent_weights=NITROGEN_AS_N_MASS_PER_MEQ,
        threshold=10.0,
    )
    metrics = pipeline.results["ion_balance"].metrics
    assert metrics["units"] == "mg/L"
    assert metrics["conversion_overrides_mg_per_meq"]["NO3"] == 14.0067
    assert metrics["conversion_overrides_mg_per_meq"]["NH4"] == 14.0067
