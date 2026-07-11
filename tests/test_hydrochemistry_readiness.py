import pandas as pd

from ai_aquatica.hydrochemistry import assess_ion_balance_inputs


def test_assess_ion_balance_inputs_reports_missing_major_ions():
    data = pd.DataFrame({"Ca": [40.0], "Mg": [12.0], "Cl": [20.0], "Alkalinity": [120.0]})

    report = assess_ion_balance_inputs(
        data,
        cations=["Ca", "Mg", "NH4"],
        anions=["Cl", "SO4", "NO3"],
        alkalinity_col="Alkalinity",
    )

    assert report["readiness_score"] < 1.0
    assert "NH4" in report["missing_columns"]
    assert "SO4" in report["missing_columns"]
    assert report["notes"]
