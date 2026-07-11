import pandas as pd

from ai_aquatica.core import WaterQualityPipeline
from ai_aquatica.hydrochemistry import add_bicarbonate_from_alkalinity, calculate_charge_balance_from_alkalinity


def test_add_bicarbonate_from_alkalinity():
    data = pd.DataFrame({"Alkalinity": [50.043]})
    result = add_bicarbonate_from_alkalinity(data)
    assert "HCO3" in result.columns
    assert abs(result.loc[0, "HCO3"] - 61.0168) < 1e-6


def test_calculate_charge_balance_from_alkalinity():
    data = pd.DataFrame({"Ca": [20.039], "Cl": [35.453], "Alkalinity": [50.043]})
    result = calculate_charge_balance_from_alkalinity(
        data,
        cations=["Ca"],
        anions=["Cl"],
        threshold=10,
    )
    assert "HCO3" in result.columns
    assert "Charge_Balance_Error_pct" in result.columns
    assert "Ion_Balance_Status" in result.columns


def test_pipeline_ion_balance_from_alkalinity():
    data = pd.DataFrame(
        {
            "Ca": [20.039, 25.0, 30.0],
            "Mg": [12.1525, 15.0, 18.0],
            "Cl": [35.453, 40.0, 45.0],
            "SO4": [48.03, 50.0, 52.0],
            "Alkalinity": [50.043, 55.0, 60.0],
        }
    )
    pipeline = WaterQualityPipeline.from_dataframe(data).ion_balance_from_alkalinity(
        cations=["Ca", "Mg"],
        anions=["Cl", "SO4"],
        threshold=10,
    )
    assert "ion_balance" in pipeline.results
    assert "HCO3" in pipeline.dataset.data.columns
