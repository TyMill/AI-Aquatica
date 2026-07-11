from pathlib import Path

import pandas as pd

from ai_aquatica.io import detect_csv_format, load_water_quality_csv
from ai_aquatica.preprocessing import normalize_water_quality_columns


def test_detects_european_csv_and_normalizes_columns(tmp_path: Path):
    csv_path = tmp_path / "water.csv"
    csv_path.write_text(
        "Station;month number ;Chl a;Temp.;Zasadowoæ\x8d;Ca\n"
        "r1;1;25,5;16,5;125;84\n",
        encoding="latin1",
    )

    detected = detect_csv_format(csv_path)
    assert detected["sep"] == ";"
    assert detected["decimal"] == ","

    data = load_water_quality_csv(csv_path)
    assert list(data.columns) == ["Station", "month", "Chl_a", "Temp", "Alkalinity", "Ca"]
    assert data.loc[0, "Chl_a"] == 25.5
    assert data.loc[0, "Temp"] == 16.5


def test_normalize_water_quality_columns_keeps_unknown_columns_unique():
    data = pd.DataFrame({"Chl a": [1], "Chl-a": [2], "custom value": [3]})
    normalized = normalize_water_quality_columns(data)
    assert "Chl_a" in normalized.columns
    assert "Chl_a_1" in normalized.columns
    assert "custom_value" in normalized.columns
