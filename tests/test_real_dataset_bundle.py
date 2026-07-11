from pathlib import Path

from ai_aquatica.io import detect_csv_format, load_water_quality_csv
from ai_aquatica.preprocessing import normalize_water_quality_columns


def test_bundled_real_dataset_can_be_loaded():
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "examples" / "data" / "water_quality.csv"

    assert data_path.exists()
    detected = detect_csv_format(data_path)
    data = load_water_quality_csv(data_path)
    data = normalize_water_quality_columns(data)

    assert detected["sep"] == ";"
    assert detected["decimal"] == ","
    assert data.shape[0] == 148
    assert "Station" in data.columns
    assert "Alkalinity" in data.columns
    assert "Chl_a" in data.columns
