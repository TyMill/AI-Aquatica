"""Example datasets bundled with AI-Aquatica."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).with_name("data")


def make_synthetic_water_quality(n_samples: int = 120, random_state: int = 42) -> pd.DataFrame:
    """Create a reproducible synthetic water-quality dataset.

    The generated data are not intended to represent a regulatory monitoring
    dataset. They provide a compact example for tutorials, tests and SoftwareX
    reproducibility checks.
    """

    rng = np.random.default_rng(random_state)
    sites = np.array(["Lake_A", "Lake_B", "River_A", "Urban_Lake"])
    site = rng.choice(sites, size=n_samples)
    temperature = rng.normal(15, 6, n_samples).clip(1, 28)
    pH = rng.normal(7.6, 0.45, n_samples).clip(6.2, 9.2)
    conductivity = rng.lognormal(mean=np.log(620), sigma=0.25, size=n_samples)
    nitrate = rng.gamma(2.2, 0.7, n_samples)
    phosphate = rng.gamma(1.5, 0.08, n_samples)
    dissolved_oxygen = (12.5 - 0.23 * temperature - 0.35 * phosphate + rng.normal(0, 0.8, n_samples)).clip(2, 14)
    chlorophyll_a = (4 + 14 * phosphate + 0.9 * nitrate + rng.normal(0, 2, n_samples)).clip(0.5, None)

    # Major ions in mg/L. Values are generated to be plausible for examples.
    ca = rng.normal(72, 13, n_samples).clip(20, None)
    mg = rng.normal(18, 5, n_samples).clip(3, None)
    na = rng.normal(38, 10, n_samples).clip(5, None)
    k = rng.normal(5, 2, n_samples).clip(0.5, None)
    hco3 = rng.normal(220, 45, n_samples).clip(40, None)
    cl = rng.normal(48, 13, n_samples).clip(5, None)
    so4 = rng.normal(75, 18, n_samples).clip(8, None)

    stress_score = phosphate * 6 + nitrate * 0.45 + chlorophyll_a * 0.03 - dissolved_oxygen * 0.18
    water_quality_class = np.where(stress_score > np.quantile(stress_score, 0.67), "poor", "good")
    water_quality_class = np.where(
        (stress_score > np.quantile(stress_score, 0.40)) & (stress_score <= np.quantile(stress_score, 0.67)),
        "moderate",
        water_quality_class,
    )

    dates = pd.date_range("2024-01-01", periods=n_samples, freq="7D")
    data = pd.DataFrame(
        {
            "sample_id": [f"AQ-{i:04d}" for i in range(n_samples)],
            "sampling_date": dates,
            "site": site,
            "temperature": temperature,
            "pH": pH,
            "conductivity": conductivity,
            "dissolved_oxygen": dissolved_oxygen,
            "nitrate": nitrate,
            "phosphate": phosphate,
            "chlorophyll_a": chlorophyll_a,
            "Ca": ca,
            "Mg": mg,
            "Na": na,
            "K": k,
            "HCO3": hco3,
            "Cl": cl,
            "SO4": so4,
            "water_quality_class": water_quality_class,
        }
    )

    # Inject a small amount of missingness to demonstrate preprocessing.
    for column in ["nitrate", "phosphate", "dissolved_oxygen"]:
        mask = rng.random(n_samples) < 0.05
        data.loc[mask, column] = np.nan
    return data


def load_example_dataset() -> pd.DataFrame:
    """Load the bundled synthetic water-quality dataset."""

    path = DATA_DIR / "example_water_quality.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["sampling_date"])
    return make_synthetic_water_quality()


__all__ = ["load_example_dataset", "make_synthetic_water_quality"]
