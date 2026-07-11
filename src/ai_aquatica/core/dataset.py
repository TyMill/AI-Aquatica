"""Core data container for AI-Aquatica workflows."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from ..io.water_quality_csv import load_water_quality_csv
from ..preprocessing.columns import normalize_water_quality_columns


@dataclass
class WaterQualityDataset:
    """A lightweight container around a water-quality DataFrame.

    The class stores the original table together with optional metadata such as
    site, date and target columns.  It deliberately keeps a pandas-first design
    so that domain scientists can inspect and modify the underlying data at any
    point in the workflow.
    """

    data: pd.DataFrame
    site_column: str | None = None
    date_column: str | None = None
    target_column: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        auto_detect: bool = True,
        normalize_columns: bool = True,
        **kwargs,
    ) -> "WaterQualityDataset":
        """Load a CSV file using water-quality-friendly defaults.

        By default this method detects common European CSV conventions such as
        semicolon separators, decimal commas and legacy encodings. Pass
        ``auto_detect=False`` to delegate fully to :func:`pandas.read_csv`.
        """

        if auto_detect or normalize_columns:
            data = load_water_quality_csv(
                path,
                auto_detect=auto_detect,
                normalize_columns=normalize_columns,
                **kwargs,
            )
        else:
            data = pd.read_csv(path, **kwargs)
        return cls(data)

    @classmethod
    def from_excel(cls, path: str | Path, **kwargs) -> "WaterQualityDataset":
        return cls(pd.read_excel(path, **kwargs))

    def copy(self) -> "WaterQualityDataset":
        return WaterQualityDataset(
            self.data.copy(),
            site_column=self.site_column,
            date_column=self.date_column,
            target_column=self.target_column,
            metadata=dict(self.metadata),
        )

    def validate_columns(self, columns: Sequence[str]) -> None:
        missing = [column for column in columns if column not in self.data.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

    def numeric_columns(self) -> list[str]:
        return list(self.data.select_dtypes(include="number").columns)

    def missingness(self) -> pd.DataFrame:
        """Return a missingness profile for every column."""

        rows = len(self.data)
        missing = self.data.isna().sum()
        return pd.DataFrame(
            {
                "column": missing.index,
                "missing_count": missing.values,
                "missing_fraction": (missing.values / rows) if rows else 0,
                "dtype": [str(self.data[column].dtype) for column in missing.index],
            }
        )

    def select_features(self, columns: Iterable[str]) -> pd.DataFrame:
        columns = list(columns)
        self.validate_columns(columns)
        return self.data.loc[:, columns]

    def normalize_columns(self) -> "WaterQualityDataset":
        """Normalize column names using water-quality domain aliases."""

        self.data = normalize_water_quality_columns(self.data)
        return self

    def set_roles(
        self,
        *,
        site_column: str | None = None,
        date_column: str | None = None,
        target_column: str | None = None,
    ) -> "WaterQualityDataset":
        for column in [site_column, date_column, target_column]:
            if column is not None:
                self.validate_columns([column])
        self.site_column = site_column if site_column is not None else self.site_column
        self.date_column = date_column if date_column is not None else self.date_column
        self.target_column = target_column if target_column is not None else self.target_column
        return self
