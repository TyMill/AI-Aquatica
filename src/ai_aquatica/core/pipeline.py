"""Fluent analysis pipeline for water-quality datasets."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .dataset import WaterQualityDataset
from .results import AnalysisResult
from ..hydrochemistry import (
    IonBalanceConfig,
    assess_ion_balance_inputs,
    calculate_charge_balance,
    calculate_charge_balance_from_alkalinity,
    summarize_ion_balance,
)
from ..reporting.html import generate_water_quality_report


@dataclass
class WaterQualityPipeline:
    """Composable water-quality analysis pipeline.

    The pipeline provides a higher-level API for publication-grade examples while
    keeping every intermediate table accessible through ``dataset.data`` and
    ``results``.  It is intentionally conservative and relies on standard
    scikit-learn estimators.
    """

    dataset: WaterQualityDataset | None = None
    feature_columns: list[str] = field(default_factory=list)
    target_column: str | None = None
    results: dict[str, AnalysisResult] = field(default_factory=dict)
    _X: pd.DataFrame | None = None
    _y: pd.Series | None = None

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame) -> "WaterQualityPipeline":
        return cls(WaterQualityDataset(data.copy()))

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs) -> "WaterQualityPipeline":
        return cls(WaterQualityDataset.from_csv(path, **kwargs))

    def select_features(self, columns: Sequence[str], target: str | None = None) -> "WaterQualityPipeline":
        self._require_dataset()
        self.dataset.validate_columns(list(columns))
        self.feature_columns = list(columns)
        self._X = self.dataset.data.loc[:, self.feature_columns].copy()
        if target is not None:
            self.dataset.validate_columns([target])
            self.target_column = target
            self._y = self.dataset.data[target].copy()
        return self

    def impute(self, strategy: str = "median") -> "WaterQualityPipeline":
        self._require_features()
        imputer = SimpleImputer(strategy=strategy)
        self._X = pd.DataFrame(imputer.fit_transform(self._X), columns=self.feature_columns, index=self._X.index)
        self.results["imputation"] = AnalysisResult("imputation", metrics={"strategy": strategy})
        return self

    def scale(self) -> "WaterQualityPipeline":
        self._require_features()
        scaler = StandardScaler()
        self._X = pd.DataFrame(scaler.fit_transform(self._X), columns=self.feature_columns, index=self._X.index)
        self.results["scaling"] = AnalysisResult("scaling", metrics={"method": "StandardScaler"})
        return self

    def describe(self) -> "WaterQualityPipeline":
        self._require_dataset()
        numeric = self.dataset.data.select_dtypes(include="number")
        result = AnalysisResult("descriptive_statistics")
        if not numeric.empty:
            table = numeric.describe().T
            table["range"] = table["max"] - table["min"]
            result.add_table("summary", table)
        result.add_table("missingness", self.dataset.missingness())
        self.results["descriptive_statistics"] = result
        return self

    def ion_balance(
        self,
        cations: Sequence[str],
        anions: Sequence[str],
        *,
        units: str = "mg/L",
        threshold: float = 5.0,
        equivalent_weights: Mapping[str, float] | None = None,
    ) -> "WaterQualityPipeline":
        self._require_dataset()
        preflight = assess_ion_balance_inputs(self.dataset.data, cations=cations, anions=anions)
        config = IonBalanceConfig(
            cations=list(cations),
            anions=list(anions),
            units=units,
            threshold=threshold,
            equivalent_weights=equivalent_weights or {},
        )
        balanced = calculate_charge_balance(self.dataset.data, config)
        self.dataset.data = balanced
        summary = summarize_ion_balance(balanced, threshold=threshold)
        self.results["ion_balance"] = AnalysisResult(
            "ion_balance",
            tables={"diagnostics": balanced},
            metrics={**summary.to_dict(), "input_readiness": preflight},
        )
        return self

    def ion_balance_from_alkalinity(
        self,
        cations: Sequence[str],
        anions: Sequence[str] | None = None,
        *,
        alkalinity_col: str = "Alkalinity",
        alkalinity_units: str = "mg_CaCO3_L",
        bicarbonate_col: str = "HCO3",
        units: str = "mg/L",
        threshold: float = 5.0,
        equivalent_weights: Mapping[str, float] | None = None,
    ) -> "WaterQualityPipeline":
        """Run charge-balance diagnostics after deriving HCO3 from alkalinity."""

        self._require_dataset()
        preflight = assess_ion_balance_inputs(
            self.dataset.data,
            cations=cations,
            anions=list(anions or []) + [bicarbonate_col],
            alkalinity_col=alkalinity_col,
        )
        balanced = calculate_charge_balance_from_alkalinity(
            self.dataset.data,
            cations=cations,
            anions=anions,
            alkalinity_col=alkalinity_col,
            alkalinity_units=alkalinity_units,
            bicarbonate_col=bicarbonate_col,
            units=units,
            threshold=threshold,
            equivalent_weights=equivalent_weights or {},
        )
        self.dataset.data = balanced
        summary = summarize_ion_balance(balanced, threshold=threshold)
        self.results["ion_balance"] = AnalysisResult(
            "ion_balance",
            tables={"diagnostics": balanced},
            metrics={
                **summary.to_dict(),
                "alkalinity_col": alkalinity_col,
                "alkalinity_units": alkalinity_units,
                "bicarbonate_col": bicarbonate_col,
                "input_readiness": preflight,
            },
        )
        return self

    def pca(self, n_components: int = 2) -> "WaterQualityPipeline":
        self._require_features()
        pca = PCA(n_components=n_components, random_state=42)
        scores = pca.fit_transform(self._X)
        columns = [f"PC{i + 1}" for i in range(n_components)]
        scores_df = pd.DataFrame(scores, columns=columns, index=self._X.index)
        self.results["pca"] = AnalysisResult(
            "pca",
            tables={"scores": scores_df},
            metrics={"explained_variance_ratio": pca.explained_variance_ratio_.tolist()},
        )
        return self

    def train_random_forest(
        self,
        *,
        task: str = "classification",
        test_size: float = 0.25,
        random_state: int = 42,
    ) -> "WaterQualityPipeline":
        self._require_features()
        if self._y is None:
            if self.target_column is None:
                raise ValueError("A target column is required for supervised modelling.")
            self._y = self.dataset.data[self.target_column]

        stratify = self._y if task == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            self._X,
            self._y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        if task == "classification":
            model = RandomForestClassifier(n_estimators=200, random_state=random_state)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            }
        elif task == "regression":
            model = RandomForestRegressor(n_estimators=200, random_state=random_state)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = {
                "r2": float(r2_score(y_test, y_pred)),
                "mae": float(mean_absolute_error(y_test, y_pred)),
            }
        else:
            raise ValueError("task must be either 'classification' or 'regression'.")

        importance = pd.DataFrame(
            {"feature": self.feature_columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
        self.results["random_forest"] = AnalysisResult(
            "random_forest",
            tables={"feature_importance": importance},
            metrics=metrics,
        )
        return self

    def export_html_report(self, output_path: str | Path) -> Path:
        self._require_dataset()
        return generate_water_quality_report(
            data=self.dataset.data,
            output_path=output_path,
            results=self.results,
            title="AI-Aquatica water quality report",
        )

    def _require_dataset(self) -> None:
        if self.dataset is None:
            raise ValueError("Load a dataset first using from_csv or from_dataframe.")

    def _require_features(self) -> None:
        self._require_dataset()
        if self._X is None or not self.feature_columns:
            raise ValueError("Select feature columns first using select_features().")


__all__ = ["WaterQualityPipeline", "WaterQualityDataset", "AnalysisResult"]
