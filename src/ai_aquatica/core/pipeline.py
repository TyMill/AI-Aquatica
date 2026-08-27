"""Fluent, leakage-safe analysis pipeline for water-quality datasets."""
from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

from ..hydrochemistry import (
    IonBalanceConfig,
    assess_ion_balance_inputs,
    calculate_charge_balance,
    calculate_charge_balance_from_alkalinity,
    summarize_ion_balance,
)
from ..reporting.html import generate_water_quality_report
from .dataset import WaterQualityDataset
from .results import AnalysisResult


@dataclass
class WaterQualityPipeline:
    """Composable water-quality analysis pipeline.

    Preprocessing operations are registered first and fitted only inside the
    training partition or validation fold. This prevents information leakage
    from imputation, scaling, and optional PCA into the evaluation data.
    Intermediate source data and result tables remain inspectable.
    """

    dataset: WaterQualityDataset | None = None
    feature_columns: list[str] = field(default_factory=list)
    target_column: str | None = None
    results: dict[str, AnalysisResult] = field(default_factory=dict)
    fitted_models: dict[str, Any] = field(default_factory=dict, repr=False)
    _X: pd.DataFrame | None = None
    _y: pd.Series | None = None
    _imputation_strategy: str | None = None
    _scale_enabled: bool = False
    _pca_components: int | None = None
    _pca_for_model: bool = False

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame) -> WaterQualityPipeline:
        return cls(WaterQualityDataset(data.copy()))

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs) -> WaterQualityPipeline:
        return cls(WaterQualityDataset.from_csv(path, **kwargs))

    @property
    def data(self) -> pd.DataFrame:
        """Backward-compatible access to the underlying DataFrame."""
        self._require_dataset()
        return self.dataset.data

    def select_features(
        self,
        columns: Sequence[str] | None = None,
        target: str | None = None,
        *,
        features: Sequence[str] | None = None,
    ) -> WaterQualityPipeline:
        """Select predictor columns and an optional supervised target.

        ``features=`` is accepted as an alias for ``columns=`` so manuscript
        examples and existing user code remain executable.
        """

        self._require_dataset()
        if columns is not None and features is not None:
            raise ValueError("Use either columns= or features=, not both.")
        selected = list(features if features is not None else (columns or []))
        if not selected:
            raise ValueError("At least one feature column must be selected.")
        self.dataset.validate_columns(selected)
        self.feature_columns = selected
        self._X = self.dataset.data.loc[:, selected].copy()
        if target is not None:
            self.dataset.validate_columns([target])
            self.target_column = target
            self._y = self.dataset.data[target].copy()
        return self

    def impute(self, strategy: str = "median") -> WaterQualityPipeline:
        """Register imputation for leakage-safe fitting inside each split/fold."""

        self._require_features()
        allowed = {"mean", "median", "most_frequent", "constant"}
        if strategy not in allowed:
            raise ValueError(f"Unsupported imputation strategy: {strategy!r}.")
        self._imputation_strategy = strategy
        self.results["imputation"] = AnalysisResult(
            "imputation",
            metrics={
                "strategy": strategy,
                "fit_scope": "training partition/fold only",
                "leakage_safe": True,
            },
        )
        return self

    def scale(self) -> WaterQualityPipeline:
        """Register standardization for fitting inside each split/fold."""

        self._require_features()
        self._scale_enabled = True
        self.results["scaling"] = AnalysisResult(
            "scaling",
            metrics={
                "method": "StandardScaler",
                "fit_scope": "training partition/fold only",
                "leakage_safe": True,
            },
        )
        return self

    def describe(self) -> WaterQualityPipeline:
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
    ) -> WaterQualityPipeline:
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
            metrics={
                **summary.to_dict(),
                "units": units,
                "conversion_overrides_mg_per_meq": dict(equivalent_weights or {}),
                "input_readiness": preflight,
            },
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
    ) -> WaterQualityPipeline:
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
                "units": units,
                "conversion_overrides_mg_per_meq": dict(equivalent_weights or {}),
                "input_readiness": preflight,
            },
        )
        return self

    def pca(self, n_components: int = 2, *, use_for_model: bool = False) -> WaterQualityPipeline:
        """Calculate exploratory PCA and optionally register PCA for modelling.

        The displayed PCA scores are exploratory and fitted on the selected
        dataset. When ``use_for_model=True``, a separate PCA transformer is also
        fitted independently inside every training partition/fold during model
        evaluation.
        """

        self._require_features()
        if n_components < 1:
            raise ValueError("n_components must be at least 1.")
        if n_components > min(self._X.shape):
            raise ValueError("n_components cannot exceed the number of samples or features.")

        exploratory = self._build_transformer(include_pca=True, pca_components=n_components)
        try:
            scores = exploratory.fit_transform(self._X)
        except ValueError as exc:
            raise ValueError(
                "PCA could not be fitted. Register imputation first when features contain missing values."
            ) from exc

        columns = [f"PC{i + 1}" for i in range(n_components)]
        scores_df = pd.DataFrame(scores, columns=columns, index=self._X.index)
        pca_model = exploratory.named_steps["pca"]
        loadings = pd.DataFrame(
            pca_model.components_.T,
            index=self.feature_columns,
            columns=columns,
        )
        self._pca_components = n_components
        self._pca_for_model = bool(use_for_model)
        self.results["pca"] = AnalysisResult(
            "pca",
            tables={"scores": scores_df, "loadings": loadings},
            metrics={
                "explained_variance_ratio": pca_model.explained_variance_ratio_.tolist(),
                "role": "predictive preprocessing" if use_for_model else "exploratory only",
                "predictive_fit_scope": (
                    "training partition/fold only" if use_for_model else "not used by predictive model"
                ),
            },
        )
        return self

    def train_random_forest(
        self,
        *,
        task: str = "classification",
        validation: str = "holdout",
        test_size: float = 0.25,
        random_state: int = 42,
        n_estimators: int = 300,
        n_splits: int = 5,
        group_column: str | None = None,
        time_column: str | None = None,
        quality_policy: str = "warn",
        quality_status_column: str = "Ion_Balance_Status",
        acceptable_statuses: Sequence[str] = ("acceptable",),
        bootstrap_iterations: int = 1000,
        result_name: str = "random_forest",
    ) -> WaterQualityPipeline:
        """Train and evaluate a leakage-safe Random Forest workflow.

        Parameters
        ----------
        validation:
            ``holdout``, ``stratified_kfold``, ``group_kfold``, or
            ``temporal_holdout``.
        quality_policy:
            ``warn`` (default), ``filter``, ``raise``, or ``ignore`` when an
            ion-balance status column is available.
        """

        self._require_features()
        if task not in {"classification", "regression"}:
            raise ValueError("task must be either 'classification' or 'regression'.")
        if validation not in {"holdout", "stratified_kfold", "group_kfold", "temporal_holdout"}:
            raise ValueError(
                "validation must be 'holdout', 'stratified_kfold', 'group_kfold', or 'temporal_holdout'."
            )
        if quality_policy not in {"warn", "filter", "raise", "ignore"}:
            raise ValueError("quality_policy must be 'warn', 'filter', 'raise', or 'ignore'.")
        if self._y is None:
            if self.target_column is None:
                raise ValueError("A target column is required for supervised modelling.")
            self._y = self.dataset.data[self.target_column]

        X = self._X.copy()
        y = self._y.copy()
        valid_target = y.notna()
        X = X.loc[valid_target]
        y = y.loc[valid_target]

        X, y, quality_meta = self._apply_quality_policy(
            X,
            y,
            policy=quality_policy,
            status_column=quality_status_column,
            acceptable_statuses=acceptable_statuses,
        )
        if X.empty:
            raise ValueError("No samples remain after target and quality-control filtering.")

        estimator = self._build_model_pipeline(
            task=task,
            n_estimators=n_estimators,
            random_state=random_state,
        )

        if validation == "holdout":
            evaluation = self._evaluate_holdout(
                estimator,
                X,
                y,
                task=task,
                test_size=test_size,
                random_state=random_state,
            )
        elif validation in {"stratified_kfold", "group_kfold"}:
            evaluation = self._evaluate_cross_validation(
                estimator,
                X,
                y,
                task=task,
                validation=validation,
                n_splits=n_splits,
                random_state=random_state,
                group_column=group_column,
            )
        else:
            evaluation = self._evaluate_temporal_holdout(
                estimator,
                X,
                y,
                task=task,
                test_size=test_size,
                time_column=time_column,
            )

        y_true = evaluation.pop("y_true")
        y_pred = evaluation.pop("y_pred")
        prediction_index = evaluation.pop("prediction_index")
        metrics = self._task_metrics(y_true, y_pred, task)
        bootstrap_groups = None
        if validation == "group_kfold" and group_column:
            bootstrap_groups = self.dataset.data.loc[prediction_index, group_column].to_numpy()
        metrics.update(
            self._bootstrap_confidence_intervals(
                y_true,
                y_pred,
                task=task,
                iterations=bootstrap_iterations,
                random_state=random_state,
                groups=bootstrap_groups,
            )
        )
        metrics["confidence_interval_method"] = (
            "cluster bootstrap by validation group"
            if bootstrap_groups is not None
            else "nonparametric bootstrap of out-of-sample predictions"
        )
        metrics.update(evaluation.pop("validation_metadata"))
        metrics.update(quality_meta)
        metrics["n_estimators"] = int(n_estimators)
        metrics["random_state"] = int(random_state)
        metrics["preprocessing"] = {
            "imputation": self._imputation_strategy,
            "scaling": self._scale_enabled,
            "pca_in_model": self._pca_for_model,
            "pca_components": self._pca_components if self._pca_for_model else None,
            "fit_scope": "training partition/fold only",
        }
        metrics.update(evaluation.pop("baseline_metrics", {}))

        predictions = pd.DataFrame(
            {"observed": np.asarray(y_true), "predicted": np.asarray(y_pred)},
            index=prediction_index,
        ).sort_index()
        tables: dict[str, pd.DataFrame] = {"predictions": predictions}
        tables.update(evaluation.pop("tables", {}))

        final_model = clone(estimator).fit(X, y)
        self.fitted_models[result_name] = final_model
        tables["feature_importance"] = self._feature_importance_table(final_model)

        result = AnalysisResult(
            result_name,
            tables=tables,
            metrics=metrics,
            figures=self._evaluation_figures(y_true, y_pred, task),
        )
        self.results[result_name] = result
        return self

    def export_html_report(self, output_path: str | Path) -> Path:
        self._require_dataset()
        return generate_water_quality_report(
            data=self.dataset.data,
            output_path=output_path,
            results=self.results,
            title="AI-Aquatica water quality report",
        )

    def export_artifacts(self, output_dir: str | Path) -> dict[str, Path]:
        """Export result metrics, tables, and figures to reproducible files."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, Path] = {}
        for result_name, result in self.results.items():
            safe_name = result_name.replace(" ", "_")
            metrics_path = output_dir / f"{safe_name}_metrics.json"
            metrics_path.write_text(
                json.dumps(result.metrics, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            artifacts[f"{result_name}:metrics"] = metrics_path
            result.artifacts["metrics"] = metrics_path
            for table_name, table in result.tables.items():
                if not isinstance(table, pd.DataFrame):
                    continue
                table_path = output_dir / f"{safe_name}_{table_name}.csv"
                table.to_csv(table_path, index=True)
                artifacts[f"{result_name}:{table_name}"] = table_path
                result.artifacts[table_name] = table_path
            for figure_name, figure in result.figures.items():
                figure_path = output_dir / f"{safe_name}_{figure_name}.png"
                figure.savefig(figure_path, dpi=200, bbox_inches="tight")
                artifacts[f"{result_name}:{figure_name}"] = figure_path
                result.artifacts[figure_name] = figure_path
        return artifacts

    def _build_transformer(
        self,
        *,
        include_pca: bool,
        pca_components: int | None = None,
    ) -> SklearnPipeline:
        steps: list[tuple[str, Any]] = []
        if self._imputation_strategy is not None:
            steps.append(("imputer", SimpleImputer(strategy=self._imputation_strategy)))
        if self._scale_enabled:
            steps.append(("scaler", StandardScaler()))
        if include_pca:
            steps.append(("pca", PCA(n_components=pca_components)))
        if not steps:
            # An identity-like transformer that preserves the Pipeline interface.
            from sklearn.preprocessing import FunctionTransformer

            steps.append(("identity", FunctionTransformer(validate=False)))
        return SklearnPipeline(steps)

    def _build_model_pipeline(
        self,
        *,
        task: str,
        n_estimators: int,
        random_state: int,
    ) -> SklearnPipeline:
        transformer = self._build_transformer(
            include_pca=self._pca_for_model,
            pca_components=self._pca_components,
        )
        steps = list(transformer.steps)
        if task == "classification":
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
            )
        steps.append(("model", model))
        return SklearnPipeline(steps)

    def _apply_quality_policy(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        policy: str,
        status_column: str,
        acceptable_statuses: Sequence[str],
    ) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
        metadata: dict[str, Any] = {
            "quality_policy": policy,
            "quality_status_column": status_column,
            "quality_filter_applied": False,
            "n_samples_before_quality_policy": int(len(X)),
        }
        if status_column not in self.dataset.data.columns:
            metadata["quality_status_available"] = False
            metadata["n_samples_after_quality_policy"] = int(len(X))
            return X, y, metadata

        metadata["quality_status_available"] = True
        status = self.dataset.data.loc[X.index, status_column].astype("string")
        counts = status.fillna("<missing>").value_counts().to_dict()
        metadata["quality_status_counts"] = {str(k): int(v) for k, v in counts.items()}
        accepted = status.isin(list(acceptable_statuses))
        n_not_accepted = int((~accepted).sum())
        metadata["n_samples_not_accepted"] = n_not_accepted

        if n_not_accepted:
            message = (
                f"{n_not_accepted} samples are not in accepted hydrochemical statuses "
                f"{list(acceptable_statuses)}."
            )
            if policy == "raise":
                raise ValueError(message)
            if policy == "warn":
                warnings.warn(
                    message + " They remain in the model because quality_policy='warn'.",
                    stacklevel=3,
                )
            if policy == "filter":
                X = X.loc[accepted]
                y = y.loc[accepted]
                metadata["quality_filter_applied"] = True

        metadata["n_samples_after_quality_policy"] = int(len(X))
        return X, y, metadata

    def _evaluate_holdout(
        self,
        estimator: SklearnPipeline,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        task: str,
        test_size: float,
        random_state: int,
    ) -> dict[str, Any]:
        stratify = y if task == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        model = clone(estimator).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        tables = self._evaluation_tables(y_test, y_pred, task)
        baseline = self._baseline_metrics(X_train, y_train, X_test, y_test, task)
        return {
            "y_true": y_test.to_numpy(),
            "y_pred": np.asarray(y_pred),
            "prediction_index": y_test.index,
            "tables": tables,
            "baseline_metrics": baseline,
            "validation_metadata": {
                "validation": "holdout",
                "test_size": float(test_size),
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
            },
        }

    def _evaluate_cross_validation(
        self,
        estimator: SklearnPipeline,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        task: str,
        validation: str,
        n_splits: int,
        random_state: int,
        group_column: str | None,
    ) -> dict[str, Any]:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        groups = None
        if validation == "group_kfold":
            if not group_column:
                raise ValueError("group_column is required for group_kfold validation.")
            self.dataset.validate_columns([group_column])
            groups = self.dataset.data.loc[X.index, group_column]
            if groups.isna().any():
                raise ValueError("group_column contains missing values.")
            if groups.nunique(dropna=True) < n_splits:
                raise ValueError("The number of unique groups must be at least n_splits.")
            splitter = GroupKFold(n_splits=n_splits)
            split_iterator = splitter.split(X, y, groups)
        else:
            if task != "classification":
                raise ValueError("stratified_kfold is available only for classification.")
            min_class = int(y.value_counts().min())
            if min_class < n_splits:
                raise ValueError("Each class must contain at least n_splits observations.")
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            split_iterator = splitter.split(X, y)

        predictions = pd.Series(index=X.index, dtype="object" if task == "classification" else "float64")
        fold_rows: list[dict[str, Any]] = []
        baseline_predictions = pd.Series(index=X.index, dtype="float64") if task == "regression" else None

        for fold, (train_idx, test_idx) in enumerate(split_iterator, start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            fold_model = clone(estimator).fit(X_train, y_train)
            fold_pred = fold_model.predict(X_test)
            predictions.loc[y_test.index] = fold_pred
            row = {"fold": fold, "n_train": len(train_idx), "n_test": len(test_idx)}
            row.update(self._task_metrics(y_test.to_numpy(), np.asarray(fold_pred), task))
            if groups is not None:
                row["test_groups"] = ",".join(map(str, sorted(pd.unique(groups.iloc[test_idx]))))
            fold_rows.append(row)

            if task == "regression":
                baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
                baseline_predictions.loc[y_test.index] = baseline.predict(X_test)

        if predictions.isna().any():
            raise RuntimeError("Cross-validation did not produce exactly one prediction for every sample.")
        y_pred = predictions.loc[y.index].to_numpy()
        tables = self._evaluation_tables(y, y_pred, task)
        tables["fold_metrics"] = pd.DataFrame(fold_rows)
        baseline_metrics: dict[str, Any] = {}
        if task == "regression" and baseline_predictions is not None:
            baseline_metrics = {
                f"baseline_{key}": value
                for key, value in self._task_metrics(
                    y.to_numpy(), baseline_predictions.loc[y.index].to_numpy(), task
                ).items()
            }
        return {
            "y_true": y.to_numpy(),
            "y_pred": y_pred,
            "prediction_index": y.index,
            "tables": tables,
            "baseline_metrics": baseline_metrics,
            "validation_metadata": {
                "validation": validation,
                "n_splits": int(n_splits),
                "group_column": group_column,
                "n_samples_evaluated": int(len(X)),
            },
        }

    def _evaluate_temporal_holdout(
        self,
        estimator: SklearnPipeline,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        task: str,
        test_size: float,
        time_column: str | None,
    ) -> dict[str, Any]:
        if not time_column:
            raise ValueError("time_column is required for temporal_holdout validation.")
        self.dataset.validate_columns([time_column])
        time_values = self.dataset.data.loc[X.index, time_column]
        if time_values.isna().any():
            raise ValueError("time_column contains missing values.")
        unique_times = np.array(sorted(pd.unique(time_values)))
        if unique_times.size < 2:
            raise ValueError("At least two unique time values are required.")
        n_test_times = max(1, int(np.ceil(unique_times.size * test_size)))
        test_times = set(unique_times[-n_test_times:])
        test_mask = time_values.isin(test_times)
        train_mask = ~test_mask
        X_train, X_test = X.loc[train_mask], X.loc[test_mask]
        y_train, y_test = y.loc[train_mask], y.loc[test_mask]
        if X_train.empty or X_test.empty:
            raise ValueError("Temporal split produced an empty training or test set.")
        model = clone(estimator).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        tables = self._evaluation_tables(y_test, y_pred, task)
        baseline = self._baseline_metrics(X_train, y_train, X_test, y_test, task)
        return {
            "y_true": y_test.to_numpy(),
            "y_pred": np.asarray(y_pred),
            "prediction_index": y_test.index,
            "tables": tables,
            "baseline_metrics": baseline,
            "validation_metadata": {
                "validation": "temporal_holdout",
                "time_column": time_column,
                "test_size": float(test_size),
                "train_time_min": str(time_values.loc[train_mask].min()),
                "train_time_max": str(time_values.loc[train_mask].max()),
                "test_time_min": str(time_values.loc[test_mask].min()),
                "test_time_max": str(time_values.loc[test_mask].max()),
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
            },
        }

    def _baseline_metrics(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task: str,
    ) -> dict[str, Any]:
        if task != "regression":
            return {}
        baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
        pred = baseline.predict(X_test)
        return {f"baseline_{key}": value for key, value in self._task_metrics(y_test, pred, task).items()}

    @staticmethod
    def _task_metrics(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> dict[str, float]:
        if task == "classification":
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(
                    recall_score(
                        y_true,
                        y_pred,
                        labels=sorted(pd.unique(pd.Series(y_true))),
                        average="macro",
                        zero_division=0,
                    )
                ),
                "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
                "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            }
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": rmse,
        }

    @staticmethod
    def _evaluation_figures(
        y_true: Sequence[Any],
        y_pred: Sequence[Any],
        task: str,
    ) -> dict[str, Any]:
        if task == "classification":
            labels = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)], ignore_index=True)))
            matrix = confusion_matrix(y_true, y_pred, labels=labels)
            fig, ax = plt.subplots(figsize=(6, 5))
            image = ax.imshow(matrix)
            fig.colorbar(image, ax=ax)
            ax.set_xticks(range(len(labels)), labels=labels)
            ax.set_yticks(range(len(labels)), labels=labels)
            ax.set_xlabel("Predicted station")
            ax.set_ylabel("Observed station")
            ax.set_title("Out-of-sample confusion matrix")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
            fig.tight_layout()
            return {"confusion_matrix": fig}

        observed = np.asarray(y_true, dtype=float)
        predicted = np.asarray(y_pred, dtype=float)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(observed, predicted, alpha=0.75)
        minimum = float(np.nanmin([observed.min(), predicted.min()]))
        maximum = float(np.nanmax([observed.max(), predicted.max()]))
        ax.plot([minimum, maximum], [minimum, maximum], linestyle="--")
        ax.set_xlabel("Observed chlorophyll-a")
        ax.set_ylabel("Predicted chlorophyll-a")
        ax.set_title("Observed versus out-of-sample predicted values")
        fig.tight_layout()
        return {"observed_vs_predicted": fig}

    @staticmethod
    def _evaluation_tables(y_true: Sequence[Any], y_pred: Sequence[Any], task: str) -> dict[str, pd.DataFrame]:
        if task == "classification":
            labels = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)], ignore_index=True)))
            report = classification_report(
                y_true,
                y_pred,
                labels=labels,
                output_dict=True,
                zero_division=0,
            )
            report_df = pd.DataFrame(report).T
            matrix = pd.DataFrame(
                confusion_matrix(y_true, y_pred, labels=labels),
                index=[f"actual_{label}" for label in labels],
                columns=[f"predicted_{label}" for label in labels],
            )
            return {"classification_report": report_df, "confusion_matrix": matrix}
        residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
        return {
            "residual_summary": pd.DataFrame(
                {
                    "metric": ["mean", "std", "median", "min", "max"],
                    "value": [
                        float(np.mean(residuals)),
                        float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0,
                        float(np.median(residuals)),
                        float(np.min(residuals)),
                        float(np.max(residuals)),
                    ],
                }
            )
        }

    def _feature_importance_table(self, fitted_pipeline: SklearnPipeline) -> pd.DataFrame:
        model = fitted_pipeline.named_steps["model"]
        if self._pca_for_model and self._pca_components:
            names = [f"PC{i + 1}" for i in range(self._pca_components)]
        else:
            names = self.feature_columns
        return pd.DataFrame(
            {"feature": names, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False, ignore_index=True)

    @classmethod
    def _bootstrap_confidence_intervals(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        task: str,
        iterations: int,
        random_state: int,
        groups: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Estimate metric confidence intervals from out-of-sample predictions.

        When validation groups are supplied, whole groups are resampled with
        replacement (cluster bootstrap). This preserves within-campaign
        dependence in grouped environmental monitoring data.
        """

        if iterations <= 0 or len(y_true) < 2:
            return {}
        rng = np.random.default_rng(random_state)
        keys = list(cls._task_metrics(y_true, y_pred, task).keys())
        values: dict[str, list[float]] = {key: [] for key in keys}
        n = len(y_true)

        group_array: np.ndarray | None = None
        unique_groups: np.ndarray | None = None
        group_indices: dict[Any, np.ndarray] | None = None
        if groups is not None:
            group_array = np.asarray(groups)
            if len(group_array) != n:
                raise ValueError("groups must have the same length as y_true and y_pred.")
            unique_groups = np.asarray(pd.unique(pd.Series(group_array)))
            if unique_groups.size < 2:
                group_array = None
                unique_groups = None
            else:
                group_indices = {
                    group: np.flatnonzero(group_array == group)
                    for group in unique_groups
                }

        for _ in range(iterations):
            if unique_groups is not None and group_indices is not None:
                sampled_groups = rng.choice(
                    unique_groups,
                    size=len(unique_groups),
                    replace=True,
                )
                idx = np.concatenate([group_indices[group] for group in sampled_groups])
            else:
                idx = rng.integers(0, n, size=n)
            try:
                sample = cls._task_metrics(y_true[idx], y_pred[idx], task)
            except ValueError:
                continue
            for key, value in sample.items():
                if np.isfinite(value):
                    values[key].append(value)
        intervals: dict[str, float] = {}
        for key, samples in values.items():
            if samples:
                intervals[f"{key}_ci95_low"] = float(np.percentile(samples, 2.5))
                intervals[f"{key}_ci95_high"] = float(np.percentile(samples, 97.5))
        return intervals

    def _require_dataset(self) -> None:
        if self.dataset is None:
            raise ValueError("Load a dataset first using from_csv or from_dataframe.")

    def _require_features(self) -> None:
        self._require_dataset()
        if self._X is None or not self.feature_columns:
            raise ValueError("Select feature columns first using select_features().")


__all__ = ["WaterQualityPipeline", "WaterQualityDataset", "AnalysisResult"]
