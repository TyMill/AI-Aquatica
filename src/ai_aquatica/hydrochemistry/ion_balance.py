"""Hydrochemical ion-balance utilities.

The module implements charge-balance calculations used for quality control of
water chemistry analyses.  Concentrations can be supplied either as
milliequivalents per litre (``meq/L``) or as milligrams per litre (``mg/L``),
provided that ion names are available in the equivalent-weight catalogue or are
supplied by the user.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

# Equivalent weights in mg/meq. Values follow molar mass / absolute ionic charge.
DEFAULT_EQUIVALENT_WEIGHTS: dict[str, float] = {
    "H": 1.0079,
    "Li": 6.941,
    "Na": 22.9898,
    "K": 39.0983,
    "NH4": 18.039,
    "Mg": 12.1525,
    "Ca": 20.039,
    "Sr": 43.81,
    "Ba": 68.665,
    "Fe2": 27.9225,
    "Mn2": 27.469,
    "Al": 8.994,
    "F": 18.998,
    "Cl": 35.453,
    "Br": 79.904,
    "NO2": 46.0055,
    "NO3": 62.0049,
    "HCO3": 61.0168,
    "CO3": 30.0045,
    "SO4": 48.03,
    "PO4": 31.657,
    "HPO4": 47.984,
    "OH": 17.007,
}


@dataclass(frozen=True)
class IonBalanceConfig:
    """Configuration for charge-balance quality control.

    Parameters
    ----------
    cations, anions:
        Names of columns containing major cation and anion concentrations.
    units:
        Either ``"meq/L"`` or ``"mg/L"``. In ``mg/L`` mode, concentrations are
        converted to milliequivalents using equivalent weights.
    threshold:
        Absolute charge-balance error (%) above which the analysis is flagged.
    equivalent_weights:
        Optional user-supplied equivalent weights in mg/meq. These values extend
        or override the built-in catalogue.
    """

    cations: Sequence[str]
    anions: Sequence[str]
    units: str = "meq/L"
    threshold: float = 5.0
    equivalent_weights: Mapping[str, float] = field(default_factory=dict)

    def weights(self) -> dict[str, float]:
        merged = DEFAULT_EQUIVALENT_WEIGHTS.copy()
        merged.update(dict(self.equivalent_weights))
        return merged


@dataclass
class IonBalanceSummary:
    """Summary statistics for an ion-balance run."""

    n_samples: int
    n_flagged: int
    threshold: float
    mean_abs_error: float
    median_abs_error: float
    max_abs_error: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n_samples": self.n_samples,
            "n_flagged": self.n_flagged,
            "threshold": self.threshold,
            "mean_abs_error": self.mean_abs_error,
            "median_abs_error": self.median_abs_error,
            "max_abs_error": self.max_abs_error,
        }


def _validate_columns(data: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise KeyError(f"Missing required ion columns: {missing}")


def _as_numeric_frame(data: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    frame = data.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    if frame.isna().all(axis=None):
        raise ValueError("Ion concentration columns do not contain numeric values.")
    return frame


def concentrations_to_meq(
    data: pd.DataFrame,
    ions: Sequence[str],
    equivalent_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Convert selected ion concentrations from mg/L to meq/L.

    Parameters
    ----------
    data:
        DataFrame containing ion concentration columns.
    ions:
        Column names to convert.
    equivalent_weights:
        Mapping from ion/column name to equivalent weight in mg/meq.

    Returns
    -------
    pandas.DataFrame
        Numeric DataFrame with converted concentrations.
    """

    _validate_columns(data, ions)
    weights = DEFAULT_EQUIVALENT_WEIGHTS.copy()
    if equivalent_weights is not None:
        weights.update(dict(equivalent_weights))

    missing_weights = [ion for ion in ions if ion not in weights]
    if missing_weights:
        raise KeyError(
            "Equivalent weights are missing for ion columns: "
            f"{missing_weights}. Provide them through equivalent_weights."
        )

    numeric = _as_numeric_frame(data, ions)
    converted = numeric.copy()
    for ion in ions:
        converted[ion] = numeric[ion] / weights[ion]
    return converted


def calculate_charge_balance(
    data: pd.DataFrame,
    config: IonBalanceConfig,
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """Calculate charge-balance error and diagnostic columns.

    The charge-balance error is calculated as:

    ``100 * (sum_cations - sum_anions) / (sum_cations + sum_anions)``

    where ion sums are expressed in milliequivalents per litre.
    """

    units_normalized = config.units.lower().replace(" ", "")
    if units_normalized not in {"meq/l", "mg/l"}:
        raise ValueError("units must be either 'meq/L' or 'mg/L'.")

    all_ions = list(config.cations) + list(config.anions)
    _validate_columns(data, all_ions)
    result = data.copy() if copy else data

    if units_normalized == "mg/l":
        cations_meq = concentrations_to_meq(result, config.cations, config.weights())
        anions_meq = concentrations_to_meq(result, config.anions, config.weights())
    else:
        cations_meq = _as_numeric_frame(result, config.cations)
        anions_meq = _as_numeric_frame(result, config.anions)

    cation_sum = cations_meq.sum(axis=1, skipna=True)
    anion_sum = anions_meq.sum(axis=1, skipna=True)
    denominator = cation_sum + anion_sum

    charge_balance = pd.Series(np.nan, index=result.index, dtype="float64")
    valid = denominator.ne(0) & denominator.notna()
    charge_balance.loc[valid] = (
        (cation_sum.loc[valid] - anion_sum.loc[valid]) / denominator.loc[valid] * 100.0
    )
    charge_balance.loc[denominator.eq(0)] = 0.0

    result["Cations_Sum_meq_L"] = cation_sum
    result["Anions_Sum_meq_L"] = anion_sum
    # Backward-compatible aliases used by the original API/tests.
    result["Cations_Sum"] = cation_sum
    result["Anions_Sum"] = anion_sum
    result["Ion_Balance"] = charge_balance
    result["Charge_Balance_Error_pct"] = charge_balance
    result["Potential_Error"] = charge_balance.abs() > config.threshold
    result["Ion_Balance_Status"] = np.where(
        result["Potential_Error"], "review", "acceptable"
    )
    return result


def summarize_ion_balance(
    data: pd.DataFrame,
    balance_column: str = "Ion_Balance",
    flag_column: str = "Potential_Error",
    threshold: float = 5.0,
) -> IonBalanceSummary:
    """Summarise charge-balance diagnostics."""

    if balance_column not in data.columns:
        raise KeyError(f"Column '{balance_column}' is required.")
    errors = pd.to_numeric(data[balance_column], errors="coerce").abs().dropna()
    if errors.empty:
        return IonBalanceSummary(0, 0, threshold, float("nan"), float("nan"), float("nan"))
    n_flagged = int(data.get(flag_column, errors > threshold).sum())
    return IonBalanceSummary(
        n_samples=int(errors.shape[0]),
        n_flagged=n_flagged,
        threshold=float(threshold),
        mean_abs_error=float(errors.mean()),
        median_abs_error=float(errors.median()),
        max_abs_error=float(errors.max()),
    )


def correct_by_proportional_scaling(
    data: pd.DataFrame,
    cations: Sequence[str],
    anions: Sequence[str],
    *,
    balance_column: str = "Ion_Balance",
) -> pd.DataFrame:
    """Return a diagnostic correction that proportionally balances ion sums.

    This is intended as a computational sensitivity tool, not as an automatic
    replacement for laboratory quality control. The original measurements are
    not overwritten; corrected values are written to ``*_corrected`` columns.
    """

    _validate_columns(data, list(cations) + list(anions))
    if balance_column not in data.columns:
        raise KeyError("Run calculate_charge_balance before correction.")

    result = data.copy()
    cation_sum = pd.to_numeric(result["Cations_Sum"], errors="coerce")
    anion_sum = pd.to_numeric(result["Anions_Sum"], errors="coerce")
    target = (cation_sum + anion_sum) / 2.0

    cation_factor = pd.Series(1.0, index=result.index)
    anion_factor = pd.Series(1.0, index=result.index)
    cation_valid = cation_sum.ne(0) & cation_sum.notna()
    anion_valid = anion_sum.ne(0) & anion_sum.notna()
    cation_factor.loc[cation_valid] = target.loc[cation_valid] / cation_sum.loc[cation_valid]
    anion_factor.loc[anion_valid] = target.loc[anion_valid] / anion_sum.loc[anion_valid]

    for column in cations:
        result[f"{column}_corrected"] = pd.to_numeric(result[column], errors="coerce") * cation_factor
    for column in anions:
        result[f"{column}_corrected"] = pd.to_numeric(result[column], errors="coerce") * anion_factor

    corrected_cations = [f"{column}_corrected" for column in cations]
    corrected_anions = [f"{column}_corrected" for column in anions]
    result["Cations_Sum_corrected"] = result[corrected_cations].sum(axis=1)
    result["Anions_Sum_corrected"] = result[corrected_anions].sum(axis=1)
    denominator = result["Cations_Sum_corrected"] + result["Anions_Sum_corrected"]
    result["Ion_Balance_corrected"] = np.where(
        denominator.ne(0),
        (result["Cations_Sum_corrected"] - result["Anions_Sum_corrected"]) / denominator * 100.0,
        0.0,
    )
    return result


def bicarbonate_from_alkalinity(
    alkalinity: pd.Series | pd.DataFrame,
    *,
    units: str = "mg_CaCO3_L",
) -> pd.Series | pd.DataFrame:
    """Convert alkalinity to bicarbonate concentration for ion-balance checks.

    Parameters
    ----------
    alkalinity:
        Alkalinity values. For most monitoring datasets this is reported as
        mg CaCO3/L.
    units:
        Supported values are ``"mg_CaCO3_L"`` and ``"meq_L"``. In the first
        case the conversion uses 50.043 mg CaCO3 per meq and 61.0168 mg HCO3
        per meq. In the second case input alkalinity is already in meq/L and is
        converted to mg HCO3/L.
    """

    numeric = pd.to_numeric(alkalinity, errors="coerce")
    units_normalized = units.lower().replace("/", "_").replace(" ", "")
    if units_normalized in {"mg_caco3_l", "mgcaco3_l", "mgcaco3l"}:
        return numeric * (DEFAULT_EQUIVALENT_WEIGHTS["HCO3"] / 50.043)
    if units_normalized in {"meq_l", "meql"}:
        return numeric * DEFAULT_EQUIVALENT_WEIGHTS["HCO3"]
    raise ValueError("Supported alkalinity units are 'mg_CaCO3_L' and 'meq_L'.")


def add_bicarbonate_from_alkalinity(
    data: pd.DataFrame,
    alkalinity_col: str = "Alkalinity",
    *,
    output_col: str = "HCO3",
    alkalinity_units: str = "mg_CaCO3_L",
    overwrite: bool = False,
    copy: bool = True,
) -> pd.DataFrame:
    """Add a bicarbonate column estimated from alkalinity.

    The original alkalinity column is preserved. The derived bicarbonate column
    is intended for charge-balance diagnostics when direct bicarbonate analyses
    are not available.
    """

    if alkalinity_col not in data.columns:
        raise KeyError(f"Column '{alkalinity_col}' is required to derive bicarbonate.")
    result = data.copy() if copy else data
    if output_col in result.columns and not overwrite:
        return result
    result[output_col] = bicarbonate_from_alkalinity(
        result[alkalinity_col], units=alkalinity_units
    )
    return result


def calculate_charge_balance_from_alkalinity(
    data: pd.DataFrame,
    *,
    cations: Sequence[str],
    anions: Sequence[str] | None = None,
    alkalinity_col: str = "Alkalinity",
    alkalinity_units: str = "mg_CaCO3_L",
    bicarbonate_col: str = "HCO3",
    units: str = "mg/L",
    threshold: float = 5.0,
    equivalent_weights: Mapping[str, float] | None = None,
    copy: bool = True,
) -> pd.DataFrame:
    """Calculate ion balance after deriving HCO3 from alkalinity."""

    result = add_bicarbonate_from_alkalinity(
        data,
        alkalinity_col=alkalinity_col,
        output_col=bicarbonate_col,
        alkalinity_units=alkalinity_units,
        overwrite=True,
        copy=copy,
    )
    final_anions = list(anions or [])
    if bicarbonate_col not in final_anions:
        final_anions.append(bicarbonate_col)
    config = IonBalanceConfig(
        cations=list(cations),
        anions=final_anions,
        units=units,
        threshold=threshold,
        equivalent_weights=equivalent_weights or {},
    )
    return calculate_charge_balance(result, config, copy=False)


def assess_ion_balance_inputs(
    data: pd.DataFrame,
    *,
    cations: Sequence[str],
    anions: Sequence[str],
    alkalinity_col: str | None = None,
) -> dict[str, object]:
    """Assess whether a dataset contains enough ions for charge-balance QC.

    The function does not perform laboratory validation. It provides a transparent
    pre-flight diagnostic that helps users understand why a charge-balance check
    may be incomplete, for example because sodium or potassium were not measured.
    This is useful for real monitoring datasets where only a subset of major ions
    is available.
    """

    requested = list(cations) + list(anions)
    if alkalinity_col:
        requested.append(alkalinity_col)
    available = [col for col in requested if col in data.columns]
    missing = [col for col in requested if col not in data.columns]
    major_cations = {"Ca", "Mg", "Na", "K"}
    major_anions = {"HCO3", "Cl", "SO4"}
    expected_major = sorted(major_cations | major_anions)
    present_major = sorted([ion for ion in expected_major if ion in data.columns or ion in requested])
    missing_major = sorted([ion for ion in expected_major if ion not in data.columns and ion not in requested])
    notes: list[str] = []
    if missing:
        notes.append("Some requested ion/alkalinity columns are missing.")
    if missing_major:
        notes.append(
            "The dataset may not include all common major ions; charge-balance errors should be interpreted as diagnostic flags rather than automatic data rejection."
        )
    if alkalinity_col:
        notes.append(
            "Bicarbonate will be derived from alkalinity; verify alkalinity units before interpreting charge-balance results."
        )
    return {
        "requested_columns": requested,
        "available_columns": available,
        "missing_columns": missing,
        "expected_major_ions": expected_major,
        "present_major_ions": present_major,
        "missing_major_ions": missing_major,
        "readiness_score": round(len(available) / len(requested), 3) if requested else 0.0,
        "notes": notes,
    }
