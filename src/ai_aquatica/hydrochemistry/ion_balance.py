"""Hydrochemical ion-balance utilities.

The module implements charge-balance calculations for laboratory and monitoring
quality control. Concentrations may be provided in ``meq/L`` or ``mg/L``. In
``mg/L`` mode, values are converted using explicit equivalent weights.

Rows with missing, non-numeric, infinite, negative, or zero-total ion data are
marked ``indeterminate`` rather than being assigned an artificial 0% error.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Default mass-per-charge conversion factors in mg/meq for concentrations reported as the ion itself.
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

# When NO3, NO2, or NH4 are reported as mg N/L rather than mg ion/L, one
# milliequivalent of each monovalent species contains one millimole of N.
# Use these reporting-basis factors through ``equivalent_weights``.
NITROGEN_AS_N_MASS_PER_MEQ: dict[str, float] = {
    "NH4": 14.0067,
    "NO2": 14.0067,
    "NO3": 14.0067,
}


@dataclass(frozen=True)
class IonBalanceConfig:
    """Configuration for charge-balance quality control.

    Parameters
    ----------
    cations, anions:
        Names of columns containing selected cation and anion concentrations.
    units:
        ``"meq/L"`` or ``"mg/L"``. In ``mg/L`` mode, values are divided by
        equivalent weights in mg/meq.
    threshold:
        Absolute charge-balance error (%) above which a complete, evaluable row
        is assigned the status ``review``.
    equivalent_weights:
        Optional user-supplied mass-per-meq conversion factors that extend or
        override the built-in ion-based catalogue. This also supports analytes
        reported on an elemental basis, for example NO3-N, NO2-N, or NH4-N in
        mg N/L.
    require_complete:
        When True (default), every selected ion must contain a valid value for a
        row to receive a numerical charge-balance error.
    """

    cations: Sequence[str]
    anions: Sequence[str]
    units: str = "meq/L"
    threshold: float = 5.0
    equivalent_weights: Mapping[str, float] = field(default_factory=dict)
    require_complete: bool = True

    def weights(self) -> dict[str, float]:
        merged = DEFAULT_EQUIVALENT_WEIGHTS.copy()
        merged.update(dict(self.equivalent_weights))
        return merged


@dataclass
class IonBalanceSummary:
    """Summary statistics for an ion-balance run."""

    n_samples: int
    n_evaluable: int
    n_indeterminate: int
    n_flagged: int
    threshold: float
    mean_abs_error: float
    median_abs_error: float
    max_abs_error: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n_samples": self.n_samples,
            "n_evaluable": self.n_evaluable,
            "n_indeterminate": self.n_indeterminate,
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


def _numeric_diagnostics(
    data: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Return numeric data and row-level validity masks."""

    selected = data.loc[:, list(columns)]
    numeric = selected.apply(pd.to_numeric, errors="coerce").astype(float)
    non_numeric = selected.notna() & numeric.isna()
    infinite = pd.DataFrame(
        np.isinf(numeric.to_numpy(dtype=float)),
        index=numeric.index,
        columns=numeric.columns,
    )
    negative = numeric.lt(0)
    invalid_cells = non_numeric | infinite | negative
    cleaned = numeric.mask(invalid_cells)
    missing_any = cleaned.isna().any(axis=1)
    invalid_any = invalid_cells.any(axis=1)
    complete = ~(missing_any | invalid_any)
    return cleaned, invalid_cells, complete, missing_any, invalid_any


def _as_numeric_frame(data: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    _validate_columns(data, columns)
    numeric, _, _, _, _ = _numeric_diagnostics(data, columns)
    if numeric.isna().all(axis=None):
        raise ValueError("Ion concentration columns do not contain valid numeric values.")
    return numeric


def concentrations_to_meq(
    data: pd.DataFrame,
    ions: Sequence[str],
    equivalent_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Convert selected reported concentrations from mg/L to meq/L.

    By default, concentrations are assumed to be reported as the ion itself.
    Supply ``equivalent_weights`` to override the mass-per-meq factor when a
    laboratory reports a constituent on an elemental basis (e.g. mg N/L).
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
    invalid_weights = [ion for ion in ions if not np.isfinite(weights[ion]) or weights[ion] <= 0]
    if invalid_weights:
        raise ValueError(f"Equivalent weights must be finite and positive for: {invalid_weights}")

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
    """Calculate charge-balance error and explicit diagnostic status.

    The charge-balance error is

    ``CBE (%) = 100 * (Σ cations - Σ anions) / (Σ cations + Σ anions)``

    where all ion concentrations are expressed in milliequivalents per litre.
    A numerical CBE is returned only for complete, valid rows with a positive
    total ionic charge. Other rows receive ``NaN`` and ``indeterminate``.
    """

    units_normalized = config.units.lower().replace(" ", "")
    if units_normalized not in {"meq/l", "mg/l"}:
        raise ValueError("units must be either 'meq/L' or 'mg/L'.")
    if not config.cations or not config.anions:
        raise ValueError("At least one cation and one anion column are required.")
    if not np.isfinite(config.threshold) or config.threshold < 0:
        raise ValueError("threshold must be a finite, non-negative percentage.")

    all_ions = list(config.cations) + list(config.anions)
    _validate_columns(data, all_ions)
    result = data.copy() if copy else data

    cations_raw, cation_invalid_cells, cation_complete, cation_missing, cation_invalid = (
        _numeric_diagnostics(result, config.cations)
    )
    anions_raw, anion_invalid_cells, anion_complete, anion_missing, anion_invalid = (
        _numeric_diagnostics(result, config.anions)
    )

    if units_normalized == "mg/l":
        weights = config.weights()
        missing_weights = [ion for ion in all_ions if ion not in weights]
        if missing_weights:
            raise KeyError(
                "Equivalent weights are missing for ion columns: "
                f"{missing_weights}. Provide them through equivalent_weights."
            )
        invalid_weights = [
            ion for ion in all_ions if not np.isfinite(weights[ion]) or weights[ion] <= 0
        ]
        if invalid_weights:
            raise ValueError(f"Equivalent weights must be finite and positive for: {invalid_weights}")
        cations_meq = cations_raw.copy()
        anions_meq = anions_raw.copy()
        for ion in config.cations:
            cations_meq[ion] = cations_raw[ion] / weights[ion]
        for ion in config.anions:
            anions_meq[ion] = anions_raw[ion] / weights[ion]
    else:
        cations_meq = cations_raw
        anions_meq = anions_raw

    if config.require_complete:
        row_complete = cation_complete & anion_complete
        cation_sum = cations_meq.sum(axis=1, min_count=len(config.cations))
        anion_sum = anions_meq.sum(axis=1, min_count=len(config.anions))
    else:
        row_complete = ~(cation_invalid | anion_invalid)
        cation_sum = cations_meq.sum(axis=1, min_count=1)
        anion_sum = anions_meq.sum(axis=1, min_count=1)

    denominator = cation_sum + anion_sum
    evaluable = row_complete & denominator.notna() & denominator.gt(0)

    charge_balance = pd.Series(np.nan, index=result.index, dtype="float64")
    charge_balance.loc[evaluable] = (
        (cation_sum.loc[evaluable] - anion_sum.loc[evaluable])
        / denominator.loc[evaluable]
        * 100.0
    )

    status = pd.Series("indeterminate", index=result.index, dtype="string")
    status.loc[evaluable & charge_balance.abs().le(config.threshold)] = "acceptable"
    status.loc[evaluable & charge_balance.abs().gt(config.threshold)] = "review"

    potential_error = pd.Series(pd.NA, index=result.index, dtype="boolean")
    potential_error.loc[evaluable] = charge_balance.loc[evaluable].abs().gt(config.threshold)

    reason = pd.Series("", index=result.index, dtype="string")
    missing_rows = cation_missing | anion_missing
    invalid_rows = cation_invalid | anion_invalid
    zero_rows = row_complete & denominator.fillna(0).eq(0)
    reason.loc[missing_rows] = "missing_selected_ion"
    reason.loc[invalid_rows] = "invalid_non_numeric_infinite_or_negative_value"
    reason.loc[zero_rows] = "zero_total_ionic_charge"
    reason.loc[status.eq("acceptable")] = "within_threshold"
    reason.loc[status.eq("review")] = "absolute_cbe_above_threshold"

    result["Cations_Sum_meq_L"] = cation_sum
    result["Anions_Sum_meq_L"] = anion_sum
    # Backward-compatible aliases.
    result["Cations_Sum"] = cation_sum
    result["Anions_Sum"] = anion_sum
    result["Ion_Balance"] = charge_balance
    result["Charge_Balance_Error_pct"] = charge_balance
    result["Ion_Set_Complete"] = row_complete.astype(bool)
    result["Ion_Invalid_Cell_Count"] = (
        cation_invalid_cells.sum(axis=1) + anion_invalid_cells.sum(axis=1)
    ).astype(int)
    result["Potential_Error"] = potential_error
    result["Ion_Balance_Status"] = status
    result["Ion_Balance_Diagnostic"] = reason
    return result


def summarize_ion_balance(
    data: pd.DataFrame,
    balance_column: str = "Ion_Balance",
    flag_column: str = "Potential_Error",
    threshold: float = 5.0,
) -> IonBalanceSummary:
    """Summarise evaluable and indeterminate charge-balance diagnostics."""

    if balance_column not in data.columns:
        raise KeyError(f"Column '{balance_column}' is required.")
    errors = pd.to_numeric(data[balance_column], errors="coerce").abs()
    evaluable = errors.notna()
    status = data.get("Ion_Balance_Status")
    if status is not None:
        n_indeterminate = int(pd.Series(status).astype("string").eq("indeterminate").sum())
        n_flagged = int(pd.Series(status).astype("string").eq("review").sum())
    else:
        flags = data.get(flag_column, errors > threshold)
        n_indeterminate = int((~evaluable).sum())
        n_flagged = int(pd.Series(flags).fillna(False).astype(bool).sum())
    valid_errors = errors.loc[evaluable]
    if valid_errors.empty:
        mean_abs = median_abs = max_abs = float("nan")
    else:
        mean_abs = float(valid_errors.mean())
        median_abs = float(valid_errors.median())
        max_abs = float(valid_errors.max())
    return IonBalanceSummary(
        n_samples=int(len(data)),
        n_evaluable=int(evaluable.sum()),
        n_indeterminate=n_indeterminate,
        n_flagged=n_flagged,
        threshold=float(threshold),
        mean_abs_error=mean_abs,
        median_abs_error=median_abs,
        max_abs_error=max_abs,
    )


def correct_by_proportional_scaling(
    data: pd.DataFrame,
    cations: Sequence[str],
    anions: Sequence[str],
    *,
    balance_column: str = "Ion_Balance",
) -> pd.DataFrame:
    """Return a non-destructive proportional balancing sensitivity analysis.

    This is a computational sensitivity tool, not a replacement for laboratory
    quality control. Original measurements are preserved and corrected values
    are written to ``*_corrected`` columns.
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
    result["Cations_Sum_corrected"] = result[corrected_cations].sum(axis=1, min_count=1)
    result["Anions_Sum_corrected"] = result[corrected_anions].sum(axis=1, min_count=1)
    denominator = result["Cations_Sum_corrected"] + result["Anions_Sum_corrected"]
    result["Ion_Balance_corrected"] = np.where(
        denominator.ne(0),
        (result["Cations_Sum_corrected"] - result["Anions_Sum_corrected"])
        / denominator
        * 100.0,
        0.0,
    )
    return result


def bicarbonate_from_alkalinity(
    alkalinity: pd.Series | pd.DataFrame,
    *,
    units: str = "mg_CaCO3_L",
) -> pd.Series | pd.DataFrame:
    """Convert alkalinity to bicarbonate concentration.

    For alkalinity in mg CaCO3/L, the conversion is
    ``HCO3 (mg/L) = alkalinity * 61.0168 / 50.043``.
    For alkalinity in meq/L, input is multiplied by 61.0168 mg/meq.
    Invalid, negative, or infinite values are returned as missing values and are
    later marked indeterminate by the charge-balance calculation.
    """

    numeric = pd.to_numeric(alkalinity, errors="coerce").astype(float)
    numeric = numeric.mask(~np.isfinite(numeric) | numeric.lt(0))
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
    """Add a bicarbonate column estimated from alkalinity."""

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
    """Assess column availability and common major-ion completeness."""

    requested = list(cations) + list(anions)
    if alkalinity_col:
        requested.append(alkalinity_col)
    derivable_hco3 = bool(
        alkalinity_col and alkalinity_col in data.columns and "HCO3" in requested
    )
    available = [col for col in requested if col in data.columns]
    if derivable_hco3 and "HCO3" not in available:
        available.append("HCO3 (derived from alkalinity)")
    missing = [
        col
        for col in requested
        if col not in data.columns and not (col == "HCO3" and derivable_hco3)
    ]
    major_cations = {"Ca", "Mg", "Na", "K"}
    major_anions = {"HCO3", "Cl", "SO4"}
    expected_major = sorted(major_cations | major_anions)
    present_major = sorted([ion for ion in expected_major if ion in data.columns])
    if alkalinity_col and alkalinity_col in data.columns and "HCO3" not in present_major:
        present_major.append("HCO3 (derived from alkalinity)")
    missing_major = sorted(
        [
            ion
            for ion in expected_major
            if ion not in data.columns
            and not (ion == "HCO3" and alkalinity_col and alkalinity_col in data.columns)
        ]
    )
    notes: list[str] = []
    if missing:
        notes.append("Some requested ion or alkalinity columns are missing.")
    if missing_major:
        notes.append(
            "The dataset lacks one or more common major ions; charge-balance results must be interpreted as diagnostics of the available ion set."
        )
    if alkalinity_col:
        notes.append(
            "Bicarbonate is derived from alkalinity; alkalinity units must be verified before interpretation."
        )

    selected_present = [column for column in requested if column in data.columns]
    if selected_present:
        numeric = data[selected_present].apply(pd.to_numeric, errors="coerce")
        row_complete_fraction = float(numeric.notna().all(axis=1).mean())
        negative_count = int(numeric.lt(0).sum().sum())
        infinite_count = int(np.isinf(numeric.to_numpy(dtype=float)).sum())
    else:
        row_complete_fraction = 0.0
        negative_count = 0
        infinite_count = 0

    return {
        "requested_columns": requested,
        "available_columns": available,
        "missing_columns": missing,
        "expected_major_ions": expected_major,
        "present_major_ions": present_major,
        "missing_major_ions": missing_major,
        "column_readiness_score": round(len(available) / len(requested), 3) if requested else 0.0,
        "readiness_score": round(len(available) / len(requested), 3) if requested else 0.0,
        "complete_row_fraction_for_available_requested_columns": round(row_complete_fraction, 3),
        "negative_value_count": negative_count,
        "infinite_value_count": infinite_count,
        "notes": notes,
    }
