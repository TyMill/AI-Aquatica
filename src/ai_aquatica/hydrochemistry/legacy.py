"""Backward-compatible ion-balance API.

New projects should prefer :mod:`ai_aquatica.hydrochemistry.ion_balance`, which
adds unit conversion, typed configuration and summary diagnostics.  The original
function names are retained to avoid breaking existing notebooks and tests.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .ion_balance import (
    IonBalanceConfig,
    calculate_charge_balance,
    correct_by_proportional_scaling,
)


def calculate_ion_balance(
    data: pd.DataFrame,
    cations: Sequence[str],
    anions: Sequence[str],
    units: str = "meq/L",
    threshold: float = 5.0,
) -> pd.DataFrame:
    """Calculate ion/charge balance error in percent.

    Parameters are kept compatible with the original implementation.  The
    returned DataFrame includes both legacy columns (``Cations_Sum``,
    ``Anions_Sum``, ``Ion_Balance``) and explicit scientific names
    (``*_meq_L`` and ``Charge_Balance_Error_pct``).
    """

    config = IonBalanceConfig(cations=list(cations), anions=list(anions), units=units, threshold=threshold)
    return calculate_charge_balance(data, config, copy=True)


def identify_potential_errors(data: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    """Flag samples whose absolute ion-balance error exceeds ``threshold``."""

    result = data.copy()
    if "Ion_Balance" not in result.columns:
        raise KeyError("Column 'Ion_Balance' is required. Run calculate_ion_balance first.")
    result["Potential_Error"] = pd.to_numeric(result["Ion_Balance"], errors="coerce").abs() > threshold
    result["Ion_Balance_Status"] = np.where(result["Potential_Error"], "review", "acceptable")
    return result


def correct_ion_discrepancies(
    data: pd.DataFrame,
    cations: Sequence[str],
    anions: Sequence[str],
) -> pd.DataFrame:
    """Balance cation and anion sums for diagnostic sensitivity analysis.

    The legacy API overwrites the supplied ion columns; this behaviour is
    retained for compatibility.  For a non-destructive workflow, use
    :func:`ai_aquatica.hydrochemistry.correct_by_proportional_scaling`.
    """

    if not cations or not anions:
        return data.copy()
    if "Cations_Sum" not in data.columns or "Anions_Sum" not in data.columns:
        data = calculate_ion_balance(data, cations, anions)

    diagnostic = correct_by_proportional_scaling(data, cations, anions)
    result = data.copy()
    for column in cations:
        result[column] = diagnostic[f"{column}_corrected"].fillna(result[column])
    for column in anions:
        result[column] = diagnostic[f"{column}_corrected"].fillna(result[column])

    result["Cations_Sum"] = result[list(cations)].sum(axis=1)
    result["Anions_Sum"] = result[list(anions)].sum(axis=1)
    result["Cations_Sum_meq_L"] = result["Cations_Sum"]
    result["Anions_Sum_meq_L"] = result["Anions_Sum"]
    total = result["Cations_Sum"] + result["Anions_Sum"]
    result["Ion_Balance"] = np.where(
        total.ne(0),
        (result["Cations_Sum"] - result["Anions_Sum"]) / total * 100.0,
        0.0,
    )
    result["Charge_Balance_Error_pct"] = result["Ion_Balance"]
    result["Potential_Error"] = result["Ion_Balance"].abs() > 5.0
    result["Ion_Balance_Status"] = np.where(result["Potential_Error"], "review", "acceptable")
    return result
