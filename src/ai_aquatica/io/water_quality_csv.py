"""Robust CSV loading for water-quality datasets."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ..preprocessing.columns import normalize_water_quality_columns


def _decode_sample(path: Path, encodings: tuple[str, ...]) -> tuple[str, str]:
    raw = path.read_bytes()[:8192]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        # latin1-like fallback never fails and preserves bytes sufficiently for parsing.
        return raw.decode("latin1", errors="replace"), "latin1"
    return raw.decode("utf-8", errors="replace"), "utf-8"


def _detect_separator(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t|")
        return dialect.delimiter
    except csv.Error:
        lines = [line for line in sample.splitlines() if line.strip()]
        header = lines[0] if lines else sample
        counts = {sep: header.count(sep) for sep in [";", ",", "\t", "|"]}
        return max(counts, key=counts.get) if any(counts.values()) else ","


def _detect_decimal(sample: str, separator: str) -> str:
    # If semicolon is used as the field separator and decimal commas are found,
    # use European decimal notation. Otherwise let pandas use the default dot.
    if separator == ";" and re.search(r"\d,\d", sample):
        return ","
    return "."


def _coerce_numeric_strings(data: pd.DataFrame, decimal: str) -> pd.DataFrame:
    result = data.copy()
    for column in result.columns:
        if result[column].dtype != object:
            continue
        series = result[column].astype(str).str.strip()
        if decimal == ",":
            candidate = series.str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
        else:
            candidate = series.str.replace(" ", "", regex=False)
        converted = pd.to_numeric(candidate, errors="coerce")
        non_missing_original = series.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA}).notna().sum()
        if non_missing_original and converted.notna().sum() / non_missing_original >= 0.85:
            result[column] = converted
    return result


def detect_csv_format(
    path: str | Path,
    *,
    encodings: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1250", "latin1"),
) -> dict[str, str]:
    """Detect encoding, separator and decimal convention for a CSV file."""

    path = Path(path)
    sample, encoding = _decode_sample(path, encodings)
    sep = _detect_separator(sample)
    decimal = _detect_decimal(sample, sep)
    return {"encoding": encoding, "sep": sep, "decimal": decimal}


def load_water_quality_csv(
    path: str | Path,
    *,
    auto_detect: bool = True,
    normalize_columns: bool = True,
    coerce_numeric: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load a water-quality CSV file with robust defaults.

    The loader handles common environmental-monitoring exports, including
    semicolon-separated European CSV files, decimal commas, and legacy encodings.
    User-provided keyword arguments override detected values.
    """

    path = Path(path)
    read_kwargs: dict[str, Any] = {}
    if auto_detect:
        read_kwargs.update(detect_csv_format(path))
    read_kwargs.update(kwargs)
    data = pd.read_csv(path, **read_kwargs)
    if normalize_columns:
        data = normalize_water_quality_columns(data)
    if coerce_numeric:
        data = _coerce_numeric_strings(data, decimal=str(read_kwargs.get("decimal", ".")))
    return data


__all__ = ["detect_csv_format", "load_water_quality_csv"]
