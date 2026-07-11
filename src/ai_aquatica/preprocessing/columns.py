"""Column-name normalization utilities for water-quality datasets."""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

import pandas as pd

DOMAIN_ALIASES: dict[str, str] = {
    "station": "Station",
    "site": "Station",
    "sampling_site": "Station",
    "month_number": "month",
    "month": "month",
    "date": "date",
    "sampling_date": "date",
    "chl_a": "Chl_a",
    "chla": "Chl_a",
    "chlorophyll_a": "Chl_a",
    "temp": "Temp",
    "temperature": "Temp",
    "ph": "pH",
    "eh": "Eh",
    "chzt_mn": "COD_Mn",
    "cod_mn": "COD_Mn",
    "chzt_cr": "COD_Cr",
    "cod_cr": "COD_Cr",
    "bzt5": "BOD5",
    "bod5": "BOD5",
    "o2rozp": "O2_dissolved",
    "o2_dissolved": "O2_dissolved",
    "dissolved_oxygen": "O2_dissolved",
    "ws": "Secchi_depth",
    "secchi": "Secchi_depth",
    "secchi_depth": "Secchi_depth",
    "no3": "NO3",
    "no2": "NO2",
    "nh4": "NH4",
    "tn": "TN",
    "srp": "SRP",
    "tp": "TP",
    "th": "TH",
    "ca": "Ca",
    "mg": "Mg",
    "na": "Na",
    "k": "K",
    "cl": "Cl",
    "so4": "SO4",
    "hco3": "HCO3",
    "alkalinity": "Alkalinity",
    "zasadowosc": "Alkalinity",
    "zasadowo": "Alkalinity",
    "acidity": "Acidity",
    "kwasowosc": "Acidity",
    "kwasowo": "Acidity",
    "feog": "Fe_total",
    "mnog": "Mn_total",
    "pb": "Pb",
    "zn": "Zn",
    "cd": "Cd",
    "cu": "Cu",
}


def _ascii_slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_value = ascii_value.strip()
    ascii_value = re.sub(r"[^0-9A-Za-z]+", "_", ascii_value)
    ascii_value = re.sub(r"_+", "_", ascii_value).strip("_")
    return ascii_value


def canonicalize_column_name(column: str, aliases: Mapping[str, str] | None = None) -> str:
    """Return a stable, publication-friendly column name.

    The function is deliberately conservative: it removes encoding artefacts,
    whitespace and punctuation, then applies water-quality aliases for common
    hydrochemical variables. Unknown columns are converted to safe snake/camel
    identifiers without changing their semantic content.
    """

    raw = str(column).strip()
    # Common mojibake/encoding artefacts observed in legacy Polish CSV exports.
    raw = raw.replace("æ", "c").replace("ć", "c").replace("\x8d", "")
    raw = raw.replace("Ť", "").replace("µ", "u")
    slug = _ascii_slug(raw)
    lookup = slug.lower()
    alias_map = dict(DOMAIN_ALIASES)
    if aliases:
        alias_map.update({str(k).lower(): str(v) for k, v in aliases.items()})
    if lookup in alias_map:
        return alias_map[lookup]
    # Prefix matching catches truncated mojibake such as "Zasadowo" and "Kwasowo".
    if lookup.startswith("zasadow"):
        return "Alkalinity"
    if lookup.startswith("kwasow"):
        return "Acidity"
    return slug or "unnamed_column"


def build_column_mapping(
    columns: list[str] | pd.Index,
    aliases: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a unique original-to-normalized column mapping."""

    mapping: dict[str, str] = {}
    used: dict[str, int] = {}
    for column in columns:
        base = canonicalize_column_name(str(column), aliases=aliases)
        candidate = base
        if candidate in used:
            used[base] += 1
            candidate = f"{base}_{used[base]}"
        else:
            used[base] = 0
        mapping[str(column)] = candidate
    return mapping


def normalize_water_quality_columns(
    data: pd.DataFrame,
    *,
    aliases: Mapping[str, str] | None = None,
    copy: bool = True,
) -> pd.DataFrame:
    """Normalize column names in a water-quality DataFrame.

    Parameters
    ----------
    data:
        Input table.
    aliases:
        Optional custom aliases where keys are canonicalized source names and
        values are final output names.
    copy:
        If ``True``, return a new DataFrame. If ``False``, rename in place.
    """

    result = data.copy() if copy else data
    result.rename(columns=build_column_mapping(result.columns, aliases=aliases), inplace=True)
    return result


__all__ = [
    "DOMAIN_ALIASES",
    "build_column_mapping",
    "canonicalize_column_name",
    "normalize_water_quality_columns",
]
