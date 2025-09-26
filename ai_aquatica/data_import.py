"""Utility wrappers for importing data into AI Aquatica projects.

This module provides a stable, high-level API that delegates to the
implementations in :mod:`ai_aquatica.data_loading`. The functions are kept
very small so that tests – and downstream users – can rely on a predictable
interface while the heavy lifting remains inside ``data_loading``.
"""
from __future__ import annotations

from typing import Any, Dict

from ai_aquatica.data_loading import (
    load_csv,
    load_excel,
    load_json,
    load_sql,
    load_mongo,
)

__all__ = [
    "import_csv",
    "import_excel",
    "import_json",
    "import_from_sql",
    "import_from_nosql",
]


def import_csv(file_path: str):
    """Load a CSV file via :func:`ai_aquatica.data_loading.load_csv`."""

    return load_csv(file_path)


def import_excel(file_path: str, sheet_name: Any = 0):
    """Load an Excel worksheet by delegating to :func:`load_excel`."""

    return load_excel(file_path, sheet_name=sheet_name)


def import_json(file_path: str):
    """Load JSON data using :func:`load_json`."""

    return load_json(file_path)


def import_from_sql(db_path: str, query: str):
    """Execute *query* against the SQLite database located at *db_path*."""

    return load_sql(sql_query=query, db_path=db_path)


def import_from_nosql(
    collection_name: str,
    db_name: str,
    query: Dict[str, Any] | None = None,
    mongo_uri: str = "mongodb://localhost:27017/",
):
    """Proxy to :func:`load_mongo` for MongoDB collections."""

    actual_query: Dict[str, Any] = query or {}
    return load_mongo(
        collection_name=collection_name,
        db_name=db_name,
        query=actual_query,
        mongo_uri=mongo_uri,
    )
