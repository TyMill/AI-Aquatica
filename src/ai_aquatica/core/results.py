"""Typed result containers for reproducible workflows."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class AnalysisResult:
    """Container for outputs produced by a pipeline step."""

    name: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    figures: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)

    def add_table(self, key: str, table: pd.DataFrame) -> "AnalysisResult":
        self.tables[key] = table
        return self

    def add_metric(self, key: str, value: Any) -> "AnalysisResult":
        self.metrics[key] = value
        return self

    def add_figure(self, key: str, figure: Any) -> "AnalysisResult":
        self.figures[key] = figure
        return self
