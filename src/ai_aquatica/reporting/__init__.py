"""HTML reporting utilities."""

from .html import HtmlReportConfig, generate_water_quality_report
from .legacy_reports import (
    generate_interpretation_report,
    generate_statistical_report,
    suggest_further_analysis,
)

__all__ = [
    "HtmlReportConfig",
    "generate_water_quality_report",
    "generate_statistical_report",
    "generate_interpretation_report",
    "suggest_further_analysis",
]
