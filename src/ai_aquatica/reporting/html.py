"""Professional HTML reporting utilities for AI-Aquatica."""
from __future__ import annotations

import base64
import html
import json
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


@dataclass(frozen=True)
class HtmlReportConfig:
    """Configuration for a standalone HTML report."""

    title: str = "AI-Aquatica water quality report"
    subtitle: str = "Reproducible environmental data analysis"
    include_raw_preview: bool = True
    max_preview_rows: int = 10


def _fig_to_base64(fig) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _safe_table(table: pd.DataFrame | pd.Series, classes: str = "data-table") -> str:
    if isinstance(table, pd.Series):
        table = table.to_frame()
    return table.to_html(classes=classes, border=0, escape=True)


def _correlation_figure(data: pd.DataFrame) -> str | None:
    numeric = data.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric.corr(), cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlation matrix")
    return _fig_to_base64(fig)


def _missingness_figure(data: pd.DataFrame) -> str | None:
    missing = data.isna().mean().sort_values(ascending=False)
    if missing.empty or missing.max() == 0:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    missing.plot(kind="bar", ax=ax)
    ax.set_ylabel("Missing fraction")
    ax.set_title("Missing values by column")
    return _fig_to_base64(fig)


def _render_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
        return f"<pre>{html.escape(text)}</pre>"
    return html.escape(str(value))


def _metrics_html(metrics: Mapping[str, Any]) -> str:
    if not metrics:
        return "<p>No metrics available.</p>"
    rows = []
    for key, value in metrics.items():
        rows.append(
            f"<tr><th>{html.escape(str(key))}</th><td>{_render_metric_value(value)}</td></tr>"
        )
    return '<table class="data-table"><tbody>' + "".join(rows) + "</tbody></table>"



def _hydrochemistry_section(data: pd.DataFrame) -> str:
    required = {"Charge_Balance_Error_pct", "Ion_Balance_Status"}
    if not required.issubset(set(data.columns)):
        return ""
    errors = pd.to_numeric(data["Charge_Balance_Error_pct"], errors="coerce")
    statuses = data["Ion_Balance_Status"].astype(str).value_counts().rename_axis("status").reset_index(name="count")
    metrics = {
        "samples_with_diagnostics": int(errors.notna().sum()),
        "mean_abs_charge_balance_error_pct": float(errors.abs().mean()) if errors.notna().any() else float("nan"),
        "median_abs_charge_balance_error_pct": float(errors.abs().median()) if errors.notna().any() else float("nan"),
        "max_abs_charge_balance_error_pct": float(errors.abs().max()) if errors.notna().any() else float("nan"),
        "flagged_samples": int((data["Ion_Balance_Status"].astype(str) == "review").sum()),
        "indeterminate_samples": int((data["Ion_Balance_Status"].astype(str) == "indeterminate").sum()),
    }
    figure_html = ""
    if errors.notna().any():
        fig, ax = plt.subplots(figsize=(8, 4))
        errors.dropna().plot(kind="hist", bins=20, ax=ax)
        ax.set_xlabel("Charge balance error (%)")
        ax.set_title("Hydrochemical charge-balance diagnostics")
        figure_html = f'<img class="figure" alt="Charge balance error distribution" src="data:image/png;base64,{_fig_to_base64(fig)}">'
    return f"""
    <section class="card">
      <h2>Hydrochemical quality control</h2>
      <p>This section summarizes ion-balance diagnostics. A high charge-balance error does not automatically invalidate a sample, but indicates that the ion set, units, laboratory results or derived alkalinity should be reviewed.</p>
      <h3>Charge-balance metrics</h3>
      {_metrics_html(metrics)}
      <h3>Status counts</h3>
      {_safe_table(statuses)}
      {figure_html}
    </section>
    """

def generate_water_quality_report(
    data: pd.DataFrame,
    output_path: str | Path,
    *,
    results: Mapping[str, Any] | None = None,
    title: str = "AI-Aquatica water quality report",
    config: HtmlReportConfig | None = None,
) -> Path:
    """Generate a standalone HTML report for a water-quality analysis.

    The report contains dataset diagnostics, descriptive statistics, embedded
    figures, and optional results produced by :class:`WaterQualityPipeline`.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = config or HtmlReportConfig(title=title)
    numeric = data.select_dtypes(include="number")
    describe_html = _safe_table(numeric.describe().T) if not numeric.empty else "<p>No numeric columns.</p>"
    missing_table = pd.DataFrame(
        {
            "column": data.columns,
            "missing_count": data.isna().sum().values,
            "missing_fraction": data.isna().mean().values,
            "dtype": [str(data[column].dtype) for column in data.columns],
        }
    )
    corr_img = _correlation_figure(data)
    missing_img = _missingness_figure(data)
    hydrochemistry_html = _hydrochemistry_section(data)

    result_sections = []
    for name, result in (results or {}).items():
        metrics = getattr(result, "metrics", {}) or {}
        tables = getattr(result, "tables", {}) or {}
        figures = getattr(result, "figures", {}) or {}
        table_blocks = []
        for table_name, table in tables.items():
            if isinstance(table, pd.DataFrame):
                table_blocks.append(
                    f"<h4>{html.escape(str(table_name))}</h4>" + _safe_table(table.head(50))
                )
        figure_blocks = []
        for figure_name, figure in figures.items():
            try:
                encoded = _fig_to_base64(figure)
            except Exception:
                continue
            figure_blocks.append(
                f'<h4>{html.escape(str(figure_name).replace("_", " ").title())}</h4>'
                f'<img class="figure" alt="{html.escape(str(figure_name))}" src="data:image/png;base64,{encoded}">'
            )
        result_sections.append(
            f"""
            <section class="card">
              <h3>{html.escape(str(name).replace('_', ' ').title())}</h3>
              <h4>Metrics</h4>
              {_metrics_html(metrics)}
              {''.join(figure_blocks)}
              {''.join(table_blocks)}
            </section>
            """
        )

    preview_html = ""
    if cfg.include_raw_preview:
        preview_html = _safe_table(data.head(cfg.max_preview_rows))

    html_content = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(cfg.title)}</title>
<style>
:root {{ --bg: #f7f9fb; --card: #ffffff; --ink: #1f2937; --muted: #667085; --line: #d0d7de; }}
body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--ink); }}
header {{ background: linear-gradient(135deg, #0f4c5c, #2a9d8f); color: white; padding: 2.5rem; }}
header h1 {{ margin: 0; font-size: 2rem; }}
header p {{ margin: .5rem 0 0; color: rgba(255,255,255,.88); }}
main {{ max-width: 1180px; margin: 0 auto; padding: 1.5rem; }}
.card {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 1.25rem; margin-bottom: 1rem; box-shadow: 0 1px 2px rgba(16,24,40,.04); }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; }}
.metric {{ background: #eef6f6; border-radius: 12px; padding: 1rem; }}
.metric strong {{ display:block; font-size: 1.6rem; }}
.metric span {{ color: var(--muted); }}
.data-table {{ border-collapse: collapse; width: 100%; font-size: .9rem; }}
.data-table th, .data-table td {{ border-bottom: 1px solid var(--line); padding: .5rem; text-align: left; }}
.data-table th {{ background: #f2f4f7; }}
pre {{ white-space: pre-wrap; margin: 0; font-size: .82rem; }}
.figure {{ max-width: 100%; border: 1px solid var(--line); border-radius: 12px; }}
footer {{ color: var(--muted); padding: 2rem; text-align: center; }}
</style>
</head>
<body>
<header>
  <h1>{html.escape(cfg.title)}</h1>
  <p>{html.escape(cfg.subtitle)}</p>
</header>
<main>
  <section class="grid">
    <div class="metric"><strong>{data.shape[0]}</strong><span>samples</span></div>
    <div class="metric"><strong>{data.shape[1]}</strong><span>columns</span></div>
    <div class="metric"><strong>{numeric.shape[1]}</strong><span>numeric variables</span></div>
    <div class="metric"><strong>{int(data.isna().sum().sum())}</strong><span>missing values</span></div>
  </section>
  <section class="card"><h2>Dataset preview</h2>{preview_html}</section>
  <section class="card"><h2>Missingness profile</h2>{_safe_table(missing_table)}{f'<img class="figure" alt="Missingness" src="data:image/png;base64,{missing_img}">' if missing_img else ''}</section>
  <section class="card"><h2>Descriptive statistics</h2>{describe_html}</section>
  <section class="card"><h2>Correlation structure</h2>{f'<img class="figure" alt="Correlation matrix" src="data:image/png;base64,{corr_img}">' if corr_img else '<p>Not enough numeric variables for correlation analysis.</p>'}</section>
  {hydrochemistry_html}
  {''.join(result_sections)}
</main>
<footer>Generated by AI-Aquatica.</footer>
</body>
</html>"""
    output_path.write_text(html_content, encoding="utf-8")
    return output_path


__all__ = ["HtmlReportConfig", "generate_water_quality_report"]
