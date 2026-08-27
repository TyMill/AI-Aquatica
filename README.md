# 🌊 AI-Aquatica

[![PyPI version](https://img.shields.io/pypi/v/ai-aquatica?color=blue)](https://pypi.org/project/ai-aquatica/)
[![Downloads](https://static.pepy.tech/badge/ai-aquatica)](https://pepy.tech/project/ai-aquatica)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://tymill.github.io/AI-Aquatica/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/TyMill/AI-Aquatica/actions/workflows/ci.yml/badge.svg)](https://github.com/TyMill/AI-Aquatica/actions/workflows/ci.yml)

**AI-Aquatica** is an open-source Python library for reproducible water-quality analysis with hydrochemical quality control, leakage-safe machine-learning validation, visualization, and standalone HTML reporting.

The project is designed for researchers, students and environmental analysts working with tabular datasets from rivers, lakes, reservoirs, coastal waters and urban aquatic systems. It provides both a backward-compatible functional API and a new publication-oriented pipeline API for complete workflows.

---

## Why AI-Aquatica?

Water-quality analysis often combines repeated steps: loading monitoring data, inspecting missing values, checking hydrochemical consistency, standardizing variables, running exploratory statistics, training models and preparing reports. In many projects these steps are implemented as separate notebooks or ad hoc scripts.

AI-Aquatica organizes these steps into a reusable Python package while keeping the workflow transparent and inspectable.

---

## Main features

- **Core workflow API**: `WaterQualityDataset`, `WaterQualityPipeline`, and typed result containers.
- **Data import**: CSV, Excel, JSON, SQL, NoSQL and API helpers.
- **Preprocessing**: missing-value handling, standardization, normalization and transformations.
- **Hydrochemistry**: explicit `mg/L` to `meq/L` conversion, charge-balance diagnostics, completeness checks, and `acceptable`, `review`, or `indeterminate` status.
- **Exploratory analysis**: descriptive statistics, correlation analysis, ANOVA and time-series decomposition.
- **Machine learning**: leakage-safe Random Forest classification and regression with holdout, stratified, grouped, and temporal validation; classical exploratory utilities remain available separately.
- **Visualization**: static exploratory plots and optional interactive Plotly charts.
- **Reporting**: standalone HTML reports with dataset diagnostics, figures, model outputs and ion-balance summaries.
- **Reproducibility assets**: a bundled synthetic tutorial dataset and the real 148-observation dataset used by the SoftwareX workflow.

---


## Real CSV import and hydrochemical quality control

AI-Aquatica can load monitoring datasets exported as European CSV files with semicolon separators, decimal commas and legacy encodings:

```python
from ai_aquatica.io import load_water_quality_csv
from ai_aquatica.core import WaterQualityPipeline
from ai_aquatica.hydrochemistry import NITROGEN_AS_N_MASS_PER_MEQ

data = load_water_quality_csv("water_quality.csv")

pipeline = (
    WaterQualityPipeline.from_dataframe(data)
    .describe()
    .ion_balance_from_alkalinity(
        cations=["Ca", "Mg", "NH4"],
        anions=["Cl", "SO4", "NO3", "NO2"],
        alkalinity_col="Alkalinity",
        alkalinity_units="mg_CaCO3_L",
        threshold=10,
        equivalent_weights=NITROGEN_AS_N_MASS_PER_MEQ,
    )
)

pipeline.export_html_report("ai_aquatica_report.html")
```

The released real dataset reports NO3, NO2, and NH4 as mg N/L; the explicit mapping above converts that reporting basis correctly for charge balance.

The HTML report contains dataset diagnostics, missingness, descriptive statistics, correlation structure and a hydrochemical ion-balance quality-control section.

## Installation

```bash
pip install ai-aquatica
```

Optional extras:

```bash
pip install "ai-aquatica[interactive]"     # Plotly charts
pip install "ai-aquatica[deep_learning]"   # optional experimental TensorFlow utilities (not used in the primary SoftwareX workflow)
pip install "ai-aquatica[database]"        # SQL support
pip install "ai-aquatica[nosql]"           # MongoDB support
pip install "ai-aquatica[all]"             # all optional dependencies
```

Development installation:

```bash
git clone https://github.com/TyMill/AI-Aquatica.git
cd AI-Aquatica
pip install -e ".[testing,interactive]"
python -m pytest
```

---

## Quick start: professional pipeline

```python
from ai_aquatica.core import WaterQualityPipeline
from ai_aquatica.datasets import load_example_dataset

# Load bundled example data
data = load_example_dataset()

features = [
    "temperature",
    "pH",
    "conductivity",
    "dissolved_oxygen",
    "nitrate",
    "phosphate",
    "chlorophyll_a",
]

pipeline = (
    WaterQualityPipeline.from_dataframe(data)
    .describe()
    .ion_balance(
        cations=["Ca", "Mg", "Na", "K"],
        anions=["HCO3", "Cl", "SO4"],
        units="mg/L",
        threshold=5.0,
    )
    .select_features(features=features, target="water_quality_class")
    .impute(strategy="median")
    .scale()
    .pca(n_components=2, use_for_model=False)
    .train_random_forest(
        task="classification",
        validation="group_kfold",
        group_column="site",
        n_splits=4,
        quality_policy="warn",
    )
)

pipeline.export_artifacts("outputs/artifacts")
pipeline.export_html_report("outputs/ai_aquatica_report.html")
```

Run the full SoftwareX-style reproducibility example:

```bash
python examples/softwarex_full_workflow.py
python examples/real_dataset_workflow.py --output outputs/real_dataset
```

Outputs are written to:

```text
outputs/softwarex_example/ai_aquatica_report.html
outputs/softwarex_example/processed_water_quality.csv
```

---


## Reproducible real-dataset validation

The complete manuscript workflow is executed by one command:

```bash
python examples/real_dataset_workflow.py --output validation/real_dataset --bootstrap 1000
```

The workflow fits imputation, scaling, and any predictive PCA **only inside each training partition or validation fold**. It exports campaign-grouped station classification, a QC-filtered sensitivity analysis, chlorophyll-a regression against a mean baseline, fold metrics, out-of-sample predictions, 95% bootstrap confidence intervals, confusion matrices, an observed-versus-predicted plot, hydrochemical diagnostics, and a standalone HTML report.

Hydrochemical status can control model training through `quality_policy="warn"`, `"filter"`, `"raise"`, or `"ignore"`.

## Hydrochemical ion balance

AI-Aquatica includes a domain-specific charge-balance module for chemical quality control.

```python
from ai_aquatica.datasets import load_example_dataset
from ai_aquatica.hydrochemistry import IonBalanceConfig, calculate_charge_balance

water = load_example_dataset()
config = IonBalanceConfig(
    cations=["Ca", "Mg", "Na", "K"],
    anions=["HCO3", "Cl", "SO4"],
    units="mg/L",
    threshold=5.0,
)

checked = calculate_charge_balance(water, config)
print(checked[["Ion_Balance", "Potential_Error", "Ion_Balance_Status"]].head())
```

The module converts mg/L concentrations to milliequivalents per litre using an explicit equivalent-weight catalogue and reports charge-balance error in percent. Rows with missing, non-numeric, infinite, negative, or zero-total selected ions are marked `indeterminate`; they are never assigned an artificial 0% error. Diagnostic correction utilities are provided only for sensitivity analysis and must not replace laboratory quality-control procedures.

---

## HTML reports

```python
from ai_aquatica.datasets import load_example_dataset
from ai_aquatica.reporting import generate_water_quality_report

data = load_example_dataset()
generate_water_quality_report(data, "water_quality_report.html")
```

Generated reports are standalone HTML files containing dataset preview, missingness profile, descriptive statistics, correlation figure and optional pipeline outputs.

---

## Package structure

```text
ai_aquatica/
  core/              # dataset container, pipeline and result objects
  hydrochemistry/    # ion/charge balance and water chemistry quality control
  reporting/         # standalone HTML reports
  datasets/          # bundled reproducible example data
  ...                # backward-compatible functional modules
```

The original functional modules remain available for existing notebooks and scripts.

---

## Reproducibility for SoftwareX review

This repository includes publication-oriented metadata and reproducibility assets:

- `CITATION.cff`
- `codemeta.json`
- `CONTRIBUTING.md`
- `CHANGELOG.md`
- `docs/softwarex_reproducibility.md`
- `docs/release_checklist.md`
- `requirements-validation.txt` (reference Python 3.13 reproducibility environment)
- `docs/softwarex_architecture.md`
- `examples/softwarex_full_workflow.py`
- bundled example dataset in `src/ai_aquatica/datasets/data/`
- `examples/real_dataset_workflow.py` with bundled real monitoring CSV file in `examples/data/water_quality.csv`

Validation commands:

```bash
python -m pip install -e ".[testing,interactive]"
python -m ruff check .
python -m pytest --cov=ai_aquatica --cov-report=term-missing --cov-fail-under=70
python -m compileall src/ai_aquatica examples
python examples/softwarex_full_workflow.py
python examples/real_dataset_workflow.py --output outputs/real_dataset
```

Current validation status for the SoftwareX-preparation branch:

```text
76 passed, 1 skipped; 72% whole-package coverage in the reference validation environment (core pipeline 79%, hydrochemistry 82%, HTML reporting 95%)
```

The skipped test concerns TensorFlow-based imputation when TensorFlow is not installed. TensorFlow is optional and is not used by the primary SoftwareX workflow. Continuous integration tests Python 3.9–3.13 on Ubuntu and runs smoke tests on Ubuntu, Windows, and macOS.

---

## Citation

Please cite the archived software release and/or the SoftwareX article once published. See `CITATION.cff` for machine-readable citation metadata.

---

## License

This project is licensed under the MIT License.

---

## Contributing

Contributions are welcome. Please see `CONTRIBUTING.md` for development setup, testing and pull-request guidance.
