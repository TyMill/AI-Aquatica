# SoftwareX reproducibility notes

This document summarizes how to reproduce the current AI-Aquatica software state for a SoftwareX-style submission.

## Installation

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[testing,interactive]"
```

Optional extras:

```bash
python -m pip install "ai-aquatica[excel]"       # Excel files
python -m pip install "ai-aquatica[api]"         # HTTP API loading
python -m pip install "ai-aquatica[nosql]"       # MongoDB loading
python -m pip install "ai-aquatica[database]"    # SQLAlchemy workflows
python -m pip install "ai-aquatica[deep_learning]"  # Autoencoder/GAN utilities
```

## Test suite

```bash
python -m pytest
python -m compileall ai_aquatica
```

Current local validation result after publication-readiness cleanup:

```text
51 passed, 1 skipped, 3 subtests passed
```

The skipped test concerns TensorFlow-based autoencoder imputation when TensorFlow is not installed. This is expected for a minimal environment because TensorFlow is an optional deep-learning dependency.

## Core reproducible workflow

Run:

```bash
python examples/water_quality_workflow.py
```

The example creates a deterministic synthetic water-quality dataset, performs cleaning, imputation, standardization, Random Forest classification, anomaly detection, ion-balance checking, and report generation.

## Software scope

AI-Aquatica covers the following workflow stages:

1. Data loading from CSV, Excel, JSON, SQLite, MongoDB, and APIs.
2. Data cleaning and missing-value imputation.
3. Hydrochemical consistency checks through ion-balance utilities.
4. Descriptive statistics, correlation analysis, ANOVA, and time-series decomposition.
5. Classical ML models for regression, classification, clustering, anomaly detection, and synthetic data generation.
6. Static and interactive visualizations.
7. HTML report generation.

## Notes for reviewers

- The package imports without optional database, HTTP, Excel, Plotly, or TensorFlow dependencies.
- Tests avoid external services and heavy deep-learning execution.
- Public metadata files are included for citation and indexing.
- The MIT license permits reuse and extension.
