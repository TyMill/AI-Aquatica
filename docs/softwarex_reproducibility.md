# SoftwareX reproducibility notes

This document describes how to reproduce the AI-Aquatica v2.3.0 major-revision release candidate.

## Installation

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[testing]"
```

Optional extras are not required by the primary manuscript workflow. TensorFlow utilities are experimental and are not used to justify the machine-learning results reported in the SoftwareX article.

For exact reproduction of the reference Python 3.13 environment used in the 2026-08-27 audit, see `requirements-validation.txt`. Ruff is pinned to 0.16.2, matching the release lint audit.

## Validation commands

```bash
python -m ruff check .
python -m pytest --cov=ai_aquatica --cov-report=term-missing --cov-fail-under=70
python -m compileall src/ai_aquatica examples
python examples/real_dataset_workflow.py --output validation/real_dataset --bootstrap 1000
python -m build
```

Reference validation result:

```text
76 passed, 1 skipped
72.3% whole-package coverage
```

The optional skip concerns TensorFlow-based functionality when TensorFlow is not installed. Module-level coverage in the reference environment is 79% for the core pipeline, 82% for hydrochemistry, 95% for HTML reporting, and 95% for the CLI.

## Continuous integration

GitHub Actions is configured to test:

- Python 3.9, 3.10, 3.11, 3.12, and 3.13 on Ubuntu;
- core smoke tests on Ubuntu, Windows, and macOS;
- Ruff static analysis for source, tests, and examples;
- coverage with a 70% whole-package threshold;
- source compilation;
- execution of the real-dataset reproducibility workflow;
- source and wheel package builds;
- independent import/version smoke testing of the built wheel.

## Exact reviewed release

Before resubmission, the authors must:

1. run the final local verification commands and retain the log;
2. regenerate `validation/real_dataset/` with `--bootstrap 1000`;
3. merge the verified code and validation outputs;
4. create the exact Git tag `v2.3.0`;
5. publish the same version on PyPI;
6. archive that GitHub release in Zenodo;
7. insert the exact v2.3.0 Zenodo DOI into the manuscript and release metadata only after it exists;
8. verify that `pip install ai-aquatica==2.3.0` installs the reviewed API.

No older Zenodo DOI should be presented as the exact v2.3.0 reviewed archive.

## Reproducible real-data workflow

```bash
python examples/real_dataset_workflow.py --output validation/real_dataset --bootstrap 1000
```

The workflow exports:

- processed data;
- hydrochemical diagnostics;
- campaign-grouped station classification;
- QC-filtered classification sensitivity analysis;
- campaign-grouped chlorophyll-a regression;
- mean-prediction regression baseline;
- per-fold metrics;
- out-of-sample predictions;
- confusion matrices;
- observed-versus-predicted plot;
- group-aware bootstrap 95% confidence intervals;
- feature-importance tables;
- standalone HTML report.

## Dataset documentation

The released real dataset is documented in `examples/data/README.md` and `examples/data/analytical_methods.csv`. It contains 148 original, non-synthetic observations from 37 monthly campaigns (January 2020-January 2023) at four urban aquatic locations in Szczecin, Poland. The dataset is licensed separately under CC BY 4.0; the software source code is MIT-licensed.
