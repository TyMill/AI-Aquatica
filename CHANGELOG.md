# Changelog

## v2.3.0 - SoftwareX major-revision release

- Explicitly enable scikit-learn's experimental `IterativeImputer` before import for clean compatibility with the pinned validation environment.
- Removed source-path import hacks from examples; examples now run against the installed package, matching CI and release usage.
- Updated the legacy tutorial classifier so supervised imputation/scaling are fitted only inside the training partition.
- Keep legacy report heatmap files beside the requested report path instead of writing into the repository root.

- Added Ruff to the testing extra and CI release lint gate.
- Modernized Python 3.9+ type annotations and warning stack levels for clean Ruff validation.
- Cleaned source-checkout example import handling and release metadata instructions.

- Eliminated preprocessing leakage by fitting imputation, scaling, and optional PCA only inside training partitions or validation folds.
- Added holdout, stratified K-fold, campaign-grouped K-fold, and temporal holdout validation.
- Added per-class precision, recall and F1, confusion matrices, fold-level metrics, out-of-sample predictions, group-aware bootstrap 95% confidence intervals, and regression baselines.
- Added RMSE, MAE and R² for regression, plus observed-versus-predicted figures.
- Added hydrochemical safeguards: incomplete, non-numeric, infinite, negative, or zero-total ion sets are now marked `indeterminate` rather than `acceptable`.
- Added explicit `warn`, `filter`, `raise`, and `ignore` quality policies linking hydrochemical QC to predictive modelling.
- Added `features=` as an API alias and `pipeline.data` for manuscript/API consistency.
- Added a single executable workflow that reproduces hydrochemical diagnostics, grouped station classification, QC-filtered sensitivity analysis, chlorophyll-a regression, tables, figures, and HTML reporting.
- Added artifact export for metrics, predictions, fold results, feature importance tables and figures.
- Expanded the test suite to 76 passed tests plus one optional TensorFlow skip, including CLI and grouped-validation safeguards.
- Added multi-version and multi-operating-system CI, coverage reporting, package building, and workflow smoke tests.
- Synchronized package, citation and CodeMeta version metadata at v2.3.0.
- Corrected the confirmed source-data transcription error for Lake Słoneczne (`r2`), campaign 27, from pH 14.9 to pH 7.49.
- Documented that NO3, NO2 and NH4 in the SoftwareX dataset are reported as mg N/L and added an explicit nitrogen-as-N mass-per-meq conversion mapping for charge-balance calculations.

## v2.2.0 - Real-dataset hardening release

- Added robust water-quality CSV loader with automatic separator, decimal and encoding detection.
- Added column normalization utilities for common hydrochemical variables and legacy CSV exports.
- Added alkalinity-to-bicarbonate conversion for ion-balance workflows.
- Added `ion_balance_from_alkalinity()` to the fluent pipeline API.
- Added a hydrochemical quality-control section to standalone HTML reports.
- Added tests for European CSV import, column normalization and alkalinity-based ion balance.
- Validated the workflow on a real 148-sample water-quality dataset.


## v1.2.0 - SoftwareX professionalisation release

### Added
- `ai_aquatica.core` package with `WaterQualityDataset`, `WaterQualityPipeline`, and `AnalysisResult`.
- `ai_aquatica.hydrochemistry` package with configurable charge/ion balance diagnostics.
- mg/L to meq/L conversion using an equivalent-weight catalogue for common major ions.
- Ion-balance summary diagnostics and non-destructive proportional correction utilities.
- `ai_aquatica.reporting` package for standalone HTML reports.
- Bundled deterministic example water-quality dataset.
- Full SoftwareX-style reproducibility workflow in `examples/softwarex_full_workflow.py`.
- Additional documentation for architecture, pipeline API and HTML reports.

### Changed
- Updated package version to `1.2.0`.
- Strengthened README around reproducibility, hydrochemical validation and reporting.
- Preserved the original functional API for backward compatibility while introducing a stronger modular structure.

### Validation
- Test suite: 54 passed, 1 skipped, 3 subtests passed.
- Example workflow generates an HTML report and processed dataset.

## v1.1.1 - SoftwareX preparation release

### Added
- `CITATION.cff`, `codemeta.json`, `CONTRIBUTING.md`, and SoftwareX reproducibility documentation.
- Deterministic water-quality workflow example.

### Fixed
- Optional dependency handling for TensorFlow, MongoDB, Plotly and Excel-related functionality.
- Visualization functions now return figure/axes objects and support non-interactive execution.

## [2.0.0] - SoftwareX professional structure

### Changed
- Reorganized package to a clean `src/` layout.
- Moved root-level functional scripts into dedicated subpackages: `io`, `preprocessing`, `analysis`, `modeling`, `visualization`, `hydrochemistry`, `reporting`, `core`, and `datasets`.
- Promoted `WaterQualityPipeline`, ion-balance diagnostics, bundled example data, and standalone HTML reporting as the main SoftwareX-facing API.

### Added
- Cleaner public imports through subpackage `__init__.py` files.
- Professional package architecture suitable for documentation and manuscript description.
- Updated tests to validate the new organized API.

### Notes
- Legacy-style functionality is retained inside appropriate subpackages where possible, but the former flat package layout has been removed to make the repository maintainable and publication-ready.
