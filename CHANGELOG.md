# Changelog

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
