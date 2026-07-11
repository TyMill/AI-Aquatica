# AI-Aquatica SoftwareX architecture

AI-Aquatica v2.0.0 uses a `src/`-based Python package layout and separates the software into domain-oriented layers.

```text
src/ai_aquatica/
  core/             Dataset, pipeline and result abstractions
  io/               CSV, Excel, JSON, SQL, MongoDB and API loaders/importers
  preprocessing/    Cleaning, missing-value handling and transformations
  analysis/         Descriptive statistics and time-series analysis
  modeling/         Classical ML, clustering, anomaly detection and synthetic data helpers
  hydrochemistry/   Charge/ion-balance diagnostics and hydrochemical quality control
  visualization/    Exploratory and model-oriented plotting
  reporting/        Standalone HTML report generation and templates
  datasets/         Bundled reproducible example datasets
```

The main SoftwareX workflow is exposed through `WaterQualityPipeline`, which combines dataset validation, descriptive analysis, ion-balance diagnostics, preprocessing, PCA, random-forest modelling and HTML report generation.

The former flat module layout was intentionally removed because it made the repository harder to review, maintain and describe as scientific software. The current structure is designed to support future extensions such as additional hydrochemical indices, geospatial/remote-sensing integration and richer reporting.
