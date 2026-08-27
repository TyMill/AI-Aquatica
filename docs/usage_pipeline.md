# Pipeline API

```python
from ai_aquatica.core import WaterQualityPipeline
from ai_aquatica.datasets import load_example_dataset

data = load_example_dataset()

features = [
    "temperature",
    "pH",
    "conductivity",
    "dissolved_oxygen",
    "nitrate",
    "phosphate",
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
        result_name="water_quality_classification",
    )
)

pipeline.export_artifacts("outputs/artifacts")
pipeline.export_html_report("outputs/report.html")
```

## Leakage-safe preprocessing

`impute()`, `scale()`, and predictive PCA are registered but not fitted on the complete dataset. They are fitted separately inside every training partition or cross-validation fold.

## PCA behaviour

- `pca(..., use_for_model=False)` produces exploratory scores and loadings only.
- `pca(..., use_for_model=True)` additionally places PCA inside the predictive scikit-learn pipeline, where it is fitted independently within every training fold.

## Validation modes

- `holdout`
- `stratified_kfold`
- `group_kfold`
- `temporal_holdout`

## Hydrochemical quality policies

- `warn`: retain non-accepted rows and emit a warning;
- `filter`: train only on accepted rows;
- `raise`: stop when non-accepted rows are present;
- `ignore`: do not use the status column.

## API compatibility

Both `.select_features(columns=[...])` and `.select_features(features=[...])` are supported. The underlying processed DataFrame is available as either `pipeline.data` or `pipeline.dataset.data`.
