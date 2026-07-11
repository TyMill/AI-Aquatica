# Pipeline API

```python
from ai_aquatica.core import WaterQualityPipeline
from ai_aquatica.datasets import load_example_dataset

data = load_example_dataset()

pipeline = (
    WaterQualityPipeline.from_dataframe(data)
    .describe()
    .ion_balance(cations=["Ca", "Mg", "Na", "K"], anions=["HCO3", "Cl", "SO4"], units="mg/L")
    .select_features(["temperature", "pH", "conductivity", "dissolved_oxygen", "nitrate", "phosphate"], target="water_quality_class")
    .impute(strategy="median")
    .scale()
    .pca(n_components=2)
    .train_random_forest(task="classification")
)

pipeline.export_html_report("report.html")
```
