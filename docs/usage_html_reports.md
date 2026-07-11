# HTML reports

AI-Aquatica can generate standalone HTML reports with dataset diagnostics, missingness profile, descriptive statistics, correlation structure and pipeline outputs.

```python
from ai_aquatica.reporting import generate_water_quality_report
from ai_aquatica.datasets import load_example_dataset

data = load_example_dataset()
generate_water_quality_report(data, "water_quality_report.html")
```
