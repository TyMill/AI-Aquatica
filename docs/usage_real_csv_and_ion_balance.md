# Real-world CSV import and ion-balance diagnostics

AI-Aquatica includes robust helpers for environmental monitoring datasets exported as regional CSV files. The loader detects common CSV conventions such as semicolon separators, decimal commas and legacy encodings.

```python
from ai_aquatica.io import load_water_quality_csv
from ai_aquatica.core import WaterQualityPipeline
from ai_aquatica.hydrochemistry import NITROGEN_AS_N_MASS_PER_MEQ

# Detects separator, decimal convention and encoding automatically.
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

In the released SoftwareX dataset, NO3, NO2, and NH4 are reported as mg N/L, so the explicit nitrogen reporting-basis conversion mapping is required.

The method `ion_balance_from_alkalinity()` derives HCO3 from alkalinity, calculates the charge-balance error and adds diagnostic columns:

- `HCO3`
- `Cations_Sum_meq_L`
- `Anions_Sum_meq_L`
- `Charge_Balance_Error_pct`
- `Potential_Error`
- `Ion_Balance_Status`

A high charge-balance error should be interpreted as a quality-control warning. It may indicate incomplete major-ion coverage, inconsistent units, derived alkalinity limitations or laboratory quality-control issues. It should not be treated as an automatic correction of measurements.
