# Real-dataset workflow

AI-Aquatica includes a single executable workflow for the real dataset used in the SoftwareX illustrative example:

```bash
python examples/real_dataset_workflow.py --output outputs/real_dataset
```

The workflow:

1. detects CP1250 encoding, semicolon separation and comma decimals;
2. loads 148 observations from four stations;
3. derives bicarbonate from alkalinity and calculates charge-balance diagnostics;
4. marks incomplete or invalid ion sets as `indeterminate`;
5. performs campaign-grouped station classification with `month` as the group;
6. performs a second classification after filtering to hydrochemically `acceptable` rows;
7. performs campaign-grouped chlorophyll-a regression;
8. compares regression performance with a mean-prediction baseline;
9. exports bootstrap 95% confidence intervals, fold metrics, out-of-sample predictions, figures and an HTML report.

Primary outputs:

```text
outputs/real_dataset/
  ai_aquatica_real_dataset_report.html
  processed_water_quality.csv
  validation_summary.csv
  artifacts/
    ion_balance_diagnostics.csv
    station_classification_grouped_metrics.json
    station_classification_grouped_confusion_matrix.csv
    station_classification_grouped_confusion_matrix.png
    station_classification_grouped_predictions.csv
    station_classification_grouped_fold_metrics.csv
    station_classification_qc_filtered_*.csv/json/png
    chlorophyll_regression_grouped_metrics.json
    chlorophyll_regression_grouped_predictions.csv
    chlorophyll_regression_grouped_observed_vs_predicted.png
    chlorophyll_regression_grouped_fold_metrics.csv
```

The script is the source of truth for numerical results reported in the revised manuscript. Results should not be copied from earlier notebooks or releases.
