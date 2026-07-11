# Real dataset workflow

AI-Aquatica includes a command-line example for real monitoring CSV files. The repository includes `examples/data/water_quality.csv`, the dataset used in the SoftwareX illustrative example, so the workflow can be reproduced without obtaining an external file. The workflow auto-detects common CSV conventions, normalizes column names, runs hydrochemical ion-balance quality control when the required chemistry columns are present, trains a station classifier when station labels are available, and exports a standalone HTML report.

```bash
python examples/real_dataset_workflow.py --output outputs/real_dataset
```

The script writes:

```text
outputs/real_dataset/ai_aquatica_real_dataset_report.html
outputs/real_dataset/processed_water_quality.csv
```

This example is useful for SoftwareX review because it demonstrates that the library can process non-trivial real monitoring exports, including semicolon-separated European CSV files with decimal commas and legacy encodings.
