# AI-Aquatica v2.3.0 validation record

Reference validation date: 2026-08-27

Reference validation environment:

- Python 3.13.5
- Linux 6.18.35 x86_64
- pandas 2.2.3
- NumPy 2.3.5
- scikit-learn 1.8.0
- matplotlib 3.10.8
- SciPy 1.17.0
- statsmodels 0.14.6

The authors should rerun the same commands in the final local/CI release environment before publication and retain the resulting logs with the release.

## Static analysis and automated tests

Release lint command:

```bash
python -m ruff check .
```

Expected result: `All checks passed!`

Reference automated-test result:

```text
76 passed, 1 skipped
```

The skipped test concerns optional TensorFlow functionality when TensorFlow is not installed.

Whole-package coverage in this environment: 72.3%.

Selected module coverage:

- core pipeline: 79%;
- hydrochemical module: 82%;
- HTML reporting: 95%;
- command-line interface: 95%.

The CI coverage gate is 70%.

## Package build

The source distribution and universal wheel build successfully. The wheel can be installed independently from the source checkout and reports version `2.3.0`.

Expected artifacts:

```text
ai_aquatica-2.3.0.tar.gz
ai_aquatica-2.3.0-py3-none-any.whl
```

## Real-dataset workflow

Command:

```bash
python examples/real_dataset_workflow.py --output validation/real_dataset --bootstrap 1000
```

Dataset:

- 148 observations;
- 30 original variables;
- four stations, 37 observations per station;
- 37 ordered sampling campaigns represented by `month`;
- no missing values in the released CSV.

### Hydrochemical diagnostics

Selected cations: Ca, Mg, NH4.

Selected anions: Cl, SO4, NO3, NO2 and HCO3 derived from alkalinity.

In the released dataset, NO3, NO2, and NH4 are reported as mg N/L. Their charge conversion therefore uses the explicit `NITROGEN_AS_N_MASS_PER_MEQ` mapping (14.0067 mg N/meq), while Ca, Mg, Cl, SO4, and derived HCO3 use their ion-based mass-per-meq factors.

- evaluable samples: 148;
- indeterminate samples: 0;
- samples above the 10% CBE threshold: 104;
- mean absolute CBE: 17.3127%;
- median absolute CBE: 16.2037%;
- maximum absolute CBE: 46.1783%.

The pre-flight diagnostic identifies Na and K as missing common major ions. CBE therefore describes closure of the available selected ion set and must not be interpreted as complete conventional major-ion closure.

### Station classification: campaign-grouped evaluation

Validation: 5-fold GroupKFold using `month` as the campaign group.

- accuracy: 0.9257;
- 95% group-cluster bootstrap CI: 0.8851-0.9595;
- balanced accuracy: 0.9257;
- macro precision: 0.9254;
- macro recall: 0.9257;
- macro F1: 0.9253;
- weighted F1: 0.9253.

All preprocessing was fitted within each training fold. PCA was exploratory and was not used by the Random Forest. Confidence intervals resample complete validation groups rather than individual observations.

### QC-filtered classification sensitivity analysis

Only samples with `Ion_Balance_Status == "acceptable"` were retained.

- samples: 44;
- validation: 4-fold GroupKFold by `month`;
- accuracy: 0.8864;
- 95% group-cluster bootstrap CI: 0.7907-0.9714;
- balanced accuracy: 0.7045;
- macro F1: 0.6893;
- weighted F1: 0.8552.

The reduced macro performance reflects the strongly imbalanced number of acceptable samples by station, especially only three accepted observations for station r2. This analysis should be reported as a sensitivity analysis, not as the primary classifier.

### Chlorophyll-a regression

Validation: 5-fold GroupKFold using `month` as the campaign group.

- R²: 0.1922;
- 95% group-cluster bootstrap CI: -0.0132 to 0.3520;
- MAE: 14.3768 µg/L;
- RMSE: 17.8882 µg/L.

Mean-prediction baseline:

- R²: -0.0139;
- MAE: 16.5820 µg/L;
- RMSE: 20.0407 µg/L.

The Random Forest improves on the baseline in point estimates, but the grouped confidence interval for R² includes zero. Predictive performance should therefore be described as modest and the regression retained as an illustrative workflow rather than an operational forecasting model.

## Dataset QA correction incorporated

The source-data audit identified a transcription error at station `r2` (Lake Słoneczne), campaign `27`: pH had been entered as `14.9`. The corresponding author checked the original record and confirmed the correct value as `7.49`. The released CSV has been corrected accordingly.

The reporting basis of nitrogen species was also confirmed: `NO3`, `NO2`, and `NH4` are reported as mg N/L. The reproducibility workflow now applies the correct elemental-nitrogen mass-per-meq conversion for charge-balance calculations.

## Remaining release actions

Before resubmission:

1. run the local release verification commands and preserve the log;
2. create and push Git tag `v2.3.0`;
3. publish the exact wheel/source release to PyPI;
4. archive the same GitHub tag in Zenodo;
5. insert the exact v2.3.0 Zenodo DOI into release metadata and the manuscript only after it exists;
6. use the files under `validation/real_dataset/` as the sole numerical source for the revised manuscript.
