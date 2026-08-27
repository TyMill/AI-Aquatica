# SoftwareX major-revision implementation notes

Manuscript: SOFTX-D-26-00838

## Central revision strategy

The revised software is positioned as a domain-specific **machine-learning and hydrochemical analysis library**, not as a deep-learning platform. The primary contribution is the integration of robust regional data import, hydrochemical quality control, leakage-safe predictive validation, artifact export and HTML reporting.

## Reviewer issues addressed in code

### Exact release and API consistency

- Package, `__version__`, `CITATION.cff`, and `codemeta.json` synchronized at v2.3.0.
- `features=` is accepted by `select_features()`.
- `pipeline.data` is available alongside `pipeline.dataset.data`.
- One executable script is the source of all manuscript metrics.

A new PyPI/Zenodo release is still required before resubmission.

### Leakage prevention

- `.impute()` and `.scale()` now register transformations rather than fitting them on the full dataset.
- Predictive transformations are fitted only inside the training partition/fold.
- Exploratory PCA is explicitly separated from predictive PCA.

### Validation rigor

- Added holdout, stratified K-fold, GroupKFold and temporal holdout.
- Added confusion matrices, per-class metrics, macro and weighted metrics, fold metrics, predictions and group-aware bootstrap confidence intervals.
- Added RMSE and a mean-prediction baseline for regression.
- The manuscript workflow uses campaign-grouped validation by `month`.

### Hydrochemical safeguards

- Missing, non-numeric, infinite, negative and zero-total selected ion data are marked `indeterminate`.
- Indeterminate samples do not receive 0% CBE.
- Added explicit diagnostic reasons and completeness columns.
- Added `warn`, `filter`, `raise` and `ignore` QC policies for predictive modelling.

### Reproducible evidence

The workflow exports:

- hydrochemical diagnostics;
- validation summary;
- out-of-fold predictions;
- fold-level metrics;
- feature importance;
- confusion matrices;
- observed-versus-predicted plot;
- HTML report;
- machine-readable JSON metrics.

## Manuscript claims that must change

- Replace “AI-based” with “machine learning-based” where the text describes the current core implementation.
- Do not use optional TensorFlow functions as evidence for the main software contribution.
- Replace the previous accuracy 0.9189 with the grouped out-of-fold result 0.9257 only after the revised protocol is fully described.
- Replace the previous regression R² 0.2695 with grouped out-of-fold R² 0.1922, MAE 14.3768 and RMSE 17.8882.
- State that the regression point estimate improves on the mean baseline, but the group-cluster bootstrap R² interval includes zero; the regression remains an illustrative, modest-predictability example.
- Explain that the 104 CBE flags reflect the available selected ion set and missing Na/K.
- Report the QC-filtered classification only as a sensitivity analysis because accepted samples are imbalanced across stations.
- Add the formal CBE equation, equivalent weights, alkalinity conversion and indeterminate-status logic from `docs/hydrochemical_methodology.md`.
- Add the validation protocol from `docs/validation_protocol.md`.
- Add figures and tables from `validation/real_dataset/artifacts/`.

## Dataset metadata status

Author-supplied dataset metadata have now been incorporated: study locations, January 2020-January 2023 monthly sampling, analytical-method descriptions, reporting units, station-code mapping, spatial-disclosure statement, and CC BY 4.0 data licensing.

The source-data QA point was resolved: the pH value 14.9 at station r2, campaign 27 was confirmed from the original laboratory record to be a transcription error; the correct value is 7.49 and the released CSV has been corrected. The reporting basis of NO3, NO2 and NH4 was confirmed as mg N/L, and the SoftwareX workflow now applies the corresponding 14.0067 mg N/meq conversion.

## Dataset metadata completed

The released dataset is now documented as monthly monitoring of Lake Głębokie, Lake Słoneczne, and the inflow and outflow of Lake Rusałka in Szczecin, Poland, from January 2020 through January 2023. A station-code map, analytical-method table, reporting units, and CC BY 4.0 data license were added.
