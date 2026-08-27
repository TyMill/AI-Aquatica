# Predictive validation protocol

The primary AI-Aquatica modelling workflow uses scikit-learn pipelines so that all data-dependent preprocessing is fitted exclusively on training data.

## Leakage prevention

The following transformations are registered by the fluent API but are not fitted immediately:

- missing-value imputation;
- standardization;
- optional predictive PCA.

During holdout or cross-validation, the sequence is fitted independently inside each training partition:

```text
training partition
  -> median imputer fit
  -> standard scaler fit
  -> optional PCA fit
  -> Random Forest fit
  -> prediction on untouched evaluation partition
```

The evaluation data are never used to determine imputation statistics, scaling parameters, or predictive PCA components.

Exploratory PCA is treated separately. `pca(..., use_for_model=False)` produces descriptive scores and loadings but does not transform the predictors used by the Random Forest. When `use_for_model=True`, a separate PCA transformer is fitted within every training fold.

## Supported validation strategies

- `holdout`: stratified random holdout for classification and random holdout for regression;
- `stratified_kfold`: shuffled stratified K-fold validation for classification;
- `group_kfold`: K-fold validation with complete group separation;
- `temporal_holdout`: earlier ordered groups for training and the latest groups for evaluation.

The SoftwareX real-dataset workflow uses `GroupKFold` with `month` as the sampling-campaign group. All four station observations from a given campaign remain in the same fold, reducing the risk that correlated observations from the same campaign appear in training and evaluation data.

## Classification outputs

The pipeline exports:

- accuracy;
- balanced accuracy;
- macro precision;
- macro recall;
- macro F1;
- weighted F1;
- per-station precision, recall and F1;
- out-of-fold confusion matrix;
- fold-level metrics;
- out-of-fold predictions;
- 95% bootstrap confidence intervals;
- feature importance fitted after evaluation on the complete analysis set.

## Regression outputs

The pipeline exports:

- R²;
- MAE;
- RMSE;
- 95% bootstrap confidence intervals;
- fold-level metrics;
- out-of-fold observed and predicted values;
- observed-versus-predicted plot;
- residual summary;
- a `DummyRegressor(strategy="mean")` baseline evaluated using the same splits.

## Reproduction command

```bash
python examples/real_dataset_workflow.py --output outputs/real_dataset
```

For grouped validation, 95% confidence intervals are obtained by cluster bootstrap of complete validation groups (campaigns), preserving within-campaign dependence. For non-grouped validation, the bootstrap resamples out-of-sample observations.

The command exports machine-readable JSON and CSV results, figures, processed data, a validation summary and a standalone HTML report.
