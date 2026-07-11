# 🕳️ Usage – Missing Data (AI-Aquatica)

This guide demonstrates multiple strategies for handling missing values in environmental datasets using the `missing_data` module.

---

## 1. 📦 Importing

```python
from ai_aquatica.preprocessing.missing import (
    fill_missing_with_mean,
    fill_missing_with_median,
    fill_missing_with_mode,
    fill_missing_with_knn,
    fill_missing_with_regression,
    fill_missing_with_autoencoder
)
```

---

## 2. 🧪 Sample dataset with missing values

```python
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'pH': [7.1, 6.9, np.nan, 7.3, 7.0],
    'NO3': [1.5, np.nan, 1.7, 1.6, 1.8]
})
```

---

## 3. 📊 Simple statistical imputations

### Fill with mean
```python
mean_filled = fill_missing_with_mean(data)
```

### Fill with median
```python
median_filled = fill_missing_with_median(data)
```

### Fill with mode
```python
mode_filled = fill_missing_with_mode(data)
```

---

## 4. 🤖 AI/ML imputations

### K-Nearest Neighbors
```python
knn_filled = fill_missing_with_knn(data, n_neighbors=3)
```

### Regression Imputation
```python
regression_filled = fill_missing_with_regression(data)
```

### Autoencoder Neural Network
```python
autoencoder_filled = fill_missing_with_autoencoder(data)
```

---

## 📘 Notes

- Make sure the data contains only numerical features for ML methods.
- Autoencoder requires a basic imputation to start training.
- KNN and regression approaches benefit from scaled data.

