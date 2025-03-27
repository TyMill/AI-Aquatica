# 🧼 Usage – Data Cleaning (AI-Aquatica)

This guide demonstrates how to use the `data_cleaning` module to prepare and clean water quality datasets before applying modeling or visualization.

---

## 1. 📦 Importing

```python
from ai_aquatica.data_cleaning import (
    remove_duplicates,
    handle_missing_values,
    normalize_data,
    standardize_data
)
```

---

## 2. 🧪 Example dataset

```python
import pandas as pd
import numpy as np

# Example dataset with missing values and duplicates
data = pd.DataFrame({
    'pH': [7.0, 6.8, np.nan, 7.2, 7.0],
    'NO3': [1.5, 1.7, 1.6, np.nan, 1.5]
})

# Add duplicate row
data.loc[5] = data.loc[0]
```

---

## 3. ❌ Remove duplicates

```python
data_no_duplicates = remove_duplicates(data)
```

---

## 4. 🕳️ Fill missing values

### Mean strategy
```python
data_filled_mean = handle_missing_values(data, strategy='mean')
```

### Median strategy
```python
data_filled_median = handle_missing_values(data, strategy='median')
```

### Interpolation
```python
data_filled_interpolated = handle_missing_values(data, strategy='interpolate')
```

---

## 5. 📏 Normalize data (range [0, 1])

```python
data_normalized = normalize_data(data_filled_mean)
```

---

## 6. 📊 Standardize data (Z-score)

```python
data_standardized = standardize_data(data_filled_mean)
```

---

## 🔎 Notes
- All functions return pandas DataFrames.
- Make sure to handle missing values before normalizing or standardizing.
- Works on numerical data only.

