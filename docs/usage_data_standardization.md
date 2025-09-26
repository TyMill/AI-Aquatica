# 📏 Usage – Data Standardization (AI-Aquatica)

This guide demonstrates how to use the `data_standardization` module to transform your data for better performance in machine learning models.

---

## 1. 📦 Importing

```python
from ai_aquatica.data_standardization import (
    normalize_data,
    standardize_data,
    log_transform,
    sqrt_transform,
    boxcox_transform
)
```

---

## 2. 🧪 Sample data

```python
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'nitrate': [1.2, 3.4, 2.1, 5.6, 4.2],
    'phosphate': [0.5, 1.5, 0.9, 2.3, 1.8]
})
```

---

## 3. 🔄 Normalize data (0–1 range)

```python
normalized = normalize_data(data)
```

---

## 4. 🧮 Standardize data (mean = 0, std = 1)

```python
standardized = standardize_data(data)
```

---

## 5. 🔢 Log transformation

```python
log_transformed = log_transform(data)
```

---

## 6. 🔳 Square root transformation

```python
sqrt_transformed = sqrt_transform(data)
```

---

## 7. 📈 Box-Cox transformation (positive values only)

```python
boxcox_data = boxcox_transform(data)
```

---

## ⚠️ Notes

- Box-Cox requires all values to be strictly positive.
- Transformations help reduce skewness and improve model convergence.
- Apply after handling missing values and before training models.

