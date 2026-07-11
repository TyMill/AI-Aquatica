# 📊 Usage – Statistical Analysis (AI-Aquatica)

This guide describes how to perform basic and advanced statistical analyses using the `statistical_analysis` module.

---

## 1. 📦 Importing

```python
from ai_aquatica.analysis.statistics import (
    calculate_basic_statistics,
    plot_distribution,
    plot_boxplot,
    calculate_correlation_matrix,
    perform_anova,
    decompose_time_series
)
```

---

## 2. 📈 Sample dataset

```python
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'NO3': [1.5, 1.7, 1.6, 1.8, 1.4],
    'pH': [7.0, 6.9, 7.1, 7.2, 6.8],
    'Region': ['A', 'A', 'B', 'B', 'A']
})
```

---

## 3. 🔹 Basic statistics

```python
stats = calculate_basic_statistics(data[['NO3', 'pH']])
print(stats)
```

---

## 4. 📊 Visualizations

### Histogram + KDE

```python
plot_distribution(data, 'NO3')
```

### Boxplot

```python
plot_boxplot(data, 'pH')
```

---

## 5. 🔗 Correlation matrix

```python
corr_matrix = calculate_correlation_matrix(data[['NO3', 'pH']])
print(corr_matrix)
```

---

## 6. 📊 ANOVA (Analysis of Variance)

```python
anova_results = perform_anova(data, 'NO3 ~ Region')
print(anova_results)
```

---

## 7. ⏱️ Time series decomposition

```python
# Requires datetime index and regular time steps
time_data = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=12, freq='M'),
    'NO3': np.random.rand(12)
}).set_index('date')

decomposition = decompose_time_series(time_data, 'NO3', model='additive', freq=12)
```

---

## 🧠 Notes

- ANOVA requires categorical variables in the formula (e.g., 'pH ~ Region').
- Time series decomposition needs a datetime index and consistent frequency.
- Visuals are shown using matplotlib and seaborn.

