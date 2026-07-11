# 📊 Usage – Data Visualization (AI-Aquatica)

This guide presents how to use the `visualization` module to create basic and advanced plots for water quality and environmental datasets.

---

## 1. 📦 Importing

```python
from ai_aquatica.visualization.plots import (
    plot_line,
    plot_bar,
    plot_pie,
    plot_scatter,
    plot_heatmap,
    plot_pca,
    plot_tsne,
    plot_interactive_bubble
)
```

---

## 2. 📈 Sample dataset

```python
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'size': np.random.randint(1, 100, 100)
})
```

---

## 3. 📉 Basic visualizations

### Line plot
```python
plot_line(data, 'feature1', 'feature2')
```

### Bar plot
```python
plot_bar(data, 'category', 'size')
```

### Pie chart
```python
plot_pie(data, 'category')
```

### Scatter plot
```python
plot_scatter(data, 'feature1', 'feature2')
```

### Heatmap of correlations
```python
plot_heatmap(data[['feature1', 'feature2']])
```

---

## 4. 📊 Advanced visualizations

### PCA (Principal Component Analysis)
```python
plot_pca(data[['feature1', 'feature2']])
```

### t-SNE (Dimensionality Reduction)
```python
plot_tsne(data[['feature1', 'feature2']])
```

### Interactive Bubble Chart
```python
plot_interactive_bubble(data, 'feature1', 'feature2', 'size', 'category')
```

---

## 🧠 Notes

- Requires `plotly` for interactive bubble chart.
- PCA and t-SNE operate on numeric, scaled features.
- Always visualize your data before and after cleaning or modeling.

