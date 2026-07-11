# 📘 Usage – AI Models (AI-Aquatica)

This guide shows how to use the AI/ML functionalities in the `ml_analysis` module of the AI-Aquatica library.

---

## 1. 🔧 Importing

```python
from ai_aquatica.modeling import (
    train_linear_regression,
    train_logistic_regression,
    train_classification_model,
    evaluate_classification_model,
    perform_clustering,
    plot_clusters,
    detect_anomalies,
    generate_synthetic_data
)
```

---

## 2. 📊 Dataset preparation

```python
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'target': np.random.randint(0, 2, 100)
})

X = data[['feature1', 'feature2']]
y = data['target']
```

---

## 3. 📈 Linear regression

```python
model = train_linear_regression(X, y)
print("Coefficients:", model.coef_)
```

---

## 4. ✅ Classification (Decision Tree, SVM, KNN, Random Forest)

```python
model = train_classification_model(X, y, model_type='random_forest')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
metrics = evaluate_classification_model(model, X_test, y_test)
print(metrics)
```

---

## 5. 🧬 Clustering (KMeans/DBSCAN)

```python
cluster_model, labels = perform_clustering(X, algorithm='kmeans', n_clusters=3)
plot_clusters(X, labels)
```

---

## 6. 🔍 Anomaly detection (Isolation Forest / LOF)

```python
anomalies = detect_anomalies(X, method='isolation_forest')
print("Detected:", anomalies)
```

---

## 7. 🧪 Generate synthetic data with GAN

```python
synthetic = generate_synthetic_data(X, model_type='gan', epochs=100)
synthetic.head()
```

---

## 📘 Notes

- All models return standard scikit-learn or Keras objects.
- You can further evaluate models using metrics from `sklearn.metrics`.

---

## 🧠 Tip

Use these tools in combination with preprocessing utilities such as `ai_aquatica.preprocessing.cleaning` or `ai_aquatica.preprocessing.transformations`, along with `ai_aquatica.visualization.plots`, for full pipelines!
