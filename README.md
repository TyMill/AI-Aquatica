# 🌊 AI-Aquatica

[![PyPI version](https://img.shields.io/pypi/v/ai-aquatica?color=blue)](https://pypi.org/project/ai-aquatica/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15096947.svg)](https://doi.org/10.5281/zenodo.15096947)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://tymill.github.io/AI-Aquatica/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AI-Aquatica** is a comprehensive open-source Python library designed to analyze water quality data using advanced AI and statistical tools.  
It facilitates preprocessing, modeling, visualization, and reporting of hydrochemical datasets with minimal effort – empowering researchers and professionals in hydrology, ecology, and environmental monitoring.

---

## ✨ Features

- ✅ **Data Import**: Load datasets from CSV, Excel, JSON, SQL, NoSQL, and APIs.
- 🧼 **Data Cleaning**: Remove duplicates and handle missing values via multiple strategies.
- 📏 **Data Standardization**: Normalize and standardize data (Z-score, MinMax, log, sqrt, Box-Cox).
- 🧠 **Missing Data Imputation**: Fill gaps with:
  - Mean, Median, Mode
  - KNN Imputer
  - Regression Imputer
  - Autoencoder Neural Network
- ⚖️ **Ion Balance**: Detect chemical inconsistencies and auto-correct based on ionic ratios.
- 📊 **Statistical Analysis**: Get descriptive statistics, correlation matrices, ANOVA, time series decomposition.
- 🤖 **AI/ML Modeling**:
  - Regression & Classification (Logistic, SVM, Tree, RF)
  - Clustering (KMeans, DBSCAN)
  - Anomaly Detection (LOF, Isolation Forest)
  - Synthetic Data (GAN-based generation)
- 📈 **Visualization**:
  - Basic: Line, Bar, Pie, Scatter, Heatmaps
  - Advanced: PCA, t-SNE, Interactive Bubble Charts
- 📝 **Report Generation**:
  - Automatic HTML reports (statistics, ML evaluation, recommendations)

---

## 🛠 Installation

```bash
pip install ai-aquatica
```

Or from GitHub:

```bash
git clone https://github.com/TyMill/AI-Aquatica.git
cd AI-Aquatica
pip install -e .
```

> Full guide: [installation.md](https://tymill.github.io/AI-Aquatica/installation)

---

## 📘 Documentation

Read the full documentation on **GitHub Pages**:  
👉 [https://tymill.github.io/AI-Aquatica/](https://tymill.github.io/AI-Aquatica/)

Explore individual usage examples:
- `usage_data_cleaning.md`
- `usage_data_loading.md`
- `usage_missing_data.md`
- `usage_statistical_analysis.md`
- ... and more!

---

## 💡 Quick Start Example

```python
from ai_aquatica.ai_models import train_classification_model
import pandas as pd
import numpy as np

# Create mock dataset
df = pd.DataFrame({
    'NO3': np.random.rand(100),
    'pH': np.random.rand(100),
    'target': np.random.randint(0, 2, 100)
})

X = df[['NO3', 'pH']]
y = df['target']

model = train_classification_model(X, y, model_type='random_forest')
print("Model trained successfully.")
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to fork the repo and submit a pull request.  
We especially welcome:
- New preprocessing or AI models
- Example notebooks / visual dashboards
- Dataset integrations

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## 🙏 Acknowledgments

Special thanks to:
- Open-source contributors
- Environmental data science community
- University of Szczecin & BNP Paribas for ongoing support

---

📫 Questions? Suggestions? Open an issue or email the maintainer.
