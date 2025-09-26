# 🌊 AI-Aquatica

[![PyPI version](https://img.shields.io/pypi/v/ai-aquatica?color=blue)](https://pypi.org/project/ai-aquatica/)
[![Downloads](https://static.pepy.tech/badge/ai-aquatica)](https://pepy.tech/project/ai-aquatica)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15096947.svg)](https://doi.org/10.5281/zenodo.15096947)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://tymill.github.io/AI-Aquatica/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/ai-aquatica?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/ai-aquatica)
[![CI](https://github.com/TyMill/energicast/actions/workflows/ci.yml/badge.svg)](https://github.com/TyMill/AI-Aquatica/actions/workflows/ci.yml)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/TyMill/energicast)](https://github.com/TyMill/ai-aquatica/releases)

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

Optional extras provide deep-learning and interactive visualization support:

```bash
# Install TensorFlow-powered utilities
pip install "ai-aquatica[deep_learning]"

# Install Plotly-based interactive charts
pip install "ai-aquatica[interactive]"

# Or grab everything
pip install "ai-aquatica[all]"
```

Or from GitHub:

```bash
git clone https://github.com/TyMill/AI-Aquatica.git
cd AI-Aquatica
pip install -e .[all]
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
from ai_aquatica.ml_analysis import train_classification_model
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

# Train a random forest classification model using AI-Aquatica's helper
model = train_classification_model(X, y, model_type='random_forest')
print("Model trained successfully.")
```

### Report Generation Templates

AI-Aquatica ships with ready-to-use [Jinja2](https://jinja.palletsprojects.com/) templates stored in
`ai_aquatica/templates`. By default the report utilities render these files to create:

- `statistical_report.html` (plus an accompanying `heatmap.png` chart),
- `interpretation_report.html`,
- `further_analysis_report.html`.

The templates are available immediately after installation, but if you installed only the
minimal dependencies make sure `jinja2` is present:

```bash
pip install jinja2
```

You can point the report functions to your own template directory by passing the `template_dir`
parameter. The directory should contain files named like the bundled templates so the engine can
find them.

```python
from ai_aquatica.report_generation import generate_statistical_report

custom_templates = "/path/to/my/templates"  # folder with statistical_report_template.html, etc.
generate_statistical_report(
    data=df,
    report_path="reports/wq_stat_report.html",
    template_dir=custom_templates,
)
```

> Need more control? Copy the files from `ai_aquatica/templates`, modify them, and point
> `template_dir` to the folder containing your customized versions.

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
