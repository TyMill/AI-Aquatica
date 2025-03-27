# 📝 Usage – Report Generation (AI-Aquatica)

This guide shows how to use the `report_generation` module to automatically generate HTML reports with statistics, model interpretation, and analysis suggestions.

---

## 1. 📦 Importing

```python
from ai_aquatica.report_generation import (
    generate_statistical_report,
    generate_interpretation_report,
    suggest_further_analysis
)
```

---

## 2. 📊 Generate a statistical report

```python
import pandas as pd

# Sample data
data = pd.DataFrame({
    'NO3': [1.5, 1.7, 1.6, 1.8, 1.4],
    'pH': [7.0, 6.9, 7.1, 7.2, 6.8]
})

generate_statistical_report(data, report_path='statistical_report.html')
```

🖼️ Note: This creates a heatmap image and uses a Jinja2 template in the `templates/` directory.

---

## 3. 🤖 Generate model interpretation report

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

X = data[['NO3']]
y = [0, 1, 0, 1, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y)
model = DecisionTreeClassifier().fit(X_train, y_train)

generate_interpretation_report(data, model, X_test, y_test, report_path='interpretation_report.html')
```

---

## 4. 💡 Generate suggestions for further analysis

```python
suggest_further_analysis(data, report_path='further_analysis_report.html')
```

---

## 📘 Notes

- All functions save HTML files using Jinja2 templates.
- Templates (`.html`) must exist in a folder named `templates/`.
- Heatmap is saved as `heatmap.png` in the working directory.
- Ideal for creating reproducible reports from water quality analyses.

