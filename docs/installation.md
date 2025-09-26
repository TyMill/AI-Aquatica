# 📦 Installation – AI-Aquatica

AI-Aquatica is a Python library for water quality data analysis using modern AI and statistical methods.

---

## 🔧 Requirements

- Python 3.7+
- pip

---

## 🧪 Recommended (optional)

- Jupyter Notebook or JupyterLab
- Optional extras: TensorFlow for deep-learning utilities, Plotly for interactive charts

---

## 🛠️ Install from PyPI

```bash
pip install ai-aquatica
```

Optional extras bundle the heavier dependencies so you can decide what to install:

```bash
# Autoencoders, GANs and other TensorFlow features
pip install "ai-aquatica[deep_learning]"

# Interactive Plotly visualisations
pip install "ai-aquatica[interactive]"

# Everything at once
pip install "ai-aquatica[all]"
```

---

## 🧬 Clone from GitHub (development version)

```bash
git clone https://github.com/TyMill/AI-Aquatica.git
cd AI-Aquatica
pip install -e .[all]
```

---

## ✅ Test the installation

```python
import ai_aquatica
print("AI-Aquatica is installed successfully!")
```

---

## 🧠 Tip

To explore examples, go to the `examples/` folder or visit our documentation pages.

