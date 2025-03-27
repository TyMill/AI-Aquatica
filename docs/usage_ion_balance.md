# ⚖️ Usage – Ion Balance (AI-Aquatica)

This guide demonstrates how to calculate ion balance, identify potential chemical inconsistencies, and perform corrections using the `ion_balance` module.

---

## 1. 📦 Importing

```python
from ai_aquatica.ion_balance import (
    calculate_ion_balance,
    identify_potential_errors,
    correct_ion_discrepancies
)
```

---

## 2. 💧 Sample dataset

```python
import pandas as pd

data = pd.DataFrame({
    'Ca2+': [2.1, 1.9, 2.3],
    'Mg2+': [1.1, 0.8, 1.0],
    'Na+': [0.7, 0.6, 0.9],
    'Cl-': [1.8, 1.7, 1.9],
    'SO4--': [1.5, 1.3, 1.4],
    'HCO3-': [1.0, 0.9, 1.2]
})

cations = ['Ca2+', 'Mg2+', 'Na+']
anions = ['Cl-', 'SO4--', 'HCO3-']
```

---

## 3. ⚖️ Calculate ion balance

```python
balanced_data = calculate_ion_balance(data, cations, anions)
print(balanced_data[['Cations_Sum', 'Anions_Sum', 'Ion_Balance']])
```

---

## 4. 🚨 Identify potential analytical errors

```python
flagged_data = identify_potential_errors(balanced_data, threshold=5.0)
print(flagged_data[['Ion_Balance', 'Potential_Error']])
```

---

## 5. 🛠️ Correct discrepancies

```python
corrected_data = correct_ion_discrepancies(balanced_data, cations, anions)
```

---

## 🧠 Notes

- Ion balance is reported as a percentage: `(cations - anions)/(cations + anions) * 100`
- A balance error > 5% typically indicates problems with sample or measurement.
- Corrections are done proportionally across ions.

