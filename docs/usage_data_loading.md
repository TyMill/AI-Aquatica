# 🗂️ Usage – Data Loading (AI-Aquatica)

This guide demonstrates how to use the `data_loading` module to import data from multiple sources such as CSV, Excel, JSON, SQLite, MongoDB, and APIs.

---

## 1. 📦 Importing

```python
from ai_aquatica.io.loaders import (
    load_csv,
    load_excel,
    load_json,
    load_sql,
    load_mongo,
    load_api
)
```

---

## 2. 📄 Load from CSV

```python
df_csv = load_csv("data/sample.csv")
```

---

## 3. 📊 Load from Excel

```python
df_excel = load_excel("data/sample.xlsx", sheet_name=0)
```

---

## 4. 🧾 Load from JSON

```python
df_json = load_json("data/sample.json")
```

---

## 5. 🗃️ Load from SQLite database

```python
query = "SELECT * FROM water_quality"
db_path = "data/odra_data.sqlite"
df_sql = load_sql(query, db_path)
```

---

## 6. 🍃 Load from MongoDB

```python
df_mongo = load_mongo(
    collection_name="measurements",
    db_name="aquatic_db",
    mongo_uri="mongodb://localhost:27017/"
)
```

---

## 7. 🌐 Load from API

```python
url = "https://api.environmentaldata.org/water"
params = {"location": "Odra"}
df_api = load_api(url, params=params)
```

---

## 🔍 Notes

- All functions return a `pandas.DataFrame`.
- Ensure appropriate access rights and credentials when accessing APIs or MongoDB.
- Excel sheets are indexed from 0.

