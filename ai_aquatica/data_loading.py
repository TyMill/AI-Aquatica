import pandas as pd
import json
import sqlite3
import pymongo
import requests

def load_csv(file_path):
    """
    Load data from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def load_excel(file_path, sheet_name=0):
    """
    Load data from an Excel file.
    """
    try:
        data = pd.read_excel(file_path, sheet_name=sheet_name)
        return data
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None

def load_json(file_path):
    """
    Load data from a JSON file.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

def load_sql(sql_query, db_path):
    """
    Load data from an SQLite database.
    """
    try:
        conn = sqlite3.connect(db_path)
        data = pd.read_sql_query(sql_query, conn)
        conn.close()
        return data
    except Exception as e:
        print(f"Error loading data from SQLite database: {e}")
        return None

def load_mongo(collection_name, db_name, query={}, mongo_uri='mongodb://localhost:27017/'):
    """
    Load data from a MongoDB collection.
    """
    try:
        client = pymongo.MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        data = list(collection.find(query))
        client.close()
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading data from MongoDB: {e}")
        return None

def load_api(url, params=None, headers=None):
    """
    Load data from an API endpoint.
    """
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading data from API: {e}")
        return None
