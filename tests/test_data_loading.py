import unittest
import pandas as pd
from ai_aquatica.data_import import (
    import_csv,
    import_excel,
    import_json,
    import_from_sql,
    import_from_nosql
)
import sqlite3
import json
import os

class TestDataImport(unittest.TestCase):

    def setUp(self):
        # Tworzenie przykładowych plików do testów
        self.csv_file = 'test_data.csv'
        self.excel_file = 'test_data.xlsx'
        self.json_file = 'test_data.json'
        
        data = {
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        }
        
        pd.DataFrame(data).to_csv(self.csv_file, index=False)
        pd.DataFrame(data).to_excel(self.excel_file, index=False)
        with open(self.json_file, 'w') as f:
            json.dump(data, f)
        
        # Tworzenie przykładowej bazy danych SQL
        self.sql_file = 'test_data.db'
        conn = sqlite3.connect(self.sql_file)
        pd.DataFrame(data).to_sql('test_table', conn, index=False, if_exists='replace')
        conn.close()

    def tearDown(self):
        # Usuwanie przykładowych plików po testach
        os.remove(self.csv_file)
        os.remove(self.excel_file)
        os.remove(self.json_file)
        os.remove(self.sql_file)

    def test_import_csv(self):
        df = import_csv(self.csv_file)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (3, 2))

    def test_import_excel(self):
        df = import_excel(self.excel_file)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (3, 2))

    def test_import_json(self):
        df = import_json(self.json_file)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (3, 2))

    def test_import_from_sql(self):
        query = 'SELECT * FROM test_table'
        df = import_from_sql(self.sql_file, query)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (3, 2))

    def test_import_from_nosql(self):
        # To test NoSQL import, we would need a running NoSQL database like MongoDB.
        # For the purpose of this example, we will skip the actual test implementation.
        pass

if __name__ == '__main__':
    unittest.main()
