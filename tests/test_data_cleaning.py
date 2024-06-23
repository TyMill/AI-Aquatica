import unittest
import pandas as pd
from ai_aquatica.data_cleaning import remove_duplicates, handle_missing_values, normalize_data, standardize_data

class TestDataCleaning(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'A': [1, 2, 2, 4, None],
            'B': [5, None, 6, 7, 8],
            'C': [9, 10, None, 12, 13]
        })

    def test_remove_duplicates(self):
        data_cleaned = remove_duplicates(self.data)
        self.assertEqual(len(data_cleaned), 4)

    def test_handle_missing_values_mean(self):
        data_filled = handle_missing_values(self.data, strategy='mean')
        self.assertFalse(data_filled.isnull().values.any())

    def test_handle_missing_values_median(self):
        data_filled = handle_missing_values(self.data, strategy='median')
        self.assertFalse(data_filled.isnull().values.any())

    def test_handle_missing_values_interpolate(self):
        data_filled = handle_missing_values(self.data, strategy='interpolate')
        self.assertFalse(data_filled.isnull().values.any())

    def test_normalize_data(self):
        data_filled = handle_missing_values(self.data, strategy='mean')
        data_normalized = normalize_data(data_filled)
        self.assertTrue((data_normalized.min() >= 0).all() and (data_normalized.max() <= 1).all())

    def test_standardize_data(self):
        data_filled = handle_missing_values(self.data, strategy='mean')
        data_standardized = standardize_data(data_filled)
        self.assertAlmostEqual(data_standardized.mean().mean(), 0, places=1)
        self.assertAlmostEqual(data_standardized.std().mean(), 1, places=1)

if __name__ == '__main__':
    unittest.main()
