import unittest
import pandas as pd
from ai_aquatica.missing_data import (
    fill_missing_with_mean,
    fill_missing_with_median,
    fill_missing_with_mode,
    fill_missing_with_knn,
    fill_missing_with_regression,
    fill_missing_with_autoencoder
)

class TestMissingData(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'A': [1, 2, None, 4, 5],
            'B': [5, None, 6, 7, 8],
            'C': [9, 10, None, 12, 13]
        })

    def test_fill_missing_with_mean(self):
        data_filled = fill_missing_with_mean(self.data)
        self.assertFalse(data_filled.isnull().values.any())
        self.assertAlmostEqual(data_filled['A'].iloc[2], self.data['A'].mean(), places=1)

    def test_fill_missing_with_median(self):
        data_filled = fill_missing_with_median(self.data)
        self.assertFalse(data_filled.isnull().values.any())
        self.assertAlmostEqual(data_filled['A'].iloc[2], self.data['A'].median(), places=1)

    def test_fill_missing_with_mode(self):
        data_filled = fill_missing_with_mode(self.data)
        self.assertFalse(data_filled.isnull().values.any())
        self.assertEqual(data_filled['A'].iloc[2], self.data['A'].mode()[0])

    def test_fill_missing_with_knn(self):
        data_filled = fill_missing_with_knn(self.data)
        self.assertFalse(data_filled.isnull().values.any())

    def test_fill_missing_with_regression(self):
        data_filled = fill_missing_with_regression(self.data)
        self.assertFalse(data_filled.isnull().values.any())

    def test_fill_missing_with_autoencoder(self):
        data_filled = fill_missing_with_autoencoder(self.data)
        self.assertFalse(data_filled.isnull().values.any())

if __name__ == '__main__':
    unittest.main()
