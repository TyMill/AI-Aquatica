import unittest
import pandas as pd
import numpy as np
from ai_aquatica.data_standardization import normalize_data, standardize_data, log_transform, sqrt_transform, boxcox_transform

class TestDataStandardization(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'A': [1, 2, 2, 4, 5],
            'B': [5, 6, 6, 7, 8],
            'C': [9, 10, 11, 12, 13]
        })

    def test_normalize_data(self):
        data_normalized = normalize_data(self.data)
        self.assertTrue((data_normalized.min() >= 0).all() and (data_normalized.max() <= 1).all())

    def test_standardize_data(self):
        data_standardized = standardize_data(self.data)
        self.assertAlmostEqual(data_standardized.mean().mean(), 0, places=1)
        self.assertAlmostEqual(data_standardized.std().mean(), 1, places=1)

    def test_log_transform(self):
        data_log_transformed = log_transform(self.data)
        self.assertFalse(np.isnan(data_log_transformed).values.any())

    def test_sqrt_transform(self):
        data_sqrt_transformed = sqrt_transform(self.data)
        self.assertFalse(np.isnan(data_sqrt_transformed).values.any())

    def test_boxcox_transform(self):
        data_boxcox_transformed = boxcox_transform(self.data)
        self.assertFalse(np.isnan(data_boxcox_transformed).values.any())

if __name__ == '__main__':
    unittest.main()
