import unittest
import pandas as pd
from ai_aquatica.statistical_analysis import (
    calculate_basic_statistics,
    calculate_correlation_matrix,
    perform_anova,
    decompose_time_series
)

class TestStatisticalAnalysis(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'Do': [8.1, 7.8, 8.2, 7.9, 8.0],
            'pH': [7.0, 7.2, 7.1, 7.3, 7.2],
            'BOD': [3.0, 2.8, 3.2, 3.1, 3.0]
        })

    def test_calculate_basic_statistics(self):
        basic_stats = calculate_basic_statistics(self.data)
        self.assertIn('mean', basic_stats.columns)
        self.assertIn('50%', basic_stats.columns)
        self.assertIn('std', basic_stats.columns)
        self.assertIn('range', basic_stats.columns)

    def test_calculate_correlation_matrix(self):
        correlation_matrix = calculate_correlation_matrix(self.data)
        self.assertEqual(correlation_matrix.shape, (3, 3))

    def test_perform_anova(self):
        formula = 'Do ~ pH + BOD'
        anova_results = perform_anova(self.data, formula)
        self.assertIn('sum_sq', anova_results.columns)

    def test_decompose_time_series(self):
        time_series_data = pd.DataFrame({
            'Date': pd.date_range(start='1/1/2020', periods=12, freq='M'),
            'Do': [8.1, 7.8, 8.2, 7.9, 8.0, 7.7, 7.6, 7.8, 7.9, 8.1, 8.0, 7.9]
        })
        time_series_data.set_index('Date', inplace=True)
        decomposition = decompose_time_series(time_series_data, 'Do', model='additive', freq=12)
        self.assertIsNotNone(decomposition)

if __name__ == '__main__':
    unittest.main()
