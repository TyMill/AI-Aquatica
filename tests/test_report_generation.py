import unittest
import pandas as pd
from ai_aquatica.report_generation import generate_statistical_report, generate_interpretation_report, suggest_further_analysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os

class TestReportGeneration(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'target': [0, 1, 0, 1, 0]
        })
        self.X = self.data[['feature1', 'feature2']]
        self.y = self.data['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)

    def tearDown(self):
        for report in ['statistical_report.html', 'interpretation_report.html', 'further_analysis_report.html', 'heatmap.png']:
            if os.path.exists(report):
                os.remove(report)

    def test_generate_statistical_report(self):
        generate_statistical_report(self.data)
        self.assertTrue(os.path.exists('statistical_report.html'))

    def test_generate_interpretation_report(self):
        generate_interpretation_report(self.data, self.model, self.X_test, self.y_test)
        self.assertTrue(os.path.exists('interpretation_report.html'))

    def test_suggest_further_analysis(self):
        suggest_further_analysis(self.data)
        self.assertTrue(os.path.exists('further_analysis_report.html'))

if __name__ == '__main__':
    unittest.main()
