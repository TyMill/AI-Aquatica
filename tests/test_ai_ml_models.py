import unittest
import pandas as pd
import numpy as np
from ai_aquatica.ml_analysis import (
    train_linear_regression,
    train_logistic_regression,
    train_classification_model,
    evaluate_classification_model,
    perform_clustering,
    detect_anomalies
)

class TestMLAnalysis(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        self.X = self.data[['feature1', 'feature2']]
        self.y = self.data['target']

    def test_train_linear_regression(self):
        model = train_linear_regression(self.X, self.y)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'coef_'))

    def test_train_logistic_regression(self):
        model = train_logistic_regression(self.X, self.y)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'coef_'))

    def test_train_classification_model(self):
        model = train_classification_model(self.X, self.y, model_type='decision_tree')
        self.assertIsNotNone(model)

    def test_evaluate_classification_model(self):
        model = train_classification_model(self.X, self.y, model_type='decision_tree')
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
        evaluation = evaluate_classification_model(model, X_test, y_test)
        self.assertIn('confusion_matrix', evaluation)
        self.assertIn('precision', evaluation)
        self.assertIn('recall', evaluation)
        self.assertIn('f1_score', evaluation)

    def test_perform_clustering(self):
        model, labels = perform_clustering(self.X, n_clusters=3, algorithm='kmeans')
        self.assertIsNotNone(model)
        self.assertIsInstance(labels, np.ndarray)

    def test_detect_anomalies(self):
        labels = detect_anomalies(self.X, method='isolation_forest')
        self.assertIsInstance(labels, np.ndarray)

if __name__ == '__main__':
    unittest.main()
