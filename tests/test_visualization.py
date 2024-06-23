import unittest
import pandas as pd
import numpy as np
from ai_aquatica.data_visualization import (
    plot_line,
    plot_bar,
    plot_pie,
    plot_scatter,
    plot_heatmap,
    plot_pca,
    plot_tsne,
    plot_interactive_bubble
)

class TestDataVisualization(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'size': np.random.randint(1, 100, 100)
        })

    def test_plot_line(self):
        try:
            plot_line(self.data, 'feature1', 'feature2')
        except Exception as e:
            self.fail(f"plot_line() raised {e}")

    def test_plot_bar(self):
        try:
            plot_bar(self.data, 'category', 'size')
        except Exception as e:
            self.fail(f"plot_bar() raised {e}")

    def test_plot_pie(self):
        try:
            plot_pie(self.data, 'category')
        except Exception as e:
            self.fail(f"plot_pie() raised {e}")

    def test_plot_scatter(self):
        try:
            plot_scatter(self.data, 'feature1', 'feature2')
        except Exception as e:
            self.fail(f"plot_scatter() raised {e}")

    def test_plot_heatmap(self):
        try:
            plot_heatmap(self.data)
        except Exception as e:
            self.fail(f"plot_heatmap() raised {e}")

    def test_plot_pca(self):
        try:
            plot_pca(self.data[['feature1', 'feature2']])
        except Exception as e:
            self.fail(f"plot_pca() raised {e}")

    def test_plot_tsne(self):
        try:
            plot_tsne(self.data[['feature1', 'feature2']])
        except Exception as e:
            self.fail(f"plot_tsne() raised {e}")

    def test_plot_interactive_bubble(self):
        try:
            plot_interactive_bubble(self.data, 'feature1', 'feature2', 'size', 'category')
        except Exception as e:
            self.fail(f"plot_interactive_bubble() raised {e}")

if __name__ == '__main__':
    unittest.main()
