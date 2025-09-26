import importlib
import sys
import unittest
from unittest import mock

import pandas as pd


class TestOptionalDependencies(unittest.TestCase):
    def test_missing_data_without_tensorflow(self):
        module_name = "ai_aquatica.missing_data"
        sys.modules.pop(module_name, None)

        real_import = importlib.import_module

        def fake_import(name, *args, **kwargs):
            if name.startswith("tensorflow"):
                raise ImportError("No module named 'tensorflow'")
            return real_import(name, *args, **kwargs)

        with mock.patch("importlib.import_module", side_effect=fake_import):
            module = importlib.import_module(module_name)
            module = importlib.reload(module)
            self.assertFalse(module.TENSORFLOW_AVAILABLE)
            with self.assertRaises(ImportError):
                module.fill_missing_with_autoencoder(pd.DataFrame({"A": [1, None, 3]}))

        sys.modules.pop(module_name, None)
        import ai_aquatica.missing_data as missing_data  # noqa: F401
        importlib.reload(missing_data)

    def test_ai_ml_models_without_tensorflow(self):
        module_name = "ai_aquatica.ai_ml_models"
        sys.modules.pop(module_name, None)

        real_import = importlib.import_module

        def fake_import(name, *args, **kwargs):
            if name.startswith("tensorflow"):
                raise ImportError("No module named 'tensorflow'")
            return real_import(name, *args, **kwargs)

        with mock.patch("importlib.import_module", side_effect=fake_import):
            module = importlib.import_module(module_name)
            module = importlib.reload(module)
            self.assertFalse(module.TENSORFLOW_AVAILABLE)
            with self.assertRaises(ImportError):
                module.generate_synthetic_data(pd.DataFrame({"A": [1, 2, 3]}))

        sys.modules.pop(module_name, None)
        import ai_aquatica.ai_ml_models as ai_ml_models  # noqa: F401
        importlib.reload(ai_ml_models)

    def test_visualization_without_plotly(self):
        module_name = "ai_aquatica.visualization"
        module = importlib.import_module(module_name)

        real_import = importlib.import_module

        def fake_import(name, *args, **kwargs):
            if name == "plotly.express":
                raise ImportError("No module named 'plotly'")
            return real_import(name, *args, **kwargs)

        with mock.patch("importlib.import_module", side_effect=fake_import):
            with self.assertWarns(RuntimeWarning):
                module.plot_interactive_bubble(
                    pd.DataFrame({
                        "x": [1, 2, 3],
                        "y": [4, 5, 6],
                        "size": [1, 2, 3],
                        "label": ["a", "b", "c"],
                    }),
                    "x",
                    "y",
                    "size",
                    "label",
                )


if __name__ == "__main__":
    unittest.main()
