"""Tests for the top-level public API re-exports.

These checks focus on a small representative subset of the package's
``__all__`` to ensure that the documented entry points remain importable
and callable without executing the heavy downstream dependencies.  Costly
I/O or plotting backends are mocked so that the test remains lightweight.
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd


def test_public_api_subset_is_accessible_and_callable():
    import ai_aquatica as aa

    public_symbols = {"import_csv", "plot_line", "train_linear_regression"}

    # The selected names should be advertised via ``__all__`` and be callable.
    for name in public_symbols:
        assert name in aa.__all__, f"{name} should be part of the public API"
        attr = getattr(aa, name)
        assert callable(attr), f"{name} should be callable"

    # ``import_csv`` is a thin wrapper around ``ai_aquatica.data_loading.load_csv``.
    with patch("ai_aquatica.data_import.load_csv", return_value={"rows": 1}) as mock_load_csv:
        result = aa.import_csv("dummy.csv")
        assert result == {"rows": 1}
        mock_load_csv.assert_called_once_with("dummy.csv")

    # ``plot_line`` delegates to the plotting backend – mock it to avoid real rendering.
    line_data = pd.DataFrame({"time": [0, 1], "value": [1.0, 2.0]})
    with patch("ai_aquatica.visualization.plt", autospec=True) as mock_plt:
        aa.plot_line(line_data, "time", "value")
        mock_plt.figure.assert_called_once()
        mock_plt.plot.assert_called_once_with(line_data["time"], line_data["value"])
        mock_plt.show.assert_called_once()

    # ``train_linear_regression`` comes from the machine-learning helpers.
    with patch("ai_aquatica.ai_ml_models.train_linear_regression", autospec=True) as mock_train:
        with patch.object(aa, "train_linear_regression", mock_train):
            aa.train_linear_regression([[0.0], [1.0]], [0, 1])
        mock_train.assert_called_once_with([[0.0], [1.0]], [0, 1])
