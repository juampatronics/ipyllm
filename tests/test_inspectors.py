# Copyright 2026 Juan Pablo de la Cruz Gutiérrez
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the type-specific, bounded ``describe_object`` inspectors."""

import importlib
import unittest

from ipyllm.inspectors import MAX_INSPECT_CHARS, describe_object


def _has(module: str) -> bool:
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True


class NumpyInspectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _has("numpy"):
            raise unittest.SkipTest("numpy not installed")

    def test_summary_is_size_independent(self):
        import numpy as np

        text = describe_object(np.arange(24.0).reshape(4, 6))
        self.assertIn("shape=[4, 6]", text)
        self.assertIn("dtype=float64", text)
        self.assertIn("ndim=2", text)
        self.assertIn("strides=", text)
        self.assertNotIn("stats:", text)

    def test_detail_adds_descriptive_stats_and_fences(self):
        import numpy as np

        text = describe_object(np.array([1.0, 2.0, 3.0, 100.0]), "detail")
        for token in ("mean=", "std=", "min=", "median=", "max=", "iqr=", "outlier_fence="):
            self.assertIn(token, text)

    def test_full_adds_sample_values(self):
        import numpy as np

        text = describe_object(np.arange(10), "full")
        self.assertIn("sample:", text)

    def test_integer_stats_are_exact(self):
        import numpy as np

        text = describe_object(np.array([1, 2, 3]), "detail")
        self.assertIn("min=1", text)
        self.assertIn("max=3", text)

    def test_non_numeric_reports_no_stats(self):
        import numpy as np

        text = describe_object(np.array(["a", "b", "c"]), "detail")
        self.assertIn("not numeric", text)


class PandasInspectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _has("pandas"):
            raise unittest.SkipTest("pandas not installed")

    def test_summary_has_shape_dtypes_columns(self):
        import pandas as pd

        frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        text = describe_object(frame)
        self.assertIn("DataFrame shape=(2, 2)", text)
        self.assertIn("columns: a, b", text)
        self.assertNotIn("mean=", text)

    def test_detail_reports_missing_and_float_stats(self):
        import pandas as pd

        frame = pd.DataFrame({"x": [1.0, 2.0, None, 100.0]})
        text = describe_object(frame, "detail")
        self.assertIn("missing=1", text)
        self.assertIn("mean=", text)
        self.assertIn("outlier_fence=", text)

    def test_full_adds_head_preview(self):
        import pandas as pd

        frame = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        text = describe_object(frame, "full")
        self.assertIn("head(5):", text)

    def test_wide_frame_bounds_per_column_scan(self):
        import pandas as pd

        wide = pd.DataFrame({f"col{i}": [1.0, 2.0, 3.0] for i in range(60)})
        text = describe_object(wide, "detail")
        self.assertIn("first 40 of 60", text)
        self.assertIn("+20 more column(s) not scanned", text)

    def test_series_detail(self):
        import pandas as pd

        text = describe_object(pd.Series([1.0, 2.0, 3.0]), "detail")
        self.assertIn("Series shape=(3,)", text)
        self.assertIn("mean=", text)


class StandardLibraryInspectorTests(unittest.TestCase):
    def test_function_summary(self):
        def sample(x, y=2, *, flag=False):
            """Add x and y.

            Returns the sum.
            """
            return x + y

        text = describe_object(sample)
        self.assertIn("function", text)
        self.assertIn("sample(", text)
        self.assertIn("Add x and y.", text)

    def test_function_full_attempts_source(self):
        def sample(x):
            """Doc."""
            return x

        text = describe_object(sample, "full")
        self.assertIn("source:", text)

    def test_builtin_function(self):
        text = describe_object(len)
        self.assertIn("builtin", text)
        self.assertIn("signature:", text)

    def test_mapping_bounded(self):
        mapping = {f"key{i}": i for i in range(60)}
        text = describe_object(mapping, "detail")
        self.assertIn("length=60", text)
        self.assertIn("+36 more", text)

    def test_sequence_bounded(self):
        text = describe_object(list(range(60)), "detail")
        self.assertIn("length=60", text)
        self.assertIn("+48 more", text)

    def test_scalar_int_exact(self):
        self.assertEqual(describe_object(42), "int: 42")

    def test_scalar_float_compact(self):
        self.assertEqual(describe_object(3.5), "float: 3.5")

    def test_none(self):
        self.assertEqual(describe_object(None), "NoneType: None")


class FallbackTests(unittest.TestCase):
    def test_unknown_type_uses_bounded_repr(self):
        class Weird:
            def __repr__(self):
                return "x" * 10_000

        text = describe_object(Weird())
        self.assertIn("Weird", text)
        self.assertLess(len(text), 5_000)

    def test_hostile_repr_does_not_raise(self):
        class Broken:
            def __repr__(self):
                raise RuntimeError("boom")

        text = describe_object(Broken())
        self.assertIn("Broken", text)

    def test_invalid_detail_falls_back_to_summary(self):
        text = describe_object([1, 2, 3], detail="bogus")
        self.assertIn("length=3", text)
        self.assertNotIn("+", text)

    def test_output_never_exceeds_hard_cap(self):
        class Huge:
            def __repr__(self):
                return "y" * 100_000

        self.assertLessEqual(len(describe_object(Huge())), MAX_INSPECT_CHARS)


class MatplotlibInspectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _has("matplotlib"):
            raise unittest.SkipTest("matplotlib not installed")
        import matplotlib

        matplotlib.use("Agg")

    def test_figure_summary_and_artists(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])
        ax.bar([1, 2, 3], [1, 2, 3])
        ax.set_title("trend")
        try:
            summary = describe_object(fig)
            detail = describe_object(fig, "detail")
            self.assertIn("axes=1", summary)
            self.assertIn("title='trend'", detail)
            self.assertIn("Line2D×1", detail)
            self.assertIn("Rectangle×3", detail)
        finally:
            plt.close(fig)


class TorchInspectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _has("torch"):
            raise unittest.SkipTest("torch not installed")

    def test_tensor_summary_and_stats(self):
        import torch

        tensor = torch.tensor([1.0, 2.0, 3.0, 100.0], requires_grad=True)
        summary = describe_object(tensor)
        detail = describe_object(tensor, "detail")
        self.assertIn("Tensor shape=[4]", summary)
        self.assertIn("requires_grad=True", summary)
        self.assertIn("outlier_fence=", detail)

    def test_module_summary_and_parameters(self):
        from torch import nn

        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        text = describe_object(model)
        self.assertIn("nn.Module", text)
        self.assertIn("parameters: total=", text)
        self.assertIn("mode=", text)


class SklearnInspectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _has("sklearn"):
            raise unittest.SkipTest("scikit-learn not installed")

    def test_estimator_fitted_status(self):
        from sklearn.linear_model import LinearRegression

        estimator = LinearRegression()
        self.assertIn("fitted: no", describe_object(estimator))

    def test_fitted_estimator_reports_fitted_attrs(self):
        import numpy as np
        from sklearn.linear_model import LinearRegression

        estimator = LinearRegression().fit(
            np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0, 3.0])
        )
        text = describe_object(estimator)
        self.assertIn("fitted: yes", text)


if __name__ == "__main__":
    unittest.main()
