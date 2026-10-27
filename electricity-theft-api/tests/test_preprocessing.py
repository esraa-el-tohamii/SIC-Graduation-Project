"""
test_preprocessing.py — Unit tests for the preprocessing module.

These tests do NOT require the trained model artifacts.
They verify that the stateless per-consumer transformations match the
exact behaviour implemented in the notebook (Cells 62–64, 74).
"""

import math
import numpy as np
import pytest

from app.config import EXPECTED_READINGS, NUM_WEEKS, PAD_DAYS
from app.preprocessing import (
    fill_missing,
    cap_outliers,
    minmax,
    preprocess_single,
    preprocess_batch,
)


# ── fill_missing ─────────────────────────────────────────────────────────

class TestFillMissing:
    def test_no_nan_passthrough(self):
        X = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = fill_missing(X)
        np.testing.assert_array_almost_equal(result, X)

    def test_interior_nan_filled_with_neighbour_average(self):
        X = np.array([[2.0, np.nan, 4.0]], dtype=np.float32)
        result = fill_missing(X)
        assert not np.isnan(result).any()
        assert math.isclose(result[0, 1], 3.0, abs_tol=1e-5)

    def test_leading_nan_filled_with_right_neighbour(self):
        # prev is also NaN → avg is NaN → fallback to 0.0
        X = np.array([[np.nan, 5.0, 6.0]], dtype=np.float32)
        result = fill_missing(X)
        assert not np.isnan(result).any()
        # prev=NaN, nxt=5 → avg=NaN (both must be present)  → 0.0
        assert result[0, 0] == 0.0

    def test_trailing_nan_falls_back_to_zero(self):
        X = np.array([[3.0, 4.0, np.nan]], dtype=np.float32)
        result = fill_missing(X)
        assert result[0, 2] == 0.0

    def test_all_nan_row_becomes_zeros(self):
        X = np.full((1, 5), np.nan, dtype=np.float32)
        result = fill_missing(X)
        assert not np.isnan(result).any()
        np.testing.assert_array_equal(result, np.zeros((1, 5)))

    def test_does_not_modify_input(self):
        X = np.array([[1.0, np.nan, 3.0]], dtype=np.float32)
        X_copy = X.copy()
        fill_missing(X)
        np.testing.assert_array_equal(X, X_copy)


# ── cap_outliers ──────────────────────────────────────────────────────────

class TestCapOutliers:
    def test_no_outliers_passthrough(self):
        X = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        result = cap_outliers(X)
        assert result.max() <= X.mean() + 2 * X.std() + 1e-5

    def test_extreme_outlier_is_capped(self):
        X = np.array([[1.0, 1.0, 1.0, 1.0, 1000.0]], dtype=np.float32)
        result = cap_outliers(X)
        mu = X.mean()
        sd = X.std()
        assert result.max() <= mu + 2 * sd + 1e-4

    def test_shape_preserved(self):
        X = np.random.rand(10, 50).astype(np.float32)
        result = cap_outliers(X)
        assert result.shape == X.shape


# ── minmax ────────────────────────────────────────────────────────────────

class TestMinmax:
    def test_output_in_0_1(self):
        X = np.array([[3.0, 6.0, 9.0]], dtype=np.float32)
        result = minmax(X)
        assert result.min() >= 0.0 - 1e-6
        assert result.max() <= 1.0 + 1e-6

    def test_flat_consumer_stays_zero(self):
        X = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
        result = minmax(X)
        np.testing.assert_array_almost_equal(result, np.zeros((1, 3)))

    def test_shape_preserved(self):
        X = np.random.rand(7, 100).astype(np.float32)
        result = minmax(X)
        assert result.shape == X.shape


# ── preprocess_single ─────────────────────────────────────────────────────

class TestPreprocessSingle:
    def _make_readings(self, n=EXPECTED_READINGS, missing_ratio=0.0):
        """Create a list of readings with a controlled number of nulls."""
        readings = [float(i % 100) for i in range(n)]
        n_missing = int(n * missing_ratio)
        for i in range(0, n_missing * 3, 3):
            if i < n:
                readings[i] = None
        return readings

    def test_correct_length(self):
        readings = self._make_readings()
        stack_row, x_1d, x_2d, miss_pct, miss_frac = preprocess_single(readings)
        assert len(stack_row) == EXPECTED_READINGS + 2   # 1036

    def test_x_1d_shape(self):
        readings = self._make_readings()
        _, x_1d, _, _, _ = preprocess_single(readings)
        assert x_1d.shape == (1, EXPECTED_READINGS + 1)  # (1, 1035)

    def test_x_2d_shape(self):
        readings = self._make_readings()
        _, _, x_2d, _, _ = preprocess_single(readings)
        assert x_2d.shape == (1, 2, NUM_WEEKS, 7)         # (1, 2, 148, 7)

    def test_x_2d_clean_channel_in_0_1(self):
        readings = self._make_readings()
        _, _, x_2d, _, _ = preprocess_single(readings)
        clean_ch = x_2d[0, 0]
        assert clean_ch.min() >= 0.0 - 1e-5
        assert clean_ch.max() <= 1.0 + 1e-5

    def test_x_2d_mask_channel_is_binary(self):
        readings = self._make_readings(missing_ratio=0.1)
        _, _, x_2d, _, _ = preprocess_single(readings)
        mask_ch = x_2d[0, 1]
        unique_vals = set(mask_ch.flatten().tolist())
        assert unique_vals.issubset({0.0, 1.0})

    def test_missing_pct_correct(self):
        n_missing = 103   # exactly 10%
        readings = [1.0] * EXPECTED_READINGS
        for i in range(n_missing):
            readings[i] = None
        _, _, _, miss_pct, miss_frac = preprocess_single(readings)
        expected_pct = n_missing / EXPECTED_READINGS * 100
        assert math.isclose(miss_pct, expected_pct, rel_tol=1e-5)
        assert math.isclose(miss_frac, n_missing / EXPECTED_READINGS, rel_tol=1e-5)

    def test_stack_row_cnn_placeholder_is_zero(self):
        readings = self._make_readings()
        stack_row, _, _, _, _ = preprocess_single(readings)
        assert stack_row[-1] == 0.0  # CNN_Prediction placeholder

    def test_stack_row_missing_feature_is_fraction(self):
        readings = [1.0] * EXPECTED_READINGS
        for i in range(103):
            readings[i] = None
        stack_row, _, _, miss_pct, miss_frac = preprocess_single(readings)
        assert math.isclose(stack_row[-2], miss_frac, rel_tol=1e-6)
        assert math.isclose(stack_row[-2], miss_pct / 100.0, rel_tol=1e-6)

    def test_stack_row_raw_readings_preserved(self):
        """Raw readings in stack_row must equal the input (NaN where null)."""
        readings = [1.0, None, 3.0] + [2.0] * (EXPECTED_READINGS - 3)
        stack_row, _, _, _, _ = preprocess_single(readings)
        assert math.isclose(stack_row[0], 1.0)
        assert math.isnan(stack_row[1])   # None → NaN preserved
        assert math.isclose(stack_row[2], 3.0)

    def test_no_nan_in_x_1d(self):
        readings = self._make_readings(missing_ratio=0.15)
        _, x_1d, _, _, _ = preprocess_single(readings)
        assert not np.isnan(x_1d).any()

    def test_no_nan_in_x_2d_clean_channel(self):
        readings = self._make_readings(missing_ratio=0.20)
        _, _, x_2d, _, _ = preprocess_single(readings)
        assert not np.isnan(x_2d[:, 0]).any()

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError, match="1034"):
            preprocess_single([1.0] * (EXPECTED_READINGS - 1))

    def test_rejects_infinite_value(self):
        readings = [1.0] * EXPECTED_READINGS
        readings[5] = float("inf")
        with pytest.raises(ValueError, match="infinite"):
            preprocess_single(readings)

    def test_rejects_negative_inf(self):
        readings = [1.0] * EXPECTED_READINGS
        readings[5] = float("-inf")
        with pytest.raises(ValueError, match="infinite"):
            preprocess_single(readings)

    def test_all_null_accepted(self):
        """All readings missing is a degenerate but valid input."""
        readings = [None] * EXPECTED_READINGS
        stack_row, x_1d, x_2d, miss_pct, miss_frac = preprocess_single(readings)
        assert math.isclose(miss_pct, 100.0)
        assert math.isclose(miss_frac, 1.0)

    def test_all_zero_readings(self):
        readings = [0.0] * EXPECTED_READINGS
        stack_row, x_1d, x_2d, miss_pct, _ = preprocess_single(readings)
        assert miss_pct == 0.0
        # minmax of flat-zero consumer → zeros
        clean_ch = x_2d[0, 0]
        assert clean_ch.max() == 0.0


# ── preprocess_batch ──────────────────────────────────────────────────────

class TestPreprocessBatch:
    def test_batch_shapes_consistent(self):
        N = 4
        readings_list = [[float(j) for j in range(EXPECTED_READINGS)]] * N
        stack_rows, x_1d_b, x_2d_b, miss_pcts = preprocess_batch(readings_list)
        assert stack_rows.shape == (N, EXPECTED_READINGS + 2)
        assert x_1d_b.shape == (N, EXPECTED_READINGS + 1)
        assert x_2d_b.shape == (N, 2, NUM_WEEKS, 7)
        assert miss_pcts.shape == (N,)

    def test_batch_results_match_single(self):
        readings = [float(i % 50) for i in range(EXPECTED_READINGS)]
        single = preprocess_single(readings)
        batch = preprocess_batch([readings, readings])
        np.testing.assert_array_almost_equal(batch[1][0], single[1][0])
        np.testing.assert_array_almost_equal(batch[2][0], single[2][0])
