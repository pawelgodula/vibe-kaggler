# Tests for target transformation by feature utilities.

import unittest
import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

# Adjust import path as necessary
try:
    from vibe_kaggler.utils.transform_target_by_feature import (
        normalize_target_by_feature,
        denormalize_predictions_by_feature
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from utils.transform_target_by_feature import (
        normalize_target_by_feature,
        denormalize_predictions_by_feature
    )

class TestTransformTargetByFeature(unittest.TestCase):

    def setUp(self):
        self.df = pl.DataFrame({
            'target': [10, 20, 30, 0, 50, 60, None],
            'feature': [2, 4, 5, 10, 0, None, 10],
            'other': [1, 1, 1, 1, 1, 1, 1]
        })
        # Expected for division:
        # 10/2=5, 20/4=5, 30/5=6, 0/10=0, 50/0=Error, 60/None=Null, None/10=Null
        # Expected for subtraction:
        # 10-2=8, 20-4=16, 30-5=25, 0-10=-10, 50-0=50, 60-None=Null, None-10=Null

    def test_normalize_divide_new_col(self):
        """Test normalization by division into a new column."""
        df_out = normalize_target_by_feature(
            self.df, 'target', 'feature', operation='divide', 
            new_target_col='target_norm', fill_value_on_error=None
        )
        expected_series = pl.Series("target_norm", [5.0, 5.0, 6.0, 0.0, None, None, None], dtype=pl.Float64)
        self.assertIn('target_norm', df_out.columns)
        self.assertIn('target', df_out.columns) # Original should still exist
        assert_series_equal(df_out['target_norm'], expected_series)
        
    def test_normalize_divide_replace_fill(self):
        """Test normalization by division, replacing original, with fill value."""
        df_out = normalize_target_by_feature(
            self.df, 'target', 'feature', operation='divide', 
            new_target_col=None, fill_value_on_error=-1.0
        )
        expected_series = pl.Series("target", [5.0, 5.0, 6.0, 0.0, -1.0, None, None], dtype=pl.Float64)
        self.assertIn('target', df_out.columns)
        assert_series_equal(df_out['target'], expected_series)
        
    def test_normalize_subtract_replace(self):
        """Test normalization by subtraction, replacing original."""
        df_out = normalize_target_by_feature(
            self.df, 'target', 'feature', operation='subtract', new_target_col=None
        )
        expected_series = pl.Series("target", [8.0, 16.0, 25.0, -10.0, 50.0, None, None], dtype=pl.Float64)
        self.assertIn('target', df_out.columns)
        assert_series_equal(df_out['target'], expected_series)

    def test_denormalize_divide(self):
        """Test denormalization for original division."""
        predictions = pl.Series("preds", [5.0, 5.0, 6.0, 0.0, -1.0, None, 0.5]) # Includes original error and null cases
        feature_vals = self.df['feature'] # [2, 4, 5, 10, 0, None, 10]
        denorm_out = denormalize_predictions_by_feature(predictions, feature_vals, operation='divide')
        # Expected: 5*2=10, 5*4=20, 6*5=30, 0*10=0, -1*0=0, None*None=None, 0.5*10=5
        expected_series = pl.Series("preds", [10.0, 20.0, 30.0, 0.0, 0.0, None, 5.0], dtype=pl.Float64)
        assert_series_equal(denorm_out, expected_series)
        
    def test_denormalize_subtract(self):
        """Test denormalization for original subtraction."""
        predictions = pl.Series("preds", [8.0, 16.0, 25.0, -10.0, 50.0, None, -15.0]) # Includes null case
        feature_vals = self.df['feature'] # [2, 4, 5, 10, 0, None, 10]
        denorm_out = denormalize_predictions_by_feature(predictions, feature_vals, operation='subtract')
        # Expected: 8+2=10, 16+4=20, 25+5=30, -10+10=0, 50+0=50, None+None=None, -15+10=-5
        expected_series = pl.Series("preds", [10.0, 20.0, 30.0, 0.0, 50.0, None, -5.0], dtype=pl.Float64)
        assert_series_equal(denorm_out, expected_series)

    def test_error_normalize_missing_col(self):
        with self.assertRaises(pl.exceptions.ColumnNotFoundError):
             normalize_target_by_feature(self.df, 'target', 'missing_feature')
        with self.assertRaises(pl.exceptions.ColumnNotFoundError):
             normalize_target_by_feature(self.df, 'missing_target', 'feature')

    def test_error_denormalize_mismatched_length(self):
        preds = pl.Series([1, 2, 3])
        features = pl.Series([1, 2])
        with self.assertRaisesRegex(ValueError, "Length of predictions Series must match"):
             denormalize_predictions_by_feature(preds, features)

    def test_error_normalize_non_numeric(self):
        df_bad = self.df.with_columns(pl.col("target").cast(pl.Utf8))
        with self.assertRaises(TypeError):
             normalize_target_by_feature(df_bad, 'target', 'feature')

    def test_error_denormalize_non_numeric(self):
        preds = pl.Series(["a", "b"])
        features = pl.Series([1, 2])
        with self.assertRaises(TypeError):
             denormalize_predictions_by_feature(preds, features)
        preds_ok = pl.Series([1.0, 2.0])
        features_bad = pl.Series(["x", "y"])
        with self.assertRaises(TypeError):
             denormalize_predictions_by_feature(preds_ok, features_bad)

if __name__ == '__main__':
    unittest.main() 