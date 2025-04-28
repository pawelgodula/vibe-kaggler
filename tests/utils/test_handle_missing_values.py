import unittest
import polars as pl
from polars.testing import assert_frame_equal # Use polars testing utilities
import numpy as np # Still needed for setup
from utils.handle_missing_values import handle_missing_values

class TestHandleMissingValues(unittest.TestCase):

    def setUp(self):
        """Set up a sample Polars DataFrame with missing values."""
        self.data = {
            'num1': [1.0, 2.0, None, 4.0, 5.0], # Mean=3.0, Median=3.0
            'num2': [10.0, None, None, 40.0, 50.0], # Mean=33.33, Median=40.0
            'cat1': ['A', 'B', 'A', None, 'B'], # Mode='A' or 'B'
            'cat2': ['X', 'Y', 'X', 'X', None], # Mode='X'
            'no_missing': [1, 2, 3, 4, 5],
            'bool_col': [True, False, True, None, False] # Mean/Median/Zero invalid
        }
        # Explicitly define schema for consistent testing
        self.schema = {
             'num1': pl.Float64,
             'num2': pl.Float64,
             'cat1': pl.Utf8,
             'cat2': pl.Utf8,
             'no_missing': pl.Int64,
             'bool_col': pl.Boolean
        }
        self.df = pl.DataFrame(self.data, schema=self.schema)

    def test_mean_imputation_specific_features(self):
        """Test mean imputation on specified numeric features."""
        df_filled = handle_missing_values(self.df, strategy='mean', features=['num1'])
        self.assertFalse(df_filled['num1'].is_null().any())
        self.assertAlmostEqual(df_filled.item(2, 'num1'), 3.0) # Mean of [1, 2, 4, 5]
        self.assertTrue(df_filled['num2'].is_null().any()) # num2 untouched
        self.assertTrue(df_filled['cat1'].is_null().any()) # cat1 untouched

    def test_mean_imputation_all_features(self):
        """Test mean imputation on all applicable (numeric, non-bool) features."""
        df_filled = handle_missing_values(self.df, strategy='mean')
        self.assertFalse(df_filled['num1'].is_null().any())
        self.assertFalse(df_filled['num2'].is_null().any())
        self.assertAlmostEqual(df_filled.item(2, 'num1'), 3.0)
        self.assertAlmostEqual(df_filled.item(1, 'num2'), 100.0 / 3.0)
        self.assertAlmostEqual(df_filled.item(2, 'num2'), 100.0 / 3.0)
        self.assertTrue(df_filled['cat1'].is_null().any()) # Cat should be skipped
        self.assertTrue(df_filled['cat2'].is_null().any())
        self.assertTrue(df_filled['bool_col'].is_null().any()) # Bool should be skipped

    def test_median_imputation(self):
        """Test median imputation."""
        df_filled = handle_missing_values(self.df, strategy='median', features=['num1', 'num2'])
        self.assertFalse(df_filled['num1'].is_null().any())
        self.assertFalse(df_filled['num2'].is_null().any())
        self.assertAlmostEqual(df_filled.item(2, 'num1'), 3.0) # Median of [1, 2, 4, 5]
        self.assertAlmostEqual(df_filled.item(1, 'num2'), 40.0) # Median of [10, 40, 50]
        self.assertAlmostEqual(df_filled.item(2, 'num2'), 40.0)

    def test_mode_imputation_specific(self):
        """Test mode imputation on specific categorical feature."""
        df_filled = handle_missing_values(self.df, strategy='mode', features=['cat2'])
        self.assertFalse(df_filled['cat2'].is_null().any())
        self.assertEqual(df_filled.item(4, 'cat2'), 'X') # Mode of [X, Y, X, X]
        self.assertTrue(df_filled['cat1'].is_null().any()) # cat1 untouched

    def test_mode_imputation_all(self):
        """Test mode imputation on all applicable features."""
        df_filled = handle_missing_values(self.df, strategy='mode')
        # Floats are skipped for mode when features=None
        self.assertTrue(df_filled['num1'].is_null().any()) 
        self.assertTrue(df_filled['num2'].is_null().any())
        # Non-floats should be filled
        self.assertFalse(df_filled['cat1'].is_null().any())
        self.assertFalse(df_filled['cat2'].is_null().any())
        self.assertFalse(df_filled['bool_col'].is_null().any()) # Mode works for bool
        # Check specific filled values where mode is deterministic
        self.assertEqual(df_filled.item(4, 'cat2'), 'X')
        # Mode of bool depends on impl when tied, just check it's filled
        self.assertIsNotNone(df_filled.item(3, 'bool_col'))
        # Mode of cat1 depends on polars impl, just check it's filled
        self.assertIsNotNone(df_filled.item(3, 'cat1'))

    def test_zero_imputation(self):
        """Test zero imputation (only on numeric non-bool)."""
        df_filled = handle_missing_values(self.df, strategy='zero', features=['num1', 'num2'])
        self.assertAlmostEqual(df_filled.item(2, 'num1'), 0.0)
        self.assertAlmostEqual(df_filled.item(1, 'num2'), 0.0)
        self.assertAlmostEqual(df_filled.item(2, 'num2'), 0.0)

    def test_literal_imputation(self):
        """Test literal value imputation."""
        fill_c = 'Unknown'
        fill_n = -99.0
        df_filled = handle_missing_values(self.df, strategy='literal', features=['cat1'], fill_value=fill_c)
        df_filled = handle_missing_values(df_filled, strategy='literal', features=['num2'], fill_value=fill_n)
        self.assertEqual(df_filled.item(3, 'cat1'), fill_c)
        self.assertAlmostEqual(df_filled.item(1, 'num2'), fill_n)
        self.assertAlmostEqual(df_filled.item(2, 'num2'), fill_n)

    def test_literal_imputation_no_fill_value(self):
        """Test literal strategy without providing fill_value."""
        with self.assertRaisesRegex(ValueError, "literal.*requires.*fill_value"):
            handle_missing_values(self.df, strategy='literal', features=['num1'])

    def test_invalid_strategy(self):
        """Test providing an invalid strategy."""
        with self.assertRaisesRegex(ValueError, "Unsupported strategy"):
            handle_missing_values(self.df, strategy='invalid_strategy')

    def test_compute_error_mean_on_cat(self):
        """Test applying mean strategy explicitly to a categorical feature."""
        with self.assertRaises(pl.exceptions.ComputeError):
            handle_missing_values(self.df, strategy='mean', features=['cat1'])
        with self.assertRaises(pl.exceptions.ComputeError):
            handle_missing_values(self.df, strategy='mean', features=['bool_col']) 

    def test_key_error_missing_feature(self):
        """Test providing a feature not in the DataFrame."""
        with self.assertRaisesRegex(KeyError, "Features not found"):
            handle_missing_values(self.df, features=['num1', 'missing'])

    def test_no_missing_values(self):
        """Test running on a column with no missing values."""
        original_col = self.df['no_missing']
        df_filled = handle_missing_values(self.df, strategy='mean', features=['no_missing'])
        # Should return the original df if no changes applied to specified column
        assert_frame_equal(self.df, df_filled) 
        # Test with features=None - should not modify no_missing column
        df_filled_all = handle_missing_values(self.df, strategy='mean')
        self.assertTrue(df_filled_all['no_missing'].equals(original_col))


if __name__ == '__main__':
    unittest.main() 