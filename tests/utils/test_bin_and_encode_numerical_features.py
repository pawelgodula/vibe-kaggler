# Tests for the bin_and_encode_numerical_features utility function.

import unittest
import polars as pl
import numpy as np
from polars.testing import assert_frame_equal
from sklearn.preprocessing import OneHotEncoder

# Adjust import path as necessary
try:
    from vibe_kaggler.utils.bin_and_encode_numerical_features import (
        bin_and_encode_numerical_features
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from utils.bin_and_encode_numerical_features import (
        bin_and_encode_numerical_features
    )

class TestBinAndEncodeFeatures(unittest.TestCase):

    def setUp(self):
        self.train_df = pl.DataFrame({
            'id': [1, 2, 3, 4, 5, 6],
            'num1': [10, 20, 30, 40, 50, 60],
            'num2': [1.1, 1.1, 2.2, 2.2, 3.3, 3.3],
            'cat1': ['A', 'B', 'A', 'B', 'A', 'B']
        })
        self.test_df = pl.DataFrame({
            'id': [7, 8, 9],
            'num1': [5, 35, 65], # Includes value below min, between, and above max
            'num2': [1.1, 2.5, 4.0],
            'cat1': ['A', 'B', 'A']
        })

    def test_quantile_binning_onehot(self):
        """Test quantile binning followed by one-hot encoding."""
        features = ['num1']
        n_bins = 3
        train_out, test_out, params = bin_and_encode_numerical_features(
            self.train_df, features, n_bins, test_df=self.test_df, strategy='quantile'
        )
        
        # Expected edges for num1 (quantiles 0, 0.33, 0.66, 1.0 of [10..60]): ~[10, 30, 50, 60]
        # Bins based on [-inf, 10, 30, 50, 60, inf]
        # Train num1 values: 10, 20, 30, 40, 50, 60
        # Train bins:      ~bin1, bin2, bin2, bin3, bin3, bin4 
        # Test num1 values: 5, 35, 65
        # Test bins:      ~bin0, bin3, bin5
        # Encoder learns 4 bins from train data. OHE should create 4 columns.
        
        self.assertIsInstance(params, dict)
        self.assertIn('encoder', params)
        self.assertIsInstance(params['encoder'], OneHotEncoder)
        self.assertIn('bin_edges', params)
        self.assertIn('num1', params['bin_edges'])
        self.assertGreaterEqual(len(params['bin_edges']['num1']), n_bins + 1) # Edges including +/- inf

        # Check output shapes and columns
        # Original 'num1' should be dropped, 'num1_bin_... OHE columns added.
        self.assertNotIn('num1', train_out.columns)
        self.assertNotIn('num1_bin', train_out.columns)
        ohe_cols = [c for c in train_out.columns if c.startswith('num1_bin_')]
        self.assertGreaterEqual(len(ohe_cols), n_bins) # Might be more if edges are exact
        self.assertEqual(train_out.height, self.train_df.height)
        self.assertEqual(test_out.height, self.test_df.height)
        self.assertEqual(set(train_out.columns) - set(self.train_df.columns) - set(ohe_cols), set())
        
        # Check a sample value transformation (qualitative check)
        # e.g., first row num1=10 should fall in the lowest bin -> first OHE col = 1?
        # This depends heavily on exact edges and OHE category ordering, difficult to assert precisely
        # Check that the output is numeric (float or int from OHE)
        self.assertTrue(all(train_out[c].dtype.is_numeric() for c in ohe_cols))
        if test_out is not None:
             self.assertTrue(all(test_out[c].dtype.is_numeric() for c in ohe_cols))


    def test_uniform_binning_onehot_multiple_features(self):
        """Test uniform binning on multiple features."""
        features = ['num1', 'num2']
        n_bins = 2
        train_out, test_out, params = bin_and_encode_numerical_features(
            self.train_df, features, n_bins, test_df=self.test_df, strategy='uniform'
        )
        
        # Num1 range [10, 60] -> edges ~[10, 35, 60] -> final [-inf, 10, 35, 60, inf]
        # Num2 range [1.1, 3.3] -> edges ~[1.1, 2.2, 3.3] -> final [-inf, 1.1, 2.2, 3.3, inf]
        # Expect ~2 OHE cols for num1_bin, ~2 OHE cols for num2_bin
        
        self.assertIsInstance(params['encoder'], OneHotEncoder)
        self.assertIn('num1', params['bin_edges'])
        self.assertIn('num2', params['bin_edges'])

        self.assertNotIn('num1', train_out.columns)
        self.assertNotIn('num2', train_out.columns)
        ohe_cols_num1 = [c for c in train_out.columns if c.startswith('num1_bin_')]
        ohe_cols_num2 = [c for c in train_out.columns if c.startswith('num2_bin_')]
        self.assertGreaterEqual(len(ohe_cols_num1), n_bins)
        self.assertGreaterEqual(len(ohe_cols_num2), n_bins)
        self.assertEqual(train_out.height, self.train_df.height)
        if test_out is not None:
             self.assertEqual(test_out.height, self.test_df.height)

    def test_no_test_df(self):
        """Test functionality when test_df is None."""
        features = ['num2']
        n_bins = 4
        train_out, test_out, params = bin_and_encode_numerical_features(
            self.train_df, features, n_bins, test_df=None
        )
        self.assertIsNone(test_out)
        self.assertIsInstance(train_out, pl.DataFrame)
        self.assertNotIn('num2', train_out.columns)
        ohe_cols = [c for c in train_out.columns if c.startswith('num2_bin_')]
        # Check that *some* OHE columns were created, not necessarily >= n_bins
        self.assertGreater(len(ohe_cols), 0) 
        self.assertIsInstance(params['encoder'], OneHotEncoder)
        
    def test_error_non_numeric_feature(self):
        """Test error if a non-numeric feature is provided."""
        features = ['num1', 'cat1']
        with self.assertRaisesRegex(ValueError, "Features must be numeric for binning"):
            bin_and_encode_numerical_features(self.train_df, features, n_bins=3, test_df=self.test_df)
            
    def test_error_missing_feature(self):
        """Test error if a feature is not in the DataFrame."""
        features = ['num1', 'non_existent']
        with self.assertRaisesRegex(ValueError, "Features not found in train_df"):
            bin_and_encode_numerical_features(self.train_df, features, n_bins=3, test_df=self.test_df)

    def test_constant_column(self):
        """Test handling of a constant numerical column."""
        df_const = self.train_df.with_columns(pl.lit(5.0).alias('num_const'))
        features = ['num_const']
        n_bins = 3
        # Expect a warning, but should produce 1 bin and 1 OHE column
        train_out, _, params = bin_and_encode_numerical_features(
            df_const, features, n_bins, test_df=None
        )
        ohe_cols = [c for c in train_out.columns if c.startswith('num_const_bin_')]
        # Depending on edge case handling, might get 1 or 2 OHE columns
        self.assertGreaterEqual(len(ohe_cols), 1) 
        self.assertEqual(train_out.height, df_const.height)
        self.assertIn('num_const', params['bin_edges'])
        
if __name__ == '__main__':
    unittest.main() 