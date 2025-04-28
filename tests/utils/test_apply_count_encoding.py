import unittest
import polars as pl
import numpy as np
from polars.testing import assert_frame_equal

# Adjust import path as necessary
try:
    from vibe_kaggler.utils.apply_count_encoding import apply_count_encoding
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from utils.apply_count_encoding import apply_count_encoding


class TestApplyCountEncoding(unittest.TestCase):

    def setUp(self):
        """Set up sample dataframes."""
        self.train_data = {
            'id': range(6),
            'cat1': ['A', 'B', 'A', 'C', 'B', None], 
            'cat2': ['X', 'X', 'Y', 'Y', 'Z', 'X']
        }
        self.train_df = pl.DataFrame(self.train_data)
        
        self.test_data = {
            'id': range(6, 10),
            'cat1': ['A', 'B', 'D', None], # D is unseen in train
            'cat2': ['X', 'Y', 'W', 'Z']  # W is unseen in train
        }
        self.test_df = pl.DataFrame(self.test_data)
        
        self.features = ['cat1', 'cat2']
        self.suffix = '_count'

    def test_count_on_train(self):
        """Test counting based on training data only."""
        train_out, test_out = apply_count_encoding(
            self.train_df, self.features, self.test_df, count_on='train'
        )
        
        # Expected counts based on train_df
        # cat1: A=2, B=2, C=1, None=1 -> Total non-null = 5
        # cat2: X=3, Y=2, Z=1 -> Total non-null = 6
        exp_train_cat1 = pl.Series('cat1_count', [2, 2, 2, 1, 2, 0], dtype=pl.UInt32) 
        exp_train_cat2 = pl.Series('cat2_count', [3, 3, 2, 2, 1, 3], dtype=pl.UInt32)
        exp_test_cat1 = pl.Series('cat1_count', [2, 2, 0, 0], dtype=pl.UInt32) # D is unknown, None is unknown
        exp_test_cat2 = pl.Series('cat2_count', [3, 2, 0, 1], dtype=pl.UInt32) # W is unknown
        
        assert_frame_equal(train_out.select(['cat1_count', 'cat2_count']),
                             pl.DataFrame([exp_train_cat1, exp_train_cat2]))
        assert_frame_equal(test_out.select(['cat1_count', 'cat2_count']),
                             pl.DataFrame([exp_test_cat1, exp_test_cat2]))

    def test_count_on_test(self):
        """Test counting based on test data only."""
        train_out, test_out = apply_count_encoding(
            self.train_df, self.features, self.test_df, count_on='test'
        )
        
        # Expected counts based on test_df
        # cat1: A=1, B=1, D=1, None=1 -> Total non-null = 3
        # cat2: X=1, Y=1, W=1, Z=1 -> Total non-null = 4
        exp_train_cat1 = pl.Series('cat1_count', [1, 1, 1, 0, 1, 0], dtype=pl.UInt32) # C unknown
        exp_train_cat2 = pl.Series('cat2_count', [1, 1, 1, 1, 1, 1], dtype=pl.UInt32)
        exp_test_cat1 = pl.Series('cat1_count', [1, 1, 1, 0], dtype=pl.UInt32)
        exp_test_cat2 = pl.Series('cat2_count', [1, 1, 1, 1], dtype=pl.UInt32)
        
        assert_frame_equal(train_out.select(['cat1_count', 'cat2_count']),
                             pl.DataFrame([exp_train_cat1, exp_train_cat2]))
        assert_frame_equal(test_out.select(['cat1_count', 'cat2_count']),
                             pl.DataFrame([exp_test_cat1, exp_test_cat2]))

    def test_count_on_combined(self):
        """Test counting based on combined train and test data."""
        train_out, test_out = apply_count_encoding(
            self.train_df, self.features, self.test_df, count_on='combined'
        )
        
        # Expected counts based on combined data
        # cat1: A=3, B=3, C=1, D=1, None=2 -> Total non-null = 8
        # cat2: X=4, Y=3, Z=2, W=1 -> Total non-null = 10
        exp_train_cat1 = pl.Series('cat1_count', [3, 3, 3, 1, 3, 0], dtype=pl.UInt32)
        exp_train_cat2 = pl.Series('cat2_count', [4, 4, 3, 3, 2, 4], dtype=pl.UInt32)
        exp_test_cat1 = pl.Series('cat1_count', [3, 3, 1, 0], dtype=pl.UInt32)
        exp_test_cat2 = pl.Series('cat2_count', [4, 3, 1, 2], dtype=pl.UInt32)
        
        assert_frame_equal(train_out.select(['cat1_count', 'cat2_count']),
                             pl.DataFrame([exp_train_cat1, exp_train_cat2]))
        assert_frame_equal(test_out.select(['cat1_count', 'cat2_count']),
                             pl.DataFrame([exp_test_cat1, exp_test_cat2]))

    def test_normalize(self):
        """Test normalization (frequency encoding)."""
        train_out, test_out = apply_count_encoding(
            self.train_df, self.features, self.test_df, count_on='train', normalize=True
        )
        
        # Expected frequencies based on train_df
        # cat1: A=2/5, B=2/5, C=1/5, None=0 -> Total non-null = 5
        # cat2: X=3/6, Y=2/6, Z=1/6 -> Total non-null = 6
        exp_train_cat1 = pl.Series('cat1_count', [0.4, 0.4, 0.4, 0.2, 0.4, 0.0])
        exp_train_cat2 = pl.Series('cat2_count', [0.5, 0.5, 2/6, 2/6, 1/6, 0.5])
        exp_test_cat1 = pl.Series('cat1_count', [0.4, 0.4, 0.0, 0.0]) # D unknown
        exp_test_cat2 = pl.Series('cat2_count', [0.5, 2/6, 0.0, 1/6]) # W unknown
        
        assert_frame_equal(train_out.select(['cat1_count', 'cat2_count']),
                             pl.DataFrame([exp_train_cat1, exp_train_cat2]), check_dtype=False, rtol=1e-6)
        assert_frame_equal(test_out.select(['cat1_count', 'cat2_count']),
                             pl.DataFrame([exp_test_cat1, exp_test_cat2]), check_dtype=False, rtol=1e-6)

    def test_handle_unknown_nan(self):
        """Test handling unknown categories with NaN."""
        train_out, test_out = apply_count_encoding(
            self.train_df, self.features, self.test_df, count_on='train', handle_unknown='nan'
        )
        exp_test_cat1 = pl.Series('cat1_count', [2.0, 2.0, np.nan, np.nan], dtype=pl.Float64) # D and None unknown
        exp_test_cat2 = pl.Series('cat2_count', [3.0, 2.0, np.nan, 1.0], dtype=pl.Float64) # W unknown
        
        self.assertTrue(test_out['cat1_count'].dtype == pl.Float64)
        self.assertTrue(test_out['cat2_count'].dtype == pl.Float64)
        assert_frame_equal(test_out.select(['cat1_count', 'cat2_count']),
                             pl.DataFrame([exp_test_cat1, exp_test_cat2]))

    def test_no_test_df(self):
        """Test behavior when test_df is None."""
        train_out, test_out = apply_count_encoding(
            self.train_df, self.features, test_df=None, count_on='train'
        )
        self.assertIsNone(test_out)
        self.assertIn('cat1_count', train_out.columns)
        self.assertIn('cat2_count', train_out.columns)

    def test_error_missing_test_df(self):
        """Test ValueError if test_df is needed but missing."""
        with self.assertRaisesRegex(ValueError, "test_df must be provided"): 
            apply_count_encoding(self.train_df, self.features, count_on='test')
        with self.assertRaisesRegex(ValueError, "test_df must be provided"): 
            apply_count_encoding(self.train_df, self.features, count_on='combined')

    def test_error_missing_columns(self):
        """Test ColumnNotFoundError if features are missing."""
        with self.assertRaises(pl.exceptions.ColumnNotFoundError):
            apply_count_encoding(self.train_df, ['cat1', 'missing'], self.test_df)
        with self.assertRaises(pl.exceptions.ColumnNotFoundError):
             bad_test = self.test_df.drop('cat2')
             apply_count_encoding(self.train_df, self.features, bad_test)

if __name__ == '__main__':
    unittest.main() 