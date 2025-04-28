# Tests for the create_nway_interaction_features utility function.

import unittest
import polars as pl
from polars.testing import assert_frame_equal

# Adjust import path as necessary
try:
    from vibe_kaggler.utils.create_nway_interaction_features import (
        create_nway_interaction_features
    )
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from utils.create_nway_interaction_features import (
        create_nway_interaction_features
    )

class TestNwayInteractionFeatures(unittest.TestCase):

    def setUp(self):
        self.df = pl.DataFrame({
            'cat1': ['A', 'B', 'A', 'B', None, 'A'],
            'cat2': ['X', 'X', 'Y', 'Y', 'X', 'Y'],
            'num1': [1, 2, 1, 2, 1, 2] # Treat as categorical for interaction
        })

    def test_2way_interaction_strings(self):
        """Test basic 2-way string concatenation interaction."""
        features = ['cat1', 'cat2']
        df_out, params = create_nway_interaction_features(self.df, features, n=2, count_encode=False)
        
        expected = pl.DataFrame({
            'cat1_cat2_inter2': ['A_X', 'B_X', 'A_Y', 'B_Y', 'null_str_X', 'A_Y']
        })
        
        self.assertIsNone(params)
        assert_frame_equal(df_out, expected)

    def test_3way_interaction_strings_propagate_nulls(self):
        """Test 3-way interaction with null propagation."""
        features = ['cat1', 'cat2', 'num1']
        # Cast num1 to string for sensible concatenation
        df_in = self.df.with_columns(pl.col('num1').cast(pl.Utf8))
        df_out, params = create_nway_interaction_features(
            df_in, features, n=3, count_encode=False, handle_nulls='propagate'
        )
        
        expected = pl.DataFrame({
            'cat1_cat2_num1_inter3': ['A_X_1', 'B_X_2', 'A_Y_1', 'B_Y_2', None, 'A_Y_2']
        })
        
        self.assertIsNone(params)
        assert_frame_equal(df_out, expected)
        
    def test_2way_interaction_count_encode(self):
        """Test 2-way interaction with count encoding."""
        features = ['cat1', 'cat2']
        df_out, params = create_nway_interaction_features(self.df, features, n=2, count_encode=True)
        
        # Expected counts for cat1_cat2 combinations:
        # A_X: 1
        # B_X: 1
        # A_Y: 2
        # B_Y: 1
        # null_str_X: 1
        expected = pl.DataFrame({
            'cat1_cat2_inter2_count': pl.Series([1, 1, 2, 1, 1, 2], dtype=pl.UInt32)
        })
        
        self.assertIsNone(params, "fitted_params should be None for window function count encoding")
        
        assert_frame_equal(df_out, expected)
        
    def test_multiple_2way_interactions(self):
        """Test generating multiple 2-way interactions at once."""
        features = ['cat1', 'cat2', 'num1']
        df_in = self.df.with_columns(pl.col('num1').cast(pl.Utf8))
        df_out, params = create_nway_interaction_features(df_in, features, n=2, count_encode=False)
        
        expected_cols = ['cat1_cat2_inter2', 'cat1_num1_inter2', 'cat2_num1_inter2']
        self.assertEqual(df_out.columns, expected_cols)
        self.assertEqual(df_out.height, self.df.height)
        self.assertIsNone(params)
        # Check one column example
        self.assertEqual(df_out['cat1_num1_inter2'][4], 'null_str_1') # 5th row, cat1=None, num1=1
        self.assertEqual(df_out['cat2_num1_inter2'][1], 'X_2') # 2nd row, cat2=X, num1=2
        
    def test_error_invalid_n(self):
        """Test error for unsupported n."""
        with self.assertRaisesRegex(ValueError, "Interaction order n must be 2 or 3"):
            create_nway_interaction_features(self.df, ['cat1', 'cat2'], n=4)
            
    def test_error_missing_feature(self):
        """Test error if a feature is missing."""
        with self.assertRaisesRegex(ValueError, "Features not found in df"):
            create_nway_interaction_features(self.df, ['cat1', 'non_existent'], n=2)


if __name__ == '__main__':
    unittest.main() 