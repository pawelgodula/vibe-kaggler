# Tests for eda_calculate_correlations function.

import unittest
import polars as pl
from polars.testing import assert_frame_equal
import numpy as np

# Adjust import path as necessary
try:
    from vibe_kaggler.utils.eda_calculate_correlations import calculate_correlations
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from utils.eda_calculate_correlations import calculate_correlations

class TestCalculateCorrelations(unittest.TestCase):

    def setUp(self):
        self.df_numeric = pl.DataFrame({
            'target': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feat1': [2.0, 4.0, 6.0, 8.0, 10.0], # Perfectly correlated with target
            'feat2': [5.0, 4.0, 3.0, 2.0, 1.0], # Perfectly negatively correlated
            'feat3': [1.0, 1.5, 1.2, 1.8, 1.1]  # Less correlated
        })
        self.df_mixed = pl.DataFrame({
            'target': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feat1': [2.0, 4.0, 6.0, 8.0, 10.0],
            'feat_cat': ['A', 'B', 'A', 'C', 'B'],
            'feat2': [5.0, 4.0, 3.0, 2.0, 1.0],
            'target_cat': ['X', 'Y', 'X', 'Y', 'X']
        })
        self.df_non_numeric_target = self.df_mixed.select([
            'feat1', 'feat_cat', 'feat2', 
            pl.col('target_cat').alias('target') 
        ])
        self.df_one_numeric_feat = pl.DataFrame({
             'target': [1.0, 2.0, 3.0],
             'feat1': [2.0, 4.0, 6.0],
             'cat': ['a','b','a']
        })
        self.df_no_numeric_feat = pl.DataFrame({
             'target': [1.0, 2.0, 3.0],
             'cat1': ['a','b','a'],
             'cat2': ['x','x','y']
        })

    def test_numeric_only(self):
        """Test with only numeric features and target."""
        feat_corr, target_corr = calculate_correlations(self.df_numeric, 'target')
        
        self.assertIsInstance(feat_corr, pl.DataFrame)
        self.assertIsInstance(target_corr, pl.DataFrame)
        
        # Check shapes (3 features -> 3x4 matrix including 'feature' col)
        self.assertEqual(feat_corr.shape, (3, 4))
        self.assertEqual(target_corr.shape, (3, 2))
        
        # Check specific perfect correlations
        self.assertAlmostEqual(feat_corr.filter(pl.col('feature') == 'feat1')['feat1'][0], 1.0)
        self.assertAlmostEqual(feat_corr.filter(pl.col('feature') == 'feat1')['feat2'][0], -1.0)
        self.assertAlmostEqual(target_corr.filter(pl.col('feature') == 'feat1')['target_correlation'][0], 1.0)
        self.assertAlmostEqual(target_corr.filter(pl.col('feature') == 'feat2')['target_correlation'][0], -1.0)
        
        # Check column names
        self.assertEqual(feat_corr.columns, ['feature', 'feat1', 'feat2', 'feat3'])
        self.assertEqual(target_corr.columns, ['feature', 'target_correlation'])
        
        # Check sorting of target_corr (descending by correlation value)
        # Corrs: feat1=1.0, feat2=-1.0, feat3=~0.05
        # Order: feat1, feat3, feat2
        self.assertEqual(target_corr['feature'].to_list(), ['feat1', 'feat3', 'feat2']) 

    def test_mixed_types(self):
        """Test with mixed feature types, only numerics should be used."""
        feat_corr, target_corr = calculate_correlations(self.df_mixed, 'target')
        
        self.assertIsInstance(feat_corr, pl.DataFrame)
        self.assertIsInstance(target_corr, pl.DataFrame)
        
        # Check shapes (2 numeric features -> 2x3 matrix)
        self.assertEqual(feat_corr.shape, (2, 3))
        self.assertEqual(target_corr.shape, (2, 2))
        
        # Check column names (only numeric features included)
        self.assertEqual(feat_corr.columns, ['feature', 'feat1', 'feat2'])
        self.assertEqual(target_corr.columns, ['feature', 'target_correlation'])
        self.assertEqual(feat_corr['feature'].to_list(), ['feat1', 'feat2'])
        self.assertEqual(target_corr['feature'].to_list(), ['feat1', 'feat2'])

    def test_non_numeric_target(self):
        """Test when the target column is not numeric."""
        feat_corr, target_corr = calculate_correlations(self.df_non_numeric_target, 'target')
        
        self.assertIsInstance(feat_corr, pl.DataFrame) # Feature corr should still work
        self.assertIsNone(target_corr) # Target corr should be None
        
        self.assertEqual(feat_corr.shape, (2, 3))
        self.assertEqual(feat_corr.columns, ['feature', 'feat1', 'feat2'])

    def test_one_numeric_feature(self):
        """Test with only one numeric feature."""
        feat_corr, target_corr = calculate_correlations(self.df_one_numeric_feat, 'target')
        
        self.assertIsNone(feat_corr) # Need >= 2 features for matrix
        self.assertIsInstance(target_corr, pl.DataFrame) # Target corr should work
        
        self.assertEqual(target_corr.shape, (1, 2))
        self.assertEqual(target_corr['feature'][0], 'feat1')
        self.assertAlmostEqual(target_corr['target_correlation'][0], 1.0)
        
    def test_no_numeric_features(self):
        """Test with no numeric features available."""
        feat_corr, target_corr = calculate_correlations(self.df_no_numeric_feat, 'target')
        self.assertIsNone(feat_corr)
        self.assertIsNone(target_corr) # Also none as no features to correlate with target

    def test_specific_feature_cols(self):
        """Test using the feature_cols argument."""
        feat_corr, target_corr = calculate_correlations(self.df_numeric, 'target', feature_cols=['feat1', 'feat3'])
        
        self.assertIsInstance(feat_corr, pl.DataFrame)
        self.assertIsInstance(target_corr, pl.DataFrame)
        
        # Check shapes (2 specified features -> 2x3 matrix)
        self.assertEqual(feat_corr.shape, (2, 3))
        self.assertEqual(target_corr.shape, (2, 2))
        
        self.assertEqual(feat_corr.columns, ['feature', 'feat1', 'feat3'])
        self.assertEqual(target_corr['feature'].to_list(), ['feat1', 'feat3']) 

    def test_missing_target_col(self):
        """Test error when target column is missing."""
        with self.assertRaisesRegex(pl.exceptions.ColumnNotFoundError, "Target column 'missing_target' not found."):
            calculate_correlations(self.df_numeric, 'missing_target')
            
    def test_missing_feature_col(self):
        """Test error when a specified feature column is missing."""
        with self.assertRaisesRegex(pl.exceptions.ColumnNotFoundError, "Feature column\(s\) not found: \['missing_feat'\]"):
            calculate_correlations(self.df_numeric, 'target', feature_cols=['feat1', 'missing_feat'])


if __name__ == '__main__':
    unittest.main() 