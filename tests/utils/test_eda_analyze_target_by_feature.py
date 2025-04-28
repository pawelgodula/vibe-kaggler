# Tests for eda_analyze_target_by_feature function.

import unittest
import polars as pl
from polars.testing import assert_frame_equal
import numpy as np

# Adjust import path as necessary
try:
    from vibe_kaggler.utils.eda_analyze_target_by_feature import analyze_target_by_feature
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from utils.eda_analyze_target_by_feature import analyze_target_by_feature

class TestAnalyzeTargetByFeature(unittest.TestCase):

    def setUp(self):
        self.df = pl.DataFrame({
            'target': [10.0, 20.0, 15.0, 5.0, 30.0, 25.0, 10.0, 12.0, 18.0, 14.0],
            'cat_feat': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
            'num_feat': [1.1, 2.2, 1.5, 3.3, 2.5, 1.8, 3.9, 2.9, 1.2, 3.1],
            'cat_high_card': [f'cat_{i % 50}' for i in range(10)],
            'num_low_unique': [1, 2, 1, 3, 2, 1, 3, 2, 1, 3]
        })
        self.df_with_nulls = pl.DataFrame({
            'target': [10.0, 20.0, None, 5.0, 30.0, 25.0, 10.0],
            'cat_feat': ['A', 'B', 'A', None, 'B', None, 'C'],
            'num_feat': [1.1, 2.2, 1.5, 3.3, None, 1.8, None],
        })

    def test_categorical_feature(self):
        """Test analysis with a standard categorical feature."""
        summary = analyze_target_by_feature(self.df, 'target', 'cat_feat')
        expected = pl.DataFrame({
            'feature_value': ['A', 'B', 'C'],
            'count': [4, 3, 3],
            'target_mean': [(10.0 + 15.0 + 25.0 + 18.0) / 4, (20.0 + 30.0 + 12.0) / 3, (5.0 + 10.0 + 14.0) / 3],
            'target_median': [16.5, 20.0, 10.0] # Medians: A=[10,15,18,25]->16.5; B=[12,20,30]->20; C=[5,10,14]->10
        }).sort('count', descending=True)
        self.assertIsInstance(summary, pl.DataFrame)
        # Sort both by feature_value before comparing to handle unstable sort order for ties in count
        assert_frame_equal(summary.sort('feature_value'), expected.sort('feature_value'), check_dtype=False, check_column_order=False)

    def test_categorical_max_categories(self):
        """Test limiting categories for high cardinality categorical feature."""
        df_high = self.df.with_columns(
            pl.col('cat_feat').map_elements(lambda x: f"cat_{x}", return_dtype=pl.Utf8).alias('cat_feat')
        )
        summary = analyze_target_by_feature(df_high, 'target', 'cat_feat', max_categories=2)
        self.assertIsInstance(summary, pl.DataFrame)
        self.assertEqual(len(summary), 2) # Should only have top 2 categories
        # Top categories are A (4) and B/C (3 each) - specific one depends on tie-breaking
        self.assertTrue(all(val in ['cat_A', 'cat_B', 'cat_C'] for val in summary['feature_value'])) 

    def test_numerical_feature_binning(self):
        """Test analysis with a numerical feature using quantile binning."""
        # Expecting 3 bins (qcut might create fewer if values are concentrated)
        summary = analyze_target_by_feature(self.df, 'target', 'num_feat', num_bins=3)
        self.assertIsInstance(summary, pl.DataFrame)
        # Check output columns
        self.assertEqual(summary.columns, ['feature_value', 'count', 'target_mean', 'target_median'])
        # Check number of bins (might be less than requested if duplicates)
        self.assertTrue(1 <= len(summary) <= 3)
        # Check bin names
        self.assertTrue(all(val.startswith('q') for val in summary['feature_value']))

    def test_numerical_feature_low_cardinality_auto(self):
        """Test numerical feature with few unique values treated as categorical in auto mode."""
        summary = analyze_target_by_feature(self.df, 'target', 'num_low_unique', feature_type='auto')
        # Expected values are calculated based on grouping by 1, 2, 3
        # Target for 1: [10.0, 15.0, 25.0, 18.0] -> mean=17.0, median=16.5
        # Target for 2: [20.0, 30.0, 12.0] -> mean=20.667, median=20.0
        # Target for 3: [5.0, 10.0, 14.0] -> mean=9.667, median=10.0
        expected = pl.DataFrame({
            'feature_value': [1, 2, 3], # Expect original Int type when treated as categorical
            'count': [4, 3, 3],
            'target_mean': [17.0, 20.666667, 9.666667],
            'target_median': [16.5, 20.0, 10.0]
        }).sort('count', descending=True).with_columns(pl.col('count').cast(pl.UInt32))
        # Sort result by count for comparison
        summary_sorted = summary.sort('count', descending=True)
        self.assertIsInstance(summary, pl.DataFrame)
        # Auto mode should detect low cardinality and treat as categorical, keeping original dtype
        self.assertEqual(summary['feature_value'].dtype, pl.Int64)
        # Sort by feature_value as well for stable comparison
        assert_frame_equal(summary_sorted.sort('feature_value'), expected.sort('feature_value'), check_dtype=True, check_column_order=False, atol=1e-5)

    def test_numerical_feature_low_cardinality_force_numerical(self):
        """Test numerical feature with few unique values forced as numerical (binning may fail/warn)."""
        # qcut might fail here or create strange bins
        summary = analyze_target_by_feature(self.df, 'target', 'num_low_unique', feature_type='numerical', num_bins=2)
        self.assertIsInstance(summary, pl.DataFrame)
        # Check if it fell back to categorical analysis due to qcut error or produced bin labels
        self.assertTrue(summary['feature_value'].dtype == pl.Categorical or summary['feature_value'].dtype == pl.Utf8) 
        self.assertTrue(all(val.startswith('q') or val in ['1','2','3'] for val in summary['feature_value']))

    def test_handle_nulls_drop(self):
        """Test dropping rows with nulls in the feature column."""
        summary_cat = analyze_target_by_feature(self.df_with_nulls, 'target', 'cat_feat', handle_nulls='drop')
        summary_num = analyze_target_by_feature(self.df_with_nulls, 'target', 'num_feat', handle_nulls='drop')
        # Expected counts after dropping nulls in feature
        # Nulls in cat_feat at index 3, 5 are dropped.
        # Nulls in num_feat at index 4, 6 are dropped.
        self.assertEqual(summary_cat.filter(pl.col('feature_value') == 'A')['count'][0], 2) # Index 0, 2 remain
        self.assertEqual(summary_cat.filter(pl.col('feature_value') == 'B')['count'][0], 2) # Index 1, 4 remain
        self.assertEqual(summary_cat.filter(pl.col('feature_value') == 'C')['count'][0], 1) # Index 6 remains
        self.assertEqual(summary_num['count'].sum(), 5) # Index 0, 1, 2, 3, 5 remain

    def test_handle_nulls_fill(self):
        """Test filling nulls in the feature column with a placeholder."""
        summary_cat = analyze_target_by_feature(self.df_with_nulls, 'target', 'cat_feat', handle_nulls='fill')
        summary_num = analyze_target_by_feature(self.df_with_nulls, 'target', 'num_feat', handle_nulls='fill')
        
        self.assertIn('_NULL_', summary_cat['feature_value'].to_list())
        self.assertIn('_NULL_', summary_num['feature_value'].to_list())
        
        # Check counts for null category (null target is ignored in agg)
        self.assertEqual(summary_cat.filter(pl.col('feature_value') == '_NULL_')['count'][0], 2) # feat=None at index 3, 5
        self.assertEqual(summary_num.filter(pl.col('feature_value') == '_NULL_')['count'][0], 2) # feat=None at index 4, 6
        # Check mean for null category (target=25 at index 5; target=30 at index 4)
        self.assertAlmostEqual(summary_cat.filter(pl.col('feature_value') == '_NULL_')['target_mean'][0], (5.0+25.0)/2) 
        self.assertAlmostEqual(summary_num.filter(pl.col('feature_value') == '_NULL_')['target_mean'][0], (30.0+10.0)/2)

    def test_non_numeric_target(self):
        """Test error when target is not numeric."""
        df_bad_target = self.df.with_columns(pl.col('target').cast(pl.Utf8))
        with self.assertRaisesRegex(TypeError, "Target column 'target' must be numeric"):
            analyze_target_by_feature(df_bad_target, 'target', 'cat_feat')

    def test_missing_columns(self):
        """Test error when columns are missing."""
        with self.assertRaisesRegex(pl.exceptions.ColumnNotFoundError, "Target column 'missing_target' not found"):
             analyze_target_by_feature(self.df, 'missing_target', 'cat_feat')
        with self.assertRaisesRegex(pl.exceptions.ColumnNotFoundError, "Feature column 'missing_feature' not found"):
             analyze_target_by_feature(self.df, 'target', 'missing_feature')

if __name__ == '__main__':
    unittest.main() 