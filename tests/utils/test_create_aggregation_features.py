import unittest
import polars as pl
import numpy as np
from polars.testing import assert_frame_equal
from utils.create_aggregation_features import create_aggregation_features

class TestCreateAggregationFeatures(unittest.TestCase):

    def setUp(self):
        """Set up a sample DataFrame."""
        self.df = pl.DataFrame({
            'group1': ['A', 'A', 'B', 'B', 'A', 'C', None, 'C'],
            'group2': [1, 1, 2, 2, 1, 1, 1, 2],
            'value1': [10, 12, 20, 22, 11, 30, 5, 33],
            'value2': [100, 100, 200, 200, 100, 50, 50, 50],
            'value3_cat': ['x', 'y', 'x', 'y', 'x', 'z', 'y', 'z']
        })

    def test_single_group_single_agg(self):
        """Test aggregation with one group column and one aggregation."""
        group_cols = ['group1']
        agg_dict = {'value1': ['mean']}
        df_out = create_aggregation_features(self.df, group_cols, agg_dict)
        
        self.assertIn('agg_value1_by_group1_mean', df_out.columns)
        self.assertEqual(df_out.shape[0], self.df.shape[0])
        self.assertEqual(df_out.shape[1], self.df.shape[1] + 1)
        
        # Check values for group A: mean(10, 12, 11) = 11
        mean_a = df_out.filter(pl.col('group1') == 'A')['agg_value1_by_group1_mean'].mean()
        self.assertAlmostEqual(mean_a, 11.0)
        # Check values for group B: mean(20, 22) = 21
        mean_b = df_out.filter(pl.col('group1') == 'B')['agg_value1_by_group1_mean'].mean()
        self.assertAlmostEqual(mean_b, 21.0)
         # Check values for group C: mean(30, 33) = 31.5
        mean_c = df_out.filter(pl.col('group1') == 'C')['agg_value1_by_group1_mean'].mean()
        self.assertAlmostEqual(mean_c, 31.5)
        # Check null group - result should be null
        self.assertTrue(df_out.filter(pl.col('group1').is_null())['agg_value1_by_group1_mean'].is_null().all())

    def test_single_group_multiple_aggs(self):
        """Test aggregation with one group column and multiple aggregations."""
        group_cols = ['group1']
        agg_dict = {
            'value1': ['mean', 'std', 'max'],
            'value2': ['count']
        }
        df_out = create_aggregation_features(self.df, group_cols, agg_dict)
        
        self.assertIn('agg_value1_by_group1_mean', df_out.columns)
        self.assertIn('agg_value1_by_group1_std', df_out.columns)
        self.assertIn('agg_value1_by_group1_max', df_out.columns)
        self.assertIn('agg_value2_by_group1_count', df_out.columns)
        self.assertEqual(df_out.shape[1], self.df.shape[1] + 4)
        
        # Check max for group A: max(10, 12, 11) = 12
        max_a = df_out.filter(pl.col('group1') == 'A')['agg_value1_by_group1_max'].first()
        self.assertEqual(max_a, 12)
        # Check count for group B: count(200, 200) = 2
        count_b = df_out.filter(pl.col('group1') == 'B')['agg_value2_by_group1_count'].first()
        self.assertEqual(count_b, 2)

    def test_multiple_group_columns(self):
        """Test aggregation with multiple grouping columns."""
        group_cols = ['group1', 'group2']
        agg_dict = {'value1': ['sum']}
        df_out = create_aggregation_features(self.df, group_cols, agg_dict)
        
        self.assertIn('agg_value1_by_group1_group2_sum', df_out.columns)
        self.assertEqual(df_out.shape[1], self.df.shape[1] + 1)

        # Check group A, 1: sum(10, 12, 11) = 33
        sum_a1 = df_out.filter((pl.col('group1') == 'A') & (pl.col('group2') == 1))['agg_value1_by_group1_group2_sum'].first()
        self.assertEqual(sum_a1, 33)
        # Check group B, 2: sum(20, 22) = 42
        sum_b2 = df_out.filter((pl.col('group1') == 'B') & (pl.col('group2') == 2))['agg_value1_by_group1_group2_sum'].first()
        self.assertEqual(sum_b2, 42)
        # Check group C, 2: sum(33) = 33
        sum_c2 = df_out.filter((pl.col('group1') == 'C') & (pl.col('group2') == 2))['agg_value1_by_group1_group2_sum'].first()
        self.assertEqual(sum_c2, 33)

    def test_size_and_nunique_aggs(self):
        """Test 'size' and 'nunique' aggregations."""
        group_cols = ['group1']
        agg_dict = {
            'value1': ['size'], # Special handling in function for name
            'value3_cat': ['nunique']
        }
        df_out = create_aggregation_features(self.df, group_cols, agg_dict)
        
        self.assertIn('agg_group1_size', df_out.columns)
        self.assertIn('agg_value3_cat_by_group1_nunique', df_out.columns)
        self.assertEqual(df_out.shape[1], self.df.shape[1] + 2)

        # Check size for group A: 3 rows
        size_a = df_out.filter(pl.col('group1') == 'A')['agg_group1_size'].first()
        self.assertEqual(size_a, 3)
        # Check nunique for group A: nunique(x, y, x) = 2
        nunique_a = df_out.filter(pl.col('group1') == 'A')['agg_value3_cat_by_group1_nunique'].first()
        self.assertEqual(nunique_a, 2)
        # Check size for group C: 2 rows
        size_c = df_out.filter(pl.col('group1') == 'C')['agg_group1_size'].first()
        self.assertEqual(size_c, 2)
        # Check nunique for group C: nunique(z, z) = 1
        nunique_c = df_out.filter(pl.col('group1') == 'C')['agg_value3_cat_by_group1_nunique'].first()
        self.assertEqual(nunique_c, 1)
        # Check size for null group: 1 row
        # The aggregated value will be null because the key was null during the left join
        self.assertTrue(df_out.filter(pl.col('group1').is_null())['agg_group1_size'].is_null().all())
        # Check nunique for null group: nunique(y) = 1, but result is null due to join key being null
        self.assertTrue(df_out.filter(pl.col('group1').is_null())['agg_value3_cat_by_group1_nunique'].is_null().all())

    def test_custom_prefix(self):
        """Test using a custom column prefix."""
        group_cols = ['group1']
        agg_dict = {'value1': ['mean']}
        prefix = "feat_"
        df_out = create_aggregation_features(self.df, group_cols, agg_dict, new_col_prefix=prefix)
        self.assertIn('feat_value1_by_group1_mean', df_out.columns)
        self.assertNotIn('agg_value1_by_group1_mean', df_out.columns)

    # --- Error Handling --- 
    def test_error_empty_group_cols(self):
        """Test error on empty group_by_cols list."""
        with self.assertRaisesRegex(ValueError, "group_by_cols cannot be empty"):
            create_aggregation_features(self.df, [], {'value1': ['mean']})

    def test_error_empty_agg_dict(self):
        """Test error on empty agg_dict."""
        with self.assertRaisesRegex(ValueError, "agg_dict cannot be empty"):
            create_aggregation_features(self.df, ['group1'], {})

    def test_error_missing_group_col(self):
        """Test error if group column is missing."""
        with self.assertRaises(pl.exceptions.ColumnNotFoundError):
            create_aggregation_features(self.df, ['group1', 'missing_group'], {'value1': ['mean']})

    def test_error_missing_agg_col(self):
        """Test error if aggregation target column is missing."""
        with self.assertRaises(pl.exceptions.ColumnNotFoundError):
            create_aggregation_features(self.df, ['group1'], {'missing_value': ['mean']})

    def test_error_invalid_agg_func(self):
        """Test error if an invalid aggregation function name is used."""
        with self.assertRaisesRegex(ValueError, "Unsupported aggregation function: 'invalid_func'"):
            create_aggregation_features(self.df, ['group1'], {'value1': ['mean', 'invalid_func']})

if __name__ == '__main__':
    unittest.main() 