# Tests for eda_calculate_feature_summary function.

import unittest
import polars as pl
from polars.testing import assert_frame_equal

# Adjust import path as necessary
try:
    from vibe_kaggler.utils.eda_calculate_feature_summary import calculate_feature_summary
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from utils.eda_calculate_feature_summary import calculate_feature_summary

class TestCalculateFeatureSummary(unittest.TestCase):

    def setUp(self):
        self.df_mixed = pl.DataFrame({
            'col_int': [1, 2, 3, 1, None],
            'col_float': [1.1, 2.2, 3.3, 1.1, 4.4],
            'col_str': ['A', 'B', 'A', 'C', 'B'],
            'col_bool': [True, False, True, False, True],
            'col_all_null': [None, None, None, None, None]
        }, schema={
            'col_int': pl.Int32,
            'col_float': pl.Float64,
            'col_str': pl.Utf8,
            'col_bool': pl.Boolean,
            'col_all_null': pl.Float32 # Give it a type
        })
        self.total_rows = 5

    def test_summary_calculation(self):
        """Test the calculation of summary statistics for various types."""
        summary = calculate_feature_summary(self.df_mixed)

        # Expected shape
        self.assertEqual(summary.shape, (5, 10)) # 5 columns, 10 stats
        
        # Check stats for integer column
        int_summary = summary.filter(pl.col('Column') == 'col_int')
        self.assertEqual(int_summary['DataType'][0], 'Int32')
        self.assertEqual(int_summary['NumUnique'][0], 3) # 1, 2, 3
        self.assertEqual(int_summary['NumNull'][0], 1)
        self.assertEqual(int_summary['PctNull'][0], 20.0)
        self.assertAlmostEqual(int_summary['Mean'][0], (1+2+3+1)/4)
        self.assertAlmostEqual(int_summary['Median'][0], 1.5)
        self.assertEqual(int_summary['Min'][0], 1)
        self.assertEqual(int_summary['Max'][0], 3)
        self.assertIsNotNone(int_summary['StdDev'][0])
        
        # Check stats for float column
        float_summary = summary.filter(pl.col('Column') == 'col_float')
        self.assertEqual(float_summary['DataType'][0], 'Float64')
        self.assertEqual(float_summary['NumUnique'][0], 4) 
        self.assertEqual(float_summary['NumNull'][0], 0)
        self.assertEqual(float_summary['PctNull'][0], 0.0)
        self.assertAlmostEqual(float_summary['Mean'][0], (1.1+2.2+3.3+1.1+4.4)/5)
        self.assertAlmostEqual(float_summary['Median'][0], 2.2)
        self.assertEqual(float_summary['Min'][0], 1.1)
        self.assertEqual(float_summary['Max'][0], 4.4)
        self.assertIsNotNone(float_summary['StdDev'][0])
        
        # Check stats for string column
        str_summary = summary.filter(pl.col('Column') == 'col_str')
        self.assertEqual(str_summary['DataType'][0], 'String') # Polars uses 'String' now
        self.assertEqual(str_summary['NumUnique'][0], 3) # A, B, C
        self.assertEqual(str_summary['NumNull'][0], 0)
        self.assertEqual(str_summary['PctNull'][0], 0.0)
        self.assertIsNone(str_summary['Mean'][0]) # Non-numeric stats should be None
        self.assertIsNone(str_summary['StdDev'][0])
        self.assertIsNone(str_summary['Median'][0])
        self.assertIsNone(str_summary['Min'][0])
        self.assertIsNone(str_summary['Max'][0])
        
        # Check stats for boolean column
        bool_summary = summary.filter(pl.col('Column') == 'col_bool')
        self.assertEqual(bool_summary['DataType'][0], 'Boolean')
        self.assertEqual(bool_summary['NumUnique'][0], 2) # True, False
        self.assertEqual(bool_summary['NumNull'][0], 0)
        self.assertIsNone(bool_summary['Mean'][0])
        
        # Check stats for all null column
        null_summary = summary.filter(pl.col('Column') == 'col_all_null')
        self.assertEqual(null_summary['DataType'][0], 'Float32')
        self.assertEqual(null_summary['NumUnique'][0], 0) # Only nulls
        self.assertEqual(null_summary['NumNull'][0], 5)
        self.assertEqual(null_summary['PctNull'][0], 100.0)
        self.assertIsNone(null_summary['Mean'][0]) 
        self.assertIsNone(null_summary['Median'][0]) 
        self.assertIsNone(null_summary['Min'][0]) 
        self.assertIsNone(null_summary['Max'][0]) 
        self.assertIsNone(null_summary['StdDev'][0]) 

    def test_empty_dataframe(self):
        """Test with an empty DataFrame input."""
        empty_df = pl.DataFrame()
        summary = calculate_feature_summary(empty_df)
        self.assertTrue(summary.is_empty())

if __name__ == '__main__':
    unittest.main() 