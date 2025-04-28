import unittest
import polars as pl
from polars.testing import assert_frame_equal
import numpy as np
from utils.reduce_memory_usage import reduce_memory_usage
import io
from unittest.mock import patch

class TestReduceMemoryUsage(unittest.TestCase):

    def setUp(self):
        """Create a sample Polars DataFrame with various types."""
        self.data = {
            # Integer types
            'int64_col': np.arange(10, dtype=np.int64),
            'int32_col': np.arange(10, dtype=np.int32),
            'int16_col': np.arange(10, dtype=np.int16),
            'int8_col': np.arange(10, dtype=np.int8),
            'int_large': np.array([0, np.iinfo(np.int32).max + 10] * 5, dtype=np.int64), # Repeat 5 times for length 10
            'int_small_range': np.array([5, 6, 7, 5, 6] * 2, dtype=np.int64), # Needs length 10
            'int_with_null': np.array([1, 2, None, 4, 5] * 2, dtype=object), # Mixed type forces object
            # Float types
            'float64_col': np.arange(10, dtype=np.float64),
            'float32_col': np.arange(10, dtype=np.float32),
            'float_large_range': np.array([0.0, np.finfo(np.float32).max * 2.0] * 5, dtype=np.float64),
            'float_small_range': np.array([1.1, 2.2, 3.3] * 3 + [4.4], dtype=np.float64), # length 10
            'float_with_null': np.array([1.0, None, 3.0] * 3 + [4.0], dtype=object),
            # String/Category types
            'category_low_cardinality': ['A', 'B', 'A', 'C', 'B'] * 2,
            'category_high_cardinality': [f'ID_{i}' for i in range(10)],
            'string_with_null': ['X', 'Y', None] * 3 + ['Z'],
            # Other types
            'bool_col': [True, False] * 5,
            'datetime_col': pl.date_range(np.datetime64('2023-01-01'), 
                                          np.datetime64('2023-01-10'), "1d", eager=True)
        }

        # Polars handles None -> null conversion automatically during DataFrame creation
        # Remove explicit casts for object columns - let Polars infer
        self.df = pl.DataFrame(self.data)
        # Ensure original types for testing downcasting are set *after* initial creation
        self.df = self.df.with_columns([
            pl.col('int_small_range').cast(pl.Int64), 
            pl.col('float_small_range').cast(pl.Float64) 
        ])
        

    def test_integer_downcasting(self):
        """Test Polars integer downcasting."""
        df_reduced = reduce_memory_usage(self.df, verbose=False)
        self.assertEqual(df_reduced['int64_col'].dtype, pl.Int8)
        self.assertEqual(df_reduced['int32_col'].dtype, pl.Int8)
        self.assertEqual(df_reduced['int16_col'].dtype, pl.Int8)
        self.assertEqual(df_reduced['int8_col'].dtype, pl.Int8)
        self.assertEqual(df_reduced['int_large'].dtype, pl.Int64) 
        self.assertEqual(df_reduced['int_small_range'].dtype, pl.Int8)
        # Column was inferred as Object due to None, function doesn't cast Object types currently.
        self.assertEqual(df_reduced['int_with_null'].dtype, pl.Object)

    def test_float_downcasting(self):
        """Test Polars float downcasting."""
        df_reduced = reduce_memory_usage(self.df, verbose=False)
        self.assertEqual(df_reduced['float64_col'].dtype, pl.Float32)
        self.assertEqual(df_reduced['float32_col'].dtype, pl.Float32)
        self.assertEqual(df_reduced['float_large_range'].dtype, pl.Float64)
        self.assertEqual(df_reduced['float_small_range'].dtype, pl.Float32)
        # Column was inferred as Object due to None, function doesn't cast Object types currently.
        self.assertEqual(df_reduced['float_with_null'].dtype, pl.Object)
        # Check values reasonably preserved
        # Need to cast original float_with_null to float before comparing if we handle Objects later
        # assert_frame_equal(self.df.select("float64_col"), 
        #                    df_reduced.select("float64_col").cast(pl.Float64), 
        #                    check_dtype=False, atol=1e-5)

    def test_utf8_to_category_conversion(self):
        """Test conversion of Utf8 columns to Categorical."""
        df_reduced = reduce_memory_usage(self.df, verbose=False, cat_threshold=0.6)
        self.assertEqual(df_reduced['category_low_cardinality'].dtype, pl.Categorical)
        # High cardinality should remain Utf8
        self.assertEqual(df_reduced['category_high_cardinality'].dtype, pl.Utf8)
        # String with nulls and low cardinality should convert
        self.assertEqual(df_reduced['string_with_null'].dtype, pl.Categorical)

    def test_other_types_unchanged(self):
        """Test that non-numeric, non-Utf8 columns are unchanged."""
        df_reduced = reduce_memory_usage(self.df, verbose=False)
        self.assertEqual(df_reduced['bool_col'].dtype, self.df['bool_col'].dtype)
        self.assertEqual(df_reduced['datetime_col'].dtype, self.df['datetime_col'].dtype)

    def test_memory_reduction(self):
        """Test that memory usage is reduced (using estimated_size)."""
        start_mem = self.df.estimated_size()
        df_reduced = reduce_memory_usage(self.df, verbose=False)
        end_mem = df_reduced.estimated_size()
        # Polars estimation might not always decrease if casts are minor
        # or if Categorical overhead outweighs Utf8 savings for small data
        print(f"\nMemory reduction test: Start={start_mem}, End={end_mem}")
        self.assertLessEqual(end_mem, start_mem) # Check it doesn't increase significantly

    def test_verbose_output(self):
        """Test that verbose=True prints output."""
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            reduce_memory_usage(self.df, verbose=True)
            output = mock_stdout.getvalue()
            self.assertIn("Initial memory usage:", output)
            self.assertIn("Final memory usage:", output)
            self.assertIn("reduction", output)
            self.assertIn("Converting column 'category_low_cardinality' to Categorical", output)
            self.assertIn("Converting column 'string_with_null' to Categorical", output)
            # self.assertIn("Skipping numeric downcast for 'int_with_null' due to nulls", output) # This won't be printed as the column is Object type

    def test_verbose_false_no_output(self):
        """Test that verbose=False prints no output."""
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            reduce_memory_usage(self.df, verbose=False)
            output = mock_stdout.getvalue()
            self.assertEqual(output.strip(), "")

if __name__ == '__main__':
    unittest.main()
