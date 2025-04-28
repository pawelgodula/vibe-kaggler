import unittest
import polars as pl
from polars.testing import assert_frame_equal
import numpy as np # Keep if needed for setup
from utils.create_interaction_features import create_interaction_features

class TestCreateInteractionFeatures(unittest.TestCase):

    def setUp(self):
        """Set up a sample Polars DataFrame."""
        self.data = {
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [-1, 0, 1],
            'D_cat': ['x', 'y', 'z']
        }
        self.df = pl.DataFrame(self.data).with_columns([
             pl.col('A').cast(pl.Int16),
             pl.col('B').cast(pl.Int16),
             pl.col('C').cast(pl.Int16),
             pl.col('D_cat').cast(pl.Utf8)
        ])

    def test_basic_interactions(self):
        """Test creating basic interactions."""
        pairs = [('A', 'B'), ('A', 'C')]
        df_interactions = create_interaction_features(self.df, pairs)

        # Polars sorts column names in select by default?
        # Check expected columns exist, order might vary or be fixed
        # Let's create expected DataFrame for comparison
        expected_data = {
            'A_x_B': [1*4, 2*5, 3*6],
            'A_x_C': [1*(-1), 2*0, 3*1]
        }
        # Cast to expected output type (result of Int16 * Int16 -> Int32? Check Polars)
        # Polars promotes Int16*Int16 to Int32
        expected_df = pl.DataFrame(expected_data).select([
            pl.col('A_x_B').cast(pl.Int32),
            pl.col('A_x_C').cast(pl.Int32),
        ])
        
        # Ensure columns are in same order for assertion
        assert_frame_equal(df_interactions.select(sorted(df_interactions.columns)),
                           expected_df.select(sorted(expected_df.columns)))

    def test_multiple_interactions(self):
        """Test creating multiple interactions including duplicates."""
        pairs = [('A', 'B'), ('B', 'C'), ('A', 'C'), ('B', 'A')] # ('B', 'A') duplicate
        df_interactions = create_interaction_features(self.df, pairs)

        # Expect unique interactions: A_x_B, A_x_C, B_x_C (sorted names)
        expected_data = {
            'A_x_B': [4, 10, 18],
            'A_x_C': [-1, 0, 3],
            'B_x_C': [-4, 0, 6]
        }
        expected_df = pl.DataFrame(expected_data).select([
            pl.all().cast(pl.Int32)
        ])

        assert_frame_equal(df_interactions.select(sorted(df_interactions.columns)),
                           expected_df.select(sorted(expected_df.columns)))

    def test_empty_pairs_list(self):
        """Test providing an empty list of pairs."""
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            create_interaction_features(self.df, [])

    def test_invalid_pair_format(self):
        """Test providing invalid items in the pairs list."""
        with self.assertRaisesRegex(ValueError, "must be a tuple of length 2"):
            create_interaction_features(self.df, [('A', 'B'), ('C',)])
        # This case might need refinement depending on how the list comprehension fails
        with self.assertRaises((TypeError, ValueError)): # Polars might raise different error here
             create_interaction_features(self.df, ['A', 'B'])

    def test_missing_feature(self):
        """Test providing a feature not in the DataFrame."""
        with self.assertRaises(pl.exceptions.ColumnNotFoundError):
            create_interaction_features(self.df, [('A', 'X')])

    def test_non_numeric_feature(self):
        """Test providing a non-numeric feature."""
        with self.assertRaises(pl.exceptions.ComputeError):
            create_interaction_features(self.df, [('A', 'D_cat')])

if __name__ == '__main__':
    unittest.main() 