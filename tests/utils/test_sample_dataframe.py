import unittest
import polars as pl
import numpy as np
from polars.testing import assert_frame_equal
from utils.sample_dataframe import sample_dataframe

class TestSampleDataFrame(unittest.TestCase):

    def setUp(self):
        """Set up a sample DataFrame."""
        self.n_rows = 50
        self.df = pl.DataFrame({
            'id': np.arange(self.n_rows),
            'value': np.random.rand(self.n_rows)
        })

    def test_sample_n(self):
        """Test sampling a fixed number of rows (n)."""
        n_sample = 10
        sampled_df = sample_dataframe(self.df, n=n_sample)
        self.assertEqual(len(sampled_df), n_sample)
        # Check if sampled ids are unique and from the original ids
        self.assertEqual(sampled_df['id'].n_unique(), n_sample)
        self.assertTrue(set(sampled_df['id'].to_list()).issubset(set(self.df['id'].to_list())))

    def test_sample_frac(self):
        """Test sampling a fraction of rows (frac)."""
        frac_sample = 0.2
        expected_len = int(self.n_rows * frac_sample)
        sampled_df = sample_dataframe(self.df, frac=frac_sample)
        self.assertEqual(len(sampled_df), expected_len)
        self.assertEqual(sampled_df['id'].n_unique(), expected_len)
        self.assertTrue(set(sampled_df['id'].to_list()).issubset(set(self.df['id'].to_list())))

    def test_reproducibility_with_seed(self):
        """Test that sampling is reproducible with the same random_state."""
        seed = 123
        n_sample = 15
        sampled_1 = sample_dataframe(self.df, n=n_sample, random_state=seed)
        sampled_2 = sample_dataframe(self.df, n=n_sample, random_state=seed)
        assert_frame_equal(sampled_1, sampled_2)

    def test_different_results_without_seed(self):
        """Test that sampling gives different results without a seed (usually)."""
        n_sample = 15
        sampled_1 = sample_dataframe(self.df, n=n_sample) # No seed
        sampled_2 = sample_dataframe(self.df, n=n_sample) # No seed
        # Highly unlikely to be identical for a reasonable sample size
        self.assertFalse(sampled_1.equals(sampled_2))
        
    def test_different_results_with_different_seeds(self):
        """Test that sampling gives different results with different seeds."""
        n_sample = 15
        sampled_1 = sample_dataframe(self.df, n=n_sample, random_state=1)
        sampled_2 = sample_dataframe(self.df, n=n_sample, random_state=2)
        self.assertFalse(sampled_1.equals(sampled_2))

    def test_shuffle_false_n(self):
        """Test shuffle=False with n returns the first n rows."""
        n_sample = 5
        sampled_df = sample_dataframe(self.df, n=n_sample, shuffle=False)
        expected_df = self.df.slice(0, n_sample)
        assert_frame_equal(sampled_df, expected_df)

    def test_shuffle_false_frac(self):
        """Test shuffle=False with frac returns the first frac * N rows."""
        frac_sample = 0.1
        expected_len = int(self.n_rows * frac_sample)
        sampled_df = sample_dataframe(self.df, frac=frac_sample, shuffle=False)
        expected_df = self.df.slice(0, expected_len)
        assert_frame_equal(sampled_df, expected_df)
        
    def test_sample_n_zero(self):
        """Test sampling n=0 rows."""
        sampled_df = sample_dataframe(self.df, n=0)
        self.assertEqual(len(sampled_df), 0)
        self.assertEqual(list(sampled_df.columns), list(self.df.columns))
        
    def test_sample_frac_zero(self):
        """Test sampling frac=0.0 rows."""
        sampled_df = sample_dataframe(self.df, frac=0.0)
        self.assertEqual(len(sampled_df), 0)
        self.assertEqual(list(sampled_df.columns), list(self.df.columns))
        
    def test_sample_n_all(self):
        """Test sampling n=len(df) rows."""
        sampled_df = sample_dataframe(self.df, n=self.n_rows, shuffle=True, random_state=42)
        self.assertEqual(len(sampled_df), self.n_rows)
        # Should be shuffled version of original
        self.assertFalse(sampled_df.equals(self.df)) 
        self.assertEqual(sorted(sampled_df['id'].to_list()), sorted(self.df['id'].to_list()))

    def test_sample_frac_all(self):
        """Test sampling frac=1.0 rows."""
        sampled_df = sample_dataframe(self.df, frac=1.0, shuffle=True, random_state=42)
        self.assertEqual(len(sampled_df), self.n_rows)
        self.assertFalse(sampled_df.equals(self.df))
        self.assertEqual(sorted(sampled_df['id'].to_list()), sorted(self.df['id'].to_list()))

    # --- Error Handling --- 
    def test_error_both_n_frac(self):
        """Test error if both n and frac are provided."""
        with self.assertRaisesRegex(ValueError, "Cannot specify both"):
            sample_dataframe(self.df, n=10, frac=0.5)

    def test_error_neither_n_frac(self):
        """Test error if neither n nor frac is provided."""
        with self.assertRaisesRegex(ValueError, "Must specify either"):
            sample_dataframe(self.df)

    def test_error_invalid_frac(self):
        """Test error for invalid frac values."""
        with self.assertRaisesRegex(ValueError, "must be between 0.0 and 1.0"):
            sample_dataframe(self.df, frac=-0.1)
        with self.assertRaisesRegex(ValueError, "must be between 0.0 and 1.0"):
            sample_dataframe(self.df, frac=1.1)

    def test_error_invalid_n(self):
        """Test error for invalid n values."""
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            sample_dataframe(self.df, n=-1)
        with self.assertRaisesRegex(ValueError, "cannot be greater than"):
            sample_dataframe(self.df, n=self.n_rows + 1)

if __name__ == '__main__':
    unittest.main() 