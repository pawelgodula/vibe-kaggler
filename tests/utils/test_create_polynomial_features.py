import unittest
import polars as pl
import numpy as np
from polars.testing import assert_frame_equal
from utils.create_polynomial_features import create_polynomial_features

class TestCreatePolynomialFeatures(unittest.TestCase):

    def setUp(self):
        """Set up a sample Polars DataFrame."""
        self.df = pl.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9],
            'D_cat': ['x', 'y', 'z']
        })
        self.features = ['A', 'B']
        self.original_cols = self.df.columns
        self.prefix = "poly_"

    def test_degree_2_defaults(self):
        """Test degree=2, interaction_only=False, include_bias=False, prefix='poly_'."""
        df_poly = create_polynomial_features(self.df, self.features)
        
        self.assertIsInstance(df_poly, pl.DataFrame)

        # Check original columns are preserved
        for col in self.original_cols:
            self.assertIn(col, df_poly.columns)
            assert_frame_equal(df_poly.select(col), self.df.select(col))

        # Expected new features for degree 2: A, B, A^2, A*B, B^2
        expected_new_cols_raw = ['A', 'B', 'A^2', 'A B', 'B^2']
        expected_new_cols = [f"{self.prefix}{name}" for name in expected_new_cols_raw]
        for col in expected_new_cols:
            self.assertIn(col, df_poly.columns)
            self.assertTrue(df_poly[col].dtype.is_float()) # Check they are float
        
        # Check final shape: original (4) + new (5) = 9 cols
        self.assertEqual(df_poly.shape, (3, len(self.original_cols) + len(expected_new_cols)))

        # Check values for the first row (A=1, B=4)
        # A=1, B=4, A^2=1, A*B=4, B^2=16
        expected_row_0_poly = [1.0, 4.0, 1.0, 4.0, 16.0]
        row_0_poly = df_poly.filter(pl.col('A')==1).select(expected_new_cols).row(0)
        np.testing.assert_array_almost_equal(row_0_poly, expected_row_0_poly)

        # Check values for the second row (A=2, B=5)
        # A=2, B=5, A^2=4, A*B=10, B^2=25
        expected_row_1_poly = [2.0, 5.0, 4.0, 10.0, 25.0]
        row_1_poly = df_poly.filter(pl.col('A')==2).select(expected_new_cols).row(0)
        np.testing.assert_array_almost_equal(row_1_poly, expected_row_1_poly)

    def test_degree_3(self):
        """Test degree=3."""
        df_poly = create_polynomial_features(self.df, self.features, degree=3)
        # Expected: A, B, A^2, A*B, B^2, A^3, A^2*B, A*B^2, B^3 (9 new cols)
        expected_new_count = 9
        self.assertEqual(df_poly.shape, (3, len(self.original_cols) + expected_new_count))
        self.assertIn(f'{self.prefix}A^3', df_poly.columns)
        self.assertIn(f'{self.prefix}A^2 B', df_poly.columns)
        self.assertIn(f'{self.prefix}A B^2', df_poly.columns)
        self.assertIn(f'{self.prefix}B^3', df_poly.columns)

        # Check A^3 for row 1 (A=2) -> 8
        self.assertAlmostEqual(df_poly.filter(pl.col('A')==2).select(f'{self.prefix}A^3').item(), 8.0)
        # Check A*B^2 for row 1 (A=2, B=5) -> 2 * 25 = 50
        self.assertAlmostEqual(df_poly.filter(pl.col('A')==2).select(f'{self.prefix}A B^2').item(), 50.0)

    def test_interaction_only(self):
        """Test interaction_only=True."""
        df_poly = create_polynomial_features(self.df, self.features, degree=2, interaction_only=True)
        # Expected new: A, B, A*B (3 new cols)
        expected_new_cols_raw = ['A', 'B', 'A B']
        expected_new_cols = [f"{self.prefix}{name}" for name in expected_new_cols_raw]
        self.assertEqual(df_poly.shape, (3, len(self.original_cols) + len(expected_new_cols)))
        for col in expected_new_cols:
            self.assertIn(col, df_poly.columns)
        self.assertNotIn(f'{self.prefix}A^2', df_poly.columns)
        self.assertNotIn(f'{self.prefix}B^2', df_poly.columns)

        # Check values for row 1 (A=2, B=5) -> A=2, B=5, A*B=10
        expected_row_1_poly = [2.0, 5.0, 10.0]
        row_1_poly = df_poly.filter(pl.col('A')==2).select(expected_new_cols).row(0)
        np.testing.assert_array_almost_equal(row_1_poly, expected_row_1_poly)

    def test_include_bias(self):
        """Test include_bias=True."""
        df_poly = create_polynomial_features(self.df, self.features, degree=2, include_bias=True)
        # Expected new: 1, A, B, A^2, A*B, B^2 (6 new cols)
        expected_new_cols_raw = ['1', 'A', 'B', 'A^2', 'A B', 'B^2']
        expected_new_cols = [f"{self.prefix}{name}" for name in expected_new_cols_raw]
        self.assertEqual(df_poly.shape, (3, len(self.original_cols) + len(expected_new_cols)))
        for col in expected_new_cols:
            self.assertIn(col, df_poly.columns)
        # Check bias column is all 1s
        np.testing.assert_array_almost_equal(df_poly[f'{self.prefix}1'].to_numpy(), [1.0, 1.0, 1.0])

    def test_custom_prefix(self):
        """Test using a custom prefix."""
        custom_prefix = "feat_"
        df_poly = create_polynomial_features(self.df, self.features, degree=2, new_col_prefix=custom_prefix)
        self.assertIn(f'{custom_prefix}A', df_poly.columns)
        self.assertIn(f'{custom_prefix}B^2', df_poly.columns)
        self.assertNotIn('poly_A', df_poly.columns) # Default prefix should not be used
        self.assertEqual(df_poly.shape, (3, len(self.original_cols) + 5)) # 5 new cols

    def test_single_feature(self):
        """Test with a single input feature."""
        df_poly = create_polynomial_features(self.df, ['A'], degree=3)
        # Expected new: A, A^2, A^3 (3 new cols)
        expected_new_cols_raw = ['A', 'A^2', 'A^3']
        expected_new_cols = [f"{self.prefix}{name}" for name in expected_new_cols_raw]
        self.assertEqual(df_poly.shape, (3, len(self.original_cols) + len(expected_new_cols)))
        for col in expected_new_cols:
            self.assertIn(col, df_poly.columns)
        # Check A^3 for row 2 (A=3) -> 27
        self.assertAlmostEqual(df_poly.filter(pl.col('A')==3).select(f'{self.prefix}A^3').item(), 27.0)

    def test_missing_feature(self):
        """Test providing a feature not in the DataFrame."""
        with self.assertRaises(pl.exceptions.ColumnNotFoundError):
            create_polynomial_features(self.df, ['A', 'X'])

    def test_non_numeric_feature(self):
        """Test providing a non-numeric feature."""
        with self.assertRaises(pl.exceptions.ComputeError):
            create_polynomial_features(self.df, ['A', 'D_cat'])

if __name__ == '__main__':
    unittest.main() 