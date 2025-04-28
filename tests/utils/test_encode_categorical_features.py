import unittest
import polars as pl
import numpy as np
from polars.testing import assert_frame_equal
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from utils.encode_categorical_features import encode_categorical_features

class TestEncodeCategoricalFeatures(unittest.TestCase):

    def setUp(self):
        """Set up sample Polars DataFrames for testing."""
        self.train_data = pl.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'cat1': ['A', 'B', 'A', 'C', 'B'],
            'cat2': ['X', 'Y', 'X', 'X', 'Z'],
            'num1': [10, 20, 15, 25, 10]
        })
        self.test_data = pl.DataFrame({
            'id': [6, 7, 8],
            'cat1': ['C', 'A', 'D'], # D is unknown
            'cat2': ['Y', 'Z', 'X'],
            'num1': [12, 18, 30]
        })
        self.features_to_encode = ['cat1', 'cat2']

    def test_onehot_encoder_defaults(self):
        """Test OneHotEncoder with default settings (drop=True, handle_unknown='ignore')."""
        train_encoded, test_encoded, encoder = encode_categorical_features(
            train_df=self.train_data,
            features=self.features_to_encode,
            test_df=self.test_data,
            encoder_type='onehot'
        )

        self.assertIsInstance(encoder, OneHotEncoder)
        self.assertIsInstance(train_encoded, pl.DataFrame)
        self.assertIsInstance(test_encoded, pl.DataFrame)
        self.assertNotIn('cat1', train_encoded.columns)
        self.assertNotIn('cat2', train_encoded.columns)
        self.assertIn('cat1_A', train_encoded.columns) # Check generated columns
        self.assertIn('cat2_X', train_encoded.columns)
        self.assertIn('num1', train_encoded.columns) # Check numeric column preserved

        # Check shapes (original: 4 cols, encoded: 1 id + 1 num + 3 cat1 + 3 cat2 = 8 cols)
        self.assertEqual(train_encoded.shape, (5, 8))
        self.assertEqual(test_encoded.shape, (3, 8))

        # Check unknown category 'D' in test_df for cat1 (handle_unknown='ignore')
        # Row index 2 (id=8) should have 0 for cat1_A, cat1_B, cat1_C
        test_row_unknown = test_encoded.filter(pl.col('id') == 8)
        self.assertEqual(test_row_unknown.select(['cat1_A', 'cat1_B', 'cat1_C']).row(0), (0.0, 0.0, 0.0))
        # Also check a known category in test - row 1 (id=7) cat1='A'
        test_row_known = test_encoded.filter(pl.col('id') == 7)
        self.assertEqual(test_row_known.select(['cat1_A', 'cat1_B', 'cat1_C']).row(0), (1.0, 0.0, 0.0))

        # Check dtypes of new columns are numeric (likely float from OHE)
        self.assertTrue(train_encoded['cat1_A'].dtype.is_numeric())
        self.assertTrue(train_encoded['cat2_Y'].dtype.is_numeric())

    def test_onehot_encoder_no_drop(self):
        """Test OneHotEncoder with drop_original=False."""
        train_encoded, _, _ = encode_categorical_features(
            train_df=self.train_data,
            features=self.features_to_encode,
            encoder_type='onehot',
            drop_original=False
        )
        self.assertIn('cat1', train_encoded.columns)
        self.assertIn('cat2', train_encoded.columns)
        self.assertIn('cat1_A', train_encoded.columns)
        # Shape: 5 rows, original 4 cols + 3 cat1 + 3 cat2 = 10 cols
        self.assertEqual(train_encoded.shape, (5, 10))

    def test_onehot_encoder_prefix(self):
        """Test OneHotEncoder with a column prefix."""
        prefix = "pref_"
        train_encoded, _, _ = encode_categorical_features(
            train_df=self.train_data,
            features=self.features_to_encode,
            encoder_type='onehot',
            new_col_prefix=prefix
            # drop=True is default
        )
        self.assertNotIn('cat1', train_encoded.columns)
        self.assertIn(f'{prefix}cat1_A', train_encoded.columns)
        self.assertIn(f'{prefix}cat2_X', train_encoded.columns)
        self.assertEqual(train_encoded.shape, (5, 8)) # Shape should be same as default

    def test_onehot_encoder_handle_unknown_error(self):
        """Test OneHotEncoder with handle_unknown='error'."""
        # Sklearn raises ValueError when unknown categories are found with handle_unknown='error'
        with self.assertRaises((ValueError, pl.exceptions.ComputeError)):
            encode_categorical_features(
                train_df=self.train_data,
                features=self.features_to_encode,
                test_df=self.test_data,
                encoder_type='onehot',
                handle_unknown='error' # Should fail on 'D' in test
            )

    def test_ordinal_encoder(self):
        """Test OrdinalEncoder successful fit/transform on known data."""
        train_encoded, test_encoded, encoder = encode_categorical_features(
            train_df=self.train_data,
            features=self.features_to_encode,
            test_df=self.test_data.filter(pl.col('cat1') != 'D'), # Test only on known values
            encoder_type='ordinal',
            drop_original=False # Keep original to compare easily
        )
        self.assertIsInstance(encoder, OrdinalEncoder)
        self.assertIn('cat1', train_encoded.columns)
        self.assertIn('cat2', train_encoded.columns)
        self.assertTrue(train_encoded['cat1'].dtype.is_integer()) # Default is integer
        self.assertTrue(train_encoded['cat2'].dtype.is_integer())
        self.assertEqual(train_encoded.shape, self.train_data.shape)

        # Check encoding values (e.g., A -> 0, B -> 1, C -> 2)
        self.assertEqual(train_encoded.filter(pl.col('id') == 1).select('cat1').item(), 0)
        self.assertEqual(train_encoded.filter(pl.col('id') == 2).select('cat1').item(), 1)
        self.assertEqual(train_encoded.filter(pl.col('id') == 4).select('cat1').item(), 2)
        self.assertEqual(train_encoded.filter(pl.col('id') == 1).select('cat2').item(), 0)
        self.assertEqual(train_encoded.filter(pl.col('id') == 2).select('cat2').item(), 1)
        self.assertEqual(train_encoded.filter(pl.col('id') == 5).select('cat2').item(), 2)
        
        # Check test encoding for known value
        self.assertEqual(test_encoded.filter(pl.col('id') == 7).select('cat1').item(), 0) # id 7 has cat1='A'

    def test_ordinal_encoder_prefix_and_drop(self):
        """Test OrdinalEncoder with prefix and drop."""
        prefix = "ordinal_"
        train_encoded, _, _ = encode_categorical_features(
            train_df=self.train_data,
            features=self.features_to_encode,
            encoder_type='ordinal',
            new_col_prefix=prefix,
            drop_original=True
        )
        self.assertNotIn('cat1', train_encoded.columns)
        self.assertIn(f'{prefix}cat1', train_encoded.columns)
        self.assertIn(f'{prefix}cat2', train_encoded.columns)
        self.assertTrue(train_encoded[f'{prefix}cat1'].dtype.is_integer())
        self.assertEqual(train_encoded.shape, self.train_data.shape) # Shape remains same

    def test_ordinal_encoder_unknown_error_default(self):
        """Test OrdinalEncoder raises error on unknown category with default settings."""
        # OrdinalEncoder raises ValueError by default for unknown categories
        with self.assertRaises((ValueError, pl.exceptions.ComputeError)) as cm:
             encode_categorical_features(
                train_df=self.train_data, # Fit on train
                features=self.features_to_encode,
                test_df=self.test_data, # Transform test (contains 'D')
                encoder_type='ordinal' # Default handle_unknown='error'
            )
        # Check the error message content if needed (can be brittle)
        self.assertTrue(
             "Found unknown categories" in str(cm.exception) or 
             "could not convert" in str(cm.exception).lower() # Polars might raise compute error
        )

    def test_ordinal_encoder_handle_unknown(self):
        """Test OrdinalEncoder with handle_unknown options."""
        # Using default np.nan for unknown categories
        train_encoded, test_encoded, encoder = encode_categorical_features(
            train_df=self.train_data,
            features=self.features_to_encode,
            test_df=self.test_data,
            encoder_type='ordinal',
            handle_unknown='use_encoded_value',
            # unknown_value=np.nan is now the default in the function
            drop_original=False # Keep original columns to check encoded values
        )
        self.assertIn('cat1', test_encoded.columns) # Column should exist now
        # Check row index 2 (id=8), cat1='D' should be encoded as NaN
        unknown_row = test_encoded.filter(pl.col('id') == 8)
        self.assertTrue(np.isnan(unknown_row.select('cat1').item()))
        # Check dtype is float because of NaN
        self.assertTrue(test_encoded['cat1'].dtype.is_float())
        # Known value 'A' in test should be encoded correctly (0.0)
        known_row = test_encoded.filter(pl.col('id') == 7)
        self.assertEqual(known_row.select('cat1').item(), 0.0)


    def test_invalid_encoder_type(self):
        """Test invalid encoder type."""
        with self.assertRaisesRegex(ValueError, "Unsupported encoder_type: 'invalid'"):
            encode_categorical_features(self.train_data, self.features_to_encode, encoder_type='invalid')

    def test_missing_feature(self):
        """Test providing a feature not in the DataFrame."""
        with self.assertRaises(pl.exceptions.ColumnNotFoundError):
            encode_categorical_features(self.train_data, ['cat1', 'missing_feature'], encoder_type='onehot')

if __name__ == '__main__':
    unittest.main() 