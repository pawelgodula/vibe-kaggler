import unittest
import polars as pl
import numpy as np
from polars.testing import assert_frame_equal
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.exceptions import NotFittedError

# Assuming the function is in 'utils.scale_numerical_features'
from utils.scale_numerical_features import scale_numerical_features

class TestScaleNumericalFeatures(unittest.TestCase):

    def setUp(self):
        """Set up test data for each test method."""
        self.train_data = pl.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'numeric1': [10.0, 20.0, 30.0, 40.0, 50.0],
            'numeric2': [1.0, 2.0, 3.0, 4.0, 5.0],
            'category': ['A', 'B', 'A', 'C', 'B']
        })
        self.test_data = pl.DataFrame({
            'id': [6, 7, 8],
            'numeric1': [60.0, 70.0, 80.0],
            'numeric2': [6.0, 7.0, 8.0],
            'category': ['A', 'C', 'B']
        })
        self.features_to_scale = ['numeric1', 'numeric2']

    def test_standard_scaler(self):
        """Test scaling with StandardScaler."""
        train_scaled, test_scaled, scaler = scale_numerical_features(
            self.train_data, self.features_to_scale, self.test_data, scaler_type='standard'
        )

        # Check types
        self.assertIsInstance(train_scaled, pl.DataFrame)
        self.assertIsInstance(test_scaled, pl.DataFrame)
        self.assertIsInstance(scaler, StandardScaler)

        # Check shapes
        self.assertEqual(train_scaled.shape, self.train_data.shape)
        self.assertEqual(test_scaled.shape, self.test_data.shape)

        # Check that non-scaled columns are untouched
        assert_frame_equal(train_scaled.select(['id', 'category']), self.train_data.select(['id', 'category']))
        assert_frame_equal(test_scaled.select(['id', 'category']), self.test_data.select(['id', 'category']))

        # Check scaled values (using numpy for easier mean/std check)
        train_scaled_np = train_scaled.select(self.features_to_scale).to_numpy()
        test_scaled_np = test_scaled.select(self.features_to_scale).to_numpy()
        
        np.testing.assert_almost_equal(train_scaled_np.mean(axis=0), [0.0, 0.0])
        np.testing.assert_almost_equal(train_scaled_np.std(axis=0), [1.0, 1.0])

        # Manually scale test data for comparison
        manual_scaler = StandardScaler().fit(self.train_data.select(self.features_to_scale).to_numpy())
        expected_test_scaled_np = manual_scaler.transform(self.test_data.select(self.features_to_scale).to_numpy())
        np.testing.assert_almost_equal(test_scaled_np, expected_test_scaled_np)

        # Check dtypes are float
        for col in self.features_to_scale:
            self.assertTrue(train_scaled[col].dtype.is_float())
            self.assertTrue(test_scaled[col].dtype.is_float())

    def test_minmax_scaler(self):
        """Test scaling with MinMaxScaler."""
        train_scaled, test_scaled, scaler = scale_numerical_features(
            self.train_data, self.features_to_scale, self.test_data, scaler_type='minmax', feature_range=(0, 1)
        )
        self.assertIsInstance(scaler, MinMaxScaler)
        train_scaled_np = train_scaled.select(self.features_to_scale).to_numpy()
        test_scaled_np = test_scaled.select(self.features_to_scale).to_numpy()

        np.testing.assert_almost_equal(train_scaled_np.min(axis=0), [0.0, 0.0])
        np.testing.assert_almost_equal(train_scaled_np.max(axis=0), [1.0, 1.0])
        
        # Manually scale test data for comparison
        manual_scaler = MinMaxScaler(feature_range=(0, 1)).fit(self.train_data.select(self.features_to_scale).to_numpy())
        expected_test_scaled_np = manual_scaler.transform(self.test_data.select(self.features_to_scale).to_numpy())
        np.testing.assert_almost_equal(test_scaled_np, expected_test_scaled_np)

    def test_robust_scaler(self):
        """Test scaling with RobustScaler."""
        train_scaled, test_scaled, scaler = scale_numerical_features(
            self.train_data, self.features_to_scale, self.test_data, scaler_type='robust'
        )
        self.assertIsInstance(scaler, RobustScaler)
        # RobustScaler centers data around the median and scales by IQR. Harder to check exact values easily.
        # We mainly check if it runs and returns correct types/shapes.
        self.assertEqual(train_scaled.shape, self.train_data.shape)
        self.assertEqual(test_scaled.shape, self.test_data.shape)
        self.assertTrue(all(train_scaled[f].dtype.is_float() for f in self.features_to_scale))
        self.assertTrue(all(test_scaled[f].dtype.is_float() for f in self.features_to_scale))


    def test_no_test_df(self):
        """Test scaling when no test_df is provided."""
        train_scaled, test_scaled, scaler = scale_numerical_features(
            self.train_data, self.features_to_scale, scaler_type='standard'
        )
        self.assertIsInstance(train_scaled, pl.DataFrame)
        self.assertIsNone(test_scaled)
        self.assertIsInstance(scaler, StandardScaler)
        self.assertEqual(train_scaled.shape, self.train_data.shape)
        np.testing.assert_almost_equal(train_scaled.select(self.features_to_scale).to_numpy().mean(axis=0), [0.0, 0.0])

    def test_scaler_kwargs(self):
        """Test passing additional kwargs to the scaler."""
        # Test RobustScaler with different quantile range
        train_scaled, _, scaler = scale_numerical_features(
            self.train_data, self.features_to_scale, scaler_type='robust', quantile_range=(30.0, 70.0)
        )
        self.assertIsInstance(scaler, RobustScaler)
        self.assertEqual(scaler.quantile_range, (30.0, 70.0))

    def test_invalid_scaler_type(self):
        """Test error handling for unsupported scaler type."""
        with self.assertRaisesRegex(ValueError, "Unsupported scaler_type: 'invalid_scaler'"):
            scale_numerical_features(self.train_data, self.features_to_scale, scaler_type='invalid_scaler')

    def test_missing_feature(self):
        """Test error handling for missing features."""
        with self.assertRaises(pl.exceptions.ColumnNotFoundError):
            scale_numerical_features(self.train_data, ['numeric1', 'non_existent'], scaler_type='standard')

    def test_non_numeric_feature(self):
        """Test error handling for non-numeric features."""
        with self.assertRaises(pl.exceptions.ComputeError):
             scale_numerical_features(self.train_data, ['numeric1', 'category'], scaler_type='standard')

    def test_empty_dataframe(self):
        """Test handling of empty DataFrames."""
        empty_df = pl.DataFrame({
            'numeric1': pl.Series([], dtype=pl.Float64),
            'numeric2': pl.Series([], dtype=pl.Float64)
        })
        # Scaling an empty DataFrame should ideally work but result in empty scaled data
        # However, scikit-learn scalers might raise errors on empty arrays.
        # Let's check for a NotFittedError or ValueError during fit/transform
        with self.assertRaises((ValueError, NotFittedError, pl.exceptions.ComputeError)):
             scale_numerical_features(empty_df, ['numeric1', 'numeric2'])
             
    def test_single_value_dataframe(self):
        """ Test handling of DataFrames with a single row (std dev = 0)."""
        single_row_df = pl.DataFrame({
            'numeric1': [10.0],
            'numeric2': [1.0]
        })
        # StandardScaler results in NaN/Inf or 0s when std dev is 0
        # We no longer check for UserWarning as it's not guaranteed.
        # Check the function handles this by producing 0s or NaNs.
        
        # Run StandardScaler
        train_scaled, _, scaler = scale_numerical_features(single_row_df, ['numeric1', 'numeric2'], scaler_type='standard')
        self.assertIsInstance(scaler, StandardScaler)
        
        # Scaled result might be all zeros or NaNs depending on sklearn version and parameters
        scaled_vals = train_scaled.select(['numeric1', 'numeric2']).to_numpy()
        self.assertTrue(np.all(scaled_vals == 0) or np.all(np.isnan(scaled_vals)), 
                        f"Expected scaled values to be all 0s or NaNs, but got {scaled_vals}")

        # MinMaxScaler should work fine and produce 0s
        train_scaled_mm, _, _ = scale_numerical_features(single_row_df, ['numeric1', 'numeric2'], scaler_type='minmax')
        np.testing.assert_almost_equal(train_scaled_mm.select(['numeric1', 'numeric2']).to_numpy(), [[0.0, 0.0]])

if __name__ == '__main__':
    unittest.main() 