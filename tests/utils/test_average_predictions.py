import unittest
import numpy as np
import polars as pl
from utils.average_predictions import average_predictions

class TestAveragePredictions(unittest.TestCase):

    def test_average_numpy_arrays(self):
        """Test averaging a list of NumPy arrays."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([3.0, 4.0, 5.0])
        arr3 = np.array([5.0, 6.0, 7.0])
        pred_list = [arr1, arr2, arr3]
        expected = np.array([3.0, 4.0, 5.0]) # (1+3+5)/3, (2+4+6)/3, (3+5+7)/3
        result = average_predictions(pred_list)
        np.testing.assert_array_almost_equal(result, expected)

    def test_average_polars_series(self):
        """Test averaging a list of Polars Series."""
        s1 = pl.Series("p1", [0.1, 0.2, 0.9])
        s2 = pl.Series("p2", [0.5, 0.6, 0.3])
        pred_list = [s1, s2]
        expected = np.array([0.3, 0.4, 0.6]) # (0.1+0.5)/2, (0.2+0.6)/2, (0.9+0.3)/2
        result = average_predictions(pred_list)
        np.testing.assert_array_almost_equal(result, expected)

    def test_average_mixed_types(self):
        """Test averaging a list of mixed NumPy arrays and Polars Series."""
        arr1 = np.array([10, 20])
        s1 = pl.Series("p1", [30, 40])
        pred_list = [arr1, s1]
        expected = np.array([20.0, 30.0]) # (10+30)/2, (20+40)/2
        result = average_predictions(pred_list)
        np.testing.assert_array_almost_equal(result, expected)
        self.assertTrue(np.issubdtype(result.dtype, np.floating)) # Should be float

    def test_average_single_prediction(self):
        """Test averaging a list with a single prediction."""
        arr1 = np.array([5, 10, 15])
        pred_list = [arr1]
        expected = np.array([5.0, 10.0, 15.0])
        result = average_predictions(pred_list)
        np.testing.assert_array_almost_equal(result, expected)
        # Also test with a Series
        s1 = pl.Series("p1", [7.0, 8.0])
        result_s = average_predictions([s1])
        np.testing.assert_array_almost_equal(result_s, np.array([7.0, 8.0]))

    def test_average_2d_arrays(self):
        """Test averaging 2D arrays (e.g., class probabilities)."""
        arr1 = np.array([[0.1, 0.9], [0.8, 0.2]])
        arr2 = np.array([[0.3, 0.7], [0.6, 0.4]])
        pred_list = [arr1, arr2]
        expected = np.array([[0.2, 0.8], [0.7, 0.3]])
        result = average_predictions(pred_list)
        np.testing.assert_array_almost_equal(result, expected)

    # --- Error Handling --- 
    def test_empty_list(self):
        """Test error when the input list is empty."""
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            average_predictions([])

    def test_mismatched_shapes(self):
        """Test error when predictions have different shapes."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5])
        with self.assertRaisesRegex(ValueError, "inconsistent shapes"):
            average_predictions([arr1, arr2])
            
        s1 = pl.Series("s1", [1, 2, 3])
        arr2d = np.array([[1, 1], [2, 2], [3, 3]])
        with self.assertRaisesRegex(ValueError, "inconsistent shapes"):
             average_predictions([s1, arr2d])
             
    def test_invalid_type_in_list(self):
        """Test error if list contains unsupported types."""
        arr1 = np.array([1, 2])
        invalid_item = [3, 4] # A Python list
        with self.assertRaises(TypeError):
            average_predictions([arr1, invalid_item])

if __name__ == '__main__':
    unittest.main() 