# Test suite for the internal _train_lgbm function.

import unittest
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any
from unittest.mock import patch, MagicMock, ANY

# Function to test - Assuming it's runnable from the project root
# or PYTHONPATH is configured.
from utils._train_lgbm import _train_lgbm

def create_mock_data(rows=50, features=5, is_classification=False, num_classes=2):
    """Helper to create mock pandas DataFrames and numpy arrays for testing."""
    X = pd.DataFrame(np.random.rand(rows, features), columns=[f'feat_{i}' for i in range(features)])
    if is_classification:
        if num_classes == 2:
            y = np.random.randint(0, 2, size=rows)
        else:
            y = np.random.randint(0, num_classes, size=rows)
    else: # Regression
        y = np.random.rand(rows) * 100
    return X, y

class TestTrainLGBM(unittest.TestCase):

    def setUp(self):
        """Set up common parameters and data for tests."""
        self.feature_cols = ['feat_0', 'feat_1', 'feat_2']
        self.cat_features = None # Default to no categorical features

        self.X_train, self.y_train = create_mock_data(rows=100, features=3)
        self.X_valid, self.y_valid = create_mock_data(rows=50, features=3)
        self.X_test, _ = create_mock_data(rows=30, features=3) # Test target not used

        self.base_model_params: Dict[str, Any] = {'random_state': 42, 'n_estimators': 5} # Small n_estimators for speed
        self.base_fit_params: Dict[str, Any] = {'early_stopping_rounds': 2, 'verbose': -1}

    @patch('lightgbm.LGBMRegressor')
    def test_regression(self, MockLGBMRegressor):
        """Test regression case."""
        mock_model_instance = MockLGBMRegressor.return_value
        # Define what predict should return based on input
        def predict_side_effect(df):
            if df.equals(self.X_valid):
                return np.arange(len(self.y_valid)) # Predict sequence 0..N-1 for validation
            elif df.equals(self.X_test):
                 return np.arange(len(self.X_test)) * 2 # Predict sequence 0..2N-2 for test
            raise ValueError("Unexpected input to predict")
        mock_model_instance.predict.side_effect = predict_side_effect
        mock_model_instance.fit.return_value = None # fit returns None

        model_params = {**self.base_model_params, 'objective': 'regression'}
        fit_params = self.base_fit_params

        model, val_preds, test_preds = _train_lgbm(
            self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test,
            model_params, fit_params, self.feature_cols, self.cat_features
        )

        MockLGBMRegressor.assert_called_once_with(**model_params)
        mock_model_instance.fit.assert_called_once()
        # Check fit call arguments more robustly
        fit_args, fit_kwargs = mock_model_instance.fit.call_args
        pd.testing.assert_frame_equal(fit_args[0], self.X_train) # X
        np.testing.assert_array_equal(fit_args[1], self.y_train) # y
        self.assertIn('eval_set', fit_kwargs)
        eval_X, eval_y = fit_kwargs['eval_set'][0]
        pd.testing.assert_frame_equal(eval_X, self.X_valid)
        np.testing.assert_array_equal(eval_y, self.y_valid)
        self.assertIn('callbacks', fit_kwargs)
        self.assertIsInstance(fit_kwargs['callbacks'], list)
        # Check if callbacks list is non-empty when early stopping is specified
        self.assertTrue(len(fit_kwargs['callbacks']) > 0)

        # Check predict calls and results
        self.assertEqual(mock_model_instance.predict.call_count, 2)
        predict_calls = mock_model_instance.predict.call_args_list
        # Ensure both valid and test were predicted, order doesn't matter
        called_with_valid = any(call[0][0].equals(self.X_valid) for call in predict_calls)
        called_with_test = any(call[0][0].equals(self.X_test) for call in predict_calls)
        self.assertTrue(called_with_valid)
        self.assertTrue(called_with_test)

        self.assertIs(model, mock_model_instance)
        self.assertIsInstance(val_preds, np.ndarray)
        self.assertEqual(val_preds.shape, (len(self.y_valid),))
        np.testing.assert_array_equal(val_preds, np.arange(len(self.y_valid))) # Check specific return value
        self.assertIsInstance(test_preds, np.ndarray)
        self.assertEqual(test_preds.shape, (len(self.X_test),))
        np.testing.assert_array_equal(test_preds, np.arange(len(self.X_test)) * 2) # Check specific return value

    @patch('lightgbm.LGBMClassifier')
    def test_binary_classification(self, MockLGBMClassifier):
        """Test binary classification case."""
        mock_model_instance = MockLGBMClassifier.return_value
        def predict_proba_side_effect(df):
             if df.equals(self.X_valid):
                 # Return probs (N, 2), ensure class 1 prob is index 1
                 return np.vstack([np.linspace(0.1, 0.9, len(self.y_valid)), 1-np.linspace(0.1, 0.9, len(self.y_valid))]).T
             elif df.equals(self.X_test):
                 return np.vstack([np.linspace(0.2, 0.8, len(self.X_test)), 1-np.linspace(0.2, 0.8, len(self.X_test))]).T
             raise ValueError("Unexpected input to predict_proba")
        mock_model_instance.predict_proba.side_effect = predict_proba_side_effect
        mock_model_instance.fit.return_value = None

        model_params = {**self.base_model_params, 'objective': 'binary'}
        fit_params = self.base_fit_params
        _, y_train_bin = create_mock_data(rows=100, features=3, is_classification=True, num_classes=2)
        _, y_valid_bin = create_mock_data(rows=50, features=3, is_classification=True, num_classes=2)

        model, val_preds, test_preds = _train_lgbm(
            self.X_train, y_train_bin, self.X_valid, y_valid_bin, self.X_test,
            model_params, fit_params, self.feature_cols, self.cat_features
        )

        MockLGBMClassifier.assert_called_once_with(**model_params)
        mock_model_instance.fit.assert_called_once()
        fit_args, fit_kwargs = mock_model_instance.fit.call_args
        _, eval_y = fit_kwargs['eval_set'][0]
        np.testing.assert_array_equal(eval_y, y_valid_bin)

        self.assertEqual(mock_model_instance.predict_proba.call_count, 2)
        self.assertIs(model, mock_model_instance)
        self.assertIsInstance(val_preds, np.ndarray)
        self.assertEqual(val_preds.shape, (len(y_valid_bin),)) # Should return proba of class 1
        self.assertIsInstance(test_preds, np.ndarray)
        self.assertEqual(test_preds.shape, (len(self.X_test),))
        self.assertTrue(np.all((val_preds >= 0) & (val_preds <= 1)))
        # Check the specific value returned (proba of class 1)
        expected_val_preds = 1 - np.linspace(0.1, 0.9, len(y_valid_bin))
        np.testing.assert_allclose(val_preds, expected_val_preds)

    @patch('lightgbm.LGBMClassifier')
    def test_multiclass_classification(self, MockLGBMClassifier):
        """Test multi-class classification case."""
        num_classes = 3
        mock_model_instance = MockLGBMClassifier.return_value
        def predict_proba_side_effect(df):
            if df.equals(self.X_valid):
                 return np.random.rand(len(self.y_valid), num_classes) # Return (N, k)
            elif df.equals(self.X_test):
                 return np.random.rand(len(self.X_test), num_classes)
            raise ValueError("Unexpected input to predict_proba")
        mock_model_instance.predict_proba.side_effect = predict_proba_side_effect
        mock_model_instance.fit.return_value = None

        model_params = {**self.base_model_params, 'objective': 'multiclass', 'num_class': num_classes}
        fit_params = self.base_fit_params
        _, y_train_multi = create_mock_data(rows=100, features=3, is_classification=True, num_classes=num_classes)
        _, y_valid_multi = create_mock_data(rows=50, features=3, is_classification=True, num_classes=num_classes)

        model, val_preds, test_preds = _train_lgbm(
            self.X_train, y_train_multi, self.X_valid, y_valid_multi, self.X_test,
            model_params, fit_params, self.feature_cols, self.cat_features
        )

        MockLGBMClassifier.assert_called_once_with(**model_params)
        mock_model_instance.fit.assert_called_once()

        self.assertEqual(mock_model_instance.predict_proba.call_count, 2)
        self.assertIs(model, mock_model_instance)
        self.assertIsInstance(val_preds, np.ndarray)
        self.assertEqual(val_preds.shape, (len(y_valid_multi), num_classes)) # Full proba matrix
        self.assertIsInstance(test_preds, np.ndarray)
        self.assertEqual(test_preds.shape, (len(self.X_test), num_classes))

    @patch('lightgbm.LGBMRegressor')
    def test_no_test_data(self, MockLGBMRegressor):
        """Test case where X_test is None."""
        mock_model_instance = MockLGBMRegressor.return_value
        mock_model_instance.predict.return_value = np.random.rand(len(self.y_valid))
        mock_model_instance.fit.return_value = None

        model_params = {**self.base_model_params, 'objective': 'regression'}
        fit_params = self.base_fit_params

        model, val_preds, test_preds = _train_lgbm(
            self.X_train, self.y_train, self.X_valid, self.y_valid, None, # X_test is None
            model_params, fit_params, self.feature_cols, self.cat_features
        )

        # Check predict called only once with validation data
        mock_model_instance.predict.assert_called_once_with(self.X_valid)
        self.assertIsNone(test_preds) # test_preds should be None

    @patch('lightgbm.LGBMRegressor')
    def test_categorical_features(self, MockLGBMRegressor):
        """Test passing categorical features."""
        mock_model_instance = MockLGBMRegressor.return_value
        mock_model_instance.predict.side_effect = [np.random.rand(len(self.y_valid)), np.random.rand(len(self.X_test))]
        mock_model_instance.fit.return_value = None
        cat_features = ['feat_0']

        model_params = {**self.base_model_params, 'objective': 'regression'}
        fit_params = self.base_fit_params

        _train_lgbm(
            self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test,
            model_params, fit_params, self.feature_cols, cat_features # Pass cat_features
        )

        # Check fit was called with categorical_feature param in kwargs
        fit_args, fit_kwargs = mock_model_instance.fit.call_args
        self.assertIn('categorical_feature', fit_kwargs)
        self.assertEqual(fit_kwargs['categorical_feature'], cat_features)

if __name__ == '__main__':
    unittest.main() 