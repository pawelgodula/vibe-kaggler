# Test suite for the internal _train_xgb function.

import unittest
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, Optional
from unittest.mock import patch, MagicMock, ANY

# Function to test
from utils._train_xgb import _train_xgb

# Re-use mock data creation helper if needed, or define specific one
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

class TestTrainXGB(unittest.TestCase):

    def setUp(self):
        """Set up common parameters and data for tests."""
        self.feature_cols = ['feat_0', 'feat_1', 'feat_2']
        self.cat_features = None # XGBoost needs explicit handling

        self.X_train, self.y_train = create_mock_data(rows=100, features=3)
        self.X_valid, self.y_valid = create_mock_data(rows=50, features=3)
        self.X_test, _ = create_mock_data(rows=30, features=3)

        # Use XGBoost specific objectives
        self.base_model_params: Dict[str, Any] = {'random_state': 42, 'tree_method': 'hist'} # Use hist for speed
        self.base_fit_params: Dict[str, Any] = {'num_boost_round': 5, 'early_stopping_rounds': 2, 'verbose': 0} # verbose=0 or False

    # Patch xgb.train and the DMatrix constructor
    @patch('xgboost.train')
    @patch('xgboost.DMatrix')
    def test_regression(self, MockDMatrix, mock_xgb_train):
        """Test regression case (e.g., reg:squarederror)."""
        mock_booster = MagicMock(spec=xgb.Booster)
        mock_booster.best_iteration = 4 # Mock early stopping
        mock_xgb_train.return_value = mock_booster

        # Mock DMatrix instances returned by the constructor
        mock_dtrain = MagicMock()
        mock_dvalid = MagicMock()
        mock_dtest = MagicMock()
        MockDMatrix.side_effect = [mock_dtrain, mock_dvalid, mock_dtest]

        # Define predict results
        mock_booster.predict.side_effect = lambda dmat, iteration_range: \
            np.arange(50) if dmat == mock_dvalid else \
            np.arange(30) * 2 if dmat == mock_dtest else \
            None

        model_params = {**self.base_model_params, 'objective': 'reg:squarederror'}
        fit_params = self.base_fit_params

        model, val_preds, test_preds = _train_xgb(
            self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test,
            model_params, fit_params, self.feature_cols, self.cat_features
        )

        # Check DMatrix constructor calls
        self.assertEqual(MockDMatrix.call_count, 3)
        dmatrix_calls = MockDMatrix.call_args_list
        # Check train call args
        train_call_args, train_call_kwargs = dmatrix_calls[0]
        pd.testing.assert_frame_equal(train_call_args[0], self.X_train[self.feature_cols])
        np.testing.assert_array_equal(train_call_kwargs['label'], self.y_train)
        self.assertEqual(train_call_kwargs['feature_names'], self.feature_cols)
        self.assertEqual(train_call_kwargs['enable_categorical'], False)
        # Check valid call args
        valid_call_args, valid_call_kwargs = dmatrix_calls[1]
        pd.testing.assert_frame_equal(valid_call_args[0], self.X_valid[self.feature_cols])
        np.testing.assert_array_equal(valid_call_kwargs['label'], self.y_valid)
        self.assertEqual(valid_call_kwargs['feature_names'], self.feature_cols)
        self.assertEqual(valid_call_kwargs['enable_categorical'], False)
        # Check test call args
        test_call_args, test_call_kwargs = dmatrix_calls[2]
        pd.testing.assert_frame_equal(test_call_args[0], self.X_test[self.feature_cols])
        self.assertNotIn('label', test_call_kwargs) # No label for test
        self.assertEqual(test_call_kwargs['feature_names'], self.feature_cols)
        self.assertEqual(test_call_kwargs['enable_categorical'], False)

        # Check xgb.train call
        mock_xgb_train.assert_called_once_with(
            params=model_params,
            dtrain=mock_dtrain,
            num_boost_round=fit_params['num_boost_round'],
            evals=[(mock_dtrain, 'train'), (mock_dvalid, 'validation')],
            early_stopping_rounds=fit_params['early_stopping_rounds'],
            verbose_eval=fit_params['verbose']
        )

        # Check predict calls (uses best_iteration)
        self.assertEqual(mock_booster.predict.call_count, 2)
        mock_booster.predict.assert_any_call(mock_dvalid, iteration_range=(0, mock_booster.best_iteration))
        mock_booster.predict.assert_any_call(mock_dtest, iteration_range=(0, mock_booster.best_iteration))

        # Check results
        self.assertIs(model, mock_booster)
        self.assertIsInstance(val_preds, np.ndarray)
        self.assertEqual(val_preds.shape, (50,))
        np.testing.assert_array_equal(val_preds, np.arange(50))
        self.assertIsInstance(test_preds, np.ndarray)
        self.assertEqual(test_preds.shape, (30,))
        np.testing.assert_array_equal(test_preds, np.arange(30) * 2)

    @patch('xgboost.train')
    @patch('xgboost.DMatrix')
    def test_binary_classification(self, MockDMatrix, mock_xgb_train):
        """Test binary classification case (e.g., binary:logistic)."""
        mock_booster = MagicMock(spec=xgb.Booster)
        mock_booster.best_iteration = 5 # No early stopping needed if rounds == num_boost_round
        mock_xgb_train.return_value = mock_booster

        mock_dtrain, mock_dvalid, mock_dtest = MagicMock(), MagicMock(), MagicMock()
        MockDMatrix.side_effect = [mock_dtrain, mock_dvalid, mock_dtest]

        # Predict usually returns probabilities for logistic objective
        mock_booster.predict.side_effect = lambda dmat, iteration_range: \
            np.linspace(0.1, 0.9, 50) if dmat == mock_dvalid else \
            np.linspace(0.2, 0.8, 30) if dmat == mock_dtest else \
            None

        model_params = {**self.base_model_params, 'objective': 'binary:logistic'}
        fit_params = {**self.base_fit_params, 'early_stopping_rounds': None} # Test without early stopping
        _, y_train_bin = create_mock_data(rows=100, features=3, is_classification=True, num_classes=2)
        _, y_valid_bin = create_mock_data(rows=50, features=3, is_classification=True, num_classes=2)

        model, val_preds, test_preds = _train_xgb(
            self.X_train, y_train_bin, self.X_valid, y_valid_bin, self.X_test,
            model_params, fit_params, self.feature_cols, self.cat_features
        )

        self.assertEqual(MockDMatrix.call_count, 3)
        # Check call without early_stopping_rounds when it's None in fit_params
        expected_train_args = {
             'params': model_params,
             'dtrain': mock_dtrain,
             'num_boost_round': fit_params['num_boost_round'],
             'evals': [(mock_dtrain, 'train'), (mock_dvalid, 'validation')],
             'verbose_eval': fit_params['verbose']
        }
        mock_xgb_train.assert_called_once_with(**expected_train_args)

        self.assertEqual(mock_booster.predict.call_count, 2)
        pred_iter_range = (0, fit_params['num_boost_round']) # Use num_boost_round if no early stopping
        mock_booster.predict.assert_any_call(mock_dvalid, iteration_range=pred_iter_range)
        mock_booster.predict.assert_any_call(mock_dtest, iteration_range=pred_iter_range)

        self.assertIs(model, mock_booster)
        self.assertIsInstance(val_preds, np.ndarray)
        self.assertEqual(val_preds.shape, (50,))
        np.testing.assert_allclose(val_preds, np.linspace(0.1, 0.9, 50))
        self.assertTrue(np.all((val_preds >= 0) & (val_preds <= 1)))
        self.assertIsInstance(test_preds, np.ndarray)
        self.assertEqual(test_preds.shape, (30,))
        np.testing.assert_allclose(test_preds, np.linspace(0.2, 0.8, 30))

    @patch('xgboost.train')
    @patch('xgboost.DMatrix')
    def test_multiclass_classification(self, MockDMatrix, mock_xgb_train):
        """Test multi-class classification case (e.g., multi:softprob)."""
        num_classes = 3
        mock_booster = MagicMock(spec=xgb.Booster)
        mock_booster.best_iteration = 5
        mock_xgb_train.return_value = mock_booster

        mock_dtrain, mock_dvalid, mock_dtest = MagicMock(), MagicMock(), MagicMock()
        MockDMatrix.side_effect = [mock_dtrain, mock_dvalid, mock_dtest]

        # Predict returns (N, K) probabilities for softprob
        mock_booster.predict.side_effect = lambda dmat, iteration_range: \
            np.random.rand(50, num_classes) if dmat == mock_dvalid else \
            np.random.rand(30, num_classes) if dmat == mock_dtest else \
            None

        model_params = {**self.base_model_params, 'objective': 'multi:softprob', 'num_class': num_classes}
        fit_params = self.base_fit_params
        _, y_train_multi = create_mock_data(rows=100, features=3, is_classification=True, num_classes=num_classes)
        _, y_valid_multi = create_mock_data(rows=50, features=3, is_classification=True, num_classes=num_classes)

        model, val_preds, test_preds = _train_xgb(
            self.X_train, y_train_multi, self.X_valid, y_valid_multi, self.X_test,
            model_params, fit_params, self.feature_cols, self.cat_features
        )

        self.assertEqual(MockDMatrix.call_count, 3)
        mock_xgb_train.assert_called_once()
        self.assertEqual(mock_booster.predict.call_count, 2)

        self.assertIs(model, mock_booster)
        self.assertIsInstance(val_preds, np.ndarray)
        self.assertEqual(val_preds.shape, (50, num_classes))
        self.assertIsInstance(test_preds, np.ndarray)
        self.assertEqual(test_preds.shape, (30, num_classes))

    @patch('xgboost.train')
    @patch('xgboost.DMatrix')
    def test_no_test_data(self, MockDMatrix, mock_xgb_train):
        """Test case where X_test is None."""
        mock_booster = MagicMock(spec=xgb.Booster)
        mock_booster.best_iteration = 4
        mock_xgb_train.return_value = mock_booster

        mock_dtrain, mock_dvalid = MagicMock(), MagicMock()
        # DMatrix constructor called only twice
        MockDMatrix.side_effect = [mock_dtrain, mock_dvalid]

        mock_booster.predict.return_value = np.random.rand(50)

        model_params = {**self.base_model_params, 'objective': 'reg:squarederror'}
        fit_params = self.base_fit_params

        model, val_preds, test_preds = _train_xgb(
            self.X_train, self.y_train, self.X_valid, self.y_valid, None, # X_test is None
            model_params, fit_params, self.feature_cols, self.cat_features
        )

        # Check DMatrix called only for train and valid
        self.assertEqual(MockDMatrix.call_count, 2)
        # Explicitly check arguments of the calls
        dmatrix_calls = MockDMatrix.call_args_list
        # Check train call
        train_call_args, train_call_kwargs = dmatrix_calls[0]
        pd.testing.assert_frame_equal(train_call_args[0], self.X_train[self.feature_cols])
        np.testing.assert_array_equal(train_call_kwargs['label'], self.y_train)
        self.assertEqual(train_call_kwargs['feature_names'], self.feature_cols)
        self.assertEqual(train_call_kwargs['enable_categorical'], False)
        # Check valid call
        valid_call_args, valid_call_kwargs = dmatrix_calls[1]
        pd.testing.assert_frame_equal(valid_call_args[0], self.X_valid[self.feature_cols])
        np.testing.assert_array_equal(valid_call_kwargs['label'], self.y_valid)
        self.assertEqual(valid_call_kwargs['feature_names'], self.feature_cols)
        self.assertEqual(valid_call_kwargs['enable_categorical'], False)
        # Ensure test call wasn't made (redundant given call_count check, but safe)
        self.assertFalse(any(call[0][0].equals(self.X_test[self.feature_cols]) for call in dmatrix_calls))

        mock_xgb_train.assert_called_once()
        # Check predict called only once with validation DMatrix
        mock_booster.predict.assert_called_once_with(mock_dvalid, iteration_range=(0, mock_booster.best_iteration))

        self.assertIsNone(test_preds)

    @patch('xgboost.train')
    @patch('xgboost.DMatrix')
    def test_enable_categorical(self, MockDMatrix, mock_xgb_train):
        """Test passing enable_categorical=True via model_params."""
        mock_booster = MagicMock(spec=xgb.Booster)
        mock_booster.best_iteration = 4
        mock_xgb_train.return_value = mock_booster

        mock_dtrain, mock_dvalid, mock_dtest = MagicMock(), MagicMock(), MagicMock()
        MockDMatrix.side_effect = [mock_dtrain, mock_dvalid, mock_dtest]
        mock_booster.predict.side_effect = [np.random.rand(50), np.random.rand(30)]

        # Set enable_categorical in model_params
        model_params = {**self.base_model_params, 'objective': 'reg:squarederror', 'enable_categorical': True}
        fit_params = self.base_fit_params

        _train_xgb(
            self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test,
            model_params, fit_params, self.feature_cols, self.cat_features
        )

        # Check DMatrix constructor calls include enable_categorical=True
        self.assertEqual(MockDMatrix.call_count, 3)
        MockDMatrix.assert_any_call(ANY, label=ANY, feature_names=ANY, enable_categorical=True)
        MockDMatrix.assert_any_call(ANY, label=ANY, feature_names=ANY, enable_categorical=True)
        MockDMatrix.assert_any_call(ANY, feature_names=ANY, enable_categorical=True)

if __name__ == '__main__':
    unittest.main() 