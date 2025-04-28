# Tests for the _train_sklearn internal utility function.

import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.exceptions import NotFittedError

# Adjust import path as necessary
try:
    from vibe_kaggler.utils._train_sklearn import _train_sklearn
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from utils._train_sklearn import _train_sklearn

class TestTrainSklearn(unittest.TestCase):

    def setUp(self):
        # Simple reproducible data
        self.X_train_cls = np.array([[1, 2], [1, 3], [2, 1], [3, 1], [3, 2]])
        self.y_train_cls = np.array([0, 0, 1, 1, 1])
        self.X_valid_cls = np.array([[1, 1], [2, 3]])
        self.y_valid_cls = np.array([0, 1]) # Not used in basic fit, but good to have
        self.X_test_cls = np.array([[3, 3], [1, 4]])
        
        self.X_train_reg = np.array([[1], [2], [3], [4], [5]])
        self.y_train_reg = np.array([2.1, 3.9, 6.1, 8.0, 9.9])
        self.X_valid_reg = np.array([[1.5], [4.5]])
        self.y_valid_reg = np.array([3.0, 9.0]) # Not used
        self.X_test_reg = np.array([[0], [6]])

    def test_train_classifier_rf(self):
        """Test training a RandomForestClassifier."""
        model_class = RandomForestClassifier
        model_params = {'n_estimators': 5, 'random_state': 42}
        fit_params = {}

        model, y_pred_valid, y_pred_test = _train_sklearn(
            self.X_train_cls,
            self.y_train_cls,
            self.X_valid_cls,
            self.y_valid_cls,
            self.X_test_cls,
            model_class,
            model_params,
            fit_params
        )

        self.assertIsInstance(model, RandomForestClassifier)
        self.assertEqual(model.n_estimators, 5)
        # Check if model is fitted (predict should work)
        try:
             model.predict(self.X_train_cls)
        except NotFittedError:
             self.fail("_train_sklearn did not fit the model")

        self.assertIsInstance(y_pred_valid, np.ndarray)
        self.assertEqual(y_pred_valid.shape, (self.X_valid_cls.shape[0],)) # predict_proba returns P(class=1)
        self.assertTrue(np.all(y_pred_valid >= 0) and np.all(y_pred_valid <= 1))
        
        self.assertIsInstance(y_pred_test, np.ndarray)
        self.assertEqual(y_pred_test.shape, (self.X_test_cls.shape[0],))
        self.assertTrue(np.all(y_pred_test >= 0) and np.all(y_pred_test <= 1))

    def test_train_regressor_ridge(self):
        """Test training a Ridge regressor."""
        model_class = Ridge
        model_params = {'alpha': 1.0, 'random_state': 42}
        fit_params = {}

        model, y_pred_valid, y_pred_test = _train_sklearn(
            self.X_train_reg,
            self.y_train_reg,
            self.X_valid_reg,
            self.y_valid_reg,
            self.X_test_reg,
            model_class,
            model_params,
            fit_params
        )

        self.assertIsInstance(model, Ridge)
        # Check if model is fitted (predict should work)
        try:
             model.predict(self.X_train_reg)
        except NotFittedError:
             self.fail("_train_sklearn did not fit the model")

        self.assertIsInstance(y_pred_valid, np.ndarray)
        self.assertEqual(y_pred_valid.shape, (self.X_valid_reg.shape[0],))
        
        self.assertIsInstance(y_pred_test, np.ndarray)
        self.assertEqual(y_pred_test.shape, (self.X_test_reg.shape[0],))

    def test_no_test_data(self):
        """Test training without test data."""
        model_class = LogisticRegression
        model_params = {'random_state': 42}
        fit_params = {}

        model, y_pred_valid, y_pred_test = _train_sklearn(
            self.X_train_cls,
            self.y_train_cls,
            self.X_valid_cls,
            self.y_valid_cls,
            None, # No test data
            model_class,
            model_params,
            fit_params
        )

        self.assertIsInstance(model, LogisticRegression)
        self.assertIsInstance(y_pred_valid, np.ndarray)
        self.assertIsNone(y_pred_test) # Expect None for test predictions
        
    def test_no_valid_data(self):
        """Test training without validation data."""
        model_class = Ridge
        model_params = {'alpha': 0.5}
        fit_params = {}

        model, y_pred_valid, y_pred_test = _train_sklearn(
            self.X_train_reg,
            self.y_train_reg,
            None, # No validation data
            None,
            self.X_test_reg,
            model_class,
            model_params,
            fit_params
        )
        self.assertIsInstance(model, Ridge)
        self.assertIsNone(y_pred_valid) # Expect None for validation predictions
        self.assertIsInstance(y_pred_test, np.ndarray)

    def test_invalid_model_params(self):
        """Test error handling for bad model parameters."""
        model_class = RandomForestClassifier
        model_params = {'invalid_param': 123} # Invalid parameter
        fit_params = {}
        with self.assertRaises(ValueError):
            _train_sklearn(
                self.X_train_cls, self.y_train_cls, None, None, None,
                model_class, model_params, fit_params
            )
            
    def test_invalid_fit_params(self):
        """Test error handling for bad fit parameters."""
        # Ridge raises TypeError for unexpected fit keyword args
        model_class = Ridge
        model_params = {}
        fit_params = {'non_existent_fit_param': True}
        # Expect ValueError from our wrapper function, which should catch the underlying TypeError
        with self.assertRaises(ValueError) as cm:
            _train_sklearn(
                self.X_train_reg, self.y_train_reg, None, None, None,
                model_class, model_params, fit_params
            )
        self.assertIn("Failed to fit model Ridge", str(cm.exception))
        self.assertIsInstance(cm.exception.__cause__, TypeError)
        self.assertIn("unexpected keyword argument 'non_existent_fit_param'", str(cm.exception.__cause__))

if __name__ == '__main__':
    unittest.main() 