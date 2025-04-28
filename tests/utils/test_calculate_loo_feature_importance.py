# Tests for calculate_loo_feature_importance function.

import unittest
from unittest.mock import patch, MagicMock
import polars as pl
import numpy as np

# Adjust import path as necessary
try:
    from vibe_kaggler.utils.calculate_loo_feature_importance import calculate_loo_feature_importance
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from utils.calculate_loo_feature_importance import calculate_loo_feature_importance

class TestCalculateLOOFeatureImportance(unittest.TestCase):

    def setUp(self):
        self.train_df = pl.DataFrame({
            'f1': [1, 2, 3, 4, 5, 6],
            'f2': [6, 5, 4, 3, 2, 1],
            'f3': [1, 1, 0, 0, 1, 0],
            'target': [1, 0, 1, 0, 1, 0]
        })
        self.features = ['f1', 'f2', 'f3']
        self.target_col = 'target'
        # Dummy CV indices (e.g., 2 folds)
        self.cv_indices = [
            (np.array([0, 1, 2]), np.array([3, 4, 5])),
            (np.array([3, 4, 5]), np.array([0, 1, 2]))
        ]
        self.model_type = 'lgbm'
        self.model_params = {'objective': 'binary'}
        self.fit_params = {}
        self.metric_name = 'auc'
        self.metric_params = {}

    @patch('utils.calculate_loo_feature_importance._evaluate_features_cv')
    def test_importance_higher_is_better(self, mock_evaluate_cv):
        """Test LOO importance calculation when higher score is better (e.g., AUC)."""
        
        # Define mock return values based on features provided
        def side_effect(*args, **kwargs):
            features = kwargs.get('feature_cols', [])
            if features == ['f1', 'f2', 'f3']: return 0.85 # Baseline
            if features == ['f2', 'f3']: return 0.70 # Without f1
            if features == ['f1', 'f3']: return 0.80 # Without f2
            if features == ['f1', 'f2']: return 0.82 # Without f3
            return 0.5 # Default/error

        mock_evaluate_cv.side_effect = side_effect

        importance = calculate_loo_feature_importance(
            train_df=self.train_df,
            target_col=self.target_col,
            features=self.features,
            cv_indices=self.cv_indices,
            model_type=self.model_type,
            model_params=self.model_params,
            fit_params=self.fit_params,
            metric_name=self.metric_name,
            metric_params=self.metric_params,
            higher_is_better=True
        )

        # Expected: baseline - score_without
        # f1: 0.85 - 0.70 = 0.15
        # f2: 0.85 - 0.80 = 0.05
        # f3: 0.85 - 0.82 = 0.03
        expected_importance = {'f1': 0.15, 'f2': 0.05, 'f3': 0.03}
        
        self.assertEqual(mock_evaluate_cv.call_count, 4) # 1 baseline + 3 features
        self.assertDictAlmostEqual(importance, expected_importance)
        # Check sorting
        self.assertEqual(list(importance.keys()), ['f1', 'f2', 'f3'])

    @patch('utils.calculate_loo_feature_importance._evaluate_features_cv')
    def test_importance_lower_is_better(self, mock_evaluate_cv):
        """Test LOO importance calculation when lower score is better (e.g., RMSE)."""
        
        # Define mock return values
        def side_effect(*args, **kwargs):
            features = kwargs.get('feature_cols', [])
            if features == ['f1', 'f2', 'f3']: return 1.5 # Baseline
            if features == ['f2', 'f3']: return 2.5 # Without f1 (worse)
            if features == ['f1', 'f3']: return 1.8 # Without f2 (worse)
            if features == ['f1', 'f2']: return 1.6 # Without f3 (slightly worse)
            return 99.0 # Default/error

        mock_evaluate_cv.side_effect = side_effect

        importance = calculate_loo_feature_importance(
            train_df=self.train_df,
            target_col=self.target_col,
            features=self.features,
            cv_indices=self.cv_indices,
            model_type=self.model_type,
            model_params=self.model_params,
            fit_params=self.fit_params,
            metric_name='rmse',
            metric_params=self.metric_params,
            higher_is_better=False
        )

        # Expected: score_without - baseline
        # f1: 2.5 - 1.5 = 1.0
        # f2: 1.8 - 1.5 = 0.3
        # f3: 1.6 - 1.5 = 0.1
        expected_importance = {'f1': 1.0, 'f2': 0.3, 'f3': 0.1}
        
        self.assertEqual(mock_evaluate_cv.call_count, 4) # 1 baseline + 3 features
        self.assertDictAlmostEqual(importance, expected_importance)
        # Check sorting
        self.assertEqual(list(importance.keys()), ['f1', 'f2', 'f3'])

    @patch('utils.calculate_loo_feature_importance._evaluate_features_cv')
    def test_no_features(self, mock_evaluate_cv):
        """Test case with an empty feature list."""
        importance = calculate_loo_feature_importance(
            train_df=self.train_df,
            target_col=self.target_col,
            features=[], # Empty list
            cv_indices=self.cv_indices,
            model_type=self.model_type,
            model_params=self.model_params,
            fit_params=self.fit_params,
            metric_name=self.metric_name,
            metric_params=self.metric_params,
            higher_is_better=True
        )
        self.assertEqual(importance, {})
        mock_evaluate_cv.assert_not_called()

    @patch('utils.calculate_loo_feature_importance._evaluate_features_cv')
    def test_single_feature(self, mock_evaluate_cv):
        """Test case with only one feature."""
        mock_evaluate_cv.return_value = 0.7 # Baseline score with the single feature
        
        importance = calculate_loo_feature_importance(
            train_df=self.train_df,
            target_col=self.target_col,
            features=['f1'],
            cv_indices=self.cv_indices,
            model_type=self.model_type,
            model_params=self.model_params,
            fit_params=self.fit_params,
            metric_name=self.metric_name,
            metric_params=self.metric_params,
            higher_is_better=True
        )
        # Expected importance is 0 because removing the only feature leaves no features to evaluate.
        expected_importance = {'f1': 0.0}
        self.assertEqual(importance, expected_importance)
        self.assertEqual(mock_evaluate_cv.call_count, 1) # Only baseline called

    @patch('utils.calculate_loo_feature_importance._evaluate_features_cv')
    def test_evaluation_error(self, mock_evaluate_cv):
        """Test handling of errors during _evaluate_features_cv calls."""
        def side_effect(*args, **kwargs):
            features = kwargs.get('feature_cols', [])
            if features == ['f1', 'f2', 'f3']: return 0.85 # Baseline OK
            if features == ['f2', 'f3']: raise ValueError("Mock evaluation error") # Error removing f1
            if features == ['f1', 'f3']: return 0.80 # Without f2 OK
            if features == ['f1', 'f2']: return 0.82 # Without f3 OK
            return 0.5 

        mock_evaluate_cv.side_effect = side_effect

        importance = calculate_loo_feature_importance(
            train_df=self.train_df,
            target_col=self.target_col,
            features=self.features,
            cv_indices=self.cv_indices,
            model_type=self.model_type,
            model_params=self.model_params,
            fit_params=self.fit_params,
            metric_name=self.metric_name,
            metric_params=self.metric_params,
            higher_is_better=True
        )
        # f1 importance defaults to 0 due to error
        # f2: 0.85 - 0.80 = 0.05
        # f3: 0.85 - 0.82 = 0.03
        expected_importance = {'f2': 0.05, 'f3': 0.03, 'f1': 0.0}
        self.assertDictAlmostEqual(importance, expected_importance)
        self.assertEqual(mock_evaluate_cv.call_count, 4) # Still called 4 times

    def assertDictAlmostEqual(self, dict1, dict2, places=7):
        """Helper to compare dictionaries with float values."""
        self.assertEqual(set(dict1.keys()), set(dict2.keys()))
        for key in dict1:
            self.assertAlmostEqual(dict1[key], dict2[key], places=places,
                                   msg=f"Value mismatch for key '{key}'")

if __name__ == '__main__':
    unittest.main() 