# Test suite for the train_and_cv orchestrator function.

import unittest
import numpy as np
import polars as pl
import pandas as pd # For comparison in assertions if needed
from typing import Dict, Any, List, Tuple, Optional
from unittest.mock import patch, MagicMock, call

# Function to test
from utils.train_and_cv import train_and_cv

# Mock the underlying train_single_fold function
# Patch it where it's imported in the train_and_cv module
@patch('utils.train_and_cv.train_single_fold')
class TestTrainAndCV(unittest.TestCase):

    def setUp(self):
        """Set up mock Polars DataFrames, CV indices, and parameters."""
        self.feature_cols = ['feat_0', 'feat_1']
        self.target_col = 'target'
        self.cat_features: Optional[List[str]] = None
        self.n_rows_train = 150
        self.n_rows_test = 60
        self.n_splits = 3

        # Create full Polars DataFrames
        self.train_pl = pl.DataFrame({
            'feat_0': np.arange(self.n_rows_train),
            'feat_1': np.arange(self.n_rows_train) * 2,
            'target': np.random.randint(0, 2, self.n_rows_train)
        })
        self.test_pl = pl.DataFrame({
            'feat_0': np.arange(self.n_rows_test) + 1000,
            'feat_1': (np.arange(self.n_rows_test) + 1000) * 2
        })

        # Create mock CV indices (simple sequential splits for testing)
        fold_size = self.n_rows_train // self.n_splits
        self.cv_indices: List[Tuple[np.ndarray, np.ndarray]] = []
        all_indices = np.arange(self.n_rows_train)
        for i in range(self.n_splits):
            start, end = i * fold_size, (i + 1) * fold_size
            valid_idx = np.arange(start, end)
            train_idx = np.setdiff1d(all_indices, valid_idx, assume_unique=True)
            self.cv_indices.append((train_idx, valid_idx))

        self.model_params: Dict[str, Any] = {'model_p': 'a'}
        self.fit_params: Dict[str, Any] = {'fit_p': 'b'}

        # Mock results from train_single_fold for each fold
        self.mock_models = [MagicMock(name=f'model_fold_{i}') for i in range(self.n_splits)]
        # OOF preds: Need shape (n_valid_rows_in_fold,) or (n_valid_rows_in_fold, n_classes)
        self.mock_oof_preds_per_fold = [
            np.ones(len(valid_idx)) * (i + 1) # Fold 1 -> 1s, Fold 2 -> 2s, etc.
            for i, (_, valid_idx) in enumerate(self.cv_indices)
        ]
        # Test preds: Need shape (n_test_rows,) or (n_test_rows, n_classes)
        self.mock_test_preds_per_fold = [
            np.ones(len(self.test_pl)) * 10 * (i + 1)
            for i in range(self.n_splits)
        ]

    def test_cv_loop_and_aggregation(self, mock_train_single_fold):
        """Test the main CV loop, data slicing, and prediction aggregation."""
        # Configure the mock train_single_fold to return specific values per call
        mock_train_single_fold.side_effect = [
            (self.mock_models[i], self.mock_oof_preds_per_fold[i], self.mock_test_preds_per_fold[i])
            for i in range(self.n_splits)
        ]

        oof_preds, test_preds_avg, models = train_and_cv(
            train_df=self.train_pl,
            test_df=self.test_pl,
            target_col=self.target_col,
            feature_cols=self.feature_cols,
            model_type='lgbm',
            cv_indices=self.cv_indices,
            model_params=self.model_params,
            fit_params=self.fit_params,
            cat_features=self.cat_features,
            verbose=False # Suppress print statements during test
        )

        # Check train_single_fold was called n_splits times
        self.assertEqual(mock_train_single_fold.call_count, self.n_splits)

        # Verify calls to train_single_fold with correct sliced data
        # Instead of assert_has_calls, check each call individually
        actual_calls = mock_train_single_fold.call_args_list
        self.assertEqual(len(actual_calls), self.n_splits)

        for i, (train_idx, valid_idx) in enumerate(self.cv_indices):
            train_fold_df_expected = self.train_pl[train_idx]
            valid_fold_df_expected = self.train_pl[valid_idx]
            current_call_kwargs = actual_calls[i].kwargs

            # Compare DataFrames using Polars equality or testing utility
            self.assertTrue(current_call_kwargs['train_fold_df'].equals(train_fold_df_expected))
            self.assertTrue(current_call_kwargs['valid_fold_df'].equals(valid_fold_df_expected))
            if self.test_pl is not None:
                 self.assertTrue(current_call_kwargs['test_df'].equals(self.test_pl))
            else:
                 self.assertIsNone(current_call_kwargs['test_df'])

            # Compare other arguments
            self.assertEqual(current_call_kwargs['target_col'], self.target_col)
            self.assertEqual(current_call_kwargs['feature_cols'], self.feature_cols)
            self.assertEqual(current_call_kwargs['model_type'], 'lgbm') # Hardcoded in this test
            self.assertEqual(current_call_kwargs['model_params'], self.model_params)
            self.assertEqual(current_call_kwargs['fit_params'], self.fit_params)
            self.assertEqual(current_call_kwargs['cat_features'], self.cat_features)

        # Check OOF predictions aggregation
        self.assertIsInstance(oof_preds, np.ndarray)
        self.assertEqual(oof_preds.shape, (self.n_rows_train,))
        # Verify OOF values were placed correctly
        expected_oof = np.full(self.n_rows_train, np.nan)
        for i, (_, valid_idx) in enumerate(self.cv_indices):
            expected_oof[valid_idx] = self.mock_oof_preds_per_fold[i]
        np.testing.assert_array_equal(oof_preds, expected_oof)

        # Check test predictions aggregation
        self.assertIsInstance(test_preds_avg, np.ndarray)
        self.assertEqual(test_preds_avg.shape, (self.n_rows_test,))
        # Verify test predictions are averaged correctly
        expected_test_avg = np.mean(np.stack(self.mock_test_preds_per_fold, axis=0), axis=0)
        np.testing.assert_allclose(test_preds_avg, expected_test_avg)

        # Check returned models list
        self.assertEqual(models, self.mock_models)

    def test_no_test_data(self, mock_train_single_fold):
        """Test behavior when test_df is None."""
        # Configure mock to return None for test predictions
        mock_train_single_fold.side_effect = [
            (self.mock_models[i], self.mock_oof_preds_per_fold[i], None)
            for i in range(self.n_splits)
        ]

        oof_preds, test_preds_avg, models = train_and_cv(
            train_df=self.train_pl,
            test_df=None, # Pass None for test_df
            target_col=self.target_col,
            feature_cols=self.feature_cols,
            model_type='xgb',
            cv_indices=self.cv_indices,
            model_params=self.model_params,
            fit_params=self.fit_params,
            verbose=False
        )

        self.assertEqual(mock_train_single_fold.call_count, self.n_splits)
        # Check test_df passed to mock was None in all calls
        for mock_call in mock_train_single_fold.call_args_list:
            self.assertIsNone(mock_call.kwargs['test_df'])

        # Check OOF preds are still generated
        self.assertIsInstance(oof_preds, np.ndarray)
        self.assertEqual(oof_preds.shape, (self.n_rows_train,))
        # Check test_preds_avg is None
        self.assertIsNone(test_preds_avg)
        # Check models are returned
        self.assertEqual(models, self.mock_models)

    def test_multiclass_aggregation(self, mock_train_single_fold):
        """Test aggregation for multiclass predictions."""
        n_classes = 3
        # Mock OOF preds: (n_valid_rows, n_classes)
        mock_oof_multi = [
            np.random.rand(len(valid_idx), n_classes)
            for _, valid_idx in self.cv_indices
        ]
        # Mock Test preds: (n_test_rows, n_classes)
        mock_test_multi = [
            np.random.rand(self.n_rows_test, n_classes) * (i+1)
            for i in range(self.n_splits)
        ]

        mock_train_single_fold.side_effect = [
            (self.mock_models[i], mock_oof_multi[i], mock_test_multi[i])
            for i in range(self.n_splits)
        ]

        oof_preds, test_preds_avg, models = train_and_cv(
            train_df=self.train_pl,
            test_df=self.test_pl,
            target_col=self.target_col,
            feature_cols=self.feature_cols,
            model_type='lgbm',
            cv_indices=self.cv_indices,
            model_params=self.model_params,
            fit_params=self.fit_params,
            verbose=False
        )

        # Check OOF shape
        self.assertEqual(oof_preds.shape, (self.n_rows_train, n_classes))
        # Verify OOF values
        expected_oof = np.full((self.n_rows_train, n_classes), np.nan)
        for i, (_, valid_idx) in enumerate(self.cv_indices):
            expected_oof[valid_idx, :] = mock_oof_multi[i]
        np.testing.assert_allclose(oof_preds, expected_oof)

        # Check test preds shape and aggregation
        self.assertEqual(test_preds_avg.shape, (self.n_rows_test, n_classes))
        expected_test_avg = np.mean(np.stack(mock_test_multi, axis=0), axis=0)
        np.testing.assert_allclose(test_preds_avg, expected_test_avg)

    def test_empty_cv_indices(self, mock_train_single_fold):
        """Test ValueError if cv_indices list is empty."""
        with self.assertRaisesRegex(ValueError, "cv_indices list cannot be empty."):
            train_and_cv(
                train_df=self.train_pl,
                test_df=self.test_pl,
                target_col=self.target_col,
                feature_cols=self.feature_cols,
                model_type='lgbm',
                cv_indices=[], # Empty list
                model_params=self.model_params,
                fit_params=self.fit_params
            )
        mock_train_single_fold.assert_not_called()

    def test_pass_cat_features(self, mock_train_single_fold):
        """Test passing cat_features through to train_single_fold."""
        cat_features = ['feat_0']
        # Mock return needs correct oof shape, even if empty/dummy
        mock_oof_return = np.full(len(self.cv_indices[0][1]), np.nan) # Shape based on first fold's valid_idx
        mock_train_single_fold.return_value = (MagicMock(), mock_oof_return, None)

        train_and_cv(
            train_df=self.train_pl,
            test_df=None,
            target_col=self.target_col,
            feature_cols=self.feature_cols,
            model_type='lgbm',
            cv_indices=self.cv_indices,
            model_params=self.model_params,
            fit_params=self.fit_params,
            cat_features=cat_features, # Pass cat features
            verbose=False
        )

        self.assertEqual(mock_train_single_fold.call_count, self.n_splits)
        # Check cat_features in the keyword arguments of each call
        for mock_call in mock_train_single_fold.call_args_list:
            self.assertEqual(mock_call.kwargs['cat_features'], cat_features)


if __name__ == '__main__':
    unittest.main() 