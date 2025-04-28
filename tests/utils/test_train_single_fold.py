# Test suite for the train_single_fold dispatcher function.

import unittest
import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock, ANY

# Function to test
from utils.train_single_fold import train_single_fold, MODEL_TRAINERS

# Mock underlying trainers (_train_lgbm, _train_xgb)
# We need to patch them where they are looked up (within the train_single_fold module)
@patch('utils.train_single_fold._train_lgbm')
@patch('utils.train_single_fold._train_xgb')
class TestTrainSingleFold(unittest.TestCase):

    def setUp(self):
        """Set up mock Polars DataFrames and parameters."""
        self.feature_cols = ['feat_0', 'feat_1']
        self.target_col = 'target'
        self.cat_features: Optional[List[str]] = None

        # Create Polars DataFrames
        self.train_fold_pl = pl.DataFrame({
            'feat_0': np.random.rand(100),
            'feat_1': np.random.rand(100),
            'target': np.random.randint(0, 2, 100)
        })
        self.valid_fold_pl = pl.DataFrame({
            'feat_0': np.random.rand(50),
            'feat_1': np.random.rand(50),
            'target': np.random.randint(0, 2, 50)
        })
        self.test_pl = pl.DataFrame({
            'feat_0': np.random.rand(30),
            'feat_1': np.random.rand(30)
            # No target column in test
        })

        self.model_params: Dict[str, Any] = {'param1': 'value1'}
        self.fit_params: Dict[str, Any] = {'fit_param1': 10}

        # Expected Pandas conversions
        self.X_train_pd = self.train_fold_pl.select(self.feature_cols).to_pandas()
        self.y_train_np = self.train_fold_pl[self.target_col].to_numpy()
        self.X_valid_pd = self.valid_fold_pl.select(self.feature_cols).to_pandas()
        self.y_valid_np = self.valid_fold_pl[self.target_col].to_numpy()
        self.X_test_pd = self.test_pl.select(self.feature_cols).to_pandas()

    def test_dispatch_to_lgbm(self, mock_train_xgb, mock_train_lgbm):
        """Test if train_single_fold correctly calls _train_lgbm."""
        # Temporarily replace dict value with the mock passed by @patch
        original_trainers = MODEL_TRAINERS.copy()
        MODEL_TRAINERS['lgbm'] = mock_train_lgbm
        MODEL_TRAINERS['lightgbm'] = mock_train_lgbm # Also patch alias

        mock_model = MagicMock()
        mock_val_preds = np.random.rand(len(self.valid_fold_pl))
        mock_test_preds = np.random.rand(len(self.test_pl))
        mock_train_lgbm.return_value = (mock_model, mock_val_preds, mock_test_preds)

        try:
            model, val_preds, test_preds = train_single_fold(
                train_fold_df=self.train_fold_pl,
                valid_fold_df=self.valid_fold_pl,
                test_df=self.test_pl,
                target_col=self.target_col,
                feature_cols=self.feature_cols,
                model_type='lgbm', # Test lowercase
                model_params=self.model_params,
                fit_params=self.fit_params,
                cat_features=self.cat_features
            )

            # Check _train_lgbm was called correctly
            mock_train_lgbm.assert_called_once()
            call_args, call_kwargs = mock_train_lgbm.call_args
            pd.testing.assert_frame_equal(call_kwargs['X_train'], self.X_train_pd)
            np.testing.assert_array_equal(call_kwargs['y_train'], self.y_train_np)
            pd.testing.assert_frame_equal(call_kwargs['X_valid'], self.X_valid_pd)
            np.testing.assert_array_equal(call_kwargs['y_valid'], self.y_valid_np)
            pd.testing.assert_frame_equal(call_kwargs['X_test'], self.X_test_pd)
            self.assertEqual(call_kwargs['model_params'], self.model_params)
            self.assertEqual(call_kwargs['fit_params'], self.fit_params)
            self.assertEqual(call_kwargs['feature_cols'], self.feature_cols)
            self.assertEqual(call_kwargs['cat_features'], self.cat_features)

            # Check _train_xgb was not called
            mock_train_xgb.assert_not_called()

            # Check return values match the mock
            self.assertIs(model, mock_model)
            np.testing.assert_array_equal(val_preds, mock_val_preds)
            np.testing.assert_array_equal(test_preds, mock_test_preds)
        finally:
            # Restore original dictionary
            MODEL_TRAINERS.clear()
            MODEL_TRAINERS.update(original_trainers)

    def test_dispatch_to_xgb(self, mock_train_xgb, mock_train_lgbm):
        """Test if train_single_fold correctly calls _train_xgb."""
        # Temporarily replace dict value with the mock passed by @patch
        original_trainers = MODEL_TRAINERS.copy()
        MODEL_TRAINERS['xgb'] = mock_train_xgb
        MODEL_TRAINERS['xgboost'] = mock_train_xgb # Also patch alias

        mock_model = MagicMock()
        mock_val_preds = np.random.rand(len(self.valid_fold_pl))
        mock_test_preds = np.random.rand(len(self.test_pl))
        mock_train_xgb.return_value = (mock_model, mock_val_preds, mock_test_preds)

        try:
            model, val_preds, test_preds = train_single_fold(
                train_fold_df=self.train_fold_pl,
                valid_fold_df=self.valid_fold_pl,
                test_df=self.test_pl,
                target_col=self.target_col,
                feature_cols=self.feature_cols,
                model_type='XGBoost', # Test mixed case
                model_params=self.model_params,
                fit_params=self.fit_params,
                cat_features=self.cat_features
            )

            # Check _train_xgb was called correctly
            mock_train_xgb.assert_called_once()
            call_args, call_kwargs = mock_train_xgb.call_args
            pd.testing.assert_frame_equal(call_kwargs['X_train'], self.X_train_pd)
            np.testing.assert_array_equal(call_kwargs['y_train'], self.y_train_np)
            pd.testing.assert_frame_equal(call_kwargs['X_valid'], self.X_valid_pd)
            np.testing.assert_array_equal(call_kwargs['y_valid'], self.y_valid_np)
            pd.testing.assert_frame_equal(call_kwargs['X_test'], self.X_test_pd)
            self.assertEqual(call_kwargs['model_params'], self.model_params)
            self.assertEqual(call_kwargs['fit_params'], self.fit_params)
            self.assertEqual(call_kwargs['feature_cols'], self.feature_cols)
            self.assertEqual(call_kwargs['cat_features'], self.cat_features)

            # Check _train_lgbm was not called
            mock_train_lgbm.assert_not_called()

            # Check return values match the mock
            self.assertIs(model, mock_model)
            np.testing.assert_array_equal(val_preds, mock_val_preds)
            np.testing.assert_array_equal(test_preds, mock_test_preds)
        finally:
            # Restore original dictionary
            MODEL_TRAINERS.clear()
            MODEL_TRAINERS.update(original_trainers)

    def test_unsupported_model_type(self, mock_train_xgb, mock_train_lgbm):
        """Test ValueError for an unknown model_type."""
        with self.assertRaisesRegex(ValueError, "Unsupported model_type: 'catboost'"):
            train_single_fold(
                train_fold_df=self.train_fold_pl,
                valid_fold_df=self.valid_fold_pl,
                test_df=self.test_pl,
                target_col=self.target_col,
                feature_cols=self.feature_cols,
                model_type='catboost',
                model_params=self.model_params,
                fit_params=self.fit_params,
            )
        mock_train_lgbm.assert_not_called()
        mock_train_xgb.assert_not_called()

    def test_missing_feature_column_train(self, mock_train_xgb, mock_train_lgbm):
        """Test ColumnNotFoundError if a feature is missing in train_fold_df."""
        with self.assertRaises(pl.exceptions.ColumnNotFoundError):
            train_single_fold(
                train_fold_df=self.train_fold_pl.drop('feat_1'), # Drop a feature
                valid_fold_df=self.valid_fold_pl,
                test_df=self.test_pl,
                target_col=self.target_col,
                feature_cols=self.feature_cols,
                model_type='lgbm',
                model_params=self.model_params,
                fit_params=self.fit_params,
            )

    def test_missing_target_column_valid(self, mock_train_xgb, mock_train_lgbm):
        """Test ColumnNotFoundError if target is missing in valid_fold_df."""
        with self.assertRaises(pl.exceptions.ColumnNotFoundError):
            train_single_fold(
                train_fold_df=self.train_fold_pl,
                valid_fold_df=self.valid_fold_pl.drop(self.target_col), # Drop target
                test_df=self.test_pl,
                target_col=self.target_col,
                feature_cols=self.feature_cols,
                model_type='lgbm',
                model_params=self.model_params,
                fit_params=self.fit_params,
            )

    def test_missing_feature_column_test(self, mock_train_xgb, mock_train_lgbm):
        """Test ColumnNotFoundError if a feature is missing in test_df."""
        with self.assertRaises(pl.exceptions.ColumnNotFoundError):
            train_single_fold(
                train_fold_df=self.train_fold_pl,
                valid_fold_df=self.valid_fold_pl,
                test_df=self.test_pl.drop('feat_0'), # Drop feature from test
                target_col=self.target_col,
                feature_cols=self.feature_cols,
                model_type='lgbm',
                model_params=self.model_params,
                fit_params=self.fit_params,
            )

    def test_no_test_data(self, mock_train_xgb, mock_train_lgbm):
        """Test handling when test_df is None."""
        # Temporarily replace dict value with the mock passed by @patch
        original_trainers = MODEL_TRAINERS.copy()
        MODEL_TRAINERS['lgbm'] = mock_train_lgbm
        MODEL_TRAINERS['lightgbm'] = mock_train_lgbm

        mock_model = MagicMock()
        mock_val_preds = np.random.rand(len(self.valid_fold_pl))
        # Expect None for test_preds from the underlying trainer
        mock_train_lgbm.return_value = (mock_model, mock_val_preds, None)

        try:
            model, val_preds, test_preds = train_single_fold(
                train_fold_df=self.train_fold_pl,
                valid_fold_df=self.valid_fold_pl,
                test_df=None, # Pass None for test_df
                target_col=self.target_col,
                feature_cols=self.feature_cols,
                model_type='lgbm',
                model_params=self.model_params,
                fit_params=self.fit_params,
            )

            mock_train_lgbm.assert_called_once()
            call_args, call_kwargs = mock_train_lgbm.call_args
            # Check that X_test passed to the trainer is None
            self.assertIsNone(call_kwargs['X_test'])

            # Check return values
            self.assertIs(model, mock_model)
            np.testing.assert_array_equal(val_preds, mock_val_preds)
            self.assertIsNone(test_preds) # test_preds should be None
        finally:
            # Restore original dictionary
            MODEL_TRAINERS.clear()
            MODEL_TRAINERS.update(original_trainers)

    def test_pass_cat_features(self, mock_train_xgb, mock_train_lgbm):
        """Test passing cat_features through to the underlying trainer."""
        # Temporarily replace dict value with the mock passed by @patch
        original_trainers = MODEL_TRAINERS.copy()
        MODEL_TRAINERS['lgbm'] = mock_train_lgbm
        MODEL_TRAINERS['lightgbm'] = mock_train_lgbm

        cat_features = ['feat_0']
        mock_train_lgbm.return_value = (MagicMock(), np.array([]), None)

        try:
            train_single_fold(
                train_fold_df=self.train_fold_pl,
                valid_fold_df=self.valid_fold_pl,
                test_df=None,
                target_col=self.target_col,
                feature_cols=self.feature_cols,
                model_type='lgbm',
                model_params=self.model_params,
                fit_params=self.fit_params,
                cat_features=cat_features # Pass specific cat features
            )

            mock_train_lgbm.assert_called_once()
            call_args, call_kwargs = mock_train_lgbm.call_args
            self.assertEqual(call_kwargs['cat_features'], cat_features)
        finally:
            # Restore original dictionary
            MODEL_TRAINERS.clear()
            MODEL_TRAINERS.update(original_trainers)


if __name__ == '__main__':
    unittest.main() 