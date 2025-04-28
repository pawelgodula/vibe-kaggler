# Tests for the train_stacking_meta_model utility function.

import unittest
import polars as pl
import numpy as np
from polars.testing import assert_series_equal
from sklearn.linear_model import Ridge

# Adjust import path as necessary
try:
    from vibe_kaggler.utils.train_stacking_meta_model import (
        train_stacking_meta_model
    )
    # We also need the internal trainers it uses
    from vibe_kaggler.utils._train_sklearn import _train_sklearn
    from vibe_kaggler.utils._train_lgbm import _train_lgbm # If testing lgbm meta
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from utils.train_stacking_meta_model import (
        train_stacking_meta_model
    )
    from utils._train_sklearn import _train_sklearn
    # from utils._train_lgbm import _train_lgbm

class TestStackingMetaModel(unittest.TestCase):

    def setUp(self):
        # Mock OOF predictions (features for meta-model)
        self.oof_preds = pl.DataFrame({
            'model1_preds': [0.1, 0.8, 0.3, 0.6, 0.9],
            'model2_preds': [0.2, 0.7, 0.4, 0.5, 0.8],
            'model3_preds': [0.15, 0.75, 0.35, 0.55, 0.85]
        })
        # Corresponding true target values
        self.y_true = pl.Series("target", [0, 1, 0, 1, 1])
        
        # Mock test predictions (same columns as OOF)
        self.test_preds_df = pl.DataFrame({
            'model1_preds': [0.2, 0.7],
            'model2_preds': [0.3, 0.6],
            'model3_preds': [0.25, 0.65]
        })

    def test_linear_meta_model(self):
        """Test training a linear (Ridge) meta-model."""
        meta_model_type = 'ridge' # Or 'linear'
        model_params = {'alpha': 1.0, 'random_state': 42}
        fit_params = {}

        final_test_preds, fitted_model = train_stacking_meta_model(
            self.oof_preds,
            self.y_true,
            self.test_preds_df,
            meta_model_type,
            model_params,
            fit_params
        )

        self.assertIsInstance(fitted_model, Ridge)
        self.assertIsInstance(final_test_preds, np.ndarray)
        self.assertEqual(final_test_preds.shape, (self.test_preds_df.height,))
        # Check coefficients were fitted (should not be all zero unless alpha is huge)
        self.assertTrue(np.any(fitted_model.coef_ != 0))

    # Add tests for other meta_model_types like 'lgbm' if needed, 
    # might require mocking or installing lgbm/xgb
    # def test_lgbm_meta_model(self):
    #     # ... setup ...
    #     # Check if lightgbm is installed before running
    #     try:
    #         import lightgbm
    #     except ImportError:
    #         self.skipTest("lightgbm not installed, skipping LGBM meta-model test")
    #     # ... rest of test ...

    def test_error_mismatched_columns(self):
        """Test error if OOF and test prediction columns differ."""
        test_preds_bad = self.test_preds_df.rename({'model1_preds': 'model_A_preds'})
        with self.assertRaisesRegex(ValueError, "Columns in oof_predictions must match"):
            train_stacking_meta_model(
                self.oof_preds, self.y_true, test_preds_bad, 'ridge', {}, {}
            )
            
    def test_error_mismatched_rows(self):
        """Test error if OOF preds and y_true rows differ."""
        y_true_bad = self.y_true.slice(0, 3) # Shorter than oof_preds
        with self.assertRaisesRegex(ValueError, "Number of rows in oof_predictions must match"):
            train_stacking_meta_model(
                self.oof_preds, y_true_bad, self.test_preds_df, 'ridge', {}, {}
            )
            
    def test_error_unsupported_type(self):
        """Test error for an unsupported meta-model type."""
        with self.assertRaisesRegex(ValueError, "Unsupported meta_model_type"):
            train_stacking_meta_model(
                self.oof_preds, self.y_true, self.test_preds_df, 'unknown_model', {}, {}
            )

if __name__ == '__main__':
    unittest.main() 