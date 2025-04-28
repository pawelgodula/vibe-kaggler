import unittest
import numpy as np
import torch
from typing import Dict, Any, Optional, List

# Assuming _train_nn is in utils._train_nn
# Adjust the import path if necessary based on your project structure
try:
    from vibe_kaggler.utils._train_nn import _train_nn, SimpleMLP
except ImportError:
    # Fallback if running tests directly from the tests directory
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from utils._train_nn import _train_nn, SimpleMLP

# Basic checks: Can we import torch? Skip if not.
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Define dummy classes if torch not available, so tests can be discovered but skipped
    class SimpleMLP: pass 
    def _train_nn(*args, **kwargs): pass

@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not installed, skipping NN tests.")
class TestTrainNN(unittest.TestCase):

    def setUp(self):
        # Simple reproducible data
        np.random.seed(42)
        torch.manual_seed(42)
        self.feature_cols = [f'feat_{i}' for i in range(5)]
        self.X_train_reg = np.random.rand(50, 5).astype(np.float32)
        self.y_train_reg = (self.X_train_reg @ np.array([1, -2, 3, -1.5, 0.5])).astype(np.float32)
        self.X_valid_reg = np.random.rand(20, 5).astype(np.float32)
        self.y_valid_reg = (self.X_valid_reg @ np.array([1, -2, 3, -1.5, 0.5])).astype(np.float32)
        self.X_test_reg = np.random.rand(10, 5).astype(np.float32)
        
        self.X_train_cls = np.random.rand(60, 5).astype(np.float32)
        self.y_train_cls = (self.X_train_cls[:, 0] + self.X_train_cls[:, 1] > 1.0).astype(np.float32)
        self.X_valid_cls = np.random.rand(30, 5).astype(np.float32)
        self.y_valid_cls = (self.X_valid_cls[:, 0] + self.X_valid_cls[:, 1] > 1.0).astype(np.float32)
        self.X_test_cls = np.random.rand(15, 5).astype(np.float32)
        
    def test_regression_task(self):
        """Test training MLP for a regression task."""
        model_params = {
            'hidden_sizes': [16, 8],
            'output_size': 1,
            'dropout_rate': 0.1
        }
        fit_params = {
            'epochs': 3, # Keep epochs low for testing
            'batch_size': 16,
            'lr': 1e-2,
            'optimizer': 'adam',
            'loss_fn': 'mse',
            'early_stopping_rounds': 0, # Disable for simple test
            'verbose': 0 # Suppress print output
        }
        
        state_dict, y_pred_valid, y_pred_test = _train_nn(
            self.X_train_reg, self.y_train_reg,
            self.X_valid_reg, self.y_valid_reg,
            self.X_test_reg,
            model_params, fit_params,
            self.feature_cols
        )
        
        self.assertIsInstance(state_dict, dict)
        self.assertIn('network.0.weight', state_dict) # Check a layer exists
        self.assertIsInstance(y_pred_valid, np.ndarray)
        self.assertEqual(y_pred_valid.shape, (self.X_valid_reg.shape[0],))
        self.assertIsInstance(y_pred_test, np.ndarray)
        self.assertEqual(y_pred_test.shape, (self.X_test_reg.shape[0],))
        
    def test_binary_classification_task(self):
        """Test training MLP for a binary classification task."""
        model_params = {
            'hidden_sizes': [20],
            'output_size': 1,
            'dropout_rate': 0.0
        }
        fit_params = {
            'epochs': 4,
            'batch_size': 32,
            'lr': 5e-3,
            'optimizer': 'sgd',
            'loss_fn': 'bce',
            'early_stopping_rounds': 2,
            'verbose': 0
        }
        
        state_dict, y_pred_valid, y_pred_test = _train_nn(
            self.X_train_cls, self.y_train_cls,
            self.X_valid_cls, self.y_valid_cls,
            self.X_test_cls,
            model_params, fit_params,
            self.feature_cols
        )
        
        self.assertIsInstance(state_dict, dict)
        self.assertIsInstance(y_pred_valid, np.ndarray)
        self.assertEqual(y_pred_valid.shape, (self.X_valid_cls.shape[0],))
        self.assertTrue(np.all(y_pred_valid >= 0) and np.all(y_pred_valid <= 1)) # Probabilities
        self.assertIsInstance(y_pred_test, np.ndarray)
        self.assertEqual(y_pred_test.shape, (self.X_test_cls.shape[0],))
        self.assertTrue(np.all(y_pred_test >= 0) and np.all(y_pred_test <= 1))

    def test_no_test_data(self):
        """Test training without test data."""
        model_params = {'hidden_sizes': [8], 'output_size': 1}
        fit_params = {'epochs': 1, 'loss_fn': 'mse', 'verbose': 0}
        
        state_dict, y_pred_valid, y_pred_test = _train_nn(
            self.X_train_reg, self.y_train_reg,
            self.X_valid_reg, self.y_valid_reg,
            None, # No test data
            model_params, fit_params,
            self.feature_cols
        )
        self.assertIsInstance(state_dict, dict)
        self.assertIsInstance(y_pred_valid, np.ndarray)
        self.assertIsNone(y_pred_test)

    def test_no_valid_data_no_early_stopping(self):
        """Test training without validation data (early stopping disabled)."""
        model_params = {'hidden_sizes': [8], 'output_size': 1}
        fit_params = {'epochs': 2, 'loss_fn': 'mse', 'early_stopping_rounds': 0, 'verbose': 0}
        
        state_dict, y_pred_valid, y_pred_test = _train_nn(
            self.X_train_reg, self.y_train_reg,
            None, None, # No validation data
            self.X_test_reg,
            model_params, fit_params,
            self.feature_cols
        )
        self.assertIsInstance(state_dict, dict)
        self.assertIsNone(y_pred_valid) # No validation preds
        self.assertIsInstance(y_pred_test, np.ndarray)

    def test_early_stopping(self):
        """Test if early stopping triggers (requires loss increase)."""
        # Use small dataset, high LR to potentially increase val loss quickly
        model_params = {'hidden_sizes': [4], 'output_size': 1}
        fit_params = {
            'epochs': 50, # Allow enough epochs
            'batch_size': 4,
            'lr': 0.1, # High LR
            'optimizer': 'sgd',
            'loss_fn': 'mse',
            'early_stopping_rounds': 3, # Stop after 3 epochs of no improvement
            'verbose': 0
        }
        # Expect warning about potential overfitting, etc.
        # This test mainly checks if the mechanism runs and potentially stops early
        # Exact stopping point is non-deterministic/hard to guarantee
        try:
            state_dict, _, _ = _train_nn(
                self.X_train_reg[:10], self.y_train_reg[:10],
                self.X_valid_reg[:5], self.y_valid_reg[:5],
                None,
                model_params, fit_params,
                self.feature_cols
            )
            self.assertIsInstance(state_dict, dict) # Check it completed
        except Exception as e:
            self.fail(f"_train_nn failed during early stopping test: {e}")
            

if __name__ == '__main__':
    unittest.main() 