import unittest
import random
import numpy as np
import os
from utils.seed_everything import seed_everything

# Check if DL frameworks are available
try:
    import tensorflow as tf
    tf_available = True
except ImportError:
    tf_available = False

try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False

class TestSeedEverything(unittest.TestCase):

    def get_random_values(self):
        """Helper to get values from different random sources."""
        py_random = random.random()
        np_random = np.random.rand(1)[0]
        tf_random = None
        torch_random = None

        if tf_available and hasattr(tf.random, 'uniform'): # Check for TF2 style
             # Ensure we use a TF function affected by the global seed
             tf_random = tf.random.uniform([1], seed=123).numpy()[0] # Use op-level seed too for consistency

        if torch_available:
            torch_random = torch.rand(1).item()

        return py_random, np_random, tf_random, torch_random

    def test_fixed_seed_reproducibility(self):
        """Test that using the same fixed seed yields the same results."""
        fixed_seed = 42

        seed_everything(fixed_seed)
        vals1_py, vals1_np, vals1_tf, vals1_torch = self.get_random_values()
        py_hash_seed1 = os.environ.get('PYTHONHASHSEED')

        # Reset (important for TF/Torch state if tests run in same process)
        # No standard way to fully reset global state, but re-seeding helps

        seed_everything(fixed_seed)
        vals2_py, vals2_np, vals2_tf, vals2_torch = self.get_random_values()
        py_hash_seed2 = os.environ.get('PYTHONHASHSEED')

        self.assertEqual(vals1_py, vals2_py, "Python random not reproducible")
        self.assertEqual(vals1_np, vals2_np, "NumPy random not reproducible")
        self.assertEqual(py_hash_seed1, str(fixed_seed))
        self.assertEqual(py_hash_seed2, str(fixed_seed))

        if tf_available and vals1_tf is not None:
            self.assertEqual(vals1_tf, vals2_tf, "TensorFlow random not reproducible")
        if torch_available and vals1_torch is not None:
            self.assertEqual(vals1_torch, vals2_torch, "PyTorch random not reproducible")

    def test_different_fixed_seeds_differ(self):
        """Test that different fixed seeds yield different results."""
        seed1 = 123
        seed2 = 456

        seed_everything(seed1)
        vals1_py, vals1_np, vals1_tf, vals1_torch = self.get_random_values()

        seed_everything(seed2)
        vals2_py, vals2_np, vals2_tf, vals2_torch = self.get_random_values()

        self.assertNotEqual(vals1_py, vals2_py, "Python random should differ")
        self.assertNotEqual(vals1_np, vals2_np, "NumPy random should differ")

        if tf_available and vals1_tf is not None and vals2_tf is not None:
            self.assertNotEqual(vals1_tf, vals2_tf, "TensorFlow random should differ")
        if torch_available and vals1_torch is not None and vals2_torch is not None:
             self.assertNotEqual(vals1_torch, vals2_torch, "PyTorch random should differ")

    def test_none_seed_non_reproducible(self):
        """Test that using seed=None is not reproducible."""
        # Note: This has a small chance of flakes if seeds collide, but very unlikely
        seed_everything(None)
        vals1_py, vals1_np, _, _ = self.get_random_values()
        hash_seed1 = os.environ.get('PYTHONHASHSEED')

        seed_everything(None)
        vals2_py, vals2_np, _, _ = self.get_random_values()
        hash_seed2 = os.environ.get('PYTHONHASHSEED')

        self.assertNotEqual(vals1_py, vals2_py, "Python random should differ with None seed")
        self.assertNotEqual(vals1_np, vals2_np, "NumPy random should differ with None seed")
        # Hash seeds should also differ
        self.assertIsNotNone(hash_seed1)
        self.assertIsNotNone(hash_seed2)
        self.assertNotEqual(hash_seed1, hash_seed2)

if __name__ == '__main__':
    # You might see warnings about TF/Torch not being tested if they aren't installed
    if not tf_available:
        print("Warning: TensorFlow not found, skipping TF seeding tests.")
    if not torch_available:
        print("Warning: PyTorch not found, skipping PyTorch seeding tests.")
    unittest.main() 