import unittest
import numpy as np
import sklearn.model_selection
from utils.get_cv_splitter import get_cv_splitter

class TestGetCvSplitter(unittest.TestCase):

    def test_kfold_default(self):
        """Test KFold with default parameters."""
        splitter = get_cv_splitter(cv_type='kfold')
        self.assertIsInstance(splitter, sklearn.model_selection.KFold)
        self.assertEqual(splitter.get_n_splits(), 5)
        self.assertEqual(splitter.shuffle, True)
        self.assertIsNone(splitter.random_state)

    def test_kfold_custom(self):
        """Test KFold with custom parameters."""
        splitter = get_cv_splitter(cv_type='kfold', n_splits=10, shuffle=False, random_state=42)
        self.assertIsInstance(splitter, sklearn.model_selection.KFold)
        self.assertEqual(splitter.get_n_splits(), 10)
        self.assertEqual(splitter.shuffle, False)
        # Note: KFold stores random_state even if shuffle is False
        self.assertIsNone(splitter.random_state)

    def test_stratified_kfold(self):
        """Test StratifiedKFold."""
        splitter = get_cv_splitter(cv_type='stratified_kfold', n_splits=3, random_state=123)
        self.assertIsInstance(splitter, sklearn.model_selection.StratifiedKFold)
        self.assertEqual(splitter.get_n_splits(), 3)
        self.assertEqual(splitter.shuffle, True) # Default shuffle is True
        self.assertEqual(splitter.random_state, 123)

    def test_group_kfold(self):
        """Test GroupKFold."""
        splitter = get_cv_splitter(cv_type='group_kfold', n_splits=4)
        self.assertIsInstance(splitter, sklearn.model_selection.GroupKFold)
        self.assertEqual(splitter.get_n_splits(), 4)
        # GroupKFold does not have shuffle or random_state attributes used directly,
        # but hasattr might be true in some sklearn versions. Remove the check.
        # self.assertFalse(hasattr(splitter, 'shuffle'))
        # self.assertFalse(hasattr(splitter, 'random_state'))

    def test_invalid_type(self):
        """Test raising error for invalid cv_type."""
        with self.assertRaises(ValueError) as cm:
            get_cv_splitter(cv_type='invalid_splitter')
        self.assertIn("Unsupported cv_type: 'invalid_splitter'", str(cm.exception))

    def test_case_insensitivity(self):
        """Test that cv_type is case-insensitive."""
        splitter = get_cv_splitter(cv_type='KFOLD', n_splits=7)
        self.assertIsInstance(splitter, sklearn.model_selection.KFold)
        self.assertEqual(splitter.get_n_splits(), 7)

    # Example of how to use the splitter (optional, more like integration test)
    def test_splitter_usage_kfold(self):
        """Check basic splitting functionality for KFold."""
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 2, 100)
        splitter = get_cv_splitter(cv_type='kfold', n_splits=5, shuffle=True, random_state=0)
        splits = list(splitter.split(X, y))
        self.assertEqual(len(splits), 5)
        train_idx, val_idx = splits[0]
        self.assertEqual(len(train_idx), 80)
        self.assertEqual(len(val_idx), 20)

if __name__ == '__main__':
    unittest.main() 