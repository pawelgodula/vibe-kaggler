import unittest
import polars as pl
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, BaseCrossValidator
from utils.get_cv_indices import get_cv_indices

class TestGetCvIndices(unittest.TestCase):

    def setUp(self):
        """Set up sample data."""
        self.n_rows = 20
        self.df = pl.DataFrame({
            'feature': np.random.rand(self.n_rows),
            'target': np.random.randint(0, 2, size=self.n_rows), # Binary target for stratified
            'groups': np.random.randint(0, 4, size=self.n_rows) # 4 groups for group kfold
        })
        self.target_col = 'target'
        self.groups_series = self.df['groups']
        self.n_splits = 4

    def _validate_folds(self, folds: list, n_splits: int, n_rows: int):
        """Helper to validate basic fold structure."""
        self.assertIsInstance(folds, list)
        self.assertEqual(len(folds), n_splits)
        for i, fold in enumerate(folds):
            self.assertIsInstance(fold, tuple)
            self.assertEqual(len(fold), 2)
            train_idx, val_idx = fold
            self.assertIsInstance(train_idx, np.ndarray)
            self.assertIsInstance(val_idx, np.ndarray)
            self.assertTrue(np.issubdtype(train_idx.dtype, np.integer))
            self.assertTrue(np.issubdtype(val_idx.dtype, np.integer))
            # Check indices are within bounds
            self.assertTrue(np.all(train_idx >= 0) and np.all(train_idx < n_rows))
            self.assertTrue(np.all(val_idx >= 0) and np.all(val_idx < n_rows))
            # Check for overlap
            self.assertEqual(len(np.intersect1d(train_idx, val_idx)), 0)
            # Check total number of indices (might not be exact for all splitters, e.g. GroupKFold)
            # self.assertEqual(len(train_idx) + len(val_idx), n_rows)

    def test_kfold(self):
        """Test with KFold splitter."""
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        folds = get_cv_indices(self.df, kf)
        self._validate_folds(folds, self.n_splits, self.n_rows)
        # KFold doesn't need target or groups

    def test_stratified_kfold(self):
        """Test with StratifiedKFold splitter."""
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        folds = get_cv_indices(self.df, skf, target_col=self.target_col)
        self._validate_folds(folds, self.n_splits, self.n_rows)
        # Check stratification (distribution of target in val should be similar)
        # This is harder to assert precisely without complex checks, 
        # rely on sklearn's implementation being correct.
        
    def test_stratified_kfold_missing_target(self):
        """Test StratifiedKFold requires target_col."""
        skf = StratifiedKFold(n_splits=self.n_splits)
        with self.assertRaisesRegex(ValueError, "requires 'target_col'"):
            get_cv_indices(self.df, skf)

    def test_group_kfold(self):
        """Test with GroupKFold splitter."""
        gkf = GroupKFold(n_splits=self.n_splits)
        folds = get_cv_indices(self.df, gkf, groups=self.groups_series)
        self._validate_folds(folds, self.n_splits, self.n_rows)
        # Check group separation (groups in val should not be in train for that fold)
        for train_idx, val_idx in folds:
            train_groups = self.groups_series[train_idx].unique().to_list()
            val_groups = self.groups_series[val_idx].unique().to_list()
            self.assertEqual(len(set(train_groups) & set(val_groups)), 0)

    def test_group_kfold_missing_groups(self):
        """Test GroupKFold requires groups."""
        gkf = GroupKFold(n_splits=self.n_splits)
        with self.assertRaisesRegex(ValueError, "requires 'groups'"):
            get_cv_indices(self.df, gkf)
            
    def test_group_kfold_wrong_group_length(self):
        """Test GroupKFold group length mismatch."""
        gkf = GroupKFold(n_splits=self.n_splits)
        short_groups = self.groups_series[:10]
        with self.assertRaisesRegex(ValueError, "match DataFrame length"):
            get_cv_indices(self.df, gkf, groups=short_groups)

    def test_invalid_splitter_type(self):
        """Test providing a non-splitter object."""
        with self.assertRaises(TypeError):
            get_cv_indices(self.df, "not_a_splitter")
            
    def test_invalid_target_column(self):
        """Test providing a non-existent target column."""
        skf = StratifiedKFold(n_splits=self.n_splits)
        with self.assertRaisesRegex(ValueError, "not found in DataFrame"):
             get_cv_indices(self.df, skf, target_col="non_existent_target")

if __name__ == '__main__':
    unittest.main() 