import unittest
import polars as pl
import numpy as np
from polars.testing import assert_frame_equal
from sklearn.model_selection import KFold # For generating realistic indices
from utils.apply_target_encoding import apply_target_encoding
# Need get_cv_indices to help set up the test
from utils.get_cv_indices import get_cv_indices 

# Adjust import path as necessary
try:
    from vibe_kaggler.utils.apply_target_encoding import apply_target_encoding
    from vibe_kaggler.utils.get_cv_indices import get_cv_indices
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from utils.apply_target_encoding import apply_target_encoding
    from utils.get_cv_indices import get_cv_indices # Need this to generate indices

# Helper function to create DataFrames
def create_test_df(n_samples=100, seed=42):
    np.random.seed(seed)
    data = {
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'category2': np.random.choice(['X', 'Y', 'Z', None], n_samples), # Include nulls
        'target': np.random.rand(n_samples) * 10 + 
                  np.select(
                      [np.random.choice(['A', 'B', 'C', 'D'], n_samples) == 'A',
                       np.random.choice(['A', 'B', 'C', 'D'], n_samples) == 'B'],
                      [5, -3], default=0
                  ) # Add some dependency
    }
    return pl.DataFrame(data)

class TestApplyTargetEncoding(unittest.TestCase):

    def setUp(self):
        """Set up sample data and CV indices."""
        self.train_df = create_test_df(n_samples=100, seed=42)
        self.test_df = create_test_df(n_samples=50, seed=123)
        # Add a category in test not in train
        self.test_df = self.test_df.with_columns(
            pl.when(pl.col('category') == 'A').then(pl.lit('E')).otherwise(pl.col('category')).alias('category')
        )
        
        self.features = ['category', 'category2']
        self.target_col = 'target'
        self.n_splits = 5
        self.cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        # Use the helper function to get indices (assumes get_cv_indices works)
        self.cv_indices = get_cv_indices(self.train_df, self.cv, target_col=self.target_col)
        self.smoothing = 5.0

        self.global_mean = self.train_df[self.target_col].mean()

    def test_basic_mean_encoding(self):
        train_encoded, test_encoded = apply_target_encoding(
            self.train_df, self.features, self.target_col, self.cv_indices,
            test_df=self.test_df, agg_stat='mean', smoothing=self.smoothing
        )
        
        # Check shapes
        self.assertEqual(train_encoded.shape[0], self.train_df.shape[0])
        self.assertEqual(test_encoded.shape[0], self.test_df.shape[0])
        
        # Check new columns exist
        for f in self.features:
            new_col = f"{f}_te"
            self.assertIn(new_col, train_encoded.columns)
            self.assertIn(new_col, test_encoded.columns)
            # Check no nulls after encoding (due to smoothing/prior fill)
            self.assertEqual(train_encoded[new_col].null_count(), 0)
            self.assertEqual(test_encoded[new_col].null_count(), 0)
        
        # Check OOF calculation logic (spot check for one category/fold)
        # For category 'A', fold 0 validation set
        val_idx_fold0 = self.cv_indices[0][1]
        train_idx_fold0 = self.cv_indices[0][0]
        
        cat_a_val_fold0 = self.train_df.filter((pl.col('category') == 'A') & pl.Series(np.arange(len(self.train_df))).is_in(val_idx_fold0))
        encoded_col_cat = f'category_te'
        
        # Calculate expected encoding for 'A' based on train_idx_fold0
        cat_a_train_fold0_target = self.train_df[train_idx_fold0].filter(pl.col('category') == 'A')[self.target_col]
        fold_prior = self.train_df[train_idx_fold0][self.target_col].mean()
        expected_stat = cat_a_train_fold0_target.mean()
        count = len(cat_a_train_fold0_target)
        expected_smoothed = (expected_stat * count + fold_prior * self.smoothing) / (count + self.smoothing)
        
        # Check if the encoded values for 'A' in validation fold 0 match expected
        actual_encoded_vals = train_encoded[val_idx_fold0].filter(pl.col('category') == 'A')[encoded_col_cat]
        if not actual_encoded_vals.is_empty():
            np.testing.assert_allclose(actual_encoded_vals[0], expected_smoothed, rtol=1e-6)
            self.assertTrue((actual_encoded_vals == expected_smoothed).all()) # All 'A' in this val set should have same OOF value
            
    def test_median_encoding(self):
        train_encoded, test_encoded = apply_target_encoding(
            self.train_df, self.features, self.target_col, self.cv_indices,
            test_df=self.test_df, agg_stat='median', smoothing=self.smoothing
        )
        self.assertIn('category_te', train_encoded.columns)
        self.assertIn('category_te', test_encoded.columns)
        self.assertEqual(train_encoded['category_te'].null_count(), 0)
        self.assertEqual(test_encoded['category_te'].null_count(), 0)
        
    def test_percentile_encoding(self):
        percentile_val = 25.0
        train_encoded, test_encoded = apply_target_encoding(
            self.train_df, self.features, self.target_col, self.cv_indices,
            test_df=self.test_df, agg_stat='percentile', percentile=percentile_val,
            smoothing=self.smoothing
        )
        self.assertIn('category_te', train_encoded.columns)
        self.assertIn('category_te', test_encoded.columns)
        self.assertEqual(train_encoded['category_te'].null_count(), 0)
        self.assertEqual(test_encoded['category_te'].null_count(), 0)

    def test_unknown_category_handling(self):
        # Test set has category 'E' which is not in train
        train_encoded, test_encoded = apply_target_encoding(
            self.train_df, ['category'], self.target_col, self.cv_indices, 
            test_df=self.test_df, agg_stat='mean', smoothing=self.smoothing
        )
        
        global_mean = self.train_df[self.target_col].mean()
        encoded_col = 'category_te'
        
        # Find encoded value for category 'E' in test set
        encoded_e_value = test_encoded.filter(pl.col('category') == 'E')[encoded_col]
        
        self.assertFalse(encoded_e_value.is_empty())
        # It should be filled with the smoothed global prior (which is just the global mean here)
        # The smoothing formula simplifies to global_mean when count is 0
        np.testing.assert_allclose(encoded_e_value[0], global_mean, rtol=1e-6)
        
    def test_no_test_df(self):
        train_encoded, test_encoded = apply_target_encoding(
            self.train_df, self.features, self.target_col, self.cv_indices, 
            test_df=None, agg_stat='mean'
        )
        self.assertIsNone(test_encoded)
        self.assertIn('category_te', train_encoded.columns)

    def test_missing_percentile_value(self):
        with self.assertRaisesRegex(ValueError, "percentile must be provided"): 
            apply_target_encoding(
                self.train_df, self.features, self.target_col, self.cv_indices,
                agg_stat='percentile' # Missing percentile value
            )

    def test_invalid_agg_stat(self):
        with self.assertRaisesRegex(ValueError, "agg_stat must be one of"): 
            apply_target_encoding(
                self.train_df, self.features, self.target_col, self.cv_indices,
                agg_stat='variance' # Invalid stat
            )

    def test_missing_column(self):
        with self.assertRaises(pl.exceptions.ColumnNotFoundError):
            apply_target_encoding(
                self.train_df, ['category', 'non_existent_col'], 
                self.target_col, self.cv_indices
            )
        with self.assertRaises(pl.exceptions.ColumnNotFoundError):
            apply_target_encoding(
                self.train_df, self.features, 'non_existent_target', 
                self.cv_indices
            )

    def test_detailed_oof_calculations(self):
        """Manually verify OOF calculations for a small, controlled dataset."""
        # 1. Define small DataFrame
        data = {
            'cat':    ['A', 'B', 'A', 'B', 'A', 'C', 'B', 'A', 'B', 'A'],
            'target': [ 10,  20,  12,  22,  14,  50,  18,  11,  25,  15]
        }
        small_df = pl.DataFrame(data)
        features = ['cat']
        target = 'target'
        smoothing = 2.0 # Use a small smoothing value for easier checks
        n_splits = 2
        
        # 2. Deterministic CV split
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=0) # Fixed seed
        cv_indices = get_cv_indices(small_df, cv)
        
        # --- Check Fold 0: train_idx=[1, 2, 3, 5, 9], val_idx=[0, 4, 6, 7, 8] ---
        train_idx_0 = cv_indices[0][0]
        val_idx_0 = cv_indices[0][1]
        
        # Calculate Fold 0 Prior (mean of target in train_idx_0)
        fold0_targets = small_df[train_idx_0][target] # Targets: [20, 12, 22, 50, 15]
        fold0_prior_mean = fold0_targets.mean() # (20+12+22+50+15)/5 = 119/5 = 23.8
        fold0_prior_median = fold0_targets.median() # sorted: [12, 15, 20, 22, 50] -> 20.0
        fold0_prior_p90 = fold0_targets.quantile(0.90, interpolation='linear') # Should be between 22 and 50
        # Calculation: 0.9*(5-1) = 3.6 -> index 3 + 0.6 * (index 4 - index 3) = 22 + 0.6*(50-22) = 22 + 0.6*28 = 22 + 16.8 = 38.8

        # Calculate stats for category 'A' in Fold 0 Train (indices 2, 9 -> targets 12, 15)
        cat_a_targets_f0 = small_df[train_idx_0].filter(pl.col('cat') == 'A')[target] # [12, 15]
        count_a_f0 = len(cat_a_targets_f0)
        mean_a_f0 = cat_a_targets_f0.mean() # (12+15)/2 = 13.5
        median_a_f0 = cat_a_targets_f0.median() # 13.5
        p90_a_f0 = cat_a_targets_f0.quantile(0.90, interpolation='linear') # 0.9*(2-1)=0.9 -> idx 0 + 0.9*(idx1-idx0) = 12 + 0.9*(15-12) = 12 + 2.7 = 14.7
        
        # Calculate expected RAW values for 'A' in Fold 0 Validation (since smoothing=False)
        exp_raw_mean_a_f0 = mean_a_f0 # 13.5
        exp_raw_median_a_f0 = median_a_f0 # 13.5
        exp_raw_p90_a_f0 = p90_a_f0 # 14.7

        # Calculate stats for category 'C' (singleton) in Fold 0 Train (index 5 -> target 50)
        cat_c_targets_f0 = small_df[train_idx_0].filter(pl.col('cat') == 'C')[target] # [50]
        count_c_f0 = len(cat_c_targets_f0)
        mean_c_f0 = cat_c_targets_f0.mean() # 50.0
        median_c_f0 = cat_c_targets_f0.median() # 50.0
        p90_c_f0 = cat_c_targets_f0.quantile(0.90, interpolation='linear') # 50.0 (only one value)
        
        # --- Run Encodings & Assert (with use_smoothing=False) --- 
        # Mean
        train_enc_mean, _ = apply_target_encoding(small_df, features, target, cv_indices, 
                                                  agg_stat='mean', smoothing=smoothing, 
                                                  use_smoothing=False)
        actual_mean_a_f0 = train_enc_mean[val_idx_0].filter(pl.col('cat') == 'A')['cat_te']
        np.testing.assert_allclose(actual_mean_a_f0.to_numpy(), exp_raw_mean_a_f0, rtol=1e-6)
        # Since C is not in val_idx_0, we can't check it directly here. Check overall OOF?
        # Let's check category B in val_idx_0 (indices 6, 8 -> target 18, 25)
        cat_b_targets_f0 = small_df[train_idx_0].filter(pl.col('cat') == 'B')[target] # [20, 22]
        mean_b_f0 = cat_b_targets_f0.mean() # 21.0
        exp_raw_mean_b_f0 = mean_b_f0 # 21.0
        actual_mean_b_f0 = train_enc_mean[val_idx_0].filter(pl.col('cat') == 'B')['cat_te']
        np.testing.assert_allclose(actual_mean_b_f0.to_numpy(), exp_raw_mean_b_f0, rtol=1e-6)

        # Median
        train_enc_median, _ = apply_target_encoding(small_df, features, target, cv_indices, 
                                                    agg_stat='median', smoothing=smoothing,
                                                    use_smoothing=False)
        actual_median_a_f0 = train_enc_median[val_idx_0].filter(pl.col('cat') == 'A')['cat_te']
        np.testing.assert_allclose(actual_median_a_f0.to_numpy(), exp_raw_median_a_f0, rtol=1e-6)
        median_b_f0 = cat_b_targets_f0.median() # 21.0
        exp_raw_median_b_f0 = median_b_f0 # 21.0
        actual_median_b_f0 = train_enc_median[val_idx_0].filter(pl.col('cat') == 'B')['cat_te']
        np.testing.assert_allclose(actual_median_b_f0.to_numpy(), exp_raw_median_b_f0, rtol=1e-6)
        
        # Percentile (90th)
        p90_val = 90.0
        train_enc_p90, _ = apply_target_encoding(small_df, features, target, cv_indices, 
                                                 agg_stat='percentile', percentile=p90_val, 
                                                 smoothing=smoothing, use_smoothing=False)
        actual_p90_a_f0 = train_enc_p90[val_idx_0].filter(pl.col('cat') == 'A')['cat_te']
        np.testing.assert_allclose(actual_p90_a_f0.to_numpy(), exp_raw_p90_a_f0, rtol=1e-6)
        p90_b_f0 = cat_b_targets_f0.quantile(0.90, interpolation='linear') # 21.8
        exp_raw_p90_b_f0 = p90_b_f0 # 21.8
        actual_p90_b_f0 = train_enc_p90[val_idx_0].filter(pl.col('cat') == 'B')['cat_te']
        np.testing.assert_allclose(actual_p90_b_f0.to_numpy(), exp_raw_p90_b_f0, rtol=1e-6)

        # Check singleton category 'C' (appears in fold 0 training, index 5)
        # It doesn't appear in fold 0 validation, so its encoded value is determined
        # by fold 1's calculation. Let's check its value in the final output.
        # Fold 1: train_idx=[0, 4, 6, 7, 8], val_idx=[1, 2, 3, 5, 9]
        train_idx_1 = cv_indices[1][0]
        val_idx_1 = cv_indices[1][1]
        fold1_targets = small_df[train_idx_1][target] # Targets: [10, 14, 18, 11, 25]
        fold1_prior_mean = fold1_targets.mean() # (10+14+18+11+25)/5 = 78/5 = 15.6
        # Cat C is not in fold 1 training data. So encoding for C in val_idx_1 (index 5)
        # should be the fold 1 prior.
        actual_mean_c_f1 = train_enc_mean[val_idx_1].filter(pl.col('cat') == 'C')['cat_te'].item()
        self.assertAlmostEqual(actual_mean_c_f1, fold1_prior_mean, places=6)

if __name__ == '__main__':
    unittest.main() 