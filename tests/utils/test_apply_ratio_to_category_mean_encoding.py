import unittest
import polars as pl
import numpy as np
from polars.testing import assert_frame_equal

# Adjust import path as necessary
try:
    from vibe_kaggler.utils.apply_ratio_to_category_mean_encoding import apply_ratio_to_category_mean_encoding
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from utils.apply_ratio_to_category_mean_encoding import apply_ratio_to_category_mean_encoding


class TestApplyRatioToCategoryMeanEncoding(unittest.TestCase):

    def setUp(self):
        """Set up sample dataframes."""
        self.train_data = {
            'group': ['A', 'B', 'A', 'C', 'B', 'A', 'C', None], 
            'value1': [10, 20, 12, 50, 22, None, 55, 30], # Include null
            'value2': [1.0, 0.0, 0.5, 2.0, 0.0, 1.5, 2.5, 1.0] # Include zero
        }
        self.train_df = pl.DataFrame(self.train_data)
        
        self.test_data = {
            'group': ['A', 'B', 'D', 'C', None], # D is unseen in train
            'value1': [15, 25, 100, 60, 40],
            'value2': [0.8, 0.0, 5.0, 2.2, 1.2]
        }
        self.test_df = pl.DataFrame(self.test_data)
        
        self.numerical_features = ['value1', 'value2']
        self.category_col = 'group'
        self.suffix = "_ratio_vs_group"

    def test_calculate_on_train(self):
        """Test calculating means on training data only."""
        train_out, test_out = apply_ratio_to_category_mean_encoding(
            self.train_df, self.numerical_features, self.category_col, 
            test_df=self.test_df, calculate_mean_on='train'
        )
        
        # Expected means on train data:
        # group A: value1=(10+12)/2=11.0, value2=(1.0+0.5+1.5)/3=1.0
        # group B: value1=(20+22)/2=21.0, value2=(0.0+0.0)/2=0.0
        # group C: value1=(50+55)/2=52.5, value2=(2.0+2.5)/2=2.25
        # group None: value1=30.0, value2=1.0
        
        # Expected ratios (train):
        # value1: [10/11, 20/21, 12/11, 50/52.5, 22/21, 1.0(null), 55/52.5, 30/30]
        # value2: [1.0/1.0, 0.0/0.0->1.0, 0.5/1.0, 2.0/2.25, 0.0/0.0->1.0, 1.5/1.0, 2.5/2.25, 1.0/1.0]
        exp_train_v1 = pl.Series("value1" + self.suffix, [10/11, 20/21, 12/11, 50/52.5, 22/21, 1.0, 55/52.5, 1.0])
        exp_train_v2 = pl.Series("value2" + self.suffix, [1.0,   1.0,   0.5/1.0, 2.0/2.25, 1.0,   1.5/1.0, 2.5/2.25, 1.0])
        
        # Expected ratios (test - using train means):
        # value1: [15/11, 25/21, 1.0(D), 60/52.5, 40/30]
        # value2: [0.8/1.0, 0.0/0.0->1.0, 1.0(D), 2.2/2.25, 1.2/1.0]
        exp_test_v1 = pl.Series("value1" + self.suffix, [15/11, 25/21, 1.0, 60/52.5, 40/30])
        exp_test_v2 = pl.Series("value2" + self.suffix, [0.8,   1.0,   1.0, 2.2/2.25, 1.2])

        assert_frame_equal(train_out.select(["value1"+self.suffix, "value2"+self.suffix]),
                             pl.DataFrame([exp_train_v1, exp_train_v2]), rtol=1e-6)
        assert_frame_equal(test_out.select(["value1"+self.suffix, "value2"+self.suffix]),
                             pl.DataFrame([exp_test_v1, exp_test_v2]), rtol=1e-6)
                             
    def test_calculate_on_combined(self):
        """Test calculating means on combined train+test data."""
        train_out, test_out = apply_ratio_to_category_mean_encoding(
            self.train_df, self.numerical_features, self.category_col,
            test_df=self.test_df, calculate_mean_on='combined'
        )

        # Expected means on combined data:
        # group A: value1=(10+12+15)/3=37/3, value2=(1.0+0.5+1.5+0.8)/4=3.8/4=0.95
        # group B: value1=(20+22+25)/3=67/3, value2=(0.0+0.0+0.0)/3=0.0
        # group C: value1=(50+55+60)/3=165/3=55.0, value2=(2.0+2.5+2.2)/3=6.7/3
        # group D: value1=100.0, value2=5.0
        # group None: value1=(30+40)/2=35.0, value2=(1.0+1.2)/2=1.1

        # Expected ratios (train - using combined means):
        # value1: [10/(37/3), 20/(67/3), 12/(37/3), 50/55, 22/(67/3), 1.0(null), 55/55, 30/35]
        # value2: [1.0/0.95, 0.0/0.0->1.0, 0.5/0.95, 2.0/(6.7/3), 0.0/0.0->1.0, 1.5/0.95, 2.5/(6.7/3), 1.0/1.1]
        exp_train_v1 = pl.Series("value1" + self.suffix, [30/37, 60/67, 36/37, 50/55, 66/67, 1.0, 1.0, 30/35])
        exp_train_v2 = pl.Series("value2" + self.suffix, [1.0/0.95, 1.0, 0.5/0.95, 6.0/6.7, 1.0, 1.5/0.95, 7.5/6.7, 1.0/1.1])

        # Expected ratios (test - using combined means):
        # value1: [15/(37/3), 25/(67/3), 100/100, 60/55, 40/35]
        # value2: [0.8/0.95, 0.0/0.0->1.0, 5.0/5.0, 2.2/(6.7/3), 1.2/1.1]
        exp_test_v1 = pl.Series("value1" + self.suffix, [45/37, 75/67, 1.0, 60/55, 40/35])
        exp_test_v2 = pl.Series("value2" + self.suffix, [0.8/0.95, 1.0, 1.0, 6.6/6.7, 1.2/1.1])

        assert_frame_equal(train_out.select(["value1"+self.suffix, "value2"+self.suffix]),
                             pl.DataFrame([exp_train_v1, exp_train_v2]), rtol=1e-6)
        assert_frame_equal(test_out.select(["value1"+self.suffix, "value2"+self.suffix]),
                             pl.DataFrame([exp_test_v1, exp_test_v2]), rtol=1e-6)

    def test_custom_fill_value(self):
        """Test using a custom fill value."""
        fill_val = -999.0
        train_out, test_out = apply_ratio_to_category_mean_encoding(
            self.train_df, ['value2'], self.category_col, # Only value2 has zero mean
            test_df=self.test_df, calculate_mean_on='train', fill_value=fill_val
        )
        # group B mean for value2 is 0 on train. Ratio should be fill_val.
        # group D in test is unseen. Ratio should be fill_val.
        exp_train_v2 = pl.Series("value2" + self.suffix, [1.0, fill_val, 0.5/1.0, 2.0/2.25, fill_val, 1.5/1.0, 2.5/2.25, 1.0])
        exp_test_v2 = pl.Series("value2" + self.suffix, [0.8/1.0, fill_val, fill_val, 2.2/2.25, 1.2/1.0])

        assert_frame_equal(train_out.select(["value2"+self.suffix]),
                             pl.DataFrame([exp_train_v2]), rtol=1e-6)
        assert_frame_equal(test_out.select(["value2"+self.suffix]),
                             pl.DataFrame([exp_test_v2]), rtol=1e-6)
                             
    def test_no_test_df(self):
        """Test behavior when test_df is None."""
        train_out, test_out = apply_ratio_to_category_mean_encoding(
            self.train_df, self.numerical_features, self.category_col, 
            test_df=None, calculate_mean_on='train'
        )
        self.assertIsNone(test_out)
        self.assertIn("value1" + self.suffix, train_out.columns)
        self.assertIn("value2" + self.suffix, train_out.columns)

    def test_error_missing_test_df(self):
        """Test ValueError if test_df is needed but missing."""
        with self.assertRaisesRegex(ValueError, "test_df must be provided"): 
            apply_ratio_to_category_mean_encoding(
                self.train_df, self.numerical_features, self.category_col, 
                calculate_mean_on='test'
            )
        with self.assertRaisesRegex(ValueError, "test_df must be provided"): 
            apply_ratio_to_category_mean_encoding(
                self.train_df, self.numerical_features, self.category_col, 
                calculate_mean_on='combined'
            )

    def test_error_missing_columns(self):
        """Test ColumnNotFoundError if columns are missing."""
        with self.assertRaises(pl.exceptions.ColumnNotFoundError):
            apply_ratio_to_category_mean_encoding(
                self.train_df, ['value1', 'missing'], self.category_col, self.test_df
            )
        with self.assertRaises(pl.exceptions.ColumnNotFoundError):
             apply_ratio_to_category_mean_encoding(
                 self.train_df, self.numerical_features, 'missing_cat', self.test_df
             )
        with self.assertRaises(pl.exceptions.ColumnNotFoundError):
             bad_test = self.test_df.drop('value1')
             apply_ratio_to_category_mean_encoding(
                 self.train_df, self.numerical_features, self.category_col, bad_test
             )

    def test_error_non_numeric_feature(self):
        """Test TypeError if a numerical feature is not numeric."""
        bad_train = self.train_df.with_columns(pl.col('value1').cast(pl.Utf8))
        with self.assertRaisesRegex(TypeError, "Column 'value1'.*not numeric"):
            apply_ratio_to_category_mean_encoding(
                 bad_train, self.numerical_features, self.category_col, self.test_df
             )

if __name__ == '__main__':
    unittest.main() 