import unittest
import polars as pl
from polars.testing import assert_frame_equal
import numpy as np
import os
# Need to import Polars load_csv to verify the output file
from utils.load_csv import load_csv
from utils.generate_submission_file import generate_submission_file

class TestGenerateSubmissionFile(unittest.TestCase):

    def setUp(self):
        """Set up test data and directory."""
        self.test_dir = "test_submission_dir"
        self.test_file = os.path.join(self.test_dir, "submission.csv")
        os.makedirs(self.test_dir, exist_ok=True)

        # Use Polars Series
        self.ids = pl.Series("test_id", [101, 102, 103], dtype=pl.Int64)
        self.predictions = pl.Series("preds", [0.9, 0.1, 0.5], dtype=pl.Float32)
        self.id_col = "ID"
        self.target_col = "Target"

    def tearDown(self):
        """Remove the generated file and directory."""
        try:
            if os.path.exists(self.test_file):
                os.remove(self.test_file)
            os.rmdir(self.test_dir)
        except OSError as e:
             print(f"Error during teardown: {e}")

    def test_generate_with_polars_series(self):
        """Test generating a submission file using Polars Series."""
        generate_submission_file(
            ids=self.ids,
            predictions=self.predictions,
            id_col_name=self.id_col,
            target_col_name=self.target_col,
            file_path=self.test_file
        )

        self.assertTrue(os.path.exists(self.test_file))
        submission_df = load_csv(self.test_file) # Load back with Polars
        
        # Create expected DataFrame
        expected_df = pl.DataFrame({
            self.id_col: self.ids.rename(self.id_col), # Rename Series before creating DF
            self.target_col: self.predictions.rename(self.target_col)
        }).with_columns([
            pl.col(self.id_col).cast(pl.Int64), # Ensure ID is Int64
            pl.col(self.target_col).cast(pl.Float64) # Expect Float64 output
        ])
        
        # Polars testing utility
        assert_frame_equal(submission_df, expected_df)

    def test_generate_with_numpy_arrays(self):
        """Test generating a submission file using numpy arrays."""
        np_ids = self.ids.to_numpy()
        np_predictions = self.predictions.to_numpy()
        
        generate_submission_file(
            ids=np_ids, 
            predictions=np_predictions,
            id_col_name="RowID",
            target_col_name="Pred",
            file_path=self.test_file
        )

        self.assertTrue(os.path.exists(self.test_file))
        submission_df = load_csv(self.test_file)
        
        # Create expected DataFrame
        expected_df = pl.DataFrame({
            "RowID": np_ids,
            "Pred": np_predictions
        }).with_columns([
             pl.col("RowID").cast(pl.Int64),
             pl.col("Pred").cast(pl.Float64) # Expect Float64 output
        ])
        
        assert_frame_equal(submission_df, expected_df)

    def test_length_mismatch(self):
        """Test error handling for input length mismatch."""
        short_predictions = pl.Series([0.1, 0.9]) # Different length
        with self.assertRaisesRegex(ValueError, "Length mismatch"):
            generate_submission_file(
                ids=self.ids,
                predictions=short_predictions,
                id_col_name=self.id_col,
                target_col_name=self.target_col,
                file_path=self.test_file
            )
        self.assertFalse(os.path.exists(self.test_file))

    def test_invalid_path(self):
        """Test error handling for an invalid save path."""
        invalid_path = "non_existent_dir/submission.csv"
        with self.assertRaises(Exception): # Expect error during save_csv
             generate_submission_file(
                ids=self.ids,
                predictions=self.predictions,
                id_col_name=self.id_col,
                target_col_name=self.target_col,
                file_path=invalid_path
            )

if __name__ == '__main__':
    unittest.main() 