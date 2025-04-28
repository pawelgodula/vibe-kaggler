# Tests for create_data_samples function.

import unittest
from unittest.mock import patch, MagicMock, call
import polars as pl
from pathlib import Path

# Adjust import path as necessary
try:
    from vibe_kaggler.utils.create_data_samples import create_data_samples
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from utils.create_data_samples import create_data_samples

# Use qualified names for patching
@patch('utils.create_data_samples.Path')
@patch('utils.create_data_samples.pl.read_csv')
@patch('utils.create_data_samples.pl.read_parquet')
class TestCreateDataSamples(unittest.TestCase):

    def setUp(self):
        self.mock_df = pl.DataFrame({'a': [1, 2], 'b': [3, 4]})
        # Mock DataFrame needs write methods
        self.mock_df.write_csv = MagicMock()
        self.mock_df.write_parquet = MagicMock()

    def test_create_samples_success(self, mock_read_parquet, mock_read_csv, mock_path):
        """Test successful creation of CSV and Parquet samples."""
        # Mock Path objects and their methods
        mock_raw_dir = MagicMock(spec=Path)
        mock_raw_dir.is_dir.return_value = True
        mock_samples_dir = MagicMock(spec=Path)
        mock_path.side_effect = [mock_raw_dir, mock_samples_dir]

        # Mock files found by iterdir
        mock_file_csv = MagicMock(spec=Path)
        mock_file_csv.is_file.return_value = True
        mock_file_csv.name = "data.csv"
        mock_file_csv.suffix = ".csv"
        
        mock_file_parquet = MagicMock(spec=Path)
        mock_file_parquet.is_file.return_value = True
        mock_file_parquet.name = "data.parquet"
        mock_file_parquet.suffix = ".parquet"
        
        mock_file_txt = MagicMock(spec=Path)
        mock_file_txt.is_file.return_value = True
        mock_file_txt.name = "notes.txt"
        mock_file_txt.suffix = ".txt"
        
        mock_raw_dir.iterdir.return_value = [mock_file_csv, mock_file_parquet, mock_file_txt]
        
        # Mock Polars read functions
        mock_read_csv.return_value = self.mock_df
        mock_read_parquet.return_value = self.mock_df
        
        # Mock the sample output paths
        mock_output_csv_path = MagicMock(spec=Path)
        mock_output_parquet_path = MagicMock(spec=Path)
        mock_samples_dir.__truediv__.side_effect = [mock_output_csv_path, mock_output_parquet_path]

        # Call the function
        create_data_samples("/raw/data", "/output/samples", n_rows=5)

        # Assertions
        mock_raw_dir.is_dir.assert_called_once()
        mock_samples_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_raw_dir.iterdir.assert_called_once()
        
        # Check CSV read/write calls
        mock_read_csv.assert_called_once_with(mock_file_csv, n_rows=5, infer_schema_length=5)
        self.mock_df.write_csv.assert_called_once_with(mock_output_csv_path)
        
        # Check Parquet read/write calls
        mock_read_parquet.assert_called_once_with(mock_file_parquet, n_rows=5)
        self.mock_df.write_parquet.assert_called_once_with(mock_output_parquet_path)
        
        # Check that __truediv__ was called correctly to create output paths
        mock_samples_dir.__truediv__.assert_has_calls([
            call(mock_file_csv.name),
            call(mock_file_parquet.name)
        ])

    def test_directory_not_found(self, mock_read_parquet, mock_read_csv, mock_path):
        """Test error if raw data directory doesn't exist."""
        mock_raw_dir = MagicMock(spec=Path)
        mock_raw_dir.is_dir.return_value = False
        mock_samples_dir = MagicMock(spec=Path)
        mock_path.side_effect = [mock_raw_dir, mock_samples_dir]
        
        with self.assertRaisesRegex(FileNotFoundError, "Raw data directory not found"):
            create_data_samples("/invalid/dir", "/output/samples")
        mock_samples_dir.mkdir.assert_not_called() # Shouldn't attempt creation

    def test_invalid_n_rows(self, mock_read_parquet, mock_read_csv, mock_path):
        """Test error if n_rows is not positive."""
        mock_raw_dir = MagicMock(spec=Path)
        mock_raw_dir.is_dir.return_value = True
        mock_samples_dir = MagicMock(spec=Path)
        # Provide enough mocks for two calls
        mock_path.side_effect = [
            mock_raw_dir, mock_samples_dir, 
            mock_raw_dir, mock_samples_dir 
        ]

        with self.assertRaisesRegex(ValueError, "n_rows must be a positive integer."):
            create_data_samples("/raw/data", "/output/samples", n_rows=0)
        with self.assertRaisesRegex(ValueError, "n_rows must be a positive integer."):
            create_data_samples("/raw/data", "/output/samples", n_rows=-5)

    def test_specific_extensions(self, mock_read_parquet, mock_read_csv, mock_path):
        """Test processing only specified extensions."""
        mock_raw_dir = MagicMock(spec=Path)
        mock_raw_dir.is_dir.return_value = True
        mock_samples_dir = MagicMock(spec=Path)
        mock_path.side_effect = [mock_raw_dir, mock_samples_dir]

        mock_file_csv = MagicMock(spec=Path, name="data.csv", suffix=".csv"); mock_file_csv.is_file.return_value=True
        mock_file_parquet = MagicMock(spec=Path, name="data.parquet", suffix=".parquet"); mock_file_parquet.is_file.return_value=True
        mock_raw_dir.iterdir.return_value = [mock_file_csv, mock_file_parquet]
        
        mock_read_parquet.return_value = self.mock_df
        mock_output_parquet_path = MagicMock(spec=Path)
        mock_samples_dir.__truediv__.return_value = mock_output_parquet_path

        # Only include parquet
        create_data_samples("/raw/data", "/output/samples", include_extensions=['.parquet'])
        
        mock_read_csv.assert_not_called()
        mock_read_parquet.assert_called_once()
        self.mock_df.write_csv.assert_not_called()
        self.mock_df.write_parquet.assert_called_once()

    def test_read_error_handling(self, mock_read_parquet, mock_read_csv, mock_path):
        """Test that errors during file reading are handled."""
        mock_raw_dir = MagicMock(spec=Path)
        mock_raw_dir.is_dir.return_value = True
        mock_samples_dir = MagicMock(spec=Path)
        mock_path.side_effect = [mock_raw_dir, mock_samples_dir]
        
        mock_file_csv = MagicMock(spec=Path, name="good.csv", suffix=".csv"); mock_file_csv.is_file.return_value=True
        mock_file_bad_csv = MagicMock(spec=Path, name="bad.csv", suffix=".csv"); mock_file_bad_csv.is_file.return_value=True
        mock_raw_dir.iterdir.return_value = [mock_file_csv, mock_file_bad_csv]
        
        # First read works, second raises error
        mock_read_csv.side_effect = [self.mock_df, pl.exceptions.ComputeError("Read failed")]
        mock_output_csv_path = MagicMock(spec=Path)
        mock_samples_dir.__truediv__.return_value = mock_output_csv_path # Only needed for good.csv
        
        create_data_samples("/raw/data", "/output/samples", n_rows=10)

        mock_read_csv.assert_has_calls([
            call(mock_file_csv, n_rows=10, infer_schema_length=10),
            call(mock_file_bad_csv, n_rows=10, infer_schema_length=10)
        ])
        self.mock_df.write_csv.assert_called_once_with(mock_output_csv_path) # Only called for good.csv

if __name__ == '__main__':
    unittest.main() 