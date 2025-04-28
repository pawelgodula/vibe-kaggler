import unittest
import polars as pl
import pandas as pd # Keep for setup if needed initially, or remove
import os
from utils.load_csv import load_csv

class TestLoadCsv(unittest.TestCase):

    def setUp(self):
        """Set up a dummy CSV file for testing using Polars."""
        self.test_dir = "test_data"
        self.test_file = os.path.join(self.test_dir, "test.csv")
        os.makedirs(self.test_dir, exist_ok=True)
        # Use Polars to create the test file
        data = {'col1': [1, 2], 'col2': [3, 4]}
        df = pl.DataFrame(data)
        df.write_csv(self.test_file)

    def tearDown(self):
        """Remove the dummy CSV file and directory after testing."""
        try:
            os.remove(self.test_file)
            # Remove separator test file if it exists
            sep_test_file = os.path.join(self.test_dir, "test_sep.csv")
            if os.path.exists(sep_test_file):
                 os.remove(sep_test_file)
            os.rmdir(self.test_dir)
        except OSError as e:
            print(f"Error during teardown: {e}") # Avoid test failures during cleanup

    def test_basic_loading(self):
        """Test basic CSV loading into Polars DataFrame."""
        df = load_csv(self.test_file)
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape, (2, 2))
        self.assertListEqual(df.columns, ['col1', 'col2'])
        self.assertEqual(df['col1'][0], 1)
        # Check dtypes (polars infers types)
        self.assertEqual(df['col1'].dtype, pl.Int64)
        self.assertEqual(df['col2'].dtype, pl.Int64)

    def test_loading_with_kwargs(self):
        """Test loading with Polars kwargs (e.g., separator, dtypes)."""
        # Create a file with a different separator and different types
        sep_test_file = os.path.join(self.test_dir, "test_sep.csv")
        data = {'colA': [10.5, 20.5], 'colB': ['x', 'y']}
        df_sep = pl.DataFrame(data)
        df_sep.write_csv(sep_test_file, separator=';')

        # Load specifying separator and potential dtype override
        df = load_csv(sep_test_file, separator=';', dtypes={'colA': pl.Float32})
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape, (2, 2))
        self.assertListEqual(df.columns, ['colA', 'colB'])
        self.assertEqual(df['colA'].dtype, pl.Float32) # Check dtype override
        self.assertEqual(df['colB'].dtype, pl.Utf8) # String type in Polars
        self.assertAlmostEqual(df['colA'][0], 10.5)

    def test_file_not_found(self):
        """Test loading a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_csv("non_existent_file.csv")

if __name__ == '__main__':
    unittest.main() 