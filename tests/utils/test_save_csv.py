import unittest
import polars as pl
import pandas as pd # Remove if load_csv is fully Polars
import os
from utils.save_csv import save_csv
# Use the refactored Polars version of load_csv
from utils.load_csv import load_csv

class TestSaveCsv(unittest.TestCase):

    def setUp(self):
        """Set up a dummy Polars DataFrame and test directory."""
        self.test_dir = "test_data_save"
        self.test_file = os.path.join(self.test_dir, "output.csv")
        os.makedirs(self.test_dir, exist_ok=True)
        data = {'colA': [10, 20], 'colB': ['apple', 'banana']}
        self.df_to_save = pl.DataFrame(data)

    def tearDown(self):
        """Remove the dummy CSV file and directory after testing."""
        try:
            if os.path.exists(self.test_file):
                os.remove(self.test_file)
            os.rmdir(self.test_dir)
        except OSError as e:
            print(f"Error during teardown: {e}")

    def test_basic_saving(self):
        """Test basic CSV saving with Polars (no index by default)."""
        save_csv(self.df_to_save, self.test_file)
        self.assertTrue(os.path.exists(self.test_file))

        # Load back and verify content
        loaded_df = load_csv(self.test_file)
        # Polars DataFrame equality check
        self.assertTrue(loaded_df.equals(self.df_to_save))

    # Polars write_csv doesn't have an 'index' parameter like pandas.
    # If index saving is needed, it would require manual addition to the DataFrame first.
    # Skipping pandas-specific index test.
    # def test_basic_saving_with_index(self):
    #     pass 

    def test_saving_with_kwargs(self):
        """Test saving with Polars kwargs (e.g., different separator, no header)."""
        save_csv(self.df_to_save, self.test_file, separator='|', include_header=False)
        self.assertTrue(os.path.exists(self.test_file))

        # Load back and verify content (note: loading without header needs care)
        loaded_df = load_csv(self.test_file, separator='|', has_header=False, new_columns=['colA', 'colB'])
        
        # Type correction after loading without header (Polars might infer all as string)
        loaded_df = loaded_df.with_columns([
            pl.col('colA').cast(pl.Int64),
            pl.col('colB').cast(pl.Utf8)
        ])
        
        self.assertTrue(loaded_df.equals(self.df_to_save))

if __name__ == '__main__':
    unittest.main() 