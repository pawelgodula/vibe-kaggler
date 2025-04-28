import unittest
import polars as pl
import pandas as pd # Remove if not needed
import os
import pytest

# Check if necessary libs installed
try:
    import pyarrow
    PYARROW_INSTALLED = True
except ImportError:
    PYARROW_INSTALLED = False

# Try importing functions
try:
    from utils.save_parquet import save_parquet
    from utils.load_parquet import load_parquet # Use Polars loader
    SAVE_PARQUET_AVAILABLE = True
except ImportError:
    SAVE_PARQUET_AVAILABLE = False
    def save_parquet(*args, **kwargs):
        raise ImportError("save_parquet or its deps (pyarrow) not available.")
    def load_parquet(*args, **kwargs):
        # Define dummy if save failed but load exists
        try: from utils.load_parquet import load_parquet
        except ImportError: raise ImportError("load/save_parquet or deps missing")


# Skip all tests if function isn't available or pyarrow is missing
pytestmark = pytest.mark.skipif(not (SAVE_PARQUET_AVAILABLE and PYARROW_INSTALLED),
                                reason="Requires save_parquet function and pyarrow library")

class TestSaveParquet(unittest.TestCase):

    def setUp(self):
        """Set up a dummy Polars DataFrame and test directory."""
        self.test_dir = "test_data_save_parquet"
        self.test_file = os.path.join(self.test_dir, "output.parquet")
        os.makedirs(self.test_dir, exist_ok=True)
        data = {'col_float': [1.1, 2.2], 'col_bool': [True, False]}
        # Use specific Polars dtypes for clarity
        self.df_to_save = pl.DataFrame(data).with_columns([
            pl.col("col_float").cast(pl.Float32),
            pl.col("col_bool").cast(pl.Boolean)
        ])

    def tearDown(self):
        """Remove the dummy Parquet file and directory after testing."""
        try:
            if os.path.exists(self.test_file):
                os.remove(self.test_file)
            os.rmdir(self.test_dir)
        except OSError as e:
             print(f"Error during teardown: {e}")

    def test_basic_saving(self):
        """Test basic Parquet saving with Polars."""
        save_parquet(self.df_to_save, self.test_file)
        self.assertTrue(os.path.exists(self.test_file))

        # Load back and verify content
        loaded_df = load_parquet(self.test_file)
        self.assertTrue(loaded_df.equals(self.df_to_save))
        # Check dtypes are preserved
        self.assertEqual(loaded_df['col_float'].dtype, pl.Float32)
        self.assertEqual(loaded_df['col_bool'].dtype, pl.Boolean)

    # Polars write_parquet doesn't have an 'index' parameter.
    # If saving row numbers/index is needed, add it as a column first.
    # def test_basic_saving_with_index(self):
    #     pass

    def test_saving_with_kwargs(self):
        """Test saving with Polars kwargs (e.g., compression, statistics)."""
        # Example: using 'gzip' compression and enabling statistics
        save_parquet(self.df_to_save, self.test_file, compression='gzip', statistics=True)
        self.assertTrue(os.path.exists(self.test_file))

        # Load back and verify content
        loaded_df = load_parquet(self.test_file)
        self.assertTrue(loaded_df.equals(self.df_to_save))
        # Potentially verify metadata if needed (more complex)

# Note: Running requires pytest and pyarrow
# Run with: `pytest tests/utils/test_save_parquet.py`
if __name__ == '__main__':
    if SAVE_PARQUET_AVAILABLE and PYARROW_INSTALLED:
        unittest.main()
    else:
        print("Skipping Parquet tests: save_parquet function or pyarrow not available.") 