import unittest
import polars as pl
import pandas as pd # Keep for setup maybe?
import os
import pytest 

# Check if necessary libs installed
try:
    import pyarrow 
    PYARROW_INSTALLED = True
except ImportError:
    PYARROW_INSTALLED = False

try:
    from utils.load_parquet import load_parquet
    LOAD_PARQUET_AVAILABLE = True 
except ImportError:
    # This could happen if polars is installed but pyarrow isn't,
    # or if the function file itself has issues.
    LOAD_PARQUET_AVAILABLE = False
    def load_parquet(*args, **kwargs):
        raise ImportError("load_parquet or its deps (pyarrow) not available.")

# Skip all tests if function isn't available or pyarrow is missing
pytestmark = pytest.mark.skipif(not (LOAD_PARQUET_AVAILABLE and PYARROW_INSTALLED),
                                reason="Requires load_parquet function and pyarrow library")

class TestLoadParquet(unittest.TestCase):

    def setUp(self):
        """Set up a dummy Parquet file for testing using Polars."""
        self.test_dir = "test_data_parquet"
        self.test_file = os.path.join(self.test_dir, "test.parquet")
        os.makedirs(self.test_dir, exist_ok=True)
        data = {'col_int': [1, 2, 3], 'col_str': ['a', 'b', 'c']}
        df = pl.DataFrame(data)
        try:
            df.write_parquet(self.test_file)
        except ImportError:
            pytest.skip("Skipping setup, write_parquet requires pyarrow.")
        except Exception as e:
             pytest.fail(f"Setup failed during write_parquet: {e}")


    def tearDown(self):
        """Remove the dummy Parquet file and directory after testing."""
        try:
            if os.path.exists(self.test_file):
                os.remove(self.test_file)
            os.rmdir(self.test_dir)
        except OSError as e:
             print(f"Error during teardown: {e}")

    def test_basic_loading(self):
        """Test basic Parquet loading into Polars DataFrame."""
        df = load_parquet(self.test_file)
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape, (3, 2))
        self.assertListEqual(df.columns, ['col_int', 'col_str'])
        self.assertEqual(df['col_int'][0], 1)
        self.assertEqual(df['col_int'].dtype, pl.Int64)
        self.assertEqual(df['col_str'].dtype, pl.Utf8)

    def test_loading_with_kwargs(self):
        """Test loading with Polars kwargs (e.g., specific columns)."""
        df = load_parquet(self.test_file, columns=['col_str'])
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape, (3, 1))
        self.assertListEqual(df.columns, ['col_str'])
        self.assertEqual(df['col_str'][1], 'b')

    def test_file_not_found(self):
        """Test loading a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_parquet("non_existent_file.parquet")

# No need for explicit import error test if covered by skipif
#   def test_import_error_handled(self):
#       pass

# Note: Running this test file requires pytest and pyarrow
# Run with: `pytest tests/utils/test_load_parquet.py`
if __name__ == '__main__':
    # Fallback to unittest runner if pytest is not used, but skip logic won't apply
    if LOAD_PARQUET_AVAILABLE and PYARROW_INSTALLED:
        unittest.main()
    else:
        print("Skipping Parquet tests: load_parquet function or pyarrow not available.") 