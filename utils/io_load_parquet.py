# Loads data from a Parquet file into a Polars DataFrame.

"""
Extended Description:
This function provides a wrapper around polars.read_parquet to load data.
It accepts the file path and any additional keyword arguments accepted by polars.read_parquet.
Polars uses pyarrow by default, which must be installed.
Ensures consistent data loading for Parquet files across the project.
"""

import polars as pl
from typing import Any

def load_parquet(file_path: str, **kwargs: Any) -> pl.DataFrame:
    """Loads data from a Parquet file into a Polars DataFrame.

    Requires 'pyarrow' to be installed.

    Args:
        file_path (str): The path to the Parquet file.
        **kwargs (Any): Additional keyword arguments passed directly to polars.read_parquet.
                        Common args: columns, n_rows.

    Returns:
        pl.DataFrame: The loaded data as a Polars DataFrame.
    """
    try:
        df = pl.read_parquet(file_path, **kwargs)
        # Consider adding logging here
        return df
    # Polars read_parquet raises specific exceptions on parse/IO errors,
    # often ComputeError or exceptions from the underlying engine (pyarrow).
    # FileNotFoundError is still relevant.
    except FileNotFoundError:
        # Consider logging the error
        print(f"Error: File not found at {file_path}")
        raise
    except ImportError:
         # This might occur if polars is installed but pyarrow isn't
         print("Error: Loading Parquet files with Polars requires 'pyarrow'.")
         print("Please install it (e.g., pip install pyarrow)")
         raise
    except Exception as e:
        # Catching other potential Polars/PyArrow errors
        print(f"Error loading Parquet file {file_path} with Polars: {e}")
        raise 