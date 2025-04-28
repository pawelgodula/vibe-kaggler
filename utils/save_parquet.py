# Saves a Polars DataFrame to a Parquet file.

"""
Extended Description:
This function provides a wrapper around polars.DataFrame.write_parquet to save data.
It accepts the DataFrame, the target file path, and any additional keyword arguments
accepted by polars.DataFrame.write_parquet (e.g., compression, statistics).
Requires 'pyarrow' to be installed.
Ensures consistent data saving for Parquet files across the project.
"""

import polars as pl
from typing import Any

def save_parquet(df: pl.DataFrame, file_path: str, **kwargs: Any) -> None:
    """Saves a Polars DataFrame to a Parquet file.

    Requires 'pyarrow' to be installed.

    Args:
        df (pl.DataFrame): The DataFrame to save.
        file_path (str): The path where the Parquet file will be saved.
        **kwargs (Any): Additional keyword arguments passed directly to
                        polars.DataFrame.write_parquet.
                        Common args: compression ('snappy', 'gzip', etc.),
                        statistics (bool, default False).
    """
    try:
        df.write_parquet(file_path, **kwargs)
        # Consider adding logging here
    except ImportError:
        # This might occur if polars is installed but pyarrow isn't
        print("Error: Saving Parquet files with Polars requires 'pyarrow'.")
        print("Please install it (e.g., pip install pyarrow)")
        raise
    except Exception as e:
        # Catching other potential Polars/PyArrow errors
        print(f"Error saving Polars DataFrame to Parquet file {file_path}: {e}")
        raise 