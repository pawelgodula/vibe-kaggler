# Loads data from a CSV file into a Polars DataFrame.

"""
Extended Description:
This function provides a wrapper around polars.read_csv to load data.
It accepts the file path and any additional keyword arguments accepted by polars.read_csv.
Ensures consistent data loading across the project using Polars.
"""

import polars as pl
from typing import Any

def load_csv(file_path: str, **kwargs: Any) -> pl.DataFrame:
    """Loads data from a CSV file into a Polars DataFrame.

    Args:
        file_path (str): The path to the CSV file.
        **kwargs (Any): Additional keyword arguments passed directly to polars.read_csv.

    Returns:
        pl.DataFrame: The loaded data as a Polars DataFrame.
    """
    try:
        df = pl.read_csv(file_path, **kwargs)
        # Consider adding logging here
        return df
    except FileNotFoundError:
        # Consider logging the error
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        # Consider logging the error
        print(f"Error loading CSV file {file_path} with Polars: {e}")
        raise 