# Saves a Polars DataFrame to a CSV file.

"""
Extended Description:
This function provides a wrapper around polars.DataFrame.write_csv to save data.
It accepts the DataFrame, the target file path, and any additional keyword arguments
accepted by polars.DataFrame.write_csv (e.g., separator, include_header).
Ensures consistent data saving across the project using Polars.
Note: Polars write_csv does not include an index by default, unlike pandas.
"""

import polars as pl
from typing import Any

def save_csv(df: pl.DataFrame, file_path: str, **kwargs: Any) -> None:
    """Saves a Polars DataFrame to a CSV file.

    Args:
        df (pl.DataFrame): The DataFrame to save.
        file_path (str): The path where the CSV file will be saved.
        **kwargs (Any): Additional keyword arguments passed directly to
                        polars.DataFrame.write_csv.
                        Common args: separator (default ","), include_header (default True).
    """
    try:
        # Polars doesn't write index by default, so no 'index' arg needed like pandas
        df.write_csv(file_path, **kwargs)
        # Consider adding logging here
    except Exception as e:
        # Consider logging the error
        print(f"Error saving Polars DataFrame to CSV file {file_path}: {e}")
        raise 