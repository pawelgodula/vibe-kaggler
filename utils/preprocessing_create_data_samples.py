# Function to create sample files from raw data files.

"""
Extended Description:
Scans a directory containing raw data files (e.g., CSV, Parquet), reads the 
first N rows from each file, and saves these smaller DataFrames as new files 
in a specified samples directory. This is useful for creating small data snippets 
for quick inspection, testing, or providing context to LLMs without loading 
large datasets.
"""

import polars as pl
import os
from pathlib import Path
from typing import List, Optional

SUPPORTED_EXTENSIONS = [".csv", ".parquet"]

def create_data_samples(
    raw_data_dir: str,
    samples_dir: str,
    n_rows: int = 10,
    include_extensions: Optional[List[str]] = None
) -> None:
    """Creates sample files by reading the first N rows from files in a directory.

    Args:
        raw_data_dir (str): Path to the directory containing the raw data files.
        samples_dir (str): Path to the directory where sample files will be saved. 
                           It will be created if it doesn't exist.
        n_rows (int, optional): Number of rows to include in each sample file. 
            Defaults to 10.
        include_extensions (Optional[List[str]], optional): List of file extensions 
            (including the dot, e.g., '.csv') to process. If None, processes 
            files with extensions in SUPPORTED_EXTENSIONS. Defaults to None.

    Raises:
        FileNotFoundError: If raw_data_dir does not exist or is not a directory.
        ValueError: If n_rows is not positive.
    """
    raw_dir_path = Path(raw_data_dir)
    samples_dir_path = Path(samples_dir)

    if not raw_dir_path.is_dir():
        raise FileNotFoundError(f"Raw data directory not found or is not a directory: {raw_data_dir}")
    if n_rows <= 0:
         raise ValueError("n_rows must be a positive integer.")

    # Determine which extensions to process
    allowed_extensions = include_extensions if include_extensions is not None else SUPPORTED_EXTENSIONS
    allowed_extensions = [ext.lower() for ext in allowed_extensions]

    # Create samples directory if it doesn't exist
    samples_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured samples directory exists: {samples_dir_path}")

    print(f"Scanning '{raw_dir_path}' for files with extensions: {allowed_extensions}...")
    processed_count = 0
    skipped_count = 0

    for item in raw_dir_path.iterdir():
        if item.is_file():
            file_ext = item.suffix.lower()
            if file_ext not in allowed_extensions:
                # print(f"  Skipping file with unsupported extension: {item.name}")
                skipped_count += 1
                continue

            print(f"  Processing file: {item.name}...")
            output_path = samples_dir_path / item.name
            df_sample: Optional[pl.DataFrame] = None

            try:
                # Read the first n_rows based on file type
                if file_ext == '.csv':
                    df_sample = pl.read_csv(item, n_rows=n_rows, infer_schema_length=n_rows)
                elif file_ext == '.parquet':
                    df_sample = pl.read_parquet(item, n_rows=n_rows)
                # Add other supported types here if needed
                
                # Write the sample file
                if df_sample is not None:
                    if file_ext == '.csv':
                        df_sample.write_csv(output_path)
                    elif file_ext == '.parquet':
                        df_sample.write_parquet(output_path)
                    # Add other supported write types here
                    print(f"    Successfully created sample: {output_path}")
                    processed_count += 1
                else:
                     print(f"    Warning: Could not read or determine type for {item.name}")
                     skipped_count += 1
                     
            except Exception as e:
                print(f"    Error processing file {item.name}: {e}")
                skipped_count += 1
                # Optionally, try reading CSV with different parameters if first fails?
                # For now, just skip on error.
                
    print(f"\nFinished creating samples.")
    print(f"  Processed files: {processed_count}")
    print(f"  Skipped files: {skipped_count}") 