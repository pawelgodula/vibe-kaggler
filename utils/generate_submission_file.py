# Generates a Kaggle submission file in CSV format using Polars.

"""
Extended Description:
This function takes prediction IDs and corresponding predicted values (as Polars Series
or numpy arrays) and creates a Polars DataFrame with specified column names.
It then saves this DataFrame to a CSV file using the `save_csv` utility function,
which handles Polars CSV writing without an index.
"""

import polars as pl
import numpy as np
from typing import Union
from .save_csv import save_csv # Use Polars version of save_csv

def generate_submission_file(
    ids: Union[pl.Series, np.ndarray],
    predictions: Union[pl.Series, np.ndarray],
    id_col_name: str,
    target_col_name: str,
    file_path: str
) -> None:
    """Creates and saves a Kaggle submission CSV file using Polars.

    Args:
        ids (Union[pl.Series, np.ndarray]): The IDs for each prediction (e.g., row ID).
        predictions (Union[pl.Series, np.ndarray]): The predicted values.
        id_col_name (str): The name for the ID column in the output CSV file.
        target_col_name (str): The name for the prediction column in the output CSV file.
        file_path (str): The full path where the submission CSV file will be saved.

    Raises:
        ValueError: If the lengths of ids and predictions do not match.
        Exception: Propagates exceptions from DataFrame creation or CSV saving.
    """
    if len(ids) != len(predictions):
        raise ValueError(
            f"Length mismatch: ids length ({len(ids)}) != predictions length ({len(predictions)})"
        )

    try:
        # Create Polars DataFrame directly
        submission_df = pl.DataFrame({
            id_col_name: ids,
            target_col_name: predictions
        })
    except Exception as e:
        print(f"Error creating Polars submission DataFrame: {e}")
        raise

    try:
        # Use the Polars-based save_csv utility
        save_csv(submission_df, file_path) # index=False is default Polars behavior
        print(f"Submission file saved successfully to: {file_path}")
    except Exception as e:
        # save_csv utility should handle its own errors/logging
        print(f"Failed to save submission file using save_csv utility.")
        raise 