# Calculates the simple average of a list of prediction arrays or series.

"""
Extended Description:
Takes a list containing multiple prediction vectors (as NumPy arrays or Polars Series)
and computes their element-wise average. It ensures all inputs have the same
shape before averaging.
"""

import numpy as np
import polars as pl
from typing import List, Union

def average_predictions(
    prediction_list: List[Union[np.ndarray, pl.Series]]
) -> np.ndarray:
    """Calculates the simple element-wise average of predictions.

    Args:
        prediction_list (List[Union[np.ndarray, pl.Series]]):
            A list containing NumPy arrays and/or Polars Series.
            All elements must have the same shape.

    Returns:
        np.ndarray: A NumPy array containing the element-wise average.

    Raises:
        ValueError: If the prediction_list is empty.
        ValueError: If the predictions in the list have different shapes.
    """
    if not prediction_list:
        raise ValueError("prediction_list cannot be empty.")

    # Convert all to NumPy arrays and check shapes
    np_predictions = []
    first_shape = None
    for i, pred in enumerate(prediction_list):
        if isinstance(pred, pl.Series):
            np_pred = pred.to_numpy()
        elif isinstance(pred, np.ndarray):
            np_pred = pred
        else:
            raise TypeError(
                f"Element {i} in prediction_list is not a NumPy array or Polars Series, got {type(pred)}"
            )
        
        if i == 0:
            first_shape = np_pred.shape
        elif np_pred.shape != first_shape:
            raise ValueError(
                f"Predictions have inconsistent shapes. Expected {first_shape}, but got {np_pred.shape} at index {i}."
            )
        
        np_predictions.append(np_pred)

    # Calculate the average
    # Stacking arrays and then taking the mean along the new axis
    try:
        average_preds = np.mean(np.stack(np_predictions, axis=0), axis=0)
    except Exception as e:
        # Handle potential errors during stacking or mean calculation
        print(f"Error calculating average: {e}")
        raise

    return average_preds 