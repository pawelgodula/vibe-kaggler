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


if __name__ == '__main__':
    # Example Usage
    print("Testing average_predictions function...")

    # Example 1: List of NumPy arrays
    preds1_np = np.array([1, 2, 3, 4, 5])
    preds2_np = np.array([5, 4, 3, 2, 1])
    preds3_np = np.array([2, 3, 4, 5, 6])
    avg_np = average_predictions([preds1_np, preds2_np, preds3_np])
    print(f"\nNumPy array inputs:\n  Preds1: {preds1_np}\n  Preds2: {preds2_np}\n  Preds3: {preds3_np}")
    print(f"  Average: {avg_np}") # Expected: [ (1+5+2)/3, (2+4+3)/3, ... ] = [2.66, 3, 4, 3.66, 4]

    # Example 2: List of Polars Series
    preds1_pl = pl.Series("p1", [10, 20, 30])
    preds2_pl = pl.Series("p2", [12, 18, 33])
    avg_pl = average_predictions([preds1_pl, preds2_pl])
    print(f"\nPolars Series inputs:\n  Preds1: {preds1_pl}\n  Preds2: {preds2_pl}")
    print(f"  Average: {avg_pl}") # Expected: [11, 19, 31.5]

    # Example 3: Mixed list
    avg_mixed = average_predictions([preds1_np[:3], preds2_pl]) # Use first 3 elements of preds1_np
    print(f"\nMixed inputs (NumPy and Polars Series):\n  Preds1 (part): {preds1_np[:3]}\n  Preds2: {preds2_pl}")
    print(f"  Average: {avg_mixed}") # Expected: [ (1+12)/2, (2+18)/2, (3+33)/2 ] = [6.5, 10, 18]
    
    # Example 4: 2D arrays (e.g., multiclass probabilities)
    preds_2d_1 = np.array([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]])
    preds_2d_2 = np.array([[0.3, 0.7], [0.6, 0.4], [0.4, 0.6]])
    avg_2d = average_predictions([preds_2d_1, preds_2d_2])
    print(f"\n2D NumPy array inputs:\n  Preds1: \n{preds_2d_1}\n  Preds2: \n{preds_2d_2}")
    print(f"  Average: \n{avg_2d}") 
    # Expected: [[0.2, 0.8], [0.7, 0.3], [0.45, 0.55]]

    # Error Cases
    print("\n--- Error Cases ---")
    try:
        average_predictions([])
    except ValueError as e:
        print(f"Caught expected error (empty list): {e}")

    try:
        preds_diff_shape1 = np.array([1, 2, 3])
        preds_diff_shape2 = np.array([4, 5])
        average_predictions([preds_diff_shape1, preds_diff_shape2])
    except ValueError as e:
        print(f"Caught expected error (different shapes): {e}")
        
    try:
        average_predictions([preds1_np, "not_an_array"])
    except TypeError as e:
        print(f"Caught expected error (invalid type): {e}") 