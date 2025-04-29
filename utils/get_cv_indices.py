# Generates train/validation row indices for cross-validation folds.

"""
Extended Description:
Takes a Polars DataFrame, a scikit-learn cross-validation splitter instance,
and optional group labels (for GroupKFold etc.). It uses the splitter's `split`
method to generate pairs of numpy arrays containing train and validation row indices
for each fold defined by the splitter.
"""

import polars as pl
import numpy as np
from sklearn.model_selection import BaseCrossValidator
from typing import List, Tuple, Optional

def get_cv_indices(
    df: pl.DataFrame,
    cv_splitter: BaseCrossValidator,
    target_col: Optional[str] = None, # Needed for Stratified splits
    groups: Optional[pl.Series] = None # Needed for Group splits
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generates train/validation row indices for each CV fold.

    Args:
        df (pl.DataFrame): The DataFrame to split. Only indices are used if target/groups are not needed.
        cv_splitter (BaseCrossValidator): An initialized scikit-learn cross-validation
                                          splitter instance (e.g., KFold(), StratifiedKFold()).
        target_col (Optional[str], optional): The name of the target column in `df`.
                                              Required for stratified splits (e.g., `StratifiedKFold`).
                                              Defaults to None.
        groups (Optional[pl.Series], optional): A Polars Series containing group labels for each row.
                                                Required for group-based splits (e.g., `GroupKFold`, `GroupShuffleSplit`).
                                                Must have the same length as `df`.
                                                Defaults to None.

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: A list where each element is a tuple
                                             containing two numpy arrays: the training
                                             indices and the validation indices for a fold.

    Raises:
        ValueError: If `target_col` is required by the splitter but not provided.
        ValueError: If `groups` are required by the splitter but not provided or have incorrect length.
        TypeError: If `cv_splitter` is not a valid scikit-learn splitter.
    """
    if not isinstance(cv_splitter, BaseCrossValidator):
        raise TypeError("cv_splitter must be an instance of sklearn.model_selection.BaseCrossValidator")

    n_rows = len(df)
    indices = np.arange(n_rows)

    # Prepare inputs for splitter.split()
    X = indices # Usually sufficient, unless splitter needs features
    y = None
    groups_np = None

    # Check arguments based on splitter type
    splitter_name = cv_splitter.__class__.__name__
    requires_target = 'Stratified' in splitter_name
    requires_groups = 'Group' in splitter_name

    if requires_target:
        if target_col is None:
             raise ValueError(f"{splitter_name} requires 'target_col' to be provided.")
        if target_col not in df.columns:
             raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
        try:
             y = df[target_col].to_numpy()
        except Exception as e:
             raise TypeError(f"Could not convert target column '{target_col}' to NumPy: {e}")

    if requires_groups:
        if groups is None:
            raise ValueError(f"{splitter_name} requires 'groups' to be provided.")
        if not isinstance(groups, pl.Series):
             raise TypeError("'groups' must be a Polars Series.")
        if len(groups) != n_rows:
            raise ValueError(f"Length of groups ({len(groups)}) must match DataFrame length ({n_rows}).")
        try:
            groups_np = groups.to_numpy()
        except Exception as e:
             raise TypeError(f"Could not convert groups Series to NumPy: {e}")

    # Generate folds - pass only necessary arguments
    split_args = {'X': X}
    if requires_target:
        split_args['y'] = y
    if requires_groups:
        split_args['groups'] = groups_np
        
    try:
        # Use **split_args to pass only the relevant arguments
        fold_indices = list(cv_splitter.split(**split_args))
    except Exception as e:
        # Catch potential errors during split generation (e.g., wrong args for specific splitter)
        print(f"Error calling cv_splitter.split with args {split_args.keys()}: {e}")
        # Consider checking specific exception types if needed
        raise ValueError(f"Failed to generate splits with {splitter_name}. Check target/groups arguments.") from e

    return fold_indices 