# Functions for applying and reversing target variable transformations for regression.

"""
Extended Description:
Provides functions to apply common target transformations (log, log1p, standard scaling,
min-max scaling) and reverse them. This is often useful in regression tasks where
models perform better with targets following a specific distribution or range.

The `apply_target_transformation` function calculates necessary parameters (mean, std,
min, max) only from the training data to prevent leakage and returns these parameters
along with the transformed target(s).
The `reverse_target_transformation` function uses these parameters to revert the
transformed target back to its original scale.
"""

import polars as pl
import numpy as np
from typing import Optional, Tuple, Literal, Dict, Any, Union

SUPPORTED_METHODS = Literal['log', 'log1p', 'standard', 'minmax', 'binning']

def apply_target_transformation(
    train_target: pl.Series,
    method: SUPPORTED_METHODS,
    test_target: Optional[pl.Series] = None,
    n_bins: Optional[int] = None
) -> Tuple[pl.Series, Optional[pl.Series], Optional[Dict[str, Any]]]:
    """Applies a specified transformation to the target variable.

    Calculates necessary parameters (mean, std, min, max, bin edges) from train_target only.

    Args:
        train_target (pl.Series): The target variable Series from the training data.
        method (Literal['log', 'log1p', 'standard', 'minmax']): Transformation method.
        test_target (Optional[pl.Series], optional): The target variable Series from the
            test/validation data to transform using parameters derived from train_target.
            Defaults to None.
        n_bins (Optional[int], optional): Number of bins to use when method is 'binning'.
            Required if method='binning'. Defaults to None.

    Returns:
        Tuple[pl.Series, Optional[pl.Series], Optional[Dict[str, float]]]:
            - Transformed training target Series.
            - Transformed test target Series (or None if test_target was None).
            - Dictionary containing parameters needed for reversal (mean, std, min, max)
              or for understanding the transformation (edges for binning).
              None if no parameters are needed (log, log1p).

    Raises:
        ValueError: If an invalid method is specified or if log is used on non-positive data.
        TypeError: If input series are not numeric.
    """
    if not train_target.dtype.is_numeric():
        raise TypeError("train_target must be a numeric Series.")
    if test_target is not None and not test_target.dtype.is_numeric():
        raise TypeError("test_target must be a numeric Series.")

    params: Optional[Dict[str, Any]] = None
    transformed_train_target: pl.Series
    transformed_test_target: Optional[pl.Series] = None

    if method == 'log':
        if (train_target <= 0).any(): # Check for non-positive values
             raise ValueError("'log' method requires all target values to be positive.")
        transformed_train_target = train_target.log()
        if test_target is not None:
             if (test_target <= 0).any():
                  print("Warning: test_target contains non-positive values for 'log' transform. Result will contain NaNs/Infs.")
             transformed_test_target = test_target.log()
    
    elif method == 'log1p':
        transformed_train_target = train_target.log1p()
        if test_target is not None:
            transformed_test_target = test_target.log1p()
            
    elif method == 'standard':
        mean = train_target.mean()
        std = train_target.std()
        if std is None or np.isnan(std): # Handle all-null input
            std = 0.0 
            mean = 0.0 # Or some other default
            print("Warning: Target std dev is null/NaN. Check input data.")
            
        params = {"mean": mean, "std": std}
        
        if std == 0:
             print("Warning: Target standard deviation is zero. Standard scaling will result in zeros (or NaNs if mean is also zero).")
             # Avoid division by zero - result is 0 if mean is non-zero, NaN otherwise? Polars might handle.
             # Let's explicitly return 0.0 for this case.
             transformed_train_target = pl.repeat(0.0, n=len(train_target), dtype=pl.Float64, eager=True).rename(train_target.name)
             if test_target is not None:
                 transformed_test_target = pl.repeat(0.0, n=len(test_target), dtype=pl.Float64, eager=True).rename(test_target.name)
        else:
            transformed_train_target = (train_target - mean) / std
            if test_target is not None:
                transformed_test_target = (test_target - mean) / std
                
    elif method == 'minmax':
        min_val = train_target.min()
        max_val = train_target.max()
        if min_val is None or max_val is None: # Handle all-null input
             min_val = 0.0
             max_val = 1.0 # Or some defaults
             print("Warning: Target min/max is null. Check input data.")
             
        params = {"min": min_val, "max": max_val}
        denominator = max_val - min_val

        if denominator == 0:
             print("Warning: Target max equals min. MinMax scaling will result in zeros.")
             # All values are the same, scale to 0
             transformed_train_target = pl.repeat(0.0, n=len(train_target), dtype=pl.Float64, eager=True).rename(train_target.name)
             if test_target is not None:
                 transformed_test_target = pl.repeat(0.0, n=len(test_target), dtype=pl.Float64, eager=True).rename(test_target.name)
        else:
            transformed_train_target = (train_target - min_val) / denominator
            if test_target is not None:
                transformed_test_target = (test_target - min_val) / denominator
                
    elif method == 'binning':
        if n_bins is None or not isinstance(n_bins, int) or n_bins <= 0:
            raise ValueError("'n_bins' must be a positive integer when method is 'binning'.")
            
        # Use quantiles to find edges based on training data
        try:
            quantiles_to_calculate = np.linspace(0, 1, n_bins + 1)
            # Calculate quantiles one by one
            edges = [train_target.quantile(q) for q in quantiles_to_calculate]
            # Filter out potential None values if target is all null or quantile calc fails
            edges = [e for e in edges if e is not None]

            # Ensure edges are unique (handling cases where quantiles might be identical)
            unique_edges = sorted(list(set(edges)))
            if len(unique_edges) < 2:
                 raise ValueError(f"Could not determine enough unique quantile bins ({len(unique_edges)-1} found) for n_bins={n_bins}. Input data might be constant or have too few unique values.")
            
            # Use unique edges directly as breaks for cut
            final_edges = unique_edges 
            # Update n_bins based on the actual number of bins created by unique edges
            actual_n_bins = len(final_edges) - 1 
            # Store the actual finite edges used for potential reference/debugging
            params = {"edges": final_edges, "n_bins": actual_n_bins}
             
            # Apply cut using the derived edges
            # Use left_closed=True to get intervals like [edge1, edge2)
            # Use Series.cut() method
            transformed_train_target_cat = train_target.cut(breaks=final_edges, left_closed=True)

            # Create mapping from category label to integer index
            # Need to get the categories generated by cut (e.g., '(-inf, 1.0]') and map them
            generated_labels = transformed_train_target_cat.unique().sort().to_list()
            # Filter out potential nulls if any
            generated_labels = [label for label in generated_labels if label is not None]
            # Ensure the number of generated labels matches the expected number of bins
            if len(generated_labels) != actual_n_bins:
                 print(f"Warning: Expected {actual_n_bins} bins but Polars cut generated {len(generated_labels)} labels: {generated_labels}")
                 # Handle this discrepancy? For now, proceed with generated labels
                 # This might happen if data falls outside the outermost edges slightly?
            
            label_map = {label: i for i, label in enumerate(generated_labels)}
            
            # Use pl.when/then with pl.coalesce for mapping
            train_cat_series_name = transformed_train_target_cat.name if transformed_train_target_cat.name else "target_cat"
            transformed_train_target_cat = transformed_train_target_cat.rename(train_cat_series_name)

            # Create a list of mapping expressions
            mapping_expr_list = [
                pl.when(pl.col(train_cat_series_name) == label).then(pl.lit(index, dtype=pl.UInt32))
                for label, index in label_map.items()
            ]
            
            # Apply the mapping expression using coalesce
            # Coalesce takes the first non-null result
            transformed_train_target = (
                transformed_train_target_cat.to_frame()
                .select(pl.coalesce(*mapping_expr_list).alias(train_cat_series_name))
                .to_series()
            )

            if test_target is not None:
                transformed_test_target_cat = test_target.cut(breaks=final_edges, left_closed=True)
                
                test_cat_series_name = transformed_test_target_cat.name if transformed_test_target_cat.name else "target_cat_test"
                transformed_test_target_cat = transformed_test_target_cat.rename(test_cat_series_name)
                
                # Use the same label_map derived from training data
                mapping_expr_list_test = [
                    pl.when(pl.col(test_cat_series_name) == label).then(pl.lit(index, dtype=pl.UInt32))
                    for label, index in label_map.items()
                ]
                
                # Apply the mapping expression to test data using coalesce
                transformed_test_target = (
                    transformed_test_target_cat.to_frame()
                    .select(pl.coalesce(*mapping_expr_list_test).alias(test_cat_series_name))
                    .to_series()
                )
                
        except Exception as e:
             print(f"Error during binning: {e}")
             # Reraise or handle more gracefully?
             raise ValueError(f"Failed to apply binning with n_bins={n_bins}. Check input data distribution.") from e

    else:
        raise ValueError(f"Unsupported transformation method: {method}. Supported methods: {SUPPORTED_METHODS}")

    return transformed_train_target, transformed_test_target, params

def reverse_target_transformation(
    target: pl.Series,
    method: SUPPORTED_METHODS,
    params: Optional[Dict[str, Any]] = None
) -> pl.Series:
    """Reverses a previously applied target transformation.

    Args:
        target (pl.Series): The transformed target Series.
        method (Literal['log', 'log1p', 'standard', 'minmax']): The transformation method 
            that was originally applied.
        params (Optional[Dict[str, Any]], optional): Dictionary containing parameters
            used during the original transformation (mean, std, min, max, edges).
            Required for 'standard' and 'minmax'. Defaults to None.

    Returns:
        pl.Series: The target Series reversed to its original scale.
        
    Raises:
        ValueError: If an invalid method is specified or required params are missing.
        TypeError: If input series is not numeric or params have wrong types.
    """
    if not target.dtype.is_numeric():
        raise TypeError("target must be a numeric Series.")
        
    reversed_target: pl.Series
    
    if method == 'log':
        reversed_target = target.exp()
    elif method == 'log1p':
        reversed_target = target.exp() - 1 # More direct than expm1 if target is Series
        # reversed_target = target.exp().map_elements(np.expm1) # Alternative using numpy
    elif method == 'standard':
        if params is None or "mean" not in params or "std" not in params:
            raise ValueError("Parameters dictionary with 'mean' and 'std' is required for reversing 'standard' scaling.")
        mean = params["mean"]
        std = params["std"]
        if not isinstance(mean, (int, float)) or not isinstance(std, (int, float)):
            raise TypeError("'mean' and 'std' in params must be numeric.")
            
        if std == 0:
             print("Warning: Original std dev was zero during scaling. Reversing may not be meaningful. Returning original mean.")
             reversed_target = pl.repeat(mean, n=len(target), dtype=pl.Float64, eager=True).rename(target.name)
        else:
             reversed_target = (target * std) + mean
             
    elif method == 'minmax':
        if params is None or "min" not in params or "max" not in params:
             raise ValueError("Parameters dictionary with 'min' and 'max' is required for reversing 'minmax' scaling.")
        min_val = params["min"]
        max_val = params["max"]
        if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
            raise TypeError("'min' and 'max' in params must be numeric.")
            
        denominator = max_val - min_val
        if denominator == 0:
             print("Warning: Original max equaled min during scaling. Reversing may not be meaningful. Returning original min/max value.")
             reversed_target = pl.repeat(min_val, n=len(target), dtype=pl.Float64, eager=True).rename(target.name)
        else:
             reversed_target = (target * denominator) + min_val
             
    elif method == 'binning':
        # Reversal is not possible/meaningful for binning
        print("Warning: Reversal requested for 'binning' method. Returning the input (bin indices/labels) unchanged.")
        reversed_target = target # Return the bin indices themselves

    else:
        raise ValueError(f"Unsupported transformation method: {method}. Supported methods: {SUPPORTED_METHODS}")
        
    return reversed_target 