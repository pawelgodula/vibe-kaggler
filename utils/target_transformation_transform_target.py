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
    if not target.dtype.is_numeric() and method != 'binning': # Binning output is int but input to reverse could be float preds
        raise TypeError("target must be a numeric Series for most reversal methods.")
    if method == 'binning' and (params is None or "edges" not in params):
         raise ValueError("Parameters dictionary with 'edges' is required for reversing 'binning'.")
        
    reversed_target: pl.Series
    
    if method == 'log':
        reversed_target = target.exp()
    elif method == 'log1p':
        reversed_target = target.exp() - 1 
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
            print("Warning: Original min and max were equal during scaling. Reversing to original min value.")
            reversed_target = pl.repeat(min_val, n=len(target), dtype=pl.Float64, eager=True).rename(target.name)
        else:
            reversed_target = (target * denominator) + min_val
            
    elif method == 'binning':
        # Reversing binning means mapping bin indices back to a representative value for that bin.
        # This is an approximation, often the mean of the bin edges or the midpoint.
        # The `target` here is assumed to be predicted bin indices (or probabilities for bins).
        # For simplicity, let's map to the midpoint of the bin edges.
        # This is a simplification; sophisticated reversal might use predicted probabilities per bin.
        
        edges = params["edges"]
        if not isinstance(edges, list) or len(edges) < 2:
            raise ValueError("'edges' in params must be a list of at least two numeric values.")

        # Ensure target is int if it represents bin indices
        if not target.dtype.is_integer():
            print(f"Warning: Reversing 'binning' but target dtype is {target.dtype}. Attempting to cast to Int32 for bin indices.")
            try:
                target = target.cast(pl.Int32)
            except Exception as e:
                raise TypeError(f"Failed to cast target to Int32 for binning reversal: {e}")

        # Calculate midpoints
        midpoints = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges)-1)]
        
        # Check if target indices are within valid range
        max_bin_index = len(midpoints) - 1
        if target.min() < 0 or target.max() > max_bin_index:
            print(f"Warning: Target bin indices ({target.min()}-{target.max()}) are outside expected range (0-{max_bin_index}). Clamping to valid range.")
            target = target.clip(0, max_bin_index)

        # Map bin indices to midpoints
        # Using apply for direct mapping based on index
        reversed_target = target.map_elements(lambda idx: midpoints[idx] if idx is not None and 0 <= idx < len(midpoints) else None, 
                                              return_dtype=pl.Float64)

    else:
        raise ValueError(f"Unsupported transformation method: {method}. Supported methods: {SUPPORTED_METHODS}")

    return reversed_target.rename(target.name)


if __name__ == '__main__':
    # --- Example Usage --- 
    print("Testing target transformation functions...")

    # Sample data
    train_target_data = pl.Series("target_train", [10, 20, 30, 40, 50, 1, 100, 5, 200])
    test_target_data = pl.Series("target_test", [15, 25, 35, 60, 0.5, 300])
    print(f"\nOriginal Train Target:\n{train_target_data.to_frame()}")
    print(f"Original Test Target:\n{test_target_data.to_frame()}")

    # --- Test Log Transform --- 
    print("\n--- Log Transform ---")
    try:
        tt_log, test_t_log, params_log = apply_target_transformation(train_target_data, 'log', test_target_data)
        print(f"Transformed Train (Log):\n{tt_log.to_frame()}")
        if test_t_log:
            print(f"Transformed Test (Log):\n{test_t_log.to_frame()}")
        rev_log = reverse_target_transformation(tt_log, 'log', params_log)
        print(f"Reversed Train (Log):\n{rev_log.to_frame()}")
        # Test with non-positive data for warning
        train_target_nonpos = pl.Series("target_train_np", [10, -2, 30])
        apply_target_transformation(train_target_nonpos, 'log') # Should raise ValueError
    except ValueError as e:
        print(f"Caught ValueError (Log): {e}")
    except Exception as e:
        print(f"Error (Log): {e}")
        
    # --- Test Log1p Transform --- 
    print("\n--- Log1p Transform ---")
    train_target_log1p_data = pl.Series("target_train_log1p", [0, 10, 20, 0, 50]) # Includes 0
    tt_log1p, test_t_log1p, params_log1p = apply_target_transformation(train_target_log1p_data, 'log1p', test_target_data)
    print(f"Transformed Train (Log1p):\n{tt_log1p.to_frame()}")
    if test_t_log1p:
        print(f"Transformed Test (Log1p):\n{test_t_log1p.to_frame()}")
    rev_log1p = reverse_target_transformation(tt_log1p, 'log1p', params_log1p)
    print(f"Reversed Train (Log1p):\n{rev_log1p.to_frame()}")

    # --- Test Standard Scaling --- 
    print("\n--- Standard Scaling ---")
    tt_std, test_t_std, params_std = apply_target_transformation(train_target_data, 'standard', test_target_data)
    print(f"Transformed Train (Std):\n{tt_std.to_frame()}")
    print(f"Params (Std): {params_std}")
    if test_t_std:
        print(f"Transformed Test (Std):\n{test_t_std.to_frame()}")
    rev_std = reverse_target_transformation(tt_std, 'standard', params_std)
    print(f"Reversed Train (Std):\n{rev_std.to_frame()}")

    # --- Test MinMax Scaling --- 
    print("\n--- MinMax Scaling ---")
    tt_mm, test_t_mm, params_mm = apply_target_transformation(train_target_data, 'minmax', test_target_data)
    print(f"Transformed Train (MinMax):\n{tt_mm.to_frame()}")
    print(f"Params (MinMax): {params_mm}")
    if test_t_mm:
        print(f"Transformed Test (MinMax):\n{test_t_mm.to_frame()}")
    rev_mm = reverse_target_transformation(tt_mm, 'minmax', params_mm)
    print(f"Reversed Train (MinMax):\n{rev_mm.to_frame()}")
    
    # --- Test Binning Transformation (Quantile) --- 
    print("\n--- Binning Transform (Quantile) ---")
    N_BINS = 4
    try:
        tt_bin, test_t_bin, params_bin = apply_target_transformation(
            train_target_data, 'binning', test_target_data, n_bins=N_BINS
        )
        print(f"Transformed Train (Binning - {N_BINS} bins):\n{tt_bin.to_frame()}")
        print(f"Params (Binning): {params_bin}")
        if test_t_bin is not None:
            print(f"Transformed Test (Binning - {N_BINS} bins):\n{test_t_bin.to_frame()}")
        
        # Reverse binning (maps bin indices to midpoints of original value bins)
        rev_bin = reverse_target_transformation(tt_bin, 'binning', params_bin)
        print(f"Reversed Train (Binning - midpoints):\n{rev_bin.to_frame()}")
        
        if test_t_bin is not None:
            rev_test_bin = reverse_target_transformation(test_t_bin, 'binning', params_bin)
            print(f"Reversed Test (Binning - midpoints):\n{rev_test_bin.to_frame()}")
            
        # Test binning with a constant series (should fail or warn)
        print("\nTesting binning with constant series:")
        constant_target = pl.Series("const_target", [5.0, 5.0, 5.0, 5.0])
        try:
            apply_target_transformation(constant_target, 'binning', n_bins=2)
        except ValueError as e_const:
            print(f"Caught expected ValueError for constant binning: {e_const}")
            
    except ValueError as e:
        print(f"ValueError during binning test: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during binning test: {e}")

    # --- Test with nulls in target ---
    print("\n--- Testing with Nulls in Target ---")
    train_target_nulls = pl.Series("target_nulls", [10, None, 30, None, 50, 1, 100, None, 200])
    try:
        tt_std_null, _, params_std_null = apply_target_transformation(train_target_nulls, 'standard')
        print(f"Transformed Train (Std with Nulls):\n{tt_std_null.to_frame()}") # Expect nulls to remain null
        print(f"Params (Std with Nulls): {params_std_null}")
        rev_std_null = reverse_target_transformation(tt_std_null, 'standard', params_std_null)
        print(f"Reversed Train (Std with Nulls):\n{rev_std_null.to_frame()}")
    except Exception as e:
        print(f"Error with nulls: {e}") 