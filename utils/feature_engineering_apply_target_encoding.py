# Applies Target Encoding to categorical features using cross-validation.

"""
Extended Description:
This function implements Target Encoding, a technique where categorical features
are replaced with a numerical value derived from the target variable.
To prevent target leakage, the encoding for the training data is calculated
using an out-of-fold (OOF) strategy based on provided cross-validation indices.
The encoding for the test data is calculated using statistics from the entire
training set.

It supports different aggregation statistics (mean, median, percentile) and includes
smoothing to regularize the encoding, especially for categories with few observations.
The smoothing blends the category-specific statistic with a global prior statistic.
"""

import polars as pl
import numpy as np
from typing import List, Tuple, Optional, Literal

def _calculate_statistic(series: pl.Series, agg_stat: str, percentile: Optional[float] = None) -> float:
    """Helper to calculate a single statistic from a Polars Series."""
    if series.is_empty() or series.null_count() == len(series):
        return np.nan # Return NaN if series is empty or all null
    
    if agg_stat == 'mean':
        return series.mean()
    elif agg_stat == 'median':
        return series.median()
    elif agg_stat == 'percentile':
        if percentile is None:
            raise ValueError("percentile value must be provided for agg_stat='percentile'")
        # Polars quantile takes 0.0 to 1.0
        return series.quantile(percentile / 100.0, interpolation='linear') 
    else:
        raise ValueError(f"Unsupported agg_stat: {agg_stat}")

def apply_target_encoding(
    train_df: pl.DataFrame,
    features: List[str],
    target_col: str,
    cv_indices: List[Tuple[np.ndarray, np.ndarray]],
    test_df: Optional[pl.DataFrame] = None,
    agg_stat: Literal['mean', 'median', 'percentile'] = 'mean',
    percentile: Optional[float] = None, # Required if agg_stat is 'percentile'
    smoothing: float = 10.0,
    use_smoothing: bool = True, # New parameter
    new_col_suffix: str = '_te'
) -> Tuple[pl.DataFrame, Optional[pl.DataFrame]]:
    """Applies target encoding using an out-of-fold strategy for training data.

    Args:
        train_df (pl.DataFrame): Training DataFrame containing features and target.
        features (List[str]): List of categorical column names to encode.
        target_col (str): Name of the target column.
        cv_indices (List[Tuple[np.ndarray, np.ndarray]]): List of (train_idx, valid_idx) tuples
                                                          defining the cross-validation folds.
        test_df (Optional[pl.DataFrame], optional): Test DataFrame to encode. Defaults to None.
        agg_stat (Literal['mean', 'median', 'percentile'], optional): Aggregation statistic
            to use for encoding. Defaults to 'mean'.
        percentile (Optional[float], optional): Percentile value (0-100) if agg_stat is 'percentile'.
                                              Defaults to None.
        smoothing (float, optional): Smoothing factor (weight of the prior). Higher values
                                  mean more smoothing towards the global statistic.
                                  Defaults to 10.0.
        use_smoothing (bool, optional): Whether to apply smoothing. If False, the raw
                                       category statistic is used. Defaults to True.
        new_col_suffix (str, optional): Suffix to append to the original feature names
                                      to create the new encoded column names.
                                      Defaults to '_te'.

    Returns:
        Tuple[pl.DataFrame, Optional[pl.DataFrame]]:
            - Training DataFrame with added target-encoded columns.
            - Test DataFrame with added target-encoded columns (or None if test_df was None).

    Raises:
        ValueError: If agg_stat is invalid or percentile is missing when needed.
        pl.exceptions.ColumnNotFoundError: If feature or target columns are missing.
    """
    if agg_stat == 'percentile' and percentile is None:
        raise ValueError("percentile must be provided when agg_stat='percentile'.")
    if agg_stat not in ['mean', 'median', 'percentile']:
        raise ValueError("agg_stat must be one of 'mean', 'median', 'percentile'.")

    train_df_out = train_df.clone()
    test_df_out = test_df.clone() if test_df is not None else None

    # --- Add original index to ensure correct joining after OOF --- 
    original_index_col = "_original_idx" 
    # Check if column name already exists to avoid collision
    temp_original_index_col = original_index_col
    idx_counter = 0
    while temp_original_index_col in train_df_out.columns:
        idx_counter += 1
        temp_original_index_col = f"{original_index_col}_{idx_counter}"
    original_index_col = temp_original_index_col
    train_df_out = train_df_out.with_row_index(original_index_col)
    # was_index_added = True # Flag to know if we should drop it later

    # --- Validate Columns ---
    required_train_cols = set(features) | {target_col}
    missing_train = required_train_cols - set(train_df.columns)
    if missing_train:
        raise pl.exceptions.ColumnNotFoundError(f"Features missing in train_df: {missing_train}")
    if test_df is not None:
        missing_test = set(features) - set(test_df.columns)
        if missing_test:
            raise pl.exceptions.ColumnNotFoundError(f"Features missing in test_df: {missing_test}")

    # --- Calculate Global Prior (from full training data) --- 
    global_prior_stat = _calculate_statistic(train_df[target_col], agg_stat, percentile)
    if np.isnan(global_prior_stat):
         print(f"Warning: Global prior statistic for target '{target_col}' is NaN. Encoding might produce NaNs.")
         global_prior_stat = 0.0 # Default to 0.0 if global prior is NaN

    # --- Encode Train Data (Out-of-Fold) --- 
    # Store OOF results temporarily (index, encoded_value)
    oof_results_dfs = [] # Store list of DataFrames (original_index_col, new_col_name_1, new_col_name_2, ...)

    for fold_idx, (train_indices, valid_indices) in enumerate(cv_indices):
        if np.max(train_indices) >= len(train_df) or np.max(valid_indices) >= len(train_df):
            raise IndexError(f"CV indices out of bounds for train_df at fold {fold_idx}")
            
        train_fold_df = train_df[train_indices]
        valid_fold_df = train_df.with_row_index(original_index_col)[valid_indices] # Get original_idx for validation set
        
        fold_prior_stat = _calculate_statistic(train_fold_df[target_col], agg_stat, percentile)
        if np.isnan(fold_prior_stat):
             fold_prior_stat = global_prior_stat

        fold_encoded_cols = [pl.col(original_index_col)] # Start with the index

        for feature in features:
            new_col_name = f"{feature}{new_col_suffix}"
            
            if agg_stat == 'mean':
                agg_expr = pl.mean(target_col).alias('encoded_value')
            elif agg_stat == 'median':
                agg_expr = pl.median(target_col).alias('encoded_value')
            else: # percentile
                agg_expr = pl.quantile(target_col, percentile / 100.0, interpolation='linear').alias('encoded_value')

            encoding_map = train_fold_df.group_by(feature).agg(
                agg_expr,
                pl.col(target_col).count().alias('count')
            )
            
            if use_smoothing:
                encoding_map = encoding_map.with_columns(
                    ((pl.col('count') * pl.col('encoded_value') + smoothing * fold_prior_stat) / 
                     (pl.col('count') + smoothing)).alias(new_col_name)
                ).select([feature, new_col_name])
            else:
                encoding_map = encoding_map.rename({"encoded_value": new_col_name}).select([feature, new_col_name])
            
            valid_fold_df = valid_fold_df.join(
                encoding_map, on=feature, how='left'
            ).with_columns(
                 pl.col(new_col_name).fill_null(fold_prior_stat) 
            )
            fold_encoded_cols.append(pl.col(new_col_name))
        
        # Select only the original index and all new encoded columns for this fold
        oof_results_dfs.append(valid_fold_df.select(fold_encoded_cols))

    # Concatenate all fold results and join to the main output DataFrame
    if oof_results_dfs:
        full_oof_encoded_df = pl.concat(oof_results_dfs).sort(original_index_col)
        train_df_out = train_df_out.join(full_oof_encoded_df, on=original_index_col, how="left")
    else: # Should not happen if cv_indices is not empty
        for feature in features:
            train_df_out = train_df_out.with_columns(pl.lit(np.nan).alias(f"{feature}{new_col_suffix}"))

    # --- Encode Test Data (Using Full Training Data) --- 
    if test_df_out is not None:
        # Ensure test_df_out has original_index_col if it was added to train_df_out for joining logic, though not strictly needed for test encoding itself.
        # if original_index_col in train_df_out.columns and original_index_col not in test_df_out.columns:
        #      test_df_out = test_df_out.with_row_index(original_index_col)

        for feature in features:
            new_col_name = f"{feature}{new_col_suffix}"
            
            if agg_stat == 'mean':
                agg_expr = pl.mean(target_col).alias('encoded_value')
            elif agg_stat == 'median':
                agg_expr = pl.median(target_col).alias('encoded_value')
            else: # percentile
                agg_expr = pl.quantile(target_col, percentile / 100.0, interpolation='linear').alias('encoded_value')
            
            full_encoding_map = train_df.group_by(feature).agg(
                agg_expr,
                pl.col(target_col).count().alias('count')
            )
            
            if use_smoothing:
                 full_encoding_map = full_encoding_map.with_columns(
                      ((pl.col('count') * pl.col('encoded_value') + smoothing * global_prior_stat) / 
                       (pl.col('count') + smoothing)).alias(new_col_name)
                 ).select([feature, new_col_name])
            else:
                 full_encoding_map = full_encoding_map.rename({"encoded_value": new_col_name}).select([feature, new_col_name])
            
            test_df_out = test_df_out.join(
                 full_encoding_map, on=feature, how='left'
            ).with_columns(
                pl.col(new_col_name).fill_null(global_prior_stat) # Fill unseen with global prior
            )

    # Clean up: Drop the temporary original index column from both DataFrames if it was added
    # We check if the original train_df had this column name initially
    original_train_had_index_col = False
    search_original_index_col = "_original_idx"
    search_idx_counter = 0
    while True:
        current_search_col = search_original_index_col
        if search_idx_counter > 0:
            current_search_col = f"{search_original_index_col}_{search_idx_counter}"
        if current_search_col == original_index_col: # This is the one we added
            if current_search_col in train_df.columns: # Did original df have this exact name?
                 original_train_had_index_col = True
            break
        if current_search_col not in train_df_out.columns: # If we overshoot, means it wasn't there
            break
        search_idx_counter += 1
        if search_idx_counter > 100: # Safety break
            print("Warning: Original index column name search exceeded limits.") 
            break 

    if not original_train_had_index_col:
        if original_index_col in train_df_out.columns:
            train_df_out = train_df_out.drop(original_index_col)
        if test_df_out is not None and original_index_col in test_df_out.columns:
            # Only drop from test_df_out if it was actually added (which it might not be based on current logic)
            # A safer check is if it was in train_df_out and now in test_df_out
            test_df_out = test_df_out.drop(original_index_col)

    return train_df_out, test_df_out


if __name__ == '__main__':
    from sklearn.model_selection import KFold
    
    # Example Data
    train_example_df = pl.DataFrame({
        'id': range(10),
        'category_A': ['apple', 'banana', 'apple', 'orange', 'banana', 'apple', 'grape', 'orange', 'grape', 'banana'],
        'category_B': ['red', 'yellow', 'green', 'orange', 'yellow', 'red', 'purple', 'orange', 'green', 'yellow'],
        'target': [10, 12, 11, 20, 13, 9, 30, 22, 28, 14]
    })
    test_example_df = pl.DataFrame({
        'id': range(10, 15),
        'category_A': ['apple', 'grape', 'banana', 'orange', 'kiwi'], # kiwi is unseen
        'category_B': ['red', 'purple', 'yellow', 'green', 'brown']  # brown is unseen
    })

    # Create CV indices
    kf = KFold(n_splits=5, shuffle=False) # Shuffle=False for predictable OOF example
    cv_idx_list = []
    for tr_idx, val_idx in kf.split(train_example_df):
        cv_idx_list.append((tr_idx, val_idx))

    features_to_encode_list = ['category_A', 'category_B']
    target_column_name = 'target'

    print("Original Train DF:")
    print(train_example_df)
    print("\nOriginal Test DF:")
    print(test_example_df)

    # --- Test 1: Mean encoding with smoothing ---
    print("\n--- Test 1: Mean encoding, smoothing=True, smoothing_factor=5 ---")
    train_t1, test_t1 = apply_target_encoding(
        train_example_df, features_to_encode_list, target_column_name, cv_idx_list, 
        test_df=test_example_df, agg_stat='mean', smoothing=5, use_smoothing=True
    )
    print("Train DF (Test 1):")
    print(train_t1)
    print("Test DF (Test 1):")
    print(test_t1)

    # --- Test 2: Median encoding without smoothing ---
    print("\n--- Test 2: Median encoding, smoothing=False ---")
    train_t2, test_t2 = apply_target_encoding(
        train_example_df, features_to_encode_list, target_column_name, cv_idx_list, 
        test_df=test_example_df, agg_stat='median', use_smoothing=False
    )
    print("Train DF (Test 2):")
    print(train_t2)
    print("Test DF (Test 2):")
    print(test_t2)

    # --- Test 3: Percentile encoding with smoothing ---
    print("\n--- Test 3: Percentile (75th) encoding, smoothing=True, smoothing_factor=2 ---")
    train_t3, test_t3 = apply_target_encoding(
        train_example_df, features_to_encode_list, target_column_name, cv_idx_list, 
        test_df=test_example_df, agg_stat='percentile', percentile=75, smoothing=2, use_smoothing=True
    )
    print("Train DF (Test 3):")
    print(train_t3)
    print("Test DF (Test 3):")
    print(test_t3)

    # --- Test 4: Global prior is NaN (e.g. target all nulls) ---
    print("\n--- Test 4: Target all NaNs (Global Prior NaN) ---")
    train_nan_target = train_example_df.with_columns(pl.lit(None, dtype=pl.Float64).alias('target'))
    try:
        train_t4, test_t4 = apply_target_encoding(
            train_nan_target, features_to_encode_list, target_column_name, cv_idx_list,
            test_df=test_example_df, smoothing=1
        )
        print("Train DF (Test 4 - NaN target):") # Expect encoded values to be 0.0 (fallback for NaN prior)
        print(train_t4)
        print("Test DF (Test 4 - NaN target):")
        print(test_t4)
    except Exception as e:
        print(f"Error in Test 4: {e}")

    # --- Test 5: No test_df ---
    print("\n--- Test 5: No test_df provided ---")
    train_t5, test_t5 = apply_target_encoding(
        train_example_df, features_to_encode_list, target_column_name, cv_idx_list, 
        test_df=None, agg_stat='mean'
    )
    print("Train DF (Test 5):")
    print(train_t5)
    assert test_t5 is None, "Test_t5 should be None when no test_df is provided"
    print("Test DF (Test 5): None (as expected)") 