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
    while original_index_col in train_df_out.columns:
        original_index_col += "_"
    train_df_out = train_df_out.with_row_index(original_index_col)
    was_index_added = True # Flag to know if we should drop it later

    # --- Validate Columns ---
    required_train_cols = set(features) | {target_col}
    missing_train = required_train_cols - set(train_df.columns)
    if missing_train:
        raise pl.exceptions.ColumnNotFoundError(f"Columns missing in train_df: {missing_train}")
    if test_df is not None:
        missing_test = set(features) - set(test_df.columns)
        if missing_test:
            raise pl.exceptions.ColumnNotFoundError(f"Features missing in test_df: {missing_test}")

    # --- Calculate Global Prior (from full training data) --- 
    global_prior_stat = _calculate_statistic(train_df[target_col], agg_stat, percentile)
    if np.isnan(global_prior_stat):
         print(f"Warning: Global prior statistic for target '{target_col}' is NaN. Encoding might produce NaNs.")
         # Consider defaulting to 0 or raising an error if global prior is crucial
         # global_prior_stat = 0.0 

    # --- Encode Train Data (Out-of-Fold) --- 
    # Store OOF results temporarily (index, encoded_value)
    oof_results = {f"{feature}{new_col_suffix}": [] for feature in features}

    for fold_idx, (train_indices, valid_indices) in enumerate(cv_indices):
        # Ensure indices are valid for the DataFrame
        if np.max(train_indices) >= len(train_df) or np.max(valid_indices) >= len(train_df):
            raise IndexError(f"CV indices out of bounds for train_df at fold {fold_idx}")
            
        train_fold_df = train_df[train_indices] # Use standard slicing
        
        # Add original index to validation rows to preserve order/mapping
        valid_fold_rows_with_idx = train_df[valid_indices].with_row_index("_original_idx")
        # Map numpy valid_indices to the row index created by with_row_index
        # This assumes with_row_index respects the order of the slice
        # A potentially safer way might be to add a unique ID before splitting folds.
        # However, let's assume indices correspond for now.
        # Alternatively, capture the original index from train_df *before* slicing.
        valid_fold_rows_with_idx = train_df.with_row_index("_original_idx")[valid_indices]

        
        # Calculate fold-specific prior (excluding validation fold)
        fold_prior_stat = _calculate_statistic(train_fold_df[target_col], agg_stat, percentile)
        if np.isnan(fold_prior_stat):
             # Use global prior if fold prior is NaN (e.g., fold target is all null)
             fold_prior_stat = global_prior_stat 
             if np.isnan(fold_prior_stat):
                 print(f"Warning: Both fold and global prior are NaN for fold {fold_idx}. Filling validation NaNs.")
                 fold_prior_stat = 0.0 # Fallback if both are NaN?

        for feature in features:
            new_col_name = f"{feature}{new_col_suffix}"
            
            # --- Determine aggregation expression based on agg_stat ---
            if agg_stat == 'mean':
                agg_expr = pl.mean(target_col).alias('encoded_value')
            elif agg_stat == 'median':
                agg_expr = pl.median(target_col).alias('encoded_value')
            elif agg_stat == 'percentile':
                # percentile is guaranteed to be non-None here due to initial check
                agg_expr = pl.quantile(target_col, percentile / 100.0, interpolation='linear').alias('encoded_value')
            # No else needed due to initial validation of agg_stat
            # --------------------------------------------------------

            # Calculate encoding map based on the training part of the fold
            encoding_map = train_fold_df.group_by(feature).agg(
                #pl.col(target_col).apply(lambda s: _calculate_statistic(s, agg_stat, percentile)).alias('encoded_value'),
                agg_expr, # Use the determined Polars aggregation expression
                pl.col(target_col).count().alias('count') # Count non-null target values
            )
            
            # Apply smoothing conditionally
            if use_smoothing:
                encoding_map = encoding_map.with_columns(
                    ((pl.col('count') * pl.col('encoded_value') + smoothing * fold_prior_stat) / (pl.col('count') + smoothing)).alias('final_encoding')
                ).select([feature, 'final_encoding'])
            else:
                # Rename 'encoded_value' to 'final_encoding' for consistency
                encoding_map = encoding_map.rename({"encoded_value": "final_encoding"}).select([feature, 'final_encoding'])
            
            # Map to the validation part of the fold (which now has _original_idx)
            # Join, fill nulls, and select the original index and the encoded value
            valid_encoded_fold = valid_fold_rows_with_idx.join(
                encoding_map, on=feature, how='left' # Join based on the feature
            ).with_columns(
                 # Fill NaNs (unseen categories in train fold) with the raw fold prior 
                 pl.col('final_encoding').fill_null(fold_prior_stat) 
            ).select([original_index_col, pl.col("final_encoding").alias(new_col_name)])

            # Append the result (DataFrame slice) to the list for this feature
            oof_results[new_col_name].append(valid_encoded_fold)

    # After iterating through folds, concatenate results for each feature and join to output df
    for feature in features:
        new_col_name = f"{feature}{new_col_suffix}"
        if oof_results[new_col_name]: # Check if list is not empty
            # Concatenate the list of DataFrames for this feature
            full_oof_feature = pl.concat(oof_results[new_col_name])
            # Sort by original index to ensure correct order
            full_oof_feature = full_oof_feature.sort(original_index_col) 
            # Join the OOF results back to the main DataFrame
            train_df_out = train_df_out.join(full_oof_feature, on=original_index_col, how="left")
            # Drop the index column if it wasn't there originally
            # if "_original_idx" not in train_df.columns: # Previous logic
            #      train_df_out = train_df_out.drop("_original_idx")
        else:
            # Handle case where cv_indices might be empty (though unlikely)
            train_df_out = train_df_out.with_columns(pl.lit(np.nan, dtype=pl.Float64).alias(new_col_name))

    # --- Encode Test Data (Using Full Training Data) --- 
    if test_df_out is not None:
        # Add original index for test data if needed for consistency, though not strictly required here
        # if "_original_idx" in train_df_out.columns and "_original_idx" not in test_df_out.columns:
        #      test_df_out = test_df_out.with_row_index("_original_idx")
        if was_index_added and original_index_col not in test_df_out.columns:
             test_df_out = test_df_out.with_row_index(original_index_col)

        for feature in features:
            new_col_name = f"{feature}{new_col_suffix}"
            
            # --- Determine aggregation expression based on agg_stat ---
            if agg_stat == 'mean':
                agg_expr = pl.mean(target_col).alias('encoded_value')
            elif agg_stat == 'median':
                agg_expr = pl.median(target_col).alias('encoded_value')
            elif agg_stat == 'percentile':
                agg_expr = pl.quantile(target_col, percentile / 100.0, interpolation='linear').alias('encoded_value')
            # --------------------------------------------------------
            
            # Calculate encoding map on the FULL training data
            full_encoding_map = train_df.group_by(feature).agg(
                #pl.col(target_col).apply(lambda s: _calculate_statistic(s, agg_stat, percentile)).alias('encoded_value'),
                agg_expr, # Use the determined Polars aggregation expression
                pl.col(target_col).count().alias('count')
            )
            
            # Apply smoothing conditionally using the GLOBAL prior
            if use_smoothing:
                 full_encoding_map = full_encoding_map.with_columns(
                      ((pl.col('count') * pl.col('encoded_value') + smoothing * global_prior_stat) / (pl.col('count') + smoothing)).alias('final_encoding')
                 ).select([feature, 'final_encoding'])
            else:
                 # Rename 'encoded_value' to 'final_encoding' for consistency
                 full_encoding_map = full_encoding_map.rename({"encoded_value": "final_encoding"}).select([feature, 'final_encoding'])
            
            # Join with test data
            test_df_out = test_df_out.join(
                 full_encoding_map, on=feature, how='left'
            )
            
            # Rename the joined column and fill missing values with the GLOBAL prior
            # Use coalesce to prioritize the joined value, then fill nulls
            test_df_out = test_df_out.with_columns(
                 pl.coalesce(pl.col('final_encoding'), pl.lit(global_prior_stat)).alias(new_col_name)
            ).drop('final_encoding') # Drop the intermediate column

        # Drop index from test if it wasn't there originally
        # if "_original_idx" in test_df_out.columns and "_original_idx" not in test_df.columns:
        #     test_df_out = test_df_out.drop("_original_idx")

    # --- Clean up index column if added --- 
    if was_index_added:
        train_df_out = train_df_out.drop(original_index_col)
        if test_df_out is not None and original_index_col in test_df_out.columns:
            test_df_out = test_df_out.drop(original_index_col)

    return train_df_out, test_df_out 