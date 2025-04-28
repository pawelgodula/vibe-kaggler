# Applies Count Encoding (Frequency Encoding) to categorical features.

"""
Extended Description:
Replaces categorical features with the count (or frequency) of each category.
The counts can be calculated based on the training set only, the test set only,
or the combined training and test sets.
Handles unseen categories by filling with zero or NaN.
Optionally normalizes counts to produce frequencies.
"""

import polars as pl
import numpy as np
from typing import List, Optional, Tuple, Literal

def apply_count_encoding(
    train_df: pl.DataFrame,
    features: List[str],
    test_df: Optional[pl.DataFrame] = None,
    count_on: Literal['train', 'test', 'combined'] = 'train',
    normalize: bool = False,
    handle_unknown: Literal['zero', 'nan'] = 'zero',
    new_col_suffix: str = '_count'
) -> Tuple[pl.DataFrame, Optional[pl.DataFrame]]:
    """Applies count or frequency encoding to specified categorical features.

    Args:
        train_df (pl.DataFrame): Training DataFrame.
        features (List[str]): List of categorical column names to encode.
        test_df (Optional[pl.DataFrame], optional): Test DataFrame. Required if
            count_on is 'test' or 'combined'. Defaults to None.
        count_on (Literal['train', 'test', 'combined'], optional): Data to use for
            calculating counts/frequencies. Defaults to 'train'.
        normalize (bool, optional): If True, calculate frequency (count / total non-null)
            instead of raw count. Defaults to False.
        handle_unknown (Literal['zero', 'nan'], optional): How to handle categories
            present in the target DataFrame but not in the counting source.
            'zero': Fill with 0.
            'nan': Fill with np.nan.
            Defaults to 'zero'.
        new_col_suffix (str, optional): Suffix for the new encoded columns.
                                      Defaults to '_count'.

    Returns:
        Tuple[pl.DataFrame, Optional[pl.DataFrame]]:
            - Training DataFrame with added count/frequency encoded columns.
            - Test DataFrame with added count/frequency encoded columns (or None).

    Raises:
        ValueError: If count_on requires test_df but it's not provided, or if
                    handle_unknown has an invalid value.
        pl.exceptions.ColumnNotFoundError: If features are not found in relevant DataFrames.
    """
    if count_on in ['test', 'combined'] and test_df is None:
        raise ValueError(f"test_df must be provided when count_on='{count_on}'.")
    if handle_unknown not in ['zero', 'nan']:
        raise ValueError("handle_unknown must be 'zero' or 'nan'.")

    train_df_out = train_df.clone()
    test_df_out = test_df.clone() if test_df is not None else None

    # --- Validate Columns --- 
    required_train_cols = set(features)
    missing_train = required_train_cols - set(train_df.columns)
    if missing_train:
        raise pl.exceptions.ColumnNotFoundError(f"Features missing in train_df: {missing_train}")
    if test_df is not None:
        missing_test = required_train_cols - set(test_df.columns)
        if missing_test:
            raise pl.exceptions.ColumnNotFoundError(f"Features missing in test_df: {missing_test}")

    # --- Determine Source Data for Counts --- 
    if count_on == 'train':
        source_df = train_df
    elif count_on == 'test':
        source_df = test_df # Already validated test_df exists
    elif count_on == 'combined':
        # Select only feature columns to avoid schema mismatch if other columns differ
        source_df = pl.concat([
            train_df.select(features),
            test_df.select(features) # Already validated test_df exists
        ], how='vertical')
    else:
        # Should be unreachable due to Literal typing, but good practice
        raise ValueError(f"Invalid count_on value: {count_on}") 

    # Determine fill value and final dtype based on options
    if handle_unknown == 'zero':
        fill_value = 0
        # If normalizing, output is Float64, otherwise UInt32 for counts
        count_col_dtype = pl.Float64 if normalize else pl.UInt32 
    else: # handle_unknown == 'nan'
        fill_value = np.nan
        # If filling with NaN, output must be Float64
        count_col_dtype = pl.Float64

    # --- Calculate and Apply Encoding --- 
    for feature in features:
        new_col_name = f"{feature}{new_col_suffix}"
        
        # Calculate value counts on the source data
        count_map = source_df.group_by(feature).agg(
            pl.len().alias("count") # len() gives count of rows in group
        )
        
        total_non_null = source_df[feature].drop_nulls().len()

        if normalize:
            if total_non_null == 0:
                 print(f"Warning: Feature '{feature}' has no non-null values in source data for normalization. Frequencies will be NaN.")
                 count_map = count_map.with_columns(
                     pl.lit(np.nan, dtype=pl.Float64).alias("encoded_value")
                 )
            else:
                 count_map = count_map.with_columns(
                     (pl.col("count") / total_non_null).alias("encoded_value")
                 )
        else:
            # If not normalizing, just rename the count column
            count_map = count_map.rename({"count": "encoded_value"})

        # Select only the feature and the final encoded value, cast BEFORE join
        # This ensures the map has the intended type, though join might still upcast later?
        count_map = count_map.select([feature, pl.col("encoded_value").cast(count_col_dtype)])

        # --- Join map to train_df_out ---
        train_df_out = train_df_out.join(
            count_map, on=feature, how='left'
        )
        # Rename the joined column and fill nulls (unseen categories)
        # Ensure the final column has the desired dtype after fill_null
        train_df_out = train_df_out.with_columns(
            pl.col('encoded_value').fill_null(fill_value).cast(count_col_dtype).alias(new_col_name)
        ).drop('encoded_value')

        # --- Join map to test_df_out (if exists) ---
        if test_df_out is not None:
            test_df_out = test_df_out.join(
                count_map, on=feature, how='left'
            )
            test_df_out = test_df_out.with_columns(
                pl.col('encoded_value').fill_null(fill_value).cast(count_col_dtype).alias(new_col_name)
            ).drop('encoded_value')

    return train_df_out, test_df_out 