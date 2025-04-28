# Applies Ratio to Category Mean Encoding.

"""
Extended Description:
Encodes numerical features by calculating their ratio relative to the mean
of that feature within groups defined by a specified categorical feature.
For example, encode 'age' relative to the mean 'age' for each 'sex'.

The means can be calculated based on the training set only, the test set only,
or the combined training and test sets.
Handles unseen categories, nulls in the numerical feature, and zero means by
filling the resulting ratio with a specified value (defaulting to 1.0).
"""

import polars as pl
import numpy as np
from typing import List, Optional, Tuple, Literal

def apply_ratio_to_category_mean_encoding(
    train_df: pl.DataFrame,
    numerical_features: List[str],
    category_col: str,
    test_df: Optional[pl.DataFrame] = None,
    calculate_mean_on: Literal['train', 'test', 'combined'] = 'train',
    fill_value: float = 1.0, 
    new_col_suffix_format: str = "_ratio_vs_{cat}"
) -> Tuple[pl.DataFrame, Optional[pl.DataFrame]]:
    """Encodes numerical features as the ratio to their mean within a category.

    Args:
        train_df (pl.DataFrame): Training DataFrame.
        numerical_features (List[str]): List of numerical features to encode.
        category_col (str): The categorical column to group by for calculating means.
        test_df (Optional[pl.DataFrame], optional): Test DataFrame. Required if
            calculate_mean_on is 'test' or 'combined'. Defaults to None.
        calculate_mean_on (Literal['train', 'test', 'combined'], optional): Data to use
            for calculating category means. Defaults to 'train'.
        fill_value (float, optional): Value to fill the ratio when the mean is zero/null,
            the category is unseen, or the numerical value itself is null.
            Defaults to 1.0.
        new_col_suffix_format (str, optional): Format string for the suffix of new columns.
            Use "{cat}" as a placeholder for the category column name.
            Defaults to "_ratio_vs_{cat}".

    Returns:
        Tuple[pl.DataFrame, Optional[pl.DataFrame]]:
            - Training DataFrame with added ratio-encoded columns.
            - Test DataFrame with added ratio-encoded columns (or None).

    Raises:
        ValueError: If calculate_mean_on requires test_df but it's not provided.
        pl.exceptions.ColumnNotFoundError: If category_col or numerical_features are missing.
        TypeError: If any numerical_features column is not numeric.
    """
    if calculate_mean_on in ['test', 'combined'] and test_df is None:
        raise ValueError(f"test_df must be provided when calculate_mean_on='{calculate_mean_on}'.")

    train_df_out = train_df.clone()
    test_df_out = test_df.clone() if test_df is not None else None

    # --- Validate Columns --- 
    required_cols_train = set(numerical_features) | {category_col}
    missing_train = required_cols_train - set(train_df.columns)
    if missing_train:
        raise pl.exceptions.ColumnNotFoundError(f"Columns missing in train_df: {missing_train}")
    
    if test_df is not None:
        required_cols_test = set(numerical_features) | {category_col}
        missing_test = required_cols_test - set(test_df.columns)
        if missing_test:
             raise pl.exceptions.ColumnNotFoundError(f"Columns missing in test_df: {missing_test}")
             
    # Validate numerical columns
    for num_feat in numerical_features:
        if not train_df[num_feat].dtype.is_numeric():
             raise TypeError(f"Column '{num_feat}' in train_df is not numeric.")
        if test_df is not None and not test_df[num_feat].dtype.is_numeric():
             raise TypeError(f"Column '{num_feat}' in test_df is not numeric.")

    # --- Determine Source Data for Means --- 
    cols_to_select = numerical_features + [category_col]
    if calculate_mean_on == 'train':
        source_df = train_df.select(cols_to_select)
    elif calculate_mean_on == 'test':
        source_df = test_df.select(cols_to_select) # Already validated test_df exists
    elif calculate_mean_on == 'combined':
        source_df = pl.concat([
            train_df.select(cols_to_select),
            test_df.select(cols_to_select) # Already validated test_df exists
        ], how='vertical')
    else:
        raise ValueError(f"Invalid calculate_mean_on value: {calculate_mean_on}") 

    # --- Calculate Category Means --- 
    # Use a placeholder for nulls to allow grouping and joining them
    null_placeholder = "_NULL_PLACEHOLDER_"
    source_df_filled = source_df.with_columns(
        pl.col(category_col).fill_null(null_placeholder)
    )
    mean_agg_exprs = [pl.mean(num_feat).alias(f"{num_feat}_mean") for num_feat in numerical_features]
    # Group by the filled column
    category_means_map = source_df_filled.group_by(category_col).agg(mean_agg_exprs)

    # --- Apply Ratio Encoding --- 
    suffix = new_col_suffix_format.format(cat=category_col)
    for num_feat in numerical_features:
        new_col_name = f"{num_feat}{suffix}"
        mean_col_name = f"{num_feat}_mean"
        
        # --- Apply to train_df_out ---
        # Fill nulls in the join key before joining
        train_df_out_filled = train_df_out.with_columns(
            pl.col(category_col).fill_null(null_placeholder).alias(f"__{category_col}_filled")
        )
        train_df_out_filled = train_df_out_filled.join(
            category_means_map.select([category_col, mean_col_name]), 
            left_on=f"__{category_col}_filled", # Join on filled column
            right_on=category_col, 
            how='left'
        )
        
        # Calculate ratio, handling nulls and zero division
        ratio_expr = pl.col(num_feat) / pl.col(mean_col_name)
        train_df_out = train_df_out_filled.with_columns(
             pl.when(
                  pl.col(num_feat).is_not_null() & 
                  pl.col(mean_col_name).is_not_null() & 
                  (pl.col(mean_col_name) != 0)
             )
             .then(ratio_expr)
             .otherwise(pl.lit(fill_value, dtype=pl.Float64))
             .alias(new_col_name)
        ).drop([mean_col_name, f"__{category_col}_filled"]) # Drop helpers

        # --- Apply to test_df_out (if exists) ---
        if test_df_out is not None:
            # Fill nulls in the join key before joining
            test_df_out_filled = test_df_out.with_columns(
                pl.col(category_col).fill_null(null_placeholder).alias(f"__{category_col}_filled")
            )
            test_df_out_filled = test_df_out_filled.join(
                category_means_map.select([category_col, mean_col_name]), 
                left_on=f"__{category_col}_filled", # Join on filled column
                right_on=category_col, 
                how='left'
            )
            test_df_out = test_df_out_filled.with_columns(
                 pl.when(
                      pl.col(num_feat).is_not_null() & 
                      pl.col(mean_col_name).is_not_null() & 
                      (pl.col(mean_col_name) != 0)
                 )
                 .then(ratio_expr)
                 .otherwise(pl.lit(fill_value, dtype=pl.Float64))
                 .alias(new_col_name)
            ).drop([mean_col_name, f"__{category_col}_filled"]) # Drop helpers

    return train_df_out, test_df_out 