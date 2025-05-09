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


if __name__ == '__main__':
    # Example Usage
    train_data = pl.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7, 8],
        'city': ['NY', 'NY', 'SF', 'SF', 'LA', 'NY', None, 'SF'],
        'income': [50000, 60000, 75000, 80000, 65000, 55000, 70000, None],
        'age': [25, 30, 28, 35, 32, 29, 40, 22]
    })
    test_data = pl.DataFrame({
        'id': [9, 10, 11, 12, 13],
        'city': ['NY', 'SF', 'LA', 'Chicago', None], # Chicago is unseen in train
        'income': [58000, None, 70000, 60000, 0], # Test zero income
        'age': [27, 33, 30, 38, 24]
    })

    print("Original Train DF:")
    print(train_data)
    print("\nOriginal Test DF:")
    print(test_data)

    num_features = ['income', 'age']
    cat_feature = 'city'

    # --- Test 1: Calculate mean on train, fill_value=1.0 ---
    print("\n--- Test 1: Mean on train, fill_value=1.0 ---")
    train_t1, test_t1 = apply_ratio_to_category_mean_encoding(
        train_data, num_features, cat_feature, test_data, 
        calculate_mean_on='train', fill_value=1.0
    )
    print("Train DF (Test 1):")
    print(train_t1)
    print("Test DF (Test 1):")
    print(test_t1)

    # --- Test 2: Calculate mean on combined, fill_value=0.0, custom suffix ---
    print("\n--- Test 2: Mean on combined, fill_value=0.0, custom suffix ---")
    train_t2, test_t2 = apply_ratio_to_category_mean_encoding(
        train_data, num_features, cat_feature, test_data, 
        calculate_mean_on='combined', fill_value=0.0, new_col_suffix_format="_ratio_MEAN_{cat}"
    )
    print("Train DF (Test 2):")
    print(train_t2)
    print("Test DF (Test 2):")
    print(test_t2)
    
    # --- Test 3: Handling zero mean in a category ---
    print("\n--- Test 3: Handling zero mean in a category ---")
    train_data_zero_mean = train_data.with_columns(
        pl.when(pl.col('city') == 'LA').then(0).otherwise(pl.col('income')).alias('income')
    )
    print("Train DF for Test 3 (LA income set to 0):")
    print(train_data_zero_mean)
    train_t3, test_t3 = apply_ratio_to_category_mean_encoding(
        train_data_zero_mean, ['income'], cat_feature, test_data, 
        calculate_mean_on='train', fill_value=999.0 # Use a distinct fill value
    )
    print("Train DF (Test 3 - LA income ratio should be 999):")
    print(train_t3.filter(pl.col('city') == 'LA'))
    print("Test DF (Test 3 - LA income ratio should be 999):")
    print(test_t3.filter(pl.col('city') == 'LA'))
    
    # --- Test 4: No test_df provided but mean_on='test' (should raise ValueError) ---
    print("\n--- Test 4: Error case - mean_on='test' but no test_df ---")
    try:
        apply_ratio_to_category_mean_encoding(train_data, num_features, cat_feature, calculate_mean_on='test')
    except ValueError as ve:
        print(f"Caught expected ValueError: {ve}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")

    # --- Test 5: Numerical feature missing (should raise ColumnNotFoundError) ---
    print("\n--- Test 5: Error case - Numerical feature missing ---")
    try:
        apply_ratio_to_category_mean_encoding(train_data, ['non_existent_feature'], cat_feature, test_data)
    except pl.exceptions.ColumnNotFoundError as cnf:
        print(f"Caught expected ColumnNotFoundError: {cnf}")
    except Exception as e:
        print(f"Caught unexpected error: {e}") 