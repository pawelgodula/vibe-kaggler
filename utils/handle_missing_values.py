# Handles missing values in a Polars DataFrame using simple strategies.

"""
Extended Description:
This function imputes missing values (nulls) in specified columns or all columns
of a Polars DataFrame using basic strategies like 'mean', 'median', 'mode', or filling
with a constant 'zero' or other specified 'literal' value.
It operates on the DataFrame using Polars expressions and returns a new DataFrame.
"""

import polars as pl
from typing import List, Optional, Any

def handle_missing_values(
    df: pl.DataFrame,
    strategy: str = 'mean',
    features: Optional[List[str]] = None,
    fill_value: Optional[Any] = None # Used for strategy='literal'
) -> pl.DataFrame:
    """Imputes missing values in a Polars DataFrame.

    Args:
        df (pl.DataFrame): The DataFrame with potential missing values.
        strategy (str, optional): The imputation strategy.
            Supported: 'mean', 'median', 'mode', 'zero', 'literal'.
            Defaults to 'mean'.
        features (Optional[List[str]], optional): List of column names to impute.
            If None, attempts imputation on all columns where the strategy is applicable.
            Defaults to None.
        fill_value (Optional[Any], optional): The value to use when strategy='literal'.
                                             Defaults to None.

    Returns:
        pl.DataFrame: A new DataFrame with missing values imputed.

    Raises:
        ValueError: If an unsupported strategy is provided, or if 'literal' strategy
                    is used without a fill_value.
        pl.exceptions.ComputeError: If a non-numeric strategy ('mean', 'median', 'zero')
                                   is applied to a non-numeric column explicitly.
        KeyError: If any specified feature is not found in the DataFrame.
    """
    df_filled = df # Polars operations create new DataFrames/Series (usually)
    strategy_lower = strategy.lower()

    if features is None:
        target_features = df_filled.columns
    else:
        missing_in_df = [f for f in features if f not in df_filled.columns]
        if missing_in_df:
            raise KeyError(f"Features not found in DataFrame: {missing_in_df}")
        target_features = features

    fill_expressions = []
    checked_features = set()

    for feature in target_features:
        if feature in checked_features:
             continue
        checked_features.add(feature)
        
        col_expr = pl.col(feature)
        col_dtype = df_filled[feature].dtype
        is_numeric = col_dtype.is_numeric()
        is_float = col_dtype.is_float()
        # Polars specific check: mean/median not directly applicable to boolean
        is_bool = col_dtype == pl.Boolean 

        # Determine fill value based on strategy
        fill_expr = None
        apply_fill = df_filled[feature].is_null().any() # Only apply if nulls exist

        if strategy_lower == 'mean':
            if is_numeric and not is_bool:
                fill_expr = col_expr.fill_null(col_expr.mean())
            elif features is not None: # Only raise error if feature was explicitly listed
                 raise pl.exceptions.ComputeError(f"Strategy 'mean' cannot be applied to non-numeric or boolean feature '{feature}'")
        elif strategy_lower == 'median':
             if is_numeric and not is_bool:
                 fill_expr = col_expr.fill_null(col_expr.median())
             elif features is not None:
                 raise pl.exceptions.ComputeError(f"Strategy 'median' cannot be applied to non-numeric or boolean feature '{feature}'")
        elif strategy_lower == 'mode':
             # Mode is unreliable for floats, disallow for now.
             if is_float:
                 if features is not None:
                      raise pl.exceptions.ComputeError(f"Strategy 'mode' is not supported for float feature '{feature}'")
                 # If features is None (all columns), just skip this float column silently
             else:
                 # Mode works for most other types in Polars
                 fill_expr = col_expr.fill_null(col_expr.mode().first()) # .mode() returns list
        elif strategy_lower == 'zero':
            if is_numeric and not is_bool: # Don't fill bools with 0
                 fill_expr = col_expr.fill_null(0)
            elif features is not None:
                 raise pl.exceptions.ComputeError(f"Strategy 'zero' cannot be applied to non-numeric or boolean feature '{feature}'")
        elif strategy_lower == 'literal':
            if fill_value is None:
                raise ValueError("Strategy 'literal' requires a 'fill_value' to be provided.")
            # Use pl.lit to ensure correct type handling
            fill_expr = col_expr.fill_null(pl.lit(fill_value))
        else:
            raise ValueError(
                f"Unsupported strategy: '{strategy}'. Supported: "
                f"'mean', 'median', 'mode', 'zero', 'literal'"
            )

        if fill_expr is not None and apply_fill:
            fill_expressions.append(fill_expr)
        elif features is None and not apply_fill:
             # If processing all columns and strategy was inapplicable OR no nulls, keep original
             pass
        # else: If features were specified and strategy inapplicable, error was already raised

    # Apply all valid fill expressions at once
    if fill_expressions:
        df_filled = df_filled.with_columns(fill_expressions)

    return df_filled 