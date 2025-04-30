# Utility function to handle outliers by clipping numerical features.

"""
Extended Description:
Provides a function to clip values in specified numerical columns of a Polars DataFrame
to fall within defined minimum and maximum bounds. This is useful for mitigating the
influence of extreme outliers.
"""

import polars as pl
from typing import List, Dict, Optional, Tuple

def clip_numerical_features(
    df: pl.DataFrame,
    feature_bounds: Dict[str, Tuple[Optional[float], Optional[float]]]
) -> pl.DataFrame:
    """Clips numerical features based on specified lower and upper bounds.

    Args:
        df (pl.DataFrame): The input Polars DataFrame.
        feature_bounds (Dict[str, Tuple[Optional[float], Optional[float]]]):
            A dictionary where keys are numerical column names and values are tuples
            of (min_bound, max_bound). Use None for a bound if only one-sided
            clipping is needed (e.g., (None, 100) to clip only maximum values).

    Returns:
        pl.DataFrame: DataFrame with the specified features clipped.

    Raises:
        pl.exceptions.ColumnNotFoundError: If a specified feature is not found.
        TypeError: If a specified feature is not numeric.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError("df must be a Polars DataFrame.")
    if not feature_bounds:
        # Return original DataFrame if no bounds specified
        return df

    exprs_to_apply = []
    for col_name, (min_bound, max_bound) in feature_bounds.items():
        if col_name not in df.columns:
            raise pl.exceptions.ColumnNotFoundError(f"Feature '{col_name}' not found in DataFrame.")
        if not df[col_name].dtype.is_numeric():
            raise TypeError(f"Feature '{col_name}' is not numeric and cannot be clipped.")

        # Use when/then/otherwise for broader compatibility
        clipped_col = pl.col(col_name)
        if min_bound is not None and max_bound is not None:
            clipped_col = (
                pl.when(pl.col(col_name) < min_bound)
                .then(pl.lit(min_bound))
                .when(pl.col(col_name) > max_bound)
                .then(pl.lit(max_bound))
                .otherwise(pl.col(col_name))
            )
        elif min_bound is not None:
            clipped_col = (
                pl.when(pl.col(col_name) < min_bound)
                .then(pl.lit(min_bound))
                .otherwise(pl.col(col_name))
            )
        elif max_bound is not None:
            clipped_col = (
                pl.when(pl.col(col_name) > max_bound)
                .then(pl.lit(max_bound))
                .otherwise(pl.col(col_name))
            )
        # No need for else, as clipped_col remains pl.col(col_name) if no bounds given

        # Only add expression if clipping actually happened
        if min_bound is not None or max_bound is not None:
             # Ensure the output type matches the original type
            original_dtype = df[col_name].dtype
            exprs_to_apply.append(clipped_col.cast(original_dtype).alias(col_name))


    if exprs_to_apply:
        df = df.with_columns(exprs_to_apply)

    return df 