# Creates aggregation features based on grouping columns.

"""
Extended Description:
Calculates aggregate statistics (e.g., mean, std, min, max, count) for numerical
features, grouped by one or more categorical columns. The results are joined back
to the original DataFrame, creating new features.
"""

import polars as pl
from typing import List, Dict, Any

# Define a mapping from common aggregation names to Polars expressions
# Extend this as needed
AGG_EXPR_MAP = {
    'mean': pl.mean,
    'std': pl.std,
    'min': pl.min,
    'max': pl.max,
    'count': pl.count, # Counts non-nulls in the target column within the group
    'size': pl.len, # Counts all rows in the group (equivalent to SQL COUNT(*))
    'nunique': pl.n_unique,
    'sum': pl.sum,
    'median': pl.median,
    # Add more complex ones like 'first', 'last', 'skew', 'kurtosis' if required
}

def create_aggregation_features(
    df: pl.DataFrame,
    group_by_cols: List[str],
    agg_dict: Dict[str, List[str]],
    new_col_prefix: str = "agg_"
) -> pl.DataFrame:
    """Generates aggregation features and joins them back to the DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame.
        group_by_cols (List[str]): A list of column names to group by.
        agg_dict (Dict[str, List[str]]): A dictionary where keys are the names
            of the columns to aggregate, and values are lists of aggregation
            function names (e.g., 'mean', 'std', 'count') supported by the
            internal AGG_EXPR_MAP.
        new_col_prefix (str, optional): Prefix for the newly created aggregation
                                        columns. Defaults to "agg_".

    Returns:
        pl.DataFrame: The original DataFrame with the generated aggregation
                      features joined back based on the group_by columns.

    Raises:
        ValueError: If group_by_cols is empty.
        ValueError: If agg_dict is empty.
        ValueError: If an unsupported aggregation function name is used in agg_dict.
        pl.exceptions.ColumnNotFoundError: If any group_by or aggregation target
                                           column is not found in the DataFrame.
    """
    if not group_by_cols:
        raise ValueError("group_by_cols cannot be empty.")
    if not agg_dict:
        raise ValueError("agg_dict cannot be empty.")

    # Validate columns exist
    all_needed_cols = set(group_by_cols)
    for col in agg_dict:
         all_needed_cols.add(col)
    missing_cols = [c for c in all_needed_cols if c not in df.columns]
    if missing_cols:
         raise pl.exceptions.ColumnNotFoundError(f"Columns not found in DataFrame: {missing_cols}")

    agg_expressions = []
    for agg_col, agg_funcs in agg_dict.items():
        if not isinstance(agg_funcs, list):
             raise TypeError(f"Value for '{agg_col}' in agg_dict must be a list of strings, got {type(agg_funcs)}")
        for agg_func_name in agg_funcs:
            agg_func_name_lower = agg_func_name.lower()
            agg_expr_func = AGG_EXPR_MAP.get(agg_func_name_lower)
            if agg_expr_func is None:
                raise ValueError(
                    f"Unsupported aggregation function: '{agg_func_name}'. Supported: "
                    f"{list(AGG_EXPR_MAP.keys())}"
                )
                
            # Handle special case for 'size' which doesn't take a column argument
            if agg_func_name_lower == 'size':
                 new_col_name = f"{new_col_prefix}{'_'.join(group_by_cols)}_size"
                 expr = agg_expr_func().alias(new_col_name)
            else:
                 new_col_name = f"{new_col_prefix}{agg_col}_by_{'_'.join(group_by_cols)}_{agg_func_name_lower}"
                 expr = agg_expr_func(agg_col).alias(new_col_name)
                 
            agg_expressions.append(expr)

    if not agg_expressions:
        # This should not happen if agg_dict is validated, but as a safeguard
        print("Warning: No valid aggregation expressions were generated.")
        return df.clone() # Return a copy if no aggregations performed

    # Perform aggregation
    try:
        df_agg = df.group_by(group_by_cols).agg(agg_expressions)
    except Exception as e:
        print(f"Error during Polars group_by or aggregation: {e}")
        raise pl.exceptions.ComputeError("Aggregation computation failed.") from e

    # Join back to the original DataFrame
    try:
        # Use outer join to handle potential cases where original df might have
        # group combinations not present during agg (unlikely but safer?)
        # Or left join if we assume all original rows should be kept.
        # Left join seems more standard for feature engineering.
        df_out = df.join(df_agg, on=group_by_cols, how="left")
    except Exception as e:
        print(f"Error joining aggregation results back to DataFrame: {e}")
        raise pl.exceptions.ComputeError("Joining aggregation features failed.") from e
        
    return df_out 