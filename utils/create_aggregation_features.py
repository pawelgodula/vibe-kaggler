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
    agg_configs: List[Dict[str, Any]] # List of config dicts
) -> pl.DataFrame:
    """Generates aggregation features based on a list of configurations.

    Args:
        df (pl.DataFrame): The input DataFrame.
        agg_configs (List[Dict[str, Any]]): A list of dictionaries, where each
            dictionary defines one aggregation and should contain keys like:
            - 'groupby_cols': List[str] - Columns to group by.
            - 'agg_col': str - Column to aggregate.
            - 'agg_func': str - Aggregation function name (e.g., 'mean').
            - 'new_col_name': str - Name for the resulting feature column.

    Returns:
        pl.DataFrame: The original DataFrame with the generated aggregation
                      features joined back.

    Raises:
        ValueError: If agg_configs list is empty.
        ValueError: If an unsupported aggregation function name is used.
        ValueError: If a config dictionary is missing required keys.
        pl.exceptions.ColumnNotFoundError: If any required column is not found.
    """
    if not agg_configs:
        raise ValueError("agg_configs list cannot be empty.")

    df_out = df.clone()
    all_agg_dfs = [] # Store intermediate aggregated DFs
    required_keys = {'groupby_cols', 'agg_col', 'agg_func', 'new_col_name'}

    # Process each aggregation configuration separately
    for i, config in enumerate(agg_configs):
        if not isinstance(config, dict):
            raise TypeError(f"Element {i} in agg_configs is not a dictionary.")
        if not required_keys.issubset(config.keys()):
            missing = required_keys - set(config.keys())
            raise ValueError(f"Config dictionary {i} is missing required keys: {missing}")

        group_by_cols = config['groupby_cols']
        agg_col = config['agg_col']
        agg_func_name = config['agg_func'].lower()
        new_col_name = config['new_col_name']

        # Validate columns for this specific config
        current_needed = set(group_by_cols) | {agg_col}
        missing_cols = [c for c in current_needed if c not in df.columns]
        if missing_cols:
            raise pl.exceptions.ColumnNotFoundError(f"Columns for config {i} not found: {missing_cols}")

        agg_expr_func = AGG_EXPR_MAP.get(agg_func_name)
        if agg_expr_func is None:
            raise ValueError(
                f"Unsupported aggregation function '{config['agg_func']}' in config {i}. Supported: "
                f"{list(AGG_EXPR_MAP.keys())}"
            )

        # Handle special case for 'size'
        if agg_func_name == 'size':
            agg_expr = agg_expr_func().alias(new_col_name)
        else:
            agg_expr = agg_expr_func(agg_col).alias(new_col_name)

        # Perform this single aggregation
        try:
            df_agg_single = df.group_by(group_by_cols).agg(agg_expr)
            all_agg_dfs.append((group_by_cols, df_agg_single))
        except Exception as e:
            print(f"Error during aggregation for config {i}: {config}")
            raise pl.exceptions.ComputeError("Aggregation computation failed.") from e

    # Join all aggregated features back one by one
    for group_keys, df_agg in all_agg_dfs:
        try:
            df_out = df_out.join(df_agg, on=group_keys, how="left")
        except Exception as e:
            agg_col_name = df_agg.columns[-1] # Get the name of the aggregated column
            print(f"Error joining aggregation result '{agg_col_name}' (groups: {group_keys}) back: {e}")
            raise pl.exceptions.ComputeError("Joining aggregation features failed.") from e
        
    return df_out 