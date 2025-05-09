# Reduces the memory usage of a Polars DataFrame by downcasting numerical columns.

"""
Extended Description:
Iterates through all columns of a Polars DataFrame. For numerical columns (integers
and floats), it attempts to downcast them to the smallest possible subtype that
can hold the range of values present in the column without loss of information.
For Utf8 columns with low cardinality, it converts them to Categorical.
Prints the memory reduction achieved if verbose is True.
"""

import polars as pl
import numpy as np # Still needed for iinfo/finfo

def reduce_memory_usage(
    df: pl.DataFrame,
    verbose: bool = True,
    cat_threshold: float = 0.5 # Threshold for object->categorical conversion
) -> pl.DataFrame:
    """Downcasts numerical columns and converts potential categoricals in Polars.

    Args:
        df (pl.DataFrame): The DataFrame to optimize.
        verbose (bool, optional): If True, print the memory reduction details.
                                  Defaults to True.
        cat_threshold (float, optional): Ratio of unique values to total length below
                                        which Utf8 columns are converted to Categorical.
                                        Defaults to 0.5.

    Returns:
        pl.DataFrame: A new DataFrame with potentially reduced memory usage.
    """
    # Polars DataFrames are immutable, operations create new ones.
    # We collect expressions to apply them efficiently at the end.
    cast_expressions = []
    
    # Use estimated_size for memory usage
    start_mem = df.estimated_size("mb")
    if verbose:
        print(f'Initial memory usage: {start_mem:.2f} MB')

    for col_name in df.columns:
        col = df[col_name]
        col_type = col.dtype
        col_expr = pl.col(col_name)

        if col_type.is_numeric() and col_type != pl.Boolean:
            # Skip if column contains nulls, as min/max would be null
            if col.null_count() > 0:
                 if verbose:
                     print(f"Skipping numeric downcast for '{col_name}' due to nulls.")
                 continue
            
            c_min = col.min()
            c_max = col.max()
            
            if col_type.is_integer():
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    cast_expressions.append(col_expr.cast(pl.Int8)) 
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    cast_expressions.append(col_expr.cast(pl.Int16))
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    cast_expressions.append(col_expr.cast(pl.Int32))
                # else keep Int64 (already smallest or needed for range)
                elif col_type != pl.Int64: # Avoid redundant cast
                     cast_expressions.append(col_expr.cast(pl.Int64))
                     
            elif col_type.is_float():
                # Check if values fit within float32 range before downcasting
                f32_info = np.finfo(np.float32)
                can_cast_to_f32 = False
                if col_type == pl.Float64:
                    # Get min/max, handle potential errors if column is empty or all null
                    try:
                        c_min = col.min()
                        c_max = col.max()
                        if c_min is not None and c_max is not None: # Ensure min/max calculation succeeded
                            if c_min >= f32_info.min and c_max <= f32_info.max:
                                can_cast_to_f32 = True
                    except Exception:
                        # Handle cases where min/max might fail (e.g., empty df after filtering?)
                        # If min/max fail, probably safest not to cast.
                        pass 
                    
                    if can_cast_to_f32:
                        cast_expressions.append(col_expr.cast(pl.Float32))
                    # else: keep Float64
                # else keep original float type (Float32)

        elif col_type == pl.Utf8: 
            # Convert to categorical if cardinality is low
            num_unique = col.n_unique()
            num_total = len(col)
            if num_total > 0 and (num_unique / num_total) < cat_threshold:
                 cast_expressions.append(col_expr.cast(pl.Categorical))
                 if verbose:
                      print(f"Converting column '{col_name}' to Categorical.")
        # else: Keep other types (Boolean, Datetime, etc.) as is

    # Apply all casts at once
    if cast_expressions:
        df_reduced = df.with_columns(cast_expressions)
    else:
        df_reduced = df # No changes applied

    end_mem = df_reduced.estimated_size("mb")
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0
        print(f'Final memory usage: {end_mem:.2f} MB ({reduction:.1f}% reduction)')

    return df_reduced 