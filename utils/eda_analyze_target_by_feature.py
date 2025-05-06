# Function to analyze the target variable grouped by a feature.

"""
Extended Description:
Analyzes the relationship between a specified feature (categorical or numerical) 
and a numerical target variable. 

- For categorical features: Calculates count, mean, and median of the target for 
  each category. Can limit to top N categories by count.
- For numerical features: Bins the feature using quantiles (qcut) and then 
  calculates count, mean, and median of the target for each bin.

Returns a Polars DataFrame containing the summary statistics.
"""

import polars as pl
from typing import Optional, Literal
import numpy as np # Add numpy import for linspace

def analyze_target_by_feature(
    df: pl.DataFrame,
    target_col: str,
    feature_col: str,
    feature_type: Optional[Literal['auto', 'categorical', 'numerical']] = 'auto',
    num_bins: int = 10, # For numerical features
    max_categories: Optional[int] = 20, # For categorical features
    handle_nulls: Literal['drop', 'fill'] = 'drop' # How to handle nulls in feature_col
) -> Optional[pl.DataFrame]:
    """Calculates target summaries grouped by feature values or bins.

    Args:
        df (pl.DataFrame): Input DataFrame.
        target_col (str): Name of the (numeric) target column.
        feature_col (str): Name of the feature column to group by.
        feature_type (Optional[Literal['auto', 'categorical', 'numerical']], optional): 
            Explicitly specify feature type, or 'auto' to infer. Defaults to 'auto'.
        num_bins (int, optional): Number of quantile bins for numerical features. 
            Defaults to 10.
        max_categories (Optional[int], optional): Max categories to analyze for 
            categorical features (top N by count). If None, analyzes all. Defaults to 20.
        handle_nulls (Literal['drop', 'fill'], optional): How to handle nulls in the 
            feature column before grouping. 'drop' removes rows with nulls, 'fill' 
            replaces nulls with a placeholder string/category. Defaults to 'drop'.

    Returns:
        Optional[pl.DataFrame]: A DataFrame with summary statistics (count, mean, median)
            of the target grouped by the feature. Columns include the feature/bin, 
            'count', 'target_mean', 'target_median'. Returns None if analysis fails 
            (e.g., target not numeric, feature not found).

    Raises:
        pl.exceptions.ColumnNotFoundError: If target_col or feature_col are not found.
        TypeError: If the target column is not numeric.
    """
    if target_col not in df.columns:
        raise pl.exceptions.ColumnNotFoundError(f"Target column '{target_col}' not found.")
    if feature_col not in df.columns:
        raise pl.exceptions.ColumnNotFoundError(f"Feature column '{feature_col}' not found.")
    if not df[target_col].dtype.is_numeric():
        raise TypeError(f"Target column '{target_col}' must be numeric for this analysis.")

    # --- Data Preparation ---
    df_analysis = df.select([feature_col, target_col])

    # Handle nulls in the feature column
    if handle_nulls == 'drop':
        df_analysis = df_analysis.drop_nulls(subset=[feature_col])
    elif handle_nulls == 'fill':
        # Fill based on type - could be improved (e.g., specific fill values)
        if df_analysis[feature_col].dtype.is_numeric():
             # Filling numerics might distort analysis, maybe warn?
             # For now, let's cast to string to treat as a separate category
              df_analysis = df_analysis.with_columns(
                  pl.col(feature_col).cast(pl.Utf8).fill_null("_NULL_").alias(feature_col)
              )
        else:
            df_analysis = df_analysis.with_columns(
                pl.col(feature_col).fill_null("_NULL_").alias(feature_col)
            )
    else:
         raise ValueError("handle_nulls must be 'drop' or 'fill'")
         
    if df_analysis.is_empty():
        print(f"Warning: No data left for feature '{feature_col}' after handling nulls.")
        return None

    # --- Determine Feature Type ---
    f_type = feature_type
    if f_type == 'auto':
        if df_analysis[feature_col].dtype.is_numeric():
            # Consider numeric if more than ~20 unique values?
            # Simple heuristic for now: treat as numeric if float or > N unique ints
            # A better approach might involve cardinality checks vs max_categories
            unique_count = df_analysis[feature_col].n_unique()
            if df_analysis[feature_col].dtype.is_float() or unique_count > (max_categories or 20):
                f_type = 'numerical'
            else:
                f_type = 'categorical'
        else: # Strings, Bool, etc. are treated as categorical
            f_type = 'categorical'

    # --- Perform Analysis --- 
    summary_df: Optional[pl.DataFrame] = None
    grouping_col = feature_col

    try:
        if f_type == 'categorical':
            print(f"Analyzing '{feature_col}' as categorical...")
            # Limit categories if needed
            value_counts = df_analysis[feature_col].value_counts()
            if max_categories is not None and len(value_counts) > max_categories:
                top_categories = value_counts.head(max_categories)[feature_col].to_list()
                print(f"  Limiting to top {max_categories} categories.")
                df_analysis = df_analysis.filter(pl.col(feature_col).is_in(top_categories))
                # Consider adding an '_OTHER_' category if needed
            
            summary_df = df_analysis.group_by(feature_col).agg([
                pl.count().alias("count"),
                pl.mean(target_col).alias("target_mean"),
                pl.median(target_col).alias("target_median")
            ]).sort("count", descending=True)

        elif f_type == 'numerical':
            print(f"Analyzing '{feature_col}' as numerical (binning into {num_bins} quantiles)...")
            bin_col = f"{feature_col}_bin"
            grouping_col = bin_col
            try:
                 # --- Modification Start ---
                 # Use polars.qcut directly to get bin edges
                 # Need the series data - handle nulls first if not already dropped
                 series_to_bin = df_analysis[feature_col].drop_nulls() # Ensure no nulls for qcut

                 # --- Add Cardinality Check --- 
                 n_unique = series_to_bin.n_unique()
                 # Define a threshold for skipping high-cardinality numerical features
                 # Adjust threshold as needed (e.g., based on df length)
                 # Example: Skip if > 1000 unique values or > 90% unique
                 max_numeric_cardinality = 1000 
                 high_cardinality_ratio = 0.9 
                 
                 if n_unique > max_numeric_cardinality or (series_to_bin.len() > 0 and n_unique / series_to_bin.len() > high_cardinality_ratio):
                     print(f"  Skipping numerical analysis for high-cardinality feature '{feature_col}' ({n_unique} unique values).")
                     # Return None to signal skipping this feature in the main report loop
                     # This avoids the problematic categorical fallback for this specific case.
                     return None 
                 # --- End Cardinality Check ---
                 
                 if series_to_bin.len() == 0:
                     raise ValueError("No non-null values to bin.")
                 if series_to_bin.n_unique() <= 1:
                     raise ValueError("Cannot bin feature with <= 1 unique non-null value.")

                 # Calculate quantiles carefully
                 # Use linspace to define quantile points (0 to 1)
                 # Explicitly convert to a Python list of floats
                 quantiles_list = np.linspace(0, 1, num_bins + 1).tolist()
                 
                 # Convert series to NumPy array to use np.quantile for multiple quantiles
                 series_data_np = series_to_bin.to_numpy() # series_to_bin is already null-dropped
                 
                 # Get bin edges using np.quantile
                 # np.quantile handles a list of quantiles correctly
                 # Note: Polars' default interpolation is 'linear', which matches np.quantile's default for older numpy
                 # For newer numpy (>=1.22), 'linear' is an explicit option matching Polars.
                 # Check your numpy version if interpolation behavior is critical and differs.
                 # For simplicity, assuming 'linear' or compatible default behavior.
                 bin_edges = np.quantile(series_data_np, quantiles_list)
                 
                 # Ensure edges are unique, otherwise qcut/cut fails. If not unique, qcut logic won't work well anyway.
                 if len(np.unique(bin_edges)) < len(bin_edges):
                     # Fallback: If too few unique edges (e.g., many identical values), treat as categorical
                     print(f"  Warning: Cannot create {num_bins} unique quantile bins for '{feature_col}' (too few distinct values). Falling back to treating as categorical.")
                     raise ValueError("Duplicate bin edges") # Trigger fallback below

                 # Adjust edges slightly for interval creation if needed (though Polars cut often handles boundaries)
                 # bin_edges[0] = -np.inf # Optionally include lowest values
                 # bin_edges[-1] = np.inf # Optionally include highest values

                 # Create labels with ranges (Format numbers for readability)
                 labels = [
                     f"q{i+1} ({bin_edges[i]:.3g}-{bin_edges[i+1]:.3g})"
                     for i in range(num_bins)
                 ]
                 
                 # Apply the cut using the calculated bin edges
                 # Using cut instead of qcut because we have the edges
                 df_analysis = df_analysis.with_columns(
                     pl.col(feature_col)
                     .cut(breaks=bin_edges[1:-1].tolist(), labels=labels)
                     .alias(bin_col)
                 )
                 # Fill any nulls created by cut (if original feature had nulls handled differently)
                 # Or handle nulls *before* cut if required
                 # If handle_nulls='fill', nulls were already handled, so cut should only affect non-nulls
                 # If handle_nulls='drop', nulls are gone.
                 # If cut somehow produces nulls (e.g., value outside explicit breaks if not using inf), fill them.
                 df_analysis = df_analysis.with_columns(
                     pl.col(bin_col).fill_null("Outside Bins") 
                 )

                 # --- Modification End ---
                 
            except Exception as e:
                 # qcut or cut can fail (e.g., insufficient distinct values, duplicate edges)
                 print(f"  Warning: Binning failed for '{feature_col}' (Error: {e}). Falling back to treating as categorical.")
                 # Fallback: Treat as categorical without binning
                 # Cast to string first for consistent grouping key type
                 df_analysis_cat = df_analysis.with_columns(pl.col(feature_col).cast(pl.Utf8))

                 # *** Important: The categorical fallback should generally NOT run for high-cardinality cases ***
                 # The check added above should `return None` before reaching here for columns like 'id'.
                 # This fallback is intended for when binning fails due to other reasons (e.g., duplicate edges in low-cardinality numerics).

                 summary_df = df_analysis_cat.group_by(feature_col).agg([
                    pl.count().alias("count"),
                    pl.mean(target_col).alias("target_mean"),
                    pl.median(target_col).alias("target_median")
                 ]).sort("count", descending=True)
                 # Ensure output column name matches expectation
                 grouping_col = feature_col 
                 summary_df = summary_df.rename({feature_col: grouping_col}) 
            
            if summary_df is None: # If qcut succeeded
                summary_df = df_analysis.group_by(bin_col).agg([
                    pl.count().alias("count"),
                    pl.mean(target_col).alias("target_mean"),
                    pl.median(target_col).alias("target_median")
                ]).sort(bin_col) # Sort by bin order

        else:
            raise ValueError("Invalid feature_type specified.")
            
    except Exception as e:
        print(f"Error during analysis of feature '{feature_col}': {e}")
        return None

    # Rename the grouping column for consistent output
    if summary_df is not None and grouping_col != "feature_value":
         summary_df = summary_df.rename({grouping_col: "feature_value"})
         # Reorder columns for clarity
         summary_df = summary_df.select(["feature_value", "count", "target_mean", "target_median"])

    return summary_df 