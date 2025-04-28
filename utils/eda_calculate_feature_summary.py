# Function to calculate summary statistics for each feature in a DataFrame.

"""
Extended Description:
Computes basic summary statistics for every column in a Polars DataFrame. 
This includes data type, counts and percentages of unique and missing values, 
and standard descriptive statistics (mean, std, min, max, median) for numerical 
columns. Useful for getting a quick overview of the data's characteristics.
"""

import polars as pl
from typing import Dict, Any

def calculate_feature_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Calculates summary statistics for each column.

    Args:
        df (pl.DataFrame): Input DataFrame.

    Returns:
        pl.DataFrame: A DataFrame where each row summarizes a column from the 
                      input DataFrame. Includes statistics like dtype, counts/
                      percentages of unique and missing values, and descriptive 
                      stats for numeric columns.
    """
    summaries = []
    total_rows = len(df)

    for col_name in df.columns:
        col = df[col_name]
        dtype = col.dtype
        n_unique = col.n_unique()
        n_null = col.null_count()
        pct_null = (n_null / total_rows) * 100 if total_rows > 0 else 0
        
        summary: Dict[str, Any] = {
            "Column": col_name,
            "DataType": str(dtype),
            "NumUnique": n_unique,
            "NumNull": n_null,
            "PctNull": round(pct_null, 2),
            "Mean": None,
            "StdDev": None,
            "Median": None,
            "Min": None,
            "Max": None,
        }

        # Add numerical stats if applicable
        if dtype.is_numeric():
            try:
                stats = df.select([
                    pl.mean(col_name).alias("mean"),
                    pl.std(col_name).alias("std"),
                    pl.median(col_name).alias("median"),
                    pl.min(col_name).alias("min"),
                    pl.max(col_name).alias("max"),
                ]).row(0, named=True)
                
                summary["Mean"] = stats.get("mean")
                summary["StdDev"] = stats.get("std")
                summary["Median"] = stats.get("median")
                summary["Min"] = stats.get("min")
                summary["Max"] = stats.get("max")
            except Exception as e:
                 print(f"  Warning: Could not compute numeric stats for column '{col_name}': {e}")
        
        summaries.append(summary)

    if not summaries:
        return pl.DataFrame() # Return empty DataFrame if input was empty
        
    # Create DataFrame from list of dictionaries
    summary_df = pl.from_dicts(summaries)
    
    # Reorder columns for better readability
    ordered_cols = [
        "Column", "DataType", "NumUnique", "NumNull", "PctNull", 
        "Mean", "StdDev", "Median", "Min", "Max"
    ]
    summary_df = summary_df.select(ordered_cols)
    
    return summary_df 