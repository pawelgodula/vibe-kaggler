# Function to calculate feature-feature and feature-target correlations.

"""
Extended Description:
Calculates the Pearson correlation coefficient between all pairs of numerical 
features and between each numerical feature and the target variable (if numeric). 
This helps identify linear relationships, potential multicollinearity among 
features, and features strongly associated with the target.

Returns two Polars DataFrames: one containing the full feature-feature 
correlation matrix, and another containing the correlations between each 
feature and the target column.
"""

import polars as pl
from typing import List, Optional, Tuple

def calculate_correlations(
    df: pl.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None
) -> Tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
    """Calculates feature-feature and feature-target correlations for numeric columns.

    Args:
        df (pl.DataFrame): Input DataFrame.
        target_col (str): The name of the target column.
        feature_cols (Optional[List[str]], optional): List of feature columns to consider. 
            If None, all numeric columns except the target column are used. 
            Defaults to None.

    Returns:
        Tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]: A tuple containing:
            - feature_corr_matrix (pl.DataFrame): Correlation matrix between numerical features.
              Returns None if fewer than 2 numerical features are found.
            - target_corr (pl.DataFrame): Correlation of each numerical feature with the target.
              Returns None if the target column is not numeric or no numerical features exist.

    Raises:
        pl.exceptions.ColumnNotFoundError: If target_col or any specified feature_cols 
            are not found in the DataFrame.
    """
    if target_col not in df.columns:
        raise pl.exceptions.ColumnNotFoundError(f"Target column '{target_col}' not found.")

    # Identify numeric features
    if feature_cols:
        if not all(col in df.columns for col in feature_cols):
            missing = [col for col in feature_cols if col not in df.columns]
            raise pl.exceptions.ColumnNotFoundError(f"Feature column(s) not found: {missing}")
        numeric_features = [col for col in feature_cols if df[col].dtype.is_numeric()]
    else:
        numeric_features = [col for col in df.columns if df[col].dtype.is_numeric() and col != target_col]

    # Filter out features with zero standard deviation to avoid NaN in correlation
    if numeric_features:
        stds = df.select(numeric_features).std().row(0)
        features_with_variance = [
            feat for feat, std_val in zip(numeric_features, stds) 
            if std_val is not None and std_val > 1e-9 # Use a small epsilon for float comparison
        ]
        if len(features_with_variance) < len(numeric_features):
            removed_features = set(numeric_features) - set(features_with_variance)
            print(f"Warning: Removed features with zero variance: {removed_features}")
        numeric_features = features_with_variance

    # Check for remaining NaNs in the selected numeric features
    if numeric_features:
        nan_check = df.select(numeric_features).null_count()
        cols_with_nans = [
            col for col in numeric_features if nan_check[col][0] > 0
        ]
        if cols_with_nans:
            print(f"Warning: The following numeric features still contain NaN values after filtering: {cols_with_nans}")
            # Depending on desired behavior, you might want to drop rows with NaNs or fill them
            # For now, we'll just report them.

    is_target_numeric = df[target_col].dtype.is_numeric()

    if not numeric_features:
        print("Warning: No numerical features found to calculate correlations.")
        return None, None

    feature_corr_matrix: Optional[pl.DataFrame] = None
    target_corr: Optional[pl.DataFrame] = None

    # Calculate feature-feature correlations
    if len(numeric_features) >= 2:
        try:
            # Select only the numeric features and drop rows with any NaNs in this subset
            df_ff_corr = df.select(numeric_features).drop_nulls()
            print(f"Original df shape: {df.shape}, Shape after dropping NaNs for feature-feature correlation: {df_ff_corr.shape}")
            feature_corr_matrix = df_ff_corr.corr()
            # Add the feature names as the first column for clarity
            # Ensure feature names list matches the columns in the correlation matrix
            feature_names_in_corr = df_ff_corr.columns # Use columns from the NaN-dropped df
            feature_corr_matrix = feature_corr_matrix.with_columns(
                pl.Series("feature", feature_names_in_corr)
            ).select(["feature"] + feature_names_in_corr)
        except Exception as e:
            print(f"Warning: Could not compute feature-feature correlations: {e}")
            feature_corr_matrix = None
    elif len(numeric_features) < 2:
         print("Warning: Need at least 2 numerical features to calculate feature-feature correlations.")

    # Calculate feature-target correlations
    if is_target_numeric:
        cols_for_target_corr = numeric_features + [target_col]
        try:
            # Select features and target, then drop rows with any NaNs in this subset
            df_ft_corr = df.select(cols_for_target_corr).drop_nulls()
            print(f"Original df shape: {df.shape}, Shape after dropping NaNs for feature-target correlation: {df_ft_corr.shape}")
            full_corr = df_ft_corr.corr()
            # Extract the target column correlations, excluding target-target corr
            # Ensure we use the feature list from the NaN-dropped df (excluding target)
            features_in_ft_corr = [col for col in df_ft_corr.columns if col != target_col]
            target_corr_series = full_corr.select(pl.col(target_col)).get_column(target_col)[:-1] # Excludes target-target corr
            target_corr = pl.DataFrame({
                "feature": features_in_ft_corr, # Features used in this specific corr calculation
                f"{target_col}_correlation": target_corr_series
            }).sort(f"{target_col}_correlation", descending=True, nulls_last=True)
        except Exception as e:
             print(f"Warning: Could not compute feature-target correlations: {e}")
             target_corr = None
    else:
        print(f"Warning: Target column '{target_col}' is not numeric. Cannot calculate feature-target correlations.")

    return feature_corr_matrix, target_corr 