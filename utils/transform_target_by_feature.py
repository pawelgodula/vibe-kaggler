# Utility functions for normalizing target by a feature and denormalizing predictions.

"""
Extended Description:
Provides functions to normalize a target variable based on a feature (e.g., 
predicting `listening_time / podcast_length` instead of `listening_time`) and 
to reverse this transformation on the model predictions.

- `normalize_target_by_feature`: Normalizes the target by dividing or subtracting a feature.
- `denormalize_predictions_by_feature`: Applies the inverse operation to model predictions.
"""

import polars as pl
from typing import Optional, Literal

SUPPORTED_OPERATIONS = Literal['divide', 'subtract']

def normalize_target_by_feature(
    df: pl.DataFrame,
    target_col: str,
    feature_col: str,
    operation: SUPPORTED_OPERATIONS = 'divide',
    new_target_col: Optional[str] = None,
    fill_value_on_error: Optional[float] = None # Value to use if feature is zero/null for division
) -> pl.DataFrame:
    """Normalizes the target column using a feature column.

    Args:
        df (pl.DataFrame): Input DataFrame.
        target_col (str): Name of the target column to normalize.
        feature_col (str): Name of the feature column to use for normalization.
        operation (Literal['divide', 'subtract'], optional): How to normalize. 
            Defaults to 'divide'.
        new_target_col (Optional[str], optional): If provided, the name for the new 
            normalized target column. If None, the original target_col is replaced. 
            Defaults to None.
        fill_value_on_error (Optional[float], optional): Value to fill in the normalized 
            target if the feature column is zero/null during division operation. 
            If None, the result will be null in those cases. Defaults to None.

    Returns:
        pl.DataFrame: DataFrame with the normalized target column.

    Raises:
        pl.exceptions.ColumnNotFoundError: If target_col or feature_col are not found.
        TypeError: If columns are not numeric.
        ValueError: If an unsupported operation is specified.
    """
    if target_col not in df.columns:
        raise pl.exceptions.ColumnNotFoundError(f"Target column '{target_col}' not found.")
    if feature_col not in df.columns:
        raise pl.exceptions.ColumnNotFoundError(f"Feature column '{feature_col}' not found.")

    if not df[target_col].dtype.is_numeric():
        raise TypeError(f"Target column '{target_col}' must be numeric.")
    if not df[feature_col].dtype.is_numeric():
        raise TypeError(f"Feature column '{feature_col}' must be numeric.")

    output_col = new_target_col if new_target_col is not None else target_col
    
    norm_expr: Optional[pl.Expr] = None
    target = pl.col(target_col)
    feature = pl.col(feature_col)

    if operation == 'divide':
        # Handle division by zero or null feature value
        norm_expr = (
            pl.when(feature.is_null() | target.is_null()) # If feature or target is null, result is null
            .then(pl.lit(None, dtype=pl.Float64))
            .when(feature == 0) # If feature is zero (and not null), use fill_value
            .then(pl.lit(fill_value_on_error, dtype=pl.Float64))
            .otherwise(target / feature) # Otherwise, perform division
        )
    elif operation == 'subtract':
        # Subtraction typically doesn't need special handling unless null propagation is undesired
        # Polars propagates nulls by default: target - null -> null
        norm_expr = target - feature
    else:
        raise ValueError(f"Unsupported operation: '{operation}'. Supported: {SUPPORTED_OPERATIONS}")

    if norm_expr is not None:
        return df.with_columns(norm_expr.alias(output_col).cast(pl.Float64))
    else:
        # Should not happen with current logic, but as a fallback
        return df

def denormalize_predictions_by_feature(
    predictions: pl.Series,
    feature_values: pl.Series,
    operation: SUPPORTED_OPERATIONS = 'divide'
) -> pl.Series:
    """Denormalizes predictions using the corresponding feature values.

    Args:
        predictions (pl.Series): The model predictions on the normalized scale.
        feature_values (pl.Series): The feature values used for the original 
            normalization. Must be aligned row-wise with predictions.
        operation (Literal['divide', 'subtract'], optional): The *original* 
            normalization operation that was applied ('divide' or 'subtract'). 
            The function applies the inverse. Defaults to 'divide'.

    Returns:
        pl.Series: Predictions denormalized to the original target scale.

    Raises:
        ValueError: If prediction length doesn't match feature_values length,
                    or if an unsupported operation is specified.
        TypeError: If input Series are not numeric.
    """
    if predictions.len() != feature_values.len():
        raise ValueError("Length of predictions Series must match length of feature_values Series.")
    if not predictions.dtype.is_numeric():
        raise TypeError("predictions Series must be numeric.")
    if not feature_values.dtype.is_numeric():
        raise TypeError("feature_values Series must be numeric.")
        
    denorm_preds: Optional[pl.Series] = None

    if operation == 'divide':
        # Inverse of division is multiplication
        # Handle nulls: null * value -> null; value * null -> null
        denorm_preds = predictions * feature_values
    elif operation == 'subtract':
        # Inverse of subtraction is addition
        denorm_preds = predictions + feature_values
    else:
        raise ValueError(f"Unsupported original operation: '{operation}'. Supported: {SUPPORTED_OPERATIONS}")

    if denorm_preds is not None:
        # Attempt to cast to Float64 for consistency
        return denorm_preds.cast(pl.Float64, strict=False) 
    else:
        # Should not happen
        raise RuntimeError("Failed to compute denormalized predictions.")

# Example Usage (if run as script)
if __name__ == '__main__':
    data = {
        'listening_time': [30, 60, 15, 90, 0, 120, 45, None],
        'podcast_length': [60, 60, 30, 100, 50, 0, 45, 100],
        'other_feature': [1, 2, 1, 3, 1, 2, 1, 5]
    }
    df = pl.DataFrame(data)
    print("Original DataFrame:")
    print(df)

    print("\n--- Normalization Examples ---")
    # Normalize by division, create new column, fill with 0.0 on error
    df_norm_div = normalize_target_by_feature(
        df.clone(), 
        target_col='listening_time',
        feature_col='podcast_length',
        operation='divide',
        new_target_col='pct_listened',
        fill_value_on_error=0.0
    )
    print("\nNormalized by Division (pct_listened):")
    print(df_norm_div)

    # Normalize by subtraction, replace original column
    df_norm_sub = normalize_target_by_feature(
        df.clone(),
        target_col='listening_time',
        feature_col='other_feature',
        operation='subtract'
    )
    print("\nNormalized by Subtraction (replaced listening_time):")
    print(df_norm_sub)
    
    print("\n--- Denormalization Examples ---")
    # Assume predictions are percentages from the division example
    # Note: Predictions might be floats even if original was int
    norm_predictions_div = df_norm_div['pct_listened'].rename("pct_listened_pred")
    feature_vals_div = df_norm_div['podcast_length'] # Use from the *same* df for alignment
    print("\nNormalized Predictions (from Division):")
    print(norm_predictions_div)
    print("\nOriginal Feature Values (Podcast Length):")
    print(feature_vals_div)

    # Denormalize (inverse of division)
    denorm_preds_div = denormalize_predictions_by_feature(
        norm_predictions_div,
        feature_vals_div,
        operation='divide' # The original operation was division
    )
    print("\nDenormalized Predictions (Original Scale):")
    print(denorm_preds_div)
    
    # Assume predictions are differences from the subtraction example
    norm_predictions_sub = df_norm_sub['listening_time'].rename("diff_pred")
    feature_vals_sub = df_norm_sub['other_feature']
    print("\nNormalized Predictions (from Subtraction):")
    print(norm_predictions_sub)
    print("\nOriginal Feature Values (Other Feature):")
    print(feature_vals_sub)
    
    # Denormalize (inverse of subtraction)
    denorm_preds_sub = denormalize_predictions_by_feature(
        norm_predictions_sub,
        feature_vals_sub,
        operation='subtract' # The original operation was subtraction
    )
    print("\nDenormalized Predictions (Original Scale):")
    print(denorm_preds_sub) 