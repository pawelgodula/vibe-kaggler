# Scales numerical features using a specified scikit-learn scaler.

"""
Extended Description:
This function applies a scaling transformation (like StandardScaler or MinMaxScaler)
to specified numerical features in a Polars training DataFrame. It fits the scaler on the
training data (converted to NumPy) and then transforms both the training and an optional
testing Polars DataFrame. It returns the transformed DataFrames (as Polars DataFrames)
and the fitted scaler object.
"""

import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import List, Tuple, Optional, Any

SCALER_MAP = {
    'standard': StandardScaler,
    'minmax': MinMaxScaler,
    'robust': RobustScaler,
}

def scale_numerical_features(
    train_df: pl.DataFrame,
    features: List[str],
    test_df: Optional[pl.DataFrame] = None,
    scaler_type: str = 'standard',
    **scaler_kwargs: Any
) -> Tuple[pl.DataFrame, Optional[pl.DataFrame], Any]:
    """Fits a scaler on training data and transforms train and optionally test data (Polars IO).

    Args:
        train_df (pl.DataFrame): The training DataFrame containing features to scale.
        features (List[str]): A list of numerical column names to scale.
        test_df (Optional[pl.DataFrame], optional): The testing DataFrame to transform.
                                                    Defaults to None.
        scaler_type (str, optional): Type of scaler ('standard', 'minmax', 'robust').
                                     Defaults to 'standard'.
        **scaler_kwargs (Any): Additional kwargs for the scaler constructor.

    Returns:
        Tuple[pl.DataFrame, Optional[pl.DataFrame], Any]:
            - A new Polars DataFrame with scaled training features.
            - A new Polars DataFrame with scaled test features (or None).
            - The fitted scaler object.

    Raises:
        ValueError: If an unsupported scaler_type is provided.
        pl.exceptions.ColumnNotFoundError: If features are not found.
        pl.exceptions.ComputeError: If features are not numerical.
    """
    # Keep original dfs for non-feature columns
    train_df_out = train_df.clone()
    test_df_out = test_df.clone() if test_df is not None else None

    scaler_cls = SCALER_MAP.get(scaler_type.lower())
    if scaler_cls is None:
        raise ValueError(f"Unsupported scaler_type: '{scaler_type}'")

    # Validate features exist and are numeric in Polars
    for df_name, df_obj in [("train_df", train_df), ("test_df", test_df)]:
        if df_obj is None: continue
        missing_features = [f for f in features if f not in df_obj.columns]
        if missing_features:
            raise pl.exceptions.ColumnNotFoundError(f"Features not found in {df_name}: {missing_features}")
        non_numeric_features = [f for f in features if not df_obj[f].dtype.is_numeric()]
        if non_numeric_features:
            raise pl.exceptions.ComputeError(f"Non-numeric features in {df_name}: {non_numeric_features}")

    scaler = scaler_cls(**scaler_kwargs)

    # Extract features as NumPy for scikit-learn
    try:
        train_features_np = train_df.select(features).to_numpy()
    except Exception as e:
        raise pl.exceptions.ComputeError(f"Could not convert train features {features} to NumPy: {e}")
        
    if test_df is not None:
        try:
            test_features_np = test_df.select(features).to_numpy()
        except Exception as e:
             raise pl.exceptions.ComputeError(f"Could not convert test features {features} to NumPy: {e}")
    else:
        test_features_np = None

    # Fit on training data and transform
    try:
        train_features_scaled_np = scaler.fit_transform(train_features_np)
    except Exception as e:
        print(f"Error fitting/transforming training data with {scaler_type} scaler: {e}")
        raise

    # Transform test data if provided
    test_features_scaled_np = None
    if test_features_np is not None:
        try:
            test_features_scaled_np = scaler.transform(test_features_np)
        except Exception as e:
            print(f"Error transforming test data with {scaler_type} scaler: {e}")
            raise

    # Update the Polars DataFrames with scaled features
    # Need to convert scaled numpy array back to Polars Series/DF
    # Important: Ensure correct dtypes (scaling usually results in floats)
    for i, feature_name in enumerate(features):
         train_df_out = train_df_out.with_columns(
             pl.Series(name=feature_name, values=train_features_scaled_np[:, i]).cast(pl.Float64) # Cast to consistent float type
         )
         if test_df_out is not None and test_features_scaled_np is not None:
             test_df_out = test_df_out.with_columns(
                 pl.Series(name=feature_name, values=test_features_scaled_np[:, i]).cast(pl.Float64)
             )

    return train_df_out, test_df_out, scaler 