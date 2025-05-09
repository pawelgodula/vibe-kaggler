# Encodes categorical features using a specified scikit-learn encoder.

"""
Extended Description:
This function applies categorical encoding (like OneHotEncoder or OrdinalEncoder)
to specified features in a Polars training DataFrame. It fits the encoder on the
training data (converted to NumPy) and then transforms both the training and an
optional testing Polars DataFrame. It returns the transformed DataFrames (as Polars
DataFrames) and the fitted encoder object. Note that some encoders (like OneHotEncoder)
change the DataFrame's columns. This function handles integrating the new columns and
dropping the original ones.
"""

import polars as pl
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder # Add others as needed
from typing import List, Tuple, Optional, Any

ENCODER_MAP = {
    'onehot': OneHotEncoder,
    'ordinal': OrdinalEncoder,
    # Add other encoders here (e.g., TargetEncoder - requires y)
}

def encode_categorical_features(
    train_df: pl.DataFrame,
    features: List[str],
    test_df: Optional[pl.DataFrame] = None,
    encoder_type: str = 'onehot',
    handle_unknown: str = 'ignore', # Common param for OHE
    drop_original: bool = True,
    new_col_prefix: str = '', # Optional prefix for new columns (esp. for OHE)
    add_one: bool = False, # Add 1 to result for OrdinalEncoder (0-based -> 1-based)
    **encoder_kwargs: Any
) -> Tuple[pl.DataFrame, Optional[pl.DataFrame], Any]:
    """Fits an encoder on training data and transforms train and optionally test data (Polars IO).

    Args:
        train_df (pl.DataFrame): Training DataFrame with features to encode.
        features (List[str]): List of categorical column names to encode.
        test_df (Optional[pl.DataFrame], optional): Testing DataFrame to transform.
                                                     Defaults to None.
        encoder_type (str, optional): Type of encoder ('onehot', 'ordinal').
                                      Defaults to 'onehot'.
        handle_unknown (str, optional): How to handle unknown categories in test data
                                       (passed to some encoders like OHE).
                                       Defaults to 'ignore'.
        drop_original (bool, optional): Whether to drop the original categorical
                                        columns after encoding. Defaults to True.
        new_col_prefix (str, optional): Prefix to add to newly generated column names,
                                        particularly useful for OneHotEncoder. Defaults to ''.
        add_one (bool, optional): If True and encoder_type is 'ordinal', adds 1 to the
                                  encoded integer result. Defaults to False.
        **encoder_kwargs (Any): Additional kwargs for the encoder constructor.

    Returns:
        Tuple[pl.DataFrame, Optional[pl.DataFrame], Any]:
            - Transformed training Polars DataFrame.
            - Transformed test Polars DataFrame (or None).
            - Fitted encoder object.

    Raises:
        ValueError: If an unsupported encoder_type is provided.
        pl.exceptions.ColumnNotFoundError: If features are not found.
        pl.exceptions.ComputeError: If feature conversion or encoding fails.
    """
    train_df_out = train_df.clone()
    test_df_out = test_df.clone() if test_df is not None else None

    encoder_cls = ENCODER_MAP.get(encoder_type.lower())
    if encoder_cls is None:
        raise ValueError(
            f"Unsupported encoder_type: '{encoder_type}'. Supported types: "
            f"{list(ENCODER_MAP.keys())}"
        )

    # Validate features exist
    for df_name, df_obj in [("train_df", train_df), ("test_df", test_df)]:
        if df_obj is None: continue
        missing_features = [f for f in features if f not in df_obj.columns]
        if missing_features:
            raise pl.exceptions.ColumnNotFoundError(f"Features not found in {df_name}: {missing_features}")

    # --- Prepare Encoder ---
    encoder_init_kwargs = encoder_kwargs.copy()
    if encoder_type.lower() == 'onehot':
        encoder_init_kwargs.setdefault('handle_unknown', handle_unknown)
        encoder_init_kwargs.setdefault('sparse_output', False)
    elif encoder_type.lower() == 'ordinal':
        if handle_unknown in ['use_encoded_value']:
             encoder_init_kwargs['handle_unknown'] = handle_unknown
             if 'unknown_value' not in encoder_init_kwargs:
                  # Default unknown_value to np.nan for ordinal when handling unknown
                  encoder_init_kwargs['unknown_value'] = np.nan 
        elif handle_unknown == 'error': # Default for OrdinalEncoder
             encoder_init_kwargs.setdefault('handle_unknown', 'error')
        # 'ignore' is not directly supported by OrdinalEncoder in the same way as OHE.
        # If 'ignore' is passed, it won't be used, and the default ('error') applies.
        
    encoder = encoder_cls(**encoder_init_kwargs)

    # --- Extract data as NumPy for sklearn ---
    try:
        # Ensure consistent dtype (e.g., Utf8) before converting, handle potential non-string categoricals?
        # For now, assume features are Utf8 or compatible. Sklearn might handle mixed types.
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

    # --- Fit & Transform ---
    try:
        encoder.fit(train_features_np)
    except Exception as e:
        print(f"Error fitting {encoder_type} encoder on training data: {e}")
        raise

    try:
        transformed_train_np = encoder.transform(train_features_np)
    except Exception as e:
        print(f"Error transforming training data with {encoder_type} encoder: {e}")
        raise

    transformed_test_np = None
    if test_features_np is not None:
        try:
            transformed_test_np = encoder.transform(test_features_np)
        except Exception as e:
            print(f"Error transforming test data with {encoder_type} encoder: {e}")
            raise # Re-raise after logging

    # --- Integrate results back into Polars DataFrames ---
    if isinstance(encoder, OneHotEncoder):
        try:
            new_col_names = [f"{new_col_prefix}{name}" for name in encoder.get_feature_names_out(features)]
            
            # Create Polars DF from the numpy array
            transformed_train_pl = pl.DataFrame(transformed_train_np, schema=new_col_names)
            # Add the new columns - use hstack for horizontal concatenation
            train_df_out = pl.concat([train_df_out, transformed_train_pl], how="horizontal")

            if test_df_out is not None and transformed_test_np is not None:
                transformed_test_pl = pl.DataFrame(transformed_test_np, schema=new_col_names)
                test_df_out = pl.concat([test_df_out, transformed_test_pl], how="horizontal")
                
        except Exception as e:
            raise pl.exceptions.ComputeError(f"Error integrating OneHotEncoder results: {e}")

    elif isinstance(encoder, OrdinalEncoder):
        try:
            # OrdinalEncoder returns shape (n_samples, n_features)
            # We update the original columns with the encoded values
            updates_train = []
            updates_test = []
            for i, feature_name in enumerate(features):
                new_name = f"{new_col_prefix}{feature_name}" if new_col_prefix else feature_name
                # OrdinalEncoder might return floats if unknown_value=np.nan, otherwise ints
                # Cast explicitly for consistency? Let's use Float64 if np.nan is possible.
                dtype = pl.Float64 if encoder_init_kwargs.get('handle_unknown') == 'use_encoded_value' and encoder_init_kwargs.get('unknown_value') is np.nan else pl.Int64
                encoded_series_train = pl.Series(new_name, transformed_train_np[:, i])
                if add_one:
                    encoded_series_train = encoded_series_train + 1
                updates_train.append(encoded_series_train.cast(dtype))
                if test_df_out is not None and transformed_test_np is not None:
                    encoded_series_test = pl.Series(new_name, transformed_test_np[:, i])
                    if add_one:
                        encoded_series_test = encoded_series_test + 1
                    updates_test.append(encoded_series_test.cast(dtype))
                     
            train_df_out = train_df_out.with_columns(updates_train)
            if test_df_out is not None:
                test_df_out = test_df_out.with_columns(updates_test)
                
        except Exception as e:
             raise pl.exceptions.ComputeError(f"Error integrating OrdinalEncoder results: {e}")
    else:
        # Handle other encoders if added later
        raise NotImplementedError(f"Integration logic not implemented for encoder type: {type(encoder)}")

    # --- Drop original columns if requested ---
    if drop_original:
        try:
            train_df_out = train_df_out.drop(features)
            if test_df_out is not None:
                test_df_out = test_df_out.drop(features)
        except pl.exceptions.ColumnNotFoundError:
             # This might happen if OrdinalEncoder was used without a prefix
             # and drop_original=False, then trying to drop again? Should be safe.
             pass

    return train_df_out, test_df_out, encoder 