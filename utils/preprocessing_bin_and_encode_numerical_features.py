# Function to bin numerical features and then apply categorical encoding.

"""
Extended Description:
Bins specified numerical features based on the distribution of the training data
(e.g., using quantiles) and then applies a specified categorical encoding method
(e.g., OneHotEncoder, CountEncoder from sklearn or category-encoders) to the resulting bins.
This is useful for converting numerical features into categorical ones that some models
can utilize differently.

Returns the transformed DataFrames and a dictionary containing the fitted encoder
and the bin edges used for each feature.
"""

import polars as pl
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Literal

# Assuming presence of encode_categorical_features utility or similar logic
# For simplicity, let's use sklearn's OneHotEncoder here directly for now
from sklearn.preprocessing import OneHotEncoder
# We might need category_encoders for things like CountEncoder
# import category_encoders as ce 

SUPPORTED_BINNING_STRATEGIES = Literal['quantile', 'uniform']
SUPPORTED_ENCODERS = Literal['onehot'] # Extend as needed (e.g., 'count', 'target')

def bin_and_encode_numerical_features(
    train_df: pl.DataFrame,
    features: List[str],
    n_bins: int,
    test_df: Optional[pl.DataFrame] = None,
    strategy: SUPPORTED_BINNING_STRATEGIES = 'quantile',
    encoder_type: SUPPORTED_ENCODERS = 'onehot',
    handle_unknown: str = 'ignore', # Parameter for OneHotEncoder
    **encoder_kwargs: Any
) -> Tuple[pl.DataFrame, Optional[pl.DataFrame], Dict[str, Any]]:
    """Bins numerical features and then encodes the bins categorically.

    Args:
        train_df (pl.DataFrame): Training DataFrame.
        features (List[str]): List of numerical feature columns to bin and encode.
        n_bins (int): The number of bins to create.
        test_df (Optional[pl.DataFrame], optional): Test DataFrame. Defaults to None.
        strategy (Literal['quantile', 'uniform'], optional): Binning strategy. 
            Defaults to 'quantile'.
        encoder_type (Literal['onehot'], optional): Categorical encoding type to apply 
            to the bins. Defaults to 'onehot'.
        handle_unknown (str, optional): How OneHotEncoder should handle unknown categories
            found during transform. Defaults to 'ignore'.
        **encoder_kwargs: Additional keyword arguments for the encoder (e.g., `sparse_output` for OHE).

    Returns:
        Tuple[pl.DataFrame, Optional[pl.DataFrame], Dict[str, Any]]:
            - train_df_transformed: Training DataFrame with original features replaced by encoded bins.
            - test_df_transformed: Test DataFrame with original features replaced by encoded bins (or None).
            - fitted_params: Dictionary containing fitted encoder object and bin edges per feature.
                             Example: {'encoder': fitted_ohe, 'bin_edges': {'feat1': [...], 'feat2': [...]}}

    Raises:
        ValueError: If features are not found, not numeric, or issues occur during binning/encoding.
        TypeError: If input types are incorrect.
    """
    if not isinstance(train_df, pl.DataFrame):
        raise TypeError("train_df must be a Polars DataFrame.")
    if test_df is not None and not isinstance(test_df, pl.DataFrame):
        raise TypeError("test_df must be a Polars DataFrame or None.")
    if not features:
        raise ValueError("Feature list cannot be empty.")
    if not all(f in train_df.columns for f in features):
        missing = [f for f in features if f not in train_df.columns]
        raise ValueError(f"Features not found in train_df: {missing}")
    if test_df is not None and not all(f in test_df.columns for f in features):
        missing = [f for f in features if f not in test_df.columns]
        raise ValueError(f"Features not found in test_df: {missing}")
        
    # Check if features are numeric
    if not all(train_df[f].dtype.is_numeric() for f in features):
        non_numeric = [f for f in features if not train_df[f].dtype.is_numeric()]
        raise ValueError(f"Features must be numeric for binning: {non_numeric}")

    train_df_out = train_df.clone()
    test_df_out = test_df.clone() if test_df is not None else None
    bin_edges_map: Dict[str, List[float]] = {}
    binned_feature_names: List[str] = []

    # --- Step 1: Binning ---    
    for feature in features:
        train_series = train_df_out[feature]
        binned_col_name = f"{feature}_bin"
        binned_feature_names.append(binned_col_name)

        try:
            # Determine bin edges from training data
            if strategy == 'quantile':
                quantiles_to_calculate = np.linspace(0, 1, n_bins + 1)
                edges = [train_series.quantile(q) for q in quantiles_to_calculate]
            elif strategy == 'uniform':
                min_val = train_series.min()
                max_val = train_series.max()
                if min_val is None or max_val is None or min_val == max_val:
                    # Handle constant or all-null columns - use single edge? Or error?
                    # For simplicity, use min/max, cut might handle it gracefully
                    print(f"Warning: Feature '{feature}' might be constant or all null. Using min/max for uniform binning.")
                    edges = [min_val, max_val] if min_val is not None else []
                else: 
                    edges = np.linspace(min_val, max_val, n_bins + 1).tolist()
            else:
                raise ValueError(f"Unsupported binning strategy: {strategy}")

            # Filter non-None, ensure uniqueness
            valid_edges = sorted(list(set(e for e in edges if e is not None)))
            
            # Handle constant columns or insufficient unique values for bins
            if len(valid_edges) < 2:
                 min_val = train_series.min()
                 max_val = train_series.max()
                 if min_val is not None and max_val is not None and min_val == max_val:
                      print(f"Warning: Feature '{feature}' is constant. Creating a single bin.")
                      # Create edges slightly around the constant value to form one bin
                      # Handle potential float precision issues
                      delta = abs(min_val * 1e-6) if min_val != 0 else 1e-6
                      final_edges = [min_val - delta, min_val + delta]
                 else:
                    # Cannot form bins even with min/max, raise error
                    raise ValueError(f"Could not determine valid bin edges for feature '{feature}'. Requires at least 2 unique finite values.")
            else:
                 # Use the unique edges derived from the strategy
                 final_edges = valid_edges
                 
            bin_edges_map[feature] = final_edges # Store the actual edges used

            # Apply binning using cut. Let cut handle values outside the derived edges.
            # Using left_closed=True: [e1, e2), [e2, e3), ...
            train_df_out = train_df_out.with_columns(
                pl.col(feature).cut(breaks=final_edges, left_closed=True)
                  .alias(binned_col_name)
                  .cast(pl.Categorical) # Ensure categorical type for encoder
            )
            if test_df_out is not None:
                test_df_out = test_df_out.with_columns(
                    pl.col(feature).cut(breaks=final_edges, left_closed=True)
                      .alias(binned_col_name)
                      .cast(pl.Categorical) 
                )

        except Exception as e:
            raise ValueError(f"Failed during binning of feature '{feature}': {e}") from e

    # --- Step 2: Encoding ---    
    fitted_encoder = None
    if not binned_feature_names:
         print("Warning: No binned features generated, skipping encoding.")
         return train_df_out, test_df_out, {'encoder': None, 'bin_edges': bin_edges_map}
         
    # Prepare data for sklearn encoder (needs 2D array)
    X_train_binned = train_df_out.select(binned_feature_names).to_pandas() # OHE works well with Pandas Categoricals
    X_test_binned = test_df_out.select(binned_feature_names).to_pandas() if test_df_out is not None else None

    try:
        if encoder_type == 'onehot':
             # Ensure sparse_output=False unless explicitly requested and handled downstream
             if 'sparse_output' not in encoder_kwargs and 'sparse' not in encoder_kwargs:
                 encoder_kwargs['sparse_output'] = False 
                 
             encoder = OneHotEncoder(handle_unknown=handle_unknown, **encoder_kwargs)
             fitted_encoder = encoder.fit(X_train_binned)
             
             # Transform and get new column names
             X_train_encoded_np = fitted_encoder.transform(X_train_binned)
             new_col_names = fitted_encoder.get_feature_names_out(binned_feature_names)
             # Ensure schema is a list of strings
             schema_list = list(new_col_names) 
             X_train_encoded = pl.DataFrame(X_train_encoded_np, schema=schema_list)

             X_test_encoded = None
             if X_test_binned is not None:
                 X_test_encoded_np = fitted_encoder.transform(X_test_binned)
                 # Use the same schema derived from training
                 X_test_encoded = pl.DataFrame(X_test_encoded_np, schema=schema_list)
        
        # Add cases for other encoder types (e.g., CountEncoder) here
        # elif encoder_type == 'count':
        #     encoder = ce.CountEncoder(cols=binned_feature_names)
        #     fitted_encoder = encoder.fit(X_train_binned) # Assuming CountEncoder needs y? Check docs.
        #     X_train_encoded = fitted_encoder.transform(X_train_binned)
        #     if X_test_binned is not None:
        #         X_test_encoded = fitted_encoder.transform(X_test_binned)
                 
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        # Drop original and intermediate binned columns, add encoded columns
        cols_to_drop = features + binned_feature_names
        train_df_out = train_df_out.drop(cols_to_drop).hstack(X_train_encoded)
        if test_df_out is not None and X_test_encoded is not None:
            test_df_out = test_df_out.drop(cols_to_drop).hstack(X_test_encoded)

    except Exception as e:
        raise ValueError(f"Failed during encoding step with {encoder_type}: {e}") from e

    fitted_params = {
        'encoder': fitted_encoder,
        'bin_edges': bin_edges_map
    }

    return train_df_out, test_df_out, fitted_params


if __name__ == '__main__':
    # Example Usage
    # Create dummy data
    train_data = pl.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'numerical_feat1': [10, 12, 15, 11, 19, 25, 22, 30, 28, 35],
        'numerical_feat2': [1.1, 1.5, 2.2, 1.3, 2.5, 3.1, 2.8, 3.5, 3.2, 4.0],
        'other_col': ['a', 'b', 'a', 'c', 'b', 'a', 'c', 'a', 'b', 'c']
    })
    test_data = pl.DataFrame({
        'id': [11, 12, 13, 14, 15],
        'numerical_feat1': [9, 16, 23, 29, 40],
        'numerical_feat2': [1.0, 2.0, 2.7, 3.3, 4.5],
        'other_col': ['b', 'a', 'c', 'a', 'b']
    })

    print("Original Train DF:")
    print(train_data)
    print("\nOriginal Test DF:")
    print(test_data)

    features_to_bin = ['numerical_feat1', 'numerical_feat2']
    num_bins = 4

    try:
        print(f"\n--- Testing bin_and_encode_numerical_features with {num_bins} quantile bins and one-hot encoding ---")
        train_transformed, test_transformed, params = bin_and_encode_numerical_features(
            train_df=train_data,
            features=features_to_bin,
            n_bins=num_bins,
            test_df=test_data,
            strategy='quantile',
            encoder_type='onehot'
        )
        
        print("\nTransformed Train DF (Quantile):")
        print(train_transformed)
        print("\nTransformed Test DF (Quantile):")
        print(test_transformed)
        print("\nFitted Parameters (Quantile):")
        print(f"  Encoder: {params['encoder']}")
        print(f"  Bin Edges: {params['bin_edges']}")

        # Test uniform binning
        print(f"\n--- Testing bin_and_encode_numerical_features with {num_bins} uniform bins and one-hot encoding ---")
        train_transformed_uni, test_transformed_uni, params_uni = bin_and_encode_numerical_features(
            train_df=train_data,
            features=features_to_bin,
            n_bins=num_bins,
            test_df=test_data,
            strategy='uniform',
            encoder_type='onehot'
        )
        print("\nTransformed Train DF (Uniform):")
        print(train_transformed_uni)
        print("\nTransformed Test DF (Uniform):")
        print(test_transformed_uni)
        print("\nFitted Parameters (Uniform):")
        print(f"  Encoder: {params_uni['encoder']}")
        print(f"  Bin Edges: {params_uni['bin_edges']}")
        
        # Test handling of constant feature
        train_data_const = train_data.with_columns(pl.lit(5).alias('numerical_feat_const'))
        features_to_bin_const = ['numerical_feat1', 'numerical_feat_const']
        print(f"\n--- Testing with a constant feature ---")
        train_transformed_const, _, params_const = bin_and_encode_numerical_features(
            train_df=train_data_const,
            features=features_to_bin_const,
            n_bins=2 # Should still work with constant
        )
        print("\nTransformed Train DF (with Constant Feature):")
        print(train_transformed_const)
        print(f"  Bin Edges for Constant: {params_const['bin_edges']['numerical_feat_const']}")

    except ValueError as ve:
        print(f"\nValueError: {ve}")
    except TypeError as te:
        print(f"\nTypeError: {te}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}") 