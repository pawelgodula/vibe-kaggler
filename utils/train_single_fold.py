# Trains a model for a single fold of cross-validation.

"""
Extended Description:
Acts as a dispatcher function that takes training and validation data for a
single fold, along with model configuration, and calls the appropriate
model-specific training function (_train_lgbm, _train_xgb, etc.).
It handles the conversion of Polars DataFrames (used for input consistency)
to the format expected by the underlying model trainers (e.g., Pandas or NumPy).
"""

import polars as pl
import pandas as pd # Needed for conversion for some model libs
import numpy as np
from typing import Tuple, Optional, Dict, Any, List

# Import internal trainers
from ._train_lgbm import _train_lgbm
from ._train_xgb import _train_xgb
from ._train_sklearn import _train_sklearn # Import the new sklearn trainer
from ._train_nn import _train_nn # Import the new NN trainer
# Add imports for _train_catboost, _train_nn etc. if implemented

# Import common sklearn models for mapping
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

MODEL_TRAINERS = {
    'lgbm': _train_lgbm,
    'lightgbm': _train_lgbm,
    'xgb': _train_xgb,
    'xgboost': _train_xgb,
    'nn': _train_nn, # Added NN trainer
    'mlp': _train_nn, # Added MLP alias for NN trainer
    # Map sklearn model types to the generic trainer and the model class
    'rf': ('sklearn', RandomForestClassifier), # Default to classifier for simplicity
    'randomforest': ('sklearn', RandomForestClassifier),
    'rf_reg': ('sklearn', RandomForestRegressor),
    'et': ('sklearn', ExtraTreesClassifier),
    'extratrees': ('sklearn', ExtraTreesClassifier),
    'et_reg': ('sklearn', ExtraTreesRegressor),
    'logreg': ('sklearn', LogisticRegression),
    'logisticregression': ('sklearn', LogisticRegression),
    'ridge': ('sklearn', Ridge),
    'lasso': ('sklearn', Lasso),
    'knn': ('sklearn', KNeighborsClassifier),
    'kneighbors': ('sklearn', KNeighborsClassifier),
    'knn_reg': ('sklearn', KNeighborsRegressor),
}

def train_single_fold(
    train_fold_df: pl.DataFrame,
    valid_fold_df: pl.DataFrame,
    test_df: Optional[pl.DataFrame],
    target_col: str,
    feature_cols: List[str],
    model_type: str,
    model_params: Dict[str, Any],
    fit_params: Dict[str, Any],
    cat_features: Optional[List[str]] = None # Pass categorical features if model supports
) -> Tuple[Any, np.ndarray, Optional[np.ndarray]]:
    """Trains a model on a single fold and returns predictions.

    Args:
        train_fold_df (pl.DataFrame): Training data for this fold.
        valid_fold_df (pl.DataFrame): Validation data for this fold.
        test_df (Optional[pl.DataFrame]): Full test data for prediction.
        target_col (str): Name of the target column.
        feature_cols (List[str]): List of feature column names.
        model_type (str): Type of model to train (e.g., 'lgbm', 'xgb').
                          Must be a key in MODEL_TRAINERS.
        model_params (Dict[str, Any]): Parameters for the model constructor.
        fit_params (Dict[str, Any]): Parameters for the model fitting process
                                    (e.g., early stopping, verbosity).
        cat_features (Optional[List[str]], optional): List of categorical features names
            if the model handles them internally. Defaults to None.

    Returns:
        Tuple[Any, np.ndarray, Optional[np.ndarray]]:
            - The fitted model object for this fold.
            - NumPy array of validation predictions.
            - NumPy array of test predictions (or None if test_df is None).

    Raises:
        ValueError: If model_type is not supported or required params missing.
        pl.exceptions.ColumnNotFoundError: If target or feature columns are missing.
    """
    model_info = MODEL_TRAINERS.get(model_type.lower())
    if model_info is None:
        raise ValueError(f"Unsupported model_type: '{model_type}'. Supported: {list(MODEL_TRAINERS.keys())}")

    # Unpack trainer function and potential model class for sklearn
    if isinstance(model_info, tuple) and model_info[0] == 'sklearn':
        model_trainer = _train_sklearn
        model_class = model_info[1]
        is_sklearn = True
    else:
        model_trainer = model_info
        model_class = None # Not needed directly for non-sklearn specialized trainers
        is_sklearn = False

    # --- Data Preparation ---
    # Ensure required columns exist
    required_cols = set(feature_cols) | {target_col}
    if not required_cols.issubset(train_fold_df.columns):
         missing = required_cols - set(train_fold_df.columns)
         raise pl.exceptions.ColumnNotFoundError(f"Columns missing in train_fold_df: {missing}")
    if not required_cols.issubset(valid_fold_df.columns):
         missing = required_cols - set(valid_fold_df.columns)
         raise pl.exceptions.ColumnNotFoundError(f"Columns missing in valid_fold_df: {missing}")
    if test_df is not None and not set(feature_cols).issubset(test_df.columns):
         missing = set(feature_cols) - set(test_df.columns)
         raise pl.exceptions.ColumnNotFoundError(f"Columns missing in test_df: {missing}")
         
    # Extract features and target
    # Convert based on model type
    # NN and Sklearn trainers expect NumPy arrays
    if model_type.lower() in ['nn', 'mlp'] or is_sklearn:
        try:
             X_train = train_fold_df.select(feature_cols).to_numpy()
             y_train = train_fold_df[target_col].to_numpy()
             X_valid = valid_fold_df.select(feature_cols).to_numpy()
             y_valid = valid_fold_df[target_col].to_numpy()
             X_test = test_df.select(feature_cols).to_numpy() if test_df is not None else None
        except Exception as e:
             print(f"Error converting Polars data to NumPy for NN/Sklearn: {e}")
             raise TypeError("Failed to convert data to NumPy.") from e
    # Default to Pandas for others (LGBM/XGB can handle it)
    else:
        try:
             X_train = train_fold_df.select(feature_cols).to_pandas()
             y_train = train_fold_df[target_col].to_numpy() # Keep target as numpy
             X_valid = valid_fold_df.select(feature_cols).to_pandas()
             y_valid = valid_fold_df[target_col].to_numpy()
             X_test = test_df.select(feature_cols).to_pandas() if test_df is not None else None
        except Exception as e:
             print(f"Error converting Polars data to Pandas for model training: {e}")
             raise TypeError("Failed to convert data to required format (Pandas/NumPy).") from e

    # --- Call model-specific trainer ---
    # Note: NN trainer returns state_dict, others return model object
    model, val_preds, test_preds = model_trainer(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        model_params=model_params,
        fit_params=fit_params,
        # Only pass model_class if using the generic sklearn trainer
        **(dict(model_class=model_class) if is_sklearn else {}),
        # Pass feature_cols and cat_features which might be needed by some trainers
        feature_cols=feature_cols, 
        cat_features=cat_features 
    )

    return model, val_preds, test_preds 