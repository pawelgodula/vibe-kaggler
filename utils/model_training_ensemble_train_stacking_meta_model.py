# Function to train a meta-model for stacking/ensembling.

"""
Extended Description:
Trains a meta-model using out-of-fold (OOF) predictions from base models as input features.
It takes a DataFrame where columns represent the OOF predictions of different base models 
and the true target variable. It trains the specified meta-model type (e.g., linear model, 
LGBM, NN) on this data.

It also takes a DataFrame of predictions from the same base models on the test set and 
uses the trained meta-model to generate final test set predictions.

This function essentially performs the second level of a stacking ensemble.
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List

# Import the necessary dispatcher/trainers
# We might need train_single_fold if we want to allow complex meta-models with validation
# Or directly use the internal trainers like _train_sklearn, _train_lgbm etc.
from ._train_sklearn import _train_sklearn # TODO: Check if this needs renaming too
from ._train_lgbm import _train_lgbm # TODO: Check if this needs renaming too
from ._train_xgb import _train_xgb # TODO: Check if this needs renaming too
# from ._train_nn import _train_nn # TODO: Check if this needs renaming too

# Import common sklearn models for mapping (similar to train_single_fold)
from sklearn.linear_model import LogisticRegression, Ridge # Common meta-models
# Add others as needed

# Define potential meta-model trainers (can reuse existing internal trainers)
META_MODEL_TRAINERS = {
    'linear': ('sklearn', Ridge), # Default to Ridge for linear
    'logreg': ('sklearn', LogisticRegression),
    'ridge': ('sklearn', Ridge),
    'lgbm': _train_lgbm,
    'xgb': _train_xgb,
    # 'nn': _train_nn,
    # Add other model types if desired as meta-models
}

def train_stacking_meta_model(
    oof_predictions: pl.DataFrame,
    y_true: pl.Series,
    test_predictions: pl.DataFrame,
    meta_model_type: str,
    model_params: Dict[str, Any],
    fit_params: Dict[str, Any]
) -> Tuple[np.ndarray, Any]: # Return test predictions and fitted meta-model
    """Trains a meta-model on OOF predictions and predicts on test predictions.

    Args:
        oof_predictions (pl.DataFrame): DataFrame where columns are OOF predictions 
            from base models (features for the meta-model). Rows must align with y_true.
        y_true (pl.Series): The true target values corresponding to oof_predictions.
        test_predictions (pl.DataFrame): DataFrame where columns are predictions 
            from base models on the test set. Columns must match oof_predictions.
        meta_model_type (str): The type of meta-model to train (e.g., 'linear', 'lgbm').
        model_params (Dict[str, Any]): Parameters for the meta-model constructor.
        fit_params (Dict[str, Any]): Parameters for the meta-model fitting process.

    Returns:
        Tuple[np.ndarray, Any]:
            - final_test_preds: NumPy array of final predictions for the test set.
            - fitted_meta_model: The fitted meta-model object.

    Raises:
        ValueError: If model type is unsupported, data shapes mismatch, or training fails.
        TypeError: If input types are incorrect.
    """
    if not isinstance(oof_predictions, pl.DataFrame) or not isinstance(test_predictions, pl.DataFrame):
        raise TypeError("oof_predictions and test_predictions must be Polars DataFrames.")
    if not isinstance(y_true, pl.Series):
        raise TypeError("y_true must be a Polars Series.")
    if oof_predictions.height != y_true.len():
        raise ValueError("Number of rows in oof_predictions must match length of y_true.")
    if oof_predictions.columns != test_predictions.columns:
        raise ValueError("Columns in oof_predictions must match columns in test_predictions.")

    model_info = META_MODEL_TRAINERS.get(meta_model_type.lower())
    if model_info is None:
        raise ValueError(f"Unsupported meta_model_type: '{meta_model_type}'. Supported: {list(META_MODEL_TRAINERS.keys())}")

    # --- Data Preparation --- 
    # Convert OOF features and test features for the meta-model
    # Most trainers here expect NumPy or Pandas
    X_meta_train: Any
    X_meta_test: Any
    y_meta_train = y_true.to_numpy()
    
    # Determine trainer and if it's sklearn
    is_sklearn_meta = False
    model_trainer: Any
    model_class: Any = None
    if isinstance(model_info, tuple) and model_info[0] == 'sklearn':
        model_trainer = _train_sklearn
        model_class = model_info[1]
        is_sklearn_meta = True
        # Sklearn trainer expects NumPy
        X_meta_train = oof_predictions.to_numpy()
        X_meta_test = test_predictions.to_numpy()
    elif meta_model_type.lower() in ['lgbm', 'xgb']:
        model_trainer = model_info 
        # LGBM/XGB can use Pandas
        X_meta_train = oof_predictions.to_pandas()
        X_meta_test = test_predictions.to_pandas()
    # Add case for NN if implemented (expects NumPy)
    # elif meta_model_type.lower() in ['nn', 'mlp']:
    #     model_trainer = model_info
    #     X_meta_train = oof_predictions.to_numpy()
    #     X_meta_test = test_predictions.to_numpy()
    else:
        # Fallback or raise error for unhandled types
        raise ValueError(f"Meta model type '{meta_model_type}' data conversion not implemented yet.")

    # --- Train Meta-Model --- 
    # Note: We train on the *entire* OOF set. No validation set needed here for the basic internal trainers.
    # If a more complex meta-model training requiring validation was needed, 
    # we might need to pass CV indices here or use train_single_fold recursively.
    
    try:
        fitted_meta_model, _, final_test_preds_opt = model_trainer(
            X_train=X_meta_train,
            y_train=y_meta_train,
            X_valid=None, # No validation set used for fitting meta-model on OOF
            y_valid=None,
            X_test=X_meta_test,
            model_params=model_params,
            fit_params=fit_params,
            # Only pass model_class if using the generic sklearn trainer
            **(dict(model_class=model_class) if is_sklearn_meta else {}),
        )
        
        if final_test_preds_opt is None:
             # This should not happen if test_predictions were provided
             raise ValueError("Meta-model failed to produce test predictions.")
             
        final_test_preds = final_test_preds_opt

    except Exception as e:
        raise ValueError(f"Failed to train or predict with meta-model {meta_model_type}: {e}") from e

    return final_test_preds, fitted_meta_model


if __name__ == '__main__':
    # --- Example Usage --- 
    print("Testing train_stacking_meta_model function...")
    
    N_samples = 100
    N_test = 50
    N_base_models = 3
    rng = np.random.RandomState(42)

    # Create dummy OOF predictions (features for meta-model)
    oof_dict = {}
    for i in range(N_base_models):
        oof_dict[f'base_model_{i+1}_oof'] = rng.rand(N_samples)
    oof_df = pl.DataFrame(oof_dict)

    # Create dummy true target values
    true_target = pl.Series("target", rng.rand(N_samples) + 
                                     0.5 * oof_df['base_model_1_oof'] + 
                                     0.3 * oof_df['base_model_2_oof'] + 
                                     0.2 * oof_df['base_model_3_oof'])

    # Create dummy test predictions from base models
    test_pred_dict = {}
    for i in range(N_base_models):
        test_pred_dict[f'base_model_{i+1}_oof'] = rng.rand(N_test) # Column names MUST match oof_df
    test_df = pl.DataFrame(test_pred_dict)

    print("\nOOF Predictions (Meta-Features):")
    print(oof_df.head())
    print("\nTrue Target:")
    print(true_target.head().to_frame())
    print("\nBase Model Test Predictions:")
    print(test_df.head())

    # Example 1: Linear Meta-Model (Ridge)
    print("\n--- Testing with Linear Meta-Model (Ridge) ---")
    linear_model_params = {'alpha': 1.0}
    linear_fit_params = {}
    try:
        linear_test_preds, linear_meta_model = train_stacking_meta_model(
            oof_df, true_target, test_df, 
            meta_model_type='ridge', 
            model_params=linear_model_params, 
            fit_params=linear_fit_params
        )
        print(f"Linear meta-model fitted: {type(linear_meta_model)}")
        print(f"Linear meta-model coefs: {linear_meta_model.coef_ if hasattr(linear_meta_model, 'coef_') else 'N/A'}")
        print(f"Final test predictions (linear meta) shape: {linear_test_preds.shape}, first 5: {linear_test_preds[:5]}")
    except Exception as e:
        print(f"Error during Linear meta-model test: {e}")

    # Example 2: LGBM Meta-Model
    print("\n--- Testing with LGBM Meta-Model ---")
    lgbm_meta_model_params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'n_estimators': 30, # Small for example
        'learning_rate': 0.05,
        'num_leaves': 5,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 43
    }
    lgbm_meta_fit_params = {'callbacks': []}
    try:
        lgbm_test_preds, lgbm_meta_model = train_stacking_meta_model(
            oof_df, true_target, test_df, 
            meta_model_type='lgbm', 
            model_params=lgbm_meta_model_params, 
            fit_params=lgbm_meta_fit_params
        )
        print(f"LGBM meta-model fitted: {type(lgbm_meta_model)}")
        # print(f"LGBM feature importances: {lgbm_meta_model.feature_importances_}")
        print(f"Final test predictions (LGBM meta) shape: {lgbm_test_preds.shape}, first 5: {lgbm_test_preds[:5]}")
    except Exception as e:
        print(f"Error during LGBM meta-model test: {e}")

    # Example 3: Error case - Mismatched columns in test_predictions
    print("\n--- Testing Error Case: Mismatched Test Columns ---")
    test_df_mismatch = test_df.rename({'base_model_1_oof': 'mismatched_col_name'})
    try:
        train_stacking_meta_model(
            oof_df, true_target, test_df_mismatch, 
            meta_model_type='ridge', model_params={}, fit_params={}
        )
    except ValueError as ve:
        print(f"Caught expected ValueError: {ve}")
    except Exception as e:
        print(f"Caught unexpected error: {e}") 