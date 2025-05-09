# Internal helper function for training an XGBoost model on a single fold.

"""
Extended Description:
This function takes pre-split training and validation data (NumPy arrays or compatible),
model parameters, and fitting parameters, then trains an XGBoost model
(either XGBClassifier or XGBRegressor based on parameters). It handles
early stopping and returns the fitted model, validation predictions, and
test predictions (if test data is provided).
"""

import xgboost as xgb
import numpy as np
import pandas as pd # Keep for consistency, though XGBoost often uses NumPy/DMatrix
from typing import Tuple, Optional, Dict, Any, List

def _train_xgb(
    X_train: pd.DataFrame, # Input as Pandas for consistency, convert to DMatrix internally
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: Optional[pd.DataFrame],
    model_params: Dict[str, Any],
    fit_params: Dict[str, Any],
    feature_cols: List[str],
    cat_features: Optional[List[str]] = None # Note: XGBoost needs specific handling like enable_categorical
) -> Tuple[Any, np.ndarray, Optional[np.ndarray]]:
    """Trains a single XGBoost model."""

    objective = model_params.get('objective', 'reg:squarederror') # Default assumption
    is_classification = 'binary:' in objective or 'multi:' in objective or 'rank:' in objective
    is_multiclass = 'multi:' in objective

    # Convert data to DMatrix for efficiency and potential categorical handling
    # Note: Set enable_categorical=True in model_params if using native cat support
    # Feature names are needed if using categorical features directly
    dtrain = xgb.DMatrix(X_train[feature_cols], label=y_train, feature_names=feature_cols, 
                         enable_categorical=model_params.get("enable_categorical", False))
    dvalid = xgb.DMatrix(X_valid[feature_cols], label=y_valid, feature_names=feature_cols,
                         enable_categorical=model_params.get("enable_categorical", False))
    dtest = xgb.DMatrix(X_test[feature_cols], feature_names=feature_cols, 
                        enable_categorical=model_params.get("enable_categorical", False)) if X_test is not None else None

    evals = [(dtrain, 'train'), (dvalid, 'validation')]
    
    # Prepare fit kwargs
    xgb_fit_params = {}
    if fit_params.get('early_stopping_rounds'):
        xgb_fit_params['early_stopping_rounds'] = fit_params['early_stopping_rounds']
    if fit_params.get('verbose') is not None:
        xgb_fit_params['verbose_eval'] = fit_params['verbose'] # Note: Different key name
    if fit_params.get('eval_metric'):
         # Pass eval_metric via model_params usually in XGBoost
         model_params['eval_metric'] = fit_params['eval_metric'] 
         
    # num_boost_round is often a key parameter for xgboost.train
    num_boost_round = fit_params.get('num_boost_round', 1000) # Default rounds if not set

    print(f"Training XGBoost with params: {model_params}")
    print(f"Fit params: {xgb_fit_params} (num_boost_round={num_boost_round})")

    try:
        # Use the functional API xgb.train for more control with DMatrix
        model = xgb.train(
            params=model_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            **xgb_fit_params
        )
        
        # Get best iteration if early stopping was used
        best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else num_boost_round
        
    except Exception as e:
         print(f"Error during XGBoost model training: {e}")
         raise

    # --- Make Predictions --- 
    try:
        # Use best_iteration for prediction if available
        pred_iteration = best_iteration
        
        if is_classification:
             # predict returns probabilities directly for binary/multi with DMatrix
             val_preds = model.predict(dvalid, iteration_range=(0, pred_iteration))
             test_preds = model.predict(dtest, iteration_range=(0, pred_iteration)) if dtest is not None else None
             # No slicing like [:, 1] needed for binary usually, XGB handles objective
             # If multiclass, output shape is (n_samples, n_classes)
        else: # Regression
            val_preds = model.predict(dvalid, iteration_range=(0, pred_iteration))
            test_preds = model.predict(dtest, iteration_range=(0, pred_iteration)) if dtest is not None else None
            
    except Exception as e:
         print(f"Error during XGBoost prediction: {e}")
         val_preds = np.full(len(X_valid), np.nan)
         test_preds = np.full(len(X_test), np.nan) if X_test is not None else None
         # raise

    # Return model and predictions
    # Note: model here is Booster object from xgb.train
    return model, val_preds, test_preds 