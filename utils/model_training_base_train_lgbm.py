# Internal helper function for training a LightGBM model on a single fold.

"""
Extended Description:
This function takes pre-split training and validation data (NumPy arrays or compatible),
model parameters, and fitting parameters, then trains a LightGBM model
(either LGBMClassifier or LGBMRegressor based on parameters). It handles
early stopping and returns the fitted model, validation predictions, and
test predictions (if test data is provided).
"""

import lightgbm as lgb
import numpy as np
import pandas as pd # LightGBM sometimes prefers pandas for feature names
from typing import Tuple, Optional, Dict, Any, List

def _train_lgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: Optional[pd.DataFrame],
    model_params: Dict[str, Any],
    fit_params: Dict[str, Any],
    feature_cols: List[str],
    cat_features: Optional[List[str]] = None
) -> Tuple[Any, np.ndarray, Optional[np.ndarray]]:
    """Trains a single LightGBM model.

    Assumes X_train, X_valid, X_test are Pandas DataFrames for potential
    categorical feature handling within LightGBM.
    """

    objective = model_params.get('objective', 'regression') # Default assumption
    is_classification = 'class' in objective or 'binary' in objective or 'multiclass' in objective

    if is_classification:
        model = lgb.LGBMClassifier(**model_params)
    else:
        model = lgb.LGBMRegressor(**model_params)

    # Prepare eval set
    eval_set = [(X_valid, y_valid)]
    eval_names = ['validation']

    # Prepare fit kwargs, converting keys if necessary (e.g., early_stopping_rounds)
    lgbm_fit_params = {}
    callbacks = []
    if fit_params.get('early_stopping_rounds'):
         callbacks.append(lgb.early_stopping(fit_params['early_stopping_rounds'], verbose=fit_params.get('verbose', False)))
         lgbm_fit_params["callbacks"] = callbacks # Use callbacks key
    # Add other relevant fit parameters like verbose logging callback if needed
    if fit_params.get('verbose') is not None and fit_params.get('verbose') > 0:
         callbacks.append(lgb.log_evaluation(period=fit_params['verbose']))
         lgbm_fit_params["callbacks"] = callbacks
         
    # Map eval_metric if provided in fit_params, otherwise LGBM uses metric from model_params
    if fit_params.get('eval_metric'):
         lgbm_fit_params['eval_metric'] = fit_params['eval_metric']
         
    # Handle categorical features - specify by name if X is Pandas
    if cat_features:
         lgbm_fit_params['categorical_feature'] = cat_features

    print(f"Training LGBM with params: {model_params}")
    print(f"Fit params: {lgbm_fit_params}")

    try:
        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            **lgbm_fit_params
        )
    except Exception as e:
         print(f"Error during LightGBM model.fit: {e}")
         raise

    # --- Make Predictions --- 
    try:
        if is_classification:
            # Predict probabilities for the positive class (or all classes if multiclass)
            if model_params.get('objective') == 'multiclass':
                 val_preds = model.predict_proba(X_valid)
                 test_preds = model.predict_proba(X_test) if X_test is not None else None
            else: # Binary or custom classification assumed to need proba of positive class
                 val_preds = model.predict_proba(X_valid)[:, 1]
                 test_preds = model.predict_proba(X_test)[:, 1] if X_test is not None else None
        else: # Regression
            val_preds = model.predict(X_valid)
            test_preds = model.predict(X_test) if X_test is not None else None
            
    except Exception as e:
         print(f"Error during LightGBM prediction: {e}")
         # Return None for predictions if they fail?
         val_preds = np.full(len(X_valid), np.nan) # Fill with NaN on error
         test_preds = np.full(len(X_test), np.nan) if X_test is not None else None
         # Optionally re-raise the error depending on desired behavior
         # raise

    return model, val_preds, test_preds 