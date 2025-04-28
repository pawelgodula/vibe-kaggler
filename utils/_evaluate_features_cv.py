# Internal helper function to evaluate a feature set using cross-validation.

"""
Extended Description:
Runs cross-validation for a given set of features using a specified model 
and metric. It trains the model on each training fold and evaluates on the 
corresponding validation fold. It aggregates the out-of-fold validation 
predictions and calculates the final CV score using the specified metric.

This function is primarily intended as a helper for feature selection routines.
"""

import polars as pl
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# Needs access to train_single_fold and calculate_metric
from .train_single_fold import train_single_fold
from .calculate_metric import calculate_metric 

def _evaluate_features_cv(
    train_df: pl.DataFrame,
    target_col: str,
    feature_cols: List[str],
    cv_indices: List[Tuple[np.ndarray, np.ndarray]],
    model_type: str,
    model_params: Dict[str, Any],
    fit_params: Dict[str, Any],
    metric_name: str,
    metric_params: Dict[str, Any], # Params for the metric function
    cat_features: Optional[List[str]] = None 
) -> float:
    """Evaluates a feature set using CV and returns the score.

    Args:
        train_df (pl.DataFrame): The full training DataFrame.
        target_col (str): Name of the target column.
        feature_cols (List[str]): The specific set of features to evaluate.
        cv_indices (List[Tuple[np.ndarray, np.ndarray]]): List of (train_idx, valid_idx) tuples.
        model_type (str): Type of model to train (e.g., 'lgbm', 'xgb', 'rf').
        model_params (Dict[str, Any]): Parameters for the model constructor.
        fit_params (Dict[str, Any]): Parameters for the model fitting process.
        metric_name (str): Name of the metric to calculate (e.g., 'rmse', 'auc').
        metric_params (Dict[str, Any]): Keyword arguments for the metric function.
        cat_features (Optional[List[str]], optional): Categorical features for the model.
             Defaults to None.

    Returns:
        float: The overall cross-validation score for the given feature set.
        
    Raises:
        Exception: Propagates exceptions from training or metric calculation.
    """
    if not feature_cols:
        raise ValueError("feature_cols cannot be empty for evaluation.")
        
    oof_preds = np.zeros(len(train_df)) # Initialize OOF predictions array
    y_true = train_df[target_col].to_numpy() # Get true target values

    print(f"Evaluating {len(feature_cols)} features: {feature_cols[:5]}... using {model_type}")

    for fold, (train_idx, valid_idx) in enumerate(cv_indices):
        print(f"  Fold {fold+1}/{len(cv_indices)}")
        train_fold_df = train_df[train_idx]
        valid_fold_df = train_df[valid_idx]
        
        # Filter categorical features that are actually in the current feature set
        current_cat_features = None
        if cat_features:
            current_cat_features = [f for f in cat_features if f in feature_cols]

        try:
            # Train model for the fold (we don't need test preds here)
            _, val_preds, _ = train_single_fold(
                train_fold_df=train_fold_df,
                valid_fold_df=valid_fold_df,
                test_df=None,
                target_col=target_col,
                feature_cols=feature_cols,
                model_type=model_type,
                model_params=model_params,
                fit_params=fit_params,
                cat_features=current_cat_features
            )
            
            # Store OOF predictions for this fold
            oof_preds[valid_idx] = val_preds
            
        except Exception as e:
             print(f"Error during training/prediction in fold {fold+1} for feature evaluation: {e}")
             # Depending on desired robustness, could return NaN/inf or re-raise
             raise e # Re-raise for now

    # Calculate overall OOF metric
    try:
        cv_score = calculate_metric(
            y_true=y_true,
            y_pred=oof_preds,
            metric_name=metric_name,
            **metric_params
        )
        print(f"  Finished evaluation. CV Score ({metric_name}): {cv_score:.6f}")
    except Exception as e:
         print(f"Error calculating final CV metric '{metric_name}': {e}")
         # Return NaN or re-raise? Re-raise seems safer.
         raise e
         
    return cv_score 