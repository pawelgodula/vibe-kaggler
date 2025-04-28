# Function to calculate feature importance using the Leave-One-Out method.

"""
Extended Description:
Calculates feature importance by evaluating the change in model performance when 
each feature is individually removed from the feature set. It first establishes a 
baseline score using all provided features via cross-validation. Then, for each 
feature, it calculates the cross-validated score without that single feature. 
The importance of a feature is typically measured as the difference (or ratio) 
between the baseline score and the score obtained after removing the feature. A larger 
drop in performance when a feature is removed indicates higher importance.
"""

import polars as pl
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import math

# Needs the helper function to evaluate CV score
from ._evaluate_features_cv import _evaluate_features_cv

def calculate_loo_feature_importance(
    train_df: pl.DataFrame,
    target_col: str,
    features: List[str],
    cv_indices: List[Tuple[np.ndarray, np.ndarray]],
    model_type: str,
    model_params: Dict[str, Any],
    fit_params: Dict[str, Any],
    metric_name: str,
    metric_params: Dict[str, Any],
    higher_is_better: bool,
    cat_features: Optional[List[str]] = None
) -> Dict[str, float]:
    """Calculates Leave-One-Out (LOO) feature importance.

    Args:
        train_df (pl.DataFrame): Full training data.
        target_col (str): Target column name.
        features (List[str]): List of features to evaluate importance for.
        cv_indices (List[Tuple[np.ndarray, np.ndarray]]): Pre-calculated CV fold indices.
        model_type (str): Model type to use for evaluation (e.g., 'lgbm').
        model_params (Dict[str, Any]): Model parameters.
        fit_params (Dict[str, Any]): Model fitting parameters.
        metric_name (str): Metric name for evaluation (e.g., 'rmse', 'auc').
        metric_params (Dict[str, Any]): Keyword arguments for the metric function.
        higher_is_better (bool): True if a higher metric score is better.
        cat_features (Optional[List[str]], optional): Categorical features for the model.
            Defaults to None.

    Returns:
        Dict[str, float]: A dictionary mapping each feature name to its importance score. 
                          The importance is calculated as the difference: 
                          baseline_score - score_without_feature (if higher_is_better=True) 
                          or score_without_feature - baseline_score (if higher_is_better=False).
                          A positive score generally indicates importance.
    """
    if not features:
        print("Warning: No features provided for importance calculation.")
        return {}

    # 1. Calculate baseline score with all features
    print(f"Calculating baseline CV score with {len(features)} features...")
    try:
        baseline_score = _evaluate_features_cv(
            train_df=train_df,
            target_col=target_col,
            feature_cols=features,
            cv_indices=cv_indices,
            model_type=model_type,
            model_params=model_params,
            fit_params=fit_params,
            metric_name=metric_name,
            metric_params=metric_params,
            cat_features=cat_features
        )
        print(f"Baseline score: {baseline_score:.6f}")
    except Exception as e:
        print(f"Error calculating baseline score: {e}")
        print("Cannot proceed with LOO importance calculation.")
        return {}

    feature_importance: Dict[str, float] = {}

    # 2. Iterate through each feature, remove it, and evaluate
    print("\nCalculating importance for each feature...")
    for feature_to_remove in features:
        print(f"  Evaluating importance of '{feature_to_remove}'...")
        current_features_to_try = [f for f in features if f != feature_to_remove]

        if not current_features_to_try:
            print(f"    Skipping '{feature_to_remove}': Removing it leaves no features.")
            # Assign a neutral importance score or handle as needed
            feature_importance[feature_to_remove] = 0.0 
            continue
            
        try:
            score_without_feature = _evaluate_features_cv(
                train_df=train_df,
                target_col=target_col,
                feature_cols=current_features_to_try,
                cv_indices=cv_indices,
                model_type=model_type,
                model_params=model_params,
                fit_params=fit_params,
                metric_name=metric_name,
                metric_params=metric_params,
                cat_features=cat_features
            )
            
            # 3. Calculate importance score
            if higher_is_better:
                importance = baseline_score - score_without_feature
            else:
                importance = score_without_feature - baseline_score
                
            feature_importance[feature_to_remove] = importance
            print(f"    Score without '{feature_to_remove}': {score_without_feature:.6f}. Importance: {importance:.6f}")

        except Exception as e:
            print(f"    Error evaluating removal of '{feature_to_remove}': {e}")
            # Assign a default importance (e.g., 0 or NaN) or skip
            feature_importance[feature_to_remove] = 0.0 # Or math.nan if preferred

    print("\nLOO Feature Importance Calculation Complete.")
    
    # Sort by importance (descending) for better readability
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_importance 