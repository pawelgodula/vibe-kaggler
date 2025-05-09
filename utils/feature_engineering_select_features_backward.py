# Function to perform backward feature selection (elimination).

"""
Extended Description:
Performs backward feature selection (elimination) by iteratively removing features 
from an initial set. In each step, it evaluates the impact of removing each single 
feature using cross-validation with a specified model and metric. The feature whose 
removal causes the least performance degradation (or largest improvement, if applicable) 
is permanently removed. The process stops when removing any feature significantly 
hurts performance or a target number of features is reached.
"""

import polars as pl
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import math

# Needs the helper function to evaluate CV score
from ._evaluate_features_cv import _evaluate_features_cv # TODO: Check if this needs renaming too

def select_features_backward(
    train_df: pl.DataFrame,
    target_col: str,
    initial_features: List[str],
    cv_indices: List[Tuple[np.ndarray, np.ndarray]],
    model_type: str,
    model_params: Dict[str, Any],
    fit_params: Dict[str, Any],
    metric_name: str,
    metric_params: Dict[str, Any],
    higher_is_better: bool,
    cat_features: Optional[List[str]] = None,
    min_features: int = 1, # Stop when this number of features is reached
    score_threshold: Optional[float] = None # Stop if score drops below this threshold
) -> List[str]:
    """Performs backward feature selection.

    Args:
        train_df (pl.DataFrame): Full training data.
        target_col (str): Target column name.
        initial_features (List[str]): Features to start with (usually all candidates).
        cv_indices (List[Tuple[np.ndarray, np.ndarray]]): Pre-calculated CV fold indices.
        model_type (str): Model type to use for evaluation (e.g., 'lgbm').
        model_params (Dict[str, Any]): Model parameters.
        fit_params (Dict[str, Any]): Model fitting parameters.
        metric_name (str): Metric name for evaluation (e.g., 'rmse', 'auc').
        metric_params (Dict[str, Any]): Keyword arguments for the metric function.
        higher_is_better (bool): True if a higher metric score is better.
        cat_features (Optional[List[str]], optional): Categorical features for the model.
            Defaults to None.
        min_features (int, optional): Minimum number of features to keep. Defaults to 1.
        score_threshold (Optional[float], optional): If provided, stop if the CV score 
            drops below this value (for higher_is_better=True) or rises above this value 
            (for higher_is_better=False). Defaults to None.

    Returns:
        List[str]: The final list of selected features.
    """
    selected_features = list(initial_features)
    
    if len(selected_features) <= min_features:
        print(f"Initial feature count ({len(selected_features)}) is already at or below min_features ({min_features}). Skipping backward selection.")
        return selected_features

    # Calculate initial score with all features
    best_score = _evaluate_features_cv(
        train_df=train_df,
        target_col=target_col,
        feature_cols=selected_features,
        cv_indices=cv_indices,
        model_type=model_type,
        model_params=model_params,
        fit_params=fit_params,
        metric_name=metric_name,
        metric_params=metric_params,
        cat_features=cat_features
    )
    print(f"Initial score with {len(selected_features)} features: {best_score:.6f}")

    while len(selected_features) > min_features:
        scores_this_iteration = {}
        print(f"\n--- Iteration: Evaluating removal of {len(selected_features)} features ---")
        
        candidate_features_to_remove = list(selected_features) # Features to try removing
        
        for candidate_to_remove in candidate_features_to_remove:
            current_features_to_try = [f for f in selected_features if f != candidate_to_remove]
            
            if not current_features_to_try:
                 print(f"  Cannot evaluate removal of '{candidate_to_remove}', would leave no features.")
                 scores_this_iteration[candidate_to_remove] = -math.inf if higher_is_better else math.inf # Penalize heavily
                 continue
                 
            try:
                score = _evaluate_features_cv(
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
                scores_this_iteration[candidate_to_remove] = score
            except Exception as e:
                 print(f"  Skipping removal of '{candidate_to_remove}' due to error during evaluation: {e}")
                 # Assign a terrible score so this removal isn't chosen
                 scores_this_iteration[candidate_to_remove] = -math.inf if higher_is_better else math.inf

        if not scores_this_iteration:
             print("No candidates could be evaluated in this iteration. Stopping.")
             break
             
        # Find the feature whose removal hurts the score *the least*
        # (i.e., results in the best score among all removal options)
        if higher_is_better:
            best_candidate_to_remove = max(scores_this_iteration, key=scores_this_iteration.get)
            best_score_after_removal = scores_this_iteration[best_candidate_to_remove]
        else:
            best_candidate_to_remove = min(scores_this_iteration, key=scores_this_iteration.get)
            best_score_after_removal = scores_this_iteration[best_candidate_to_remove]
            
        print(f"Best candidate to remove this iteration: '{best_candidate_to_remove}' resulting score {best_score_after_removal:.6f}")

        # Check if the best score after removal is still acceptable
        is_acceptable = True
        if score_threshold is not None:
             if higher_is_better and best_score_after_removal < score_threshold:
                 is_acceptable = False
             elif not higher_is_better and best_score_after_removal > score_threshold:
                 is_acceptable = False
        
        # Also check if the score degraded significantly compared to the *previous* best score
        # (or if it actually improved by removing the feature!)
        score_change = best_score_after_removal - best_score if higher_is_better else best_score - best_score_after_removal
        
        # Decide whether to remove the feature
        # Remove if: 
        # 1) The score after removal is still acceptable (above threshold) AND
        # 2) Removing this feature resulted in the smallest drop (or even an improvement) 
        #    compared to removing other features (this is implicitly handled by finding the max/min score).
        # We essentially remove the feature that gives the best score *after* removal, provided that score is acceptable.
        
        if is_acceptable:
            print(f"Removing feature '{best_candidate_to_remove}'. Score change vs previous best: {score_change:.6f}")
            selected_features.remove(best_candidate_to_remove)
            best_score = best_score_after_removal # Update the best score to the score after removal
        else:
            print(f"Stopping removal. Best score after removal ({best_score_after_removal:.6f}) is below threshold or removal degrades score too much.")
            break # Stop if the best removal option is unacceptable

    print(f"\nBackward selection finished. Selected {len(selected_features)} features:")
    print(selected_features)
    return selected_features


if __name__ == '__main__':
    # --- Example Usage of Backward Feature Selection --- 
    print("\n--- Example: Backward Feature Selection ---")
    
    # 0. Generate dummy data (Polars DataFrame)
    N_samples = 200
    rng = np.random.RandomState(42)
    data = pl.DataFrame({
        'feature1': rng.rand(N_samples),
        'feature2': rng.rand(N_samples) * 0.5,
        'feature3_noisy': rng.rand(N_samples) * 0.1, # Less important
        'feature4': rng.rand(N_samples) * 2,
        'feature5_irrelevant': rng.rand(N_samples), # Irrelevant
        'target': 2 * rng.rand(N_samples) + 3 * rng.rand(N_samples) * 0.5 + 0.5 * rng.rand(N_samples) * 2 + rng.randn(N_samples) * 0.1
    })
    # Correcting target based on actual features used in its generation
    data = data.with_columns(
        (2 * pl.col('feature1') + 3 * pl.col('feature2') + 0.5 * pl.col('feature4') + 
         rng.randn(N_samples) * 0.1).alias('target')
    )

    target_name = 'target'
    all_features = ['feature1', 'feature2', 'feature3_noisy', 'feature4', 'feature5_irrelevant']
    
    # 1. Create CV indices (e.g., 5-fold)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_idx_list = list(kf.split(data))

    # 2. Define model and metric parameters
    lgbm_params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'n_estimators': 50, # Keep it small for example
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'num_leaves': 10,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        'boosting_type': 'gbdt',
    }
    lgbm_fit_params = {
        'eval_metric': 'mae',
        'callbacks': [] # No early stopping for simplicity here
    }
    metric_name_config = 'mae'
    metric_params_config = {}
    higher_is_better_config = False # Lower MAE is better

    try:
        final_selected_features = select_features_backward(
            train_df=data,
            target_col=target_name,
            initial_features=all_features,
            cv_indices=cv_idx_list,
            model_type='lgbm',
            model_params=lgbm_params,
            fit_params=lgbm_fit_params,
            metric_name=metric_name_config,
            metric_params=metric_params_config,
            higher_is_better=higher_is_better_config,
            min_features=2, # Stop when we have at least 2 features
            score_threshold=None # No hard score threshold for this example
        )
        print(f"\nFinal features selected by backward selection: {final_selected_features}")
    except ImportError as e:
        print(f"ImportError during example: {e}. Make sure scikit-learn and lightgbm are installed.")
    except Exception as e:
        print(f"An error occurred during the backward selection example: {e}") 