# Function to perform forward feature selection.

"""
Extended Description:
Performs forward feature selection by iteratively adding features from a candidate
pool to an initial set. In each step, it evaluates all potential single feature
additions using cross-validation with a specified model and metric. The feature 
that provides the best improvement according to the metric
is permanently added. The process stops when no further improvement is made or all
candidate features are added.
"""

import polars as pl
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import math

# Needs the helper function to evaluate CV score
from ._evaluate_features_cv import _evaluate_features_cv # TODO: Check if this needs renaming too

def select_features_forward(
    train_df: pl.DataFrame,
    target_col: str,
    initial_features: List[str],
    candidate_features: List[str],
    cv_indices: List[Tuple[np.ndarray, np.ndarray]],
    model_type: str,
    model_params: Dict[str, Any],
    fit_params: Dict[str, Any],
    metric_name: str,
    metric_params: Dict[str, Any],
    higher_is_better: bool,
    cat_features: Optional[List[str]] = None,
    min_improvement: float = 1e-5 # Minimum improvement required to add a feature
) -> List[str]:
    """Performs forward feature selection.

    Args:
        train_df (pl.DataFrame): Full training data.
        target_col (str): Target column name.
        initial_features (List[str]): Features to start with. Can be empty.
        candidate_features (List[str]): Features to potentially add.
        cv_indices (List[Tuple[np.ndarray, np.ndarray]]): Pre-calculated CV fold indices.
        model_type (str): Model type to use for evaluation (e.g., 'lgbm').
        model_params (Dict[str, Any]): Model parameters.
        fit_params (Dict[str, Any]): Model fitting parameters.
        metric_name (str): Metric name for evaluation (e.g., 'rmse', 'auc').
        metric_params (Dict[str, Any]): Keyword arguments for the metric function.
        higher_is_better (bool): True if a higher metric score is better.
        cat_features (Optional[List[str]], optional): Categorical features for the model.
            Defaults to None.
        min_improvement (float, optional): The minimum absolute improvement in the CV score
            required to add a feature. Defaults to 1e-5.

    Returns:
        List[str]: The final list of selected features.
    """
    selected_features = list(initial_features) # Start with initial features
    remaining_candidates = list(set(candidate_features) - set(selected_features))
    
    if not remaining_candidates:
        print("No candidate features to add beyond the initial set.")
        return selected_features

    # Calculate initial score
    if selected_features:
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
    else:
        # If starting with no features, initialize best_score to be easily beaten
        best_score = -math.inf if higher_is_better else math.inf
        print("Starting forward selection with an empty feature set.")

    while remaining_candidates:
        scores_this_iteration = {}
        print(f"\n--- Iteration: Evaluating {len(remaining_candidates)} candidates ---")
        
        for candidate in remaining_candidates:
            current_features_to_try = selected_features + [candidate]
            
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
                scores_this_iteration[candidate] = score
            except Exception as e:
                 print(f"  Skipping candidate '{candidate}' due to error during evaluation: {e}")
                 # Assign a terrible score so it won't be chosen
                 scores_this_iteration[candidate] = -math.inf if higher_is_better else math.inf

        if not scores_this_iteration:
             print("No candidates could be evaluated in this iteration. Stopping.")
             break
             
        # Find the best performing candidate in this iteration
        if higher_is_better:
            best_candidate_this_iter = max(scores_this_iteration, key=scores_this_iteration.get)
            best_score_this_iter = scores_this_iteration[best_candidate_this_iter]
        else:
            best_candidate_this_iter = min(scores_this_iteration, key=scores_this_iteration.get)
            best_score_this_iter = scores_this_iteration[best_candidate_this_iter]
            
        print(f"Best candidate this iteration: '{best_candidate_this_iter}' with score {best_score_this_iter:.6f}")

        # Check for improvement
        improvement = best_score_this_iter - best_score if higher_is_better else best_score - best_score_this_iter
        
        if improvement > min_improvement:
            print(f"Improvement found ({improvement:.6f} > {min_improvement:.6f}). Adding feature: '{best_candidate_this_iter}'")
            selected_features.append(best_candidate_this_iter)
            remaining_candidates.remove(best_candidate_this_iter)
            best_score = best_score_this_iter # Update the best score
        else:
            print(f"No significant improvement ({improvement:.6f} <= {min_improvement:.6f}). Stopping selection.")
            break # Stop if no candidate improved the score sufficiently

    print(f"\nForward selection finished. Selected {len(selected_features)} features:")
    print(selected_features)
    return selected_features


if __name__ == '__main__':
    # --- Example Usage of Forward Feature Selection --- 
    print("\n--- Example: Forward Feature Selection ---")
    
    # 0. Generate dummy data (Polars DataFrame)
    N_samples = 200
    rng = np.random.RandomState(42)
    data = pl.DataFrame({
        'feature1_strong': rng.rand(N_samples),
        'feature2_medium': rng.rand(N_samples) * 0.5,
        'feature3_weak': rng.rand(N_samples) * 0.1, 
        'feature4_irrelevant': rng.rand(N_samples),
        'feature5_another_strong': rng.rand(N_samples) * 2
    })
    data = data.with_columns(
        (2 * pl.col('feature1_strong') + 
         3 * pl.col('feature2_medium') + 
         0.5 * pl.col('feature5_another_strong') + 
         rng.randn(N_samples) * 0.1 # Noise
        ).alias('target')
    )

    target_name = 'target'
    candidate_feature_pool = ['feature1_strong', 'feature2_medium', 'feature3_weak', 'feature4_irrelevant', 'feature5_another_strong']
    initial_feature_set = [] # Start from scratch
    
    # 1. Create CV indices (e.g., 5-fold)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_idx_list = list(kf.split(data))

    # 2. Define model and metric parameters
    lgbm_params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'n_estimators': 50, # Small for example
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
        'callbacks': [] # No early stopping for simplicity
    }
    metric_name_config = 'mae'
    metric_params_config = {}
    higher_is_better_config = False # Lower MAE is better

    try:
        final_selected_features = select_features_forward(
            train_df=data,
            target_col=target_name,
            initial_features=initial_feature_set,
            candidate_features=candidate_feature_pool,
            cv_indices=cv_idx_list,
            model_type='lgbm',
            model_params=lgbm_params,
            fit_params=lgbm_fit_params,
            metric_name=metric_name_config,
            metric_params=metric_params_config,
            higher_is_better=higher_is_better_config,
            min_improvement=0.001 # Require at least some small improvement
        )
        print(f"\nFinal features selected by forward selection: {final_selected_features}")
    except ImportError as e:
        print(f"ImportError during example: {e}. Make sure scikit-learn and lightgbm are installed.")
    except Exception as e:
        print(f"An error occurred during the forward selection example: {e}") 