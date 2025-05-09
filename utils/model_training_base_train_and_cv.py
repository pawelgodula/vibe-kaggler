# Orchestrates model training across multiple cross-validation folds.

"""
Extended Description:
This function manages the cross-validation training loop. It takes the full
training and optional test DataFrames, cross-validation indices, model configuration,
and iterates through each fold. For each fold, it calls `train_single_fold`
to train a model. It collects out-of-fold (OOF) predictions for the training
data and averages the predictions made on the test data across all folds.
"""

import polars as pl
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union

# from .train_single_fold import train_single_fold # TODO: This needs to be updated after renaming
from .model_training_base_train_single_fold import train_single_fold
from ..evaluation.calculate_metric import calculate_metric
from ..utils.logging_utils import create_logger

def train_and_cv(
    train_df: pl.DataFrame,
    test_df: Optional[pl.DataFrame],
    target_col: str,
    feature_cols: List[str],
    model_type: str,
    cv_indices: List[Tuple[np.ndarray, np.ndarray]],
    model_params: Dict[str, Any],
    fit_params: Dict[str, Any],
    cat_features: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[Any]]:
    """Trains a model using cross-validation and returns predictions.

    Args:
        train_df (pl.DataFrame): Full training data.
        test_df (Optional[pl.DataFrame]): Full test data.
        target_col (str): Name of the target column.
        feature_cols (List[str]): List of feature column names.
        model_type (str): Type of model to train (e.g., 'lgbm', 'xgb').
        cv_indices (List[Tuple[np.ndarray, np.ndarray]]): List of (train_idx, valid_idx)
            tuples, one for each CV fold.
        model_params (Dict[str, Any]): Parameters for the model constructor.
        fit_params (Dict[str, Any]): Parameters for the model fitting process.
        cat_features (Optional[List[str]], optional): Categorical features list.
        verbose (bool, optional): If True, prints fold progress. Defaults to True.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray], List[Any]]:
            - oof_preds (np.ndarray): Out-of-fold predictions for the training data.
                                      Shape (len(train_df),) or (len(train_df), n_classes).
            - test_preds_avg (Optional[np.ndarray]): Averaged predictions for the test data
                                                 across all folds. Shape (len(test_df),)
                                                 or (len(test_df), n_classes). None if
                                                 test_df is None.
            - models (List[Any]): List of fitted model objects from each fold.

    Raises:
        ValueError: If inputs are invalid (e.g., empty cv_indices).
        Exception: Propagates exceptions from underlying training functions.
    """
    if not cv_indices:
        raise ValueError("cv_indices list cannot be empty.")

    num_folds = len(cv_indices)
    oof_preds = None # Initialize OOF predictions array
    test_preds_list = [] # Store test predictions from each fold
    models = [] # Store models from each fold

    for fold, (train_idx, valid_idx) in enumerate(cv_indices):
        if verbose:
            print(f"===== Fold {fold + 1}/{num_folds} ====")

        # --- Prepare fold data --- 
        try:
            train_fold_df = train_df[train_idx]
            valid_fold_df = train_df[valid_idx]
            # Make sure test_df is passed correctly (can be None)
            current_test_df = test_df
        except IndexError as e:
             print(f"Error slicing DataFrame for fold {fold + 1}: {e}. Check cv_indices.")
             raise
        except Exception as e:
            print(f"Unexpected error preparing data for fold {fold + 1}: {e}")
            raise

        # --- Train on fold --- 
        model, val_preds, test_preds_fold = train_single_fold(
            train_fold_df=train_fold_df,
            valid_fold_df=valid_fold_df,
            test_df=current_test_df,
            target_col=target_col,
            feature_cols=feature_cols,
            model_type=model_type,
            model_params=model_params,
            fit_params=fit_params,
            cat_features=cat_features
        )

        # --- Store results --- 
        models.append(model)

        # Initialize OOF array on the first fold based on validation prediction shape
        if oof_preds is None:
            if val_preds.ndim == 1:
                oof_preds = np.full(len(train_df), np.nan)
            else: # Handle multiclass case
                oof_preds = np.full((len(train_df), val_preds.shape[1]), np.nan)
        
        # Place validation predictions into the correct OOF slots
        try:
            oof_preds[valid_idx] = val_preds
        except IndexError as e:
             print(f"Error assigning validation predictions to OOF array for fold {fold + 1}: {e}. Check shape/indices.")
             print(f"OOF shape: {oof_preds.shape}, valid_idx shape: {valid_idx.shape}, val_preds shape: {val_preds.shape}")
             raise
        except ValueError as e:
             print(f"Error assigning validation predictions (ValueError) for fold {fold + 1}: {e}. Check shape/indices.")
             print(f"OOF shape: {oof_preds.shape}, valid_idx shape: {valid_idx.shape}, val_preds shape: {val_preds.shape}")
             raise

        if test_preds_fold is not None:
            test_preds_list.append(test_preds_fold)

    # --- Aggregate test predictions --- 
    test_preds_avg = None
    if test_preds_list:
        # Stack predictions and average across folds (axis=0)
        try:
             stacked_preds = np.stack(test_preds_list, axis=0)
             test_preds_avg = np.mean(stacked_preds, axis=0)
        except ValueError as e:
             print(f"Error stacking or averaging test predictions: {e}. Check shapes consistency across folds.")
             # Attempt to provide more debug info
             shapes = [p.shape for p in test_preds_list]
             print(f"Shapes of test predictions per fold: {shapes}")
             # Optionally return None or raise, depending on desired strictness
             test_preds_avg = None 
             # raise # Uncomment to make errors fatal

    if verbose:
        print("===== Training Complete ====")

    return oof_preds, test_preds_avg, models 