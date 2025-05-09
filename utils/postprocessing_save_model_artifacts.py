#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility function for saving model artifacts like OOF predictions, test predictions,
feature importance plots, and submission files.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import necessary utility functions this module will use
from .eda_plotting_plot_feature_importance import plot_feature_importance
from .postprocessing_generate_submission_file import generate_submission_file

def save_model_artifacts(
    results: Dict[str, Dict[str, Any]],
    features_list: List[str],
    test_ids: Any, # Typically pd.Series or np.ndarray
    target_col: str,
    oof_pred_base_path: str,
    test_pred_base_path: str,
    submission_base_path: str,
    feature_importance_base_path: str,
    best_model_name: str, # Added to save a specific "best_submission.csv"
    logger: Optional[logging.Logger] = None
) -> None:
    """Saves model artifacts for each model in the results dictionary.

    Args:
        results (Dict[str, Dict[str, Any]]): Dictionary containing model results.
            Expected structure: 
            {
                model_name: {
                    'best_model_instance': trained_model_object (optional),
                    'oof_preds': np.ndarray,
                    'test_preds_original': np.ndarray
                }, ...
            }
        features_list (List[str]): List of feature names used in the model.
        test_ids (Any): Series or array of IDs for test predictions.
        target_col (str): Name of the target column for submission file.
        oof_pred_base_path (str): Base path string for saving OOF prediction files.
        test_pred_base_path (str): Base path string for saving test prediction files.
        submission_base_path (str): Base path string for saving submission files.
        feature_importance_base_path (str): Base path string for saving feature importance plots.
        best_model_name (str): Name of the model considered the best, for saving a dedicated submission file.
        logger (Optional[logging.Logger], optional): Logger instance. Defaults to None.
    """
    log_print = logger.info if logger else print

    for model_name_iter_save in results:
        model_data = results[model_name_iter_save]

        # Save Feature Importance Plot
        if model_data.get('best_model_instance') is not None:
            log_print(f"Generating feature importance plot for {model_name_iter_save}...")
            fi_plot_path = feature_importance_base_path + f"_{model_name_iter_save}_feature_importance.png"
            try:
                plot_feature_importance(
                    model=model_data['best_model_instance'],
                    model_name=model_name_iter_save,
                    feature_names=features_list,
                    output_path=fi_plot_path
                )
            except Exception as e:
                log_print_err = logger.error if logger else print
                log_print_err(f"Error generating feature importance for {model_name_iter_save}: {e}")

        # Save OOF Predictions
        if 'oof_preds' in model_data:
            oof_path = oof_pred_base_path + f"_{model_name_iter_save}_oof.npy"
            try:
                np.save(oof_path, model_data['oof_preds'])
                log_print(f"{model_name_iter_save} OOF predictions saved to: {oof_path}")
            except Exception as e:
                log_print_err = logger.error if logger else print
                log_print_err(f"Error saving OOF for {model_name_iter_save}: {e}")

        # Save Test Predictions
        if 'test_preds_original' in model_data:
            test_path = test_pred_base_path + f"_{model_name_iter_save}_test.npy"
            try:
                np.save(test_path, model_data['test_preds_original'])
                log_print(f"{model_name_iter_save} test predictions saved to: {test_path}")
            except Exception as e:
                log_print_err = logger.error if logger else print
                log_print_err(f"Error saving test preds for {model_name_iter_save}: {e}")

            # Save Submission File for this model
            submission_path = submission_base_path + f"_{model_name_iter_save}_submission.csv"
            try:
                generate_submission_file(
                    ids=test_ids,
                    predictions=model_data['test_preds_original'],
                    id_col_name='id', # Assuming 'id' is standard
                    target_col_name=target_col,
                    file_path=str(submission_path)
                )
                log_print(f"{model_name_iter_save} submission file saved to: {submission_path}")
            except Exception as e:
                log_print_err = logger.error if logger else print
                log_print_err(f"Error generating submission for {model_name_iter_save}: {e}")

    # Save submission file for the best model
    if best_model_name in results and 'test_preds_original' in results[best_model_name]:
        best_submission_path_str = submission_base_path + "_best_submission.csv"
        try:
            generate_submission_file(
                ids=test_ids,
                predictions=results[best_model_name]['test_preds_original'],
                id_col_name='id', # Assuming 'id' is standard
                target_col_name=target_col,
                file_path=best_submission_path_str
            )
            log_print(f"Best model ({best_model_name}) submission file saved to: {best_submission_path_str}")
        except Exception as e:
            log_print_err = logger.error if logger else print
            log_print_err(f"Error generating best submission for {best_model_name}: {e}")
    else:
        log_print(f"Warning: Best model '{best_model_name}' or its test predictions not found in results. Cannot save best_submission.csv.") 