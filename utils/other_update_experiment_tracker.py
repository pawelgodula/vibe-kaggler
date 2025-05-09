#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility function for updating the experiment tracker CSV file.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional

def update_experiment_tracker(
    experiments_csv_path: Path,
    experiment_name: str,
    cv_score: float,
    best_model_name: str,
    column_updates: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> None:
    """Updates or adds an entry to the experiments.csv tracker.

    Args:
        experiments_csv_path (Path): Path to the experiments.csv file.
        experiment_name (str): Name of the current experiment script.
        cv_score (float): The calculated CV score for the experiment.
        best_model_name (str): Name of the best performing model in this experiment.
        column_updates (Dict[str, Any]): A dictionary containing additional columns and their values
                                         to update/set for the experiment entry. Expected keys include:
                                         'why', 'what', 'how', 'experiment_type', 
                                         'base_experiment', 'new_feature'.
        logger (Optional[logging.Logger], optional): Logger instance. Defaults to None.
    """
    log_print = logger.info if logger else print

    try:
        if not experiments_csv_path.exists():
            log_print(f"Experiments file not found at {experiments_csv_path}. Creating it.")
            headers = ['script_file_name', 'why', 'what', 'how', 'cv_score', 'lb_score', 'status', 'experiment_type', 'base_experiment', 'new_feature']
            pd.DataFrame(columns=headers).to_csv(experiments_csv_path, index=False, sep=';')
            
        experiments_df = pd.read_csv(experiments_csv_path, sep=';')
        exp_mask = experiments_df['script_file_name'] == experiment_name

        default_values = {
            'lb_score': None,
            'status': 'done'
        }

        if exp_mask.sum() > 0:
            idx_to_update = experiments_df[exp_mask].index[0]
            experiments_df.loc[idx_to_update, 'cv_score'] = cv_score
            experiments_df.loc[idx_to_update, 'status'] = default_values['status']
            for col, value in column_updates.items():
                if col in experiments_df.columns:
                    experiments_df.loc[idx_to_update, col] = value
                else:
                    log_print(f"Warning: Column '{col}' not found in experiments.csv. Skipping update for this column.")
            # Update 'how' to include best_model_name if not already structured that way
            if 'how' in column_updates and best_model_name not in str(column_updates['how']):
                 experiments_df.loc[idx_to_update, 'how'] = f"{column_updates['how']} Best model: {best_model_name}."
            elif 'how' not in column_updates:
                 experiments_df.loc[idx_to_update, 'how'] = f"Best model: {best_model_name}."

            log_print(f"Updated existing experiment '{experiment_name}' with CV Score: {cv_score:.5f}")
        else:
            new_row_data = {
                'script_file_name': experiment_name,
                'cv_score': cv_score,
                **default_values,
                **column_updates
            }
            if 'how' in new_row_data and best_model_name not in str(new_row_data['how']):
                 new_row_data['how'] = f"{new_row_data['how']} Best model: {best_model_name}."
            elif 'how' not in new_row_data:
                 new_row_data['how'] = f"Best model: {best_model_name}."

            # Ensure all columns from experiments_df are present in new_row_data, adding None if missing
            for col_header in experiments_df.columns:
                if col_header not in new_row_data:
                    new_row_data[col_header] = None
            
            new_row_df = pd.DataFrame([new_row_data])
            # Ensure column order matches the existing DataFrame
            new_row_df = new_row_df[experiments_df.columns.tolist()]

            experiments_df = pd.concat([experiments_df, new_row_df], ignore_index=True)
            log_print(f"Added new experiment '{experiment_name}' to tracker with CV Score: {cv_score:.5f}")
            
        experiments_df.to_csv(experiments_csv_path, index=False, sep=';')
        log_print("Experiments tracker updated successfully.")

    except Exception as e:
        log_print_err = logger.error if logger else print
        log_print_err(f"Error updating experiment tracker: {e}", exc_info=True if logger else False)
        # Optionally re-raise if this should halt execution
        # raise 