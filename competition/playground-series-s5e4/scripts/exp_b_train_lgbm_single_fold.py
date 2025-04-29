# Script for Experiment B: Train LightGBM on a single fold.

"""
Extended Description:
This script runs the first baseline experiment: training a LightGBM model 
on the first cross-validation fold (fold 0). It uses pre-computed CV indices, 
loads the training data, calls the `train_single_fold` utility, calculates 
the validation score (RMSE assumed), and updates the `experiments.csv` tracker.

Corresponds to experiment 'exp_b_train_lgbm_single_fold.py' in experiments.csv.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb # Import necessary model library
import polars as pl

# --- Add project root to sys.path --- 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.load_csv import load_csv
    from utils.train_single_fold import train_single_fold
    from utils.calculate_metric import calculate_metric # Import metric calculation
    from utils.generate_submission_file import generate_submission_file # Import submission utility
    from utils.setup_logging import setup_logging
    from utils.seed_everything import seed_everything
except ImportError as e:
     print(f"Error importing utility functions: {e}")
     print("Ensure utils directory is at the project root.")
     print("Consider running this script from the project root or as a module.")
     sys.exit(1)

# --- Configuration ---
EXPERIMENT_NAME = "exp_b_train_lgbm_single_fold.py" # For updating experiments.csv
FOLD_ID = 0
SEED = 42
TARGET_COL = "Listening_Time_minutes"

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
COMPETITION_DIR = SCRIPT_DIR.parent
RAW_DATA_DIR = COMPETITION_DIR / "data" / "raw"
PROCESSED_DATA_DIR = COMPETITION_DIR / "data" / "processed"
CV_INDICES_DIR = PROCESSED_DATA_DIR / "cv_indices"
MODEL_OUTPUT_DIR = COMPETITION_DIR / "models"
OOF_PREDS_DIR = COMPETITION_DIR / "predictions" / "oof"
TEST_PREDS_DIR = COMPETITION_DIR / "predictions" / "test" # Add directory for test preds/submission
EXPERIMENTS_CSV = COMPETITION_DIR / "experiments.csv"

INPUT_CSV = RAW_DATA_DIR / "train.csv"
TEST_CSV = RAW_DATA_DIR / "test.csv" # Path to test data
TRAIN_IDX_PATH = CV_INDICES_DIR / f"fold_{FOLD_ID}_train.npy"
VALID_IDX_PATH = CV_INDICES_DIR / f"fold_{FOLD_ID}_valid.npy"
MODEL_OUTPUT_PATH = MODEL_OUTPUT_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_fold{FOLD_ID}.lgb"
OOF_PRED_OUTPUT_PATH = OOF_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_fold{FOLD_ID}.npy"
SUBMISSION_OUTPUT_PATH = TEST_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_submission.csv" # Path for submission file

# Create output directories if they don't exist
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OOF_PREDS_DIR.mkdir(parents=True, exist_ok=True)
TEST_PREDS_DIR.mkdir(parents=True, exist_ok=True) # Create test preds dir

# Features (Start with numeric only for baseline)
# Manually list numeric features based on EDA/prior knowledge
# TODO: Refine this feature list based on EDA
NUMERIC_FEATURES = [
    'Episode_Length_minutes',
    'Host_Popularity_percentage',
    'Guest_Popularity_percentage',
    'Number_of_Ads'
]
FEATURES = NUMERIC_FEATURES # Use only numeric for this baseline

# Model Parameters (Example - use defaults or define specifics)
LGBM_PARAMS = {
    'objective': 'regression_l1', # MAE objective, common for time-related regression
    'metric': 'rmse', # Report RMSE during training
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,
    'seed': SEED,
    'boosting_type': 'gbdt',
    # Early stopping needs to be in fit_params
}

# Parameters for the fit/train method (e.g., early stopping)
# Check train_single_fold and underlying _train_lgbm for exact names
LGBM_FIT_PARAMS = {
    'early_stopping_rounds': 100,
    'verbose': 100, # Log training progress every 100 rounds
}


def main():
    """Main function to run the single fold training experiment."""
    # --- Basic Setup ---
    print("Setting up logger...") # Add print statement
    logger = setup_logging(log_file=str(COMPETITION_DIR / "training.log"))
    if logger is None:
        print("ERROR: Logger setup returned None!")
        sys.exit(1)
    print("Logger setup apparently successful.") # Add print statement
    
    print("Setting seed...") # Add print statement
    seed_everything(SEED)
    print("Seed set.") # Add print statement

    try:
        logger.info(f"--- Starting Experiment: {EXPERIMENT_NAME} ---")
        logger.info(f"Training on Fold: {FOLD_ID}")
    except Exception as e:
        print(f"ERROR logging initial messages: {e}")
        sys.exit(1)

    print("Initial log messages attempted.") # Add print statement

    # --- Load Data and Indices (Check Only) ---
    try:
        print("Loading training data...") # Add print statement
        logger.info(f"Loading training data from: {INPUT_CSV}")
        df_train_full = load_csv(str(INPUT_CSV))
        logger.info(f"Full training data shape: {df_train_full.shape}")
        logger.info(f"Type of df_train_full after load_csv: {type(df_train_full)}") 
        print("Training data loaded.") # Add print statement
        
        logger.info(f"Loading test data from: {TEST_CSV}")
        df_test = load_csv(str(TEST_CSV))
        logger.info(f"Test data shape: {df_test.shape}")
        test_ids = df_test['id'] # Assuming 'id' column exists for submission

        # Add row index column if it doesn't exist (needed for filtering below)
        if 'row_index' not in df_train_full.columns:
            df_train_full = df_train_full.with_row_index('row_index')
            logger.info("Added 'row_index' column.")

        print("Loading indices...") # Add print statement
        logger.info(f"Loading indices: {TRAIN_IDX_PATH}, {VALID_IDX_PATH}")
        train_idx = np.load(TRAIN_IDX_PATH)
        valid_idx = np.load(VALID_IDX_PATH)
        logger.info(f"Train indices shape: {train_idx.shape}, Valid indices shape: {valid_idx.shape}")
        print("Indices loaded.") # Add print statement

    except FileNotFoundError as e:
        logger.error(f"Error loading data or indices: {e}")
        print(f"ERROR: File not found during load: {e}") # Add print
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading: {e}")
        print(f"ERROR: Unexpected load error: {e}") # Add print
        sys.exit(1)

    print("Data and indices loading successful (apparently).") # Add print statement

    # --- Feature Selection & Preprocessing (Minimal for Baseline) ---
    logger.info(f"Using features: {FEATURES}")
    # Simple NaN handling for baseline - fill numeric NaNs with median
    # TODO: Move sophisticated handling to dedicated preprocessing script/utility
    
    # Add check before loop
    logger.info(f"Type of df_train_full before NaN loop: {type(df_train_full)}")
    print("Starting NaN filling loop...") # Add print
    for col in FEATURES:
        # Ensure column access creates a Polars Series
        col_series = df_train_full[col]
        logger.debug(f"Processing column '{col}', type: {type(col_series)}")
        # Check using Polars dtypes
        if col_series.dtype.is_numeric(): # Check Polars dtype
             if col_series.null_count() > 0:
                 median_val = col_series.median()
                 # Use Polars expression for updating
                 df_train_full = df_train_full.with_columns(pl.col(col).fill_null(median_val))
                 logger.info(f"Filled NaNs in '{col}' with median: {median_val}")
        else:
             # Handle potential non-numeric features if added to FEATURES list later
             logger.warning(f"Feature '{col}' is not numeric. Skipping NaN fill.")
    print("Finished NaN filling loop.") # Add print
             
    # --- Preprocess Test Data ---
    logger.info("Preprocessing test data...")
    for col in FEATURES:
        if col in df_test.columns:
             if df_test[col].dtype.is_numeric():
                 if df_test[col].null_count() > 0:
                     # IMPORTANT: Use median from TRAINING data to fill test NaNs
                     # Re-calculate or store training median if needed. For simplicity here, re-calculating.
                     # A better approach is to save/load scalers/imputers.
                     train_median = df_train_full[col].median() 
                     df_test = df_test.with_columns(pl.col(col).fill_null(train_median))
                     logger.info(f"Filled NaNs in test '{col}' with train median: {train_median}")
             else:
                  logger.warning(f"Test feature '{col}' is not numeric. Skipping NaN fill.")
        else:
             logger.warning(f"Feature '{col}' not found in test data. Skipping preprocessing.")
             
    # --- Train Model --- 
    logger.info("Starting model training...")
    print("Starting model training section...") # Add print
    try:
        # Extract validation target for score calculation later
        y_valid = df_train_full.filter(pl.col('row_index').is_in(valid_idx))[TARGET_COL].to_numpy()
        
        # Call train_single_fold - returns model, val_preds, test_preds
        model, val_preds, test_preds = train_single_fold(
            train_fold_df=df_train_full.lazy().filter(pl.col('row_index').is_in(train_idx)).collect(), 
            valid_fold_df=df_train_full.lazy().filter(pl.col('row_index').is_in(valid_idx)).collect(),
            test_df=df_test, # Pass preprocessed test data
            feature_cols=FEATURES, # Correct keyword from function signature
            target_col=TARGET_COL,
            model_type='lgbm',
            model_params=LGBM_PARAMS,
            fit_params=LGBM_FIT_PARAMS, # Pass fit parameters
            cat_features=None,
            # Optional: Pass fold_id, seed, logger if train_single_fold accepts them
            # fold_id=FOLD_ID, 
            # seed=SEED,
            # logger=logger 
        )
        logger.info(f"Training complete for fold {FOLD_ID}.")
        
        # Calculate CV score (RMSE assumed)
        score = calculate_metric(y_true=y_valid, y_pred=val_preds, metric_name='rmse') # Correct arg name
        logger.info(f"Fold {FOLD_ID} CV Score (RMSE): {score:.5f}")
        print(f"Training apparently complete. Score: {score:.5f}") # Add print

        # --- Generate Submission File ---
        if test_preds is not None:
            logger.info("Generating submission file...")
            print("Generating submission file...") # Add print
            try:
                generate_submission_file(
                    ids=test_ids, 
                    predictions=test_preds, 
                    id_col_name='id', 
                    target_col_name=TARGET_COL, 
                    file_path=str(SUBMISSION_OUTPUT_PATH)
                )
                logger.info(f"Submission file saved to: {SUBMISSION_OUTPUT_PATH}")
                print(f"Submission file saved: {SUBMISSION_OUTPUT_PATH}")
            except Exception as e_sub:
                logger.error(f"Failed to generate submission file: {e_sub}")
                print(f"ERROR generating submission: {e_sub}")
        else:
             logger.warning("Test predictions are None, skipping submission file generation.")
             print("WARNING: No test predictions generated.")

    except Exception as e:
        logger.error(f"An error occurred during model training or submission generation: {e}")
        print(f"ERROR during training: {e}") # Add print
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    # --- Update Experiments Tracker ---
    print("Starting tracker update section...") # Add print
    try:
        logger.info(f"Updating experiments tracker: {EXPERIMENTS_CSV}")
        experiments_df = pd.read_csv(EXPERIMENTS_CSV, sep=';')
        
        # Find the row for the current experiment
        exp_mask = experiments_df['script_file_name'] == EXPERIMENT_NAME
        if exp_mask.sum() == 0:
            logger.warning(f"Experiment '{EXPERIMENT_NAME}' not found in {EXPERIMENTS_CSV}. Cannot update score.")
            print(f"WARNING: Experiment {EXPERIMENT_NAME} not found in tracker.") # Add print
        elif exp_mask.sum() > 1:
            logger.warning(f"Multiple entries for experiment '{EXPERIMENT_NAME}' found in {EXPERIMENTS_CSV}. Updating the first one.")
            print(f"WARNING: Multiple entries for {EXPERIMENT_NAME} found.") # Add print
            experiments_df.loc[exp_mask.idxmax(), 'cv_score'] = score
            experiments_df.loc[exp_mask.idxmax(), 'status'] = 'done'
        else:
            experiments_df.loc[exp_mask, 'cv_score'] = score
            experiments_df.loc[exp_mask, 'status'] = 'done'
            logger.info(f"Updated experiment '{EXPERIMENT_NAME}' with CV score: {score:.5f} and status: done")
            print(f"Updated tracker for {EXPERIMENT_NAME}.") # Add print

        experiments_df.to_csv(EXPERIMENTS_CSV, index=False, sep=';')
        logger.info("Experiments tracker updated successfully.")
        print("Tracker saved.") # Add print

    except FileNotFoundError:
        logger.error(f"Experiments tracker file not found: {EXPERIMENTS_CSV}")
        print(f"ERROR: Tracker file not found: {EXPERIMENTS_CSV}") # Add print
    except Exception as e:
        logger.error(f"An error occurred while updating experiments tracker: {e}")
        print(f"ERROR updating tracker: {e}") # Add print

    logger.info(f"--- Experiment Finished: {EXPERIMENT_NAME} ---")
    # print("--- Reached end of simplified main function ---") # Remove print from simplified version


if __name__ == "__main__":
    # Minimal check, load within main for clarity now
    main() 