# Script for Experiment F: Average Ensemble of LGBM and XGBoost CV Predictions

"""
Extended Description:
This script implements Experiment F, which creates a simple average ensemble 
of the cross-validated predictions from Experiment C (LightGBM) and 
Experiment E (XGBoost). It loads the OOF and test predictions from these 
experiments, calculates their average, computes the CV score of the averaged 
OOF predictions against the true target, saves the ensembled predictions, 
generates a submission file, and updates the `experiments.csv` tracker.

Corresponds to experiment 'exp_f_ensemble_avg.py' in experiments.csv.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

# --- Add project root to sys.path --- 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.load_csv import load_csv
    from utils.calculate_metric import calculate_metric 
    from utils.generate_submission_file import generate_submission_file 
    from utils.setup_logging import setup_logging
    # from utils.seed_everything import seed_everything # Not strictly necessary for ensembling
except ImportError as e:
     print(f"Error importing utility functions: {e}")
     print("Ensure utils directory is at the project root.")
     print("Consider running this script from the project root or as a module.")
     sys.exit(1)

# --- Configuration ---
EXPERIMENT_NAME = "exp_f_ensemble_avg.py" # For updating experiments.csv
TARGET_COL = "Listening_Time_minutes"

# Input Experiments (Source of predictions)
EXP_C_NAME = "exp_c_train_lgbm_cv"
EXP_E_NAME = "exp_e_train_xgb_cv"

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
COMPETITION_DIR = SCRIPT_DIR.parent
RAW_DATA_DIR = COMPETITION_DIR / "data" / "raw"
OOF_PREDS_DIR = COMPETITION_DIR / "predictions" / "oof"
TEST_PREDS_DIR = COMPETITION_DIR / "predictions" / "test" 
EXPERIMENTS_CSV = COMPETITION_DIR / "experiments.csv"

# Input Prediction Paths
EXP_C_OOF_PATH = OOF_PREDS_DIR / f"{EXP_C_NAME}_oof.npy"
EXP_C_TEST_PATH = TEST_PREDS_DIR / f"{EXP_C_NAME}_test_avg.npy"
EXP_E_OOF_PATH = OOF_PREDS_DIR / f"{EXP_E_NAME}_oof.npy"
EXP_E_TEST_PATH = TEST_PREDS_DIR / f"{EXP_E_NAME}_test_avg.npy"

# Data paths (for target and IDs)
TRAIN_CSV = RAW_DATA_DIR / "train.csv"
TEST_CSV = RAW_DATA_DIR / "test.csv" 

# Output Prediction Paths
ENSEMBLE_OOF_OUTPUT_PATH = OOF_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_oof.npy"
ENSEMBLE_TEST_OUTPUT_PATH = TEST_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_test_avg.npy"
ENSEMBLE_SUBMISSION_OUTPUT_PATH = TEST_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_submission.csv"

# Create output directories if they don't exist (although they should from previous exps)
OOF_PREDS_DIR.mkdir(parents=True, exist_ok=True)
TEST_PREDS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    """Main function to run the averaging ensemble experiment."""
    logger = setup_logging(log_file=str(COMPETITION_DIR / "training.log"))
    logger.info(f"--- Starting Experiment: {EXPERIMENT_NAME} ---")

    # --- Load Input Predictions and Data ---
    try:
        logger.info("Loading input predictions...")
        oof_preds_c = np.load(EXP_C_OOF_PATH)
        test_preds_c = np.load(EXP_C_TEST_PATH)
        logger.info(f"Loaded predictions from {EXP_C_NAME}")
        
        oof_preds_e = np.load(EXP_E_OOF_PATH)
        test_preds_e = np.load(EXP_E_TEST_PATH)
        logger.info(f"Loaded predictions from {EXP_E_NAME}")

        logger.info(f"Loading training data for target: {TRAIN_CSV}")
        df_train_full = load_csv(str(TRAIN_CSV))
        y_true_full = df_train_full[TARGET_COL]
        logger.info(f"Training data shape: {df_train_full.shape}")

        logger.info(f"Loading test data for IDs: {TEST_CSV}")
        df_test = load_csv(str(TEST_CSV))
        test_ids = df_test['id']
        logger.info(f"Test data shape: {df_test.shape}")

        # Basic shape validation
        assert oof_preds_c.shape == oof_preds_e.shape == y_true_full.shape, "OOF prediction shapes or target shape mismatch."
        assert test_preds_c.shape == test_preds_e.shape == test_ids.shape, "Test prediction shapes or test ID shape mismatch."

    except FileNotFoundError as e:
        logger.error(f"Error loading prediction file or data: {e}")
        logger.error("Ensure experiments C and E have been run successfully and their outputs exist.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during loading: {e}")
        sys.exit(1)

    # --- Create Ensemble Predictions ---
    logger.info("Averaging predictions...")
    ensemble_oof_preds = (oof_preds_c + oof_preds_e) / 2.0
    ensemble_test_preds = (test_preds_c + test_preds_e) / 2.0
    logger.info("Ensemble predictions calculated.")

    # --- Calculate Ensemble CV Score ---
    logger.info("Calculating ensemble CV score...")
    try:
        # Handle potential NaNs in input predictions if necessary
        if np.isnan(ensemble_oof_preds).any():
             logger.warning("NaN values found in ensemble OOF predictions. Calculating metric only on non-NaN values.")
             valid_oof_mask = ~np.isnan(ensemble_oof_preds)
             # Ensure y_true_full is filtered correctly (Polars needs boolean literal)
             y_true_filtered = y_true_full.filter(pl.lit(valid_oof_mask))
             ensemble_cv_score = calculate_metric(y_true=y_true_filtered, y_pred=ensemble_oof_preds[valid_oof_mask], metric_name='rmse')
        else:
             ensemble_cv_score = calculate_metric(y_true=y_true_full, y_pred=ensemble_oof_preds, metric_name='rmse')
        logger.info(f"Ensemble Overall CV Score (RMSE): {ensemble_cv_score:.5f}")
    except Exception as e:
        logger.error(f"An error occurred during CV score calculation: {e}")
        # Decide if we should exit or continue to save predictions
        # Let's try to continue and save what we have
        ensemble_cv_score = np.nan # Mark score as invalid

    # --- Save Ensemble Artifacts ---
    try:
        logger.info("Saving ensemble predictions...")
        np.save(ENSEMBLE_OOF_OUTPUT_PATH, ensemble_oof_preds)
        logger.info(f"Ensemble OOF predictions saved to: {ENSEMBLE_OOF_OUTPUT_PATH}")
        
        np.save(ENSEMBLE_TEST_OUTPUT_PATH, ensemble_test_preds)
        logger.info(f"Ensemble test predictions saved to: {ENSEMBLE_TEST_OUTPUT_PATH}")

        logger.info("Generating ensemble submission file...")
        generate_submission_file(ids=test_ids, predictions=ensemble_test_preds, id_col_name='id', target_col_name=TARGET_COL, file_path=str(ENSEMBLE_SUBMISSION_OUTPUT_PATH))
        logger.info(f"Ensemble submission file saved to: {ENSEMBLE_SUBMISSION_OUTPUT_PATH}")
    except Exception as e:
        logger.error(f"An error occurred during artifact saving: {e}")
        sys.exit(1) # Exit if we cannot save results

    # --- Update Experiments Tracker ---
    if not np.isnan(ensemble_cv_score):
        try:
            logger.info(f"Updating experiments tracker: {EXPERIMENTS_CSV}")
            experiments_df = pd.read_csv(EXPERIMENTS_CSV, sep=';')
            exp_mask = experiments_df['script_file_name'] == EXPERIMENT_NAME
            if exp_mask.sum() == 1:
                experiments_df.loc[exp_mask, 'cv_score'] = ensemble_cv_score
                experiments_df.loc[exp_mask, 'status'] = 'done'
                logger.info(f"Updated experiment '{EXPERIMENT_NAME}' with CV score: {ensemble_cv_score:.5f} and status: done")
            else:
                logger.warning(f"Experiment '{EXPERIMENT_NAME}' not found or found multiple times in tracker. Check file.")
            experiments_df.to_csv(EXPERIMENTS_CSV, index=False, sep=';')
            logger.info("Experiments tracker updated successfully.")
        except Exception as e:
            logger.error(f"An error occurred while updating experiments tracker: {e}")
    else:
        logger.warning("Skipping experiments tracker update due to invalid CV score.")


    logger.info(f"--- Experiment Finished: {EXPERIMENT_NAME} ---")

if __name__ == "__main__":
    main() 