# Script for Experiment C: Train LightGBM using all CV folds.

"""
Extended Description:
This script runs the second baseline experiment: training a LightGBM model 
using all pre-computed cross-validation folds. It utilizes the `train_and_cv` 
utility, which handles the fold iteration and calls `train_single_fold` 
internally. After training, it calculates the overall CV score from the 
combined OOF predictions, saves the OOF and averaged test predictions, 
and updates the `experiments.csv` tracker.

Corresponds to experiment 'exp_c_train_lgbm_cv.py' in experiments.csv.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb 
import joblib # For saving models list

# --- Add project root to sys.path --- 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.load_csv import load_csv
    from utils.train_and_cv import train_and_cv # Use the CV trainer
    from utils.calculate_metric import calculate_metric # Need this for overall score
    from utils.generate_submission_file import generate_submission_file 
    from utils.setup_logging import setup_logging
    from utils.seed_everything import seed_everything
except ImportError as e:
     print(f"Error importing utility functions: {e}")
     print("Ensure utils directory is at the project root.")
     print("Consider running this script from the project root or as a module.")
     sys.exit(1)

# --- Configuration ---
EXPERIMENT_NAME = "exp_c_train_lgbm_cv.py" # For updating experiments.csv
N_FOLDS = 5 # Must match the number of folds saved in cv_indices
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
TEST_PREDS_DIR = COMPETITION_DIR / "predictions" / "test" 
EXPERIMENTS_CSV = COMPETITION_DIR / "experiments.csv"

INPUT_CSV = RAW_DATA_DIR / "train.csv"
TEST_CSV = RAW_DATA_DIR / "test.csv" 
MODEL_LIST_OUTPUT_PATH = MODEL_OUTPUT_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_models.joblib"
OOF_PRED_OUTPUT_PATH = OOF_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_oof.npy"
TEST_PRED_OUTPUT_PATH = TEST_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_test_avg.npy" # Saved averaged preds
SUBMISSION_OUTPUT_PATH = TEST_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_submission.csv"

# Create output directories
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OOF_PREDS_DIR.mkdir(parents=True, exist_ok=True)
TEST_PREDS_DIR.mkdir(parents=True, exist_ok=True)

# Features (Use same as Exp B for consistency)
NUMERIC_FEATURES = [
    'Episode_Length_minutes',
    'Host_Popularity_percentage',
    'Guest_Popularity_percentage',
    'Number_of_Ads'
]
FEATURES = NUMERIC_FEATURES

# Model Parameters (Same as Exp B)
LGBM_PARAMS = {
    'objective': 'regression_l1', 
    'metric': 'rmse', 
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
}

# Fit Parameters (Same as Exp B)
LGBM_FIT_PARAMS = {
    'early_stopping_rounds': 100,
    'verbose': 100, 
}

def main():
    """Main function to run the cross-validated training experiment."""
    logger = setup_logging(log_file=str(COMPETITION_DIR / "training.log"))
    seed_everything(SEED)
    logger.info(f"--- Starting Experiment: {EXPERIMENT_NAME} ---")

    # --- Load Data and Indices ---
    try:
        logger.info(f"Loading training data from: {INPUT_CSV}")
        df_train_full = load_csv(str(INPUT_CSV))
        y_true_full = df_train_full[TARGET_COL] # Get target for final CV score calc
        logger.info(f"Full training data shape: {df_train_full.shape}")
        
        logger.info(f"Loading test data from: {TEST_CSV}")
        df_test = load_csv(str(TEST_CSV))
        test_ids = df_test['id']
        logger.info(f"Test data shape: {df_test.shape}")

        logger.info(f"Loading CV indices from: {CV_INDICES_DIR}")
        cv_indices = []
        for i in range(N_FOLDS):
            train_idx = np.load(CV_INDICES_DIR / f"fold_{i}_train.npy")
            valid_idx = np.load(CV_INDICES_DIR / f"fold_{i}_valid.npy")
            cv_indices.append((train_idx, valid_idx))
        logger.info(f"Loaded {len(cv_indices)} sets of fold indices.")

    except FileNotFoundError as e:
        logger.error(f"Error loading data or indices: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading: {e}")
        sys.exit(1)

    # --- Preprocessing (Train and Test) ---
    logger.info(f"Using features: {FEATURES}")
    logger.info("Preprocessing train and test data (NaN filling)..." )
    # Train
    for col in FEATURES:
        if df_train_full[col].dtype.is_numeric():
             if df_train_full[col].null_count() > 0:
                 median_val = df_train_full[col].median()
                 df_train_full = df_train_full.with_columns(pl.col(col).fill_null(median_val))
                 # Also fill test with train median
                 if col in df_test.columns:
                     df_test = df_test.with_columns(pl.col(col).fill_null(median_val))
                 logger.info(f"Filled NaNs in '{col}' with train median: {median_val}")
        else:
             logger.warning(f"Feature '{col}' is not numeric. Skipping NaN fill.")
    logger.info("Preprocessing complete.")

    # --- Train CV Model --- 
    logger.info("Starting cross-validated model training...")
    try:
        oof_preds, test_preds_avg, models = train_and_cv(
            train_df=df_train_full,
            test_df=df_test,
            target_col=TARGET_COL,
            feature_cols=FEATURES,
            model_type='lgbm',
            cv_indices=cv_indices,
            model_params=LGBM_PARAMS,
            fit_params=LGBM_FIT_PARAMS,
            cat_features=None,
            verbose=True # Show fold progress
        )
        logger.info("Cross-validation training complete.")

        # Calculate overall CV score from OOF predictions
        # Ensure no NaNs remain in OOF (can happen if a fold fails or indices are wrong)
        if np.isnan(oof_preds).any():
             logger.warning("NaN values found in OOF predictions. Calculating score only on non-NaN rows.")
             valid_oof_mask = ~np.isnan(oof_preds)
             overall_cv_score = calculate_metric(
                 y_true=y_true_full.filter(pl.lit(valid_oof_mask)), # Filter Polars Series
                 y_pred=oof_preds[valid_oof_mask],
                 metric_name='rmse'
             )
        else:
            overall_cv_score = calculate_metric(
                y_true=y_true_full, 
                y_pred=oof_preds, 
                metric_name='rmse'
            )
        logger.info(f"Overall CV Score (RMSE): {overall_cv_score:.5f}")

        # --- Save Artifacts ---
        logger.info("Saving OOF predictions, test predictions, and models...")
        np.save(OOF_PRED_OUTPUT_PATH, oof_preds)
        logger.info(f"OOF predictions saved to: {OOF_PRED_OUTPUT_PATH}")
        if test_preds_avg is not None:
            np.save(TEST_PRED_OUTPUT_PATH, test_preds_avg)
            logger.info(f"Averaged test predictions saved to: {TEST_PRED_OUTPUT_PATH}")
            
            # Generate Submission File
            logger.info("Generating submission file...")
            generate_submission_file(
                ids=test_ids,
                predictions=test_preds_avg,
                id_col_name='id',
                target_col_name=TARGET_COL,
                file_path=str(SUBMISSION_OUTPUT_PATH)
            )
            logger.info(f"Submission file saved to: {SUBMISSION_OUTPUT_PATH}")
        else:
             logger.warning("Average test predictions are None. Skipping saving and submission.")
             
        # Save models (optional)
        try:
            joblib.dump(models, MODEL_LIST_OUTPUT_PATH)
            logger.info(f"Models list saved to: {MODEL_LIST_OUTPUT_PATH}")
        except Exception as e_model_save:
            logger.warning(f"Failed to save models list: {e_model_save}")
            
    except Exception as e:
        logger.error(f"An error occurred during CV training or artifact saving: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    # --- Update Experiments Tracker ---
    try:
        logger.info(f"Updating experiments tracker: {EXPERIMENTS_CSV}")
        experiments_df = pd.read_csv(EXPERIMENTS_CSV, sep=';')
        exp_mask = experiments_df['script_file_name'] == EXPERIMENT_NAME
        if exp_mask.sum() == 1:
            experiments_df.loc[exp_mask, 'cv_score'] = overall_cv_score
            experiments_df.loc[exp_mask, 'status'] = 'done'
            logger.info(f"Updated experiment '{EXPERIMENT_NAME}' with CV score: {overall_cv_score:.5f} and status: done")
        else:
            logger.warning(f"Experiment '{EXPERIMENT_NAME}' not found or found multiple times in tracker. Check file.")

        experiments_df.to_csv(EXPERIMENTS_CSV, index=False, sep=';')
        logger.info("Experiments tracker updated successfully.")

    except Exception as e:
        logger.error(f"An error occurred while updating experiments tracker: {e}")

    logger.info(f"--- Experiment Finished: {EXPERIMENT_NAME} ---")

if __name__ == "__main__":
    main() 