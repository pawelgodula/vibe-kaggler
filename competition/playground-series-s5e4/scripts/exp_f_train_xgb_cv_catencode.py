"""
Experiment F: Train XGBoost using CV (5 folds) with OneHotEncoded categorical features.

Same as Exp D, but uses XGBoost instead of LightGBM.
"""
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

# Add the project root to the Python path
# Revert to parents logic like exp_d, but corrected to parents[3]
project_root = str(Path(__file__).resolve().parents[3])
# Remove explicit utils_path addition
# utils_path = os.path.join(project_root, 'utils')
# if utils_path not in sys.path:
#      sys.path.insert(0, utils_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Use insert instead of append? Match exp_d

# Imports - remove incorrect tracker imports
from utils.setup_logging import setup_logging
# from utils.data_structures.experiment import Experiment
# from utils.tracker import ExperimentTracker
from utils.calculate_metric import calculate_metric
from utils.seed_everything import seed_everything
from utils.train_and_cv import train_and_cv
from utils.encode_categorical_features import encode_categorical_features
from utils.load_csv import load_csv

# Define necessary constants locally
KAGGLE_COMPETITION_NAME = "playground-series-s5e4"
# KAGGLE_COMPETITION_PATH should now be correct with the right project_root
KAGGLE_COMPETITION_PATH = Path(project_root) / "competition" / KAGGLE_COMPETITION_NAME
SEED = 42

# Configuration
EXPERIMENT_NAME = "exp_f_train_xgb_cv_catencode"
N_FOLDS = 5
TARGET_COL = "Listening_Time_minutes"

# Paths
EXPERIMENT_DIR = KAGGLE_COMPETITION_PATH / "experiments" / EXPERIMENT_NAME
SUBMISSION_DIR = KAGGLE_COMPETITION_PATH / "predictions" / "test"
OOF_PREDICTIONS_DIR = KAGGLE_COMPETITION_PATH / "predictions" / "oof"
# Add paths for raw data and correct CV indices location
RAW_DATA_DIR = KAGGLE_COMPETITION_PATH / "data" / "raw"
PROCESSED_DATA_DIR = KAGGLE_COMPETITION_PATH / "data" / "processed"
CV_INDICES_DIR = PROCESSED_DATA_DIR / "cv_indices" # Correct path

# Create directories if they don't exist
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
OOF_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = EXPERIMENT_DIR / f"{EXPERIMENT_NAME}_log.log"
SUBMISSION_FILE = SUBMISSION_DIR / f"{EXPERIMENT_NAME}_submission.csv"
OOF_PREDICTIONS_FILE = OOF_PREDICTIONS_DIR / f"{EXPERIMENT_NAME}_oof.csv"
EXPERIMENTS_TRACKER_FILE = KAGGLE_COMPETITION_PATH / "experiments.csv"

# Setup logging
setup_logging(log_file=str(LOG_FILE))

# Set seed for reproducibility
seed_everything(SEED)
logging.info(f"Using fixed seed: {SEED}")

# Load RAW data using CSV
# train_path = KAGGLE_COMPETITION_PATH / "data/processed/train_processed.parquet" # Incorrect
# test_path = KAGGLE_COMPETITION_PATH / "data/processed/test_processed.parquet" # Incorrect
train_path = RAW_DATA_DIR / "train.csv"
test_path = RAW_DATA_DIR / "test.csv"
# Load CV indices from .npy files
# cv_indices_path = (
#     KAGGLE_COMPETITION_PATH / f"cv_indices/train_cv_folds_{N_FOLDS}_seed{SEED}.parquet"
# ) # Incorrect

logging.info("Loading data...")
try:
    # Use load_csv
    train_data = load_csv(str(train_path))
    test_data = load_csv(str(test_path))
    # Load CV indices from .npy files
    cv_indices = []
    for i in range(N_FOLDS):
        train_idx_path = CV_INDICES_DIR / f"fold_{i}_train.npy"
        valid_idx_path = CV_INDICES_DIR / f"fold_{i}_valid.npy"
        train_idx = np.load(train_idx_path)
        valid_idx = np.load(valid_idx_path)
        cv_indices.append((train_idx, valid_idx))
    logging.info(f"Train data shape: {train_data.shape}")
    logging.info(f"Test data shape: {test_data.shape}")
    logging.info(f"Loaded {len(cv_indices)} CV folds from .npy files.")
except FileNotFoundError as e:
    logging.error(f"Error loading data or CV indices: {e}")
    raise
except Exception as e:
    logging.error(f"An unexpected error occurred during data loading: {e}")
    raise

# Store test_ids before potential modifications
test_ids = test_data['id'] 
y_true_full = train_data[TARGET_COL] # Store target before potential modifications

# Identify categorical and numerical features
# Exclude 'id' and the target column
all_cols = [
    col for col in train_data.columns if col not in ["id", TARGET_COL]
]
categorical_cols = [
    col for col in all_cols if train_data[col].dtype in [pl.Categorical, pl.Utf8]
]
numerical_cols = [
    col for col in all_cols if train_data[col].dtype in [pl.Float64, pl.Int64]
]

logging.info(f"Identified {len(categorical_cols)} categorical columns.")
logging.info(f"Identified {len(numerical_cols)} numerical columns.")

# --- Preprocessing Steps (like exp_d) ---
logging.info("Starting preprocessing...")
try:
    # Ensure categorical columns are Utf8 for encoder
    for col in categorical_cols:
         if col in train_data.columns:
             train_data = train_data.with_columns(pl.col(col).cast(pl.Utf8))
         if test_data is not None and col in test_data.columns:
             test_data = test_data.with_columns(pl.col(col).cast(pl.Utf8))

    # --- Categorical Feature Encoding ---
    logging.info("Encoding categorical features using One-Hot Encoding...")
    # Pass dataframes directly, encoder handles train/test fit/transform
    encoded_train_data, encoded_test_data, encoder = (
        encode_categorical_features(
            train_df=train_data,
            test_df=test_data,
            features=categorical_cols,
            encoder_type="onehot",
            handle_unknown='ignore',
            drop_original=True
        )
    )
    # Note: encode_categorical_features returns the encoder itself now
    # We don't get back encoded_feature_names directly, need to derive them

    # Derive encoded feature names
    encoded_feature_names = [col for col in encoded_train_data.columns if col not in train_data.columns and col != TARGET_COL]

    logging.info(f"Encoded train data shape after OHE: {encoded_train_data.shape}")
    logging.info(f"Encoded test data shape after OHE: {encoded_test_data.shape}")
    logging.info(f"Number of encoded categorical features: {len(encoded_feature_names)}")

    # --- NaN Filling for Numeric Features ---
    logging.info("Filling NaNs in Numeric Features...")
    for col in numerical_cols:
        if encoded_train_data[col].null_count() > 0:
            median_val = encoded_train_data[col].median()
            encoded_train_data = encoded_train_data.with_columns(pl.col(col).fill_null(median_val))
            if col in encoded_test_data.columns:
                encoded_test_data = encoded_test_data.with_columns(pl.col(col).fill_null(median_val))
            logging.info(f"Filled NaNs in numeric '{col}' with train median: {median_val}")

    logging.info("Preprocessing complete.")

except Exception as e:
    logging.error(f"An error occurred during preprocessing: {e}", exc_info=True)
    raise

# Combine numerical and encoded categorical features
features = numerical_cols + encoded_feature_names
logging.info(f"Total number of features after encoding: {len(features)}")

# Prepare data for training
X = encoded_train_data[features].to_pandas()
y = encoded_train_data[TARGET_COL].to_pandas() # Use target from encoded df
X_test = encoded_test_data[features].to_pandas()
# cv_indices already loaded correctly from .npy files

# --- Model Training ---
logging.info("Starting XGBoost training with Cross-Validation...")

# XGBoost parameters - Reusing from Exp E
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0.0,
    "lambda": 1.0,
    "alpha": 0.0,
    "seed": SEED,
    "nthread": -1,  # Use all available threads
}

# XGBoost fit parameters - Reusing from Exp E
XGB_FIT_PARAMS = {
    "early_stopping_rounds": 100,
    "num_boost_round": 1000,
    # verbose_eval=100, # Controlled by trainer's log_period
}


try:
    oof_preds, test_preds, models, scores = train_and_cv(
        X=X,
        y=y,
        X_test=X_test,
        cv_indices=cv_indices,
        model_type="xgb",
        model_params=XGB_PARAMS,
        fit_params=XGB_FIT_PARAMS,
        target_col=TARGET_COL,
        log_period=100,
    )
    logging.info("Training complete.")
    logging.info(f"Fold scores (RMSE): {scores}")
    mean_oof_score = np.mean(scores)
    logging.info(f"Mean OOF RMSE: {mean_oof_score:.4f}")
except Exception as e:
    logging.error(f"An error occurred during training: {e}", exc_info=True)
    raise

# --- Post-Training ---
# Save OOF predictions
logging.info("Saving OOF predictions...")
oof_df = pd.DataFrame({TARGET_COL: oof_preds}, index=train_data["id"].to_pandas()) # Use original train_data for ID
oof_df.to_csv(OOF_PREDICTIONS_FILE)
logging.info(f"OOF predictions saved to {OOF_PREDICTIONS_FILE}")

# Verify OOF score (optional but recommended)
calculated_oof_score = calculate_metric(
    y_true=y, y_pred=oof_preds, metric_name="rmse"
)
logging.info(f"Calculated OOF RMSE from saved predictions: {calculated_oof_score:.4f}")
if not np.isclose(mean_oof_score, calculated_oof_score):
     logging.warning(
         "Mean OOF score from folds differs slightly from score calculated "
         f"on combined OOF predictions ({mean_oof_score:.4f} vs {calculated_oof_score:.4f}). "
         "This can happen due to floating point differences or if folds have different sizes."
     )


# Save submission file
logging.info("Saving submission file...")
submission_df = pd.DataFrame(
    {TARGET_COL: test_preds}, index=test_ids # Use stored test_ids
)
submission_df.to_csv(SUBMISSION_FILE)
logging.info(f"Submission file saved successfully to: {SUBMISSION_FILE}")

# --- Experiment Tracking ---
logging.info("Updating experiment tracker...")
# Manual CSV update logic (similar to exp_c/exp_e)
try:
    experiments_df = pd.read_csv(EXPERIMENTS_TRACKER_FILE, sep=';')
    exp_mask = experiments_df['script_file_name'] == EXPERIMENT_NAME
    if exp_mask.sum() == 1:
        experiments_df.loc[exp_mask, 'cv_score'] = calculated_oof_score
        experiments_df.loc[exp_mask, 'status'] = 'completed'
        logging.info(f"Updated experiment '{EXPERIMENT_NAME}' with CV score: {calculated_oof_score:.4f} and status: completed")
    elif exp_mask.sum() == 0:
        logging.warning(f"Experiment '{EXPERIMENT_NAME}' not found in {EXPERIMENTS_TRACKER_FILE}. Adding new row.")
        new_exp = pd.DataFrame([{
            "script_file_name": EXPERIMENT_NAME,
            "description": "Train XGBoost using CV (5 folds) with OneHotEncoded categorical features",
            "model_type": "xgb",
            "features": "numeric+categorical",
            "categorical_encoding": "ohe",
            "cv_folds": N_FOLDS,
            "cv_seed": SEED,
            "status": "completed",
            "oof_score": calculated_oof_score,
            "submission_score": None,
            # Add placeholder columns if needed based on CSV structure
            "why": "Assess XGBoost performance with categorical features.", 
            "what": "Train XGBoost using CV folds with basic categorical encoding (OHE).",
            "how": "Modify Exp D script to use XGBoost instead of LightGBM."
        }])
        # Ensure columns match exactly, handle potential missing columns if needed
        experiments_df = pd.concat([experiments_df, new_exp], ignore_index=True)

    else:
         logger.warning(f"Multiple entries for '{EXPERIMENT_NAME}' found. Updating first.")
         # Get the index of the first match
         first_match_idx = experiments_df.index[exp_mask].tolist()[0]
         experiments_df.loc[first_match_idx, 'cv_score'] = calculated_oof_score
         experiments_df.loc[first_match_idx, 'status'] = 'completed'

    # Define column order explicitly based on previous experiments.csv structure
    column_order = ["script_file_name", "why", "what", "how", "cv_score", "lb_score", "status"]
    # Reorder and select only necessary columns if new ones were added implicitly by DataFrame creation
    experiments_df = experiments_df[column_order]

    experiments_df.to_csv(EXPERIMENTS_TRACKER_FILE, index=False, sep=';')
    logging.info("Experiment tracker updated.")
except FileNotFoundError:
    logging.error(f"Experiments tracker file not found: {EXPERIMENTS_TRACKER_FILE}")
except Exception as e:
    logging.error(f"Error updating experiment tracker: {e}", exc_info=True) 