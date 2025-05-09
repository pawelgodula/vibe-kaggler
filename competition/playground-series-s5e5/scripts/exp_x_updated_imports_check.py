# Script for Experiment X: Updated Imports Check (based on Exp V)
# MODIFIED for Playground Series S5E5: Predict Calorie Expenditure

"""
Extended Description:
This script is a copy of Experiment V (exp_v_multi_model_direct_log_target.py)
with updated import paths for utility functions after a major refactoring
of the utils directory. The purpose is to verify that the core logic and
CV score remain consistent after these import changes.

It has been further refactored to move common functions (add_feature_cross_terms,
plot_feature_importance, experiment tracking, artifact saving) into the utils directory.

It includes:
- Loading train and test data
- Basic preprocessing (label encoding for categorical features)
- Feature engineering through cross-terms between numerical features (using new utility)
- Training LightGBM, XGBoost, and CatBoost models using 5-fold CV
- Comparing model performance
- Generating feature importance visualizations (via new utility)
- Creating submission file with the best model's predictions (via new utility)
- Updating the experiments tracker (using new utility)

Corresponds to experiment 'exp_x_updated_imports_check.py' in experiments.csv.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import time
import lightgbm as lgb
from lightgbm import LGBMRegressor
import xgboost as xgb
from xgboost import XGBRegressor
import catboost as cb
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
# Matplotlib and Seaborn are now primarily used within the utility functions
# import matplotlib.pyplot as plt 
# import seaborn as sns

# --- Add project root to sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    # Core utilities
    from utils.other_setup_logging import setup_logging
    from utils.other_seed_everything import seed_everything
    # Refactored utilities
    from utils.feature_engineering_add_cross_terms import add_feature_cross_terms
    from utils.postprocessing_save_model_artifacts import save_model_artifacts
    from utils.other_update_experiment_tracker import update_experiment_tracker
    # generate_submission_file is now called by save_model_artifacts
    # plot_feature_importance is now called by save_model_artifacts
except ImportError as e:
    print(f"Error importing utility functions: {e}")
    print("Ensure utils directory is at the project root and contains all required files.")
    print("Consider running this script from the project root or as a module.")
    sys.exit(1)

# --- Configuration ---
EXPERIMENT_NAME = "exp_x_updated_imports_check.py" 
N_FOLDS = 5
SEED = 42 
TARGET_COL = "Calories"
NUMERICAL_FEATURES = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
CATEGORICAL_FEATURES = ['Sex']

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
COMPETITION_DIR = SCRIPT_DIR.parent
RAW_DATA_DIR = COMPETITION_DIR / "data" / "raw"
MODEL_OUTPUT_DIR = COMPETITION_DIR / "models"
OOF_PREDS_DIR = COMPETITION_DIR / "predictions" / "oof"
TEST_PREDS_DIR = COMPETITION_DIR / "predictions" / "test"
REPORTS_DIR = COMPETITION_DIR / "reports"
EXPERIMENTS_CSV = COMPETITION_DIR / "experiments.csv"

# Input Data Paths
TRAIN_CSV = RAW_DATA_DIR / "train.csv"
TEST_CSV = RAW_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = RAW_DATA_DIR / "sample_submission.csv"

# Output Paths base - these will be used by save_model_artifacts utility
OOF_PRED_BASE_PATH = str(OOF_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}")
TEST_PRED_BASE_PATH = str(TEST_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}")
SUBMISSION_BASE_PATH = str(TEST_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}")
FEATURE_IMPORTANCE_BASE_PATH = str(REPORTS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}")

# Create output directories (utilities might also do this, but good practice here too)
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OOF_PREDS_DIR.mkdir(parents=True, exist_ok=True)
TEST_PREDS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Model Parameters ---
LGBM_PARAMS = {
    'n_estimators': 2000,
    'learning_rate': 0.02,
    'max_depth': 10,
    'colsample_bytree': 0.7,
    'subsample': 0.9,
    'random_state': SEED,
    'verbose': -1
}
XGB_PARAMS = {
    'max_depth': 10,
    'colsample_bytree': 0.7,
    'subsample': 0.9,
    'n_estimators': 2000,
    'learning_rate': 0.02,
    'gamma': 0.01,
    'max_delta_step': 2,
    'early_stopping_rounds': 100,
    'eval_metric': 'rmse',
    'enable_categorical': True, 
    'random_state': SEED
}
CATBOOST_PARAMS = {
    'verbose': 100,
    'random_seed': SEED,
    'cat_features': ['Sex'], 
    'early_stopping_rounds': 100
}

# --- Main Function ---
def main():
    logger = setup_logging(log_file=str(COMPETITION_DIR / "training.log"))
    seed_everything(SEED)
    logger.info(f"--- Starting Experiment: {EXPERIMENT_NAME} ---")

    try:
        # --- Load Data ---
        logger.info(f"Loading training data from: {TRAIN_CSV}")
        train_df = pd.read_csv(TRAIN_CSV)
        logger.info(f"Train shape: {train_df.shape}")

        logger.info(f"Loading test data from: {TEST_CSV}")
        test_df = pd.read_csv(TEST_CSV)
        logger.info(f"Test shape: {test_df.shape}")

        test_ids = test_df['id']
        
        # --- Basic Preprocessing ---
        logger.info("Performing basic preprocessing...")
        le = LabelEncoder()
        train_df['Sex'] = le.fit_transform(train_df['Sex'])
        test_df['Sex'] = le.transform(test_df['Sex'])
        train_df['Sex'] = train_df['Sex'].astype('category')
        test_df['Sex'] = test_df['Sex'].astype('category')
        logger.info(f"Label encoded 'Sex' and set as categorical type")
        
        # --- Feature Engineering (using utility) ---
        logger.info("Creating cross-term features...")
        train_df = add_feature_cross_terms(train_df, NUMERICAL_FEATURES)
        test_df = add_feature_cross_terms(test_df, NUMERICAL_FEATURES)
        logger.info(f"Train shape after adding cross-terms: {train_df.shape}")
        logger.info(f"Test shape after adding cross-terms: {test_df.shape}")
        
        # --- Prepare Data for Model Training ---
        logger.info("Preparing data for modeling...")
        X = train_df.drop(columns=['id', TARGET_COL])
        y = np.log1p(train_df[TARGET_COL]) 
        logger.info("Target transformed using np.log1p(Calories)")
        X_test = test_df.drop(columns=['id'])
        logger.info(f"X shape: {X.shape}, y shape: {len(y)}, X_test shape: {X_test.shape}")
        features_list = X.columns.tolist()
        logger.info(f"Total features: {len(features_list)}")
        
        # --- Cross-Validation and Model Training ---
        logger.info(f"Starting {N_FOLDS}-Fold Cross-Validation with multiple models...")
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        model_configs = {
            'LightGBM': LGBMRegressor(**LGBM_PARAMS),
            'XGBoost': XGBRegressor(**XGB_PARAMS),
            'CatBoost': CatBoostRegressor(**CATBOOST_PARAMS)
        }
        results = {
            name: {
                'oof_preds': np.zeros(len(train_df)),
                'test_preds_sum': np.zeros(len(test_df)),
                'rmsle_scores': [],
                'best_model_instance': None, 
                'training_time': 0
            } for name in model_configs
        }
        
        for model_name, model_template in model_configs.items():
            logger.info(f"\n=== Training {model_name} ===")
            start_time_model = time.time()
            for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
                logger.info(f"--- Fold {fold + 1}/{N_FOLDS} for {model_name} ---")
                X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
                X_valid_fold, y_valid_fold = X.iloc[valid_idx], y.iloc[valid_idx]
                fold_start_time = time.time()
                logger.info(f"Training {model_name} fold {fold + 1}...")
                current_model_fold = model_template
                if model_name == 'CatBoost': 
                    current_model_fold = CatBoostRegressor(**CATBOOST_PARAMS)
                if model_name == 'LightGBM':
                    current_model_fold.fit(X_train_fold, y_train_fold, categorical_feature=['Sex'])
                elif model_name == 'XGBoost':
                    current_model_fold.fit(X_train_fold, y_train_fold, eval_set=[(X_valid_fold, y_valid_fold)], verbose=100)
                elif model_name == 'CatBoost':
                    current_model_fold.fit(X_train_fold, y_train_fold, eval_set=(X_valid_fold, y_valid_fold))
                if fold == N_FOLDS - 1: 
                    results[model_name]['best_model_instance'] = current_model_fold
                fold_oof_preds_log = current_model_fold.predict(X_valid_fold)
                fold_test_preds_log = current_model_fold.predict(X_test)
                fold_oof_preds_orig = np.expm1(fold_oof_preds_log)
                results[model_name]['oof_preds'][valid_idx] = fold_oof_preds_orig
                results[model_name]['test_preds_sum'] += fold_test_preds_log
                fold_rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_valid_fold), fold_oof_preds_orig))
                results[model_name]['rmsle_scores'].append(fold_rmsle)
                fold_time = time.time() - fold_start_time
                logger.info(f"Fold {fold + 1} RMSLE: {fold_rmsle:.5f}, Training time: {fold_time:.2f}s")
            results[model_name]['training_time'] = time.time() - start_time_model
            final_test_preds_log = results[model_name]['test_preds_sum'] / N_FOLDS
            results[model_name]['test_preds_original'] = np.expm1(final_test_preds_log)
            results[model_name]['test_preds_original'] = np.clip(results[model_name]['test_preds_original'], 1, 314)
            results[model_name]['oof_preds'] = np.clip(results[model_name]['oof_preds'], 1, 314)
            overall_cv_score = np.sqrt(mean_squared_log_error(train_df[TARGET_COL], results[model_name]['oof_preds']))
            results[model_name]['overall_cv_score'] = overall_cv_score
            logger.info(f"{model_name} - Overall CV Score (RMSLE from OOF preds): {overall_cv_score:.5f}")
            logger.info(f"{model_name} - Mean fold RMSLE: {np.mean(results[model_name]['rmsle_scores']):.5f}, Std: {np.std(results[model_name]['rmsle_scores']):.5f}")
            logger.info(f"{model_name} - Total training time: {results[model_name]['training_time']:.2f}s")
        
        logger.info("\n=== Model Comparison ===")
        for model_name_iter in results:
            logger.info(f"{model_name_iter} - CV Score (from OOF): {results[model_name_iter]['overall_cv_score']:.5f}, Mean Fold RMSLE: {np.mean(results[model_name_iter]['rmsle_scores']):.5f}, Training time: {results[model_name_iter]['training_time']:.2f}s")
        best_model_name = min(results, key=lambda x: np.mean(results[x]['rmsle_scores']))
        logger.info(f"\nBest Model (based on mean fold RMSLE): {best_model_name} with Mean Fold RMSLE: {np.mean(results[best_model_name]['rmsle_scores']):.5f}")
        
        # --- Save Artifacts (using utility) ---
        save_model_artifacts(
            results=results,
            features_list=features_list,
            test_ids=test_ids,
            target_col=TARGET_COL,
            oof_pred_base_path=OOF_PRED_BASE_PATH,
            test_pred_base_path=TEST_PRED_BASE_PATH,
            submission_base_path=SUBMISSION_BASE_PATH,
            feature_importance_base_path=FEATURE_IMPORTANCE_BASE_PATH,
            best_model_name=best_model_name,
            logger=logger
        )
        
        # --- Update Experiment Tracker (using utility) ---
        logger.info(f"Updating experiments tracker: {EXPERIMENTS_CSV}")
        final_cv_for_tracker = np.mean(results[best_model_name]['rmsle_scores'])
        experiment_details = {
            'why': "Verify CV after refactoring: add_cross_terms, plot_fi, save_artifacts, update_tracker to utils.",
            'what': f'Run exp_x logic with major components moved to utility functions.',
            'how': f'exp_x refactored. Core modeling loop remains. Best model: {best_model_name}.',
            'experiment_type': 'multi_model_refactored',
            'base_experiment': 'exp_x_updated_imports_check.py',
            'new_feature': 'Major refactor of helper functions into utils.'
        }
        update_experiment_tracker(
            experiments_csv_path=EXPERIMENTS_CSV,
            experiment_name=EXPERIMENT_NAME, # Still exp_x, but this run is the refactored version
            cv_score=final_cv_for_tracker,
            best_model_name=best_model_name,
            column_updates=experiment_details,
            logger=logger
        )
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info(f"--- Experiment Finished: {EXPERIMENT_NAME} ---")

if __name__ == "__main__":
    main() 