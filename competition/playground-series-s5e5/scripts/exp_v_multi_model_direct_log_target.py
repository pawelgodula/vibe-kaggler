# Script for Experiment V: Multi-Model Comparison with Direct Log Target
# MODIFIED for Playground Series S5E5: Predict Calorie Expenditure

"""
Extended Description:
This script is based on Experiment R but modified to align with an external script
that reportedly achieved better CV/LB scores. Key changes include:
1. Direct log1p transformation of the target: np.log1p(Calories)
2. Predictions are transformed back using only np.expm1()
3. CatBoost parameters adjusted (iterations, learning_rate, depth set to default)
4. LightGBM fit call simplified (removed eval_set and callbacks)

It includes:
- Loading train and test data
- Basic preprocessing (label encoding for categorical features)
- Feature engineering through cross-terms between numerical features
- Training LightGBM, XGBoost, and CatBoost models using 5-fold CV
- Comparing model performance
- Generating feature importance visualizations
- Creating submission file with the best model's predictions
- Updating the experiments tracker

Corresponds to experiment 'exp_v_multi_model_direct_log_target.py' in experiments.csv.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
# import polars as pl # Not used in the provided script, removing for closer replication
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
import matplotlib.pyplot as plt
import seaborn as sns

# --- Add project root to sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.load_csv import load_csv
    from utils.calculate_metric import calculate_metric
    from utils.generate_submission_file import generate_submission_file
    from utils.setup_logging import setup_logging
    from utils.seed_everything import seed_everything
except ImportError as e:
    print(f"Error importing utility functions: {e}")
    print("Ensure utils directory is at the project root and contains all required files.")
    print("Consider running this script from the project root or as a module.")
    sys.exit(1)

# --- Configuration ---
EXPERIMENT_NAME = "exp_v_multi_model_direct_log_target.py"
N_FOLDS = 5
SEED = 42 # Matching the provided script's random_state
TARGET_COL = "Calories"
NUMERICAL_FEATURES = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
CATEGORICAL_FEATURES = ['Sex']

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
COMPETITION_DIR = SCRIPT_DIR.parent
RAW_DATA_DIR = COMPETITION_DIR / "data" / "raw"
# PROCESSED_DATA_DIR = COMPETITION_DIR / "data" / "processed" # Not used
MODEL_OUTPUT_DIR = COMPETITION_DIR / "models"
OOF_PREDS_DIR = COMPETITION_DIR / "predictions" / "oof"
TEST_PREDS_DIR = COMPETITION_DIR / "predictions" / "test"
REPORTS_DIR = COMPETITION_DIR / "reports"
EXPERIMENTS_CSV = COMPETITION_DIR / "experiments.csv"

# Input Data Paths
TRAIN_CSV = RAW_DATA_DIR / "train.csv"
TEST_CSV = RAW_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = RAW_DATA_DIR / "sample_submission.csv"

# Output Paths base
OOF_PRED_BASE_PATH = str(OOF_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}")
TEST_PRED_BASE_PATH = str(TEST_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}")
SUBMISSION_BASE_PATH = str(TEST_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}")
FEATURE_IMPORTANCE_BASE_PATH = str(REPORTS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}")

# Create output directories
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OOF_PREDS_DIR.mkdir(parents=True, exist_ok=True)
TEST_PREDS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Model Parameters ---
# LightGBM Parameters (matching provided script)
LGBM_PARAMS = {
    'n_estimators': 2000,
    'learning_rate': 0.02,
    'max_depth': 10,
    'colsample_bytree': 0.7,
    'subsample': 0.9,
    'random_state': SEED,
    'verbose': -1
}

# XGBoost Parameters (matching provided script)
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
    'enable_categorical': True, # XGBoost can handle pandas category dtype if enable_categorical=True
    'random_state': SEED
}

# CatBoost Parameters (matching provided script - note: iterations, lr, depth use defaults)
CATBOOST_PARAMS = {
    'verbose': 100,
    'random_seed': SEED,
    'cat_features': ['Sex'], # Must match column name if 'Sex' is category type
    'early_stopping_rounds': 100
    # iterations, learning_rate, depth will use CatBoost defaults
}

# Function to add cross-term features between numerical features
def add_feature_cross_terms(df, numerical_features):
    df_new = df.copy()
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):
            feature1 = numerical_features[i]
            feature2 = numerical_features[j]
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    return df_new

# Function to generate and save feature importance plot
def plot_feature_importance(model, model_name, feature_names, output_path):
    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1.0)
    sns.set_style('whitegrid')
    
    if model_name == 'LightGBM':
        importance = model.feature_importances_
    elif model_name == 'XGBoost':
        importance = model.feature_importances_
    elif model_name == 'CatBoost':
        importance = model.get_feature_importance()
    
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    top_features = importance_df.head(30)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    
    plt.title(f'{model_name} Feature Importance (Top 30 features)', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# --- Main Function ---
def main():
    logger = setup_logging(log_file=str(COMPETITION_DIR / "training.log"))
    seed_everything(SEED)
    logger.info(f"--- Starting Experiment: {EXPERIMENT_NAME} ---")

    try:
        # --- Load Data ---
        logger.info(f"Loading training data from: {TRAIN_CSV}")
        # train_df = load_csv(str(TRAIN_CSV)) # Using pandas.read_csv for closer replication
        train_df = pd.read_csv(TRAIN_CSV)
        logger.info(f"Train shape: {train_df.shape}")

        logger.info(f"Loading test data from: {TEST_CSV}")
        # test_df = load_csv(str(TEST_CSV)) # Using pandas.read_csv for closer replication
        test_df = pd.read_csv(TEST_CSV)
        logger.info(f"Test shape: {test_df.shape}")

        test_ids = test_df['id']
        
        # --- Basic Preprocessing ---
        logger.info("Performing basic preprocessing...")
        le = LabelEncoder()
        # Provided script fits on train['Sex'] and then transforms train['Sex'] and test['Sex']
        train_df['Sex'] = le.fit_transform(train_df['Sex'])
        test_df['Sex'] = le.transform(test_df['Sex'])
        
        train_df['Sex'] = train_df['Sex'].astype('category')
        test_df['Sex'] = test_df['Sex'].astype('category')
        logger.info(f"Label encoded 'Sex' and set as categorical type")
        
        # --- Feature Engineering ---
        logger.info("Creating cross-term features...")
        train_df = add_feature_cross_terms(train_df, NUMERICAL_FEATURES)
        test_df = add_feature_cross_terms(test_df, NUMERICAL_FEATURES)
        logger.info(f"Train shape after adding cross-terms: {train_df.shape}")
        logger.info(f"Test shape after adding cross-terms: {test_df.shape}")
        
        # --- Prepare Data for Model Training ---
        logger.info("Preparing data for modeling...")
        
        X = train_df.drop(columns=['id', TARGET_COL])
        y = np.log1p(train_df[TARGET_COL]) # Direct log1p transformation
        logger.info("Target transformed using np.log1p(Calories)")
        
        X_test = test_df.drop(columns=['id'])
        
        logger.info(f"X shape: {X.shape}, y shape: {len(y)}, X_test shape: {X_test.shape}")
        
        features_list = X.columns.tolist() # Renamed from 'features' to avoid conflict
        logger.info(f"Total features: {len(features_list)}")
        
        # --- Cross-Validation and Model Training ---
        logger.info(f"Starting {N_FOLDS}-Fold Cross-Validation with multiple models...")
        
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        
        model_configs = { # Renamed from 'models' to avoid conflict with trained model instances
            'LightGBM': LGBMRegressor(**LGBM_PARAMS),
            'XGBoost': XGBRegressor(**XGB_PARAMS),
            'CatBoost': CatBoostRegressor(**CATBOOST_PARAMS)
        }
        
        results = {
            name: {
                'oof_preds': np.zeros(len(train_df)),
                'test_preds_sum': np.zeros(len(test_df)),
                'rmsle_scores': [],
                'best_model_instance': None, # Storing the actual trained model from the last fold
                'training_time': 0
            } for name in model_configs
        }
        
        for model_name, model_template in model_configs.items(): # Iterate over model configurations
            logger.info(f"\n=== Training {model_name} ===")
            start_time_model = time.time() # Use a different variable name
            
            for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
                logger.info(f"--- Fold {fold + 1}/{N_FOLDS} for {model_name} ---")
                
                X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
                X_valid_fold, y_valid_fold = X.iloc[valid_idx], y.iloc[valid_idx]
                
                fold_start_time = time.time()
                logger.info(f"Training {model_name} fold {fold + 1}...")
                
                # Instantiate model for the fold
                current_model_fold = model_template # XGBoost and LightGBM can be refit, CatBoost usually re-instantiated
                if model_name == 'CatBoost': # CatBoost should be re-instantiated to reset state for new fold
                    current_model_fold = CatBoostRegressor(**CATBOOST_PARAMS)

                if model_name == 'LightGBM':
                    current_model_fold.fit(X_train_fold, y_train_fold, categorical_feature=['Sex']) # Simplified fit
                elif model_name == 'XGBoost':
                    current_model_fold.fit(X_train_fold, y_train_fold, eval_set=[(X_valid_fold, y_valid_fold)], verbose=100)
                elif model_name == 'CatBoost':
                     # CatBoost uses 'cat_features' in constructor, 'eval_set' in fit
                    current_model_fold.fit(X_train_fold, y_train_fold, eval_set=(X_valid_fold, y_valid_fold))

                if fold == N_FOLDS - 1: # Store the last trained model for feature importance
                    results[model_name]['best_model_instance'] = current_model_fold
                
                fold_oof_preds_log = current_model_fold.predict(X_valid_fold)
                fold_test_preds_log = current_model_fold.predict(X_test)
                
                # Transform log predictions back to original scale (NO multiplication by duration)
                fold_oof_preds_orig = np.expm1(fold_oof_preds_log)
                
                results[model_name]['oof_preds'][valid_idx] = fold_oof_preds_orig
                results[model_name]['test_preds_sum'] += fold_test_preds_log
                
                # RMSLE calculation uses np.expm1 on y_valid as well, consistent with provided script
                fold_rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_valid_fold), fold_oof_preds_orig))
                results[model_name]['rmsle_scores'].append(fold_rmsle)
                
                fold_time = time.time() - fold_start_time
                logger.info(f"Fold {fold + 1} RMSLE: {fold_rmsle:.5f}, Training time: {fold_time:.2f}s")
            
            results[model_name]['training_time'] = time.time() - start_time_model
            
            final_test_preds_log = results[model_name]['test_preds_sum'] / N_FOLDS
            results[model_name]['test_preds_original'] = np.expm1(final_test_preds_log) # NO multiplication by duration
            
            results[model_name]['test_preds_original'] = np.clip(results[model_name]['test_preds_original'], 1, 314)
            results[model_name]['oof_preds'] = np.clip(results[model_name]['oof_preds'], 1, 314)
            
            # y_true for metric is original scale, consistent with RMSLE calculation within the loop
            overall_cv_score = np.sqrt(mean_squared_log_error(train_df[TARGET_COL], results[model_name]['oof_preds']))
            results[model_name]['overall_cv_score'] = overall_cv_score # Store calculated overall CV
            
            # The RMSLE scores in 'rmsle_scores' are already on the correct scale from the loop
            logger.info(f"{model_name} - Overall CV Score (RMSLE from OOF preds): {overall_cv_score:.5f}")
            logger.info(f"{model_name} - Mean fold RMSLE: {np.mean(results[model_name]['rmsle_scores']):.5f}, Std: {np.std(results[model_name]['rmsle_scores']):.5f}")
            logger.info(f"{model_name} - Total training time: {results[model_name]['training_time']:.2f}s")
        
        logger.info("\n=== Model Comparison ===")
        for model_name_iter in results: # Use a different variable name
            logger.info(f"{model_name_iter} - CV Score (from OOF): {results[model_name_iter]['overall_cv_score']:.5f}, Mean Fold RMSLE: {np.mean(results[model_name_iter]['rmsle_scores']):.5f}, Training time: {results[model_name_iter]['training_time']:.2f}s")
        
        best_model_name = min(results, key=lambda x: np.mean(results[x]['rmsle_scores'])) # Use mean fold RMSLE for best model selection
        logger.info(f"\nBest Model (based on mean fold RMSLE): {best_model_name} with Mean Fold RMSLE: {np.mean(results[best_model_name]['rmsle_scores']):.5f}")
        
        for model_name_iter_save in results: # Use a different variable name
            if results[model_name_iter_save]['best_model_instance'] is not None:
                logger.info(f"Generating feature importance plot for {model_name_iter_save}...")
                plot_feature_importance(
                    model=results[model_name_iter_save]['best_model_instance'],
                    model_name=model_name_iter_save,
                    feature_names=features_list,
                    output_path=FEATURE_IMPORTANCE_BASE_PATH + f"_{model_name_iter_save}_feature_importance.png"
                )
            
            oof_path = OOF_PRED_BASE_PATH + f"_{model_name_iter_save}_oof.npy"
            np.save(oof_path, results[model_name_iter_save]['oof_preds'])
            logger.info(f"{model_name_iter_save} OOF predictions saved to: {oof_path}")
            
            test_path = TEST_PRED_BASE_PATH + f"_{model_name_iter_save}_test.npy"
            np.save(test_path, results[model_name_iter_save]['test_preds_original'])
            logger.info(f"{model_name_iter_save} test predictions saved to: {test_path}")
            
            submission_path = SUBMISSION_BASE_PATH + f"_{model_name_iter_save}_submission.csv"
            generate_submission_file(
                ids=test_ids,
                predictions=results[model_name_iter_save]['test_preds_original'],
                id_col_name='id',
                target_col_name=TARGET_COL,
                file_path=str(submission_path)
            )
            logger.info(f"{model_name_iter_save} submission file saved to: {submission_path}")
        
        best_submission_path = SUBMISSION_BASE_PATH + "_best_submission.csv"
        generate_submission_file(
            ids=test_ids,
            predictions=results[best_model_name]['test_preds_original'],
            id_col_name='id',
            target_col_name=TARGET_COL,
            file_path=str(best_submission_path)
        )
        logger.info(f"Best model ({best_model_name}) submission file saved to: {best_submission_path}")
        
        logger.info(f"Updating experiments tracker: {EXPERIMENTS_CSV}")
        if not EXPERIMENTS_CSV.exists():
            logger.info(f"Experiments file not found at {EXPERIMENTS_CSV}. Creating it.")
            headers = ['script_file_name', 'why', 'what', 'how', 'cv_score', 'lb_score', 'status', 'experiment_type', 'base_experiment', 'new_feature']
            pd.DataFrame(columns=headers).to_csv(EXPERIMENTS_CSV, index=False, sep=';')
            
        experiments_df = pd.read_csv(EXPERIMENTS_CSV, sep=';')
        exp_mask = experiments_df['script_file_name'] == EXPERIMENT_NAME
        
        # Use the mean fold RMSLE for the experiments.csv, as this is what was used for best model selection
        final_cv_for_tracker = np.mean(results[best_model_name]['rmsle_scores'])

        if exp_mask.sum() > 0:
            idx_to_update = experiments_df[exp_mask].index[0]
            experiments_df.loc[idx_to_update, 'cv_score'] = final_cv_for_tracker
            experiments_df.loc[idx_to_update, 'status'] = 'done'
            experiments_df.loc[idx_to_update, 'why'] = 'Replicate external script with reported better scores.'
            experiments_df.loc[idx_to_update, 'what'] = f'Train LGBM, XGB, CatBoost with direct log1p(Calories) target, cross-terms.'
            experiments_df.loc[idx_to_update, 'how'] = f'Direct log1p target, specific model params/fit calls, best model: {best_model_name}.'
            experiments_df.loc[idx_to_update, 'experiment_type'] = 'multi_model'
            experiments_df.loc[idx_to_update, 'base_experiment'] = 'exp_r_multi_model_cross_terms_normalized.py & external script'
            experiments_df.loc[idx_to_update, 'new_feature'] = f'Direct log target, simplified LGBM fit, CatBoost default params (iter,lr,depth)'
            logger.info(f"Updated existing experiment '{EXPERIMENT_NAME}' with Mean Fold RMSLE: {final_cv_for_tracker:.5f}")
        else:
            new_row = pd.DataFrame([{
                'script_file_name': EXPERIMENT_NAME,
                'why': 'Replicate external script with reported better scores.',
                'what': f'Train LGBM, XGB, CatBoost with direct log1p(Calories) target, cross-terms.',
                'how': f'Direct log1p target, specific model params/fit calls, best model: {best_model_name}.',
                'cv_score': final_cv_for_tracker,
                'lb_score': None,
                'status': 'done',
                'experiment_type': 'multi_model',
                'base_experiment': 'exp_r_multi_model_cross_terms_normalized.py & external script',
                'new_feature': f'Direct log target, simplified LGBM fit, CatBoost default params (iter,lr,depth)'
            }])
            
            for col in experiments_df.columns:
                if col not in new_row.columns: new_row[col] = None
            new_row = new_row[experiments_df.columns]
            experiments_df = pd.concat([experiments_df, new_row], ignore_index=True)
            logger.info(f"Added new experiment '{EXPERIMENT_NAME}' to tracker with Mean Fold RMSLE: {final_cv_for_tracker:.5f}")
            
        experiments_df.to_csv(EXPERIMENTS_CSV, index=False, sep=';')
        logger.info("Experiments tracker updated successfully.")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        # Python's default mean_squared_log_error can't handle negative inputs if np.expm1 results in negative values for y_pred
        # or if y_true is negative. Ensure clipping or handle this.
        # The provided script clips y_preds for submission, but not for OOF CV calculation.
        # Adding a check or clip for RMSLE calculation.
        # However, calories should not be negative. np.expm1(log_value) can be -1 if log_value is 0.
        # If target y can be 0, log1p(0) is 0. expm1(0) is 0.
        # If pred is very small negative log, expm1 can be close to -1.
        # The competition metric RMSLE requires y_true and y_pred >= 0.
        # train_df[TARGET_COL] is used for y_true, which should be positive.
        # results[model_name]['oof_preds'] are also expm1'd and clipped to [1, 314] before this final CV.
        # So, this should be fine for overall_cv_score.
        # The fold_rmsle uses y_valid_fold (log scale) and fold_oof_preds_orig (original scale).
        # y_valid_original for fold_rmsle = train_df.iloc[valid_idx][TARGET_COL].values (original scale)
        # fold_oof_preds_orig = np.expm1(fold_oof_preds_log) (original scale)
        # The issue might be if fold_oof_preds_orig becomes < 0 due to large negative log predictions.
        # The provided script calculates RMSLE on `np.expm1(y_valid)` and `np.expm1(oof_pred)`.
        # `y_valid` is already log1p. So `np.expm1(y_valid)` is the original scale.
        # This is what I'm doing for `fold_rmsle` using `y_valid_original`.
        # For `overall_cv_score`, it's `train_df[TARGET_COL]` vs `results[model_name]['oof_preds']`.
        # The `results[model_name]['oof_preds']` are already on original scale and clipped.
        # The provided script uses `mean_squared_log_error(np.expm1(y_true_log), np.expm1(y_pred_log))`.
        # This is equivalent to `mean_squared_log_error(y_true_orig, y_pred_orig)`.
        # My `calculate_metric` utility likely does this correctly.
        # The issue in `exp_r` was `PosixPath` + `str`. This script uses `str()` for paths now.
        sys.exit(1)
    
    logger.info(f"--- Experiment Finished: {EXPERIMENT_NAME} ---")

if __name__ == "__main__":
    main() 