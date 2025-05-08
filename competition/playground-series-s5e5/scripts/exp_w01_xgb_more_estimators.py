# Script for Experiment W01: XGBoost with More Estimators (Direct Log Target)
# MODIFIED for Playground Series S5E5: Predict Calorie Expenditure

"""
Extended Description:
This script builds on Experiment V (direct log1p(Calories) target, multi-model).
Key change:
1. XGBoost: Increased n_estimators to 5000 and decreased learning_rate to 0.01.
2. Other models (LightGBM, CatBoost) and their parameters remain as in exp_v.

It includes:
- Loading train and test data
- Basic preprocessing (label encoding for categorical features)
- Feature engineering through cross-terms between numerical features
- Training LightGBM, XGBoost (modified), and CatBoost models using 5-fold CV
- Comparing model performance
- Generating feature importance visualizations
- Creating submission file with the best model's predictions
- Updating the experiments tracker

Corresponds to experiment 'exp_w01_xgb_more_estimators.py' in experiments.csv.
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
import matplotlib.pyplot as plt
import seaborn as sns

# --- Add project root to sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.load_csv import load_csv
    # from utils.calculate_metric import calculate_metric # Using mean_squared_log_error directly
    from utils.generate_submission_file import generate_submission_file
    from utils.setup_logging import setup_logging
    from utils.seed_everything import seed_everything
except ImportError as e:
    print(f"Error importing utility functions: {e}")
    print("Ensure utils directory is at the project root and contains all required files.")
    print("Consider running this script from the project root or as a module.")
    sys.exit(1)

# --- Configuration ---
EXPERIMENT_NAME = "exp_w01_xgb_more_estimators.py"
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

TRAIN_CSV = RAW_DATA_DIR / "train.csv"
TEST_CSV = RAW_DATA_DIR / "test.csv"

OOF_PRED_BASE_PATH = str(OOF_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}")
TEST_PRED_BASE_PATH = str(TEST_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}")
SUBMISSION_BASE_PATH = str(TEST_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}")
FEATURE_IMPORTANCE_BASE_PATH = str(REPORTS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}")

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

# Modified XGBoost Parameters for this experiment
XGB_PARAMS = {
    'max_depth': 10,
    'colsample_bytree': 0.7,
    'subsample': 0.9,
    'n_estimators': 5000,       # Increased
    'learning_rate': 0.01,      # Decreased
    'gamma': 0.01,
    'max_delta_step': 2,
    'early_stopping_rounds': 100, # Will need a long patience if lr is very small
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

def add_feature_cross_terms(df, numerical_features):
    df_new = df.copy()
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):
            feature1 = numerical_features[i]
            feature2 = numerical_features[j]
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    return df_new

def plot_feature_importance(model, model_name, feature_names, output_path):
    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1.0)
    sns.set_style('whitegrid')
    if model_name == 'LightGBM': importance = model.feature_importances_
    elif model_name == 'XGBoost': importance = model.feature_importances_
    elif model_name == 'CatBoost': importance = model.get_feature_importance()
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    top_features = importance_df.head(30)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'{model_name} Feature Importance (Top 30 features)', fontsize=14)
    plt.xlabel('Importance', fontsize=12); plt.ylabel('Feature', fontsize=12)
    plt.tight_layout(); plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close()

def main():
    logger = setup_logging(log_file=str(COMPETITION_DIR / "training.log"))
    seed_everything(SEED)
    logger.info(f"--- Starting Experiment: {EXPERIMENT_NAME} ---")

    try:
        train_df = pd.read_csv(TRAIN_CSV)
        test_df = pd.read_csv(TEST_CSV)
        logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        test_ids = test_df['id']

        le = LabelEncoder()
        train_df['Sex'] = le.fit_transform(train_df['Sex'])
        test_df['Sex'] = le.transform(test_df['Sex'])
        train_df['Sex'] = train_df['Sex'].astype('category')
        test_df['Sex'] = test_df['Sex'].astype('category')
        logger.info("Label encoded 'Sex' and set as categorical type")

        train_df = add_feature_cross_terms(train_df, NUMERICAL_FEATURES)
        test_df = add_feature_cross_terms(test_df, NUMERICAL_FEATURES)
        logger.info(f"Train shape after cross-terms: {train_df.shape}")

        X = train_df.drop(columns=['id', TARGET_COL])
        y = np.log1p(train_df[TARGET_COL])
        X_test = test_df.drop(columns=['id'])
        features_list = X.columns.tolist()
        logger.info(f"Target: direct log1p. X shape: {X.shape}, Total features: {len(features_list)}")

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        model_configs = {
            'LightGBM': LGBMRegressor(**LGBM_PARAMS),
            'XGBoost': XGBRegressor(**XGB_PARAMS), # Using modified params
            'CatBoost': CatBoostRegressor(**CATBOOST_PARAMS)
        }
        results = {name: {'oof_preds': np.zeros(len(train_df)), 'test_preds_sum': np.zeros(len(test_df)),
                           'rmsle_scores': [], 'best_model_instance': None, 'training_time': 0}
                   for name in model_configs}

        for model_name, model_template in model_configs.items():
            logger.info(f"\n=== Training {model_name} ===")
            start_time_model = time.time()
            for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
                logger.info(f"--- Fold {fold + 1}/{N_FOLDS} for {model_name} ---")
                X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
                X_valid_fold, y_valid_fold = X.iloc[valid_idx], y.iloc[valid_idx]
                fold_start_time = time.time()

                current_model_fold = model_template
                if model_name == 'CatBoost': current_model_fold = CatBoostRegressor(**CATBOOST_PARAMS)
                
                if model_name == 'LightGBM':
                    current_model_fold.fit(X_train_fold, y_train_fold, categorical_feature=['Sex'])
                elif model_name == 'XGBoost':
                    current_model_fold.fit(X_train_fold, y_train_fold, eval_set=[(X_valid_fold, y_valid_fold)], verbose=500) # Increased verbose for long training
                elif model_name == 'CatBoost':
                    current_model_fold.fit(X_train_fold, y_train_fold, eval_set=(X_valid_fold, y_valid_fold))

                if fold == N_FOLDS - 1: results[model_name]['best_model_instance'] = current_model_fold
                
                fold_oof_preds_log = current_model_fold.predict(X_valid_fold)
                fold_test_preds_log = current_model_fold.predict(X_test)
                fold_oof_preds_orig = np.expm1(fold_oof_preds_log)
                results[model_name]['oof_preds'][valid_idx] = fold_oof_preds_orig
                results[model_name]['test_preds_sum'] += fold_test_preds_log
                
                # Clip predictions for RMSLE calculation if they go below 0 after expm1, to avoid NaN
                fold_oof_preds_orig_clipped = np.maximum(0, fold_oof_preds_orig)
                y_valid_orig_clipped = np.maximum(0, np.expm1(y_valid_fold))
                fold_rmsle = np.sqrt(mean_squared_log_error(y_valid_orig_clipped, fold_oof_preds_orig_clipped))

                results[model_name]['rmsle_scores'].append(fold_rmsle)
                logger.info(f"Fold {fold + 1} RMSLE: {fold_rmsle:.5f}, Time: {(time.time() - fold_start_time):.2f}s")
            
            results[model_name]['training_time'] = time.time() - start_time_model
            final_test_preds_log = results[model_name]['test_preds_sum'] / N_FOLDS
            results[model_name]['test_preds_original'] = np.expm1(final_test_preds_log)
            results[model_name]['test_preds_original'] = np.clip(results[model_name]['test_preds_original'], 1, 314)
            results[model_name]['oof_preds'] = np.clip(results[model_name]['oof_preds'], 1, 314)
            
            # Overall CV from OOF predictions (already on original scale and clipped)
            overall_cv_score = np.sqrt(mean_squared_log_error(train_df[TARGET_COL], results[model_name]['oof_preds']))
            results[model_name]['overall_cv_score'] = overall_cv_score
            
            logger.info(f"{model_name} - Overall CV (OOF): {overall_cv_score:.5f}, Mean Fold RMSLE: {np.mean(results[model_name]['rmsle_scores']):.5f}, Std: {np.std(results[model_name]['rmsle_scores']):.5f}, Total Time: {results[model_name]['training_time']:.2f}s")

        logger.info("\n=== Model Comparison ===")
        for name in results: logger.info(f"{name} - Mean Fold RMSLE: {np.mean(results[name]['rmsle_scores']):.5f}")
        best_model_name = min(results, key=lambda x: np.mean(results[x]['rmsle_scores']))
        logger.info(f"\nBest Model: {best_model_name} with Mean Fold RMSLE: {np.mean(results[best_model_name]['rmsle_scores']):.5f}")

        for name, data in results.items():
            if data['best_model_instance']: plot_feature_importance(data['best_model_instance'], name, features_list, FEATURE_IMPORTANCE_BASE_PATH + f"_{name}_feature_importance.png")
            np.save(OOF_PRED_BASE_PATH + f"_{name}_oof.npy", data['oof_preds'])
            np.save(TEST_PRED_BASE_PATH + f"_{name}_test.npy", data['test_preds_original'])
            generate_submission_file(test_ids, data['test_preds_original'], 'id', TARGET_COL, str(SUBMISSION_BASE_PATH + f"_{name}_submission.csv"))
        generate_submission_file(test_ids, results[best_model_name]['test_preds_original'], 'id', TARGET_COL, str(SUBMISSION_BASE_PATH + "_best_submission.csv"))

        logger.info(f"Updating experiments tracker: {EXPERIMENTS_CSV}")
        experiments_df = pd.read_csv(EXPERIMENTS_CSV, sep=';')
        exp_mask = experiments_df['script_file_name'] == EXPERIMENT_NAME
        final_cv_for_tracker = np.mean(results[best_model_name]['rmsle_scores'])
        how_str = f'XGB n_est=5000,lr=0.01. Direct log1p target. Best: {best_model_name}.'

        if exp_mask.sum() > 0:
            idx = experiments_df[exp_mask].index[0]
            experiments_df.loc[idx, ['cv_score', 'status', 'how']] = [final_cv_for_tracker, 'done', how_str]
        else:
            new_row = pd.DataFrame([{
                'script_file_name': EXPERIMENT_NAME, 'why': 'Increase n_estimators for XGBoost in exp_v setup.',
                'what': 'Train XGBoost with n_estimators=5000, lr=0.01. Other models from exp_v.',
                'how': how_str, 'cv_score': final_cv_for_tracker, 'status': 'done',
                'experiment_type': 'multi_model', 'base_experiment': 'exp_v_multi_model_direct_log_target.py',
                'new_feature': 'XGBoost: n_estimators=5000, lr=0.01'
            }])
            new_row = new_row.reindex(columns=experiments_df.columns)
            experiments_df = pd.concat([experiments_df, new_row], ignore_index=True)
        experiments_df.to_csv(EXPERIMENTS_CSV, index=False, sep=';')
        logger.info(f"Tracker updated for {EXPERIMENT_NAME} with CV: {final_cv_for_tracker:.5f}")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)
    logger.info(f"--- Experiment Finished: {EXPERIMENT_NAME} ---")

if __name__ == "__main__":
    main() 