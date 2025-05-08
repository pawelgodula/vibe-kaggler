# Script for Experiment R: Multi-Model Comparison with Cross-Terms and Normalized Target
# MODIFIED for Playground Series S5E5: Predict Calorie Expenditure

"""
Extended Description:
This script extends Experiment Q by adding multiple model types:
1. LightGBM (same as exp_q)
2. XGBoost 
3. CatBoost

It maintains the same approach for:
- Feature engineering through cross-terms between numerical features
- Target normalization by dividing Calories by Duration

It includes:
- Loading train and test data
- Basic preprocessing (label encoding for categorical features)
- Feature engineering through cross-terms between numerical features
- Target normalization by dividing Calories by Duration before applying log1p
- Training LightGBM, XGBoost, and CatBoost models using 5-fold CV
- Transforming predictions back to the original scale
- Comparing model performance
- Generating feature importance visualizations
- Creating submission file with the best model's predictions
- Updating the experiments tracker

Corresponds to experiment 'exp_r_multi_model_cross_terms_normalized.py' in experiments.csv.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import time
import lightgbm as lgb
from lightgbm import LGBMRegressor
import xgboost as xgb
from xgboost import XGBRegressor
import catboost as cb
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
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
EXPERIMENT_NAME = "exp_r_multi_model_cross_terms_normalized.py"
N_FOLDS = 5
SEED = 42
TARGET_COL = "Calories"
NUMERICAL_FEATURES = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
CATEGORICAL_FEATURES = ['Sex']

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
COMPETITION_DIR = SCRIPT_DIR.parent
RAW_DATA_DIR = COMPETITION_DIR / "data" / "raw"
PROCESSED_DATA_DIR = COMPETITION_DIR / "data" / "processed"
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
# LightGBM Parameters
LGBM_PARAMS = {
    'n_estimators': 2000,
    'learning_rate': 0.02,
    'max_depth': 10,
    'colsample_bytree': 0.7,
    'subsample': 0.9,
    'random_state': SEED,
    'verbose': -1
}

# XGBoost Parameters
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

# CatBoost Parameters
CATBOOST_PARAMS = {
    'verbose': 100,
    'random_seed': SEED,
    'cat_features': ['Sex'],
    'early_stopping_rounds': 100,
    'iterations': 2000,
    'learning_rate': 0.02,
    'depth': 10
}

# Function to add cross-term features between numerical features
def add_feature_cross_terms(df, numerical_features):
    """
    Create cross-term features by multiplying each pair of numerical features

    Args:
        df (pandas.DataFrame): Input dataframe
        numerical_features (list): List of numerical feature names
        
    Returns:
        pandas.DataFrame: Dataframe with added cross-term features
    """
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
    """
    Generate and save a feature importance plot for the specified model.
    
    Args:
        model: Trained model
        model_name (str): Name of the model
        feature_names (list): List of feature names
        output_path (str or Path): Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1.0)
    sns.set_style('whitegrid')
    
    # Get feature importance based on model type
    if model_name == 'LightGBM':
        importance = model.feature_importances_
    elif model_name == 'XGBoost':
        importance = model.feature_importances_
    elif model_name == 'CatBoost':
        importance = model.get_feature_importance()
    
    # Create dataframe of feature importances
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot top 30 features
    top_features = importance_df.head(30)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    
    # Add labels and title
    plt.title(f'{model_name} Feature Importance (Top 30 features)', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    # Save figure
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
        train_df = pd.read_csv(TRAIN_CSV)
        logger.info(f"Train shape: {train_df.shape}")

        logger.info(f"Loading test data from: {TEST_CSV}")
        test_df = pd.read_csv(TEST_CSV)
        logger.info(f"Test shape: {test_df.shape}")

        # Store test IDs for submission
        test_ids = test_df['id']
        
        # --- Basic Preprocessing ---
        logger.info("Performing basic preprocessing...")
        
        # Label encode categorical features
        logger.info(f"Label encoding categorical features: {CATEGORICAL_FEATURES}")
        le = LabelEncoder()
        for col in CATEGORICAL_FEATURES:
            if col in train_df.columns:
                train_df[col] = le.fit_transform(train_df[col])
                test_df[col] = le.transform(test_df[col])
                
                # Set as categorical type for LightGBM and others
                train_df[col] = train_df[col].astype('category')
                test_df[col] = test_df[col].astype('category')
                
                logger.info(f"Encoded {col} and set as categorical type")
        
        # --- Feature Engineering ---
        logger.info("Creating cross-term features...")
        train_df = add_feature_cross_terms(train_df, NUMERICAL_FEATURES)
        test_df = add_feature_cross_terms(test_df, NUMERICAL_FEATURES)
        logger.info(f"Train shape after adding cross-terms: {train_df.shape}")
        logger.info(f"Test shape after adding cross-terms: {test_df.shape}")
        
        # --- Prepare Data for Model Training ---
        logger.info("Preparing data for modeling...")
        
        # Store Duration for un-normalization later
        train_duration = train_df['Duration'].values
        test_duration = test_df['Duration'].values
        
        # Prepare features and target
        X = train_df.drop(columns=['id', TARGET_COL])
        
        # Normalize target by Duration before applying log1p
        normalized_target = train_df[TARGET_COL] / train_df['Duration']
        y = np.log1p(normalized_target)
        logger.info("Normalized target by Duration (calories per minute) and applied log1p transformation")
        
        X_test = test_df.drop(columns=['id'])
        
        logger.info(f"X shape: {X.shape}, y shape: {len(y)}, X_test shape: {X_test.shape}")
        
        # Get feature list
        features = X.columns.tolist()
        logger.info(f"Total features: {len(features)}")
        
        # --- Cross-Validation and Model Training ---
        logger.info(f"Starting {N_FOLDS}-Fold Cross-Validation with multiple models...")
        
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        
        # Dictionary to store models and results
        models = {
            'LightGBM': LGBMRegressor(**LGBM_PARAMS),
            'XGBoost': XGBRegressor(**XGB_PARAMS),
            'CatBoost': CatBoostRegressor(**CATBOOST_PARAMS)
        }
        
        results = {
            name: {
                'oof_preds': np.zeros(len(train_df)),
                'test_preds_sum': np.zeros(len(test_df)),
                'rmsle_scores': [],
                'best_model': None,
                'training_time': 0
            } for name in models
        }
        
        # Training loop for each model
        for model_name, model in models.items():
            logger.info(f"\n=== Training {model_name} ===")
            start_time = time.time()
            
            for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
                logger.info(f"--- Fold {fold + 1}/{N_FOLDS} for {model_name} ---")
                
                # Slice data for the fold
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_valid_fold = X.iloc[valid_idx]
                y_valid_fold = y.iloc[valid_idx]
                valid_duration = train_duration[valid_idx]
                
                fold_start_time = time.time()
                logger.info(f"Training {model_name} fold {fold + 1}...")
                
                # Model specific training
                if model_name == 'LightGBM':
                    model_fold = LGBMRegressor(**LGBM_PARAMS)
                    model_fold.fit(
                        X_train_fold, y_train_fold,
                        eval_set=[(X_train_fold, y_train_fold), (X_valid_fold, y_valid_fold)],
                        eval_names=['train', 'validation'],
                        callbacks=[
                            lgb.early_stopping(30, verbose=False),
                            lgb.log_evaluation(100)
                        ],
                        categorical_feature=CATEGORICAL_FEATURES
                    )
                elif model_name == 'XGBoost':
                    model_fold = XGBRegressor(**XGB_PARAMS)
                    model_fold.fit(
                        X_train_fold, y_train_fold,
                        eval_set=[(X_valid_fold, y_valid_fold)],
                        verbose=100
                    )
                elif model_name == 'CatBoost':
                    model_fold = CatBoostRegressor(**CATBOOST_PARAMS)
                    model_fold.fit(
                        X_train_fold, y_train_fold,
                        eval_set=(X_valid_fold, y_valid_fold),
                        verbose=100
                    )
                
                # Save the model from the last fold
                if fold == N_FOLDS - 1:
                    results[model_name]['best_model'] = model_fold
                
                # Predict on validation fold (log scale of normalized target)
                fold_oof_preds_log = model_fold.predict(X_valid_fold)
                fold_test_preds_log = model_fold.predict(X_test)
                
                # Transform log predictions back to original scale:
                # 1. expm1 to undo log1p
                # 2. multiply by Duration to get total calories from calories/minute
                fold_oof_preds_orig = np.expm1(fold_oof_preds_log) * valid_duration
                
                # Store predictions (original scale)
                results[model_name]['oof_preds'][valid_idx] = fold_oof_preds_orig
                
                # For test predictions, we'll accumulate the log predictions and transform at the end
                results[model_name]['test_preds_sum'] += fold_test_preds_log
                
                # Calculate fold RMSLE on original scale
                y_valid_original = train_df.iloc[valid_idx][TARGET_COL].values
                fold_rmsle = calculate_metric(y_true=y_valid_original, y_pred=fold_oof_preds_orig, metric_name='rmsle')
                results[model_name]['rmsle_scores'].append(fold_rmsle)
                
                fold_time = time.time() - fold_start_time
                logger.info(f"Fold {fold + 1} RMSLE: {fold_rmsle:.5f}, Training time: {fold_time:.2f}s")
            
            # Record total training time
            results[model_name]['training_time'] = time.time() - start_time
            
            # Transform average test predictions back to original scale
            # 1. Average the log predictions
            # 2. expm1 to undo log1p
            # 3. multiply by test Duration to get total calories
            final_test_preds_log = results[model_name]['test_preds_sum'] / N_FOLDS
            results[model_name]['test_preds_original'] = np.expm1(final_test_preds_log) * test_duration
            
            # Clip final predictions
            results[model_name]['test_preds_original'] = np.clip(results[model_name]['test_preds_original'], 1, 314)
            results[model_name]['oof_preds'] = np.clip(results[model_name]['oof_preds'], 1, 314)
            
            # Calculate overall CV score
            y_true_for_metric = train_df[TARGET_COL].values
            overall_cv_score = calculate_metric(
                y_true=y_true_for_metric, 
                y_pred=results[model_name]['oof_preds'],
                metric_name='rmsle'
            )
            results[model_name]['overall_cv_score'] = overall_cv_score
            
            logger.info(f"{model_name} - Overall CV Score (RMSLE): {overall_cv_score:.5f}")
            logger.info(f"{model_name} - Mean fold RMSLE: {np.mean(results[model_name]['rmsle_scores']):.5f}, Std: {np.std(results[model_name]['rmsle_scores']):.5f}")
            logger.info(f"{model_name} - Total training time: {results[model_name]['training_time']:.2f}s")
        
        # --- Model Comparison ---
        logger.info("\n=== Model Comparison ===")
        for model_name in results:
            logger.info(f"{model_name} - CV Score: {results[model_name]['overall_cv_score']:.5f}, Training time: {results[model_name]['training_time']:.2f}s")
        
        # Find best model
        best_model_name = min(results, key=lambda x: results[x]['overall_cv_score'])
        logger.info(f"\nBest Model: {best_model_name} with CV Score: {results[best_model_name]['overall_cv_score']:.5f}")
        
        # --- Feature Importance and Saving Results ---
        for model_name in results:
            # Generate Feature Importance Plot
            if results[model_name]['best_model'] is not None:
                logger.info(f"Generating feature importance plot for {model_name}...")
                plot_feature_importance(
                    model=results[model_name]['best_model'],
                    model_name=model_name,
                    feature_names=features,
                    output_path=FEATURE_IMPORTANCE_BASE_PATH + f"_{model_name}_feature_importance.png"
                )
            
            # Save OOF predictions
            oof_path = OOF_PRED_BASE_PATH + f"_{model_name}_oof.npy"
            np.save(oof_path, results[model_name]['oof_preds'])
            logger.info(f"{model_name} OOF predictions saved to: {oof_path}")
            
            # Save test predictions
            test_path = TEST_PRED_BASE_PATH + f"_{model_name}_test.npy"
            np.save(test_path, results[model_name]['test_preds_original'])
            logger.info(f"{model_name} test predictions saved to: {test_path}")
            
            # Generate submission file for this model
            submission_path = SUBMISSION_BASE_PATH + f"_{model_name}_submission.csv"
            generate_submission_file(
                ids=test_ids,
                predictions=results[model_name]['test_preds_original'],
                id_col_name='id',
                target_col_name=TARGET_COL,
                file_path=str(submission_path)
            )
            logger.info(f"{model_name} submission file saved to: {submission_path}")
        
        # Create Best Model's submission
        best_submission_path = SUBMISSION_BASE_PATH + "_best_submission.csv"
        generate_submission_file(
            ids=test_ids,
            predictions=results[best_model_name]['test_preds_original'],
            id_col_name='id',
            target_col_name=TARGET_COL,
            file_path=str(best_submission_path)
        )
        logger.info(f"Best model ({best_model_name}) submission file saved to: {best_submission_path}")
        
        # --- Update Experiments Tracker ---
        logger.info(f"Updating experiments tracker: {EXPERIMENTS_CSV}")
        if not EXPERIMENTS_CSV.exists():
            logger.info(f"Experiments file not found at {EXPERIMENTS_CSV}. Creating it.")
            headers = ['script_file_name', 'why', 'what', 'how', 'cv_score', 'lb_score', 'status', 'experiment_type', 'base_experiment', 'new_feature']
            pd.DataFrame(columns=headers).to_csv(EXPERIMENTS_CSV, index=False, sep=';')
            
        experiments_df = pd.read_csv(EXPERIMENTS_CSV, sep=';')
        exp_mask = experiments_df['script_file_name'] == EXPERIMENT_NAME
        if exp_mask.sum() > 0:
            # Update existing row
            idx_to_update = experiments_df[exp_mask].index[0]
            experiments_df.loc[idx_to_update, 'cv_score'] = results[best_model_name]['overall_cv_score']
            experiments_df.loc[idx_to_update, 'status'] = 'done'
            experiments_df.loc[idx_to_update, 'why'] = 'Compare multiple models with the best feature engineering approach.'
            experiments_df.loc[idx_to_update, 'what'] = f'Train LightGBM, XGBoost and CatBoost with cross-term features and normalized target.'
            experiments_df.loc[idx_to_update, 'how'] = f'Create cross-terms, normalize target by Duration, train 3 model types, best model: {best_model_name}.'
            experiments_df.loc[idx_to_update, 'experiment_type'] = 'multi_model'
            experiments_df.loc[idx_to_update, 'base_experiment'] = 'exp_q_lgbm_cross_terms_normalized.py'
            experiments_df.loc[idx_to_update, 'new_feature'] = f'Multiple model comparison (best: {best_model_name})'
            logger.info(f"Updated existing experiment '{EXPERIMENT_NAME}' with CV score: {results[best_model_name]['overall_cv_score']:.5f}")
        else:
            # Add as new row
            new_row = pd.DataFrame([{
                'script_file_name': EXPERIMENT_NAME,
                'why': 'Compare multiple models with the best feature engineering approach.',
                'what': f'Train LightGBM, XGBoost and CatBoost with cross-term features and normalized target.',
                'how': f'Create cross-terms, normalize target by Duration, train 3 model types, best model: {best_model_name}.',
                'cv_score': results[best_model_name]['overall_cv_score'],
                'lb_score': None,
                'status': 'done',
                'experiment_type': 'multi_model',
                'base_experiment': 'exp_q_lgbm_cross_terms_normalized.py',
                'new_feature': f'Multiple model comparison (best: {best_model_name})'
            }])
            
            # Ensure columns align
            for col in experiments_df.columns:
                if col not in new_row.columns:
                    new_row[col] = None
            
            new_row = new_row[experiments_df.columns]
            experiments_df = pd.concat([experiments_df, new_row], ignore_index=True)
            logger.info(f"Added new experiment '{EXPERIMENT_NAME}' to tracker.")
            
        experiments_df.to_csv(EXPERIMENTS_CSV, index=False, sep=';')
        logger.info("Experiments tracker updated successfully.")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info(f"--- Experiment Finished: {EXPERIMENT_NAME} ---")

if __name__ == "__main__":
    main() 