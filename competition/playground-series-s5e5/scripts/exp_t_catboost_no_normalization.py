# Script for Experiment T: CatBoost without Target Normalization
# MODIFIED for Playground Series S5E5: Predict Calorie Expenditure

"""
Extended Description:
This script builds on the successful CatBoost model from exp_r but removes target normalization:
1. Keeps cross-term features between numerical features
2. Removes target normalization (uses direct log1p(Calories) instead of log1p(Calories/Duration))
3. Uses CatBoost as the sole model

It includes:
- Loading train and test data
- Basic preprocessing (label encoding for categorical features)
- Feature engineering through cross-terms between numerical features
- Training CatBoost model using 5-fold CV with direct log1p(Calories) as target
- Generating feature importance visualizations
- Creating submission file with predictions
- Updating the experiments tracker

Corresponds to experiment 'exp_t_catboost_no_normalization.py' in experiments.csv.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import time
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
EXPERIMENT_NAME = "exp_t_catboost_no_normalization.py"
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

# Output Paths
OOF_PRED_OUTPUT_PATH = OOF_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_oof.npy"
TEST_PRED_OUTPUT_PATH = TEST_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_test.npy"
SUBMISSION_OUTPUT_PATH = TEST_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_submission.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_feature_importance.png"

# Create output directories
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OOF_PREDS_DIR.mkdir(parents=True, exist_ok=True)
TEST_PREDS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Model Parameters ---
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
def plot_feature_importance(model, feature_names, output_path):
    """
    Generate and save a feature importance plot for the CatBoost model.
    
    Args:
        model: Trained CatBoost model
        feature_names (list): List of feature names
        output_path (str or Path): Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1.0)
    sns.set_style('whitegrid')
    
    # Get feature importance
    importance = model.get_feature_importance()
    
    # Create dataframe of feature importances
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot top 30 features
    top_features = importance_df.head(30)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    
    # Add labels and title
    plt.title(f'CatBoost Feature Importance (Top 30 features)', fontsize=14)
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
                
                # Set as categorical type for CatBoost
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
        
        # Prepare features and target
        X = train_df.drop(columns=['id', TARGET_COL])
        
        # Use direct log1p(Calories) as target (NO NORMALIZATION)
        y = np.log1p(train_df[TARGET_COL])
        logger.info("Using direct log1p(Calories) as target (no normalization)")
        
        X_test = test_df.drop(columns=['id'])
        
        logger.info(f"X shape: {X.shape}, y shape: {len(y)}, X_test shape: {X_test.shape}")
        
        # Get feature list
        features = X.columns.tolist()
        logger.info(f"Total features: {len(features)}")
        
        # --- Cross-Validation and Model Training ---
        logger.info(f"Starting {N_FOLDS}-Fold Cross-Validation with CatBoost...")
        
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        
        oof_preds = np.zeros(len(train_df))
        test_preds_sum = np.zeros(len(test_df))
        rmsle_scores = []
        training_time = 0
        best_model = None
        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
            logger.info(f"--- Fold {fold + 1}/{N_FOLDS} ---")
            
            # Slice data for the fold
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_valid_fold = X.iloc[valid_idx]
            y_valid_fold = y.iloc[valid_idx]
            
            fold_start_time = time.time()
            logger.info(f"Training CatBoost fold {fold + 1}...")
            
            # Initialize and train CatBoost model
            model_fold = CatBoostRegressor(**CATBOOST_PARAMS)
            model_fold.fit(
                X_train_fold, y_train_fold,
                eval_set=(X_valid_fold, y_valid_fold),
                verbose=100
            )
            
            # Save the model from the last fold
            if fold == N_FOLDS - 1:
                best_model = model_fold
            
            # Predict on validation fold (log scale)
            fold_oof_preds_log = model_fold.predict(X_valid_fold)
            fold_test_preds_log = model_fold.predict(X_test)
            
            # Transform predictions back to original scale
            fold_oof_preds_orig = np.expm1(fold_oof_preds_log)
            
            # Store predictions
            oof_preds[valid_idx] = fold_oof_preds_orig
            test_preds_sum += fold_test_preds_log
            
            # Calculate fold RMSLE
            y_valid_original = train_df.iloc[valid_idx][TARGET_COL].values
            fold_rmsle = calculate_metric(y_true=y_valid_original, y_pred=fold_oof_preds_orig, metric_name='rmsle')
            rmsle_scores.append(fold_rmsle)
            
            fold_time = time.time() - fold_start_time
            training_time += fold_time
            logger.info(f"Fold {fold + 1} RMSLE: {fold_rmsle:.5f}, Training time: {fold_time:.2f}s")
        
        # --- Process Results ---
        # Transform average test predictions back to original scale
        final_test_preds_log = test_preds_sum / N_FOLDS
        final_test_preds_original = np.expm1(final_test_preds_log)
        
        # Clip final predictions
        final_test_preds_original = np.clip(final_test_preds_original, 1, 314)
        oof_preds = np.clip(oof_preds, 1, 314)
        
        # Calculate overall CV score
        y_true_for_metric = train_df[TARGET_COL].values
        overall_cv_score = calculate_metric(
            y_true=y_true_for_metric, 
            y_pred=oof_preds,
            metric_name='rmsle'
        )
        
        logger.info(f"CatBoost - Overall CV Score (RMSLE): {overall_cv_score:.5f}")
        logger.info(f"CatBoost - Mean fold RMSLE: {np.mean(rmsle_scores):.5f}, Std: {np.std(rmsle_scores):.5f}")
        logger.info(f"CatBoost - Total training time: {training_time:.2f}s")
        
        # --- Feature Importance and Saving Results ---
        # Generate Feature Importance Plot
        if best_model is not None:
            logger.info(f"Generating feature importance plot...")
            plot_feature_importance(
                model=best_model,
                feature_names=features,
                output_path=FEATURE_IMPORTANCE_PATH
            )
            logger.info(f"Feature importance plot saved to: {FEATURE_IMPORTANCE_PATH}")
        
        # Save OOF predictions
        np.save(OOF_PRED_OUTPUT_PATH, oof_preds)
        logger.info(f"OOF predictions saved to: {OOF_PRED_OUTPUT_PATH}")
        
        # Save test predictions
        np.save(TEST_PRED_OUTPUT_PATH, final_test_preds_original)
        logger.info(f"Test predictions saved to: {TEST_PRED_OUTPUT_PATH}")
        
        # Generate submission file
        generate_submission_file(
            ids=test_ids,
            predictions=final_test_preds_original,
            id_col_name='id',
            target_col_name=TARGET_COL,
            file_path=str(SUBMISSION_OUTPUT_PATH)
        )
        logger.info(f"Submission file saved to: {SUBMISSION_OUTPUT_PATH}")
        
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
            experiments_df.loc[idx_to_update, 'cv_score'] = overall_cv_score
            experiments_df.loc[idx_to_update, 'status'] = 'done'
            experiments_df.loc[idx_to_update, 'why'] = 'Evaluate CatBoost performance without target normalization.'
            experiments_df.loc[idx_to_update, 'what'] = 'Train CatBoost with cross-term features but without target normalization.'
            experiments_df.loc[idx_to_update, 'how'] = 'Create cross-terms, use direct log1p(Calories) target without normalization by Duration.'
            experiments_df.loc[idx_to_update, 'experiment_type'] = 'single_model'
            experiments_df.loc[idx_to_update, 'base_experiment'] = 'exp_r_multi_model_cross_terms_normalized.py'
            experiments_df.loc[idx_to_update, 'new_feature'] = 'No target normalization (direct log1p)'
            logger.info(f"Updated existing experiment '{EXPERIMENT_NAME}' with CV score: {overall_cv_score:.5f}")
        else:
            # Add as new row
            new_row = pd.DataFrame([{
                'script_file_name': EXPERIMENT_NAME,
                'why': 'Evaluate CatBoost performance without target normalization.',
                'what': 'Train CatBoost with cross-term features but without target normalization.',
                'how': 'Create cross-terms, use direct log1p(Calories) target without normalization by Duration.',
                'cv_score': overall_cv_score,
                'lb_score': None,
                'status': 'done',
                'experiment_type': 'single_model',
                'base_experiment': 'exp_r_multi_model_cross_terms_normalized.py',
                'new_feature': 'No target normalization (direct log1p)'
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