# Script for Experiment N: Evaluate Impact of Soft Clipping on CV Score
# MODIFIED for Playground Series S5E5: Predict Calorie Expenditure

"""
Extended Description:
This script evaluates how soft clipping affects the CV score without retraining the model:
1. Loads the OOF predictions from experiment L (which used ratio-to-category mean encoding)
2. Applies a 'soft clipping' to high values rather than hard clipping:
   - Values below threshold (250) remain unchanged
   - Values above threshold are transformed using: threshold + log(1 + value - threshold)
3. Recalculates the CV score using the soft-clipped predictions
4. Saves the transformed OOF predictions, test predictions, and generates a new submission
5. Updates the experiments tracker with the new CV score

This experiment tests if a less aggressive transformation of high values can preserve more
information while still reducing the impact of outliers.

Corresponds to experiment 'exp_n_soft_clipping.py' in experiments.csv.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

# --- Add project root to sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.load_csv import load_csv
    from utils.calculate_metric import calculate_metric
    from utils.generate_submission_file import generate_submission_file
    from utils.setup_logging import setup_logging
except ImportError as e:
    print(f"Error importing utility functions: {e}")
    print("Ensure utils directory is at the project root and contains all required files.")
    print("Consider running this script from the project root or as a module.")
    sys.exit(1)

# --- Configuration ---
EXPERIMENT_NAME = "exp_n_soft_clipping.py"
SOURCE_EXPERIMENT = "exp_l_lgbm_ratio_encoding.py"
TARGET_COL = "Calories"
CLIP_MIN = 1
THRESHOLD = 250  # Values beyond this will be transformed with log

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
COMPETITION_DIR = SCRIPT_DIR.parent
RAW_DATA_DIR = COMPETITION_DIR / "data" / "raw"
OOF_PREDS_DIR = COMPETITION_DIR / "predictions" / "oof"
TEST_PREDS_DIR = COMPETITION_DIR / "predictions" / "test"
REPORTS_DIR = COMPETITION_DIR / "reports"
EXPERIMENTS_CSV = COMPETITION_DIR / "experiments.csv"

# Input Data Paths
TRAIN_CSV = RAW_DATA_DIR / "train.csv"
TEST_CSV = RAW_DATA_DIR / "test.csv"

# Source and Output Paths
SOURCE_OOF_PATH = OOF_PREDS_DIR / f"{SOURCE_EXPERIMENT.replace('.py', '')}_oof.npy"
SOURCE_TEST_PATH = TEST_PREDS_DIR / f"{SOURCE_EXPERIMENT.replace('.py', '')}_test_avg.npy"

OOF_PRED_OUTPUT_PATH = OOF_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_oof.npy"
TEST_PRED_OUTPUT_PATH = TEST_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_test_avg.npy"
SUBMISSION_OUTPUT_PATH = TEST_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_submission.csv"
TRANSFORM_PLOT_PATH = REPORTS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_transformation.png"

def soft_clip(values, threshold, min_val=None):
    """
    Apply soft clipping transformation:
    - Values below threshold remain unchanged
    - Values above threshold are transformed using log: threshold + log(1 + value - threshold)
    - Values below min_val (if provided) are clipped to min_val
    
    Args:
        values (numpy.ndarray): Array of values to transform
        threshold (float): Threshold above which to apply log transformation
        min_val (float, optional): Minimum value to clip to
        
    Returns:
        numpy.ndarray: Transformed values
    """
    result = values.copy()
    
    # Apply minimum value clipping if specified
    if min_val is not None:
        result = np.maximum(min_val, result)
    
    # Apply log transformation to values above threshold
    mask = result > threshold
    if np.any(mask):
        result[mask] = threshold + np.log1p(result[mask] - threshold)
    
    return result

def plot_transformation(values_range, transformed_values, threshold, output_path):
    """
    Generate a plot showing the effect of the soft clipping transformation.
    
    Args:
        values_range (numpy.ndarray): Array of original values
        transformed_values (numpy.ndarray): Array of transformed values
        threshold (float): Threshold used in the transformation
        output_path (str or Path): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot the transformation
    plt.plot(values_range, values_range, 'b--', label='Original (y=x)')
    plt.plot(values_range, transformed_values, 'r-', label='Soft Clipped')
    
    # Add reference lines
    plt.axvline(x=threshold, color='g', linestyle=':', label=f'Threshold = {threshold}')
    plt.axhline(y=threshold, color='g', linestyle=':')
    
    # Add labels and legend
    plt.xlabel('Original Value')
    plt.ylabel('Transformed Value')
    plt.title('Soft Clipping Transformation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    logger = setup_logging(log_file=str(COMPETITION_DIR / "evaluation.log"))
    logger.info(f"--- Starting Experiment: {EXPERIMENT_NAME} ---")
    
    # Create output directory if it doesn't exist
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load original training data to get actual target values
    try:
        logger.info(f"Loading training data from: {TRAIN_CSV}")
        df_train = load_csv(str(TRAIN_CSV))
        logger.info(f"Train shape: {df_train.shape}")
        
        # Extract the true target values
        y_true = df_train[TARGET_COL].to_numpy()
        
        # Load test IDs for submission file generation
        logger.info(f"Loading test data IDs from: {TEST_CSV}")
        df_test = load_csv(str(TEST_CSV))
        test_ids = df_test['id'].to_pandas()
        
    except FileNotFoundError as e:
        logger.error(f"Error loading data file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading: {e}")
        sys.exit(1)
    
    # Load OOF predictions from source experiment
    try:
        logger.info(f"Loading OOF predictions from: {SOURCE_OOF_PATH}")
        oof_preds_original = np.load(SOURCE_OOF_PATH)
        
        logger.info(f"Loading test predictions from: {SOURCE_TEST_PATH}")
        test_preds_original = np.load(SOURCE_TEST_PATH)
        
        logger.info(f"Original OOF predictions shape: {oof_preds_original.shape}")
        logger.info(f"Original test predictions shape: {test_preds_original.shape}")
        
    except FileNotFoundError as e:
        logger.error(f"Error loading prediction files: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred loading predictions: {e}")
        sys.exit(1)
    
    # Analyze original predictions
    logger.info(f"Original OOF predictions - Min: {oof_preds_original.min():.2f}, Max: {oof_preds_original.max():.2f}")
    logger.info(f"Original test predictions - Min: {test_preds_original.min():.2f}, Max: {test_preds_original.max():.2f}")
    
    # Plot the transformation function
    values_range = np.linspace(0, 350, 1000)
    transformed_values = soft_clip(values_range, THRESHOLD, CLIP_MIN)
    plot_transformation(values_range, transformed_values, THRESHOLD, TRANSFORM_PLOT_PATH)
    logger.info(f"Transformation plot saved to: {TRANSFORM_PLOT_PATH}")
    
    # Calculate original CV score 
    original_cv_score = calculate_metric(y_true=y_true, y_pred=oof_preds_original, metric_name='rmsle')
    logger.info(f"Original CV Score (RMSLE): {original_cv_score:.5f}")
    
    # Apply soft clipping
    logger.info(f"Applying soft clipping with threshold={THRESHOLD}, min={CLIP_MIN}")
    oof_preds_transformed = soft_clip(oof_preds_original, THRESHOLD, CLIP_MIN)
    test_preds_transformed = soft_clip(test_preds_original, THRESHOLD, CLIP_MIN)
    
    # Analyze transformed predictions
    logger.info(f"Transformed OOF predictions - Min: {oof_preds_transformed.min():.2f}, Max: {oof_preds_transformed.max():.2f}")
    logger.info(f"Transformed test predictions - Min: {test_preds_transformed.min():.2f}, Max: {test_preds_transformed.max():.2f}")
    
    # Calculate changes from transformation
    num_modified_oof = np.sum((oof_preds_original != oof_preds_transformed))
    pct_modified_oof = 100 * num_modified_oof / len(oof_preds_original)
    logger.info(f"Modified {num_modified_oof} OOF predictions ({pct_modified_oof:.2f}%)")
    
    num_modified_test = np.sum((test_preds_original != test_preds_transformed))
    pct_modified_test = 100 * num_modified_test / len(test_preds_original)
    logger.info(f"Modified {num_modified_test} test predictions ({pct_modified_test:.2f}%)")
    
    # Calculate new CV score with transformed predictions
    transformed_cv_score = calculate_metric(y_true=y_true, y_pred=oof_preds_transformed, metric_name='rmsle')
    logger.info(f"Transformed CV Score (RMSLE): {transformed_cv_score:.5f}")
    
    # Calculate score improvement (if any)
    score_diff = original_cv_score - transformed_cv_score
    pct_improvement = 100 * score_diff / original_cv_score
    logger.info(f"Score difference: {score_diff:.5f} ({pct_improvement:.4f}% {'improvement' if score_diff > 0 else 'worse'})")
    
    # Save transformed predictions
    logger.info("Saving transformed OOF and test predictions...")
    np.save(OOF_PRED_OUTPUT_PATH, oof_preds_transformed)
    logger.info(f"Transformed OOF predictions saved to: {OOF_PRED_OUTPUT_PATH}")
    np.save(TEST_PRED_OUTPUT_PATH, test_preds_transformed)
    logger.info(f"Transformed test predictions saved to: {TEST_PRED_OUTPUT_PATH}")
    
    # Generate new submission file
    logger.info("Generating submission file with transformed predictions...")
    generate_submission_file(
        ids=test_ids,
        predictions=test_preds_transformed,
        id_col_name='id',
        target_col_name=TARGET_COL,
        file_path=str(SUBMISSION_OUTPUT_PATH)
    )
    logger.info(f"Submission file saved to: {SUBMISSION_OUTPUT_PATH}")
    
    # --- Update Experiments Tracker ---
    try:
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
            experiments_df.loc[idx_to_update, 'cv_score'] = transformed_cv_score
            experiments_df.loc[idx_to_update, 'status'] = 'done'
            experiments_df.loc[idx_to_update, 'why'] = f'Evaluate impact of soft clipping with threshold={THRESHOLD}.'
            experiments_df.loc[idx_to_update, 'what'] = 'Load existing predictions, apply logarithmic transformation to high values, recalculate CV score.'
            experiments_df.loc[idx_to_update, 'how'] = f'Loaded OOF/test predictions from {SOURCE_EXPERIMENT}, applied threshold + log1p(value - threshold) for values > {THRESHOLD}, recalculated RMSLE.'
            experiments_df.loc[idx_to_update, 'experiment_type'] = 'post_processing'
            experiments_df.loc[idx_to_update, 'base_experiment'] = SOURCE_EXPERIMENT
            experiments_df.loc[idx_to_update, 'new_feature'] = f'Soft clipping with threshold={THRESHOLD}'
            logger.info(f"Updated existing experiment '{EXPERIMENT_NAME}' with CV score: {transformed_cv_score:.5f}")
        else:
            # Add as new row
            new_row = pd.DataFrame([{
                'script_file_name': EXPERIMENT_NAME,
                'why': f'Evaluate impact of soft clipping with threshold={THRESHOLD}.',
                'what': 'Load existing predictions, apply logarithmic transformation to high values, recalculate CV score.',
                'how': f'Loaded OOF/test predictions from {SOURCE_EXPERIMENT}, applied threshold + log1p(value - threshold) for values > {THRESHOLD}, recalculated RMSLE.',
                'cv_score': transformed_cv_score,
                'lb_score': None,
                'status': 'done',
                'experiment_type': 'post_processing',
                'base_experiment': SOURCE_EXPERIMENT,
                'new_feature': f'Soft clipping with threshold={THRESHOLD}'
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
        logger.error(f"An error occurred while updating experiments tracker: {e}")

    logger.info(f"--- Experiment Finished: {EXPERIMENT_NAME} ---")

if __name__ == "__main__":
    main() 