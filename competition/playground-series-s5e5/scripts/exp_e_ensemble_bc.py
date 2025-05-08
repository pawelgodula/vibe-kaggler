# Script for Experiment E: Ensemble Average of Experiments B and C
# MODIFIED for Playground Series S5E5: Predict Calorie Expenditure

"""
Extended Description:
This script creates an ensemble of two previous experiments by simple averaging:
- exp_b_lgbm_combinations_te.py (100% data, log1p transform)
- exp_c_lgbm_normalized_target.py (100% data, calories/minute approach)

It loads the OOF and test predictions from each experiment, calculates their average,
evaluates the ensemble performance, and generates a submission file.

This experiment differs from exp_d by excluding exp_a (which used only 10% of the data).

Corresponds to experiment 'exp_e_ensemble_bc.py' in experiments.csv.
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
    from utils.seed_everything import seed_everything
except ImportError as e:
    print(f"Error importing utility functions: {e}")
    print("Ensure utils directory is at the project root and contains all required files.")
    print("Consider running this script from the project root or as a module.")
    sys.exit(1)

# --- Configuration ---
EXPERIMENT_NAME = "exp_e_ensemble_bc.py"
SEED = 42
TARGET_COL = "Calories"

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
COMPETITION_DIR = SCRIPT_DIR.parent
RAW_DATA_DIR = COMPETITION_DIR / "data" / "raw"
OOF_PREDS_DIR = COMPETITION_DIR / "predictions" / "oof"
TEST_PREDS_DIR = COMPETITION_DIR / "predictions" / "test"
EXPERIMENTS_CSV = COMPETITION_DIR / "experiments.csv"

# Input Data Paths
TEST_CSV = RAW_DATA_DIR / "test.csv"
TRAIN_CSV = RAW_DATA_DIR / "train.csv"

# Input Predictions Paths - Only using experiments b and c
EXPERIMENT_DETAILS = [
    {
        'name': 'exp_b_lgbm_combinations_te',
        'oof_path': OOF_PREDS_DIR / 'exp_b_lgbm_combinations_te_oof.npy',
        'test_path': TEST_PREDS_DIR / 'exp_b_lgbm_combinations_te_test_avg.npy',
        'weight': 1.0
    },
    {
        'name': 'exp_c_lgbm_normalized_target',
        'oof_path': OOF_PREDS_DIR / 'exp_c_lgbm_normalized_target_oof.npy',
        'test_path': TEST_PREDS_DIR / 'exp_c_lgbm_normalized_target_test_avg.npy',
        'weight': 1.0
    }
]

# Output Paths
OOF_PRED_OUTPUT_PATH = OOF_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_oof.npy"
TEST_PRED_OUTPUT_PATH = TEST_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_test_avg.npy"
SUBMISSION_OUTPUT_PATH = TEST_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_submission.csv"

# --- Main Function ---
def main():
    logger = setup_logging(log_file=str(COMPETITION_DIR / "training.log"))
    seed_everything(SEED)
    logger.info(f"--- Starting Experiment: {EXPERIMENT_NAME} ---")

    try:
        # Load the original train data to get the target for evaluation
        logger.info(f"Loading training data from: {TRAIN_CSV}")
        df_train = load_csv(str(TRAIN_CSV))
        
        # Load the test IDs for creating the submission file
        logger.info(f"Loading test data from: {TEST_CSV}")
        df_test = load_csv(str(TEST_CSV))
        test_ids = df_test['id']
        
        # Load the OOF predictions from each experiment
        logger.info("Loading OOF predictions from previous experiments...")
        oof_preds_list = []
        for exp in EXPERIMENT_DETAILS:
            try:
                logger.info(f"Loading OOF predictions from {exp['name']}...")
                oof_preds = np.load(exp['oof_path'])
                oof_preds_list.append(oof_preds)
                logger.info(f"Loaded OOF predictions from {exp['name']} with shape {oof_preds.shape}")
            except Exception as e:
                logger.warning(f"Could not load OOF predictions from {exp['name']}: {e}")
        
        # Load the test predictions from each experiment
        logger.info("Loading test predictions from previous experiments...")
        test_preds_list = []
        for exp in EXPERIMENT_DETAILS:
            try:
                logger.info(f"Loading test predictions from {exp['name']}...")
                test_preds = np.load(exp['test_path'])
                test_preds_list.append(test_preds)
                logger.info(f"Loaded test predictions from {exp['name']} with shape {test_preds.shape}")
            except Exception as e:
                logger.warning(f"Could not load test predictions from {exp['name']}: {e}")
        
        # Check if we have predictions to average
        if len(oof_preds_list) < 2:
            logger.warning(f"Not enough OOF predictions to create an ensemble! Found {len(oof_preds_list)}")
            if len(oof_preds_list) == 1:
                logger.info("Using the single available OOF prediction set")
                oof_ensemble_preds = oof_preds_list[0]
            else:
                logger.error("No valid OOF predictions found. Cannot continue.")
                sys.exit(1)
        else:
            # Create the OOF ensemble by averaging
            logger.info(f"Creating OOF ensemble from {len(oof_preds_list)} sets of predictions...")
            oof_ensemble_preds = np.zeros_like(oof_preds_list[0])
            for i, preds in enumerate(oof_preds_list):
                logger.info(f"Adding {EXPERIMENT_DETAILS[i]['name']} predictions to ensemble...")
                oof_ensemble_preds += preds * EXPERIMENT_DETAILS[i]['weight']
            
            # Normalize by sum of weights
            sum_weights = sum(EXPERIMENT_DETAILS[i]['weight'] for i in range(len(oof_preds_list)))
            oof_ensemble_preds /= sum_weights
            logger.info(f"Created OOF ensemble predictions with shape {oof_ensemble_preds.shape}")
        
        # Check if we have test predictions to average
        if len(test_preds_list) < 2:
            logger.warning(f"Not enough test predictions to create an ensemble! Found {len(test_preds_list)}")
            if len(test_preds_list) == 1:
                logger.info("Using the single available test prediction set")
                test_ensemble_preds = test_preds_list[0]
            else:
                logger.error("No valid test predictions found. Cannot continue.")
                sys.exit(1)
        else:
            # Create the test ensemble by averaging
            logger.info(f"Creating test ensemble from {len(test_preds_list)} sets of predictions...")
            test_ensemble_preds = np.zeros_like(test_preds_list[0])
            for i, preds in enumerate(test_preds_list):
                logger.info(f"Adding {EXPERIMENT_DETAILS[i]['name']} predictions to ensemble...")
                test_ensemble_preds += preds * EXPERIMENT_DETAILS[i]['weight']
            
            # Normalize by sum of weights
            sum_weights = sum(exp['weight'] for exp in EXPERIMENT_DETAILS[:len(test_preds_list)])
            test_ensemble_preds /= sum_weights
            logger.info(f"Created test ensemble predictions with shape {test_ensemble_preds.shape}")
        
        # Evaluate the ensemble performance
        logger.info("Evaluating ensemble performance...")
        # Load the actual target values from the full train set
        y_true = df_train[TARGET_COL].to_numpy()
        
        # Make sure the lengths match
        if len(y_true) != len(oof_ensemble_preds):
            logger.warning(f"Target length ({len(y_true)}) doesn't match OOF predictions length ({len(oof_ensemble_preds)})")
        
        # Calculate RMSLE on the ensemble predictions
        ensemble_cv_score = calculate_metric(y_true=y_true, y_pred=oof_ensemble_preds, metric_name='rmsle')
        logger.info(f"Ensemble OOF Score (RMSLE): {ensemble_cv_score:.5f}")
        
        # Save the ensemble predictions
        logger.info("Saving ensemble predictions...")
        np.save(OOF_PRED_OUTPUT_PATH, oof_ensemble_preds)
        logger.info(f"Ensemble OOF predictions saved to: {OOF_PRED_OUTPUT_PATH}")
        np.save(TEST_PRED_OUTPUT_PATH, test_ensemble_preds)
        logger.info(f"Ensemble test predictions saved to: {TEST_PRED_OUTPUT_PATH}")
        
        # Generate submission file
        logger.info("Generating submission file...")
        generate_submission_file(
            ids=test_ids.to_pandas(),
            predictions=test_ensemble_preds,
            id_col_name='id',
            target_col_name=TARGET_COL,
            file_path=str(SUBMISSION_OUTPUT_PATH)
        )
        logger.info(f"Submission file saved to: {SUBMISSION_OUTPUT_PATH}")
        
        # Update the experiments tracker
        try:
            logger.info(f"Updating experiments tracker: {EXPERIMENTS_CSV}")
            if not EXPERIMENTS_CSV.exists():
                logger.info(f"Experiments file not found at {EXPERIMENTS_CSV}. Creating it.")
                headers = ['script_file_name', 'why', 'what', 'how', 'cv_score', 'lb_score', 'status']
                pd.DataFrame(columns=headers).to_csv(EXPERIMENTS_CSV, index=False, sep=';')
            
            experiments_df = pd.read_csv(EXPERIMENTS_CSV, sep=';')
            exp_mask = experiments_df['script_file_name'] == EXPERIMENT_NAME
            
            if exp_mask.sum() > 0:
                # Update existing row
                idx_to_update = experiments_df[exp_mask].index[0]
                experiments_df.loc[idx_to_update, 'cv_score'] = ensemble_cv_score
                experiments_df.loc[idx_to_update, 'status'] = 'done'
                experiments_df.loc[idx_to_update, 'why'] = 'Ensemble only the predictions from exp_b and exp_c (both using 100% data).'
                experiments_df.loc[idx_to_update, 'what'] = 'Simple average ensemble of two full-data LGBM approaches.'
                experiments_df.loc[idx_to_update, 'how'] = 'Average OOF/test predictions from exp_b (100%, log1p) and exp_c (100%, normalized target).'
                logger.info(f"Updated existing experiment '{EXPERIMENT_NAME}' with CV score: {ensemble_cv_score:.5f}")
            else:
                # Add as new row
                new_row = pd.DataFrame([{
                    'script_file_name': EXPERIMENT_NAME,
                    'why': 'Ensemble only the predictions from exp_b and exp_c (both using 100% data).',
                    'what': 'Simple average ensemble of two full-data LGBM approaches.',
                    'how': 'Average OOF/test predictions from exp_b (100%, log1p) and exp_c (100%, normalized target).',
                    'cv_score': ensemble_cv_score,
                    'lb_score': None,
                    'status': 'done'
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
    
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info(f"--- Experiment Finished: {EXPERIMENT_NAME} ---")

if __name__ == "__main__":
    main() 