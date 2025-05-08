# Script for Experiment F: LightGBM with Enhanced Interaction Features
# MODIFIED for Playground Series S5E5: Predict Calorie Expenditure

"""
Extended Description:
This script builds upon Experiment B but adds enhanced interaction features:
- Multiplication (feature1 * feature2)
- Division (feature1 / feature2) 
- Power (feature1 ^ feature2)

It includes:
- Loading train and test data
- Preprocessing steps (outlier clipping, ordinal encoding)
- Feature engineering (polynomials, aggregations, N-way interactions)
- Enhanced interaction features (multiplication, division, power)
- Out-of-fold Target Encoding using the `apply_target_encoding` utility
- Training a LightGBM model using KFold cross-validation (7 folds)
- Calculating OOF predictions and averaging test predictions
- Clipping final predictions and saving results
- Updating the experiments tracker

Corresponds to experiment 'exp_f_lgbm_enhanced_interactions.py' in experiments.csv.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
import joblib
from sklearn.model_selection import KFold

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
    from utils.handle_outliers import clip_numerical_features
    from utils.encode_categorical_features import encode_categorical_features
    from utils.create_polynomial_features import create_polynomial_features
    from utils.create_aggregation_features import create_aggregation_features
    from utils.create_nway_interaction_features import create_nway_interaction_features
    from utils.apply_target_encoding import apply_target_encoding
    # Import the enhanced interaction features utility
    from utils.create_enhanced_interaction_features import create_enhanced_interaction_features
except ImportError as e:
    print(f"Error importing utility functions: {e}")
    print("Ensure utils directory is at the project root and contains all required files.")
    print("Consider running this script from the project root or as a module.")
    sys.exit(1)

# --- Configuration ---
EXPERIMENT_NAME = "exp_f_lgbm_enhanced_interactions.py"
N_FOLDS = 7
SEED = 42
TARGET_COL = "Calories"

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
COMPETITION_DIR = SCRIPT_DIR.parent
RAW_DATA_DIR = COMPETITION_DIR / "data" / "raw"
PROCESSED_DATA_DIR = COMPETITION_DIR / "data" / "processed"
MODEL_OUTPUT_DIR = COMPETITION_DIR / "models"
OOF_PREDS_DIR = COMPETITION_DIR / "predictions" / "oof"
TEST_PREDS_DIR = COMPETITION_DIR / "predictions" / "test"
EXPERIMENTS_CSV = COMPETITION_DIR / "experiments.csv"

# Input Data Paths
TRAIN_CSV = RAW_DATA_DIR / "train.csv"
TEST_CSV = RAW_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = RAW_DATA_DIR / "sample_submission.csv"

# Output Paths
OOF_PRED_OUTPUT_PATH = OOF_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_oof.npy"
TEST_PRED_OUTPUT_PATH = TEST_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_test_avg.npy"
SUBMISSION_OUTPUT_PATH = TEST_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_submission.csv"

# Create output directories
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OOF_PREDS_DIR.mkdir(parents=True, exist_ok=True)
TEST_PREDS_DIR.mkdir(parents=True, exist_ok=True)

# --- Feature Engineering & Preprocessing Params ---
# Outlier Clipping
OUTLIER_BOUNDS = {
    # No specific bounds initially defined - can be added based on EDA
}

# Columns for Ordinal Encoding (+1)
ORDINAL_ENCODE_COLS = ['Sex']

# Polynomial Features
POLY_FEATURES_CONFIG = {
    'Duration': {'sqrt': True, 'squared': True}
}

# Aggregation Features
AGG_FEATURES_CONFIG = [
    {'groupby_cols': ['Sex'], 'agg_col': 'Duration', 'agg_func': 'mean', 'new_col_name': 'Sex_Duration_mean'},
    {'groupby_cols': ['Sex'], 'agg_col': 'Heart_Rate', 'agg_func': 'mean', 'new_col_name': 'Sex_Heart_Rate_mean'}
]

# N-way Interaction Features
NWAY_INTERACTION_BASE_COLS = [
    'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Sex'
]
NWAY_ORDERS = [2, 3]

# Enhanced Interaction Features - NEW!
ENHANCED_INTERACTION_CONFIG = [
    # Multiplication
    {'features': ('Age', 'Weight'), 'operation': 'multiply'},
    {'features': ('Height', 'Weight'), 'operation': 'multiply'},
    {'features': ('Duration', 'Heart_Rate'), 'operation': 'multiply'},
    {'features': ('Weight', 'Duration'), 'operation': 'multiply'},
    {'features': ('Age', 'Heart_Rate'), 'operation': 'multiply'},  # NEW: Age x Heart Rate
    {'features': ('Body_Temp', 'Heart_Rate'), 'operation': 'multiply'},  # NEW: Body Temp x Heart Rate
    
    # Division - with fill value for division by zero
    {'features': ('Weight', 'Height'), 'operation': 'divide', 'fill_value': 0.0},
    {'features': ('Heart_Rate', 'Duration'), 'operation': 'divide', 'fill_value': 0.0},
    {'features': ('Calories', 'Duration'), 'operation': 'divide', 'fill_value': 0.0}, # Calories per minute
    {'features': ('Weight', 'Age'), 'operation': 'divide', 'fill_value': 0.0},
    {'features': ('Heart_Rate', 'Age'), 'operation': 'divide', 'fill_value': 0.0},  # NEW: Heart Rate / Age
    {'features': ('Calories', 'Weight'), 'operation': 'divide', 'fill_value': 0.0},  # NEW: Calories per kg
    
    # Power operations
    {'features': ('Weight', 'Height'), 'operation': 'power', 'fill_value': 0.0},
    {'features': ('Duration', 'Age'), 'operation': 'power', 'fill_value': 0.0},
]

# --- LightGBM Parameters ---
LGBM_PARAMS = {
    'objective': 'regression_l1',
    'metric': 'rmse',
    'learning_rate': 0.03,
    'lambda_l1': 5.0,
    'lambda_l2': 1.0,
    'num_leaves': 255,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.85,
    'bagging_freq': 1,
    'min_child_samples': 50,
    'seed': SEED,
    'boosting_type': 'gbdt',
    'n_jobs': -1,
    'verbose': -1,
}

# Fit Parameters
LGBM_FIT_PARAMS = {
    'num_boost_round': 2000,
    'callbacks': [
        lgb.early_stopping(30, verbose=False),
        lgb.log_evaluation(period=50)
    ]
}

# --- Main Function ---
def main():
    logger = setup_logging(log_file=str(COMPETITION_DIR / "training.log"))
    seed_everything(SEED)
    logger.info(f"--- Starting Experiment: {EXPERIMENT_NAME} ---")

    # --- Load Data ---
    try:
        logger.info(f"Loading training data from: {TRAIN_CSV}")
        df_train = load_csv(str(TRAIN_CSV))
        logger.info(f"Using full training data. Train shape: {df_train.shape}")

        logger.info(f"Loading test data from: {TEST_CSV}")
        df_test = load_csv(str(TEST_CSV))
        logger.info(f"Original shapes: Train={df_train.shape}, Test={df_test.shape}")

        # Store test IDs
        test_ids = df_test['id']
        train_len = len(df_train)
        test_len = len(df_test)

        # Define approximate column types for consistency before concat
        numeric_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
        categorical_cols = ['Sex']

        # Apply casting
        cast_exprs_numeric = [pl.col(c).cast(pl.Float64, strict=False) for c in numeric_cols if c in df_train.columns]
        cast_exprs_categorical = [pl.col(c).cast(pl.Utf8, strict=False) for c in categorical_cols if c in df_train.columns]
        df_train = df_train.with_columns(cast_exprs_numeric + cast_exprs_categorical)

        cast_exprs_numeric = [pl.col(c).cast(pl.Float64, strict=False) for c in numeric_cols if c in df_test.columns]
        cast_exprs_categorical = [pl.col(c).cast(pl.Utf8, strict=False) for c in categorical_cols if c in df_test.columns]
        df_test = df_test.with_columns(cast_exprs_numeric + cast_exprs_categorical)

        # Add missing target column to test set for concatenation
        if TARGET_COL not in df_test.columns:
            df_test = df_test.with_columns(pl.lit(None, dtype=pl.Float64).alias(TARGET_COL))

        # Ensure target is Float64 in train_df if it exists
        if TARGET_COL in df_train.columns:
            df_train = df_train.with_columns(pl.col(TARGET_COL).cast(pl.Float64, strict=False))

        # --- Concatenate Data ---
        df = pl.concat([
            df_train.drop('id'),
            df_test.drop('id')
        ], how='vertical')
        logger.info(f"Concatenated shape: {df.shape}")

        # --- Deduplication ---
        initial_rows = len(df)
        logger.info(f"Shape after unique: {df.shape} (Removed {initial_rows - len(df)} duplicate rows)")

    except FileNotFoundError as e:
        logger.error(f"Error loading data file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading: {e}")
        sys.exit(1)

    # --- Preprocessing ---
    logger.info("Starting preprocessing...")
    try:
        # Outlier Clipping (Numerical)
        df = clip_numerical_features(df, OUTLIER_BOUNDS)
        logger.info("Applied numerical clipping.")

        # Ordinal Encoding for categorical features
        object_cols_to_encode = df.select(pl.col(pl.Utf8)).columns
        ordinal_features_to_encode = [col for col in ORDINAL_ENCODE_COLS if col in object_cols_to_encode]
        if ordinal_features_to_encode:
             df, _, _ = encode_categorical_features(
                 train_df=df,
                 features=ordinal_features_to_encode,
                 test_df=None,
                 encoder_type='ordinal',
                 add_one=True,
                 drop_original=False,
                 handle_unknown='use_encoded_value',
                 unknown_value=-1
             )
             logger.info(f"Applied Ordinal Encoding (+1) to: {ordinal_features_to_encode}")
        else:
             logger.info("No remaining object columns found for ordinal encoding.")

        # Fill NaNs in numeric cols needed for FE
        if 'Duration' in df.columns and df['Duration'].null_count() > 0:
            median_val = df['Duration'].median()
            df = df.with_columns(pl.col('Duration').fill_null(median_val))
            logger.info(f"Filled NaNs in 'Duration' with median: {median_val}")
        else:
            logger.info("Checked 'Duration': No NaNs found or column not present, skipping fill_null.")

        # Create power features
        if 'Duration' in df.columns:
            df = df.with_columns([
                pl.col('Duration').sqrt().alias('Duration_sqrt'),
                (pl.col('Duration') ** 2).alias('Duration_squared')
            ])
            logger.info("Created sqrt and squared features for Duration.")
        else:
            logger.warning("Column 'Duration' not found, skipping creation of sqrt and squared features for it.")
            
        # Calculate BMI - NEW!
        if 'Weight' in df.columns and 'Height' in df.columns:
            # BMI = weight(kg) / height(m)^2
            # Height is in cm, so divide by 100 to get meters
            df = df.with_columns([
                (pl.col('Weight') / ((pl.col('Height') / 100) ** 2)).alias('BMI')
            ])
            logger.info("Created BMI feature.")
        else:
            logger.warning("Weight or Height columns not found, skipping BMI calculation.")

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}", exc_info=True)
        sys.exit(1)

    # --- Feature Engineering ---
    logger.info("Starting feature engineering...")
    try:
        # Aggregation Features
        df = create_aggregation_features(df, AGG_FEATURES_CONFIG)
        logger.info("Created aggregation features.")

        # N-way Interaction Features
        existing_nway_base_cols = [c for c in NWAY_INTERACTION_BASE_COLS if c in df.columns]
        logger.info(f"Creating N-way interactions for orders {NWAY_ORDERS} using base features: {existing_nway_base_cols}")
        df_interactions = create_nway_interaction_features(
            df=df.select(existing_nway_base_cols + [TARGET_COL]),
            features=existing_nway_base_cols,
            n_ways=NWAY_ORDERS,
            encode_type='label',
            add_one=True,
            handle_nulls='ignore'
        )
        nway_feature_names = df_interactions.columns
        logger.info(f"Created {len(nway_feature_names)} N-way interaction features.")
        df = pl.concat([df, df_interactions], how="horizontal")
        logger.info(f"DataFrame shape after N-way features: {df.shape}")

        # NEW: Enhanced Interaction Features
        logger.info("Creating enhanced interaction features (multiply, divide, power)...")
        df_enhanced_interactions = create_enhanced_interaction_features(
            df=df,
            feature_pairs=ENHANCED_INTERACTION_CONFIG
        )
        enhanced_feature_names = df_enhanced_interactions.columns
        logger.info(f"Created {len(enhanced_feature_names)} enhanced interaction features.")
        df = pl.concat([df, df_enhanced_interactions], how="horizontal")
        logger.info(f"DataFrame shape after enhanced interaction features: {df.shape}")

        # Final Type Casting
        df = df.cast({col: pl.Float32 for col in df.columns if df[col].dtype.is_numeric() and col != TARGET_COL})
        logger.info("Casted numeric features to Float32.")

    except Exception as e:
        logger.error(f"Error during feature engineering: {e}", exc_info=True)
        sys.exit(1)

    # --- Prepare for Target Encoding and CV ---
    logger.info("Preparing for Target Encoding and Cross-Validation...")
    try:
        # Split back into Train and Test
        df_train_processed = df[:train_len]
        df_test_processed = df[train_len:]

        # Separate target
        target_series_original = df_train_processed[TARGET_COL]
        # Transform target for training
        target_series_log = df_train_processed.with_columns(
            pl.col(TARGET_COL).log1p()
        )[TARGET_COL]
        df_train_processed = df_train_processed.drop(TARGET_COL)
        df_test_processed = df_test_processed.drop(TARGET_COL)

        # Ensure original target wasn't accidentally dropped or is all null
        if target_series_original.null_count() == len(target_series_original):
            raise ValueError(f"Original Target column '{TARGET_COL}' is all null after preprocessing.")
        # Ensure log target doesn't have issues
        if target_series_log.is_infinite().any() or target_series_log.is_nan().any():
            logger.warning("Log-transformed target contains NaN or Inf values after transformation. Check input data.")

        # Identify all features for potential target encoding
        features_for_te = list(df_train_processed.columns)
        logger.info(f"Preparing to target encode {len(features_for_te)} features.")

        # Generate CV indices for Target Encoding
        kf_te = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        cv_indices_for_te = list(kf_te.split(np.zeros(train_len)))
        logger.info(f"Generated {len(cv_indices_for_te)} CV fold indices for Target Encoding.")

        # --- Apply Target Encoding ---
        logger.info("Applying Target Encoding...")
        # Ensure target column is present in the df passed to apply_target_encoding
        train_df_with_target_temp = df_train_processed.with_columns(target_series_log.alias(TARGET_COL))

        train_df_te, test_df_te = apply_target_encoding(
            train_df=train_df_with_target_temp,
            features=features_for_te,
            target_col=TARGET_COL,
            cv_indices=cv_indices_for_te,
            test_df=df_test_processed,
            agg_stat='mean',
            use_smoothing=True,
            smoothing=10.0,
            new_col_suffix='_te'
        )
        logger.info("Target Encoding applied.")

        # Remove the temporary target column from train_df_te
        train_df_te = train_df_te.drop(TARGET_COL)

        # Define final features - all columns in the TE dataframe
        FINAL_FEATURES = train_df_te.columns
        logger.info(f"Final number of features after TE: {len(FINAL_FEATURES)}")

        # Convert Polars DFs to Pandas for LightGBM
        logger.info("Converting DataFrames to Pandas for LightGBM...")
        train_pd_te = train_df_te.to_pandas()
        test_pd_te = test_df_te.to_pandas()
        target_pd_log_full = target_series_log.to_pandas()
        target_pd_original_full = target_series_original.to_pandas()
        logger.info("Conversion to Pandas complete.")

    except Exception as e:
        logger.error(f"Error during TE or CV setup: {e}", exc_info=True)
        sys.exit(1)

    # --- Cross-Validation and Training ---
    # Handle NaN in target before CV loop
    nan_target_mask = target_pd_log_full.isna()
    if nan_target_mask.any():
        logger.warning(f"Found {nan_target_mask.sum()} NaN values in log-transformed target variable. Dropping corresponding rows from training data and original target.")
        train_pd_clean = train_pd_te[~nan_target_mask].reset_index(drop=True)
        target_pd_log_clean = target_pd_log_full[~nan_target_mask].reset_index(drop=True)
        target_pd_original_clean = target_pd_original_full[~nan_target_mask].reset_index(drop=True)
        logger.info(f"Cleaned training data shape after dropping NaN targets: {train_pd_clean.shape}")
    else:
        train_pd_clean = train_pd_te
        target_pd_log_clean = target_pd_log_full
        target_pd_original_clean = target_pd_original_full

    # Generate CV indices *after* potentially dropping rows
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_indices_for_training = list(kf.split(train_pd_clean))
    logger.info(f"Generated {len(cv_indices_for_training)} CV fold indices for training data shape {train_pd_clean.shape}.")

    logger.info(f"Starting {N_FOLDS}-Fold Cross-Validation with LightGBM...")
    oof_preds_original_scale = np.zeros(len(train_pd_clean))
    test_preds_sum_original_scale = np.zeros(len(test_pd_te))
    models = []

    try:
        for fold, (train_idx, valid_idx) in enumerate(cv_indices_for_training):
            logger.info(f"--- Fold {fold + 1}/{N_FOLDS} ---")

            # Slice data for the fold
            X_train_fold = train_pd_clean.iloc[train_idx]
            y_train_fold = target_pd_log_clean.iloc[train_idx]
            X_valid_fold = train_pd_clean.iloc[valid_idx]
            y_valid_fold = target_pd_log_clean.iloc[valid_idx]

            logger.info(f"Training LightGBM fold {fold + 1}...")
            # Create LightGBM datasets
            lgb_train = lgb.Dataset(X_train_fold, label=y_train_fold, feature_name=FINAL_FEATURES)
            lgb_valid = lgb.Dataset(X_valid_fold, label=y_valid_fold, feature_name=FINAL_FEATURES, reference=lgb_train)

            # Train the model (objective/metric = l1/rmse)
            model = lgb.train(
                params=LGBM_PARAMS,
                train_set=lgb_train,
                valid_sets=[lgb_train, lgb_valid],
                valid_names=['train', 'validation'],
                **LGBM_FIT_PARAMS
            )

            # Store model if desired
            models.append(model)

            # Predict (log scale initially)
            fold_oof_preds_log = model.predict(X_valid_fold, num_iteration=model.best_iteration)
            fold_test_preds_log = model.predict(test_pd_te, num_iteration=model.best_iteration)

            # Transform predictions back to original scale
            fold_oof_preds_orig = np.expm1(fold_oof_preds_log)
            fold_test_preds_orig = np.expm1(fold_test_preds_log)

            # Store original scale predictions
            oof_preds_original_scale[valid_idx] = fold_oof_preds_orig
            test_preds_sum_original_scale += fold_test_preds_orig
            logger.info(f"Fold {fold + 1} prediction complete. Best Iteration: {model.best_iteration}")
            logger.info("-" * 50)

        # --- Post-Processing and Saving ---
        logger.info("Cross-validation finished. Processing results...")
        final_test_preds_original_scale = test_preds_sum_original_scale / N_FOLDS

        # Clip final predictions
        final_test_preds_original_scale = np.maximum(1.0, final_test_preds_original_scale)
        oof_preds_original_scale = np.maximum(1.0, oof_preds_original_scale)
        logger.info(f"Final predictions clipped (min 1.0). Max OOF: {oof_preds_original_scale.max():.2f}, Max Test: {final_test_preds_original_scale.max():.2f}")

        # Calculate overall CV score
        y_true_for_metric = np.array(target_pd_original_clean.copy())
        overall_cv_score = calculate_metric(y_true=y_true_for_metric, y_pred=oof_preds_original_scale, metric_name='rmsle')
        logger.info(f"Overall CV Score (RMSLE on original scale): {overall_cv_score:.5f}")

        # Save predictions
        logger.info("Saving OOF and Test predictions...")
        np.save(OOF_PRED_OUTPUT_PATH, oof_preds_original_scale)
        logger.info(f"OOF predictions saved to: {OOF_PRED_OUTPUT_PATH}")
        np.save(TEST_PRED_OUTPUT_PATH, final_test_preds_original_scale)
        logger.info(f"Averaged test predictions saved to: {TEST_PRED_OUTPUT_PATH}")

        # Generate submission file
        logger.info("Generating submission file...")
        generate_submission_file(
            ids=test_ids.to_pandas(),
            predictions=final_test_preds_original_scale,
            id_col_name='id',
            target_col_name=TARGET_COL,
            file_path=str(SUBMISSION_OUTPUT_PATH)
        )
        logger.info(f"Submission file saved to: {SUBMISSION_OUTPUT_PATH}")

    except Exception as e:
        logger.error(f"An error occurred during CV training or saving: {e}", exc_info=True)
        sys.exit(1)

    # --- Update Experiments Tracker ---
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
            experiments_df.loc[idx_to_update, 'cv_score'] = overall_cv_score
            experiments_df.loc[idx_to_update, 'status'] = 'done'
            experiments_df.loc[idx_to_update, 'why'] = 'Add enhanced interaction features (multiply, divide, power) to exp_b.'
            experiments_df.loc[idx_to_update, 'what'] = 'Train LGBM with additional feature engineering using enhanced interactions.'
            experiments_df.loc[idx_to_update, 'how'] = 'Load 100% data, add division and power operations to feature set, log1p target, KFold(7) LGBM, expm1 preds, final CV RMSLE.'
            logger.info(f"Updated existing experiment '{EXPERIMENT_NAME}' with CV score: {overall_cv_score:.5f}")
        else:
             # Add as new row
             new_row = pd.DataFrame([{
                 'script_file_name': EXPERIMENT_NAME,
                 'why': 'Add enhanced interaction features (multiply, divide, power) to exp_b.',
                 'what': 'Train LGBM with additional feature engineering using enhanced interactions.',
                 'how': 'Load 100% data, add division and power operations to feature set, log1p target, KFold(7) LGBM, expm1 preds, final CV RMSLE.',
                 'cv_score': overall_cv_score,
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

    logger.info(f"--- Experiment Finished: {EXPERIMENT_NAME} ---")

if __name__ == "__main__":
    main() 