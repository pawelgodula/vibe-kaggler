# Script for Experiment J: LightGBM (L2 Objective) with Combination Features and COUNT Encoding

"""
Extended Description:
This script builds on Experiment I by replacing the Target Encoding step with
Count Encoding (Frequency Encoding) using the `apply_count_encoding` utility.
All other aspects (data loading, preprocessing, feature engineering, model, objective,
CV strategy) remain the same as Experiment I.

Corresponds to experiment 'exp_j_lgbm_l2_combinations_ce.py' in experiments.csv.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd # For experiments.csv handling
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
    from utils.apply_count_encoding import apply_count_encoding
except ImportError as e:
    print(f"Error importing utility functions: {e}")
    print("Ensure utils directory is at the project root and contains all required files.")
    print("Consider running this script from the project root or as a module.")
    sys.exit(1)

# --- Configuration ---
EXPERIMENT_NAME = "exp_j_lgbm_l2_combinations_ce.py"
N_FOLDS = 7
SEED = 42
TARGET_COL = "Listening_Time_minutes"

# Paths (copied from exp_i, EXPERIMENT_NAME substitution handles output paths)
SCRIPT_DIR = Path(__file__).resolve().parent
COMPETITION_DIR = SCRIPT_DIR.parent
RAW_DATA_DIR = COMPETITION_DIR / "data" / "raw"
PROCESSED_DATA_DIR = COMPETITION_DIR / "data" / "processed"
EXTERNAL_DATA_DIR = COMPETITION_DIR / "data" / "external"
MODEL_OUTPUT_DIR = COMPETITION_DIR / "models"
OOF_PREDS_DIR = COMPETITION_DIR / "predictions" / "oof"
TEST_PREDS_DIR = COMPETITION_DIR / "predictions" / "test"
EXPERIMENTS_CSV = COMPETITION_DIR / "experiments.csv"
TRAIN_CSV = RAW_DATA_DIR / "train.csv"
TEST_CSV = RAW_DATA_DIR / "test.csv"
EXTERNAL_CSV = RAW_DATA_DIR / "podcast_dataset.csv"
SAMPLE_SUBMISSION_CSV = RAW_DATA_DIR / "sample_submission.csv"
OOF_PRED_OUTPUT_PATH = OOF_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_oof.npy"
TEST_PRED_OUTPUT_PATH = TEST_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_test_avg.npy"
SUBMISSION_OUTPUT_PATH = TEST_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_submission.csv"

MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OOF_PREDS_DIR.mkdir(parents=True, exist_ok=True)
TEST_PREDS_DIR.mkdir(parents=True, exist_ok=True)

# --- Feature Engineering & Preprocessing Params (copied from exp_i) ---
OUTLIER_BOUNDS = {
    'Episode_Length_minutes': (0, 120),
    'Host_Popularity_percentage': (20, 100),
    'Guest_Popularity_percentage': (0, 100),
}
ADS_CLIP_THRESHOLD = 3
ADS_CLIP_VALUE = 0
DAY_MAPPING = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
TIME_MAPPING = {'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4}
SENTIMENT_MAP = {'Negative': 1, 'Neutral': 2, 'Positive': 3}
ORDINAL_ENCODE_COLS = ['Podcast_Name', 'Genre']
POLY_FEATURES_CONFIG = {'Episode_Length_minutes': {'sqrt': True, 'squared': True}}
AGG_FEATURES_CONFIG = [
    {'groupby_cols': ['Episode_Sentiment'], 'agg_col': 'Episode_Length_minutes', 'agg_func': 'mean', 'new_col_name': 'Episode_Sentiment_EP'},
    {'groupby_cols': ['Genre'], 'agg_col': 'Episode_Length_minutes', 'agg_func': 'mean', 'new_col_name': 'Genre_EP'},
    {'groupby_cols': ['Publication_Day'], 'agg_col': 'Episode_Length_minutes', 'agg_func': 'mean', 'new_col_name': 'Publication_Day_EP'},
    {'groupby_cols': ['Podcast_Name'], 'agg_col': 'Episode_Length_minutes', 'agg_func': 'mean', 'new_col_name': 'Podcast_Name_EP'},
    {'groupby_cols': ['Episode_Title'], 'agg_col': 'Episode_Length_minutes', 'agg_func': 'mean', 'new_col_name': 'Episode_Title_EP'},
    {'groupby_cols': ['Guest_Popularity_percentage'], 'agg_col': 'Episode_Length_minutes', 'agg_func': 'mean', 'new_col_name': 'Guest_Popularity_percentage_EP'},
    {'groupby_cols': ['Host_Popularity_percentage'], 'agg_col': 'Episode_Length_minutes', 'agg_func': 'mean', 'new_col_name': 'Host_Popularity_percentage_EP'},
    {'groupby_cols': ['Number_of_Ads'], 'agg_col': 'Episode_Length_minutes', 'agg_func': 'mean', 'new_col_name': 'Number_of_Ads_EP'},
]
NWAY_INTERACTION_BASE_COLS = [
    'Episode_Length_minutes', 'Episode_Title', 'Publication_Time',
    'Host_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment',
    'Publication_Day', 'Podcast_Name', 'Genre', 'Guest_Popularity_percentage'
]
NWAY_ORDERS = [2, 3, 5, 7]

# --- Count Encoding Parameters ---
# Identify columns to apply Count Encoding (original categoricals + N-way interactions)
ORIGINAL_CATEGORICAL_BASE_COLS = [ # From preproc section
    'Publication_Day', 'Publication_Time', 'Episode_Sentiment',
    'Episode_Title', 'Podcast_Name', 'Genre', 'Number_of_Ads'
]
CE_COUNT_ON = 'train' # Calculate counts only on training data to prevent leakage
CE_NORMALIZE = False # Use raw counts
CE_HANDLE_UNKNOWN = 'zero' # Fill unknown categories in test with 0
CE_SUFFIX = '_ce'

# --- LightGBM Parameters (copied from exp_i) ---
LGBM_PARAMS = {
    'objective': 'regression_l2',
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
LGBM_FIT_PARAMS = {
    'num_boost_round': 10000,
    'callbacks': [lgb.early_stopping(30, verbose=False),
                  lgb.log_evaluation(period=500)]
}


# --- Main Function ---
def main():
    logger = setup_logging(log_file=str(COMPETITION_DIR / "training.log"))
    seed_everything(SEED)
    logger.info(f"--- Starting Experiment: {EXPERIMENT_NAME} ---")

    # --- Load Data (copied from exp_i) ---
    try:
        logger.info(f"Loading training data from: {TRAIN_CSV}")
        df_train = load_csv(str(TRAIN_CSV))
        logger.info(f"Loading test data from: {TEST_CSV}")
        df_test = load_csv(str(TEST_CSV))
        logger.info(f"Loading external data from: {EXTERNAL_CSV}")
        df_external = load_csv(str(EXTERNAL_CSV))
        logger.info(f"Original shapes: Train={df_train.shape}, Test={df_test.shape}, External={df_external.shape}")
        test_ids = df_test['id']
        train_len = len(df_train)
        test_len = len(df_test)
        numeric_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']
        categorical_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
        cast_exprs_numeric = [pl.col(c).cast(pl.Float64, strict=False) for c in numeric_cols if c in df_train.columns]
        cast_exprs_categorical = [pl.col(c).cast(pl.Utf8, strict=False) for c in categorical_cols if c in df_train.columns]
        df_train = df_train.with_columns(cast_exprs_numeric + cast_exprs_categorical)
        cast_exprs_numeric = [pl.col(c).cast(pl.Float64, strict=False) for c in numeric_cols if c in df_test.columns]
        cast_exprs_categorical = [pl.col(c).cast(pl.Utf8, strict=False) for c in categorical_cols if c in df_test.columns]
        df_test = df_test.with_columns(cast_exprs_numeric + cast_exprs_categorical)
        if TARGET_COL not in df_test.columns:
            df_test = df_test.with_columns(pl.lit(None, dtype=pl.Float64).alias(TARGET_COL))
        cast_exprs_numeric = [pl.col(c).cast(pl.Float64, strict=False) for c in numeric_cols if c in df_external.columns]
        cast_exprs_categorical = [pl.col(c).cast(pl.Utf8, strict=False) for c in categorical_cols if c in df_external.columns]
        df_external = df_external.with_columns(cast_exprs_numeric + cast_exprs_categorical)
        if TARGET_COL in df_train.columns:
             df_train = df_train.with_columns(pl.col(TARGET_COL).cast(pl.Float64, strict=False))
        df = pl.concat([
            df_train.drop('id'),
            df_external,
            df_test.drop('id')
        ], how='vertical')
        logger.info(f"Concatenated shape: {df.shape}")
        initial_rows = len(df)
        logger.info(f"Shape after unique: {df.shape} (Removed {initial_rows - len(df)} duplicate rows)")
    except FileNotFoundError as e:
        logger.error(f"Error loading data file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading: {e}")
        sys.exit(1)

    # --- Preprocessing (copied from exp_i) ---
    logger.info("Starting preprocessing...")
    try:
        df = clip_numerical_features(df, OUTLIER_BOUNDS)
        logger.info("Applied numerical clipping.")
        df = df.with_columns(
            pl.when(pl.col('Number_of_Ads') > ADS_CLIP_THRESHOLD)
            .then(pl.lit(ADS_CLIP_VALUE))
            .otherwise(pl.col('Number_of_Ads'))
            .alias('Number_of_Ads')
        )
        logger.info("Applied Number_of_Ads clipping.")
        df = df.with_columns([
            pl.col('Publication_Day').replace(DAY_MAPPING, default=None).cast(pl.Int8),
            pl.col('Publication_Time').replace(TIME_MAPPING, default=None).cast(pl.Int8),
            pl.col('Episode_Sentiment').replace(SENTIMENT_MAP, default=None).cast(pl.Int8)
        ])
        logger.info("Applied manual mappings.")
        df = df.with_columns(
            pl.col('Episode_Title').str.replace('Episode ', '').cast(pl.Int32)
        )
        logger.info("Processed Episode_Title.")
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
        if df['Episode_Length_minutes'].null_count() > 0:
            median_val = df['Episode_Length_minutes'].median()
            df = df.with_columns(pl.col('Episode_Length_minutes').fill_null(median_val))
            logger.info(f"Filled NaNs in 'Episode_Length_minutes' with median: {median_val}")
        df = df.with_columns([
            pl.col('Episode_Length_minutes').sqrt().alias('Episode_Length_minutes_sqrt'),
            (pl.col('Episode_Length_minutes') ** 2).alias('Episode_Length_minutes_squared')
        ])
        logger.info("Created sqrt and squared features for Episode_Length_minutes.")
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}", exc_info=True)
        sys.exit(1)

    # --- Feature Engineering (copied from exp_i) ---
    logger.info("Starting feature engineering...")
    try:
        df = create_aggregation_features(df, AGG_FEATURES_CONFIG)
        logger.info("Created aggregation features.")
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
        df = df.cast({col: pl.Float32 for col in df.columns if df[col].dtype.is_numeric() and col != TARGET_COL})
        logger.info("Casted numeric features to Float32.")
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}", exc_info=True)
        sys.exit(1)

    # --- Prepare for Count Encoding and CV ---
    logger.info("Preparing for Count Encoding and Cross-Validation...")
    try:
        # Split back into Train and Test
        df_train_processed = df[:train_len]
        df_test_processed = df[-test_len:]

        # Separate target
        target_series = df_train_processed[TARGET_COL]
        df_train_processed = df_train_processed.drop(TARGET_COL)
        df_test_processed = df_test_processed.drop(TARGET_COL)

        if target_series.null_count() == len(target_series):
            raise ValueError(f"Target column '{TARGET_COL}' is all null after preprocessing.")

        # Identify columns for Count Encoding
        n_way_encoded_cols = [col for col in df_train_processed.columns if '__' in col]
        features_for_ce_all = [c for c in ORIGINAL_CATEGORICAL_BASE_COLS if c in df_train_processed.columns] + n_way_encoded_cols
        features_for_ce = list(dict.fromkeys([c for c in features_for_ce_all if c in df_train_processed.columns])) # Ensure unique and exist
        logger.info(f"Identified {len(features_for_ce)} features for Count Encoding: {features_for_ce}")

        # --- Apply Count Encoding --- (Replaces Target Encoding block)
        logger.info("Applying Count Encoding...")
        train_df_ce, test_df_ce = apply_count_encoding(
            train_df=df_train_processed,
            features=features_for_ce,
            test_df=df_test_processed,
            count_on=CE_COUNT_ON,
            normalize=CE_NORMALIZE,
            handle_unknown=CE_HANDLE_UNKNOWN,
            new_col_suffix=CE_SUFFIX
        )
        logger.info("Count Encoding applied.")

        # Drop the original columns that were count encoded to avoid redundancy
        logger.info(f"Dropping original columns used for Count Encoding: {len(features_for_ce)} columns")
        train_df_encoded = train_df_ce.drop(features_for_ce)
        test_df_encoded = test_df_ce.drop(features_for_ce)

        # Define final features - all columns in the encoded dataframe
        FINAL_FEATURES = train_df_encoded.columns
        logger.info(f"Final number of features after CE and dropping originals: {len(FINAL_FEATURES)}")

        # Convert Polars DFs to Pandas for LightGBM
        logger.info("Converting DataFrames to Pandas for LightGBM...")
        train_pd_encoded = train_df_encoded.to_pandas()
        test_pd_encoded = test_df_encoded.to_pandas()
        target_pd_full = target_series.to_pandas()
        logger.info("Conversion to Pandas complete.")

    except Exception as e:
        logger.error(f"Error during Count Encoding or CV setup: {e}", exc_info=True)
        sys.exit(1)

    # --- Cross-Validation and Training ---

    # Handle NaN in target before CV loop
    nan_target_mask = target_pd_full.isna()
    if nan_target_mask.any():
        logger.warning(f"Found {nan_target_mask.sum()} NaN values in target variable. Dropping corresponding rows from training data.")
        train_pd_clean = train_pd_encoded[~nan_target_mask].reset_index(drop=True)
        target_pd_clean = target_pd_full[~nan_target_mask].reset_index(drop=True)
        logger.info(f"Cleaned training data shape after dropping NaN targets: {train_pd_clean.shape}")
    else:
        train_pd_clean = train_pd_encoded
        target_pd_clean = target_pd_full

    # Generate CV indices *after* potentially dropping rows
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_indices_for_training = list(kf.split(train_pd_clean))
    logger.info(f"Generated {len(cv_indices_for_training)} CV fold indices for training data shape {train_pd_clean.shape}.")

    logger.info(f"Starting {N_FOLDS}-Fold Cross-Validation with LightGBM...")
    oof_preds = np.zeros(len(train_pd_clean))
    test_preds_sum = np.zeros(len(test_pd_encoded))
    models = []

    try:
        for fold, (train_idx, valid_idx) in enumerate(cv_indices_for_training):
            logger.info(f"--- Fold {fold + 1}/{N_FOLDS} ---")
            X_train_fold = train_pd_clean.iloc[train_idx]
            y_train_fold = target_pd_clean.iloc[train_idx]
            X_valid_fold = train_pd_clean.iloc[valid_idx]
            y_valid_fold = target_pd_clean.iloc[valid_idx]

            logger.info(f"Training LightGBM fold {fold + 1}...")
            lgb_train = lgb.Dataset(X_train_fold, label=y_train_fold, feature_name=FINAL_FEATURES)
            lgb_valid = lgb.Dataset(X_valid_fold, label=y_valid_fold, feature_name=FINAL_FEATURES, reference=lgb_train)

            model = lgb.train(
                params=LGBM_PARAMS,
                train_set=lgb_train,
                valid_sets=[lgb_train, lgb_valid],
                valid_names=['train', 'validation'],
                **LGBM_FIT_PARAMS
            )
            models.append(model)
            fold_oof_preds = model.predict(X_valid_fold, num_iteration=model.best_iteration)
            fold_test_preds = model.predict(test_pd_encoded, num_iteration=model.best_iteration) # Use test_pd_encoded here

            oof_preds[valid_idx] = fold_oof_preds
            test_preds_sum += fold_test_preds
            logger.info(f"Fold {fold + 1} prediction complete. Best Iteration: {model.best_iteration}")
            logger.info("-" * 50)

        # --- Post-Processing and Saving ---
        logger.info("Cross-validation finished. Processing results...")
        final_test_preds = test_preds_sum / N_FOLDS
        final_test_preds = np.maximum(0, np.minimum(120, final_test_preds))
        oof_preds = np.maximum(0, np.minimum(120, oof_preds))
        logger.info("Final predictions clipped between 0 and 120.")
        y_true_cv = target_pd_clean.copy()
        overall_cv_score = calculate_metric(y_true=y_true_cv, y_pred=oof_preds, metric_name='rmse')
        logger.info(f"Overall CV Score (RMSE): {overall_cv_score:.5f}")
        logger.info("Saving OOF and Test predictions...")
        np.save(OOF_PRED_OUTPUT_PATH, oof_preds)
        logger.info(f"OOF predictions saved to: {OOF_PRED_OUTPUT_PATH}")
        np.save(TEST_PRED_OUTPUT_PATH, final_test_preds)
        logger.info(f"Averaged test predictions saved to: {TEST_PRED_OUTPUT_PATH}")
        logger.info("Generating submission file...")
        generate_submission_file(
            ids=test_ids.to_pandas(),
            predictions=final_test_preds,
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
        experiments_df = pd.read_csv(EXPERIMENTS_CSV, sep=';')
        exp_mask = experiments_df['script_file_name'] == EXPERIMENT_NAME
        if exp_mask.sum() > 0:
            logger.warning(f"Experiment '{EXPERIMENT_NAME}' already exists in tracker. Updating existing entry.")
            experiments_df.loc[exp_mask, 'cv_score'] = overall_cv_score
            experiments_df.loc[exp_mask, 'status'] = 'done'
            logger.info(f"Updated existing experiment '{EXPERIMENT_NAME}' with CV score: {overall_cv_score:.5f} and status: done")
        else:
             logger.info(f"Adding new experiment '{EXPERIMENT_NAME}' to tracker.")
             new_row = pd.DataFrame([{
                 'script_file_name': EXPERIMENT_NAME,
                 'why': 'Replace TE with CE in Exp I (LGBM L2)',
                 'what': 'Train LGBM (L2 objective) with extensive FE including N-way label encoded combinations and Count Encoding',
                 'how': 'Same as Exp I but apply_count_encoding used instead of apply_target_encoding.',
                 'cv_score': overall_cv_score,
                 'lb_score': None,
                 'status': 'done'
             }])
             for col in experiments_df.columns:
                  if col not in new_row.columns:
                      new_row[col] = None
             new_row = new_row[experiments_df.columns]
             experiments_df = pd.concat([experiments_df, new_row], ignore_index=True)
        experiments_df.to_csv(EXPERIMENTS_CSV, index=False, sep=';')
        logger.info("Experiments tracker updated successfully.")
    except Exception as e:
        logger.error(f"An error occurred while updating experiments tracker: {e}")

    logger.info(f"--- Experiment Finished: {EXPERIMENT_NAME} ---")

if __name__ == "__main__":
    main() 