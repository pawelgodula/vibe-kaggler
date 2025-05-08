# Script for Experiment L: LightGBM with Count Encoding, Target Normalization, and Ratio-to-Category Mean Encoding
# MODIFIED for Playground Series S5E5: Predict Calorie Expenditure

"""
Extended Description:
This script is based on Experiment K (LightGBM with Count Encoding and Target Normalization) 
with the addition of ratio-to-category mean encoding features:
1. Normalizes the target variable (Calories) by dividing by Duration, predicting 
   calories burned per minute instead of total calories
2. Applies count encoding using TEST data as the source for frequencies
3. Adds ratio-to-category mean encoding for numerical features relative to Sex category
4. After prediction, transforms predictions back to the original scale by multiplying by Duration

It includes:
- Loading train and test data
- Preprocessing steps (outlier clipping, ordinal encoding)
- Target normalization (calories per minute)
- Converting numerical features to categorical (string) representation
- Feature engineering (polynomials, aggregations, N-way interactions)
- Count Encoding using TEST data as the source for frequencies
- Ratio-to-Category Mean Encoding for numerical features (relative to Sex)
- Training a LightGBM model using KFold cross-validation (7 folds)
- Calculating OOF predictions and averaging test predictions
- Generating and saving feature importance visualizations
- Transforming predictions back to original scale (total calories)
- Clipping final predictions and saving results
- Updating the experiments tracker

Corresponds to experiment 'exp_l_lgbm_ratio_encoding.py' in experiments.csv.
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
    from utils.handle_outliers import clip_numerical_features
    from utils.encode_categorical_features import encode_categorical_features
    from utils.create_polynomial_features import create_polynomial_features
    from utils.create_aggregation_features import create_aggregation_features
    from utils.create_nway_interaction_features import create_nway_interaction_features
    from utils.apply_count_encoding import apply_count_encoding
    # Add import for ratio-to-category mean encoding
    from utils.apply_ratio_to_category_mean_encoding import apply_ratio_to_category_mean_encoding
except ImportError as e:
    print(f"Error importing utility functions: {e}")
    print("Ensure utils directory is at the project root and contains all required files.")
    print("Consider running this script from the project root or as a module.")
    sys.exit(1)

# --- Configuration ---
EXPERIMENT_NAME = "exp_l_lgbm_ratio_encoding.py"
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
REPORTS_DIR = COMPETITION_DIR / "reports"
EXPERIMENTS_CSV = COMPETITION_DIR / "experiments.csv"

# Input Data Paths
TRAIN_CSV = RAW_DATA_DIR / "train.csv"
TEST_CSV = RAW_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = RAW_DATA_DIR / "sample_submission.csv"

# Output Paths
OOF_PRED_OUTPUT_PATH = OOF_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_oof.npy"
TEST_PRED_OUTPUT_PATH = TEST_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_test_avg.npy"
SUBMISSION_OUTPUT_PATH = TEST_PREDS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_submission.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / f"{EXPERIMENT_NAME.replace('.py', '')}_feature_importance.png"

# Create output directories
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OOF_PREDS_DIR.mkdir(parents=True, exist_ok=True)
TEST_PREDS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Feature Engineering & Preprocessing Params ---
# Outlier Clipping
OUTLIER_BOUNDS = {
    # No specific bounds initially defined - can be added based on EDA
}

# Columns for Ordinal Encoding (+1)
ORDINAL_ENCODE_COLS = ['Sex']

# Numerical features to convert to categorical
NUMERIC_TO_CATEGORICAL = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

# Binning parameters for numeric-to-categorical conversion
# Will be used to bin continuous features into discrete categories
BINNING_PARAMS = {
    'Age': 10,        # Bin age by decade
    'Height': 5,      # 5cm bins
    'Weight': 5,      # 5kg bins
    'Duration': 1,    # 1-minute bins
    'Heart_Rate': 5,  # 5 bpm bins
    'Body_Temp': 0.2  # 0.2°C bins
}

# Polynomial Features
POLY_FEATURES_CONFIG = {
    'Duration': {'sqrt': True, 'squared': True}
}

# Aggregation Features
AGG_FEATURES_CONFIG = [
    {'groupby_cols': ['Sex'], 'agg_col': 'Duration', 'agg_func': 'mean', 'new_col_name': 'Sex_Duration_mean'},
    {'groupby_cols': ['Sex'], 'agg_col': 'Heart_Rate', 'agg_func': 'mean', 'new_col_name': 'Sex_Heart_Rate_mean'}
]

# N-way Interaction Features - Will use the string-converted features
NWAY_INTERACTION_BASE_COLS = [
    'Age_cat', 'Height_cat', 'Weight_cat', 'Duration_cat', 'Heart_Rate_cat', 'Body_Temp_cat', 'Sex'
]
NWAY_ORDERS = [2, 3]

# Ratio-to-Category Mean Encoding configuration
# Numerical features to encode as ratio to their category mean
RATIO_FEATURES_CONFIG = {
    'category_col': 'Sex',
    'numerical_features': ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Duration_sqrt', 'Duration_squared'],
    'calculate_mean_on': 'combined'  # Use both train and test data for more stable estimates
}

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

# Function to generate and save feature importance plot
def plot_feature_importance(models, feature_names, output_path, top_n=30):
    """
    Generate and save a feature importance plot for LightGBM models.
    
    Args:
        models (list): List of trained LightGBM models
        feature_names (list): List of feature names
        output_path (str or Path): Path to save the plot
        top_n (int): Number of top features to show
    """
    # Calculate average feature importance across all folds
    importance_df = pd.DataFrame()
    
    for i, model in enumerate(models):
        fold_importance = pd.DataFrame()
        fold_importance['feature'] = feature_names
        fold_importance['importance'] = model.feature_importance(importance_type='gain')
        fold_importance['fold'] = i + 1
        importance_df = pd.concat([importance_df, fold_importance], axis=0)
    
    # Group by feature and calculate stats
    feature_importance = importance_df.groupby('feature').importance.agg(['mean', 'std']).sort_values(by='mean', ascending=False)
    feature_importance = feature_importance.reset_index()
    
    # Get top N features
    if len(feature_importance) > top_n:
        top_features = feature_importance.head(top_n)
    else:
        top_features = feature_importance
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1.0)
    sns.set_style('whitegrid')
    
    # Plot mean importance
    sns.barplot(
        data=top_features,
        y='feature',
        x='mean',
        color='skyblue',
        alpha=0.8
    )
    
    # Add labels and title
    plt.title(f'LightGBM Feature Importance (Top {len(top_features)} features)', fontsize=14)
    plt.xlabel('Mean Importance (Gain)', fontsize=12)
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
        for col in NUMERIC_TO_CATEGORICAL:
            if col in df.columns and df[col].null_count() > 0:
                median_val = df[col].median()
                df = df.with_columns(pl.col(col).fill_null(median_val))
                logger.info(f"Filled NaNs in '{col}' with median: {median_val}")
            elif col in df.columns:
                logger.info(f"Checked '{col}': No NaNs found, skipping fill_null.")

        # Create power features
        if 'Duration' in df.columns:
            df = df.with_columns([
                pl.col('Duration').sqrt().alias('Duration_sqrt'),
                (pl.col('Duration') ** 2).alias('Duration_squared')
            ])
            logger.info("Created sqrt and squared features for Duration.")
        else:
            logger.warning("Column 'Duration' not found, skipping creation of sqrt and squared features for it.")

        # Convert numerical features to categorical by discretizing/binning
        logger.info("Converting numerical features to categorical...")
        for col in NUMERIC_TO_CATEGORICAL:
            if col in df.columns:
                # Get bin width from parameters or default to 10
                bin_width = BINNING_PARAMS.get(col, 10)
                
                # Round to the nearest bin width to create categories
                df = df.with_columns([
                    ((pl.col(col) / bin_width).floor() * bin_width).cast(pl.Utf8).alias(f"{col}_cat")
                ])
                logger.info(f"Converted {col} to categorical with bin width {bin_width}")
        
        # Add categorical versions of derived features
        if 'Duration_sqrt' in df.columns:
            df = df.with_columns([
                pl.col('Duration_sqrt').round(1).cast(pl.Utf8).alias('Duration_sqrt_cat')
            ])
            logger.info("Converted Duration_sqrt to categorical")
            
        if 'Duration_squared' in df.columns:
            df = df.with_columns([
                (pl.col('Duration_squared') / 10).floor().cast(pl.Utf8).alias('Duration_squared_cat')
            ])
            logger.info("Converted Duration_squared to categorical")

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}", exc_info=True)
        sys.exit(1)

    # --- Feature Engineering ---
    logger.info("Starting feature engineering...")
    try:
        # Aggregation Features
        df = create_aggregation_features(df, AGG_FEATURES_CONFIG)
        logger.info("Created aggregation features.")
        
        # Convert the aggregation features to categorical too
        for col in ['Sex_Duration_mean', 'Sex_Heart_Rate_mean']:
            if col in df.columns:
                df = df.with_columns([
                    pl.col(col).round(1).cast(pl.Utf8).alias(f"{col}_cat")
                ])
                logger.info(f"Converted {col} to categorical")

        # N-way Interaction Features using categorical versions
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

        # Split temporarily to apply ratio-to-category mean encoding
        temp_train_df = df[:train_len]
        temp_test_df = df[train_len:]
        
        # --- Apply Ratio-to-Category Mean Encoding ---
        logger.info(f"Applying Ratio-to-Category Mean Encoding...")
        
        # Check if all numerical features exist
        valid_num_features = [f for f in RATIO_FEATURES_CONFIG['numerical_features'] if f in temp_train_df.columns]
        
        if RATIO_FEATURES_CONFIG['category_col'] in temp_train_df.columns:
            temp_train_df, temp_test_df = apply_ratio_to_category_mean_encoding(
                train_df=temp_train_df,
                numerical_features=valid_num_features,
                category_col=RATIO_FEATURES_CONFIG['category_col'],
                test_df=temp_test_df,
                calculate_mean_on=RATIO_FEATURES_CONFIG['calculate_mean_on'],
                fill_value=1.0,
                new_col_suffix_format="_ratio_vs_{cat}"
            )
            logger.info(f"Applied Ratio-to-Category Mean Encoding for {len(valid_num_features)} numerical features relative to {RATIO_FEATURES_CONFIG['category_col']}")
        else:
            logger.warning(f"Category column {RATIO_FEATURES_CONFIG['category_col']} not found. Skipping Ratio-to-Category Mean Encoding.")
        
        # Merge back into a single DataFrame
        df = pl.concat([temp_train_df, temp_test_df], how="vertical")
        logger.info(f"DataFrame shape after Ratio-to-Category Mean Encoding: {df.shape}")

        # Final Type Casting
        df = df.cast({col: pl.Float32 for col in df.columns if df[col].dtype.is_numeric() and col != TARGET_COL})
        logger.info("Casted numeric features to Float32.")

    except Exception as e:
        logger.error(f"Error during feature engineering: {e}", exc_info=True)
        sys.exit(1)

    # --- Prepare Data for Model Training ---
    logger.info("Preparing data for modeling...")
    try:
        # Split back into Train and Test
        df_train_processed = df[:train_len]
        df_test_processed = df[train_len:]

        # Store Duration for later un-normalization
        train_duration = df_train_processed['Duration'].to_numpy()
        test_duration = df_test_processed['Duration'].to_numpy()
        
        # Separate target
        target_series_original = df_train_processed[TARGET_COL]
        
        # Normalize target by Duration and then apply log1p
        target_series_normalized = df_train_processed.with_columns(
            (pl.col(TARGET_COL) / pl.col('Duration')).alias("normalized_target")
        )["normalized_target"]
        
        # Apply log1p to the normalized target
        target_series_log = target_series_normalized.log1p()
        
        logger.info(f"Normalized target by Duration and applied log1p transformation")
        
        # Drop target from training and test data
        df_train_processed = df_train_processed.drop(TARGET_COL)
        df_test_processed = df_test_processed.drop(TARGET_COL)

        # Find all string (categorical) columns for count encoding
        categorical_cols_for_ce = df_train_processed.select(pl.col(pl.Utf8)).columns
        logger.info(f"Preparing to count encode {len(categorical_cols_for_ce)} categorical features.")

        # Generate CV indices for training
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        cv_indices = list(kf.split(np.zeros(train_len)))
        logger.info(f"Generated {len(cv_indices)} CV fold indices.")

        # --- Apply Count Encoding ---
        logger.info("Applying Count Encoding using TEST data as source...")
        
        train_df_ce, test_df_ce = apply_count_encoding(
            train_df=df_train_processed,
            features=categorical_cols_for_ce,
            test_df=df_test_processed,
            count_on='test',  # Use TEST data for counts
            normalize=True,    # Use frequency (normalized counts)
            handle_unknown='zero',
            new_col_suffix='_ce'
        )
        logger.info("Count Encoding applied using test data frequencies.")
        
        # Drop the original categorical columns since LightGBM can't handle them
        # Keep only the encoded versions (_ce suffix)
        train_df_ce = train_df_ce.drop(categorical_cols_for_ce)
        test_df_ce = test_df_ce.drop(categorical_cols_for_ce)
        
        logger.info(f"Dropped {len(categorical_cols_for_ce)} original categorical columns, keeping only count-encoded versions")

        # Define final features - all columns in the CE dataframe
        FINAL_FEATURES = train_df_ce.columns
        logger.info(f"Final number of features after CE: {len(FINAL_FEATURES)}")

        # Convert Polars DFs to Pandas for LightGBM
        logger.info("Converting DataFrames to Pandas for LightGBM...")
        train_pd_ce = train_df_ce.to_pandas()
        test_pd_ce = test_df_ce.to_pandas()
        target_pd_log_full = target_series_log.to_pandas()
        target_pd_original_full = target_series_original.to_pandas()
        logger.info("Conversion to Pandas complete.")

    except Exception as e:
        logger.error(f"Error during data preparation: {e}", exc_info=True)
        sys.exit(1)

    # --- Cross-Validation and Training ---
    # Handle NaN in target before CV loop
    nan_target_mask = target_pd_log_full.isna()
    if nan_target_mask.any():
        logger.warning(f"Found {nan_target_mask.sum()} NaN values in log-transformed target variable. Dropping corresponding rows from training data and original target.")
        train_pd_clean = train_pd_ce[~nan_target_mask].reset_index(drop=True)
        target_pd_log_clean = target_pd_log_full[~nan_target_mask].reset_index(drop=True)
        target_pd_original_clean = target_pd_original_full[~nan_target_mask].reset_index(drop=True)
        train_duration_clean = train_duration[~nan_target_mask]
        logger.info(f"Cleaned training data shape after dropping NaN targets: {train_pd_clean.shape}")
    else:
        train_pd_clean = train_pd_ce
        target_pd_log_clean = target_pd_log_full
        target_pd_original_clean = target_pd_original_full
        train_duration_clean = train_duration

    # Generate CV indices *after* potentially dropping rows
    cv_indices_for_training = list(kf.split(train_pd_clean))
    logger.info(f"Generated {len(cv_indices_for_training)} CV fold indices for training data shape {train_pd_clean.shape}.")

    logger.info(f"Starting {N_FOLDS}-Fold Cross-Validation with LightGBM...")
    oof_preds_original_scale = np.zeros(len(train_pd_clean))
    test_preds_sum_original_scale = np.zeros(len(test_pd_ce))
    models = []

    try:
        for fold, (train_idx, valid_idx) in enumerate(cv_indices_for_training):
            logger.info(f"--- Fold {fold + 1}/{N_FOLDS} ---")

            # Slice data for the fold
            X_train_fold = train_pd_clean.iloc[train_idx]
            y_train_fold = target_pd_log_clean.iloc[train_idx]
            X_valid_fold = train_pd_clean.iloc[valid_idx]
            y_valid_fold = target_pd_log_clean.iloc[valid_idx]
            
            # Get the duration values for this fold's validation set
            valid_duration = train_duration_clean[valid_idx]

            logger.info(f"Training LightGBM fold {fold + 1}...")
            # Create LightGBM datasets
            lgb_train = lgb.Dataset(X_train_fold, label=y_train_fold, feature_name=FINAL_FEATURES)
            lgb_valid = lgb.Dataset(X_valid_fold, label=y_valid_fold, feature_name=FINAL_FEATURES, reference=lgb_train)

            # Train the model
            model = lgb.train(
                params=LGBM_PARAMS,
                train_set=lgb_train,
                valid_sets=[lgb_train, lgb_valid],
                valid_names=['train', 'validation'],
                **LGBM_FIT_PARAMS
            )

            # Store model
            models.append(model)

            # Predict (log scale of normalized target)
            fold_oof_preds_log = model.predict(X_valid_fold, num_iteration=model.best_iteration)
            fold_test_preds_log = model.predict(test_pd_ce, num_iteration=model.best_iteration)

            # Transform predictions back to original scale:
            # 1. expm1 to undo log1p
            # 2. multiply by Duration to get total calories from calories/minute
            fold_oof_preds_orig = np.expm1(fold_oof_preds_log) * valid_duration
            fold_test_preds_orig = np.expm1(fold_test_preds_log) * test_duration

            # Store original scale predictions
            oof_preds_original_scale[valid_idx] = fold_oof_preds_orig
            test_preds_sum_original_scale += fold_test_preds_orig
            logger.info(f"Fold {fold + 1} prediction complete. Best Iteration: {model.best_iteration}")
            logger.info("-" * 50)

        # --- Generate Feature Importance Plot ---
        logger.info("Generating feature importance plot...")
        plot_feature_importance(
            models=models, 
            feature_names=FINAL_FEATURES,
            output_path=FEATURE_IMPORTANCE_PATH,
            top_n=30
        )
        logger.info(f"Feature importance plot saved to: {FEATURE_IMPORTANCE_PATH}")

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
            headers = ['script_file_name', 'why', 'what', 'how', 'cv_score', 'lb_score', 'status', 'experiment_type', 'base_experiment', 'new_feature']
            pd.DataFrame(columns=headers).to_csv(EXPERIMENTS_CSV, index=False, sep=';')

        experiments_df = pd.read_csv(EXPERIMENTS_CSV, sep=';')
        exp_mask = experiments_df['script_file_name'] == EXPERIMENT_NAME
        if exp_mask.sum() > 0:
            # Update existing row
            idx_to_update = experiments_df[exp_mask].index[0]
            experiments_df.loc[idx_to_update, 'cv_score'] = overall_cv_score
            experiments_df.loc[idx_to_update, 'status'] = 'done'
            experiments_df.loc[idx_to_update, 'why'] = 'Test ratio-to-category mean encoding with normalized target.'
            experiments_df.loc[idx_to_update, 'what'] = 'Train LGBM using count encoding, ratio-to-category mean encoding, and target normalization.'
            experiments_df.loc[idx_to_update, 'how'] = 'Normalize target by dividing Calories by Duration, apply count encoding and ratio-to-category mean encoding, KFold(7) LGBM, generate feature importance.'
            experiments_df.loc[idx_to_update, 'experiment_type'] = 'single_model'
            experiments_df.loc[idx_to_update, 'base_experiment'] = 'exp_k_lgbm_normalized_target.py'
            experiments_df.loc[idx_to_update, 'new_feature'] = 'Ratio-to-category mean encoding'
            logger.info(f"Updated existing experiment '{EXPERIMENT_NAME}' with CV score: {overall_cv_score:.5f}")
        else:
             # Add as new row
             new_row = pd.DataFrame([{
                 'script_file_name': EXPERIMENT_NAME,
                 'why': 'Test ratio-to-category mean encoding with normalized target.',
                 'what': 'Train LGBM using count encoding, ratio-to-category mean encoding, and target normalization.',
                 'how': 'Normalize target by dividing Calories by Duration, apply count encoding and ratio-to-category mean encoding, KFold(7) LGBM, generate feature importance.',
                 'cv_score': overall_cv_score,
                 'lb_score': None,
                 'status': 'done',
                 'experiment_type': 'single_model',
                 'base_experiment': 'exp_k_lgbm_normalized_target.py',
                 'new_feature': 'Ratio-to-category mean encoding'
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