# Kaggle Tabular Competition Pipeline Plan

This document outlines the essential utility functions planned for the `utils/` directory to support a typical tabular Kaggle competition workflow using Polars. Each function aims to be pure and reusable.

## Core Function Signatures

### 1. Data Loading & Saving

- **`load_csv(file_path: str, **kwargs) -> pl.DataFrame`**: Loads data from a CSV file into a Polars DataFrame.
- **`load_parquet(file_path: str, **kwargs) -> pl.DataFrame`**: Loads data from a Parquet file into a Polars DataFrame.
- **`save_csv(df: pl.DataFrame, file_path: str, **kwargs) -> None`**: Saves a Polars DataFrame to a CSV file.
- **`save_parquet(df: pl.DataFrame, file_path: str, **kwargs) -> None`**: Saves a Polars DataFrame to a Parquet file.

### 2. Cross-Validation Strategy

- **`get_cv_splitter(cv_type: str = 'kfold', n_splits: int = 5, shuffle: bool = True, random_state: int | None = None, **kwargs) -> sklearn.model_selection.BaseCrossValidator`**: Returns a configured scikit-learn cross-validation splitter instance (e.g., KFold, StratifiedKFold, GroupKFold).
- **`get_cv_indices(df: pl.DataFrame, cv_splitter: sklearn.model_selection.BaseCrossValidator, target_col: Optional[str] = None, groups: Optional[pl.Series] = None) -> List[Tuple[np.ndarray, np.ndarray]]`**: Generates lists of train/validation row indices for each fold using a DataFrame and a splitter.

### 3. Preprocessing (Fit/Transform Pattern)

*Note: Preprocessing functions that require fitting (like scalers, encoders) return the fitted object along with the transformed data.*

- **`scale_numerical_features(train_df: pl.DataFrame, test_df: Optional[pl.DataFrame] = None, features: list[str], scaler_type: str = 'standard', **scaler_kwargs) -> tuple[pl.DataFrame, Optional[pl.DataFrame], Any]`**: Fits a scaler on the training data (NumPy conversion), transforms train/test Polars DataFrames, and returns transformed DFs and the fitted scaler.
- **`encode_categorical_features(train_df: pl.DataFrame, test_df: Optional[pl.DataFrame] = None, features: list[str], encoder_type: str = 'onehot', handle_unknown: str = 'ignore', **encoder_kwargs) -> tuple[pl.DataFrame, Optional[pl.DataFrame], Any]`**: Fits an encoder on training data (NumPy conversion), transforms train/test Polars DataFrames, and returns transformed DFs and the fitted encoder.
- **`bin_and_encode_numerical_features(train_df: pl.DataFrame, test_df: Optional[pl.DataFrame] = None, features: list[str], n_bins: int, strategy: str = 'quantile', encoder_type: str = 'onehot', **kwargs) -> tuple[pl.DataFrame, Optional[pl.DataFrame], Any]`**: (Conceptual) Bins numerical features based on training data distribution (e.g., quantiles) and then applies categorical encoding (e.g., OneHot, Count) to the bins. Returns transformed DFs and fitted encoder/bin edges.
- **`handle_missing_values(df: pl.DataFrame, strategy: str = 'mean', features: list[str] | None = None, fill_value: Optional[Any] = None) -> pl.DataFrame`**: Imputes missing values using a specified strategy (Polars expressions).
- **(Note)** Complex missing value imputation (e.g., model-based, rule-based inference like filling building IDs based on lat/lon) or data standardization (e.g., address cleaning 'st' vs 'street') are often highly custom and typically implemented in competition-specific scripts.

### 4. Feature Engineering

*Note: Feature engineering is often highly competition-specific.*

- **`create_polynomial_features(df: pl.DataFrame, features: list[str], degree: int = 2, interaction_only: bool = False, include_bias: bool = False, new_col_prefix: str = 'poly_') -> pl.DataFrame`**: Generates polynomial/interaction features (using sklearn via NumPy) and appends them to the DataFrame.
- **`create_interaction_features(df: pl.DataFrame, feature_pairs: list[tuple[str, str]]) -> pl.DataFrame`**: Creates interaction terms for specified feature pairs (using Polars expressions) and returns *only* the new features.
- **`(Conceptual) create_nway_interaction_features(df: pl.DataFrame, features: list[str], n: int = 2, encoder_type: Optional[str] = 'count', **kwargs) -> pl.DataFrame`**: (Conceptual) Generates N-way interaction features (e.g., concatenating string representations of binned/categorical features) and potentially encodes them (e.g., count encoding). Complexity grows rapidly with N.
- **(Example) `create_aggregation_features(df: pl.DataFrame, group_by_cols: list[str], agg_dict: dict, new_col_prefix: str = 'agg_') -> pl.DataFrame`**: Generates aggregation features (e.g., mean, std, count) based on grouping columns and appends them.
- **(Example) `apply_target_encoding(train_df: pl.DataFrame, test_df: Optional[pl.DataFrame], features: list[str], target_col: str, cv_indices: List[Tuple[np.ndarray, np.ndarray]], smoothing: float = 10.0) -> tuple[pl.DataFrame, Optional[pl.DataFrame]]`**: Applies target encoding using a cross-validation scheme to prevent leakage.
- **(Note)** Advanced feature engineering based on duplicate detection (e.g., grouping by near-identical listings, creating lag/lead/rank features within groups based on time) is highly problem-dependent and usually requires custom competition-specific scripts.

### 5. Feature Selection (Conceptual)

*Note: Feature selection often involves iterative training and evaluation.*

- **`(Conceptual) select_features_forward(train_df: pl.DataFrame, target_col: str, initial_features: List[str], candidate_features: List[str], cv_indices: List[Tuple[np.ndarray, np.ndarray]], model_type: str, model_params: dict, fit_params: dict, metric_name: str, higher_is_better: bool) -> List[str]`**: Performs forward feature selection based on CV improvement.
- **`(Conceptual) select_features_backward(train_df: pl.DataFrame, target_col: str, initial_features: List[str], cv_indices: List[Tuple[np.ndarray, np.ndarray]], model_type: str, model_params: dict, fit_params: dict, metric_name: str, higher_is_better: bool) -> List[str]`**: Performs backward feature elimination based on CV performance drop.

### 6. Model Training (Conceptual - Usually Competition Specific)

*Note: Model training logic is often implemented in competition-specific scripts/notebooks rather than generic utils, adapting these patterns.*

- **`train_and_cv(train_df: pl.DataFrame, test_df: Optional[pl.DataFrame], target_col: str, feature_cols: List[str], model_type: str, cv_indices: List[Tuple[np.ndarray, np.ndarray]], model_params: dict, fit_params: dict, **kwargs) -> Tuple[np.ndarray, np.ndarray, List[Any]]`**: 
    - Orchestrates model training across specified CV folds.
    - Takes train/test data, feature/target names, model type (e.g., 'lgbm', 'xgb', 'catboost', 'nn', 'rf', 'et', 'knn', 'linear'), CV indices, model parameters, and fit parameters.
    - Iterates through folds, calling `train_single_fold` for each.
    - Returns out-of-fold (OOF) predictions for the training set, averaged predictions for the test set, and a list of fitted models (or model performance metrics).

- **`train_single_fold(train_fold_df: pl.DataFrame, valid_fold_df: pl.DataFrame, test_df: Optional[pl.DataFrame], target_col: str, feature_cols: List[str], model_type: str, model_params: dict, fit_params: dict, **kwargs) -> Tuple[Any, np.ndarray, np.ndarray]`**: 
    - Trains a specified model type on a single fold.
    - Takes train/validation fold data, optional test data, feature/target names, model type, model parameters, and fit parameters.
    - Calls the appropriate underlying model training function (e.g., `_train_lgbm`, `_train_sklearn`).
    - Handles data conversion (e.g., Polars to NumPy/Pandas) as needed by the underlying trainer.
    - Returns the fitted model object, validation predictions, and test predictions for that fold.

- **`(Internal) _train_lgbm(X_train, y_train, X_valid, y_valid, X_test, model_params, fit_params)`**: (Example internal function for LightGBM)
- **`(Internal) _train_xgb(X_train, y_train, X_valid, y_valid, X_test, model_params, fit_params)`**: (Example internal function for XGBoost)
- **`(Internal) _train_catboost(...)`**
- **`(Internal) _train_nn(...)`**
- **`(Internal) _train_sklearn(X_train, y_train, X_valid, y_valid, X_test, model_class, model_params, fit_params)`**: (Conceptual internal function for scikit-learn compatible models like RandomForest, ExtraTrees, LogisticRegression, Ridge, Lasso, KNN, etc.). Requires passing the specific model class.

### 7. Model Evaluation

- **`calculate_metric(y_true: np.ndarray | pl.Series, y_pred: np.ndarray | pl.Series, metric_name: str, **kwargs) -> float`**: Calculates a specified evaluation metric (e.g., 'accuracy', 'rmse', 'auc', 'logloss'). Wraps common metrics from libraries like scikit-learn.

### 8. Ensembling (Examples)

- **`average_predictions(prediction_list: List[np.ndarray | pl.Series]) -> np.ndarray`**: Calculates the simple average of a list of prediction arrays/series.
- **`(Conceptual) blend_predictions(prediction_map: Dict[str, np.ndarray | pl.Series], weights: Dict[str, float]) -> np.ndarray`**: Calculates a weighted blend of predictions.
- **`(Conceptual) train_stacking_meta_model(oof_predictions: pl.DataFrame, y_true: pl.Series | np.ndarray, test_predictions: pl.DataFrame, meta_model_type: str, cv_indices: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray, Any]`**: Trains a meta-model (e.g., Linear, LGBM, NN) on out-of-fold predictions from base models. Returns final OOF predictions, final test predictions, and the fitted meta-model. Requires OOF predictions from `train_and_cv` as input.

### 9. Submission Generation

- **`generate_submission_file(ids: pl.Series | np.ndarray, predictions: pl.Series | np.ndarray, id_col_name: str, target_col_name: str, file_path: str) -> None`**: Creates a Kaggle submission CSV file in the standard format.

### 10. Utilities

- **`sample_dataframe(df: pl.DataFrame, n: Optional[int] = None, frac: Optional[float] = None, shuffle: bool = True, random_state: Optional[int] = None) -> pl.DataFrame`**: Samples rows from a Polars DataFrame.
- **`setup_logging(log_level: str = 'INFO', log_file: str | None = None) -> None`**: Configures Python's standard logging.
- **`seed_everything(seed: int) -> None`**: Sets random seeds for Python, NumPy, and potentially frameworks like TensorFlow/PyTorch for reproducibility.
- **`timer_context_manager() -> contextlib.contextmanager`**: A context manager to time code blocks.
- **`reduce_memory_usage(df: pl.DataFrame, verbose: bool = True) -> pl.DataFrame`**: Downcasts numerical columns to reduce DataFrame memory usage (using Polars types).

### 11. Experiment Automation (Note)

- **(Note)** Building a system for automatically generating and evaluating models by treating preprocessing choices (null handling, scaling, binning), feature transformations (interactions), model types, and hyperparameters as tunable parameters (as described by the GM) is a significant undertaking. It often involves external libraries (like Optuna, Hyperopt) or custom frameworks built on top of the foundational utilities listed above. Such automation frameworks are typically beyond the scope of simple, reusable `utils` functions.

## Implementation Notes

- Each function will reside in its own file within the `utils/` directory, named after the function (e.g., `utils/load_csv.py`).
- Each file will include a one-line description, an extended description, type hints, the function definition, and associated unit tests (in a separate `tests/utils/` directory).
- Dependencies like `polars`, `numpy`, `scikit-learn`, and potentially model libraries (LightGBM, XGBoost, CatBoost, PyTorch/TensorFlow) will be required. 