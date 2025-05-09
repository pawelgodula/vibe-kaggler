# Calculates a specified evaluation metric for given true and predicted values.

"""
Extended Description:
This function acts as a unified interface to various evaluation metrics, primarily
from scikit-learn. It takes true and predicted values (as numpy arrays or Polars Series)
and the name of the metric (e.g., 'accuracy', 'rmse', 'auc', 'logloss')
as a string. It handles mapping the string name to the corresponding scikit-learn
function, converting inputs to NumPy if needed, and calculates the score.
"""

import numpy as np
import polars as pl
from sklearn import metrics
from typing import Union, Any

METRIC_MAP = {
    # Classification Metrics
    'accuracy': metrics.accuracy_score,
    'precision': metrics.precision_score,
    'recall': metrics.recall_score,
    'f1': metrics.f1_score,
    'roc_auc': metrics.roc_auc_score,
    'auc': metrics.roc_auc_score, # Alias for roc_auc
    'logloss': metrics.log_loss,
    'balanced_accuracy': metrics.balanced_accuracy_score,
    # Add more classification metrics (e.g., cohen_kappa_score) if needed

    # Regression Metrics
    'mse': metrics.mean_squared_error,
    'rmse': lambda y_true, y_pred, **kwargs: np.sqrt(metrics.mean_squared_error(y_true, y_pred, **kwargs)),
    'mae': metrics.mean_absolute_error,
    'r2': metrics.r2_score,
    'msle': metrics.mean_squared_log_error,
    'rmsle': lambda y_true, y_pred, **kwargs: np.sqrt(metrics.mean_squared_log_error(y_true, y_pred, **kwargs)),
    'mape': metrics.mean_absolute_percentage_error,
    # Add more regression metrics if needed
}

def calculate_metric(
    y_true: Union[np.ndarray, pl.Series],
    y_pred: Union[np.ndarray, pl.Series],
    metric_name: str,
    **kwargs: Any
) -> float:
    """Calculates a specified evaluation metric.

    Args:
        y_true (Union[np.ndarray, pl.Series]): Ground truth target values.
        y_pred (Union[np.ndarray, pl.Series]): Predicted target values.
            For metrics like 'roc_auc' or 'logloss', this should be predicted
            probabilities for the positive class or class probabilities.
        metric_name (str): The name of the metric to calculate.
            Supported names are keys in the internal METRIC_MAP.
        **kwargs (Any): Additional keyword arguments passed directly to the
                        underlying scikit-learn metric function (e.g., `average`
                        for f1_score, `squared=False` for rmse using mse).

    Returns:
        float: The calculated metric score.

    Raises:
        ValueError: If an unsupported metric_name is provided.
        Exception: Propagates exceptions from the underlying metric calculation.
    """
    metric_name_lower = metric_name.lower()

    metric_func = METRIC_MAP.get(metric_name_lower)

    if metric_func is None:
        raise ValueError(
            f"Unsupported metric_name: '{metric_name}'. Supported metrics: "
            f"{list(METRIC_MAP.keys())}"
        )

    # Convert Polars Series to NumPy arrays for scikit-learn compatibility
    if isinstance(y_true, pl.Series):
        y_true_np = y_true.to_numpy()
    else:
        y_true_np = y_true # Assume it's already a numpy array or compatible
        
    if isinstance(y_pred, pl.Series):
        y_pred_np = y_pred.to_numpy()
    else:
        y_pred_np = y_pred # Assume it's already a numpy array or compatible

    # Special handling for RMSE/RMSLE if not directly using the lambda
    # (The lambda approach is generally cleaner)
    # Example: if metric_name_lower == 'rmse':
    #     kwargs.setdefault('squared', False)
    #     metric_func = metrics.mean_squared_error

    try:
        score = metric_func(y_true_np, y_pred_np, **kwargs)
        return float(score) # Ensure return type is float
    except ValueError as ve:
        print(f"ValueError during '{metric_name}' calculation: {ve}")
        print("Check if y_pred format is correct for the metric (e.g., probabilities for AUC/logloss).")
        raise
    except Exception as e:
        print(f"Error calculating metric '{metric_name}': {e}")
        raise


if __name__ == '__main__':
    # Example Usage
    print("Testing calculate_metric function...")

    # Regression examples
    y_true_reg = np.array([3, -0.5, 2, 7])
    y_pred_reg = np.array([2.5, 0.0, 2, 8])
    print(f"\n--- Regression Metrics ---")
    print(f"MAE: {calculate_metric(y_true_reg, y_pred_reg, 'mae')}") # Expected: 0.5
    print(f"MSE: {calculate_metric(y_true_reg, y_pred_reg, 'mse')}") # Expected: 0.375
    print(f"RMSE: {calculate_metric(y_true_reg, y_pred_reg, 'rmse')}") # Expected: sqrt(0.375) = 0.61237
    print(f"R2: {calculate_metric(y_true_reg, y_pred_reg, 'r2')}") # Expected: 0.9486

    y_true_reg_pos = np.array([1, 2, 3, 4, 5])
    y_pred_reg_pos = np.array([1.1, 2.2, 2.8, 4.3, 5.1])
    print(f"MSLE: {calculate_metric(y_true_reg_pos, y_pred_reg_pos, 'msle')}")
    print(f"RMSLE: {calculate_metric(y_true_reg_pos, y_pred_reg_pos, 'rmsle')}")

    # Classification examples
    y_true_clf = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    y_pred_clf_labels = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    y_pred_clf_probs = np.array([0.1, 0.9, 0.6, 0.8, 0.2, 0.3, 0.95, 0.15])
    print(f"\n--- Classification Metrics ---")
    print(f"Accuracy: {calculate_metric(y_true_clf, y_pred_clf_labels, 'accuracy')}") # Expected: 0.75
    print(f"Precision (binary): {calculate_metric(y_true_clf, y_pred_clf_labels, 'precision')}") # TP=3, FP=1 -> 3/4 = 0.75
    print(f"Recall (binary): {calculate_metric(y_true_clf, y_pred_clf_labels, 'recall')}")    # TP=3, FN=1 -> 3/4 = 0.75
    print(f"F1 (binary): {calculate_metric(y_true_clf, y_pred_clf_labels, 'f1')}")          # Expected: 0.75
    print(f"Balanced Accuracy: {calculate_metric(y_true_clf, y_pred_clf_labels, 'balanced_accuracy')}") # (3/4 + 3/4)/2 = 0.75
    print(f"ROC AUC: {calculate_metric(y_true_clf, y_pred_clf_probs, 'roc_auc')}")
    print(f"LogLoss: {calculate_metric(y_true_clf, y_pred_clf_probs, 'logloss')}")

    # Example with Polars Series
    y_true_pl = pl.Series("true", y_true_reg)
    y_pred_pl = pl.Series("pred", y_pred_reg)
    print(f"\n--- Polars Series Input ---")
    print(f"MAE (Polars): {calculate_metric(y_true_pl, y_pred_pl, 'mae')}")

    # Error case: unsupported metric
    print("\n--- Error Case: Unsupported Metric ---")
    try:
        calculate_metric(y_true_reg, y_pred_reg, 'unsupported_metric_blah')
    except ValueError as e:
        print(f"Caught expected error: {e}") 