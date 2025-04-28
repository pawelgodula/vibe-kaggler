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