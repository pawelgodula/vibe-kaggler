# Internal utility function to train scikit-learn compatible models.

"""
Extended Description:
Trains a scikit-learn compatible model instance on a single fold of data.
Handles fitting the model and generating predictions for validation and test sets.
Assumes input data (X_train, X_valid, X_test) are numerical arrays (e.g., NumPy).
It supports models with both `predict` and `predict_proba` methods, choosing the appropriate
one based on standard scikit-learn conventions (classification models usually have `predict_proba`).
"""

import numpy as np
from typing import Optional, Tuple, Any, Dict
from sklearn.base import BaseEstimator # Use BaseEstimator for type hint

def _train_sklearn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: Optional[np.ndarray],
    y_valid: Optional[np.ndarray],
    X_test: Optional[np.ndarray],
    model_class: type, # Pass the actual class, e.g., RandomForestClassifier
    model_params: Dict[str, Any],
    fit_params: Dict[str, Any]
) -> Tuple[BaseEstimator, Optional[np.ndarray], Optional[np.ndarray]]:
    """Trains a scikit-learn compatible model on one fold.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target.
        X_valid (Optional[np.ndarray]): Validation features.
        y_valid (Optional[np.ndarray]): Validation target (currently unused by basic fit/predict).
        X_test (Optional[np.ndarray]): Test features.
        model_class (type): The scikit-learn model class to instantiate (e.g., RandomForestClassifier).
        model_params (Dict[str, Any]): Parameters to initialize the model class.
        fit_params (Dict[str, Any]): Parameters to pass to the model's `fit` method (e.g., sample_weight).

    Returns:
        Tuple[BaseEstimator, Optional[np.ndarray], Optional[np.ndarray]]:
            - fitted_model: The trained scikit-learn model object.
            - y_pred_valid: Predictions on the validation set (or None if X_valid is None).
            - y_pred_test: Predictions on the test set (or None if X_test is None).

    Raises:
        ValueError: If model instantiation or fitting fails.
        AttributeError: If the model lacks expected fit/predict methods.
    """
    try:
        model = model_class(**model_params)
    except Exception as e:
        raise ValueError(f"Failed to instantiate model {model_class.__name__} with params {model_params}: {e}") from e

    try:
        # Note: Sklearn models generally don't use validation set directly in fit
        # unless using specific callbacks or meta-estimators not handled here.
        # Fit params like sample_weight can be passed.
        model.fit(X_train, y_train, **fit_params)
        print(f"Trained {model_class.__name__} with params: {model_params}")
    except Exception as e:
        raise ValueError(f"Failed to fit model {model_class.__name__}: {e}") from e

    y_pred_valid: Optional[np.ndarray] = None
    y_pred_test: Optional[np.ndarray] = None

    # Determine prediction method (predict_proba for classifiers, predict otherwise)
    predict_method_name = "predict"
    if hasattr(model, "predict_proba"):
         # Check if it's likely a classifier (common convention)
         # A more robust check might involve inspecting _estimator_type attribute
        if getattr(model, "_estimator_type", None) == "classifier":
              predict_method_name = "predict_proba"
              
    predict_method = getattr(model, predict_method_name)

    if X_valid is not None:
        try:
            y_pred_valid = predict_method(X_valid)
            # For predict_proba, often want probability of the positive class (class 1)
            if predict_method_name == "predict_proba" and y_pred_valid.ndim == 2 and y_pred_valid.shape[1] >= 2:
                 # Handle binary vs multiclass probability outputs
                 if y_pred_valid.shape[1] == 2:
                     y_pred_valid = y_pred_valid[:, 1] # Probability of class 1 for binary
                 # else: return all probabilities for multiclass (shape [n_samples, n_classes])

        except Exception as e:
            print(f"Warning: Failed to generate validation predictions with {model_class.__name__}: {e}")
            # Optionally raise, but often prefer to continue to get test preds if possible

    if X_test is not None:
        try:
            y_pred_test = predict_method(X_test)
            # For predict_proba, often want probability of the positive class (class 1)
            if predict_method_name == "predict_proba" and y_pred_test.ndim == 2 and y_pred_test.shape[1] >= 2:
                 if y_pred_test.shape[1] == 2:
                     y_pred_test = y_pred_test[:, 1] # Probability of class 1 for binary
                 # else: return all probabilities for multiclass

        except Exception as e:
            print(f"Warning: Failed to generate test predictions with {model_class.__name__}: {e}")
            # Optionally raise

    return model, y_pred_valid, y_pred_test 