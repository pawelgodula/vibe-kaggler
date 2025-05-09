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
    fit_params: Dict[str, Any],
    # These are added to match signature of other _train_* funcs, but not always used by sklearn
    feature_cols: Optional[list[str]] = None, 
    cat_features: Optional[list[str]] = None 
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
        feature_cols (Optional[list[str]]): Names of feature columns (primarily for metadata, not direct use by sklearn fit).
        cat_features (Optional[list[str]]): Names of categorical features (primarily for metadata, not direct use by sklearn fit).

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


if __name__ == '__main__':
    # Example Usage
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_regression, make_classification
    from sklearn.model_selection import train_test_split

    print("Testing _train_sklearn function...")

    # Regression Example
    print("\n--- Regression Example (RandomForestRegressor) ---")
    X_reg, y_reg = make_regression(n_samples=100, n_features=10, random_state=42)
    X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    X_test_reg = rng.rand(20, 10) # Dummy test data
    
    reg_model_params = {'n_estimators': 10, 'random_state': 42}
    reg_fit_params = {}
    
    try:
        reg_model, reg_val_preds, reg_test_preds = _train_sklearn(
            X_train_reg, y_train_reg, X_val_reg, y_val_reg, X_test_reg,
            RandomForestRegressor, reg_model_params, reg_fit_params
        )
        print(f"Fitted Regressor: {type(reg_model)}")
        if reg_val_preds is not None:
            print(f"Validation preds shape: {reg_val_preds.shape}, first 5: {reg_val_preds[:5]}")
        if reg_test_preds is not None:
            print(f"Test preds shape: {reg_test_preds.shape}, first 5: {reg_test_preds[:5]}")
    except Exception as e:
        print(f"Error in regression example: {e}")

    # Classification Example (Logistic Regression)
    print("\n--- Classification Example (LogisticRegression) ---")
    X_clf, y_clf = make_classification(n_samples=100, n_features=10, random_state=42)
    X_train_clf, X_val_clf, y_train_clf, y_val_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    X_test_clf = rng.rand(20,10)
    
    clf_model_params = {'solver': 'liblinear', 'random_state': 42}
    clf_fit_params = {}

    try:
        clf_model, clf_val_preds, clf_test_preds = _train_sklearn(
            X_train_clf, y_train_clf, X_val_clf, y_val_clf, X_test_clf,
            LogisticRegression, clf_model_params, clf_fit_params
        )
        print(f"Fitted Classifier: {type(clf_model)}")
        if clf_val_preds is not None:
            print(f"Validation preds (probs class 1) shape: {clf_val_preds.shape}, first 5: {clf_val_preds[:5]}")
        if clf_test_preds is not None:
            print(f"Test preds (probs class 1) shape: {clf_test_preds.shape}, first 5: {clf_test_preds[:5]}")
            
        # Test with predict_proba giving multiclass output (though LogisticRegression is binary here)
        # For true multiclass, y_pred_valid/test would be [n_samples, n_classes]
        # This part is more to test the ndim == 2 and shape[1] >=2 logic
        if hasattr(clf_model, 'predict_proba'):
            raw_probs_val = clf_model.predict_proba(X_val_clf)
            print(f"Raw validation predict_proba output shape: {raw_probs_val.shape}")
            
    except Exception as e:
        print(f"Error in classification example: {e}") 