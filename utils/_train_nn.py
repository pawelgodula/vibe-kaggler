# Internal utility function to train a simple MLP for tabular data using PyTorch.

"""
Extended Description:
Trains a Multi-Layer Perceptron (MLP) model on a single fold of data using PyTorch.
Handles model definition, optimizer setup, loss calculation, training loop,
validation, early stopping, and prediction generation.
Assumes input data (X_train, X_valid, X_test) are NumPy arrays.
Returns the trained model's state_dict, validation predictions, and test predictions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, Tuple, Any, Dict, List
from copy import deepcopy # For saving best model state

# Define a simple MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, dropout_rate: float = 0.2):
        super(SimpleMLP, self).__init__()
        layers = []
        layer_sizes = [input_size] + hidden_sizes
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            layers.append(nn.Dropout(dropout_rate))
            
        # Output layer
        layers.append(nn.Linear(layer_sizes[-1], output_size))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def _train_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: Optional[np.ndarray],
    y_valid: Optional[np.ndarray],
    X_test: Optional[np.ndarray],
    model_params: Dict[str, Any],
    fit_params: Dict[str, Any],
    feature_cols: List[str], # For input_size determination
    cat_features: Optional[List[str]] = None # Currently unused by this simple MLP
) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
    """Trains a SimpleMLP model on one fold using PyTorch.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target.
        X_valid (Optional[np.ndarray]): Validation features.
        y_valid (Optional[np.ndarray]): Validation target.
        X_test (Optional[np.ndarray]): Test features.
        model_params (Dict[str, Any]): Parameters for the MLP constructor 
            (e.g., hidden_sizes, output_size, dropout_rate).
        fit_params (Dict[str, Any]): Parameters for the training process 
            (e.g., epochs, batch_size, lr, optimizer, loss_fn, 
             early_stopping_rounds, device).
        feature_cols (List[str]): List of feature names (used to infer input_size).
        cat_features (Optional[List[str]]): Currently unused.

    Returns:
        Tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
            - best_model_state_dict: State dictionary of the best model based on validation loss.
            - y_pred_valid: Predictions on the validation set (or None).
            - y_pred_test: Predictions on the test set (or None).

    Raises:
        ValueError: If required parameters are missing or invalid.
        ImportError: If torch is not installed.
    """
    
    # --- Parameter Extraction and Validation --- 
    input_size = X_train.shape[1]
    hidden_sizes = model_params.get('hidden_sizes', [64, 32])
    output_size = model_params.get('output_size', 1)
    dropout_rate = model_params.get('dropout_rate', 0.2)
    
    epochs = fit_params.get('epochs', 100)
    batch_size = fit_params.get('batch_size', 64)
    learning_rate = fit_params.get('lr', 1e-3)
    optimizer_name = fit_params.get('optimizer', 'adam').lower()
    loss_fn_name = fit_params.get('loss_fn', 'mse').lower() # Default to regression
    early_stopping_rounds = fit_params.get('early_stopping_rounds', 10)
    device_name = fit_params.get('device', 'auto')
    verbose_eval = fit_params.get('verbose', 10) # How often to print loss

    if device_name == 'auto':
         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
         device = torch.device(device_name)
    print(f"Using device: {device}")

    # --- Model, Optimizer, Loss --- 
    model = SimpleMLP(input_size, hidden_sizes, output_size, dropout_rate).to(device)
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # Add more optimizers if needed
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Determine task type and loss function
    # Convert y data to appropriate type and shape based on loss
    if loss_fn_name == 'mse' or loss_fn_name == 'rmse': # Regression
        criterion = nn.MSELoss()
        is_regression = True
        y_train_proc = y_train.astype(np.float32).reshape(-1, 1)
        y_valid_proc = y_valid.astype(np.float32).reshape(-1, 1) if y_valid is not None else None
    elif loss_fn_name == 'bce' or loss_fn_name == 'binary_crossentropy': # Binary Classification
        criterion = nn.BCEWithLogitsLoss() # Handles sigmoid internally
        is_regression = False
        if output_size != 1:
             print(f"Warning: BCE loss typically used with output_size=1, but got {output_size}")
        y_train_proc = y_train.astype(np.float32).reshape(-1, 1)
        y_valid_proc = y_valid.astype(np.float32).reshape(-1, 1) if y_valid is not None else None
    elif loss_fn_name == 'ce' or loss_fn_name == 'crossentropy': # Multiclass Classification
        criterion = nn.CrossEntropyLoss()
        is_regression = False
        # Target should be class indices (long tensor), not one-hot
        y_train_proc = y_train.astype(np.int64)
        y_valid_proc = y_valid.astype(np.int64) if y_valid is not None else None
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn_name}")

    # --- Data Loaders ---    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_proc, dtype=torch.float32 if is_regression or loss_fn_name == 'bce' else torch.long)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_loader = None
    if X_valid is not None and y_valid_proc is not None:
        X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
        y_valid_tensor = torch.tensor(y_valid_proc, dtype=torch.float32 if is_regression or loss_fn_name == 'bce' else torch.long)
        valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size * 2) # Larger batch size for validation
    elif early_stopping_rounds > 0:
         print("Warning: Early stopping enabled but no validation data provided.")
         early_stopping_rounds = 0 # Disable early stopping
         
    # --- Training Loop --- 
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state_dict = deepcopy(model.state_dict()) # Initial state

    for epoch in range(epochs):
        model.train() # Set model to training mode
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # --- Validation --- 
        val_loss = float('inf') # Default if no validation
        if valid_loader:
            model.eval() # Set model to evaluation mode
            avg_val_loss = 0.0
            with torch.no_grad():
                for batch_X_val, batch_y_val in valid_loader:
                    batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                    outputs_val = model(batch_X_val)
                    loss_val = criterion(outputs_val, batch_y_val)
                    avg_val_loss += loss_val.item() * batch_X_val.size(0)
            avg_val_loss /= len(valid_loader.dataset)
            val_loss = avg_val_loss # Use average for comparison
            
            if verbose_eval > 0 and (epoch + 1) % verbose_eval == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # --- Early Stopping --- 
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state_dict = deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if early_stopping_rounds > 0 and epochs_no_improve >= early_stopping_rounds:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
        else:
             # No validation, just print train loss and save current model state as best
            if verbose_eval > 0 and (epoch + 1) % verbose_eval == 0:
                 print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
            best_model_state_dict = deepcopy(model.state_dict()) 

    # --- Prediction --- 
    # Load best model state
    model.load_state_dict(best_model_state_dict)
    model.eval() # Ensure model is in eval mode for predictions
    
    y_pred_valid: Optional[np.ndarray] = None
    y_pred_test: Optional[np.ndarray] = None

    # Generate validation predictions
    if valid_loader:
        all_val_preds = []
        with torch.no_grad():
            for batch_X_val, _ in valid_loader:
                batch_X_val = batch_X_val.to(device)
                outputs = model(batch_X_val)
                # Apply activation for prediction output if needed
                if loss_fn_name == 'bce': # BCEWithLogitsLoss
                    outputs = torch.sigmoid(outputs)
                elif loss_fn_name == 'ce': # CrossEntropyLoss
                    outputs = torch.softmax(outputs, dim=1)
                # No activation needed for MSE regression outputs
                all_val_preds.append(outputs.cpu().numpy())
        y_pred_valid = np.concatenate(all_val_preds, axis=0)
        # Reshape binary classification outputs
        if loss_fn_name == 'bce' and y_pred_valid.ndim == 2 and y_pred_valid.shape[1] == 1:
            y_pred_valid = y_pred_valid.flatten() 
        # Reshape regression outputs
        elif is_regression and y_pred_valid.ndim == 2 and y_pred_valid.shape[1] == 1:
             y_pred_valid = y_pred_valid.flatten()

    # Generate test predictions
    if X_test is not None:
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        test_dataset = TensorDataset(X_test_tensor) # No labels needed
        test_loader = DataLoader(test_dataset, batch_size=batch_size * 2)
        
        all_test_preds = []
        with torch.no_grad():
            for batch_X_tuple in test_loader: # DataLoader yields tuples/lists
                batch_X_tensor = batch_X_tuple[0].to(device) # Get the tensor
                # Ensure the input tensor is 2D, even if batch size is 1
                if batch_X_tensor.ndim == 1:
                     # This case should ideally not happen if X_test has > 1 feature
                     # unless the DataLoader somehow drops dimensions for bs=1.
                     # Let's ensure it's treated as (1, num_features)
                     batch_X_tensor = batch_X_tensor.view(1, -1) 
                     
                outputs = model(batch_X_tensor)
                # Apply activation for prediction output if needed
                if loss_fn_name == 'bce': # BCEWithLogitsLoss
                    outputs = torch.sigmoid(outputs)
                elif loss_fn_name == 'ce': # CrossEntropyLoss
                    outputs = torch.softmax(outputs, dim=1)
                # No activation needed for MSE regression outputs
                all_test_preds.append(outputs.cpu().numpy())
        y_pred_test = np.concatenate(all_test_preds, axis=0)
        # Reshape binary classification outputs
        if loss_fn_name == 'bce' and y_pred_test.ndim == 2 and y_pred_test.shape[1] == 1:
            y_pred_test = y_pred_test.flatten()
        # Reshape regression outputs
        elif is_regression and y_pred_test.ndim == 2 and y_pred_test.shape[1] == 1:
             y_pred_test = y_pred_test.flatten()
            
    return best_model_state_dict, y_pred_valid, y_pred_test 