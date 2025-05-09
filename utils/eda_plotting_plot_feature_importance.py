#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility function for plotting feature importance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Any # Any for model type

def plot_feature_importance(model: Any, model_name: str, feature_names: List[str], output_path: str) -> None:
    """Generates and saves a feature importance plot.

    Args:
        model (Any): Trained model instance (LightGBM, XGBoost, CatBoost).
        model_name (str): Name of the model type (e.g., 'LightGBM', 'XGBoost', 'CatBoost').
        feature_names (List[str]): List of feature names.
        output_path (str): Path to save the generated plot.
    """
    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1.0)
    sns.set_style('whitegrid')
    
    importance = None
    if model_name == 'LightGBM':
        importance = model.feature_importances_
    elif model_name == 'XGBoost':
        importance = model.feature_importances_
    elif model_name == 'CatBoost':
        importance = model.get_feature_importance()
    else:
        print(f"Warning: Feature importance plotting not specifically implemented for model type '{model_name}'. Attempting generic .feature_importances_.")
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            print(f"Error: Cannot get feature importance for model {model_name}. No .feature_importances_ or specific handler.")
            return

    if importance is None:
        print(f"Error: Importance values were not extracted for model {model_name}.")
        return

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    top_features = importance_df.head(30)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    
    plt.title(f'{model_name} Feature Importance (Top 30 features)', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot for {model_name} saved to {output_path}") 