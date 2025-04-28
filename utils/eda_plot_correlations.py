# Functions to plot correlation results.

"""
Extended Description:
Provides functions to visualize correlation matrices calculated by 
`eda_calculate_correlations`. 

- `plot_correlation_heatmap`: Generates a heatmap for the feature-feature 
  correlation matrix.
- `plot_target_correlation_bar`: Generates a bar plot for the correlations 
  between features and the target variable.

Requires `matplotlib` and `seaborn` to be installed.
"""

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
from typing import Optional, Tuple

def plot_correlation_heatmap(
    feature_corr_matrix: pl.DataFrame,
    figsize: Tuple[int, int] = (12, 10),
    title: str = "Feature Correlation Matrix",
    annot: bool = False, # Annotate cells only if matrix is small
    max_size_for_annot: int = 20 # Max features for annotation
) -> str:
    """Generates a heatmap for the feature correlation matrix and returns base64.

    Args:
        feature_corr_matrix (pl.DataFrame): DataFrame containing the feature-feature
            correlation matrix (output from `calculate_correlations`). Assumes the 
            first column is 'feature' name and the rest are correlation values.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 10).
        title (str, optional): Plot title. Defaults to "Feature Correlation Matrix".
        annot (bool, optional): Whether to annotate cells with correlation values. 
            Defaults to False. Annotation is automatically disabled if the number 
            of features exceeds `max_size_for_annot`.
        max_size_for_annot (int, optional): Maximum number of features for enabling 
            cell annotations by default. Defaults to 20.

    Returns:
        str: Base64 encoded string of the generated heatmap image (PNG format).
             Returns an empty string if the input matrix is invalid.
    """
    if feature_corr_matrix is None or feature_corr_matrix.is_empty():
        print("Warning: No feature correlation matrix provided to plot.")
        return ""
        
    # Exclude the 'feature' name column for plotting
    plot_data = feature_corr_matrix.select(pl.all().exclude("feature"))
    feature_names = feature_corr_matrix["feature"].to_list()
    num_features = len(feature_names)
    
    # Convert to numpy for seaborn heatmap
    corr_np = plot_data.to_numpy()

    # Disable annotations if matrix is too large
    should_annot = annot and num_features <= max_size_for_annot

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_np, 
        annot=should_annot, 
        cmap='coolwarm', 
        fmt=".2f", 
        linewidths=.5,
        xticklabels=feature_names,
        yticklabels=feature_names,
        cbar=True
    )
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close() # Close the plot to free memory
    buf.seek(0)

    # Encode buffer to base64 string
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return image_base64

def plot_target_correlation_bar(
    target_corr: pl.DataFrame,
    target_col: str, # Name of the original target column for labeling
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    max_features_to_plot: Optional[int] = 50 # Plot top N features if too many
) -> str:
    """Generates a bar plot for feature-target correlations and returns base64.

    Args:
        target_corr (pl.DataFrame): DataFrame containing feature-target correlations 
            (output from `calculate_correlations`). Expected columns: 'feature' and 
            '<target_col>_correlation'.
        target_col (str): The name of the target column (used for title).
        figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 8).
        title (Optional[str], optional): Plot title. If None, a default is generated.
        max_features_to_plot (Optional[int], optional): Maximum number of features 
            to display (plots top N by absolute correlation). If None, plots all. 
            Defaults to 50.

    Returns:
        str: Base64 encoded string of the generated bar plot image (PNG format).
             Returns an empty string if the input DataFrame is invalid.
    """
    if target_corr is None or target_corr.is_empty():
        print("Warning: No target correlation data provided to plot.")
        return ""
        
    correlation_col_name = f"{target_col}_correlation"
    if correlation_col_name not in target_corr.columns or "feature" not in target_corr.columns:
         print(f"Warning: Target correlation DataFrame missing expected columns ('feature', '{correlation_col_name}').")
         return ""

    # Sort by absolute correlation for plotting top N
    plot_data = target_corr.with_columns(
        pl.col(correlation_col_name).abs().alias("abs_corr")
    ).sort("abs_corr", descending=True)
    
    if max_features_to_plot is not None and len(plot_data) > max_features_to_plot:
        plot_data = plot_data.head(max_features_to_plot)
        plot_title = title if title is not None else f'Top {max_features_to_plot} Features Correlated with {target_col}'
    else:
         plot_title = title if title is not None else f'Feature Correlation with {target_col}'
         
    # Sort again by actual correlation value for plotting order
    plot_data = plot_data.sort(correlation_col_name, descending=True)

    plt.figure(figsize=figsize)
    sns.barplot(x=correlation_col_name, y="feature", data=plot_data.to_pandas(), palette="vlag")
    plt.title(plot_title, fontsize=14)
    plt.xlabel(f'Correlation with {target_col}')
    plt.ylabel('Feature')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close() # Close the plot to free memory
    buf.seek(0)

    # Encode buffer to base64 string
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return image_base64 