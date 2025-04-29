# Function to plot the summary analysis of target grouped by a feature.

"""
Extended Description:
Generates plots based on the summary DataFrame created by 
`analyze_target_by_feature`. It typically shows the relationship between the 
feature values (categories or bins) and the target's mean or median, often 
accompanied by the count in each group.

Requires `matplotlib` and `seaborn` to be installed.
"""

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Optional, Tuple, Literal

def plot_feature_summary(
    summary_df: pl.DataFrame,
    feature_col: str,
    target_col: str,
    plot_metric: Literal['mean', 'median'] = 'mean',
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    include_count: bool = True
) -> str:
    """Generates plots for target summary grouped by feature and returns base64.

    Plots the specified metric (mean or median) of the target against feature values/bins.
    Optionally includes the count as bar height or a secondary plot.

    Args:
        summary_df (pl.DataFrame): Summary DataFrame from `analyze_target_by_feature`.
            Expected columns: 'feature_value', 'count', 'target_mean', 'target_median'.
        feature_col (str): The name of the original feature column (used for title/labels).
        target_col (str): The name of the target column (used for title/labels).
        plot_metric (Literal['mean', 'median'], optional): Which metric to plot against feature 
            values. Defaults to 'mean'.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 6).
        title (Optional[str], optional): Plot title. If None, a default is generated.
        include_count (bool, optional): If True, includes count information in the plot 
            (e.g., using a secondary y-axis or varying point size). Defaults to True.

    Returns:
        str: Base64 encoded string of the generated plot image (PNG format).
             Returns an empty string if the input DataFrame is invalid.
    """
    required_cols = ["feature_value", "count", "target_mean", "target_median"]
    if summary_df is None or summary_df.is_empty() or not all(col in summary_df.columns for col in required_cols):
        print(f"Warning: Invalid or empty summary DataFrame provided for feature '{feature_col}'. Cannot plot.")
        return ""

    metric_col = f"target_{plot_metric}"
    if metric_col not in summary_df.columns:
         # This check is somewhat redundant with the required_cols check, but explicit.
         print(f"Warning: Metric column '{metric_col}' not found in summary DataFrame for feature '{feature_col}'. Cannot plot.")
         return ""
         
    # Convert feature_value to string for consistent plotting, especially for bins
    plot_data = summary_df.with_columns(pl.col("feature_value").cast(pl.Utf8)).to_pandas()

    plot_title = title if title is not None else f'{plot_metric.capitalize()} of {target_col} by {feature_col}'
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Primary axis: Plot the chosen metric (mean or median)
    color_metric = 'tab:red'
    sns.lineplot(x='feature_value', y=metric_col, data=plot_data, marker='o', ax=ax1, color=color_metric, sort=False, label=f'{plot_metric.capitalize()} Target')
    ax1.set_xlabel(f'{feature_col} (Value/Bin)')
    ax1.set_ylabel(f'{plot_metric.capitalize()} {target_col}', color=color_metric)
    ax1.tick_params(axis='y', labelcolor=color_metric)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    if include_count:
        # Secondary axis: Plot the count
        ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
        color_count = 'tab:blue'
        sns.barplot(x='feature_value', y='count', data=plot_data, ax=ax2, alpha=0.5, color=color_count, label='Count')
        ax2.set_ylabel('Count', color=color_count)  
        ax2.tick_params(axis='y', labelcolor=color_count)
        ax2.grid(False) # Turn off grid for secondary axis bars
        # Align bars with line plot points if needed (might require adjustments)
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    else:
        ax1.legend(loc="best")

    plt.title(plot_title, fontsize=14)
    fig.tight_layout()  # Adjust layout to prevent overlap

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig) # Close the specific figure
    buf.seek(0)

    # Encode buffer to base64 string
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return image_base64 