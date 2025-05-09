# Function to plot the distribution of the target variable.

"""
Extended Description:
Generates a histogram and/or a kernel density estimate (KDE) plot for the 
specified target column in a Polars DataFrame. Helps visualize the shape, 
central tendency, and spread of the target variable, which is crucial for 
understanding the prediction task (e.g., regression vs. classification, 
skewness, potential outliers).

Requires `matplotlib` and `seaborn` to be installed.
"""

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Optional, Literal, Tuple


def plot_target_distribution(
    df: pl.DataFrame,
    target_col: str,
    plot_type: Literal['hist', 'kde', 'both'] = 'both',
    bins: Optional[int] = 30,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None
) -> str:
    """Generates a plot of the target variable's distribution and returns it as base64.

    Args:
        df (pl.DataFrame): Input DataFrame containing the target column.
        target_col (str): The name of the target column.
        plot_type (Literal['hist', 'kde', 'both'], optional): Type of plot.
            'hist' for histogram, 'kde' for density plot, 'both' for overlay.
            Defaults to 'both'.
        bins (Optional[int], optional): Number of bins for the histogram. 
            Defaults to 30. Ignored if plot_type is 'kde'.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).
        title (Optional[str], optional): Custom title for the plot. If None, a 
            default title is generated. Defaults to None.

    Returns:
        str: Base64 encoded string of the generated plot image (PNG format).

    Raises:
        pl.exceptions.ColumnNotFoundError: If target_col is not found in the DataFrame.
        TypeError: If the target column is not numeric.
    """
    if target_col not in df.columns:
        raise pl.exceptions.ColumnNotFoundError(f"Target column '{target_col}' not found.")
    if not df[target_col].dtype.is_numeric():
        raise TypeError(f"Target column '{target_col}' must be numeric for distribution plot.")

    # Convert to Pandas Series for seaborn/matplotlib compatibility
    # Handle potential nulls by dropping them for plotting distribution
    target_series_pd = df[target_col].drop_nulls().to_pandas()

    if target_series_pd.empty:
        print(f"Warning: Target column '{target_col}' contains only null values. Cannot plot distribution.")
        # Return an empty placeholder or raise error?
        # For now, return empty string - caller should handle.
        return ""

    plt.figure(figsize=figsize)
    
    plot_title = title if title is not None else f'Distribution of Target: {target_col}'
    plt.title(plot_title, fontsize=14)

    if plot_type == 'hist' or plot_type == 'both':
        sns.histplot(target_series_pd, bins=bins, kde=(plot_type == 'both'), stat="density", alpha=0.6)
    elif plot_type == 'kde':
        sns.kdeplot(target_series_pd, fill=True, alpha=0.6)
    else:
        # Should not happen with Literal, but as fallback
         raise ValueError(f"Invalid plot_type: '{plot_type}'. Must be 'hist', 'kde', or 'both'.")

    plt.xlabel(target_col)
    plt.ylabel('Density')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()  # Close the plot to free memory
    buf.seek(0)

    # Encode buffer to base64 string
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return image_base64 