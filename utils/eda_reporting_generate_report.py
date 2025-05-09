# Function to generate a comprehensive EDA HTML report.

"""
Extended Description:
Orchestrates various EDA utility functions to perform analysis and generate 
a self-contained HTML report summarizing the data. The report includes:
- Target variable distribution plot.
- Feature-feature correlation heatmap (for numerical features).
- Feature-target correlation bar plot (if target is numeric).
- Individual analysis for specified features (or all suitable ones):
    - Summary table (count, target mean/median) grouped by feature value/bin.
    - Plot showing target mean/median vs. feature value/bin.

Requires `polars`, `matplotlib`, `seaborn`.
"""

import polars as pl
from typing import List, Optional, Dict
import datetime

# Import other EDA utils
from .eda_plot_target_distribution import plot_target_distribution
from .eda_calculate_correlations import calculate_correlations
from .eda_plot_correlations import plot_correlation_heatmap, plot_target_correlation_bar
from .eda_analyze_target_by_feature import analyze_target_by_feature
from .eda_plot_feature_summary import plot_feature_summary
# Import the new summary function
from .eda_calculate_feature_summary import calculate_feature_summary

# Basic HTML Template elements
HTML_TEMPLATE_START = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #eee; border-radius: 5px; background-color: #f9f9f9; }}
        .plot-container {{ text-align: center; margin-bottom: 20px; }}
        img {{ max-width: 90%; height: auto; border: 1px solid #ddd; }}
        table {{ border-collapse: collapse; width: auto; margin: 20px auto; border: 1px solid #ccc; }}
        th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        pre {{ background-color: #eee; padding: 10px; border-radius: 3px; white-space: pre-wrap; word-wrap: break-word; }}
        details > summary {{ cursor: pointer; font-weight: bold; margin-bottom: 10px; }}
        details > summary::marker {{ content: '\25B6  '; /* Right-pointing triangle */ }}
        details[open] > summary::marker {{ content: '\25BC  '; /* Down-pointing triangle */ }}
    </style>
</head>
<body>
<h1>{title}</h1>
<p>Report generated on: {generation_date}</p>
"""

HTML_SECTION_START = """
<div class="section">
<h2>{section_title}</h2>
"""

HTML_PLOT = """
<div class="plot-container">
    <h3>{plot_title}</h3>
    <img src="data:image/png;base64,{image_base64}" alt="{plot_title}">
</div>
"""

HTML_TABLE = """
<div>
    <h3>{table_title}</h3>
    {table_html}
</div>
"""

HTML_SECTION_END = """
</div>
"""

HTML_TEMPLATE_END = """
</body>
</html>
"""

def _df_to_html(df: Optional[pl.DataFrame], max_rows: int = 50) -> str:
    """Converts a Polars DataFrame to an HTML table string, handling None."""
    if df is None or df.is_empty():
        return "<p>No data available.</p>"
    # Use pandas for robust HTML conversion
    df_pd = df.head(max_rows).to_pandas()
    html = df_pd.to_html(index=False, border=0, classes=['dataframe'])
    if len(df) > max_rows:
        html += f"<p><i>(Showing top {max_rows} rows)</i></p>"
    return html

def generate_eda_report(
    df: pl.DataFrame,
    target_col: str,
    output_html_path: str,
    report_title: str = "Exploratory Data Analysis Report",
    features_to_analyze: Optional[List[str]] = None, # If None, analyze all suitable
    max_categories_per_feature: int = 20,
    num_bins_per_numerical: int = 10,
    max_corr_features_plot: int = 50
):
    """Generates a comprehensive EDA HTML report.

    Args:
        df (pl.DataFrame): Input DataFrame.
        target_col (str): Name of the target column.
        output_html_path (str): Path to save the generated HTML report.
        report_title (str, optional): Title for the HTML report. 
            Defaults to "Exploratory Data Analysis Report".
        features_to_analyze (Optional[List[str]], optional): Specific features to include 
            in the individual analysis section. If None, attempts to analyze all 
            non-target columns suitable for analysis (numeric target required for 
            mean/median analysis). Defaults to None.
        max_categories_per_feature (int, optional): Max categories for categorical 
            feature analysis. Defaults to 20.
        num_bins_per_numerical (int, optional): Number of bins for numerical 
            feature analysis. Defaults to 10.
        max_corr_features_plot (int, optional): Max features to show in the 
            target correlation bar plot. Defaults to 50.
    """
    print(f"Starting EDA report generation for target '{target_col}'...")
    html_content: List[str] = []
    generation_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- HTML Header ---
    html_content.append(HTML_TEMPLATE_START.format(title=report_title, generation_date=generation_date))

    # --- 0. Data Summary --- (New Section)
    print("Calculating data summary...")
    html_content.append(HTML_SECTION_START.format(section_title="0. Data Summary & Overview"))
    try:
        df_summary = calculate_feature_summary(df)
        html_content.append(HTML_TABLE.format(
            table_title="Feature Summary Statistics",
            table_html=_df_to_html(df_summary, max_rows=len(df.columns) + 5) # Show all columns
        ))
    except Exception as e:
        html_content.append(f"<p>Error generating data summary table: {e}</p>")
    html_content.append(HTML_SECTION_END)

    # --- 1. Target Distribution ---
    print("Analyzing target distribution...")
    html_content.append(HTML_SECTION_START.format(section_title="1. Target Variable Distribution"))
    try:
        target_dist_b64 = plot_target_distribution(df, target_col)
        if target_dist_b64:
            html_content.append(HTML_PLOT.format(
                plot_title=f"Distribution of {target_col}", 
                image_base64=target_dist_b64
            ))
        else:
            html_content.append("<p>Could not generate target distribution plot (target might be non-numeric or all nulls).</p>")
    except Exception as e:
        html_content.append(f"<p>Error generating target distribution plot: {e}</p>")
    html_content.append(HTML_SECTION_END)

    # --- 2. Correlations --- 
    print("Analyzing correlations...")
    html_content.append(HTML_SECTION_START.format(section_title="2. Correlation Analysis"))
    try:
        feature_corr_matrix, target_corr = calculate_correlations(df, target_col)
        
        # Feature-Feature Heatmap
        if feature_corr_matrix is not None:
            heatmap_b64 = plot_correlation_heatmap(feature_corr_matrix, max_size_for_annot=max_corr_features_plot)
            if heatmap_b64:
                html_content.append(HTML_PLOT.format(
                    plot_title="Feature Correlation Matrix", 
                    image_base64=heatmap_b64
                ))
            html_content.append(HTML_TABLE.format(
                 table_title="Feature Correlation Matrix (Table)",
                 table_html=_df_to_html(feature_corr_matrix)
            ))
        else:
             html_content.append("<p>Could not generate feature correlation heatmap (not enough numeric features?).</p>")
             
        # Feature-Target Bar Plot & Table
        if target_corr is not None:
            target_bar_b64 = plot_target_correlation_bar(
                target_corr, target_col, max_features_to_plot=max_corr_features_plot
            )
            if target_bar_b64:
                 html_content.append(HTML_PLOT.format(
                     plot_title=f"Feature Correlation with {target_col}", 
                     image_base64=target_bar_b64
                 ))
            html_content.append(HTML_TABLE.format(
                 table_title=f"Feature Correlation with {target_col} (Table)",
                 table_html=_df_to_html(target_corr)
            ))
        else:
             html_content.append(f"<p>Could not generate feature-target correlations (target '{target_col}' might not be numeric).</p>")
             
    except Exception as e:
        html_content.append(f"<p>Error during correlation analysis: {e}</p>")
    html_content.append(HTML_SECTION_END)

    # --- 3. Individual Feature Analysis --- 
    print("Analyzing individual features...")
    html_content.append(HTML_SECTION_START.format(section_title="3. Individual Feature Analysis"))
    
    if features_to_analyze is None:
        features_to_analyze = [col for col in df.columns if col != target_col]
        print(f"  Analyzing all {len(features_to_analyze)} non-target features.")
    else:
        # Validate provided features
        valid_features = []
        for f in features_to_analyze:
            if f not in df.columns:
                print(f"  Warning: Specified feature '{f}' not found in DataFrame. Skipping.")
            elif f == target_col:
                print(f"  Warning: Skipping target column '{f}' specified in features_to_analyze.")
            else:
                valid_features.append(f)
        features_to_analyze = valid_features
        print(f"  Analyzing specified features: {features_to_analyze}")

    if not df[target_col].dtype.is_numeric():
         html_content.append(f"<p>Target column '{target_col}' is not numeric. Skipping detailed feature analysis requiring target mean/median.</p>")
    elif not features_to_analyze:
         html_content.append("<p>No features selected for individual analysis.</p>")
    else:
        for feature in features_to_analyze:
            print(f"  Analyzing feature: {feature}...")
            html_content.append(f"<details><summary>{feature}</summary>")
            html_content.append(f"<div><h3>Analysis for Feature: {feature}</h3>")
            try:
                summary_df = analyze_target_by_feature(
                    df, 
                    target_col=target_col, 
                    feature_col=feature,
                    max_categories=max_categories_per_feature,
                    num_bins=num_bins_per_numerical
                )
                
                if summary_df is not None:
                    # Plot the summary (mean target vs feature value/bin)
                    summary_plot_b64 = plot_feature_summary(summary_df, feature, target_col, plot_metric='mean')
                    if summary_plot_b64:
                        html_content.append(HTML_PLOT.format(
                            plot_title=f"Mean {target_col} vs {feature}", 
                            image_base64=summary_plot_b64
                        ))
                    else:
                        html_content.append(f"<p>Could not generate plot for feature '{feature}'.</p>")
                    
                    # Display summary table
                    html_content.append(HTML_TABLE.format(
                        table_title=f"Summary Statistics for {target_col} by {feature}",
                        table_html=_df_to_html(summary_df)
                    ))
                else:
                    html_content.append(f"<p>Analysis for feature '{feature}' was skipped or returned no data (e.g., high cardinality numerical, all nulls).</p>")
                    
            except Exception as e:
                html_content.append(f"<p>Error during analysis of feature '{feature}': {e}</p>")
            finally:
                 html_content.append("</div></details>") # Close details for the feature
                 
    html_content.append(HTML_SECTION_END) # Close Individual Feature Analysis section

    # --- HTML Footer ---
    html_content.append(HTML_TEMPLATE_END)

    # Write to file
    try:
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(html_content))
        print(f"EDA report successfully generated: {output_html_path}")
    except IOError as e:
        print(f"Error writing HTML report to {output_html_path}: {e}")


if __name__ == '__main__':
    # Example Usage (assumes other EDA utils are in the same directory or PYTHONPATH)
    print("Running EDA Report Generation Example...")
    # Create dummy data
    n_rows = 200
    data = {
        'id': list(range(n_rows)),
        'numerical_feat1': np.random.rand(n_rows) * 100,
        'numerical_feat2': np.random.randn(n_rows) * 50 + 20,
        'categorical_feat1': np.random.choice(['A', 'B', 'C', 'D', 'E'], size=n_rows),
        'categorical_feat2': np.random.choice(['X', 'Y', 'Z', None, 'X', 'Y'], size=n_rows),
        'target_numeric': np.random.rand(n_rows) * 10 + np.random.randn(n_rows),
        'target_binary': np.random.randint(0, 2, size=n_rows)
    }
    sample_df = pl.DataFrame(data)
    
    # Add some nulls to features
    sample_df = sample_df.with_columns([
        pl.when(pl.col("numerical_feat1") < 10).then(None).otherwise(pl.col("numerical_feat1")).alias("numerical_feat1"),
        pl.when(pl.col("categorical_feat1") == 'A').then(None).otherwise(pl.col("categorical_feat1")).alias("categorical_feat1")
    ])

    output_path = "./eda_report_example.html"
    
    # Generate report for a numeric target
    generate_eda_report(
        df=sample_df, 
        target_col='target_numeric', 
        output_html_path=output_path,
        report_title="Sample EDA Report (Numeric Target)",
        features_to_analyze=['numerical_feat1', 'categorical_feat1', 'categorical_feat2', 'id'] # Include 'id' to test high cardinality skip
    )
    print(f"Example report generated at {output_path}")

    # Generate report for a binary target (some analyses might be less informative or skipped)
    output_path_binary = "./eda_report_example_binary_target.html"
    generate_eda_report(
        df=sample_df, 
        target_col='target_binary', 
        output_html_path=output_path_binary,
        report_title="Sample EDA Report (Binary Target)"
        # features_to_analyze=None # Analyze all
    )
    print(f"Example report for binary target generated at {output_path_binary}") 