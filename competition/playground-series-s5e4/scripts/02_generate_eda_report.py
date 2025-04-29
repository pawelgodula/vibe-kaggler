# Script to perform Exploratory Data Analysis (EDA) and generate an HTML report.

"""
Extended Description:
This script loads the training data (typically a sample for quick analysis) 
and utilizes the `generate_eda_report` utility function to create a 
comprehensive HTML report. The report includes target distribution, correlation 
analysis, and individual feature summaries.

This corresponds to step 2 in the typical workflow outlined in the main README.
"""

import sys
import os
from pathlib import Path
import re # Import regex module

# Import necessary utility functions
try:
    from utils.eda_generate_report import generate_eda_report
    from utils.load_csv import load_csv
except ImportError as e:
    # Attempt to adjust path if running script directly
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
        try:
            from utils.eda_generate_report import generate_eda_report
            from utils.load_csv import load_csv
        except ImportError as e:
             print(f"Error importing utility function: {e}")
             print("Make sure the utils directory exists at the project root.")
             print("Consider running this script as a module: python -m competition.playground-series-s5e4.scripts.02_generate_eda_report")
             sys.exit(1)
    else:
        print(f"Error importing utility function: {e}")
        print("Make sure the utils directory exists at the project root.")
        print("Consider running this script as a module: python -m competition.playground-series-s5e4.scripts.02_generate_eda_report")
        sys.exit(1)

# --- Configuration ---
# Determine paths relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
COMPETITION_DIR = SCRIPT_DIR.parent
# Use the full raw data instead of the sample
RAW_DATA_DIR = COMPETITION_DIR / "data" / "raw"
INPUT_CSV = RAW_DATA_DIR / "train.csv" # <-- Changed from SAMPLES_DATA_DIR
REPORTS_DIR = COMPETITION_DIR / "reports" # Save reports here
OUTPUT_HTML = REPORTS_DIR / "eda_report.html"
OUTPUT_HTML_NO_PLOTS = REPORTS_DIR / "eda_report_no_plots.html" # Path for no-plots version
TARGET_COL = "Listening_Time_minutes" # <--- Corrected target column

# Optional: Specify a subset of features for detailed analysis, or None for all
FEATURES_TO_ANALYZE = None 

def main():
    """Main function to load data and generate the EDA report."""
    print("--- Starting EDA Report Generation ---")
    print(f"Input data file: {INPUT_CSV}")
    print(f"Target column: {TARGET_COL}")
    print(f"Output report file: {OUTPUT_HTML}")

    # Create reports directory if it doesn't exist
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Load the data sample
        print(f"\nLoading data sample from {INPUT_CSV}...")
        df_sample = load_csv(str(INPUT_CSV))
        print(f"Loaded DataFrame shape: {df_sample.shape}")

        if TARGET_COL not in df_sample.columns:
             print(f"\nError: Target column '{TARGET_COL}' not found in the loaded data.")
             print(f"Available columns: {df_sample.columns}")
             print("Please update the TARGET_COL variable in the script.")
             sys.exit(1)
             
        # Generate the report
        print(f"\nGenerating EDA report for target '{TARGET_COL}'...")
        generate_eda_report(
            df=df_sample,
            target_col=TARGET_COL,
            output_html_path=str(OUTPUT_HTML),
            report_title=f"EDA Report: {COMPETITION_DIR.name}",
            features_to_analyze=FEATURES_TO_ANALYZE
            # Add other parameters like max_categories_per_feature if needed
        )
        print(f"--- EDA Report Generation Complete ({OUTPUT_HTML}) ---")

        # --- Create version without plots ---
        try:
            print(f"\nCreating report version without plots...")
            with open(OUTPUT_HTML, 'r', encoding='utf-8') as f_in:
                html_content = f_in.read()
            
            # Remove plot containers using regex
            # re.DOTALL makes . match newline characters
            content_no_plots = re.sub(r'<div class="plot-container">.*?</div>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
            
            with open(OUTPUT_HTML_NO_PLOTS, 'w', encoding='utf-8') as f_out:
                f_out.write(content_no_plots)
            print(f"--- No-plots report generated ({OUTPUT_HTML_NO_PLOTS}) ---")
            
        except Exception as e_no_plots:
            print(f"\nWarning: Failed to create report version without plots: {e_no_plots}")
            # Continue even if this step fails

    except FileNotFoundError:
        print(f"\nError: Input data file not found at {INPUT_CSV}")
        print("Please ensure the sample data file exists (e.g., run 01_create_data_samples.py first).")
        sys.exit(1)
    except ImportError as e:
         # Handle potential plotting library import errors within generate_eda_report
         print(f"\nError: An import error occurred, likely missing a plotting library (matplotlib/seaborn): {e}")
         sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 