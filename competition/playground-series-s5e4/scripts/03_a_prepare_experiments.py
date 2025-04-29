# Script to initialize the experiments tracker CSV.

"""
Extended Description:
This script creates the initial `experiments.csv` file in the competition's 
root directory. It populates the CSV with the first set of planned experiments 
as outlined in the main README.md (Step 5: Initial Modeling Experiments). 
The structure follows the fields defined in the `utils.data_structures.experiment.Experiment` 
dataclass.

This corresponds to step 3a in the workflow, preparing for the initial modeling runs.
"""

import pandas as pd
from pathlib import Path

# --- Configuration ---
# Determine paths relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
COMPETITION_DIR = SCRIPT_DIR.parent
OUTPUT_CSV = COMPETITION_DIR / "experiments.csv" 

# Define columns based on Experiment dataclass
# (script_file_name, why, what, how, cv_score=None, lb_score=None, status="planned")
EXPERIMENT_COLUMNS = [
    "script_file_name", 
    "why", 
    "what", 
    "how", 
    "cv_score", 
    "lb_score", 
    "status"
]

# Define initial experiments based on README Step 5
initial_experiments = [
    {
        "script_file_name": "03_b_train_lgbm_single_fold.py",
        "why": "Establish a very basic baseline performance.",
        "what": "Train LightGBM on a single fold of the data.",
        "how": "Use train_single_fold.py utility with a default LGBM configuration. No special feature engineering.",
        "cv_score": None,
        "lb_score": None,
        "status": "planned"
    },
    {
        "script_file_name": "03_c_train_lgbm_cv.py",
        "why": "Get a more robust performance estimate using cross-validation.",
        "what": "Train LightGBM using all CV folds.",
        "how": "Use train_and_cv.py utility with default LGBM configuration.",
        "cv_score": None,
        "lb_score": None,
        "status": "planned"
    },
    {
        "script_file_name": "03_d_train_lgbm_cv_catencode.py",
        "why": "Assess the impact of basic categorical feature handling.",
        "what": "Train LightGBM using CV folds with basic categorical encoding.",
        "how": "Modify 03_c script/config to include encode_categorical_features.py (e.g., OneHotEncoding) before training.",
        "cv_score": None,
        "lb_score": None,
        "status": "planned"
    },
    {
        "script_file_name": "03_e_train_xgb_cv.py",
        "why": "Evaluate an alternative gradient boosting model.",
        "what": "Train XGBoost using all CV folds.",
        "how": "Use train_and_cv.py utility with default XGBoost configuration.",
        "cv_score": None,
        "lb_score": None,
        "status": "planned"
    },
    {
        "script_file_name": "03_f_ensemble_avg.py",
        "why": "Check if a simple ensemble improves over individual models.",
        "what": "Create an average ensemble of multi-fold LGBM and XGBoost predictions.",
        "how": "Use average_predictions.py utility on the OOF/test predictions from experiments 2 (03_c) and 4 (03_e).",
        "cv_score": None,
        "lb_score": None,
        "status": "planned"
    }
]

def main():
    """Main function to create and save the experiments DataFrame."""
    print("--- Initializing Experiments Tracker ---")
    
    # Create DataFrame
    try:
        experiments_df = pd.DataFrame(initial_experiments, columns=EXPERIMENT_COLUMNS)
        print(f"Created DataFrame with {len(experiments_df)} initial experiments.")

        # Save to CSV
        print(f"Saving experiments to: {OUTPUT_CSV}")
        experiments_df.to_csv(OUTPUT_CSV, index=False)
        print("--- Experiments Tracker Initialized Successfully ---")

    except Exception as e:
        print(f"\nError creating or saving experiments CSV: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 