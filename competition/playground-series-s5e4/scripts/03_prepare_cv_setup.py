# Script to generate and save cross-validation fold indices.

"""
Extended Description:
This script loads the training data, defines a cross-validation strategy 
(e.g., StratifiedKFold), and uses the `get_cv_indices` utility to generate 
train and validation index arrays for each fold. These index arrays are then 
saved as .npy files to the specified output directory for consistent use in 
later training scripts.

This corresponds to step 3 (CV Setup) in the typical workflow.
"""

import sys
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold # Use KFold for regression

# --- Add project root to sys.path --- 
# Allows importing utils even if script is run directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.load_csv import load_csv
    from utils.get_cv_indices import get_cv_indices
except ImportError as e:
     print(f"Error importing utility functions: {e}")
     print("Ensure utils directory is at the project root.")
     print("Consider running this script from the project root directory.")
     sys.exit(1)

# --- Configuration ---
# Determine paths relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
COMPETITION_DIR = SCRIPT_DIR.parent
RAW_DATA_DIR = COMPETITION_DIR / "data" / "raw"
PROCESSED_DATA_DIR = COMPETITION_DIR / "data" / "processed"
OUTPUT_DIR = PROCESSED_DATA_DIR / "cv_indices"
INPUT_CSV = RAW_DATA_DIR / "train.csv" 

# CV Strategy Configuration
N_FOLDS = 5
RANDOM_STATE = 42
TARGET_COL = "Listening_Time_minutes" # Keep for reference, but not used by KFold
# CV_SPLITTER = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
CV_SPLITTER = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

def main():
    """Main function to load data, generate CV indices, and save them."""
    print("--- Starting CV Index Generation ---")
    print(f"Input data file: {INPUT_CSV}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"CV Strategy: {CV_SPLITTER.__class__.__name__}")
    print(f"Number of folds: {N_FOLDS}")
    # print(f"Target column (for stratification): {TARGET_COL}") # Not used for KFold

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {OUTPUT_DIR}")

    try:
        # Load data (only need target for stratification, or just length otherwise)
        print(f"\nLoading data from {INPUT_CSV}...")
        df_train = load_csv(str(INPUT_CSV))
        print(f"Loaded DataFrame shape: {df_train.shape}")

        # Generate CV indices
        print("\nGenerating CV fold indices...")
        fold_indices = get_cv_indices(
            df=df_train, 
            cv_splitter=CV_SPLITTER 
            # target_col=TARGET_COL # Not needed for KFold
        )
        print(f"Generated indices for {len(fold_indices)} folds.")

        # Save indices
        print("\nSaving fold indices...")
        for i, (train_idx, valid_idx) in enumerate(fold_indices):
            fold_label = f"fold_{i}"
            train_path = OUTPUT_DIR / f"{fold_label}_train.npy"
            valid_path = OUTPUT_DIR / f"{fold_label}_valid.npy"
            
            np.save(train_path, train_idx)
            np.save(valid_path, valid_idx)
            print(f"  Saved {fold_label}: train ({train_idx.shape[0]} rows), valid ({valid_idx.shape[0]} rows)")
            
        print("--- CV Index Generation Complete ---")

    except FileNotFoundError:
        print(f"\nError: Input data file not found at {INPUT_CSV}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 