# Script to generate sample data files for the competition.

"""
Extended Description:
This script uses the `create_data_samples` utility to generate smaller 
sample files (containing the first N rows) from the raw data files located 
in the competition\'s `data/raw/` directory. The generated samples are saved 
to the `data/samples/` directory.

This corresponds to step 1.b in the typical workflow outlined in the main README.
"""

import sys
import os
from pathlib import Path

# Removed sys.path manipulation as this script should be run as a module

# Import the utility function
# This import should work when run as a module from the project root
try:
    from utils.create_data_samples import create_data_samples
except ImportError as e:
    print(f"Error importing utility function: {e}")
    print("Make sure the utils directory exists at the project root and you are running this script as a module from the root.")
    print("Example: python -m competition.playground-series-s5e4.scripts.01_create_data_samples")
    sys.exit(1)

# --- Configuration ---
# Determine paths relative to this script's location when run as part of a package
SCRIPT_DIR = Path(__file__).resolve().parent
COMPETITION_DIR = SCRIPT_DIR.parent 
RAW_DATA_DIR = COMPETITION_DIR / "data" / "raw"
SAMPLES_DIR = COMPETITION_DIR / "data" / "samples"
N_ROWS = 10 # Number of rows for each sample

def main():
    """Main function to execute the data sample creation."""
    print("--- Starting Data Sample Creation ---")
    print(f"Raw data directory: {RAW_DATA_DIR}")
    print(f"Samples directory: {SAMPLES_DIR}")
    print(f"Number of rows per sample: {N_ROWS}")

    try:
        create_data_samples(
            raw_data_dir=str(RAW_DATA_DIR),
            samples_dir=str(SAMPLES_DIR),
            n_rows=N_ROWS
        )
        print("--- Data Sample Creation Complete ---")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the raw data directory exists and contains data files.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 