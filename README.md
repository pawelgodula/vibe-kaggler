# Vibe Kaggler Project

## Overview

The Vibe Kaggler project is a repository template for participating in Kaggle competitions using a pure functions and data structures approach in Python. It enforces code reuse, maintainability, and compatibility with LLM-based workflows.

## Assumptions and Conventions

1. **Python and Type Annotations**  
   All code is written in Python 3 with full use of type hints to ensure static typing and improved code clarity.

2. **Directory Structure**  
   We maintain two top-level directories:  
   - `utils/`: Contains reusable utility functions and modules.  
   - `competition/`: Contains data, code, and files specific to each Kaggle competition.

3. **Standalone Utilities Repository**  
   The `utils/` folder is designed to be extracted into a standalone repository in the future. This will allow sharing common utilities across multiple competition repositories and avoid code duplication.

4. **Pure Functions and Data Structures**  
   All computations and transformations are implemented as pure functions operating on immutable data structures. This approach enhances testability and reproducibility.

5. **File Organization for LLM Compatibility**  
   Each function or class resides in its own file following a consistent structure to facilitate parsing and understanding by large language models:  
   - **One-line description**: A brief summary of its purpose.  
   - **Extended description**: One paragraph elaborating on the functionality and usage.  
   - **Definition**: The function or class implementation with full type annotations.  
   - **Tests**: Unit tests validating the behavior.

## Directory Layout

```plaintext
.
├── README.md
├── utils/
│   └── <utility_module>.py
└── competition/
    └── <competition_name>/
        ├── data/
        ├── notebooks/
        ├── scripts/
        └── README.md
```

## Getting Started

1. Clone the repository.  
2. (Optional) Create a virtual environment and install dependencies.  
3. Start by adding or extending utilities under `utils/`.  
4. Create a new directory under `competition/` for your specific competition.  
5. Write pure functions and corresponding tests as per the file organization guidelines. 

## Typical Kaggle Tabular Competition Workflow

This outlines a general process for approaching a typical tabular data competition on Kaggle using this project structure:

1.  **Data Acquisition & Initial Inspection:**
    *   Download competition data into the `competition/<competition_name>/data/raw/` directory.
    *   Generate small sample files (e.g., using `head -n 5 > sample_<file_name>.csv`) and save them in `competition/<competition_name>/data/samples/` for quick inspection (useful for LLMs).

2.  **Exploratory Data Analysis (EDA):**
    *   Use notebooks in `competition/<competition_name>/notebooks/` for exploration.
    *   Analyze data distributions, correlations, missing values, and potential feature engineering ideas.
    *   Document findings and visualizations. Consider generating an HTML report from the EDA notebook (e.g., using `nbconvert`).

3.  **Cross-Validation Setup:**
    *   Define a robust cross-validation (CV) strategy using `utils/get_cv_splitter.py`.
    *   Choose an appropriate CV type (KFold, StratifiedKFold, GroupKFold, etc.) based on the data and evaluation metric.
    *   Save the fold assignments or the splitter configuration for consistent evaluation.

4.  **Baseline Pipeline & Submission:**
    *   Implement a minimal end-to-end pipeline: load data, minimal preprocessing, train a simple model (e.g., Logistic Regression, basic LGBM), generate predictions.
    *   Use the defined CV strategy to evaluate the baseline locally.
    *   Generate a submission file using `utils/generate_submission_file.py` and submit to Kaggle to establish a Leaderboard (LB) baseline score.

5.  **Initial Modeling Experiments:**
    *   Conduct experiments with common baseline models and techniques:
        *   Train LightGBM (LGBM) on a single fold.
        *   Train LGBM using all CV folds.
        *   Add basic categorical feature encoding (e.g., using `utils/encode_categorical_features.py`).
        *   Train XGBoost using all CV folds.
        *   Create a simple average ensemble of the multi-fold LGBM and XGBoost models.
    *   Track CV scores and corresponding LB scores for each experiment.

6.  **Determine Minimum Viable Data Size (Optional but Recommended):**
    *   Experiment with training on progressively smaller random samples of the training data (e.g., 10%, 25%, 50%, 75%).
    *   Re-run key experiments (like multi-fold LGBM) on these samples.
    *   Identify the smallest data size where the CV scores maintain a reasonable correlation with scores on the full dataset and/or LB scores. This helps iterate faster by reducing experiment cost.

7.  **Feature Engineering:**
    *   Based on EDA and initial results, implement more advanced feature engineering using functions in `utils/` or competition-specific scripts in `competition/<competition_name>/scripts/`.
    *   Examples: Polynomial features (`create_polynomial_features`), interaction terms (`create_interaction_features`), aggregations, target encoding.
    *   Evaluate the impact of new features using the established CV setup and potentially the minimum viable data size.

8.  **Model Ensembling/Stacking:**
    *   Experiment with different ensembling techniques:
        *   Weighted averaging.
        *   Stacking (using predictions from base models as features for a meta-model).
    *   Evaluate ensemble performance using CV.

9.  **Final Submission Selection:**
    *   Based on robust CV scores and LB feedback, select the best performing models or ensembles.
    *   Consider factors like model complexity, prediction time, and diversity of selected models.
    *   Generate final submission files.

## Guidelines for Iteration

You are allowed only to use/modify/create functions from utils . if the function does not exist or needs modification - please do so in the utils. Preserve generality - if you need to make changes fto function to make it workf in a new scenario, please make sure already implemented scenarios work too.

## Extended LLM Implementation Guide

This section contains detailed guidelines, templates, and examples for implementing the `utils` and `competition` code according to our conventions. LLMs can use these templates to generate, parse, and validate code consistently.

### 1. Implementation Goals

- Enforce pure functions and immutable data structures.  
- Maintain one responsibility per file.  
- Include comprehensive type annotations and docstrings.  
- Provide unit tests alongside each implementation.

### 2. Utils Module Template

Each utility function or class in `utils/` should follow this structure:

```python
# one-line description

"""
Extended Description:
A paragraph describing the function's purpose, parameters, return values, and any important details.
"""

from typing import ...  # import necessary types

def function_name(
    param1: Type,
    param2: Type,
) -> ReturnType:
    """One-line summary.

    Extended Description:
    Detailed explanation of the function's behavior, edge cases, and usage.
    """
    # implementation here
```

Tests for `function_name` should be placed in a corresponding test file in `tests/utils/`, e.g., `tests/utils/test_function_name.py`:

```python
import unittest
from utils.function_name import function_name

class TestFunctionName(unittest.TestCase):
    def test_basic_case(self):
        input_data = ...
        expected = ...
        self.assertEqual(function_name(input_data), expected)

if __name__ == '__main__':
    unittest.main()
```

### 3. Competition Module Template

For each competition under `competition/<competition_name>/`, follow:

```plaintext
competition/<competition_name>/
├── data/
│   ├── raw/        # unprocessed input files
│   ├── processed/  # cleaned and feature-engineered data
│   └── external/   # third-party or reference datasets
├── notebooks/      # exploratory and analysis notebooks
│   └── exploration.ipynb
├── scripts/        # reusable processing scripts as functions
│   ├── data_processing.py
│   └── feature_engineering.py
└── README.md       # competition-specific description and instructions
```

#### Example script file in `scripts/data_processing.py`:

```python
# one-line description of this module

"""
Extended Description:
Reads raw data, applies cleaning, and outputs a DataFrame for modeling.
"""

import pandas as pd
from typing import DataFrame

def load_and_clean_data(
    file_path: str
) -> DataFrame:
    """Load raw CSV file and perform cleaning.

    Extended Description:
    - Handles missing values.
    - Converts types.
    - Returns a pandas DataFrame.
    """
    df = pd.read_csv(file_path)
    # cleaning steps...
    return df
```

Tests for competition scripts should go in `tests/competition/<competition_name>/`, matching the module path.

### 4. Files and Naming Conventions

- File names in `utils/` and `competition/` must be lowercase with underscores, matching the function/class they contain.  
- Test file names follow `test_<module_name>.py`.  
- Use consistent import paths: `from utils.module_name import function_name`. 