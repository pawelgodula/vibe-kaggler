# Creates interaction terms for specified pairs of numerical features using Polars.

"""
Extended Description:
This function generates new features by multiplying specified pairs of existing
numerical features in a Polars DataFrame using efficient Polars expressions.
It provides a way to explicitly define interaction terms.
It returns a new DataFrame containing only the generated interaction features.
"""

import polars as pl
from typing import List, Tuple

def create_interaction_features(
    df: pl.DataFrame,
    feature_pairs: List[Tuple[str, str]]
) -> pl.DataFrame:
    """Creates interaction features by multiplying pairs of existing features.

    Args:
        df (pl.DataFrame): DataFrame containing the input features.
        feature_pairs (List[Tuple[str, str]]): A list of tuples, where each tuple
                                               contains two feature names to multiply.

    Returns:
        pl.DataFrame: A new DataFrame containing the generated interaction features.
                      Column names are formatted as 'feature1_x_feature2'.

    Raises:
        pl.exceptions.ColumnNotFoundError: If any specified feature is not found.
        pl.exceptions.ComputeError: If multiplication is attempted on non-numeric types.
        ValueError: If feature_pairs is empty or contains non-pairs.
    """
    if not feature_pairs:
        raise ValueError("feature_pairs list cannot be empty.")

    interaction_expressions = []
    generated_col_names = set()

    # Pre-check features existence and type
    all_features_needed = set()
    for pair in feature_pairs:
        if len(pair) != 2:
            raise ValueError(f"Each element in feature_pairs must be a tuple of length 2. Found: {pair}")
        all_features_needed.add(pair[0])
        all_features_needed.add(pair[1])
    
    missing_features = [f for f in all_features_needed if f not in df.columns]
    if missing_features:
         # Raise error compatible with Polars/KeyError style
         raise pl.exceptions.ColumnNotFoundError(f"Features not found in DataFrame: {missing_features}")

    non_numeric_features = [f for f in all_features_needed if not df[f].dtype.is_numeric()]
    if non_numeric_features:
         raise pl.exceptions.ComputeError(f"Non-numeric features specified for interactions: {non_numeric_features}")

    for feat1, feat2 in feature_pairs:
        # Create name, ensuring consistent order (e.g., A_x_B not B_x_A)
        sorted_pair = tuple(sorted((feat1, feat2)))
        interaction_col_name = f"{sorted_pair[0]}_x_{sorted_pair[1]}"

        if interaction_col_name not in generated_col_names:
            interaction_expressions.append(
                (pl.col(feat1) * pl.col(feat2)).cast(pl.Int32).alias(interaction_col_name)
            )
            generated_col_names.add(interaction_col_name)
        # else: skip duplicate interaction

    if not interaction_expressions:
        # Return empty DataFrame if all pairs were duplicates or list was empty
        return pl.DataFrame()
        
    # Select only the generated interaction columns
    df_interactions = df.select(interaction_expressions)
    
    return df_interactions 