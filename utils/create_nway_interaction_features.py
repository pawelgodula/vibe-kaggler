# Function to create N-way interaction features by concatenating feature values.

"""
Extended Description:
Generates interaction features by combining values from specified categorical or 
discrete features. It typically works best on features with relatively low cardinality 
or features that have been binned first (e.g., using `bin_and_encode_numerical_features`).

The interaction is created by concatenating the string representations of the feature values
for each combination. Optionally, it can apply count encoding to the generated interaction strings.

Returns a DataFrame containing only the new interaction features (either raw strings or count encoded).
"""

import polars as pl
from typing import List, Optional, Literal, Tuple, Dict, Any
from itertools import combinations

def create_nway_interaction_features(
    df: pl.DataFrame,
    features: List[str],
    n: Literal[2, 3], # Start with support for 2 and 3-way interactions
    sep: str = '_',
    count_encode: bool = False,
    handle_nulls: Literal['ignore', 'propagate'] = 'ignore' # 'ignore' treats null as a distinct category 'null_str'
) -> Tuple[pl.DataFrame, Optional[Dict[str, Any]]]:
    """Creates N-way interaction features via string concatenation, optionally count encodes.

    Args:
        df (pl.DataFrame): Input DataFrame.
        features (List[str]): List of feature columns to interact. Assumed to be
            categorical or string-like.
        n (Literal[2, 3]): The order of interaction (2 for pairs, 3 for triplets).
        sep (str, optional): Separator used for concatenating feature values. Defaults to '_'.
        count_encode (bool, optional): If True, apply count encoding to the generated
            interaction strings based on their frequency in the input `df`. Defaults to False.
        handle_nulls (Literal['ignore', 'propagate'], optional): How to handle nulls.
            'ignore': Treat nulls as a string category 'null_str' during concatenation.
            'propagate': If any feature in a combination is null, the interaction is null.
            Defaults to 'ignore'.

    Returns:
        Tuple[pl.DataFrame, Optional[Dict[str, Any]]]:
            - df_interactions: DataFrame containing only the new interaction features.
                               If count_encode=True, these are count values (UInt32). 
                               Otherwise, they are the concatenated string values.
            - fitted_params: If count_encode=True, a dictionary mapping each interaction
                             column name to its value->count mapping dictionary. Otherwise None.

    Raises:
        ValueError: If features are not found, n is not 2 or 3, or issues occur.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError("df must be a Polars DataFrame.")
    if not features:
        raise ValueError("Feature list cannot be empty.")
    if n not in [2, 3]:
        raise ValueError(f"Interaction order n must be 2 or 3, got {n}.")
    if not all(f in df.columns for f in features):
        missing = [f for f in features if f not in df.columns]
        raise ValueError(f"Features not found in df: {missing}")

    interaction_exprs = []
    new_col_names = []
    
    for combo in combinations(features, n):
        interaction_col_name = sep.join(combo) + f'_inter{n}'
        new_col_names.append(interaction_col_name)
        
        # Prepare list of expressions for concatenation
        concat_cols = []
        for col_name in combo:
            if handle_nulls == 'ignore':
                # Treat null as a specific string category
                concat_cols.append(pl.col(col_name).fill_null("null_str").cast(pl.Utf8))
            else: # handle_nulls == 'propagate'
                # If any col is null, the concat result will be null
                concat_cols.append(pl.col(col_name).cast(pl.Utf8))
                
        # Create concatenation expression
        interaction_expr = pl.concat_str(concat_cols, separator=sep).alias(interaction_col_name)
        interaction_exprs.append(interaction_expr)

    if not interaction_exprs:
        return pl.DataFrame(), None # Return empty DataFrame if no combinations

    # Calculate all interaction strings first
    df_interactions = df.select(interaction_exprs)
    
    fitted_params = None
    # --- Optional Count Encoding ---    
    if count_encode:
        count_encoded_exprs = []
        # We don't really have fitted params for simple count encoding this way
        # If mapping needed later (e.g. on test set without re-calculating), 
        # one would need to calculate and store the map separately.
        fitted_params = None # No fitted map returned in this simple implementation
        
        for col_name in new_col_names:
            # Calculate counts using a window function over the interaction column
            count_expr = pl.count().over(col_name).cast(pl.UInt32).alias(f"{col_name}_count")
            count_encoded_exprs.append(count_expr)
            
        if count_encoded_exprs:
            # Apply count encoding and select only the count columns
            # Need to add the count columns and then select them
            df_interactions = df_interactions.with_columns(count_encoded_exprs)
            final_count_cols = [expr.meta.output_name() for expr in count_encoded_exprs]
            df_interactions = df_interactions.select(final_count_cols)
        else:
            df_interactions = pl.DataFrame() # Should not happen if new_col_names is populated
            
    return df_interactions, fitted_params
