# Function to create N-way interaction features by concatenating and encoding feature values.

"""
Extended Description:
Generates interaction features by combining values from specified categorical or 
discrete features for various interaction orders (e.g., 2-way, 3-way, etc.). 
It typically works best on features with relatively low cardinality or features that
have been binned first.

The interaction is created by concatenating the string representations of the feature values
for each combination. Optionally, it can apply count or label encoding to the generated
interaction strings.

Returns a DataFrame containing only the new interaction features (either raw strings or encoded).
"""

import polars as pl
from typing import List, Optional, Literal, Tuple, Dict, Any
from itertools import combinations

def create_nway_interaction_features(
    df: pl.DataFrame,
    features: List[str],
    n_ways: List[int],  # List of interaction orders (e.g., [2, 3])
    sep: str = '_',
    encode_type: Literal['none', 'count', 'label'] = 'none',
    add_one: bool = False,  # If True and encode_type='label', add 1 to result
    handle_nulls: Literal['ignore', 'propagate'] = 'ignore' # 'ignore' treats null as 'null_str'
) -> pl.DataFrame:
    """Creates N-way interaction features via string concatenation, optionally encodes.

    Args:
        df (pl.DataFrame): Input DataFrame.
        features (List[str]): List of feature columns to interact. Assumed to be
            categorical or string-like.
        n_ways (List[int]): The orders of interaction (e.g., [2, 3] for pairs and triplets).
        sep (str, optional): Separator used for concatenating feature values. Defaults to '_'.
        encode_type (Literal['none', 'count', 'label'], optional):
            Type of encoding to apply to the generated interaction strings.
            - 'none': Return raw concatenated strings.
            - 'count': Apply count encoding based on frequency in the input `df`.
            - 'label': Apply label encoding (map unique strings to integers).
            Defaults to 'none'.
        add_one (bool, optional): If True and encode_type is 'label', adds 1 to the
            label encoded integer result (0-based -> 1-based). Defaults to False.
        handle_nulls (Literal['ignore', 'propagate'], optional): How to handle nulls.
            'ignore': Treat nulls as a string category 'null_str' during concatenation.
            'propagate': If any feature in a combination is null, the interaction is null.
            Defaults to 'ignore'.

    Returns:
        pl.DataFrame: DataFrame containing only the new interaction features.
                      Output type depends on encode_type (Utf8, UInt32, or UInt32/Int64).

    Raises:
        ValueError: If features are not found, n_ways is invalid, or encode_type is unknown.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError("df must be a Polars DataFrame.")
    if not features:
        raise ValueError("Feature list cannot be empty.")
    if not n_ways or not all(isinstance(n, int) and n >= 2 for n in n_ways):
        raise ValueError("n_ways must be a list of integers >= 2.")
    if encode_type not in ['none', 'count', 'label']:
        raise ValueError(f"Unknown encode_type: {encode_type}")
    if not all(f in df.columns for f in features):
        missing = [f for f in features if f not in df.columns]
        raise ValueError(f"Features not found in df: {missing}")

    all_interaction_exprs = {}
    all_new_col_names = []

    # Generate expressions for all combinations and N-ways first
    for n in n_ways:
        for combo in combinations(features, n):
            interaction_col_name = sep.join(combo) + f'_inter{n}'
            all_new_col_names.append(interaction_col_name)

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
            all_interaction_exprs[interaction_col_name] = interaction_expr

    if not all_interaction_exprs:
        return pl.DataFrame() # Return empty DataFrame if no combinations

    # --- Calculate interaction strings and apply encoding --- 
    if encode_type == 'none':
        # Just select the concatenated string columns
        df_interactions = df.select(list(all_interaction_exprs.values()))

    else:
        # Calculate interaction strings first
        df_strings = df.select(list(all_interaction_exprs.values()))
        
        final_encoded_exprs = []
        for col_name in all_new_col_names:
            if encode_type == 'count':
                encoded_col_name = f"{col_name}_count"
                count_expr = pl.count().over(col_name).cast(pl.UInt32).alias(encoded_col_name)
                final_encoded_exprs.append(count_expr)
                
            elif encode_type == 'label':
                encoded_col_name = f"{col_name}_label"
                # Polars native label encoding: use rank('dense') over the string column
                label_expr = pl.col(col_name).rank("dense", descending=False).cast(pl.UInt32) - 1 # rank is 1-based, make it 0-based
                if add_one:
                    label_expr = label_expr + 1
                final_encoded_exprs.append(label_expr.alias(encoded_col_name))
        
        if final_encoded_exprs:
            # Add encoded columns and select only them
            df_interactions = df_strings.with_columns(final_encoded_exprs)
            final_col_names = [expr.meta.output_name() for expr in final_encoded_exprs]
            df_interactions = df_interactions.select(final_col_names)
        else:
             df_interactions = pl.DataFrame()

    return df_interactions 