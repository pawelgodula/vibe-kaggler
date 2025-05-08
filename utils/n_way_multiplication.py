"""
Utility function to create n-way multiplication features.

This function extends the simple cross-term approach by allowing
multiplications of n features at a time (not just pairs).
"""

import itertools
import pandas as pd
import numpy as np


def create_n_way_multiplication_features(df, features, n_way_list=None, max_n_way=2, name_sep="_x_"):
    """
    Create n-way multiplication features from a list of input features.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        features (list): List of feature names to generate multiplications from
        n_way_list (list, optional): List of n values to generate (e.g. [2, 3] for 2-way and 3-way) 
            If None, will use range(2, max_n_way+1)
        max_n_way (int, optional): Maximum number of features to multiply together, default is 2
        name_sep (str, optional): Separator string for feature names
    
    Returns:
        pandas.DataFrame: Dataframe with the original features and added n-way multiplication features
    
    Example:
        # Create 2-way and 3-way multiplication features
        df_new = create_n_way_multiplication_features(df, ['A', 'B', 'C'], max_n_way=3)
        # This will add features: A_x_B, A_x_C, B_x_C, A_x_B_x_C
    """
    df_new = df.copy()
    
    # Determine which n values to generate
    if n_way_list is None:
        n_way_list = list(range(2, max_n_way + 1))
    
    # Ensure n_way_list contains valid values
    n_way_list = [n for n in n_way_list if n >= 2 and n <= len(features)]
    
    # Get valid features from the dataframe
    valid_features = [f for f in features if f in df.columns]
    
    # Generate features for each n value
    created_features = []
    
    for n in n_way_list:
        # Generate all combinations of n features
        feature_combinations = list(itertools.combinations(valid_features, n))
        
        for combo in feature_combinations:
            # Create feature name
            feature_name = name_sep.join(combo)
            
            # Calculate the multiplication
            product = df_new[combo[0]].copy()
            for feature in combo[1:]:
                product = product * df_new[feature]
            
            # Add the new feature
            df_new[feature_name] = product
            created_features.append(feature_name)
    
    return df_new, created_features 