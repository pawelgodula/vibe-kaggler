#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility function for creating cross-term features.
"""

import pandas as pd
from typing import List

def add_feature_cross_terms(df: pd.DataFrame, numerical_features: List[str]) -> pd.DataFrame:
    """Adds cross-term features (multiplication) between specified numerical features.

    Args:
        df (pd.DataFrame): Input DataFrame.
        numerical_features (List[str]): List of numerical feature names to create cross-terms from.

    Returns:
        pd.DataFrame: DataFrame with added cross-term features.
                      New features are named as 'feature1_x_feature2'.
    """
    df_new = df.copy()
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):
            feature1 = numerical_features[i]
            feature2 = numerical_features[j]
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    return df_new 