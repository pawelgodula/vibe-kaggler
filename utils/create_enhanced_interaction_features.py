# Creates enhanced interaction features (multiply, divide, power) using Polars.

"""
Extended Description:
This function generates new features by applying various operations between specified pairs 
of existing numerical features in a Polars DataFrame using efficient Polars expressions.
Supported operations include:
- Multiplication (feature1 * feature2)
- Division (feature1 / feature2) with handling for division by zero
- Power (feature1 ^ feature2) with safety limits

It returns a new DataFrame containing only the generated interaction features.
"""

import polars as pl
from typing import List, Tuple, Dict, Literal, Optional, Union

OperationType = Literal["multiply", "divide", "power"]

def create_enhanced_interaction_features(
    df: pl.DataFrame,
    feature_pairs: List[Dict[str, Union[Tuple[str, str], OperationType, Optional[float]]]]
) -> pl.DataFrame:
    """Creates enhanced interaction features between pairs of existing features.

    Args:
        df (pl.DataFrame): DataFrame containing the input features.
        feature_pairs (List[Dict]): A list of dictionaries, where each dictionary contains:
            - 'features': Tuple[str, str] - pair of feature names to interact
            - 'operation': "multiply", "divide", or "power" 
            - 'fill_value': Optional[float] - value to use for division by zero or overflow (default: None)

    Returns:
        pl.DataFrame: A new DataFrame containing the generated interaction features.
                      Column names are formatted based on the operation:
                      - Multiply: 'feature1_mul_feature2'
                      - Divide: 'feature1_div_feature2'
                      - Power: 'feature1_pow_feature2'

    Raises:
        pl.exceptions.ColumnNotFoundError: If any specified feature is not found.
        pl.exceptions.ComputeError: If operation is attempted on non-numeric types.
        ValueError: If feature_pairs is empty or contains invalid configurations.
    """
    if not feature_pairs:
        raise ValueError("feature_pairs list cannot be empty.")

    interaction_expressions = []
    generated_col_names = set()

    # Pre-check features existence and type
    all_features_needed = set()
    for pair_dict in feature_pairs:
        if 'features' not in pair_dict or not isinstance(pair_dict['features'], tuple) or len(pair_dict['features']) != 2:
            raise ValueError(f"Each element in feature_pairs must contain a 'features' key with a tuple of length 2. Found: {pair_dict}")
        
        if 'operation' not in pair_dict or pair_dict['operation'] not in ["multiply", "divide", "power"]:
            raise ValueError(f"Each element in feature_pairs must contain an 'operation' key with value 'multiply', 'divide', or 'power'. Found: {pair_dict}")
        
        feat1, feat2 = pair_dict['features']
        all_features_needed.add(feat1)
        all_features_needed.add(feat2)
    
    missing_features = [f for f in all_features_needed if f not in df.columns]
    if missing_features:
         # Raise error compatible with Polars/KeyError style
         raise pl.exceptions.ColumnNotFoundError(f"Features not found in DataFrame: {missing_features}")

    non_numeric_features = [f for f in all_features_needed if not df[f].dtype.is_numeric()]
    if non_numeric_features:
         raise pl.exceptions.ComputeError(f"Non-numeric features specified for interactions: {non_numeric_features}")

    for pair_dict in feature_pairs:
        feat1, feat2 = pair_dict['features']
        operation = pair_dict['operation']
        fill_value = pair_dict.get('fill_value', None)
        
        # Create appropriate column name based on operation
        if operation == "multiply":
            # For multiplication, ensure consistent ordering
            sorted_pair = tuple(sorted((feat1, feat2)))
            interaction_col_name = f"{sorted_pair[0]}_mul_{sorted_pair[1]}"
        else:
            # For division and power, order matters
            interaction_col_name = f"{feat1}_{operation[:3]}_{feat2}"

        if interaction_col_name not in generated_col_names:
            if operation == "multiply":
                # Simple multiplication
                expr = (pl.col(feat1) * pl.col(feat2)).alias(interaction_col_name)
            
            elif operation == "divide":
                # Division with zero-division handling
                if fill_value is not None:
                    # Use provided fill value for division by zero
                    expr = (pl.col(feat1) / pl.when(pl.col(feat2) != 0).then(pl.col(feat2)).otherwise(float('nan')))
                    expr = expr.fill_nan(fill_value).alias(interaction_col_name)
                else:
                    # Use null for division by zero
                    expr = (pl.col(feat1) / pl.when(pl.col(feat2) != 0).then(pl.col(feat2)).otherwise(None)).alias(interaction_col_name)
            
            elif operation == "power":
                # Power operation with safety limits
                # First, handle special cases like negative bases with fractional exponents
                if fill_value is not None:
                    # Use a complex expression to handle potential issues
                    # 1. Check base is not negative when exponent is not an integer
                    # 2. Apply reasonable limits to avoid overflow
                    # For simplified implementation, limit exponent range and handle negative bases
                    
                    # Clamp exponent to reasonable range [-10, 10] to avoid overflow
                    clamped_exp = pl.col(feat2).clip(-10, 10)
                    
                    # For negative bases, only allow integer exponents
                    safe_power = pl.when(
                        (pl.col(feat1) >= 0) | (clamped_exp.floor() == clamped_exp)
                    ).then(
                        pl.col(feat1).pow(clamped_exp)
                    ).otherwise(
                        fill_value
                    ).alias(interaction_col_name)
                    
                    expr = safe_power
                else:
                    # Without fill_value, use null for invalid operations
                    clamped_exp = pl.col(feat2).clip(-10, 10)
                    safe_power = pl.when(
                        (pl.col(feat1) >= 0) | (clamped_exp.floor() == clamped_exp)
                    ).then(
                        pl.col(feat1).pow(clamped_exp)
                    ).otherwise(
                        None
                    ).alias(interaction_col_name)
                    
                    expr = safe_power
            
            interaction_expressions.append(expr)
            generated_col_names.add(interaction_col_name)

    if not interaction_expressions:
        # Return empty DataFrame if no valid interactions
        return pl.DataFrame()
        
    # Select only the generated interaction columns
    df_interactions = df.select(interaction_expressions)
    
    return df_interactions 