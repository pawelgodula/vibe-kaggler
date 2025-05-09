# Samples rows from a Polars DataFrame.

"""
Extended Description:
Provides a utility to randomly sample a fixed number of rows or a fraction
of rows from a Polars DataFrame. It leverages the `.sample()` method
of Polars DataFrames.
"""

import polars as pl
from typing import Optional

def sample_dataframe(
    df: pl.DataFrame,
    n: Optional[int] = None,
    frac: Optional[float] = None,
    shuffle: bool = True,
    random_state: Optional[int] = None
) -> pl.DataFrame:
    """Samples rows from a Polars DataFrame.

    Args:
        df (pl.DataFrame): The DataFrame to sample from.
        n (Optional[int], optional): The number of rows to sample.
                                    Cannot be used with `frac`. Defaults to None.
        frac (Optional[float], optional): The fraction of rows to sample (0.0 to 1.0).
                                         Cannot be used with `n`. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the DataFrame before sampling.
                                Defaults to True.
        random_state (Optional[int], optional): Seed for the random number generator
                                                for reproducibility when shuffle=True.
                                                Defaults to None.

    Returns:
        pl.DataFrame: A new DataFrame containing the sampled rows.

    Raises:
        ValueError: If both `n` and `frac` are specified.
        ValueError: If neither `n` nor `frac` is specified.
        ValueError: If `frac` is not between 0.0 and 1.0.
        ValueError: If `n` is negative or larger than the DataFrame size.
    """
    if n is not None and frac is not None:
        raise ValueError("Cannot specify both 'n' and 'frac'.")
    if n is None and frac is None:
        raise ValueError("Must specify either 'n' or 'frac'.")

    if frac is not None and not (0.0 <= frac <= 1.0):
        raise ValueError(f"'frac' must be between 0.0 and 1.0, got {frac}")
    
    if n is not None and n < 0:
         raise ValueError(f"'n' must be non-negative, got {n}")

    # Polars handles n > len(df) gracefully by returning the whole df if shuffle=False,
    # but raises error if shuffle=True. Let's raise error consistently for clarity.
    if n is not None and n > len(df):
         raise ValueError(f"'n' ({n}) cannot be greater than the number of rows ({len(df)})")

    # Use slice for non-shuffled sampling, sample otherwise
    if not shuffle:
        if n is not None:
            return df.slice(0, n)
        else: # frac must be non-None due to earlier checks
            return df.slice(0, int(len(df) * frac))
    else:
        # Polars DataFrame.sample() method for shuffled sampling
        try:
            sampled_df = df.sample(
                n=n,
                fraction=frac,
                shuffle=True, # Explicitly True here
                seed=random_state
            )
            return sampled_df
        except Exception as e:
            # Catch potential Polars errors
            print(f"Error during DataFrame sampling: {e}")
            raise


if __name__ == '__main__':
    # Example Usage
    data = {
        'id': list(range(1, 101)),
        'value': [i * 10 for i in range(1, 101)]
    }
    example_df = pl.DataFrame(data)
    print("Original DataFrame:")
    print(example_df.head())
    print(f"Shape: {example_df.shape}")

    # --- Sample N rows --- 
    print("\n--- Sampling 5 rows (shuffle=True, with seed) ---")
    sampled_n = sample_dataframe(example_df, n=5, random_state=42)
    print(sampled_n)
    print(f"Shape: {sampled_n.shape}")

    # --- Sample Fraction of rows --- 
    print("\n--- Sampling 20% of rows (shuffle=True, no seed) ---")
    sampled_frac = sample_dataframe(example_df, frac=0.2)
    print(sampled_frac.head())
    print(f"Shape: {sampled_frac.shape}")

    # --- Sample without shuffle (head) --- 
    print("\n--- Sampling first 3 rows (shuffle=False) ---")
    sampled_no_shuffle_n = sample_dataframe(example_df, n=3, shuffle=False)
    print(sampled_no_shuffle_n)
    
    print("\n--- Sampling first 10% of rows (shuffle=False) ---")
    sampled_no_shuffle_frac = sample_dataframe(example_df, frac=0.1, shuffle=False)
    print(sampled_no_shuffle_frac)
    print(f"Shape: {sampled_no_shuffle_frac.shape}")

    # --- Error Cases ---
    print("\n--- Error Cases ---")
    try:
        sample_dataframe(example_df, n=5, frac=0.1)
    except ValueError as e:
        print(f"Caught expected error: {e}")
    try:
        sample_dataframe(example_df)
    except ValueError as e:
        print(f"Caught expected error: {e}")
    try:
        sample_dataframe(example_df, frac=1.5)
    except ValueError as e:
        print(f"Caught expected error: {e}")
    try:
        sample_dataframe(example_df, n=200)
    except ValueError as e:
        print(f"Caught expected error: {e}") 