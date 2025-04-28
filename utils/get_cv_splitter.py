# Returns a configured scikit-learn cross-validation splitter instance.

"""
Extended Description:
This function acts as a factory for various scikit-learn cross-validation splitters.
It allows specifying the type of splitter (e.g., 'kfold', 'stratified_kfold',
'group_kfold') and its parameters like n_splits, shuffle, and random_state.
This centralizes CV strategy definition.
"""

import sklearn.model_selection
from typing import Any, Optional

def get_cv_splitter(
    cv_type: str = 'kfold',
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None,
    **kwargs: Any
) -> sklearn.model_selection.BaseCrossValidator:
    """Creates and returns a scikit-learn cross-validation splitter.

    Args:
        cv_type (str, optional): The type of cross-validator to create.
            Supported: 'kfold', 'stratified_kfold', 'group_kfold'.
            Defaults to 'kfold'.
        n_splits (int, optional): Number of folds. Defaults to 5.
        shuffle (bool, optional): Whether to shuffle the data before splitting.
            Applicable to KFold and StratifiedKFold. Defaults to True.
        random_state (Optional[int], optional): Controls shuffling. Used when
            shuffle=True. Defaults to None.
        **kwargs (Any): Additional keyword arguments specific to the chosen
                        splitter (e.g., 'groups' for GroupKFold).

    Returns:
        sklearn.model_selection.BaseCrossValidator: An instance of the specified
            cross-validation splitter.

    Raises:
        ValueError: If an unsupported cv_type is provided.
    """
    cv_type_lower = cv_type.lower()

    # Common arguments, adjusted based on shuffle
    common_args = {
        'n_splits': n_splits,
        'shuffle': shuffle,
    }
    if shuffle:
        common_args['random_state'] = random_state

    if cv_type_lower == 'kfold':
        return sklearn.model_selection.KFold(**common_args, **kwargs)
    elif cv_type_lower == 'stratified_kfold':
        # StratifiedKFold specific args (if any) could be added here
        return sklearn.model_selection.StratifiedKFold(**common_args, **kwargs)
    elif cv_type_lower == 'group_kfold':
        # GroupKFold does not support shuffle or random_state in constructor
        # Warning is less useful here as we control args passed
        # if shuffle is not True or random_state is not None:
        #      print(f"Warning: 'shuffle' and 'random_state' are ignored for GroupKFold.")
        gkf_args = {'n_splits': n_splits}
        gkf_args.update(kwargs) # Pass only relevant extra args
        return sklearn.model_selection.GroupKFold(**gkf_args)
    # Add other CV strategies here as needed (e.g., TimeSeriesSplit)
    else:
        raise ValueError(
            f"Unsupported cv_type: '{cv_type}'. Supported types: "
            f"'kfold', 'stratified_kfold', 'group_kfold'" # Update if more are added
        ) 