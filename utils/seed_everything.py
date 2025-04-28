# Sets random seeds for various libraries to ensure reproducibility.

"""
Extended Description:
This function sets the random seed for Python's built-in `random` module,
`numpy`, and optionally for deep learning frameworks like `tensorflow` and `pytorch`
if they are installed. Setting a fixed seed helps in obtaining reproducible
results across multiple runs, which is crucial for experiments and debugging.
Note: Full reproducibility, especially with GPU operations or certain complex
algorithms, might require additional settings or environment configurations.
"""

import random
import os
import numpy as np
from typing import Optional

def seed_everything(seed: Optional[int] = None) -> None:
    """Sets the random seed for Python, NumPy, and optionally TF/PyTorch.

    Args:
        seed (Optional[int], optional): The seed value to use. If None, a random seed
                                        will be generated, which means results
                                        will not be reproducible across runs.
                                        Defaults to None.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        print(f"Generated random seed: {seed}")
    else:
        print(f"Using fixed seed: {seed}")

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    # Optional: Seed TensorFlow if installed
    try:
        import tensorflow as tf
        # tf.random.set_seed(seed) # For TF 2.x
        # For TF 1.x compatibility, might need different approach or session config
        # Check TF version or use compatibility modes if needed
        if hasattr(tf.random, 'set_seed'):
            tf.random.set_seed(seed)
            # Additional settings for GPU reproducibility (may impact performance)
            # os.environ['TF_DETERMINISTIC_OPS'] = '1' # TF 2.x+
            # os.environ['TF_CUDNN_DETERMINISTIC'] = '1' # Might require specific cuDNN versions
        else:
             print("TensorFlow found, but tf.random.set_seed not available (likely TF 1.x?). Manual seeding might be required.")
    except ImportError:
        pass # TensorFlow not installed
    except Exception as e:
        print(f"Error seeding TensorFlow: {e}")

    # Optional: Seed PyTorch if installed
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) # For multi-GPU
            # Additional settings for GPU reproducibility (may impact performance)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
    except ImportError:
        pass # PyTorch not installed
    except Exception as e:
        print(f"Error seeding PyTorch: {e}") 