# Provides a context manager for timing blocks of code.

"""
Extended Description:
This function returns a context manager that measures the execution time
of the code block it wraps. When the block is entered, it records the start time.
When the block exits (either normally or via an exception), it records the end time
and prints the elapsed time in seconds.
It's useful for quickly profiling sections of code.
"""

import time
import contextlib
import logging
from typing import Generator, Optional

# Get a logger for this module (optional, could just use print)
logger = logging.getLogger(__name__)
# Ensure logger has a handler if setup_logging hasn't run or is skipped
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler()) # Default to stderr
    logger.propagate = False # Avoid double logging if root is also configured

@contextlib.contextmanager
def timer_context_manager(block_name: Optional[str] = None) -> Generator[None, None, None]:
    """A context manager to time a code block.

    Args:
        block_name (Optional[str], optional): An optional name for the code block
                                            to include in the output message.
                                            Defaults to None.

    Yields:
        Generator[None, None, None]: Yields control to the code block.

    Example:
        >>> with timer_context_manager("Data Loading"):
        ...     # Code to load data
        ...     time.sleep(0.5)
        # Output (approx): [INFO] Block 'Data Loading' executed in 0.501 seconds
    """
    start_time = time.perf_counter() # Use perf_counter for high resolution
    prefix = f"Block '{block_name}'" if block_name else "Code block"
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # Use logger if available and configured, otherwise print
        if logger.isEnabledFor(logging.INFO):
             logger.info(f"{prefix} executed in {elapsed_time:.3f} seconds")
        else:
             print(f"{prefix} executed in {elapsed_time:.3f} seconds")

# Example Usage:
# if __name__ == '__main__':
#     # Configure logging to see logger output
#     try:
#         from .setup_logging import setup_logging
#         setup_logging(log_level='INFO')
#     except ImportError:
#         print("setup_logging not found, using basic print output for timer.")
#         pass # Fallback to print if setup_logging is not available

#     print("Starting timer tests...")

#     with timer_context_manager("Simple Sleep"):
#         time.sleep(0.1)

#     with timer_context_manager(): # Without name
#         x = [i**2 for i in range(100000)]

#     print("Timer tests finished.") 