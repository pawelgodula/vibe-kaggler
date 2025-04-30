# Configures Python's standard logging module.

"""
Extended Description:
Sets up basic logging to either the console (stderr) or a specified file.
It configures the root logger with a specific format including timestamp,
log level, and message. Allows setting the minimum logging level (e.g., INFO, DEBUG).
This ensures consistent logging format and destination across the project.
"""

import logging
import sys
from typing import Optional

def setup_logging(
    log_level: str = 'INFO',
    log_file: Optional[str] = None
) -> logging.Logger:
    """Configures the root logger and returns it.

    Removes existing handlers and adds a new one (Stream or File).

    Args:
        log_level (str, optional): The minimum logging level (e.g., 'DEBUG', 'INFO',
                                   'WARNING', 'ERROR', 'CRITICAL').
                                   Defaults to 'INFO'.
        log_file (Optional[str], optional): Path to a file to log messages to.
                                          If None, logs to stderr.
                                          Defaults to None.

    Raises:
        ValueError: If an invalid log_level is provided.
    """
    level = getattr(logging, log_level.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_format, date_format)

    root_logger = logging.getLogger() # Get the root logger

    # --- Always remove existing handlers --- 
    for handler in root_logger.handlers[:]:
        try:
            handler.close() # Close existing handlers
        except Exception:
            pass # Ignore errors during handler close
        root_logger.removeHandler(handler)
    # --- End removal ---

    # --- Add Handlers (File and/or Stream) --- 
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Always add a StreamHandler to stderr
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    # --- End adding handlers --- 
    
    # Set the level on the root logger
    root_logger.setLevel(level)
    
    return root_logger # Return the configured root logger

# Example usage:
# if __name__ == '__main__':
#     print("First setup...")
#     setup_logging(log_level='INFO')
#     logging.info("Info message 1")
#     logging.getLogger("module1").warning("Module 1 warning")
# 
#     print("\nSecond setup (file, debug, force)...")
#     setup_logging(log_level='DEBUG', log_file='app.log', force_reconfigure=True)
#     logging.debug("Debug message 2")
#     logging.getLogger("module2").info("Module 2 info")
# 
#     print("\nThird setup (skipped)...")
#     setup_logging(log_level='WARNING') # Should be skipped
#     logging.debug("Debug message 3 (should not show)")
#     logging.warning("Warning message 3 (should show in file)")
# 
#     if os.path.exists('app.log'): os.remove('app.log')
#
#     # Example of getting a logger for a specific module
#     logger = logging.getLogger('my_module')
#     logger.info("Message from my_module.") 