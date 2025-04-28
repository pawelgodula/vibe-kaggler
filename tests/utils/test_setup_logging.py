import unittest
import logging
import os
import io
from unittest.mock import patch, MagicMock
from utils.setup_logging import setup_logging # Removed import of _logging_configured

class TestSetupLogging(unittest.TestCase):

    def setUp(self):
        """Reset logging configuration before each test."""
        # No need to reset global flag
        root_logger = logging.getLogger()
        if root_logger.hasHandlers():
            for handler in root_logger.handlers[:]: 
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                root_logger.removeHandler(handler)
        root_logger.setLevel(logging.CRITICAL + 10)

        self.log_file = "test_logging.log"
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def tearDown(self):
        """Clean up log file if created."""
        root_logger = logging.getLogger()
        if root_logger.hasHandlers():
            for handler in root_logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
        
        if os.path.exists(self.log_file):
             try:
                 os.remove(self.log_file)
             except PermissionError:
                 print(f"Warning: Could not remove {self.log_file} in teardown.")

    def test_log_to_stderr_default_info(self):
        """Test logging to stderr with default INFO level."""
        with patch('sys.stderr', new_callable=io.StringIO) as mock_stderr:
            setup_logging() # Call with defaults
            logger = logging.getLogger("test_stderr")
            logger.debug("This should not be logged.")
            logger.info("This should be logged.")
            logger.warning("This should also be logged.")

            log_output = mock_stderr.getvalue()
            # print(f"Stderr log output:\n{log_output}") # Debug print
            self.assertNotIn("DEBUG", log_output)
            self.assertIn("INFO", log_output)
            self.assertIn("This should be logged.", log_output)
            self.assertIn("WARNING", log_output)
            self.assertIn("This should also be logged.", log_output)
            root_logger = logging.getLogger()
            self.assertEqual(root_logger.level, logging.INFO)
            self.assertEqual(len(root_logger.handlers), 1)
            self.assertTrue(isinstance(root_logger.handlers[0], logging.StreamHandler))

    def test_log_to_file_debug_level(self):
        """Test logging to a file with DEBUG level."""
        setup_logging(log_level='DEBUG', log_file=self.log_file)
        logger = logging.getLogger("test_file")
        logger.debug("File debug message.")
        logger.info("File info message.")

        root_logger = logging.getLogger()
        self.assertEqual(len(root_logger.handlers), 1) # Should only have file handler
        self.assertTrue(isinstance(root_logger.handlers[0], logging.FileHandler))
        self.assertEqual(root_logger.level, logging.DEBUG)

        # Close handler before reading
        for handler in root_logger.handlers[:]:
             if isinstance(handler, logging.FileHandler):
                 handler.close()
                 root_logger.removeHandler(handler) # Remove to avoid issues in other tests

        self.assertTrue(os.path.exists(self.log_file))
        with open(self.log_file, 'r') as f:
            log_content = f.read()
        # print(f"File log content:\n{log_content}") # Debug print
        self.assertIn("DEBUG", log_content)
        self.assertIn("File debug message.", log_content)
        self.assertIn("INFO", log_content)
        self.assertIn("File info message.", log_content)
        

    def test_invalid_log_level(self):
        """Test providing an invalid log level string."""
        with self.assertRaises(ValueError) as cm:
            setup_logging(log_level='INVALID')
        self.assertIn("Invalid log level: INVALID", str(cm.exception))

    # Removed tests for reconfiguration logic (test_reconfiguration_skipped_by_default, test_force_reconfigure)
    # The new function always reconfigures.

if __name__ == '__main__':
    unittest.main() 