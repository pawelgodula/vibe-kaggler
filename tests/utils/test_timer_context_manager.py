import unittest
import time
import io
import logging
from unittest.mock import patch
from utils.timer_context_manager import timer_context_manager

# Configure a logger specifically for testing the timer's output
log_stream = io.StringIO()
# Use a handler that writes to our string stream
test_handler = logging.StreamHandler(log_stream)
formatter = logging.Formatter('%(levelname)s - %(message)s') # Simplified format for easy checking
test_handler.setFormatter(formatter)

# Get the specific logger used within timer_context_manager
timer_logger = logging.getLogger('utils.timer_context_manager')
# Clear existing handlers if any and add our test handler
timer_logger.handlers.clear()
timer_logger.addHandler(test_handler)
timer_logger.setLevel(logging.INFO)
timer_logger.propagate = False # Prevent interference from root logger config

class TestTimerContextManager(unittest.TestCase):

    def setUp(self):
        """Clear the log stream before each test."""
        log_stream.seek(0)
        log_stream.truncate(0)

    def test_timing_accuracy(self):
        """Test if the timer measures sleep duration approximately correctly."""
        sleep_duration = 0.1
        with timer_context_manager("Sleep Test"):
            time.sleep(sleep_duration)

        log_output = log_stream.getvalue()
        self.assertIn("INFO - Block 'Sleep Test' executed in", log_output)
        # Extract the time value
        try:
            reported_time_str = log_output.split("in ")[-1].split(" seconds")[0]
            reported_time = float(reported_time_str)
        except (IndexError, ValueError):
            self.fail(f"Could not parse time from log output: {log_output}")

        # Check if reported time is close to sleep duration (allow some tolerance)
        self.assertAlmostEqual(reported_time, sleep_duration, delta=0.05) # 50ms tolerance

    def test_block_name_in_output(self):
        """Test that the provided block name appears in the log output."""
        block_name = "My Special Block"
        with timer_context_manager(block_name):
            pass # No operation
        log_output = log_stream.getvalue()
        self.assertIn(f"Block '{block_name}'", log_output)

    def test_no_block_name(self):
        """Test the output when no block name is provided."""
        with timer_context_manager():
            pass
        log_output = log_stream.getvalue()
        self.assertIn("Code block executed in", log_output) # Check default prefix
        self.assertNotIn("Block 'None'", log_output)

    def test_exception_handling(self):
        """Test that the timer still reports time even if an exception occurs."""
        class MyException(Exception):
            pass

        start_time = time.perf_counter()
        with self.assertRaises(MyException):
            with timer_context_manager("Exception Test"):
                time.sleep(0.05) # Simulate some work before exception
                raise MyException("Test exception inside timer")
        end_time = time.perf_counter()
        actual_duration = end_time - start_time

        log_output = log_stream.getvalue()
        self.assertIn("INFO - Block 'Exception Test' executed in", log_output)
        # Check if reported time is roughly correct
        try:
            reported_time_str = log_output.split("in ")[-1].split(" seconds")[0]
            reported_time = float(reported_time_str)
        except (IndexError, ValueError):
            self.fail(f"Could not parse time from log output: {log_output}")
        self.assertAlmostEqual(reported_time, actual_duration, delta=0.05)

    @patch('utils.timer_context_manager.logger') # Patch the logger instance used by the context manager
    def test_print_fallback(self, mock_logger):
        """Test that it falls back to print if logger is not enabled for INFO."""
        mock_logger.isEnabledFor.return_value = False # Simulate logger level > INFO

        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            with timer_context_manager("Print Fallback"):
                time.sleep(0.01)

            print_output = mock_stdout.getvalue()
            # Logger should not have been called
            mock_logger.info.assert_not_called()
            # Check print output
            self.assertIn("Block 'Print Fallback' executed in", print_output)
            self.assertIn("seconds", print_output)

if __name__ == '__main__':
    unittest.main() 