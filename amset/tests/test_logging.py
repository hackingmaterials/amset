import os
import unittest
import logging

from amset.logging import LoggableMixin, initialize_logger, \
    initialize_null_logger, logger_base_name

_run_dir = os.getcwd()


class LoggableMixinSubclass(LoggableMixin):
    """
    A class for testing logging mixin classes.

    Args:
        logger (bool or logging.Logger): The logging object.
    """
    def __init__(self, logger=True):
        self._logger = self.get_logger(logger)


class TestLoggableMixin(unittest.TestCase):

    def test_LoggableMixin(self):
        lms = LoggableMixinSubclass(logger=True)
        self.assertTrue(hasattr(lms, "logger"))
        self.assertTrue(isinstance(lms.logger, logging.Logger))


class TestLogging(unittest.TestCase):

    def test_logger_initialization(self):
        log = initialize_logger(logger_base_name, level=logging.DEBUG)
        log.info("Test logging.")
        log.debug("Test debug.")
        log.warning("Test warning.")

        # test the log is written to run dir (e.g. where the script was called
        # from and not the location of this test file
        log_file = os.path.join(_run_dir, logger_base_name + ".log")
        self.assertTrue(os.path.isfile(log_file))

        with open(log_file, 'r') as f:
            lines = f.readlines()

        self.assertTrue("logging" in lines[0])
        self.assertTrue("debug" in lines[1])
        self.assertTrue("warning" in lines[2])

        null = initialize_null_logger("matbench_null")
        null.info("Test null log 1.")
        null.debug("Test null log 2.")
        null.warning("Test null log 3.")

        null_log_file = os.path.join(_run_dir, logger_base_name + "_null.log")
        self.assertFalse(os.path.isfile(null_log_file))
