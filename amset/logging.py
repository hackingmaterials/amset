"""
Utils for logging.
"""

import logging
import os
import sys

logger_base_name = "amset"


def initialize_logger(name, filepath='.', filename=None, level=None):
    """Initialize the default logger with stdout and file handlers.

    Args:
        name (str): The package name.
        filepath (str): Path to the folder where the log file will be written.
        filename (str): The log filename.
        level (int): The log level. For example logging.DEBUG.

    Returns:
        (Logger): A logging instance with customized formatter and handlers.
    """

    level = level or logging.INFO
    filename = filename or name + ".log"

    logger = logging.getLogger(name)
    logger.handlers = []  # reset logging handlers if they already exist

    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    handler = logging.FileHandler(os.path.join(filepath, filename),
                                  mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger.addHandler(screen_handler)
    logger.addHandler(handler)

    logger.setLevel(level)
    return logger


def initialize_null_logger(name):
    """Initialize the a dummy logger which will swallow all logging commands.

    Returns:
        (Logger): The package name.
        (Logger): A dummy logging instance with no output.
    """
    logger = logging.getLogger(name + "_null")
    logger.addHandler(logging.NullHandler())

    return logger


class LoggableMixin:
    """A mixin class for easy logging (or absence of it)."""

    @property
    def logger(self):
        """Get the class logger.

        If the logger is None, the logging calls will be redirected to a dummy
        logger that has no output.
        """
        if hasattr(self, "_logger"):
            return self._logger
        else:
            raise AttributeError("Loggable object has no _logger attribute!")

    @staticmethod
    def get_logger(logger, level=None):
        """Get the class logger, initializing it if necessary.

        Args:
            logger (Logger, bool): A custom logger object to use for logging.
                Alternatively, if set to True, the default amset logger
                will be used. If set to False, then no logging will occur.
            level (int): The log level. For example logging.DEBUG.

        Return:
            A ``logging.Logger`` object.
        """
        # need comparison to True and False to avoid overwriting Logger objects
        if logger is True:
            logger = logging.getLogger(logger_base_name)

            if not logger.handlers:
                initialize_logger(logger_base_name, level=level)

        elif logger is False:
            logger = logging.getLogger(logger_base_name + "_null")

            if not logger.handlers:
                initialize_null_logger(logger_base_name)

        return logger

    def log_raise(self, exception, msg, *args, **kwargs):
        """Log and raise exception.

        Args:
            exception: The exception type.
            msg: The error message.
            *args: Additional arguments for the exception.
            **kwargs: Keyword arguments for the exception.
        """
        self.logger.error(msg)
        raise exception(msg, *args, **kwargs)

