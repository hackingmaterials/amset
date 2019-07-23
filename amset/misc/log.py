import datetime
import logging
import math
import os
import sys
import textwrap
import time

from amset.misc.constants import output_width

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"
__date__ = "June 21, 2019"

logger = logging.getLogger(__name__)


def initialize_amset_logger(filepath='.', filename=None, level=None,
                            log_error_traceback=False):
    """Initialize the default logger with stdout and file handlers.

    Args:
        filepath (str): Path to the folder where the log file will be written.
        filename (str): The log filename.
        level (int): The log level. For example logging.DEBUG.
        log_error_traceback: Whether to log the traceback if an error occurs.

    Returns:
        (Logger): A logging instance with customized formatter and handlers.
    """

    level = level or logging.DEBUG
    filename = filename or "amset.log"

    log = logging.getLogger("amset")
    log.setLevel(level)
    log.handlers = []  # reset logging handlers if they already exist

    formatter = WrappingFormatter(fmt='%(message)s')

    handler = logging.FileHandler(os.path.join(filepath, filename),
                                  mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    log.addHandler(screen_handler)
    log.addHandler(handler)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        now = datetime.datetime.now()
        exit_msg = "amset exiting on {} at {}".format(now.strftime("%d %b %Y"),
                                                      now.strftime("%H:%M"))

        if log_error_traceback:
            log.error("\n  ERROR: {}".format(exit_msg),
                         exc_info=(exc_type, exc_value, exc_traceback))
        else:
            log.error("\n  ERROR: {}\n\n  {}".format(
                str(exc_value), exit_msg))

    sys.excepthook = handle_exception

    return log


class WrappingFormatter(logging.Formatter):

    def __init__(self, fmt=None, datefmt=None, style='%', width=output_width):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.wrapper = textwrap.TextWrapper(
            width=width, subsequent_indent="  ",
            replace_whitespace=True, drop_whitespace=False)

    def format(self, record):
        text = super().format(record)
        if "└" in text or "├" in text:
            # don't have blank line when reporting time
            return "  " + text
        else:
            return "\n" + "\n".join([
                self.wrapper.fill("  " + s) for s in text.splitlines()])


def log_time_taken(t0: float):
    logger.info("  └── time: {:.4f} s".format(time.perf_counter() - t0))


def log_banner(text):
    width = output_width - 2
    nstars = (width - (len(text) + 2)) / 2
    logger.info("\n{} {} {}".format(
        '~' * math.ceil(nstars), text, '~' * math.floor(nstars)))


def log_list(list_strings, prefix="  ", level=logging.INFO):
    for i, text in enumerate(list_strings):
        if i == len(list_strings) - 1:
            pipe = "└"
        else:
            pipe = "├"
        logger.log(level, "{}{}── {}".format(prefix, pipe, text))
