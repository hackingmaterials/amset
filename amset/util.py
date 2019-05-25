import collections
import copy
import datetime
import logging
import math
import os
import sys
import textwrap
import time
from multiprocessing.sharedctypes import RawArray

import numpy as np
import scipy

from amset import amset_defaults
from amset.constants import output_width
from pymatgen import Spin

spin_name = {Spin.up: "spin-up", Spin.down: "spin-down"}

logger = logging.getLogger(__name__)


def initialize_amset_logger(filepath='.', filename=None, level=None,
                            log_traceback=False):
    """Initialize the default logger with stdout and file handlers.

    Args:
        filepath (str): Path to the folder where the log file will be written.
        filename (str): The log filename.
        level (int): The log level. For example logging.DEBUG.
        log_traceback: Whether to log the traceback if an error occurs.

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

        if log_traceback:
            log.error("\n  ERROR: {}".format(exit_msg),
                         exc_info=(exc_type, exc_value, exc_traceback))
        else:
            log.error("\n  ERROR: {}\n\n  {}".format(
                str(exc_value), exit_msg))

    sys.excepthook = handle_exception

    return log


def create_shared_array(data, return_buffer=False):
    data = np.asarray(data)
    shared_data = RawArray("d", int(np.prod(data.shape)))
    buffered_data = np.frombuffer(shared_data).reshape(data.shape)
    buffered_data[:] = data[:]

    if return_buffer:
        return shared_data, buffered_data
    else:
        return shared_data


def validate_settings(user_settings):
    settings = copy.deepcopy(amset_defaults)

    def recursive_update(d, u):
        """ Recursive dict update."""
        for k, v in u.items():
            if isinstance(v, collections.Mapping):
                d[k] = recursive_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    recursive_update(settings, user_settings)

    # validate the type of some settings
    if isinstance(settings["general"]["doping"], (int, float)):
        settings["general"]["doping"] = [settings["general"]["doping"]]

    if isinstance(settings["general"]["temperatures"], (int, float)):
        settings["general"]["temperatures"] = [
            settings["general"]["temperatures"]]

    settings["general"]["doping"] = np.asarray(settings["general"]["doping"])
    settings["general"]["temperatures"] = np.asarray(
        settings["general"]["temperatures"])

    return settings


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


def star_log(text):
    width = output_width - 2
    nstars = (width - (len(text) + 2)) / 2
    logger.info("\n{} {} {}".format(
        '~' * math.ceil(nstars), text, '~' * math.floor(nstars)))


def log_list(list_strings, prefix="  "):
    for i, text in enumerate(list_strings):
        if i == len(list_strings) - 1:
            pipe = "└"
        else:
            pipe = "├"
        logger.info("{}{}── {}".format(prefix, pipe, text))


def tensor_average(tensor):
    return np.average(scipy.linalg.eigvalsh(tensor))

def groupby(a, b):
    # Get argsort indices, to be used to sort a and b in the next steps
    sidx = b.argsort(kind='mergesort')
    a_sorted = a[sidx]
    b_sorted = b[sidx]

    # Get the group limit indices (start, stop of groups)
    cut_idx = np.flatnonzero(
        np.r_[True, b_sorted[1:] != b_sorted[:-1], True])

    # Split input array with those start, stop ones
    out = np.array(
        [a_sorted[i:j] for i, j in zip(cut_idx[:-1], cut_idx[1:])])
    return out
