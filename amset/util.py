import numpy as np

from multiprocessing.sharedctypes import RawArray


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

    level = level or logging.DEBUG
    filename = filename or name + ".log"

    logger = logging.getLogger(name)
    logger.setLevel(level)
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

    return logger


def create_shared_array(data, return_buffer=False):
    data = np.asarray(data)
    shared_data = RawArray("d", np.prod(data.shape))
    buffered_data = np.frombuffer(shared_data).reshape(data.shape)
    buffered_data[:] = data[:]

    if return_buffer:
        return shared_data, buffered_data
    else:
        return shared_data

