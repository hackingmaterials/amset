import importlib
import logging
import numpy as np
import os
import sys

"""
General tools used in other methods such as vector manipulation tools used in 
other methods as well as create_plot visualization method.
"""

class AmsetError(Exception):
    """
    Exception class for Amset. Raised when Amset gives an error. The purpose
    of this class is to be explicit about the exceptions raised specifically
    due to Amset input/output requirements instead of a generic ValueError, etc
    """
    def __init__(self, logger, msg):
        self.msg = msg
        logger.error(self.msg)

    def __str__(self):
        return "AmsetError : " + self.msg


def setup_custom_logger(name, filepath, filename, level=None):
    """
    Custom logger with both screen and file handlers. This is particularly
    useful if there are other programs (e.g. BoltzTraP2) that call on logging
    in which case the log results and their levels are distict and clear.

    Args:
        name (str): logger name to distinguish between different codes.
        filepath (str): path to the folder where the logfile is meant to be
        filename (str): log file filename
        level (int): log level in logging package; example: logging.DEBUG

    Returns: a logging instance with customized formatter and handlers
    """
    importlib.reload(logging)
    level = level or logging.DEBUG
    logger = logging.getLogger(name)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler(os.path.join(filepath, filename), mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(screen_handler)
    logger.addHandler(handler)
    return logger


def remove_from_grid(grid, grid_rm_list):
    """
    Deletes dictionaries storing properties that are no longer needed from
    a given grid (i.e. kgrid or egrid)
    """
    for tp in ["n", "p"]:
        for rm in grid_rm_list:
            try:
                del (grid[tp][rm])
            except:
                pass
    return grid


def norm(v):
    """
    Quickly calculates the norm of a vector (v: 1x3 or 3x1) as np.linalg.norm
    can be slower if called individually for each vector.
    """
    return (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5


def outer(v1, v2):
    """returns the outer product of vectors v1 and v2. This is to be used
    instead of numpy.outer which is ~3x slower if only used for 2 vectors."""
    return np.array([[v1[0] * v2[0], v1[0] * v2[1], v1[0] * v2[2]],
                     [v1[1] * v2[0], v1[1] * v2[1], v1[1] * v2[2]],
                     [v1[2] * v2[0], v1[2] * v2[1], v1[2] * v2[2]]])


def cos_angle(v1, v2):
    """
    Returns cosine of the angle between two 3x1 or 1x3 vectors
    """
    norm_v1, norm_v2 = norm(v1), norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 1.0  # In case of the two points are the origin, we assume 0 degree; i.e. no scattering: 1-X==0
    else:
        return np.dot(v1, v2) / (norm_v1 * norm_v2)


def get_angle(v1, v2):
    """
    Returns the actual angles (in radian) between 2 vectors not its cosine.
    """
    x = cos_angle(v1,v2)
    if x < -1:
        x = -1 # just to avoid consequence of numerical instability.
    elif x > 1:
        x = 1
    return np.arccos(x)


def sort_angles(vecs):
    """
    Sort a list of vectors based on their pair angles using a greedy algorithm.

    Args:
        vecs ([nx1 list or numpy.ndarray]): list of nd vectors

    Returns (tuple): sorted vecs, indexes of the initial vectors in the order
        that results in sorted vectors.
    """
    sorted_vecs = []
    indexes = range(len(vecs))
    final_idx = []
    vecs = list(vecs)
    while len(vecs) > 1:
        angles = [get_angle(vecs[0], vecs[i]) for i in range(len(vecs))]
        sort_idx = np.argsort(angles)
        vecs = [vecs[i] for i in sort_idx]
        indexes = [indexes[i] for i in sort_idx]
        sorted_vecs.append(vecs.pop(0)) # reduntant step for the first element
        final_idx.append(indexes.pop(0))
    vecs.extend(sorted_vecs)
    indexes.extend(final_idx)
    return np.array(vecs), indexes


def create_plots(x_title, y_title, tp,
                 file_suffix, fontsize, ticksize, path, margins, fontfamily,
                 plot_data, mode='offline', names=None, labels=None, x_label_short='',
                 y_label_short=None, xy_modes='markers', y_axis_type='linear',
                 title=None, empty_markers=True, **kwargs):
    """
    A wrapper function with args mostly consistent with
    matminer.figrecipes.plot.PlotlyFig

    Args:
        x_title (str): label of the x-axis
        y_title (str): label of the y-axis
        tp (str): "n" or "p"
        file_suffix (str): small suffix for filename (NOT a file format)
        fontsize (int):
        ticksize (int):
        path (str): root folder where the plot will be saved.
        margins (float or [float]): figrecipe PlotlyFig margins
        fontfamily (str):
        plot_data ([(x_data, y_data) tuples]): the actual data to be plotted
        mode (str): plot mode. "offline" and "static" recommended. "static"
            would automatically set the file format to .png
        names ([str]): names of the traces
        labels ([str]): the labels of the scatter points
        x_label_short (str): used for distinguishing filenames
        y_label_short (str):  used for distinguishing filenames
        xy_modes (str): mode of the xy scatter plots: "markers", "lines+markers"
        y_axis_type (str): e.g. "log" for logscale
        title (str): the title of the plot appearing at the top
        empty_markers (bool): whether the markers are empty (filled if False)
        **kwargs: other keyword arguments of matminer.figrecipes.plot.PlotlyFig
                for example, for setting plotly credential when mode=="static"

    Returns (None): to return the dict

    """
    from matminer.figrecipes.plot import PlotlyFig
    plot_data = list(plot_data)
    marker_symbols = range(44)
    if empty_markers:
        marker_symbols = [i+100 for i in marker_symbols]
    tp_title = {"n": "conduction band(s)", "p": "valence band(s)"}
    if title is None:
        title = '{} for {}'.format(y_title, tp_title[tp])
    if y_label_short is None:
        y_label_short = y_title
    if not x_label_short:
        filename = os.path.join(path, "{}_{}".format(y_label_short, file_suffix))
    else:
        filename = os.path.join(path, "{}_{}_{}".format(y_label_short, x_label_short, file_suffix))
    if mode == "static":
        if not filename.endswith(".png"):
            filename += ".png"
    pf = PlotlyFig(x_title=x_title, y_title=y_title, y_scale=y_axis_type,
                    title=title, fontsize=fontsize,
                   mode=mode, filename=filename, ticksize=ticksize,
                    margins=margins, fontfamily=fontfamily, **kwargs)
    pf.xy(plot_data, names=names, labels=labels, modes=xy_modes, marker_scale=1.1,
          markers=[{'symbol': marker_symbols[i],
                   'line': {'width': 2,'color': 'black'}} for i, _ in enumerate(plot_data)])
