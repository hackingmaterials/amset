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


def remove_from_grid(grid, grid_rm_list):
    """
    Deletes dictionaries storing properties that are no longer needed from
    a given grid (i.e. kgrid or egrid)
    """
    for tp in ["n", "p"]:
        for rm in grid_rm_list:
            try:
                del(grid[tp][rm])
            except KeyError:
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
        # In case of the two points are the origin, we assume 0 degree;
        # i.e. no scattering: 1-X==0
        return 1.0
    else:
        return np.dot(v1, v2) / (norm_v1 * norm_v2)


def get_angle(v1, v2):
    """
    Returns the actual angles (in radian) between 2 vectors not its cosine.
    """
    x = cos_angle(v1, v2)
    if x < -1:
        x = -1  # just to avoid consequence of numerical instability.
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
        sorted_vecs.append(vecs.pop(0))  # redundant step for the first element
        final_idx.append(indexes.pop(0))
    vecs.extend(sorted_vecs)
    indexes.extend(final_idx)
    return np.array(vecs), indexes


def get_tp(doping_concentration):
    """Get the doping type.

    Args:
        doping_concentration: The carrier concentration.

    Returns:
        (str): "n" for n-type doping (i.e. doping_concentration < 0) and
        "p" for p-type doping (doping_concentration > 0).
    """
    if doping_concentration < 0:
        return "n"
    elif doping_concentration > 0:
        return "p"
    else:
        raise ValueError("The carrier concentration cannot be zero.")
