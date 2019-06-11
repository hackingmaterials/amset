import itertools
import logging
import sys
import time

import numpy as np
import numexpr as ne

from typing import Optional, Tuple, Union
from multiprocessing import cpu_count

from joblib import Parallel, delayed
from scipy.spatial.qhull import Voronoi, ConvexHull
from scipy.stats import binned_statistic_dd
from tqdm import tqdm

from amset import amset_defaults
from amset.constants import output_width
from amset.log import log_time_taken

pdefaults = amset_defaults["performance"]
logger = logging.getLogger(__name__)


class PeriodicVoronoi(object):
    """
    Note this class works because most of the points are on a regular grid so
    it is valid to calculate the Voronoi diagram in blocks.
    """

    def __init__(self,
                 frac_points: np.ndarray,
                 reciprocal_lattice_matrix: Optional[np.ndarray] = None,
                 original_mesh: Optional[np.ndarray] = None,
                 nworkers: int = pdefaults["nworkers"]):
        """

        Args:
            frac_points: A list of points in fractional coordinates. Points must
                have coordinates between -0.5 and +0.5.
            original_mesh:
            nworkers:
        """
        self._nworkers = nworkers if nworkers != -1 else cpu_count()
        self._original_mesh = original_mesh
        self.frac_points = frac_points

        if reciprocal_lattice_matrix is None:
            reciprocal_lattice_matrix = np.diag([1, 1, 1])

        if original_mesh is None:
            self._grid_length_by_axis = [0.05] * 3
        else:
            self._grid_length_by_axis = 5 / original_mesh

        self._n_blocks_by_axis = np.ceil(
            1 / self._grid_length_by_axis).astype(int)

        # In order to take into account periodic boundary conditions we repeat
        # the points a number of times in each direction. Note this method
        # might not be robust to cells with very small/large cell angles.
        # A better method would be to calculate the supercell mesh needed to
        # converge the Voronoi volumes, but in most cases this will be fine.
        dim = [-1, 0, 1]
        periodic_points = []

        for image in itertools.product(dim, dim, dim):
            if image[0] == image[1] == image[2] == 0:
                # don't add the original points here
                continue
            periodic_points.append(frac_points + image)

        # add the original points at the end so we know their indices
        periodic_points.append(frac_points)

        self._periodic_points = np.concatenate(periodic_points)
        self._periodic_points_cart = np.dot(
            self._periodic_points, reciprocal_lattice_matrix)
        self._periodic_idx = np.arange(len(self._periodic_points))
        self._n_buffer_points = len(self._periodic_idx) - len(self.frac_points)

        # limits is ((xmin, xmax), (ymin, ymax), (zmin, zmax))
        limits = np.stack(((-0.5 - self._grid_length_by_axis),
                           (0.5 + self._grid_length_by_axis)), axis=1)

        # group the points by their coordinates, with the group cutoffs defined
        # by multiples of inner_grid_length. Include two extra groups on either
        # side of the unit cell. I.e. if inner_grid_length is 0.2,
        # there will be 7 groups for each dimension, resulting in 343 total
        # groups, with the limits ranging from -0.7 to 0.7. The groups are
        # indexed using nx, ny, nz, which range from 1 to
        # n_blocks_by_axis + 2, the bins containing the original point set
        # range from 2 to n_blocks_by_axis, however, these bins
        # may also contain additional points due to aliasing errors.
        _, _, self._groups_idx = binned_statistic_dd(
            self._periodic_points, np.ones(len(self._periodic_points)),
            bins=self._n_blocks_by_axis + 2, range=limits,
            expand_binnumbers=True)

    def compute_volumes(self):
        t0 = time.perf_counter()
        logger.info("Calculating k-point Voronoi volumes in blocks:")
        logger.debug("  ├── # total k-points: {}".format(len(self.frac_points)))
        logger.debug("  ├── # blocks: {:d}".format(np.product(
            self._n_blocks_by_axis)))

        # buffer is the list of buffer cells, with the form:
        # [[1, x], [1, y], [1, z]] where x, y, and z are the buffer cells on the
        # far side of the unitcell
        buffer = np.stack(([1, 1, 1], self._n_blocks_by_axis + 2), axis=1)

        blocks = []
        for nx, ny, nz in np.ndindex(tuple(self._n_blocks_by_axis + 2)):
            # block indices start at 1
            nx += 1
            ny += 1
            nz += 1

            if nx in buffer[0] or ny in buffer[1] or nz in buffer[2]:
                continue

            blocks.append((nx, ny, nz))

        blocks = tqdm(blocks, ncols=output_width,
                      desc="    ├── progress",  file=sys.stdout,
                      bar_format='{l_bar}{bar}| {elapsed}<{remaining}{postfix}')

        results = Parallel(n_jobs=self._nworkers, prefer="threads")(
            delayed(voronoi_worker)(
                self._groups_idx, self._periodic_points_cart,
                self._periodic_idx, self._n_buffer_points, nx, ny, nz)
            for nx, ny, nz in blocks)

        volumes = np.zeros(self.frac_points.shape[0])
        for indices, vol in results:
            volumes[indices] = vol

        zero_vols = volumes == 0
        if any(zero_vols):
            logger.warning("{} volumes are zero".format(np.sum(zero_vols)))

        inf_vols = volumes == np.inf
        if any(inf_vols):
            logger.warning("{} volumes are infinite".format(np.sum(inf_vols)))

        log_time_taken(t0)
        return volumes

def voronoi_worker(groups_idx,
                   periodic_points_cart,
                   periodic_idx, n_buffer_points, nx, ny, nz):
    # get the indices of the block we are interested in
    block_idx = _get_idx_by_group(groups_idx, periodic_idx, nx, ny, nz)

    # get the indices of the points to include when calculating the
    # Voronoi diagram, this includes the block of interest and the
    # blocks immediately surrounding it
    voro_idx = _get_idx_by_group(
        groups_idx, periodic_idx, (nx - 1, nx + 1), (ny - 1, ny + 1),
        (nz - 1, nz + 1))

    # get the indices of the block points in voro_idx, I.e.
    # it is the index of the point in voro_idx, not the index of the
    # point in the original periodic mesh. Allows us to map from the
    # voro results to the original periodic mesh
    voro_to_block_idx = _get_loc(voro_idx, block_idx)

    # Now get the indices of the unit cell points (in the periodic mesh)
    # that we are including in the voronoi diagram, so we can calculate
    # the volumes for just these points
    points_in_voro = voro_idx[voro_to_block_idx]

    # finally, normalise these indices to get their corresponding index
    # in the original frac_points mesh. As we put the original
    # frac_points at the end of the periodic_points mesh this is easy.
    points_in_voro -= n_buffer_points

    voro = Voronoi(periodic_points_cart[voro_idx])

    return points_in_voro, _get_voronoi_volumes(voro, voro_to_block_idx)


def _get_idx_by_group(groups_idx,
                      periodic_idx,
                      nx: Union[int, Tuple[int, int]],
                      ny: Union[int, Tuple[int, int]],
                      nz: Union[int, Tuple[int, int]]):
    xgroups = groups_idx[0]
    ygroups = groups_idx[1]
    zgroups = groups_idx[2]

    if isinstance(nx, tuple):
        xmin = nx[0]
        xmax = nx[1]
        evaluate_str = "(xgroups >= xmin) & (xgroups <= xmax)"
    else:
        evaluate_str = "(xgroups == nx)"

    if isinstance(ny, tuple):
        ymin = ny[0]
        ymax = ny[1]
        evaluate_str += " & (ygroups >= ymin) & (ygroups <= ymax)"
    else:
        evaluate_str += " & (ygroups == ny)"

    if isinstance(nz, tuple):
        zmin = nz[0]
        zmax = nz[1]
        evaluate_str += " & (zgroups >= zmin) & (zgroups <= zmax)"
    else:
        evaluate_str += " & (zgroups == nz)"

    mask = ne.evaluate(evaluate_str)
    return periodic_idx[mask]


def _get_voronoi_volumes(voronoi, volume_indices) -> np.ndarray:

    def get_volume(indices):
        if -1 in indices:
            # some regions can be open
            return np.inf
        else:
            return ConvexHull(voronoi.vertices[indices]).volume

    return np.array([get_volume(voronoi.regions[x])
                     for x in voronoi.point_region[volume_indices]])


def _get_loc(x, y):
    """
    Based on https://stackoverflow.com/questions/8251541

    Args:
        x:
        y:

    Returns:

    """
    # len(x) > len(y)
    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y)
    yindex = np.take(index, sorted_index, mode="clip")
    mask = x[yindex] == y
    return yindex[mask]
