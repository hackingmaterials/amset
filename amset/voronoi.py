import itertools
import logging
import math
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
                 original_mesh: Optional[np.ndarray] = None,
                 max_points_per_chunk: int = 80000,
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
        self._max_points_per_chunk = max_points_per_chunk
        self.frac_points = frac_points

        if original_mesh is None:
            self._grid_length_by_axis = [0.05] * 3
        else:
            self._grid_length_by_axis = 1 / original_mesh

        self._n_blocks_by_axis = np.ceil(
            1 / self._grid_length_by_axis).astype(int)

        # In order to take into account periodic boundary conditions we repeat
        # the points a number of times in each direction. Note this method
        # might not be robust to cells with very small/large cell angles.
        # A better method would be to calculate the supercell mesh needed to
        # converge the Voronoi volumes, but in most cases this will be fine.
        dim = [-1, 0, 1]
        periodic_points = []

        # only include periodic points in the vicinity of the unit cell
        # limits is ((xmin, xmax), (ymin, ymax), (zmin, zmax))
        limits = np.stack(((-0.50001 - self._grid_length_by_axis),
                           (0.50001 + self._grid_length_by_axis)), axis=1)

        for image in itertools.product(dim, dim, dim):
            if image[0] == image[1] == image[2] == 0:
                # don't add the original points here
                continue
            points = frac_points + image

            # filter points far from unit cell
            mask = ((points[:, 0] >= limits[0][0]) &
                    (points[:, 0] <= limits[0][1]) &
                    (points[:, 1] >= limits[1][0]) &
                    (points[:, 1] <= limits[1][1]) &
                    (points[:, 2] >= limits[2][0]) &
                    (points[:, 2] <= limits[2][1]))

            periodic_points.append(points[mask])

        # add the original points at the end so we know their indices
        periodic_points.append(frac_points)

        self._periodic_points = np.concatenate(periodic_points)
        self._periodic_idx = np.arange(len(self._periodic_points))
        self._frac_points_idx = self._periodic_idx[-len(self.frac_points):]
        self._n_buffer_points = len(self._periodic_idx) - len(self.frac_points)

        print(self._n_blocks_by_axis + 2)

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
    # if len(self._periodic_points) < self._max_points_per_chunk:
        # we can treat all points simultaneously
        # logger.info("Calculating k-point Voronoi diagram:")
        # logger.debug("  ├── num total k-points: {}".format(
        #     len(self.frac_points)))
        #
        # t0 = time.perf_counter()
        # voro = Voronoi(self._periodic_points)
        # log_time_taken(t0)
        #
        # regions = voro.point_region[self._frac_points_idx]
        # indices = np.array(voro.regions)[regions]
        # vertices = np.array([voro.vertices[i] for i in indices])
        #
        # from monty.serialization import dumpfn
        # dumpfn(indices, "indices.json")
        # dumpfn(vertices.tolist(), "vertices.json")
    # else:
        self._max_points_per_chunk = 80000
        # we break the cell up into chunks, each containing a certain number
        # of blocks, and calculate the Voronoi diagrams for each chunk
        # individually. Each chunk has a buffer layer one block thick
        # surrounding it. The Voronoi diagrams for the points in the buffer
        # are discarded.
        n_chunks = math.ceil(self._periodic_idx.shape[0] /
                             self._max_points_per_chunk)

        logger.info("Calculating k-point Voronoi diagram in chunks:")
        logger.debug("  ├── num total k-points: {}".format(
            len(self.frac_points)))
        logger.debug("  ├── num chunks: {:d}".format(n_chunks))

        # split the axis with the most blocks into n_chunks
        chunk_axis_sorted = self._n_blocks_by_axis.argsort()

        # n_blocks_per_chunk is sorted with the largest axis first
        n_blocks_per_chunk = np.ceil(
                self._n_blocks_by_axis[chunk_axis_sorted] /
                (n_chunks, 1, 1)).astype(int)

        # unsort n_block_per_chunk back to the original order
        inverse_sort = chunk_axis_sorted.argsort()
        n_blocks_per_chunk = n_blocks_per_chunk[inverse_sort]

        # chunk_axis is equal to 0 for axes which are not chunked, and
        # 1 for the axis that is chunked. E.g. if the z axis is chunked,
        # chunk_axis will be (0, 0, 1)
        chunk_axis = np.array(n_blocks_per_chunk < self._n_blocks_by_axis,
                              dtype=int)

        chunks = []
        for i in range(n_chunks):
            min_blocks = 2 + (i * n_blocks_per_chunk * chunk_axis)
            max_blocks = n_blocks_per_chunk * (i * chunk_axis + 1) + 1

            # add buffer blocks
            min_blocks -= 1
            max_blocks += 1

            # append chunks as ((xmin, xmax), (ymin, ymax), (zmin, zmax))
            chunks.append(np.stack((min_blocks, max_blocks), axis=1))

        chunks = tqdm(
            chunks,
            ncols=output_width,
            desc="    ├── Voronoi",
            file=sys.stdout,
            bar_format='{l_bar}{bar}| {elapsed}<{remaining}{postfix}')

        t0 = time.perf_counter()
        results = Parallel(n_jobs=self._nworkers, prefer="processes")(
            delayed(voronoi_worker)(
                self._groups_idx, self._periodic_points,
                self._periodic_idx, self._n_buffer_points, nx, ny, nz)
            for nx, ny, nz in chunks)
        log_time_taken(t0)

        indices = np.empty((len(self.frac_points)), dtype=object)
        vertices = np.empty((len(self.frac_points)), dtype=object)

        for frac_idx, chunk_indices, chunk_vertices in results:
            indices[frac_idx] = chunk_indices
            vertices[frac_idx] = chunk_vertices

        nones = [i is None for i in vertices]
        print(self.frac_points[nones].max())
        print(self.frac_points[nones].min())

        print("Num Nones!!!", np.sum(nones))

        from monty.serialization import dumpfn
        dumpfn(indices, "indices_para.json")
        dumpfn(vertices.tolist(), "vertices_para.json")

        # volumes = _get_volumes(regions, indices)
        volumes = 0

        zero_vols = volumes == 0
        if any(zero_vols):
            logger.warning("{} volumes are zero".format(np.sum(zero_vols)))

        inf_vols = volumes == np.inf
        if any(inf_vols):
            logger.warning("{} volumes are infinite".format(np.sum(inf_vols)))

        return volumes


def voronoi_worker(groups_idx, periodic_points,
                   periodic_idx, n_buffer_points, nx, ny, nz):
    print((nx, ny, nz))
    inner_nx = (nx[0] + 1, nx[1] - 1)
    inner_ny = (ny[0] + 1, ny[1] - 1)
    inner_nz = (nz[0] + 1, nz[1] - 1)

    # get the indices of the inner block (i.e. unit cell points) we are
    # interested in
    inner_idx = _get_idx_by_group(groups_idx, periodic_idx, inner_nx,
                                  inner_ny, inner_nz)

    # get the indices of the points to include when calculating the
    # Voronoi diagram, this includes the points of interest (inner_idx) and the
    # buffer blocks immediately surrounding it
    voro_idx = _get_idx_by_group(groups_idx, periodic_idx, nx, ny, nz)

    # get the indices of the inner points in voro_idx
    inner_in_voro_idx = _get_loc(voro_idx, inner_idx)

    # get the indices of the inner points in self._periodic_points
    periodic_points_in_voro_idx = voro_idx[inner_in_voro_idx]

    # finally, normalise these indices to get the index of the inner points
    # in self.frac_points. As we put the original frac_points at the end
    # of the periodic_points, this is easy.
    frac_points_in_voro_idx = periodic_points_in_voro_idx - n_buffer_points

    voro = Voronoi(periodic_points[voro_idx])

    regions = voro.point_region[inner_in_voro_idx]
    indices = np.array(voro.regions)[regions]
    vertices = np.array([voro.vertices[i] for i in indices])

    return frac_points_in_voro_idx, indices, vertices


def _get_idx_by_group(groups_idx,
                      periodic_idx,
                      nx: Union[int, Tuple[int, int]],
                      ny: Union[int, Tuple[int, int]],
                      nz: Union[int, Tuple[int, int]]):
    xgroups = groups_idx[0]
    ygroups = groups_idx[1]
    zgroups = groups_idx[2]

    if isinstance(nx, int):
        evaluate_str = "(xgroups == nx)"
    else:
        xmin = nx[0]
        xmax = nx[1]
        evaluate_str = "(xgroups >= xmin) & (xgroups <= xmax)"

    if isinstance(ny, int):
        evaluate_str += " & (ygroups == ny)"
    else:
        ymin = ny[0]
        ymax = ny[1]
        evaluate_str += " & (ygroups >= ymin) & (ygroups <= ymax)"

    if isinstance(nz, int):
        evaluate_str += " & (zgroups == nz)"
    else:
        zmin = nz[0]
        zmax = nz[1]
        evaluate_str += " & (zgroups >= zmin) & (zgroups <= zmax)"

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

