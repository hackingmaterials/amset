import itertools
import logging
import sys
from typing import Optional, Tuple, Union

import numpy as np
from scipy.spatial.qhull import Voronoi, ConvexHull
from scipy.stats import binned_statistic_dd
from tqdm import tqdm

from amset import amset_defaults
from amset.constants import output_width
from amset.log import log_list

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
        self._nworkers = nworkers
        self._original_mesh = original_mesh
        self.frac_points = frac_points

        if not reciprocal_lattice_matrix:
            reciprocal_lattice_matrix = np.diag([1, 1, 1])

        if original_mesh is None:
            self._grid_length_by_axis = [0.05] * 3
        else:
            self._grid_length_by_axis = 2 / original_mesh

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

        # frac_points_idx is the index of the original frac points in the
        # periodic mesh
        self._frac_points_idx = self._periodic_idx[-len(frac_points):]

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
        logger.info("Calculating k-point Voronoi volumes in blocks:")
        logger.debug("  ├── # total k-points: {}".format(len(self.frac_points)))
        logger.debug("  ├── # blocks: {:d}".format(np.product(
            self._n_blocks_by_axis)))

        # b_idx is the list of buffer cells, with the form:
        # [[1, x], [1, y], [1, z]] where x, y, and z are the buffer cells on the
        # far side of the unitcell
        b_idx = np.stack(([1, 1, 1], self._n_blocks_by_axis + 2), axis=1)

        volumes = np.zeros(self.frac_points.shape[0])
        pbar = tqdm(total=volumes.shape[0], ncols=output_width,
                    desc="    ├── progress", file=sys.stdout,
                    bar_format='{l_bar}{bar}| {elapsed}<{remaining}{postfix}')

        for nx, ny, nz in np.ndindex(tuple(self._n_blocks_by_axis + 2)):
            # group numbers start from 1
            nx += 1
            ny += 1
            nz += 1

            if nx in b_idx[0] or ny in b_idx[1] or nz in b_idx[2]:
                # cell is in the buffer zone outside the unitcell therefore
                # don't calculate voronoi weights
                continue

            # get the indices of the points to include when calculating the
            # Voronoi diagram, this includes the block of interest and the
            # blocks immediately surrounding it
            voro_idx = self._get_idx_by_group(
                (nx - 1, nx + 1), (ny - 1, ny + 1), (nz - 1, nz + 1))
            print("n1 group", self._get_idx_by_group(5, 5, 5))

            # get the indices of any unit cell points in voro_idx, I.e.
            # it is the index of the point in voro_idx, not the index of the
            # point in the original periodic mesh. Allows us to map from the
            # voro results to the original periodic mesh
            voro_to_block_idx = _get_loc(voro_idx, self._frac_points_idx)
            print(voro_to_block_idx.shape)

            # Now get the indices of the unit cell points (in the periodic mesh)
            # that we are including in the voronoi diagram
            points_in_voro = voro_idx[voro_to_block_idx]
            # print(points_in_voro)

            # finally, normalise these indices to get their corresponding index
            # in the original frac_points mesh. As we put the original
            # frac_points at the end of the periodic_points mesh this is easy.
            points_in_voro -= len(self._periodic_points) - len(self.frac_points)

            voro = Voronoi(self._periodic_points_cart[voro_idx])
            volumes[points_in_voro] = _get_voronoi_volumes(
                voro, voro_to_block_idx)

            pbar.update(len(points_in_voro))

        return volumes

    def _get_idx_by_group(self,
                          nx: Union[int, Tuple[int, int]],
                          ny: Union[int, Tuple[int, int]],
                          nz: Union[int, Tuple[int, int]]):
        if isinstance(nx, tuple):
            xmask = ((self._groups_idx[0] >= nx[0]) &
                     (self._groups_idx[0] <= nx[1]))
        else:
            xmask = self._groups_idx[0] == nx

        if isinstance(ny, tuple):
            ymask = ((self._groups_idx[1] >= ny[0]) &
                     (self._groups_idx[1] <= ny[1]))
        else:
            ymask = self._groups_idx[1] == ny

        if isinstance(nz, tuple):
            zmask = ((self._groups_idx[2] >= nz[0]) &
                     (self._groups_idx[2] <= nz[1]))
        else:
            zmask = self._groups_idx[2] == nz

        return self._periodic_idx[xmask & ymask & zmask]


def _get_voronoi_volumes(voronoi, volume_indices) -> np.ndarray:
    volumes = np.zeros(len(volume_indices))

    for i, reg_num in enumerate(voronoi.point_region[volume_indices]):
        indices = voronoi.regions[reg_num]
        if -1 in indices:
            # some regions can be opened
            volumes[i] = np.inf
        else:
            volumes[i] = ConvexHull(voronoi.vertices[indices]).volume
    return volumes


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
