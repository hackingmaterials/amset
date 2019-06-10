from typing import Optional

import numpy as np
from amset import amset_defaults

pdefaults = amset_defaults["performance"]


class PeriodicVoronoi(object):
    """
    Note this class works because most of the points are on a regular grid.
    """

    def __init__(self,
                 frac_points: np.ndarray,
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

        if original_mesh:
            grid_length_by_axis = 5 / original_mesh
        else:
            grid_length_by_axis = [0.05] * 3

        n_blocks_by_axis = np.ceil(1/grid_length_by_axis)

        # In order to take into account periodic boundary conditions we repeat
        # the points a number of times in each direction. Note this method
        # might not be robust to cells with very small/large cell angles.
        # A better method would be to calculate the supercell mesh needed to
        # converge the Voronoi volumes, but in most cases this will be fine.
        dim = [-1, 0, 1]
        periodic_points = []

        for image in np.itertools.product(dim, dim, dim):
            if image[0] == image[1] == image[2] == 0:
                # don't add the original points here
                continue
            periodic_points.append(frac_points + image)

        # limits is ((xmin, xmax), (ymin, ymax), (zmin, zmax))
        limits = np.stack(((-0.5 - grid_length_by_axis),
                           (0.5 + grid_length_by_axis)), axis=1)

        # group the points by their coordinates, with the group cutoffs defined
        # by multiples of inner_grid_length. Include two extra groups on either
        # side of the unit cell. I.e. if inner_grid_length is 0.2,
        # there will be 7 groups for each dimension, resulting in 343 total
        # groups, with the limits ranging from -0.7 to 0.7. The groups are
        # indexed using nx, ny, nz, which range from 0 to
        # n_blocks_by_axis + 2, the bins containing the original point set
        # range from 1 to n_blocks_by_axis - 1, however, these bins
        # may also contain additional points due to aliasing errors.

        # doesn't work...
        self._groups_idx, _ = np.histogramdd(
            periodic_points, n_blocks_by_axis + 2, range=limits)

    def compute(self, return_volumes):













