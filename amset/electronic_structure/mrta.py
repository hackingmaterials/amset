import logging

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from amset.constants import numeric_types
from amset.electronic_structure.kpoints import kpoints_to_first_bz

try:
    from interpolation.splines import eval_linear, UCGrid
    from interpolation.splines import extrap_options as xto

except ImportError:
    eval_linear = None

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

logger = logging.getLogger(__name__)


class MRTACalculator(object):
    def __init__(self, kpoints, kpoint_mesh, velocities):
        logger.info("Initializing momentum relaxation time factor calculator")

        # k-points has to cover the full BZ
        kpoints = kpoints_to_first_bz(kpoints)
        kpoint_mesh = tuple(kpoint_mesh)

        round_dp = int(np.log10(1 / 1e-6))
        kpoints = np.round(kpoints, round_dp)

        # get the indices to sort the k-points on the Z, then Y, then X columns
        sort_idx = np.lexsort((kpoints[:, 2], kpoints[:, 1], kpoints[:, 0]))

        # put the kpoints into a 3D grid so that they can be indexed as
        # kpoints[ikx][iky][ikz] = [kx, ky, kz]
        grid_kpoints = kpoints[sort_idx].reshape(kpoint_mesh + (3,))

        # Expand the k-point mesh to account for periodic boundary conditions
        grid_kpoints = np.pad(
            grid_kpoints, ((1, 1), (1, 1), (1, 1), (0, 0)), mode="wrap"
        )
        grid_kpoints[0, :, :] -= [1, 0, 0]
        grid_kpoints[:, 0, :] -= [0, 1, 0]
        grid_kpoints[:, :, 0] -= [0, 0, 1]
        grid_kpoints[-1, :, :] += [1, 0, 0]
        grid_kpoints[:, -1, :] += [0, 1, 0]
        grid_kpoints[:, :, -1] += [0, 0, 1]

        x = grid_kpoints[:, 0, 0, 0]
        y = grid_kpoints[0, :, 0, 1]
        z = grid_kpoints[0, 0, :, 2]

        self.interpolators = {}
        for spin, spin_velocities in velocities.items():
            nbands = spin_velocities.shape[0]

            # sort the coefficients then reshape them into the grid. The coefficients
            # can now be indexed as coefficients[iband][ikx][iky][ikz]
            sorted_velocities = spin_velocities[:, sort_idx]
            grid_shape = (nbands,) + kpoint_mesh + (3,)
            grid_velocities = sorted_velocities.reshape(grid_shape)

            # wrap the velocities to account for PBC
            grid_velocities = np.pad(
                grid_velocities, ((0, 0), (1, 1), (1, 1), (1, 1), (0, 0)), mode="wrap"
            )

            if nbands == 1:
                # this can cause a bug in RegularGridInterpolator. Have to fake
                # having at least two bands
                nbands = 2
                grid_velocities = np.tile(grid_velocities, (2, 1, 1, 1, 1))

            if eval_linear:
                grid = UCGrid(
                    (0, nbands - 1, nbands),
                    (x[0], x[-1], len(x)),
                    (y[0], y[-1], len(y)),
                    (z[0], z[-1], len(z)),
                )
                self.interpolators[spin] = (grid, grid_velocities)
            else:
                logger.warning(
                    "Install the 'interpolation' package for improved performance: "
                    "https://pypi.org/project/interpolation"
                )
                interp_range = (np.arange(nbands), x, y, z)
                self.interpolators[spin] = RegularGridInterpolator(
                    interp_range, grid_velocities, bounds_error=False, fill_value=None
                )

    def get_mrta_factor(self, spin, band_a, kpoint_a, band_b, kpoint_b):
        # k-points should be in fractional
        kpoint_a = np.asarray(kpoint_a)
        kpoint_b = np.asarray(kpoint_b)
        v1 = np.array([[band_a] + kpoint_a.tolist()])

        single_factor = False
        if isinstance(band_b, numeric_types):
            # only one band index given

            if len(kpoint_b.shape) > 1:
                # multiple k-point indices given
                band_b = np.array([band_b] * len(kpoint_b))

            else:
                band_b = np.array([band_b])
                kpoint_b = [kpoint_b]
                single_factor = True

        else:
            band_b = np.asarray(band_b)

        # v2 now has shape of (nkpoints_b, 4)
        v2 = np.concatenate([band_b[:, None], kpoint_b], axis=1)

        # get a big array of all the k-points to interpolate
        all_v = np.vstack([v1, v2])

        # get the interpolate projections for the k-points; p1 is the projections for
        # kpoint_a, p2 is a list of projections for the kpoint_b

        if eval_linear:
            p1, *p2 = eval_linear(*self.interpolators[spin], all_v, xto.LINEAR)
        else:
            p1, *p2 = self.interpolators[spin](all_v)

        factor = 1 - np.dot(p1, np.asarray(p2).T) / np.linalg.norm(p1) ** 2

        if single_factor:
            return factor[0]
        else:
            return factor
