import logging

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from amset.electronic_structure.kpoints import kpoints_to_first_bz

try:
    from interpolation.splines import eval_linear, UCGrid
    from interpolation.splines import extrap_options as xto

except ImportError:
    eval_linear = None
# eval_linear = None


__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

logger = logging.getLogger(__name__)


class ResponseCalculator(object):
    def __init__(self, kpoints, kpoint_mesh, coefficients):
        logger.info("Initializing linear response calculator")

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
        for spin, spin_coefficients in coefficients.items():
            nbands = spin_coefficients.shape[0]
            self._shape = spin_coefficients.shape[2:]

            # sort the coefficients then reshape them into the grid. The coefficients
            # can now be indexed as coefficients[iband][ikx][iky][ikz]
            sorted_coefficients = spin_coefficients[:, sort_idx]
            grid_shape = (nbands,) + kpoint_mesh + spin_coefficients.shape[2:]
            grid_coefficients = sorted_coefficients.reshape(grid_shape)

            # wrap the velocities to account for PBC
            grid_coefficients = np.pad(
                grid_coefficients,
                ((0, 0), (1, 1), (1, 1), (1, 1), (0, 0), (0, 0), (0, 0)),
                mode="wrap",
            )

            if nbands == 1:
                # this can cause a bug in RegularGridInterpolator. Have to fake
                # having at least two bands
                nbands = 2
                grid_coefficients = np.tile(grid_coefficients, (2, 1, 1, 1, 1, 1, 1))

            # print(x)
            # print(y)
            # import matplotlib.pyplot as plt
            # # coeffs = grid_coefficients[3, :, 30, 30, 0, 0]
            # # xx = np.arange(29)
            # # coeffs = grid_coefficients[3, xx, xx, xx, -1, -1]
            # # coeffs = grid_coefficients[3, xx, xx, xx, -1, -1]
            # # coeffs = grid_coefficients[3, xx, xx, xx, -1, -1]
            # # norm_c = np.linalg.norm(coeffs, axis=1)
            # import math
            # c = math.ceil(len(x)/2) -1
            # normx = np.linalg.norm(grid_coefficients[3, :, c, c, -1, -1], axis=1)
            # normy = np.linalg.norm(grid_coefficients[3, c, :, c, -1, -1], axis=1)
            # normz = np.linalg.norm(grid_coefficients[3, c, c, :, -1, -1], axis=1)
            # # print(coeffs)
            # # print(grid_kpoints[x, x, x])
            # plt.plot(np.linspace(0, 1, len(normx)), normx, label="x")
            # plt.plot(np.linspace(0, 1, len(normy)), normy, label="y")
            # # plt.plot(np.linspace(0, 1, len(norm_c)), norm_c, label="diag")
            # # plt.plot(np.linspace(0, 1, 29), np.linalg.norm(grid_coefficients[3, :, 14, 14, -1, -1], axis=1), label="line")
            # plt.legend()
            # plt.semilogy()
            # plt.show()

            if eval_linear:
                grid = UCGrid(
                    (0, nbands - 1, nbands),
                    (x[0], x[-1], len(x)),
                    (y[0], y[-1], len(y)),
                    (z[0], z[-1], len(z)),
                )
                self.interpolators[spin] = (grid, grid_coefficients)
            else:
                logger.warning(
                    "Install the 'interpolation' package for improved performance: "
                    "https://pypi.org/project/interpolation"
                )
                interp_range = (np.arange(nbands), x, y, z)
                self.interpolators[spin] = RegularGridInterpolator(
                    interp_range,
                    grid_coefficients,
                    bounds_error=False,
                    fill_value=None,
                    method="nearest",
                )

    def get_coefficients(self, spin, bands, kpoints):
        v = np.concatenate([np.asarray(bands)[:, None], np.asarray(kpoints)], axis=1)

        if eval_linear:
            grid, coeffs = self.interpolators[spin]
            interp_coeffs = np.zeros((len(v),) + self._shape)
            eval_linear(grid, coeffs, v, interp_coeffs)
        else:
            interp_coeffs = self.interpolators[spin](v)

        return interp_coeffs.transpose((1, 2, 0, 3))
