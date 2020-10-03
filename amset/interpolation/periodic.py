import logging

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from amset.constants import numeric_types
from amset.electronic_structure.kpoints import (
    get_mesh_from_kpoint_numbers,
    kpoints_to_first_bz,
)

try:
    import interpolation
except ImportError:
    interpolation = None


logger = logging.getLogger(__name__)


class PeriodicLinearInterpolator(object):
    def __init__(self, kpoints, data):
        grid_kpoints, mesh_dim, sort_idx = self._grid_kpoints(kpoints)
        self._setup_interpolators(data, grid_kpoints, mesh_dim, sort_idx)

    def _setup_interpolators(self, data, grid_kpoints, mesh_dim, sort_idx):
        x = grid_kpoints[:, 0, 0, 0]
        y = grid_kpoints[0, :, 0, 1]
        z = grid_kpoints[0, 0, :, 2]

        self.nbands = {s: c.shape[0] for s, c in data.items()}
        self.interpolators = {}
        for spin, spin_data in data.items():
            data_shape = spin_data.shape[2:]
            nbands = self.nbands[spin]
            self.data_shape = data_shape

            # sort the data then reshape them into the grid. The data
            # can now be indexed as data[iband][ikx][iky][ikz]
            sorted_data = spin_data[:, sort_idx]
            grid_shape = (nbands,) + mesh_dim + data_shape
            grid_data = sorted_data.reshape(grid_shape)

            # wrap the data to account for PBC
            pad_size = ((0, 0), (1, 1), (1, 1), (1, 1)) + ((0, 0),) * len(data_shape)
            grid_data = np.pad(grid_data, pad_size, mode="wrap")

            if nbands == 1:
                # this can cause a bug in RegularGridInterpolator. Have to fake
                # having at least two bands
                nbands = 2
                grid_data = np.tile(grid_data, (2, 1, 1, 1) + (1,) * len(data_shape))

            if interpolation is not None:
                from interpolation.splines import UCGrid

                grid = UCGrid(
                    (0, nbands - 1, nbands),
                    (x[0], x[-1], len(x)),
                    (y[0], y[-1], len(y)),
                    (z[0], z[-1], len(z)),
                )
                # flatten remaining axes
                grid_shape = grid_data.shape[:4] + (-1,)
                self.interpolators[spin] = (grid, grid_data.reshape(grid_shape))
            else:
                logger.warning(
                    "Install the 'interpolation' package for improved performance: "
                    "https://pypi.org/project/interpolation"
                )
                interp_range = (np.arange(nbands), x, y, z)

                self.interpolators[spin] = RegularGridInterpolator(
                    interp_range,
                    grid_data,
                    bounds_error=False,
                    fill_value=None,  # , method="nearest"
                )

    @staticmethod
    def _grid_kpoints(kpoints):
        # k-points has to cover the full BZ
        kpoints = kpoints_to_first_bz(kpoints)
        mesh_dim = get_mesh_from_kpoint_numbers(kpoints)
        if np.product(mesh_dim) != len(kpoints):
            raise ValueError("K-points do not cover full Brillouin zone.")

        kpoints = np.around(kpoints, 5)

        # get the indices to sort the k-points on the Z, then Y, then X columns
        sort_idx = np.lexsort((kpoints[:, 2], kpoints[:, 1], kpoints[:, 0]))

        # put the kpoints into a 3D grid so that they can be indexed as
        # kpoints[ikx][iky][ikz] = [kx, ky, kz]
        grid_kpoints = kpoints[sort_idx].reshape(mesh_dim + (3,))

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
        return grid_kpoints, mesh_dim, sort_idx

    def interpolate(self, spin, bands, kpoints):
        v = np.concatenate([np.asarray(bands)[:, None], np.asarray(kpoints)], axis=1)

        if interpolation:
            from interpolation.splines import eval_linear
            from interpolation.splines import extrap_options as xto

            grid, data = self.interpolators[spin]
            if np.iscomplexobj(data):
                # only allows interpolating floats, so have to separate real and imag
                interp_data = np.empty((len(v),) + self.data_shape, dtype=np.complex)
                interp_data.real = eval_linear(grid, data.real, v, xto.LINEAR).reshape(
                    -1, *self.data_shape
                )
                interp_data.imag = eval_linear(grid, data.imag, v, xto.LINEAR).reshape(
                    -1, *self.data_shape
                )
            else:
                interp_data = eval_linear(grid, data, v, xto.LINEAR).reshape(
                    -1, *self.data_shape
                )
        else:
            interp_data = self.interpolators[spin](v)

        return interp_data


def group_bands_and_kpoints(band_a, kpoint_a, band_b, kpoint_b):
    kpoint_a = np.asarray(kpoint_a)
    kpoint_b = np.asarray(kpoint_b)

    return_single = False
    if isinstance(band_b, numeric_types):
        # only one band index given
        if len(kpoint_b.shape) > 1:
            # multiple k-point indices given
            band_b = np.array([band_b] * len(kpoint_b))
        else:
            band_b = np.array([band_b])
            kpoint_b = [kpoint_b]
            return_single = True

    else:
        band_b = np.asarray(band_b)

    bands = np.concatenate([[band_a], band_b])
    kpoints = np.concatenate([[kpoint_a], kpoint_b])

    return bands, kpoints, return_single
