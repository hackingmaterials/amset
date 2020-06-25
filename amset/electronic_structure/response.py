import logging

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from BoltzTraP2 import units
from amset.electronic_structure.fd import fd, dfdde
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
    def __init__(self, kpoints, kpoint_mesh, coefficients, velocities, rates):
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
        self.vel_inter = {}
        self.rate_inter = {}
        for spin, spin_coefficients in coefficients.items():
            nbands = spin_coefficients.shape[0]
            self._shape = spin_coefficients.shape[2:]

            # sort the coefficients then reshape them into the grid. The coefficients
            # can now be indexed as coefficients[iband][ikx][iky][ikz]
            sorted_coefficients = spin_coefficients[:, sort_idx]
            grid_shape = (nbands,) + kpoint_mesh + spin_coefficients.shape[2:]
            grid_coefficients = sorted_coefficients.reshape(grid_shape)
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

            sorted_velocities = velocities[spin][:, sort_idx]
            grid_shape = (nbands,) + kpoint_mesh + (3, )
            grid_velocities = sorted_velocities.reshape(grid_shape)
            grid_velocities = np.pad(
                grid_velocities,
                ((0, 0), (1, 1), (1, 1), (1, 1), (0, 0)),
                mode="wrap",
            )

            if nbands == 1:
                nbands = 2
                grid_velocities = np.tile(grid_velocities, (2, 1, 1, 1, 1, 1, 1))

            sorted_rates = rates[spin][:, sort_idx]
            grid_shape = (nbands,) + kpoint_mesh + rates[spin].shape[2:]
            grid_rates = sorted_rates.reshape(grid_shape)
            grid_rates = np.pad(
                grid_rates,
                ((0, 0), (1, 1), (1, 1), (1, 1), (0, 0), (0, 0)),
                mode="wrap",
            )

            if nbands == 1:
                nbands = 2
                # remember to change this
                grid_rates = np.tile(grid_rates, (2, 1, 1, 1, 1, 1, 1))

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
                self.vel_inter[spin] = (grid, grid_velocities)
                self.rate_inter[spin] = (grid, grid_rates)
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

    def get_coefficients(self, spin, bands, kpoints, energy, amset_data):
        v = np.concatenate([np.asarray(bands)[:, None], np.asarray(kpoints)], axis=1)

        dfde = _get_fd(energy, amset_data)[None, :, :, None]

        if eval_linear:
            grid, coeffs = self.interpolators[spin]
            # coeffs = 1/(coeffs + 1e-40)
            # coeffs[np.isnan(coeffs)] = 0
            coeffs = coeffs + 1e-40

            mag = np.linalg.norm(coeffs, axis=-1)
            unit_v = coeffs / mag[..., None]
            unit_v[np.isnan(unit_v)] = 0
            # unit_v = np.sign(coeffs)

            mag = np.log(1/mag)
            # mag = np.log(1/np.abs(coeffs))

            # mag[np.isnan(mag)] = -20
            interp_coeffs = np.zeros((len(v),) + self._shape)
            eval_linear(grid, unit_v, v, interp_coeffs)

            interp_mag = np.zeros((len(v),) + self._shape[:-1])
            eval_linear(grid, mag, v, interp_mag)

            interp_mag = 1/np.exp(interp_mag) - 1e-40
            interp_coeffs *= interp_mag[..., None]
            # print(v[:20])
            # interp_coeffs *= 0.9

            # test = np.zeros((len(v),) + self._shape)
            # eval_linear(grid, coeffs - 1e-40, v, test)
            # print("test", test[:10, 0, 0])
            # print("interp", interp_coeffs[:10, 0, 0])

            # coeffs = 1/coeffs

            # interp_coeffs = np.zeros((len(v),) + self._shape)
            # eval_linear(grid, coeffs, v, interp_coeffs)
            #
            # interp_coeffs = (1/(interp_coeffs)) - 1e-40
            # interp_coeffs[np.isnan(interp_coeffs)] = 0

            grid, vel = self.vel_inter[spin]
            interp_vel = np.zeros((len(v), 3))
            eval_linear(grid, vel, v, interp_vel)
            interp_vel = interp_vel[:, None, None, :]

            grid, rate = self.rate_inter[spin]
            # rate = np.log(rate)
            # rate[np.isnan(rate)] = -20
            rate[np.isnan(rate)] = 0
            interp_rate = np.zeros((len(v),) + self._shape[:2])
            eval_linear(grid, rate, v, interp_rate)
            # lifetimes = 1 / np.exp(interp_rate[..., None])
            lifetimes = 1 / interp_rate[..., None]
            lifetimes[np.isnan(lifetimes)] = 0

            interp_coeffs = (interp_vel * lifetimes * dfde) + lifetimes * interp_coeffs
        else:
            interp_coeffs = self.interpolators[spin](v)

        interp_coeffs[np.isnan(interp_coeffs)] = 0
        interp_coeffs[np.isinf(interp_coeffs)] = 0

        return interp_coeffs.transpose((1, 2, 0, 3))


def _get_fd(energy, amset_data):
    f = np.zeros(amset_data.fermi_levels.shape)

    for n, t in np.ndindex(amset_data.fermi_levels.shape):
        f[n, t] = dfdde(
            energy,
            amset_data.fermi_levels[n, t],
            amset_data.temperatures[t] * units.BOLTZMANN,
        )
    return f
