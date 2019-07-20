import itertools
import logging
import math
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d

from BoltzTraP2 import units
from BoltzTraP2.bandlib import DOS, dFDde
from scipy.ndimage import uniform_filter1d

from amset import amset_defaults as defaults
from amset.data import AmsetData
from amset.interpolation.interpolate import Interpolater
from amset.misc.log import log_list
from amset.misc.util import kpoints_to_first_bz
from amset.interpolation.voronoi import PeriodicVoronoi

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"
__date__ = "June 21, 2019"

logger = logging.getLogger(__name__)
gdefaults = defaults["general"]
pdefaults = defaults["performance"]


class BandDensifier(object):

    def __init__(self,
                 interpolater: Interpolater,
                 amset_data: AmsetData,
                 dos_estep: float = pdefaults["dos_estep"],
                 energy_cutoff: Optional[float] = None,
                 scissor: float = None,
                 bandgap: float = None):
        if amset_data.fermi_levels is None:
            raise RuntimeError(
                "amset_data doesn't contain Fermi level information")

        self._interpolater = interpolater
        self._amset_data = amset_data
        self._dos_estep = dos_estep
        self._energy_cutoff = energy_cutoff
        self._bandgap = bandgap
        self._scissor = scissor
        self._mesh = amset_data.kpoint_mesh

        # get the indices to sort the kpoints from on the Z, then Y,
        # then X columns
        sort_idx = np.lexsort((self._mesh[:, 2],
                               self._mesh[:, 1],
                               self._mesh[:, 0]))

        self._grid_energies = []
        for spin in amset_data.spins:
            # sort the energies then reshape them into the grid. The energies
            # can now be indexed as energies[ikx][iky][ikz]
            sorted_energies = amset_data.energies[spin][:, sort_idx]
            self._grid_energies.extend(sorted_energies.reshape(
                (-1, ) + tuple(self._mesh)))

        # put the kpoints into a 3D grid so that they can be indexed as
        # kpoints[ikx][iky][ikz] = [kx, ky, kz]
        self._grid_kpoints = amset_data.full_kpoints[sort_idx].reshape(
            tuple(self._mesh) + (3,))

    def densify(self, target_de: float = gdefaults["target_de"]):
        logger.info("Densifying band structure around Fermi integrals")

        fine_mesh_dims = np.zeros(self._grid_kpoints.shape)

        for band_energies in self._grid_energies:
            # effectively make a supercell of the energies on the regular grid
            # containing one extra plane of energies per dimension, on either
            # face of the 3D energy mesh
            band_energies = np.pad(band_energies, 1, "wrap")

            x_diffs = np.abs(np.diff(band_energies, axis=2))
            y_diffs = np.abs(np.diff(band_energies, axis=1))
            z_diffs = np.abs(np.diff(band_energies, axis=0))

            # remove the diffs related to the extra padding
            x_diffs = x_diffs[1:-1, 1:-1, :].astype(float)
            y_diffs = y_diffs[1:-1, :, 1:-1].astype(float)
            z_diffs = z_diffs[:, 1:-1, 1:-1].astype(float)

            #  calculate moving averages
            x_diff_averages = uniform_filter1d(x_diffs, 2, axis=2)[:, :, 1:]
            y_diff_averages = uniform_filter1d(y_diffs, 2, axis=1)[:, 1:, :]
            z_diff_averages = uniform_filter1d(z_diffs, 2, axis=0)[1:, :, :]

            # stack the averages to get the formatted energy different array
            ndims = np.stack((x_diff_averages, y_diff_averages,
                              z_diff_averages), axis=-1)

            # take the dimensions if they are greater than the current
            # dimensions
            fine_mesh_dims = np.maximum(fine_mesh_dims, ndims)

        fine_mesh_dims = np.floor(fine_mesh_dims / target_de).astype(int)

        additional_kpoints = []

        for i, j, k in np.ndindex(tuple(dim)):
            d = ndims[i, j, k]
            if all(d == 0):
                continue
            d[d == 0] = 1

            #     kpts = get_dense_kpoint_mesh(d)
            kpts = get_dense_kpoint_mesh_spglib(d + 1)
            kpts /= dim
            kpts += mesh_grid[i, j, k]
            additional_kpoints.append(kpts)

        if additional_kpoints:
            additional_kpoints = np.concatenate(additional_kpoints)

        # add additional points in concenctric spheres around the k-points
        #
        extra_kpoints = np.concatenate(
            [_generate_points(n_extra, max_dist) + kpoint
             for kpoint, n_extra in zip(k_coords, n_points_per_kpoint)])
        log_list(["# extra kpoints: {}".format(total_points),
                  "max frac k-distance: {:.5f}".format(max_dist)])

        # from monty.serialization import dumpfn
        # dumpfn(extra_kpoints, "extra_kpoints.json")

        extra_kpoints = kpoints_to_first_bz(extra_kpoints)

        skip = 5 / self._interpolater.interpolation_factor
        energies, vvelocities, projections = self._interpolater.get_energies(
            extra_kpoints, energy_cutoff=self._energy_cutoff,
            bandgap=self._bandgap, scissor=self._scissor,
            return_velocity=True, return_effective_mass=False,
            return_projections=True, atomic_units=True,
            return_vel_outer_prod=True, skip_coefficients=skip)

        voronoi = PeriodicVoronoi(
            self._amset_data.structure.lattice.reciprocal_lattice,
            self._amset_data.full_kpoints,
            self._amset_data.kpoint_mesh,
            extra_kpoints)
        kpoint_weights = voronoi.compute_volumes()

        # note k-point weights is for all k-points, whereas the other properties
        # are just for the additional k-points
        return (extra_kpoints, energies, vvelocities, projections,
                kpoint_weights)


def _generate_points_cube(n_extra_kpoints: int, max_distance: float):
    n_p = math.ceil(np.cbrt(n_extra_kpoints) / 2)
    # forward_p = np.geomspace(max_distance/20, max_distance, n_p)
    forward_p = np.linspace(max_distance/20, max_distance, n_p)
    all_p = np.concatenate((-forward_p, forward_p))
    return np.array(list(itertools.product(all_p, all_p, all_p)))


def _generate_points(n_extra_kpoints: int, max_distance: float,
                     min_n_points_per_sphere: int = 32,
                     max_n_points_per_sphere: int = 256,
                     default_n_spheres: int = 5):

    # each sphere must contain at least min_n_points_per_sphere
    actual_n_spheres = min(
        default_n_spheres, math.ceil(n_extra_kpoints / min_n_points_per_sphere))

    actual_n_spheres = max(
        actual_n_spheres, math.ceil(n_extra_kpoints / max_n_points_per_sphere))

    # generate radii for spheres, getting logarithmically further away, up to
    # max_distance. Note, not all spheres may be used
    if actual_n_spheres > default_n_spheres:
        radii = np.geomspace(max_distance/10, max_distance, actual_n_spheres)
    else:
        radii = np.geomspace(max_distance/10, max_distance, default_n_spheres)

    n_points_per_sphere = math.ceil(n_extra_kpoints / actual_n_spheres)

    sphere_points = np.concatenate([
        sunflower_sphere(radius=r, samples=n)
        for r, n in zip(radii, [n_points_per_sphere] * n_points_per_sphere)])

    return sphere_points[:n_extra_kpoints]


def fibonacci_sphere(radius=1, samples=1, randomize=False):
    rnd = 1.
    if randomize:
        rnd = np.random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x, y, z])

    return np.array(points) * radius


def sunflower_sphere(radius=1, samples=1):
    indices = np.arange(0, samples) + 0.5

    phi = np.arccos(1 - 2 * indices / samples)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    return np.stack((x, y, z), axis=1) * radius
