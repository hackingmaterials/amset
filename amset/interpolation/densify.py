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
from amset.misc.util import kpoints_to_first_bz, get_dense_kpoint_mesh_spglib
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
        sort_idx = np.lexsort((self._amset_data.full_kpoints[:, 2],
                               self._amset_data.full_kpoints[:, 1],
                               self._amset_data.full_kpoints[:, 0]))

        self._grid_energies = []
        self._grid_weights = []
        for spin in amset_data.spins:
            # sort the energies then reshape them into the grid. The energies
            # can now be indexed as energies[ikx][iky][ikz]
            sorted_energies = amset_data.energies[spin][:, sort_idx]
            self._grid_energies.extend(sorted_energies.reshape(
                (-1, ) + tuple(self._mesh)))

            sorted_weights = amset_data.fd_weights[spin][:, sort_idx]
            self._grid_weights.extend(sorted_weights.reshape(
                (-1, ) + tuple(self._mesh)))

        # put the kpoints into a 3D grid so that they can be indexed as
        # kpoints[ikx][iky][ikz] = [kx, ky, kz]
        self._grid_kpoints = amset_data.full_kpoints[sort_idx].reshape(
            tuple(self._mesh) + (3,))

    def get_fine_mesh(self, target_de: float = gdefaults["fine_mesh_de"],
                      scale_by_fd_weights: bool = True):
        fine_mesh_dims = np.zeros(self._grid_kpoints.shape)
        fd_cutoffs = self._amset_data.fd_cutoffs

        for fd_weights, band_energies in zip(
                self._grid_weights, self._grid_energies):
            # effectively make a supercell of the energies on the regular grid
            # containing one extra plane of energies per dimension, on either
            # face of the 3D energy mesh
            pad_energies = np.pad(band_energies, 1, "wrap")

            x_diffs = np.abs(np.diff(pad_energies, axis=2))
            y_diffs = np.abs(np.diff(pad_energies, axis=1))
            z_diffs = np.abs(np.diff(pad_energies, axis=0))

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

            if fd_cutoffs:
                # if the energies do not lie within the Fermi Dirac cutoffs
                # set the dims to 0 as there is no point of interpolating
                # around these k-points
                mask = ((band_energies < fd_cutoffs[0]) |
                        (band_energies > fd_cutoffs[1]))
                ndims[mask] = np.array([0, 0, 0])
                print(self._grid_kpoints[np.invert(mask)])

            if scale_by_fd_weights:
                pass
                # ndims *= np.power(fd_weights[..., np.newaxis], 1/4)
                # ndims *= fd_weights[..., np.newaxis]

            # take the dimensions if they are greater than the current
            # dimensions
            fine_mesh_dims = np.maximum(fine_mesh_dims, ndims)

        # add test for all zero fine mesh points

        # print(np.max(fine_mesh_dims) / units.eV)
        fine_mesh_dims = np.floor(fine_mesh_dims / (target_de * units.eV)
                                  ).astype(int)
        # print(np.max(fine_mesh_dims))
        # print(np.average(fine_mesh_dims[fine_mesh_dims != 0]))
        # print(fine_mesh_dims)
        additional_kpoints = []
        logger.info("Getting ")
        kpt_count = 0
        for i, j, k in np.ndindex(tuple(self._mesh)):
            d = fine_mesh_dims[i, j, k]
            if all(d == 0):
                kpt_count += 1
                continue

            # if zero in only 1 direction then we have to add at least one
            # point to make a mesh
            d[d == 0] = 1


            # mesh = d + 1
            # kx = np.zeros((mesh[0], 3))
            # kx[:, 0] = np.arange(0, mesh[0])
            # kx[:, 0] /= mesh[0]
            # kx[:, 0] -= (1 / mesh[0]) / 2
            #
            # ky = np.zeros((mesh[1], 3))
            # ky[:, 1] = np.arange(0, mesh[1])
            # ky[:, 1] /= mesh[1]
            # ky[:, 1] -= (1 / mesh[1]) / 2
            #
            # kz = np.zeros((mesh[2], 3))
            # kz[:, 2] = np.arange(0, mesh[2])
            # kz[:, 2] /= mesh[2]
            # kz[:, 2] -= (1 / mesh[2]) / 2

            # kpts = np.concatenate([kx, ky, kz])

            #     kpts = get_dense_kpoint_mesh(d)
            m = d + 1
            m = (m % 2 == 0) + m
            kpts = get_dense_kpoint_mesh_spglib(m)
            kpts /= self._mesh
            kpts += self._grid_kpoints[i, j, k]
            additional_kpoints.append(kpts)

        print(np.product(self._mesh) - kpt_count)

        if additional_kpoints:
            additional_kpoints = np.concatenate(additional_kpoints)

        return additional_kpoints

    def densify(self, target_de: float = gdefaults["fine_mesh_de"]):
        logger.info("Densifying band structure around Fermi integrals")

        additional_kpoints = self.get_fine_mesh(target_de=target_de)
        log_list(["# extra kpoints: {}".format(len(additional_kpoints)),
                  "fine mesh de: {} eV".format(target_de)])

        extra_kpoints = kpoints_to_first_bz(additional_kpoints)

        from pymatgen import Structure
        s = Structure(self._amset_data.structure.lattice.reciprocal_lattice.matrix * 10,
                      ['H'] * len(extra_kpoints),
                      (extra_kpoints * 3) + 0.5, coords_are_cartesian=False)

        s.to(filename="test2.vasp", fmt="poscar")

        skip = 50 / self._interpolater.interpolation_factor
        # skip = None
        energies, vvelocities, projections = self._interpolater.get_energies(
            extra_kpoints, energy_cutoff=self._energy_cutoff,
            bandgap=self._bandgap, scissor=self._scissor,
            return_velocity=True, return_effective_mass=False,
            return_projections=True, atomic_units=True,
            return_vel_outer_prod=True, skip_coefficients=skip)
        from pymatgen import Spin
        print(np.min(energies[Spin.up][3]))
        print(np.min(energies[Spin.up][3]) / units.eV)
        print(extra_kpoints)
        print("CBM", np.min(self._amset_data.energies[Spin.up][3] / units.eV))

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

