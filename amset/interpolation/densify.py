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
from amset.misc.util import kpoints_to_first_bz, get_dense_kpoint_mesh_spglib, \
    get_symmetry_equivalent_kpoints, SymmetryEquivalizer
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

        self._idxs = np.arange(len(amset_data.full_kpoints))[sort_idx].reshape(
            tuple(self._mesh))
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

        self._grid_weights = np.array(self._grid_weights)

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
                mask = ((band_energies < fd_cutoffs[0] - 0.01 * units.eV) |
                        (band_energies > fd_cutoffs[1] + 0.01 * units.eV))
                ndims[mask] = np.array([0, 0, 0])

            if scale_by_fd_weights:
                # pass
                # w = (fd_weights[..., np.newaxis] + 0.5) / 1.5
                # w = (np.power(fd_weights[..., np.newaxis], 1/10) + 4) / 5
                # w = fd_weights[..., np.newaxis]
                # w = np.log1p(fd_weights[..., np.newaxis])
                # w = np.power(fd_weights[..., np.newaxis], 1/8)

                # t = 0.5
                #
                # w[w > t] = 1
                # w[w <= t] /= t
                # w = np.power(fd_weights[..., np.newaxis], 1/7)
                # print("min w", np.min(w), "max w", np.max(w))
                # ndims *= w
                ndims[fd_weights < 0.05] *= 0
                # ndims *= 1
                # ndims *=
                # ndims /= 2
                # ndims *= np.power(fd_weights[..., np.newaxis], 1/5)
                # ndims *= fd_weights[..., np.newaxis]

            # take the dimensions if they are greater than the current


            # dimensions
            fine_mesh_dims = np.maximum(fine_mesh_dims, ndims)

        # TODO: add test for all zero fine mesh points

        # print(np.max(fine_mesh_dims) / units.eV)
        # fine_mesh_dims = np.floor(fine_mesh_dims / (target_de * units.eV)
        #                           ).astype(int)

        # fine_mesh_dims = np.floor(fine_mesh_dims / (target_de * units.eV))

        # fine_mesh_dims = fine_mesh_dims / (target_de * units.eV)
        dim = fine_mesh_dims.reshape(
            self._amset_data.full_kpoints.shape) / (target_de * units.eV)
        #
        dim = dim[(dim[:, 0] != 0) | (dim[:, 1] != 0) | (dim[:, 2] != 0)]
        dim = np.floor(dim.mean(axis=0)).astype(int)
        dim[dim == 0] = 1
        print(dim)
        # fine_mesh_dims = np.floor(
        #     (fine_mesh_dims / (target_de * units.eV)).mean(axis=-1)).astype(int)

        # print(np.max(fine_mesh_dims))
        # print(np.average(fine_mesh_dims[fine_mesh_dims != 0]))
        # print(fine_mesh_dims)
        additional_kpoints = []
        kpt_count = 0
        # default_weight = 1 / np.product(self._amset_data.kpoint_mesh)
        # weight = 1 / np.product(dim) * default_weight

        # weights = np.full(len(self._amset_data.full_kpoints), default_weight)
        # additional_weights = []

        se = SymmetryEquivalizer(
            self._amset_data.structure, symprec=0.1, tol=1e-6)

        np.random.seed(0)
        fine_mesh_dims[fine_mesh_dims == 0] = 1

        default_kpts = get_dense_kpoint_mesh_spglib(
            [2, 2, 2], shift=0, spg_order=True)
        default_kpts = default_kpts[1:]
        default_kpts /= self._mesh

        for i, j, k in np.ndindex(tuple(self._mesh)):
            d = fine_mesh_dims[i, j, k]

            if all(d == 1):
                kpt_count += 1
                continue

            # if zero in only 1 direction then we have to add at least one
            # point to make a mesh
            # d[d == 1] = 2
            # m = d * 2
            # m[m < 5] = 5
            # m = (m % 2 == 0) + m
            # kpts = get_dense_kpoint_mesh_spglib(d, shift=0, spg_order=True)

            rnd_frac = np.max(self._grid_weights[:, i, j, k])

            # if rnd_frac * np.product(d) > 1:
            if True:
                # kpts = get_dense_kpoint_mesh_spglib(d, shift=0, spg_order=True)
                kpts = get_dense_kpoint_mesh_spglib(dim, shift=0, spg_order=True)
                kpts = kpts[1:]
                kpts /= self._mesh
                kpts += self._grid_kpoints[i, j, k]

                # skip = math.ceil(len(kpts)/(rnd_frac * len(kpts)))

                # kpts = kpts[::skip]

                # if rnd_frac * len(kpts) <= 27:
                #     choice = np.random.choice(
                #         len(kpts),
                #         math.ceil(len(kpts) * rnd_frac))
                #     kpts = kpts[choice]
                # else:
                #     ir_kpoints_idx, ir_to_full_idx, mapping = \
                #         se.get_equivalent_kpoints(kpts)
                #
                #     choice = np.random.choice(
                #         len(ir_kpoints_idx),
                #         math.ceil(len(ir_kpoints_idx) * rnd_frac))
                #     ir_kpts = ir_kpoints_idx[choice]
                #     kpts = kpts[np.isin(mapping, mapping[ir_kpts])]

            else:

                kpts = default_kpts.copy()
                kpts += self._grid_kpoints[i, j, k]

            additional_kpoints.append(kpts)

        if additional_kpoints:
            additional_kpoints = np.concatenate(additional_kpoints)

        print(len(additional_kpoints))

        return additional_kpoints

    def densify(self, target_de: float = gdefaults["fine_mesh_de"],
                symprec: float = pdefaults["symprec"]):
        logger.info("Densifying band structure around Fermi integrals")
        log_list(["fine mesh de: {} eV".format(target_de)])

        additional_kpoints = self.get_fine_mesh(target_de=target_de)

        extra_kpoints = kpoints_to_first_bz(additional_kpoints)

        # rnd_frac = 0.05
        # choice = np.random.choice(
        #     len(extra_kpoints),
        #     math.ceil(len(extra_kpoints) * rnd_frac))
        # extra_kpoints = extra_kpoints[choice]
        # print(len(extra_kpoints))

        # _, _, ir_kpoints_idx, ir_to_full_idx, mapping, _ = get_symmetry_equivalent_kpoints(
        #     self._amset_data.structure, extra_kpoints, symprec=0.1, tol=1e-6,
        #     return_inverse=True, time_reversal_symmetry=True)
        #
        # rnd_frac = 0.1
        # choice = np.random.choice(
        #     len(ir_kpoints_idx),
        #     math.ceil(len(ir_kpoints_idx) * rnd_frac))
        # ir_kpts = ir_kpoints_idx[choice]
        # extra_kpoints = extra_kpoints[np.isin(mapping, mapping[ir_kpts])]
        # print(len(extra_kpoints))
        #
        # _, _, ir_kpoints_idx, ir_to_full_idx, mapping, _ = get_symmetry_equivalent_kpoints(
        #     self._amset_data.structure, kpts, symprec=0.1, tol=1e-6,
        #     return_inverse=True, time_reversal_symmetry=True)
        #
        # rnd_frac = 0.2
        # choice = np.random.choice(
        #     len(ir_kpoints_idx),
        #     math.ceil(len(ir_kpoints_idx) * rnd_frac))
        # ir_kpts = ir_kpoints_idx[choice]
        #
        # kpts = kpts[np.isin(mapping, mapping[ir_kpts])]

        # ir_kpoints, _, ir_kpoints_idx, ir_to_full_idx, mapping = \
        #     get_symmetry_equivalent_kpoints(
        #         self._amset_data.structure, extra_kpoints, symprec=symprec,
        #         return_inverse=True)
        # print("n_ir_kpoints", len(ir_kpoints))

        # from pymatgen import Structure
        # s = Structure(self._amset_data.structure.lattice.reciprocal_lattice.matrix * 10,
        #               ['H'] * len(extra_kpoints),
        #               extra_kpoints * 5 + 0.5, coords_are_cartesian=False)
        # s.to(filename="test2.vasp", fmt="poscar")

        skip = 10 / self._interpolater.interpolation_factor
        # skip = None
        energies, vvelocities, projections, mapping_info = \
            self._interpolater.get_energies(
                extra_kpoints, energy_cutoff=self._energy_cutoff,
                bandgap=self._bandgap, scissor=self._scissor,
                return_velocity=True, return_effective_mass=False,
                return_projections=True, atomic_units=True,
                return_vel_outer_prod=True, skip_coefficients=skip,
                return_kpoint_mapping=True, symprec=symprec)

        voronoi = PeriodicVoronoi(
            self._amset_data.structure.lattice.reciprocal_lattice,
            self._amset_data.full_kpoints,
            self._amset_data.kpoint_mesh,
            extra_kpoints)
        kpoint_weights = voronoi.compute_volumes()

        # note k-point weights is for all k-points, whereas the other properties
        # are just for the additional k-points
        return (extra_kpoints, energies, vvelocities, projections,
                kpoint_weights, mapping_info["ir_kpoints_idx"],
                mapping_info["ir_to_full_idx"])

