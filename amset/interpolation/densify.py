import logging
from typing import Optional

import numpy as np

from BoltzTraP2 import units
from scipy.ndimage import maximum_filter1d, maximum_filter

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

_kpt_str = "[{:6.3f} {:6.3f} {:6.3f}]"
_dim_str = "[{:2d} {:2d} {:2d}]"


class BandDensifier(object):
    def __init__(
        self,
        interpolater: Interpolater,
        amset_data: AmsetData,
        dos_estep: float = pdefaults["dos_estep"],
        energy_cutoff: Optional[float] = None,
        inverse_screening_length_sq: Optional[float] = None
    ):
        if amset_data.fermi_levels is None:
            raise RuntimeError("amset_data doesn't contain Fermi level information")

        self._interpolater = interpolater
        self._amset_data = amset_data
        self._dos_estep = dos_estep
        self._energy_cutoff = energy_cutoff
        self._mesh = tuple(amset_data.kpoint_mesh)

        if inverse_screening_length_sq:
            self._minimum_dim = get_minimum_imp_mesh(
                self._amset_data, inverse_screening_length_sq)
        else:
            self._minimum_dim = None

        # get the indices to sort the kpoints from on the Z, then Y,
        # then X columns
        sort_idx = np.lexsort(
            (
                self._amset_data.full_kpoints[:, 2],
                self._amset_data.full_kpoints[:, 1],
                self._amset_data.full_kpoints[:, 0],
            )
        )

        nkpoints = len(amset_data.full_kpoints)
        self._idxs = np.arange(nkpoints)[sort_idx].reshape(self._mesh)
        self._grid_energies = []
        for spin in amset_data.spins:
            # sort the energies then reshape them into the grid. The energies
            # can now be indexed as energies[ikx][iky][ikz]
            sorted_energies = amset_data.energies[spin][:, sort_idx]
            self._grid_energies.extend(sorted_energies.reshape((-1,) + self._mesh))

        # put the kpoints into a 3D grid so that they can be indexed as
        # kpoints[ikx][iky][ikz] = [kx, ky, kz]
        self._grid_kpoints = amset_data.full_kpoints[sort_idx].reshape(
            self._mesh + (3,)
        )

    def get_fine_mesh(
        self,
        target_de: float = gdefaults["fine_mesh_de"],
        minimum_dim: Optional[np.ndarray] = None,
    ):
        fine_mesh_dims = np.zeros(self._grid_kpoints.shape)
        fd_cutoffs = self._amset_data.fd_cutoffs

        for band_energies in self._grid_energies:
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

            #  calculate moving maxes
            x_diff_averages = maximum_filter1d(x_diffs, 2, axis=2)[:, :, 1:]
            y_diff_averages = maximum_filter1d(y_diffs, 2, axis=1)[:, 1:, :]
            z_diff_averages = maximum_filter1d(z_diffs, 2, axis=0)[1:, :, :]

            # stack the averages to get the formatted energy different array
            ndims = np.stack(
                (z_diff_averages, y_diff_averages, x_diff_averages), axis=-1
            )
            if fd_cutoffs:
                # if the energies do not lie within the Fermi Dirac cutoffs
                # set the dims to 0 as there is no point of interpolating
                # around these k-points
                # mask = (band_energies < fd_cutoffs[0] - 0.01 * units.eV) | (
                #     band_energies > fd_cutoffs[1] + 0.01 * units.eV
                # )
                # ndims[mask] = np.array([0, 0, 0])

                mask = (band_energies > fd_cutoffs[0]) & (band_energies < fd_cutoffs[1])

                # expand the mask as the Fermi surface defined by the cutoffs
                # may also fall in the volume defined by an adjacent k-point
                # even if the k-point itself is not included in the Fermi
                # surface.
                mask = maximum_filter(mask, footprint=np.ones((3, 3, 3)), mode="wrap")
                ndims[~mask] = np.array([0, 0, 0])

            # take the dimensions if they are greater than the current
            # dimensions
            fine_mesh_dims = np.maximum(fine_mesh_dims, ndims)

        # TODO: add test for all zero fine mesh points
        fine_mesh_dims = np.ceil(fine_mesh_dims / (target_de * units.eV)).astype(int)
        zeros_and_ones = ((fine_mesh_dims == 0) | (fine_mesh_dims == 1)).all(axis=3)

        if minimum_dim is not None:
            fine_mesh_dims = np.maximum(fine_mesh_dims, minimum_dim)

        fine_mesh_dims[zeros_and_ones] = 1
        fine_mesh_dims[fine_mesh_dims == 0] = 1

        additional_kpoints = []
        kpoint_log = []

        for i, j, k in np.ndindex(self._mesh):
            dim = fine_mesh_dims[i, j, k]

            if (dim == 1).all():
                continue

            grid_point = self._grid_kpoints[i, j, k]
            kpt_str = _kpt_str.format(*grid_point)
            dim_str = _dim_str.format(*dim)
            kpoint_log.append("kpt: {}, mesh: {}".format(kpt_str, dim_str))
            kpoints = get_dense_kpoint_mesh_spglib(dim, spg_order=True)

            # remove [0, 0, 0] as other wise this will overlap with the
            # existing mesh
            kpoints = kpoints[1:]
            kpoints /= self._mesh
            kpoints += grid_point
            additional_kpoints.append(kpoints)

        n_points = len(additional_kpoints)
        if additional_kpoints:
            additional_kpoints = np.concatenate(additional_kpoints)
        logger.info(
            "Densified {} kpoints with {} extra points".format(
                n_points, len(additional_kpoints)
            )
        )
        log_list(kpoint_log, level=logging.DEBUG)

        return kpoints_to_first_bz(additional_kpoints)

    def densify(
        self,
        target_de: float = gdefaults["fine_mesh_de"],
        symprec: float = pdefaults["symprec"],
    ):
        densify_info = ["fine mesh de: {} eV".format(target_de)]

        if self._minimum_dim is not None:
            dim_str = _dim_str.format(*self._minimum_dim)
            densify_info.append("minimum dim for IMP: {}".format(dim_str))

        logger.info("Densifying band structure around Fermi integrals")
        log_list(densify_info)

        additional_kpoints = self.get_fine_mesh(
            target_de=target_de, minimum_dim=self._minimum_dim
        )

        # have to use amset_data scissor and not the user specified band gap,
        # as there is no guarantee that the extra k-points will go through the
        # CBM & VBM
        energies, vvelocities, projections, mapping_info = self._interpolater.get_energies(
            additional_kpoints,
            energy_cutoff=self._energy_cutoff,
            scissor=self._amset_data.scissor / units.eV,
            return_velocity=True,
            return_effective_mass=False,
            return_projections=True,
            atomic_units=True,
            return_vel_outer_prod=True,
            return_kpoint_mapping=True,
            symprec=symprec,
        )

        voronoi = PeriodicVoronoi(
            self._amset_data.structure.lattice.reciprocal_lattice,
            self._amset_data.full_kpoints,
            self._amset_data.kpoint_mesh,
            additional_kpoints,
        )
        kpoint_weights = voronoi.compute_volumes()

        # note k-point weights is for all k-points, whereas the other properties
        # are just for the additional k-points
        return (
            additional_kpoints,
            energies,
            vvelocities,
            projections,
            kpoint_weights,
            mapping_info["ir_kpoints_idx"],
            mapping_info["ir_to_full_idx"],
        )


def get_minimum_imp_mesh(amset_data, inverse_screening_length_sq):
    # todo: explain this magic number
    # conv_dx = 0.3727593720314942
    # conv_dx = 0.29470517025518095
    conv_dx = 0.14563484775012445
    dk_sq = conv_dx * inverse_screening_length_sq
    dk = np.sqrt(dk_sq)
    dk_angstrom = dk * 0.1
    k_frac = np.array([dk_angstrom, dk_angstrom, dk_angstrom])
    k_cart = amset_data.structure.lattice.reciprocal_lattice.get_fractional_coords(
        k_frac)
    dim = (1 / k_cart)

    # now get factor of dim in current mesh
    return np.ceil(dim / amset_data.kpoint_mesh).astype(int)
