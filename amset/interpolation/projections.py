import logging

import numba
import numpy as np
from interpolation.splines import eval_linear
from interpolation.splines import extrap_options as xto
from pymatgen import Spin
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.util.coord import pbc_diff

from amset.constants import defaults
from amset.electronic_structure.common import get_ibands, get_vb_idx
from amset.electronic_structure.symmetry import expand_kpoints
from amset.interpolation.periodic import (
    PeriodicLinearInterpolator,
    group_bands_and_kpoints,
)

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

logger = logging.getLogger(__name__)


class ProjectionOverlapCalculator(PeriodicLinearInterpolator):
    def __init__(self, nbands, data_shape, interpolators, rotation_mask, band_centers):
        super().__init__(nbands, data_shape, interpolators)
        self.rotation_mask = rotation_mask
        self.band_centers = band_centers

    def to_reference(self):
        interpolator_references = self._interpolators_to_reference()
        return (
            self.nbands,
            self.data_shape,
            interpolator_references,
            self.rotation_mask,
            self.band_centers,
        )

    @classmethod
    def from_data(
        cls, kpoints, data, rotation_mask=None, band_centers=None, gaussian=False
    ):
        logger.info("Initializing orbital overlap calculator")
        if rotation_mask is None or band_centers is None:
            raise ValueError("rotation_mask and band_centers are both required.")

        grid_kpoints, mesh_dim, sort_idx = cls._grid_kpoints(kpoints)
        nbands, data_shape, interpolators = cls._setup_interpolators(
            data, grid_kpoints, mesh_dim, sort_idx, gaussian
        )
        return cls(nbands, data_shape, interpolators, rotation_mask, band_centers)

    @classmethod
    def from_band_structure(
        cls,
        band_structure: BandStructure,
        energy_cutoff=defaults["energy_cutoff"],
        symprec=defaults["symprec"],
    ):
        kpoints = np.array([k.frac_coords for k in band_structure.kpoints])
        efermi = band_structure.efermi
        structure = band_structure.structure

        full_kpoints, _, _, _, _, ir_to_full_idx = expand_kpoints(
            structure, kpoints, symprec=symprec, return_mapping=True, time_reversal=True
        )

        ibands = get_ibands(energy_cutoff, band_structure)
        vb_idx = get_vb_idx(energy_cutoff, band_structure)

        energies = {s: e[ibands[s]] for s, e in band_structure.bands.items()}
        energies = {s: e[:, ir_to_full_idx] for s, e in energies.items()}
        projections = {s: p[ibands[s]] for s, p in band_structure.projections.items()}

        band_centers = get_band_centers(full_kpoints, energies, vb_idx, efermi)
        rotation_mask = get_rotation_mask(projections)

        full_projections = {}
        for spin, spin_projections in projections.items():
            nbands = spin_projections.shape[0]
            nkpoints = len(full_kpoints)

            spin_projections = spin_projections[:, ir_to_full_idx].reshape(
                (nbands, nkpoints, -1), order="F"
            )
            spin_projections /= np.linalg.norm(spin_projections, axis=2)[..., None]
            spin_projections[np.isnan(spin_projections)] = 0
            full_projections[spin] = spin_projections

        return cls.from_data(
            full_kpoints,
            full_projections,
            rotation_mask=rotation_mask,
            band_centers=band_centers,
        )

    def get_coefficients(self, spin, bands, kpoints):
        return self.interpolate(spin, bands, kpoints)

    def get_overlap(self, spin, band_a, kpoint_a, band_b, kpoint_b):
        bands, kpoints, single_overlap = group_bands_and_kpoints(
            band_a, kpoint_a, band_b, kpoint_b
        )
        grid, data = self.interpolators[spin]
        v = np.concatenate([np.asarray(bands)[:, None], np.asarray(kpoints)], axis=1)

        centers = self.band_centers[spin][band_a]
        center = centers[np.argmin(np.linalg.norm(pbc_diff(centers, kpoint_a), axis=1))]

        overlap = _get_overlap(
            grid, data, v, self.data_shape[0], center, self.rotation_mask
        )

        if single_overlap:
            return overlap[0]
        else:
            return overlap


@numba.njit
def _get_overlap(grid, data, points, ncoeffs, center, rotation_mask):
    initial = eval_linear(grid, data.real, points[0], xto.LINEAR,)

    # this is pbc_diff(kpoint_a, center)
    kpoint_a = points[0, 1:]
    shift_a = kpoint_a - center
    for i in range(3):
        shift_a[i] -= np.round_(shift_a[i])
    shift_a_norm = np.linalg.norm(shift_a)

    res = np.zeros(points.shape[0] - 1)
    inv_rotation_mask = 1 - rotation_mask
    finals = eval_linear(grid, data.real, points[1:], xto.LINEAR,)
    for i in range(1, points.shape[0]):

        # this is pbc_diff(kpoint_b, center)
        kpoint_b = points[i, 1:]
        shift_b = kpoint_b - center
        for j in range(3):
            shift_b[j] -= np.round_(shift_b[j])
        shift_b_norm = np.linalg.norm(shift_b)

        if shift_a_norm == 0 or shift_b_norm == 0:
            cosine = 1
        else:
            cosine = np.dot(shift_a, shift_b) / (shift_a_norm * shift_b_norm)

        # weight the angle masks by the contribution of the transition in that direction
        # use the mask to get the angle scaling factor which gives how much the angle
        # is applied to each projection. I.e., if the scaling factor is 1 for a
        # particular projection, then the projection is scaled by 1 * the angle. If the
        # scaling factor is 0, the projection is scaled by 0 * the angle.
        # this allows us to make s orbitals immune to the cosine weight
        sum_ = 0
        for j in range(10):
            p = initial[j] * finals[i, j]
            sum_ += (p * inv_rotation_mask[j]) + (p * rotation_mask[j] * cosine)
        res[i - 1] = sum_ ** 2
    return res


def get_rotation_mask(projections):
    nprojections, natoms = projections[Spin.up].shape[2:]
    mask = np.ones(nprojections)
    mask[0] = 0
    return np.tile(mask, natoms)


# def cosine(v1, v2):
#     # v2 can be list of vectors or single vector
#
#     return_single = False
#     if len(v2.shape) == 1:
#         v2 = v2[None, :]
#         return_single = True
#
#     v1_norm = np.linalg.norm(v1)
#     v2_norm = np.linalg.norm(v2, axis=1)
#
#     v_dot = np.dot(v1, v2.T)
#     v_angle = v_dot / (v1_norm * v2_norm)
#     v_angle[np.isnan(v_angle)] = 1
#
#     if return_single:
#         return v_angle[0]
#     else:
#         return v_angle


def get_band_centers(kpoints, energies, vb_idx, efermi, tol=0.0001):
    band_centers = {}

    for spin, spin_energies in energies.items():
        spin_centers = []
        for i, band_energies in enumerate(spin_energies):
            if vb_idx is None:
                # handle metals
                k_idxs = np.abs(band_energies - efermi) < tol

            elif i <= vb_idx[spin]:
                k_idxs = (np.max(band_energies) - band_energies) < tol

            else:
                k_idxs = (band_energies - np.min(band_energies)) < tol

            if len(k_idxs) > 0:
                k_idxs = [0]

            spin_centers.append(kpoints[k_idxs])
        band_centers[spin] = spin_centers
    return band_centers
