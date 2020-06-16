import logging

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from amset.constants import defaults, numeric_types
from amset.electronic_structure.common import get_ibands, get_vb_idx
from amset.electronic_structure.kpoints import (
    expand_kpoints,
    get_mesh_dim_from_kpoints,
    kpoints_to_first_bz,
)
from amset.electronic_structure.wavefunction import load_coefficients
from pymatgen import Spin
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.util.coord import pbc_diff

try:
    from interpolation.splines import eval_linear, UCGrid
    from interpolation.splines import extrap_options as xto
except ImportError:
    eval_linear = None

# eval_linear = None
__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

_p_mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
_d_mask = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

logger = logging.getLogger(__name__)


class WavefunctionOverlapCalculator(object):
    def __init__(self, structure, kpoints, coefficients):
        logger.info("Initializing wavefunction overlap calculator")
        self.structure = structure

        # k-points has to cover the full BZ
        kpoints = kpoints_to_first_bz(kpoints)
        mesh_dim = get_mesh_dim_from_kpoints(kpoints, tol=1e-4)

        round_dp = int(np.log10(1 / 1e-6))
        kpoints = np.round(kpoints, round_dp)

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

        x = grid_kpoints[:, 0, 0, 0]
        y = grid_kpoints[0, :, 0, 1]
        z = grid_kpoints[0, 0, :, 2]

        self.nbands = {s: c.shape[0] for s, c in coefficients.items()}

        self.interpolators = {}
        for spin, spin_coefficients in coefficients.items():
            nbands = spin_coefficients.shape[0]
            ncoefficients = spin_coefficients.shape[-1]

            # sort the coefficients then reshape them into the grid. The coefficients
            # can now be indexed as coefficients[iband][ikx][iky][ikz]
            sorted_coefficients = spin_coefficients[:, sort_idx]
            grid_shape = (nbands,) + mesh_dim + (ncoefficients,)
            grid_coefficients = sorted_coefficients.reshape(grid_shape)

            # wrap the coefficients to account for PBC
            grid_coefficients = np.pad(
                grid_coefficients, ((0, 0), (1, 1), (1, 1), (1, 1), (0, 0)), mode="wrap"
            )

            if nbands == 1:
                # this can cause a bug in RegularGridInterpolator. Have to fake
                # having at least two bands
                nbands = 2
                grid_coefficients = np.tile(grid_coefficients, (2, 1, 1, 1, 1))

            if eval_linear:
                grid = UCGrid(
                    (0, nbands - 1, nbands),
                    (x[0], x[-1], len(x)),
                    (y[0], y[-1], len(y)),
                    (z[0], z[-1], len(z)),
                )
                self.interpolators[spin] = (grid, grid_coefficients)
            else:
                interp_range = (np.arange(nbands), x, y, z)

                self.interpolators[spin] = RegularGridInterpolator(
                    interp_range,
                    grid_coefficients,
                    bounds_error=False,
                    fill_value=None,  # , method="nearest"
                )

    @classmethod
    def from_file(cls, filename):
        coeff, kpoints, structure = load_coefficients(filename)
        return cls(structure, kpoints, coeff)

    def get_coefficients(self, spin, bands, kpoints):
        v = np.concatenate([np.asarray(bands)[:, None], np.asarray(kpoints)], axis=1)

        if eval_linear:
            grid, coeffs = self.interpolators[spin]

            # only allows interpolating floats, so have to separate real and imag parts
            interp_coeffs = np.empty((len(v), coeffs.shape[-1]), dtype=np.complex)
            interp_coeffs.real = eval_linear(grid, coeffs.real, v, xto.LINEAR)
            interp_coeffs.imag = eval_linear(grid, coeffs.imag, v, xto.LINEAR)
        else:
            interp_coeffs = self.interpolators[spin](v)

        interp_coeffs /= np.linalg.norm(interp_coeffs, axis=-1)[:, None]
        return interp_coeffs

    def get_overlap(self, spin, band_a, kpoint_a, band_b, kpoint_b):
        # k-points should be in fractional
        kpoint_a = np.asarray(kpoint_a)
        kpoint_b = np.asarray(kpoint_b)
        v1 = np.array([[band_a] + kpoint_a.tolist()])

        single_overlap = False
        if isinstance(band_b, numeric_types):
            # only one band index given
            if len(kpoint_b.shape) > 1:
                # multiple k-point indices given
                band_b = np.array([band_b] * len(kpoint_b))

            else:
                band_b = np.array([band_b])
                kpoint_b = [kpoint_b]
                single_overlap = True

        else:
            band_b = np.asarray(band_b)

        # v2 now has shape of (nkpoints_b, 4)
        v2 = np.concatenate([band_b[:, None], kpoint_b], axis=1)

        # get a big array of all the k-points to interpolate
        all_v = np.vstack([v1, v2])

        # get the interpolated coefficients for the k-points
        if eval_linear:
            grid, coeffs = self.interpolators[spin]

            # only allows interpolating floats, so have to separate real and imag parts
            p = np.empty((len(all_v), coeffs.shape[-1]), dtype=np.complex)
            p.real = eval_linear(grid, coeffs.real, all_v, xto.LINEAR)
            p.imag = eval_linear(grid, coeffs.imag, all_v, xto.LINEAR)
        else:
            p = self.interpolators[spin](all_v)

        p /= np.linalg.norm(p, axis=-1)[:, None]

        overlap = np.abs(np.dot(np.conj(p[0]), p[1:].T)) ** 2

        if single_overlap:
            return overlap[0]
        else:
            return overlap


class ProjectionOverlapCalculator(object):
    def __init__(
        self,
        structure,
        kpoints,
        projections,
        band_centers,
        symprec=defaults["symprec"],
        kpoint_symmetry_mapping=None,
    ):
        logger.info("Initializing orbital overlap calculator")

        # k-points have to be on a regular grid, even if only the irreducible part of
        # the grid is used. If the irreducible part is given, we have to expand it
        # to the full BZ. Also need to expand the projections to the full BZ using
        # the rotation mapping
        if kpoint_symmetry_mapping:
            full_kpoints, ir_to_full_idx, rot_mapping = kpoint_symmetry_mapping
        else:
            full_kpoints, ir_to_full_idx, rot_mapping = expand_kpoints(
                structure, kpoints, symprec=symprec
            )

        mesh_dim = get_mesh_dim_from_kpoints(full_kpoints)

        round_dp = int(np.log10(1 / 1e-6))
        full_kpoints = np.round(full_kpoints, round_dp)

        # get the indices to sort the k-points on the Z, then Y, then X columns
        sort_idx = np.lexsort(
            (full_kpoints[:, 2], full_kpoints[:, 1], full_kpoints[:, 0])
        )

        # put the kpoints into a 3D grid so that they can be indexed as
        # kpoints[ikx][iky][ikz] = [kx, ky, kz]
        grid_kpoints = full_kpoints[sort_idx].reshape(mesh_dim + (3,))

        x = grid_kpoints[:, 0, 0, 0]
        y = grid_kpoints[0, :, 0, 1]
        z = grid_kpoints[0, 0, :, 2]

        self.nbands = {s: p.shape[0] for s, p in projections.items()}

        # TODO: Expand the k-point mesh to account for periodic boundary conditions
        self.interpolators = {}
        for spin, spin_projections in projections.items():
            nbands = spin_projections.shape[0]
            nkpoints = len(ir_to_full_idx)
            nprojections = np.product(spin_projections.shape[2:])

            expand_projections = spin_projections[:, ir_to_full_idx]
            flat_projections = expand_projections.reshape(
                (nbands, nkpoints, -1), order="F"
            )

            # aim is to get the wavefunction coefficients
            norm_projection = (
                flat_projections
                / np.sqrt((flat_projections ** 2).sum(axis=2))[..., None]
            )
            norm_projection[np.isnan(norm_projection)] = 0
            coefficients = norm_projection

            # sort the coefficients then reshape them into the grid. The coefficients
            # can now be indexed as coefficients[iband][ikx][iky][ikz]
            sorted_coefficients = coefficients[:, sort_idx]
            grid_shape = (nbands,) + mesh_dim + (nprojections,)
            grid_coefficients = sorted_coefficients.reshape(grid_shape)

            if nbands == 1:
                # this can cause a bug in RegularGridInterpolator. Have to fake
                # having at least two bands
                nbands = 2
                grid_coefficients = np.tile(grid_coefficients, (2, 1, 1, 1, 1))

            interp_range = (np.arange(nbands), x, y, z)

            self.interpolators[spin] = RegularGridInterpolator(
                interp_range, grid_coefficients, bounds_error=False, fill_value=None
            )

        self.rotation_masks = get_rotation_masks(projections)
        self.band_centers = band_centers

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

        full_kpoints, ir_to_full_idx, rot_mapping = expand_kpoints(
            structure, kpoints, symprec=symprec
        )

        ibands = get_ibands(energy_cutoff, band_structure)
        vb_idx = get_vb_idx(energy_cutoff, band_structure)

        # energies = {
        #     s: e[ibands[s], ir_to_full_idx] for s, e in band_structure.bands.items()
        # }
        energies = {s: e[ibands[s]] for s, e in band_structure.bands.items()}
        energies = {s: e[:, ir_to_full_idx] for s, e in energies.items()}
        projections = {s: p[ibands[s]] for s, p in band_structure.projections.items()}

        band_centers = get_band_centers(full_kpoints, energies, vb_idx, efermi)

        return cls(
            structure,
            kpoints,
            projections,
            band_centers,
            kpoint_symmetry_mapping=(full_kpoints, ir_to_full_idx, rot_mapping),
        )

    def get_coefficients(self, spin, bands, kpoints):
        v = np.concatenate([np.asarray(bands)[:, None], kpoints], axis=1)
        return self.interpolators[spin](v)

    def get_overlap(self, spin, band_a, kpoint_a, band_b, kpoint_b):
        # k-points should be in fractional
        kpoint_a = np.asarray(kpoint_a)
        kpoint_b = np.asarray(kpoint_b)

        v1 = np.array([[band_a] + kpoint_a.tolist()])

        single_overlap = False
        if isinstance(band_b, numeric_types):
            # only one band index given

            if len(kpoint_b.shape) > 1:
                # multiple k-point indices given
                band_b = np.array([band_b] * len(kpoint_b))

            else:
                band_b = np.array([band_b])
                kpoint_b = [kpoint_b]
                single_overlap = True

        else:
            band_b = np.asarray(band_b)

        centers = self.band_centers[spin][band_a]
        center = centers[np.argmin(np.linalg.norm(pbc_diff(centers, kpoint_a), axis=1))]

        shift_a = pbc_diff(kpoint_a, center)
        shift_b = pbc_diff(kpoint_b, center)

        angles = cosine(shift_a, shift_b)
        angle_weights = np.abs(pbc_diff(kpoint_a, kpoint_b))
        angle_weights /= np.max(angle_weights, axis=1)[:, None]
        angle_weights[np.isnan(angle_weights)] = 0

        # v2 now has shape of (nkpoints_b, 4)
        v2 = np.concatenate([band_b[:, None], kpoint_b], axis=1)

        # get a big array of all the k-points to interpolate
        all_v = np.vstack([v1, v2])

        # get the interpolate projections for the k-points; p1 is the projections for
        # kpoint_a, p2 is a list of projections for the kpoint_b
        p1, *p2 = self.interpolators[spin](all_v)

        # weight the angle masks by the contribution of the transition in that direction
        weighted_mask = self.rotation_masks[None, ...] * angle_weights[..., None]

        # use the mask to get the angle scaling factor which gives how much the angle
        # is applied to each projection. I.e., if the scaling factor is 1 for a
        # particular projection, then the projection is scaled by 1 * the angle. If the
        # scaling factor is 0.5, the projection is scaled by 0.5 * the angle.
        # this allows us to only weight specific orbitals
        scaling_factor = np.max(weighted_mask, axis=1)

        p_product = p1 * p2

        overlap = np.sum(
            p_product * (1 - scaling_factor)
            + p_product * scaling_factor * angles[:, None],
            axis=1,
        )
        # overlap = np.ones_like(overlap)

        if single_overlap:
            return overlap[0] ** 2
        else:
            return overlap ** 2


def get_rotation_masks(projections):
    nprojections, natoms = projections[Spin.up].shape[2:]
    mask = np.zeros((3, nprojections))

    if nprojections >= 4:
        mask[:, 1:4] = _p_mask

    if nprojections >= 9:
        mask[:, 4:9] = _d_mask

    if nprojections > 9:
        # rotate f-orbitals?
        mask[:, 9:] = 1

    return np.tile(mask, natoms)


def cosine(v1, v2):
    # v2 can be list of vectors or single vector

    return_single = False
    if len(v2.shape) == 1:
        v2 = v2[None, :]
        return_single = True

    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2, axis=1)

    v_dot = np.dot(v1, v2.T)
    v_angle = v_dot / (v1_norm * v2_norm)
    v_angle[np.isnan(v_angle)] = 1

    if return_single:
        return v_angle[0]
    else:
        return v_angle


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
