import logging

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from amset.constants import numeric_types
from amset.kpoints import (
    expand_kpoints,
    get_mesh_dim_from_kpoints,
    similarity_transformation,
)
from pymatgen import Spin
from pymatgen.util.coord import pbc_diff

_p_orbital_order = [2, 0, 1]  # VASP sorts p orbitals as py, pz, px
_d_orbital_order = [[4, 0, 3], [0, 4, 1], [3, 1, 2]]
_select_d_order = ([2, 0, 1, 0, 1], [2, 2, 2, 1, 1])  # selects dz2 dxz dyz dxy dx2

_p_mask = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
_d_mask = np.array([[0, 1, 0, 1, 1], [0, 0, 1, 1, 1], [1, 1, 1, 0, 0]])

logger = logging.getLogger(__name__)


class OverlapCalculator(object):
    def __init__(self, structure, kpoints, projections):

        logger.info("Initializing orbital overlap calculator")

        # k-points have to be on a regular grid, even if only the irreducible part of
        # the grid is used. If the irreducible part is given, we have to expand it
        # to the full BZ. Also need to expand the projections to the full BZ using
        # the rotation mapping
        full_kpoints, ir_to_full_idx, rot_mapping = expand_kpoints(structure, kpoints)
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

        # TODO: Expand the k-point mesh to account for periodic boundary conditions
        self.interpolators = {}
        for spin, spin_projections in projections.items():
            nbands = spin_projections.shape[0]
            nkpoints = len(ir_to_full_idx)
            nprojections = np.product(spin_projections.shape[2:])

            expand_projections = spin_projections[:, ir_to_full_idx]
            rot_projections = rotate_projections(
                expand_projections, rot_mapping, structure
            )
            # flat_projections = expand_projections.reshape((nbands, nkpoints, -1), order='F')
            flat_projections = rot_projections.reshape(
                (nbands, nkpoints, -1), order="F"
            )

            # aim is to get the wavefunction coefficients
            # norm_projection = flat_projections / flat_projections.sum(axis=2)[..., None]
            norm_projection = (
                flat_projections
                / np.sqrt((flat_projections ** 2).sum(axis=2))[..., None]
            )
            norm_projection[np.isnan(norm_projection)] = 0
            # coefficients = np.sqrt(norm_projection)
            coefficients = norm_projection

            # sort the coefficients then reshape them into the grid. The coefficients
            # can now be indexed as coefficients[iband][ikx][iky][ikz]
            sorted_coefficients = coefficients[:, sort_idx]
            grid_shape = (nbands,) + mesh_dim + (nprojections,)
            grid_coefficients = sorted_coefficients.reshape(grid_shape)

            interp_range = (np.arange(nbands), x, y, z)

            self.interpolators[spin] = RegularGridInterpolator(
                interp_range, grid_coefficients, bounds_error=False, fill_value=None
            )

        self.rotation_masks = get_rotation_masks(projections)

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

        # centre = [0.5, 0.5, 0.5]
        centre = [0.0, 0, 0]

        shift_a = pbc_diff(kpoint_a, centre)
        shift_b = pbc_diff(kpoint_b, centre)

        angles = cosine(shift_a, shift_b)
        # angles = cosine(shift_a, pbc_diff(shift_b, shift_a) + shift_a)
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

        # print("\nka", kpoint_a)
        # print("\nkb", kpoint_b[0])
        # print("\npa", p1)
        # print("\npb", p2[0])
        # print("\npp", p1 * p2[0])
        # print(p_product[:2])

        # overlap = np.sum(p_product, axis=1)
        overlap = np.sum(
            p_product * (1 - scaling_factor)
            + p_product * scaling_factor * angles[:, None],
            axis=1,
        )
        # print("\nscale", scaling_factor[:2])

        if single_overlap:
            return overlap[0] ** 2
        else:
            return overlap ** 2


def rotate_projections(projections, rotation_mapping, structure):
    rlat = structure.lattice.reciprocal_lattice.matrix

    similarity_matrix = [similarity_transformation(rlat, r) for r in rotation_mapping]
    inv_similarity_matrix = [np.linalg.inv(s) for s in similarity_matrix]

    # projections given as (nbands, nkpoints, nprojections, natoms)
    nprojections = projections.shape[2]
    rotated_projections = projections.copy()

    if nprojections >= 4:
        # includes p orbitals
        rotated_projections[:, :, 1:4, :] = rotate_p_orbitals(
            projections[:, :, 1:4, :], similarity_matrix
        )

    if nprojections >= 9:
        # includes 9 orbitals
        rotated_projections[:, :, 4:9, :] = rotate_d_orbitals(
            projections[:, :, 4:9, :], similarity_matrix, inv_similarity_matrix
        )

    # TODO: Rotate f-orbitals?
    return rotated_projections


def rotate_p_orbitals(p_orbital_projections, similarity_matrix):
    # p_orbital_projections has the shape (nbands, nkpoints, 3, natoms)
    # similarity_matrix has the shape (nkpoints, 3, 3)

    # this function both reorders and rotates the p orbitals
    nbands, nkpoints, _, natoms = p_orbital_projections.shape

    rotated_projections = np.zeros_like(p_orbital_projections)
    for b_idx, k_idx, a_idx in np.ndindex((nbands, nkpoints, natoms)):
        rotated_projections[b_idx, k_idx, :, a_idx] = np.dot(
            similarity_matrix[k_idx],
            p_orbital_projections[b_idx, k_idx, _p_orbital_order, a_idx],
        )

    return np.abs(rotated_projections)


def rotate_d_orbitals(d_orbital_projections, similarity_matrix, inv_similarity_matrix):
    # d_orbital_projections has the shape (nbands, nkpoints, 3, natoms)
    # similarity_matrix has the shape (nkpoints, 3, 3)

    # this function both reorders and rotates the p orbitals
    nbands, nkpoints, _, natoms = d_orbital_projections.shape

    rotated_projections = np.zeros_like(d_orbital_projections)
    for b_idx, k_idx, a_idx in np.ndindex((nbands, nkpoints, natoms)):
        r1 = np.dot(
            d_orbital_projections[b_idx, k_idx, _d_orbital_order, a_idx],
            similarity_matrix[k_idx],
        )

        rotated_projections[b_idx, k_idx, :, a_idx] = np.dot(
            inv_similarity_matrix[k_idx], r1
        )[_select_d_order]

    return np.abs(rotated_projections)


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
