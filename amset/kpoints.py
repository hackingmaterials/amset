from typing import Union, List, Tuple

import numpy as np
from spglib import spglib

from amset.tetrahedron import get_tetrahedra
from pymatgen import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from amset import amset_defaults

_SYMPREC = amset_defaults["performance"]["symprec"]
_KTOL = 1e-5


def kpoints_to_first_bz(kpoints: np.ndarray, tol=_KTOL) -> np.ndarray:
    """Translate fractional k-points to the first Brillouin zone.

    I.e. all k-points will have fractional coordinates:
        -0.5 <= fractional coordinates < 0.5

    Args:
        kpoints: The k-points in fractional coordinates.

    Returns:
        The translated k-points.
    """
    kp = kpoints - np.round(kpoints)

    # account for small rounding errors for 0.5
    round_dp = int(np.log10(1 / tol))
    krounded = np.round(kp, round_dp)

    kp[krounded == 0.5] = -0.5
    return kp


def get_dense_kpoint_mesh_spglib(mesh, spg_order=False, shift=0):
    """This is a reimplementation of the spglib c function get_all_grid_addresses

    Given a k-point mesh, gives the full k-point mesh that covers
    the first Brillouin zone. Uses the same convention as spglib,
    in that k-points on the edge of the Brillouin zone only
    appear as positive numbers. I.e. coordinates will always be
    +0.5 rather than -0.5. Similarly, the same ordering scheme is
    used.

    The only difference between this function and the function implemented
    in spglib is that here we return the final fraction coordinates
    whereas spglib returns the grid_addresses (integer numbers).
    """
    mesh = np.asarray(mesh)

    addresses = np.stack(np.mgrid[0:mesh[0], 0:mesh[1], 0:mesh[2]],
                         axis=-1).reshape(np.product(mesh), -1)

    if spg_order:
        # order the kpoints using the same ordering scheme as spglib
        idx = addresses[:, 2] * (mesh[0] * mesh[1]) + addresses[:, 1] * mesh[
            0] + addresses[:, 0]
        addresses = addresses[idx]

    addresses -= mesh * (addresses > mesh / 2)

    full_kpoints = (addresses + shift) / mesh
    sort_idx = np.lexsort((full_kpoints[:, 2], full_kpoints[:, 2] < 0,
                           full_kpoints[:, 1], full_kpoints[:, 1] < 0,
                           full_kpoints[:, 0], full_kpoints[:, 0] < 0))
    full_kpoints = full_kpoints[sort_idx]
    return full_kpoints


def get_symmetry_equivalent_kpoints(structure, kpoints, symprec=_SYMPREC, tol=_KTOL,
                                    return_inverse=False,
                                    time_reversal_symmetry=True):
    round_dp = int(np.log10(1 / tol))

    def shift_and_round(k):
        k = kpoints_to_first_bz(k)
        k = np.round(k, round_dp)
        return list(map(tuple, k))

    kpoints = np.asarray(kpoints)
    round_kpoints = shift_and_round(kpoints)

    rotation_matrices = get_reciprocal_point_group_operations(
        structure,
        symprec=symprec,
        time_reversal_symmetry=time_reversal_symmetry
    )

    equiv_points_mapping = {}
    rotation_matrix_mapping = {}
    mapping = []
    rot_mapping = []

    for i, point in enumerate(round_kpoints):

        if point in equiv_points_mapping:
            map_idx = equiv_points_mapping[point]
            mapping.append(map_idx)
            rot_mapping.append(rotation_matrix_mapping[map_idx][point])
        else:
            new_points = shift_and_round(np.dot(rotation_matrices, kpoints[i]))

            equiv_points_mapping.update(zip(new_points, [i] * len(new_points)))
            rotation_matrix_mapping[i] = dict(zip(new_points, rotation_matrices))

            mapping.append(i)
            rot_mapping.append(np.eye(3))

    ir_kpoints_idx, ir_to_full_idx, weights = np.unique(
        mapping, return_inverse=True, return_counts=True)

    ir_kpoints = kpoints[ir_kpoints_idx]

    if return_inverse:
        return (ir_kpoints, weights, ir_kpoints_idx, ir_to_full_idx,
                np.array(mapping), np.array(rot_mapping))
    else:
        return ir_kpoints, weights


def similarity_transformation(rot, mat):
    """ R x M x R^-1 """
    return np.dot(rot, np.dot(mat, np.linalg.inv(rot)))


def symmetrize_kpoints(structure, kpoints, symprec=_SYMPREC, tol=_KTOL,
                       time_reversal_symmetry=True):
    round_dp = int(np.log10(1 / tol))

    def shift_and_round(k):
        k = kpoints_to_first_bz(k)
        k = np.round(k, round_dp)
        return list(map(tuple, k))

    rotation_matrices = get_reciprocal_point_group_operations(
        structure,
        symprec=symprec,
        time_reversal_symmetry=time_reversal_symmetry
    )

    kpoints = kpoints_to_first_bz(kpoints)

    # symmetrized_kpoints = np.concatenate(
    #     [np.dot(rotation_matrices, k) for k in kpoints])
    symmetrized_kpoints = np.concatenate(
        [np.dot(kpoints, rot.T) for rot in rotation_matrices])
    symmetrized_kpoints = kpoints_to_first_bz(symmetrized_kpoints)
    _, unique_idxs = np.unique(shift_and_round(symmetrized_kpoints),
                               axis=0, return_index=True)

    return symmetrized_kpoints[unique_idxs]


def get_kpoints(kpoint_mesh: Union[float, List[int]],
                structure: Structure,
                symprec: float = _SYMPREC,
                return_full_kpoints: bool = False,
                boltztrap_ordering: bool = True
                ) -> Tuple[np.ndarray, ...]:
    """Gets the symmetry inequivalent k-points from a k-point mesh.

    Follows the same process as SpacegroupAnalyzer.get_ir_reciprocal_mesh
    but is faster and allows returning of the full k-point mesh and mapping.

    Args:
        kpoint_mesh: The k-point mesh as a 1x3 array. E.g.,``[6, 6, 6]``.
            Alternatively, if a single value is provided this will be
            treated as a k-point spacing cut-off and the k-points will be generated
            automatically.  Cutoff is length in Angstroms and corresponds to
            non-overlapping radius in a hypothetical supercell (Moreno-Soler length
            cutoff).
        structure: A structure.
        symprec: Symmetry tolerance used when determining the symmetry
            inequivalent k-points on which to interpolate.
        return_full_kpoints: Whether to return the full list of k-points
            covering the entire Brillouin zone and the indices of
            inequivalent k-points.
        boltztrap_ordering: Whether to return the k-points in the same order as
            given by the BoltzTraP2.fite.getBTPBands.

    Returns:
        The irreducible k-points and their weights as tuple, formatted as::

            (ir_kpoints, weights)

        If return_full_kpoints, the data will be returned as::

            (ir_kpoints, weights, full_kpoints, ir_kpoints_idx, ir_to_full_idx)

        Where ``ir_kpoints_idx`` is the index of the unique irreducible k-points
        in ``full_kpoints``. ``ir_to_full_idx`` is a list of indices that can be
        used to construct the full Brillouin zone from the ir_mesh. Note the
        ir -> full conversion will only work with calculated scalar properties
        such as energy (not vector properties such as velocity).
    """
    if isinstance(kpoint_mesh, (int, float)):
        kpoint_mesh = get_kpoint_mesh(structure, kpoint_mesh)

    atoms = AseAtomsAdaptor().get_atoms(structure)

    if not symprec:
        symprec = 1e-8

    mapping, grid = spglib.get_ir_reciprocal_mesh(
        kpoint_mesh, atoms, symprec=symprec)
    full_kpoints = grid / kpoint_mesh
    full_kpoints = kpoints_to_first_bz(full_kpoints)

    if boltztrap_ordering:
        sort_idx = np.lexsort((full_kpoints[:, 2], full_kpoints[:, 2] < 0,
                               full_kpoints[:, 1], full_kpoints[:, 1] < 0,
                               full_kpoints[:, 0], full_kpoints[:, 0] < 0))
        full_kpoints = full_kpoints[sort_idx]
        mapping = mapping[sort_idx]

        mapping_dict = {}
        new_mapping = []
        for i, n in enumerate(mapping):
            if n in mapping_dict:
                new_mapping.append(mapping_dict[n])
            else:
                mapping_dict[n] = i
                new_mapping.append(i)
        mapping = new_mapping

    ir_kpoints_idx, ir_to_full_idx, weights = np.unique(
        mapping, return_inverse=True, return_counts=True)
    ir_kpoints = full_kpoints[ir_kpoints_idx]

    if return_full_kpoints:
        return ir_kpoints, weights, full_kpoints, ir_kpoints_idx, ir_to_full_idx
    else:
        return ir_kpoints, weights


def sort_boltztrap_to_spglib(kpoints):
    sort_idx = np.lexsort((kpoints[:, 2], kpoints[:, 2] < 0,
                           kpoints[:, 1], kpoints[:, 1] < 0,
                           kpoints[:, 0], kpoints[:, 0] < 0))
    return sort_idx[sort_idx]


def get_kpoints_tetrahedral(
        kpoint_mesh: Union[float, List[int]],
        structure: Structure,
        symprec: float = _SYMPREC,
        return_full_kpoints: bool = False,
        ) -> Tuple[np.ndarray, ...]:
    """Gets the symmetry inequivalent k-points from a k-point mesh.

    Follows the same process as SpacegroupAnalyzer.get_ir_reciprocal_mesh
    but is faster and allows returning of the full k-point mesh and mapping.

    Args:
        kpoint_mesh: The k-point mesh as a 1x3 array. E.g.,``[6, 6, 6]``.
            Alternatively, if a single value is provided this will be
            treated as a k-point spacing cut-off and the k-points will be generated
            automatically.  Cutoff is length in Angstroms and corresponds to
            non-overlapping radius in a hypothetical supercell (Moreno-Soler length
            cutoff).
        structure: A structure.
        symprec: Symmetry tolerance used when determining the symmetry
            inequivalent k-points on which to interpolate.
        return_full_kpoints: Whether to return the full list of k-points
            covering the entire Brillouin zone and the indices of
            inequivalent k-points.

    Returns:
        The irreducible k-points and their weights as tuple, formatted as::

            (ir_kpoints, weights)

        If return_full_kpoints, the data will be returned as::

            (ir_kpoints, weights, full_kpoints, ir_kpoints_idx, ir_to_full_idx)

        Where ``ir_kpoints_idx`` is the index of the unique irreducible k-points
        in ``full_kpoints``. ``ir_to_full_idx`` is a list of indices that can be
        used to construct the full Brillouin zone from the ir_mesh. Note the
        ir -> full conversion will only work with calculated scalar properties
        such as energy (not vector properties such as velocity).
    """
    if isinstance(kpoint_mesh, (int, float)):
        kpoint_mesh = get_kpoint_mesh(structure, kpoint_mesh)

    atoms = AseAtomsAdaptor().get_atoms(structure)

    if not symprec:
        symprec = 1e-8

    grid_mapping, grid_address = spglib.get_ir_reciprocal_mesh(
        kpoint_mesh, atoms, symprec=symprec)
    full_kpoints = grid_address / kpoint_mesh

    tetra, ir_tetrahedra_idx, ir_tetrahedra_to_full_idx, tet_weights = get_tetrahedra(
        structure.lattice.reciprocal_lattice.matrix,
        grid_address,
        kpoint_mesh,
        grid_mapping
    )

    ir_kpoints_idx, ir_to_full_idx, weights = np.unique(
        grid_mapping, return_inverse=True, return_counts=True)
    ir_kpoints = full_kpoints[ir_kpoints_idx]

    return (ir_kpoints, weights, full_kpoints, ir_kpoints_idx, ir_to_full_idx,
            tetra, ir_tetrahedra_idx, ir_tetrahedra_to_full_idx, tet_weights)


def get_kpoint_mesh(structure: Structure, cutoff_length: float, force_odd: bool = True):
    """Calculate reciprocal-space sampling with real-space cut-off.

    """
    reciprocal_lattice = structure.lattice.reciprocal_lattice_crystallographic

    # Get reciprocal cell vector magnitudes
    abc_recip = np.array(reciprocal_lattice.abc)

    mesh = np.ceil(abc_recip * 2 * cutoff_length).astype(int)

    if force_odd:
        mesh += (mesh + 1) % 2

    return mesh


def get_reciprocal_point_group_operations(
        structure: Structure,
        symprec: float = _SYMPREC,
        time_reversal_symmetry: bool = True
):
    frac_real_to_frac_recip = np.dot(
        np.linalg.inv(structure.lattice.reciprocal_lattice.matrix.T),
        structure.lattice.matrix.T
    )
    frac_recip_to_frac_real = np.linalg.inv(frac_real_to_frac_recip)

    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    isomorphic_ops = [op.rotation_matrix for op in sga.get_symmetry_operations()]

    parity = -np.eye(3)

    if time_reversal_symmetry:
        reciprocal_ops = [-np.eye(3)]
    else:
        reciprocal_ops = [np.eye(3)]

    for op in isomorphic_ops:

        # convert to reciprocal primitive basis
        op = np.around(
            np.dot(frac_real_to_frac_recip, np.dot(op, frac_recip_to_frac_real)),
            decimals=2
        )

        reciprocal_ops.append(op)
        if time_reversal_symmetry:
            reciprocal_ops.append(np.dot(parity, op))

    return np.unique(reciprocal_ops, axis=0)
