from typing import List, Tuple, Union

import numpy as np
from spglib import spglib

from amset.constants import defaults
from pymatgen import Structure
from pymatgen.io.ase import AseAtomsAdaptor

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

_SYMPREC = defaults["symprec"]
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


def get_symmetry_equivalent_kpoints(
    structure,
    kpoints,
    symprec=_SYMPREC,
    tol=_KTOL,
    return_inverse=False,
    time_reversal_symmetry=True,
):
    round_dp = int(np.log10(1 / tol))

    def shift_and_round(k):
        k = kpoints_to_first_bz(k)
        k = np.round(k, round_dp)
        return list(map(tuple, k))

    kpoints = np.asarray(kpoints)
    round_kpoints = shift_and_round(kpoints)

    rotation_matrices = get_reciprocal_point_group_operations(
        structure, symprec=symprec, time_reversal_symmetry=time_reversal_symmetry
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
        mapping, return_inverse=True, return_counts=True
    )

    ir_kpoints = kpoints[ir_kpoints_idx]

    if return_inverse:
        return (
            ir_kpoints,
            weights,
            ir_kpoints_idx,
            ir_to_full_idx,
            np.array(mapping),
            np.array(rot_mapping),
        )
    else:
        return ir_kpoints, weights


def similarity_transformation(rot, mat):
    """ R x M x R^-1 """
    return np.dot(rot, np.dot(mat, np.linalg.inv(rot)))


def sort_boltztrap_to_spglib(kpoints):
    sort_idx = np.lexsort(
        (
            kpoints[:, 2],
            kpoints[:, 2] < 0,
            kpoints[:, 1],
            kpoints[:, 1] < 0,
            kpoints[:, 0],
            kpoints[:, 0] < 0,
        )
    )
    boltztrap_kpoints = kpoints[sort_idx]

    sort_idx = np.lexsort(
        (
            boltztrap_kpoints[:, 0],
            boltztrap_kpoints[:, 0] < 0,
            boltztrap_kpoints[:, 1],
            boltztrap_kpoints[:, 1] < 0,
            boltztrap_kpoints[:, 2],
            boltztrap_kpoints[:, 2] < 0,
        )
    )
    return sort_idx


def get_kpoints_tetrahedral(
    kpoint_mesh: Union[float, List[int]],
    structure: Structure,
    symprec: float = _SYMPREC,
    time_reversal_symmetry: bool = True,
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
        time_reversal_symmetry: Whether the system has time reversal symmetry.

    Returns:
        The irreducible k-points and their weights as tuple, formatted as::

            (ir_kpoints, weights)

        If return_full_kpoints, the data will be returned as::

            (ir_kpoints, weights, kpoints, ir_kpoints_idx, ir_to_full_idx)

        Where ``ir_kpoints_idx`` is the index of the unique irreducible k-points
        in ``kpoints``. ``ir_to_full_idx`` is a list of indices that can be
        used to construct the full Brillouin zone from the ir_mesh. Note the
        ir -> full conversion will only work with calculated scalar properties
        such as energy (not vector properties such as velocity).
    """
    from amset.electronic_structure.tetrahedron import get_tetrahedra

    if isinstance(kpoint_mesh, (int, float)):
        kpoint_mesh = get_kpoint_mesh(structure, kpoint_mesh)

    atoms = AseAtomsAdaptor().get_atoms(structure)

    if not symprec:
        symprec = 1e-8

    grid_mapping, grid_address = spglib.get_ir_reciprocal_mesh(
        kpoint_mesh, atoms, symprec=symprec, is_time_reversal=time_reversal_symmetry
    )
    full_kpoints = grid_address / kpoint_mesh

    tetra, ir_tetrahedra_idx, ir_tetrahedra_to_full_idx, tet_weights = get_tetrahedra(
        structure.lattice.reciprocal_lattice.matrix,
        grid_address,
        kpoint_mesh,
        grid_mapping,
    )

    ir_kpoints_idx, ir_to_full_idx, weights = np.unique(
        grid_mapping, return_inverse=True, return_counts=True
    )
    ir_kpoints = full_kpoints[ir_kpoints_idx]

    return (
        ir_kpoints,
        weights,
        full_kpoints,
        ir_kpoints_idx,
        ir_to_full_idx,
        tetra,
        ir_tetrahedra_idx,
        ir_tetrahedra_to_full_idx,
        tet_weights,
    )


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


def expand_kpoints(
    structure, kpoints, symprec=_SYMPREC, tol=_KTOL, time_reversal_symmetry=True
):
    round_dp = int(np.log10(1 / tol))

    def shift_and_round(k):
        k = kpoints_to_first_bz(k)
        k = np.round(k, round_dp)
        return list(map(tuple, k))

    rotation_matrices = get_reciprocal_point_group_operations(
        structure, symprec=symprec, time_reversal_symmetry=time_reversal_symmetry
    )

    kpoints = kpoints_to_first_bz(kpoints)
    full_kpoints = []
    reduced_to_full_idx = []
    rot_mapping = []
    equiv_points_mapping = {}

    for i, kpoint in enumerate(kpoints):

        symmetrized_kpoints = kpoints_to_first_bz(np.dot(rotation_matrices, kpoint))
        symmetrized_round_kpoints = shift_and_round(symmetrized_kpoints)
        _, unique_idxs = np.unique(symmetrized_round_kpoints, axis=0, return_index=True)

        for ui in unique_idxs:
            spoint = symmetrized_kpoints[ui]
            spoint_round = symmetrized_round_kpoints[ui]
            rotation_matrix = rotation_matrices[ui]

            if spoint_round not in equiv_points_mapping:
                full_kpoints.append(spoint)
                equiv_points_mapping[spoint_round] = i
                reduced_to_full_idx.append(i)
                rot_mapping.append(rotation_matrix)

    return np.array(full_kpoints), np.array(reduced_to_full_idx), np.array(rot_mapping)


def get_mesh_dim_from_kpoints(kpoints, tol=_KTOL):
    round_dp = int(np.log10(1 / tol))

    kpoints = kpoints_to_first_bz(kpoints)
    round_kpoints = np.round(kpoints, round_dp)

    nx = len(np.unique(round_kpoints[:, 0]))
    ny = len(np.unique(round_kpoints[:, 1]))
    nz = len(np.unique(round_kpoints[:, 2]))

    if nx * ny * nz != len(kpoints):
        raise ValueError("Something went wrong getting k-point mesh dimensions")

    return nx, ny, nz


def get_reciprocal_point_group_operations(
    structure: Structure, symprec: float = _SYMPREC, time_reversal_symmetry: bool = True
):
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    frac_real_to_frac_recip = np.dot(
        np.linalg.inv(structure.lattice.reciprocal_lattice.matrix.T),
        structure.lattice.matrix.T,
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
            decimals=2,
        )

        reciprocal_ops.append(op)
        if time_reversal_symmetry:
            reciprocal_ops.append(np.dot(parity, op))

    return np.unique(reciprocal_ops, axis=0)
