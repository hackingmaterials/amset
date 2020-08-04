import logging
from typing import List, Tuple, Union

import numpy as np
from pymatgen import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from spglib import spglib

from amset.constants import defaults
from amset.log import log_list

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

_SYMPREC = defaults["symprec"]
_KTOL = 1e-5

logger = logging.getLogger(__name__)


def kpoints_to_first_bz(kpoints: np.ndarray, tol=_KTOL) -> np.ndarray:
    """Translate fractional k-points to the first Brillouin zone.

    I.e. all k-points will have fractional coordinates:
        -0.5 < fractional coordinates <= 0.5

    Args:
        kpoints: The k-points in fractional coordinates.

    Returns:
        The translated k-points.
    """
    kp = kpoints - np.round(kpoints)

    # account for small rounding errors for 0.5
    round_dp = int(np.log10(1 / tol))
    krounded = np.round(kp, round_dp)

    kp[krounded == -0.5] = 0.5
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

    rotation_matrices, _ = get_reciprocal_point_group_operations(
        structure, symprec=symprec, time_reversal=time_reversal_symmetry
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


def get_mesh_from_kpoints(kpoints):
    kpoints = np.array(kpoints)
    nx = 1 / np.min(np.diff(np.unique(kpoints[:, 0])))
    ny = 1 / np.min(np.diff(np.unique(kpoints[:, 1])))
    nz = 1 / np.min(np.diff(np.unique(kpoints[:, 2])))

    # due to limited precission of the input k-points, the mesh is returned as a float
    return np.array([nx, ny, nz])


def expand_kpoints(
    structure, kpoints, symprec=_SYMPREC, return_mapping=False, time_reversal=True
):
    logger.info("Desymmetrizing k-point mesh")
    kpoints = np.array(kpoints).round(8)

    # due to limited input precision of the k-points, the mesh is returned as a float
    mesh = get_mesh_from_kpoints(kpoints)
    status_info = ["Found initial mesh: {:.3f} x {:.3f} x {:.3f}".format(*mesh)]

    # to avoid issues to limited input precision, recalculate the input k-points
    # so that the mesh is integer and the k-points are not truncated
    # to a small precision
    addresses = np.rint(kpoints * mesh)
    mesh = np.rint(mesh)
    kpoints = addresses / mesh
    status_info.append("Integer mesh: {} x {} x {}".format(*map(int, mesh)))

    rotations, translations, is_tr = get_reciprocal_point_group_operations(
        structure, symprec=symprec, time_reversal=time_reversal
    )
    n_ops = len(rotations)
    status_info.append("Using {} symmetry operations".format(n_ops))
    log_list(status_info)

    # rotate all-kpoints
    all_rotated_kpoints = []
    for r in rotations:
        all_rotated_kpoints.append(np.dot(r, kpoints.T).T)
    all_rotated_kpoints = np.concatenate(all_rotated_kpoints)

    # map to first BZ
    all_rotated_kpoints -= np.rint(all_rotated_kpoints)
    all_rotated_kpoints = all_rotated_kpoints.round(8)
    all_rotated_kpoints[all_rotated_kpoints == -0.5] = 0.5

    # Find unique points
    unique_rotated_kpoints, unique_idxs = np.unique(
        all_rotated_kpoints, return_index=True, axis=0
    )

    # find integer addresses
    unique_addresses = unique_rotated_kpoints * mesh
    unique_addresses -= np.rint(unique_addresses)
    in_uniform_mesh = (np.abs(unique_addresses) < 1e-5).all(axis=1)

    n_mapped = int(np.sum(in_uniform_mesh))
    n_expected = int(np.product(mesh))
    if n_mapped != n_expected:
        raise ValueError("Expected {} points but found {}".format(n_expected, n_mapped))

    full_kpoints = unique_rotated_kpoints[in_uniform_mesh]
    full_idxs = unique_idxs[in_uniform_mesh]

    if not return_mapping:
        return full_kpoints

    op_mapping = np.floor(full_idxs / len(kpoints)).astype(int)
    kp_mapping = (full_idxs % len(kpoints)).astype(int)

    return full_kpoints, rotations, translations, is_tr, op_mapping, kp_mapping


def get_mesh_dim_from_kpoints(kpoints, tol=_KTOL):
    round_dp = int(np.log10(1 / tol))

    kpoints = kpoints_to_first_bz(kpoints)
    round_kpoints = np.round(kpoints, round_dp)

    nx = len(np.unique(round_kpoints[:, 0]))
    ny = len(np.unique(round_kpoints[:, 1]))
    nz = len(np.unique(round_kpoints[:, 2]))

    return nx, ny, nz


def get_reciprocal_point_group_operations(
    structure: Structure, symprec: float = _SYMPREC, time_reversal: bool = True
):
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    rotations = sga.get_symmetry_dataset()["rotations"].transpose((0, 2, 1))
    translations = sga.get_symmetry_dataset()["translations"]
    is_tr = np.full(len(rotations), False, dtype=bool)

    if time_reversal:
        rotations = np.concatenate([rotations, -rotations])
        translations = np.concatenate([translations, -translations])
        is_tr = np.concatenate([is_tr, ~is_tr])

        rotations, unique_ops = np.unique(rotations, axis=0, return_index=True)
        translations = translations[unique_ops]
        is_tr = is_tr[unique_ops]

    # put identify first and time-reversal last
    sort_idx = np.argsort(np.abs(rotations - np.eye(3)).sum(axis=(1, 2)) + is_tr * 10)

    return rotations[sort_idx], translations[sort_idx], is_tr[sort_idx]


# def get_reciprocal_point_group_operations(
#     structure: Structure, symprec: float = _SYMPREC, time_reversal: bool = True
# ):
#     from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
#
#     frac_real_to_frac_recip = np.dot(
#         np.linalg.inv(structure.lattice.reciprocal_lattice.matrix.T),
#         structure.lattice.matrix.T,
#     )
#     frac_recip_to_frac_real = np.linalg.inv(frac_real_to_frac_recip)
#
#     sga = SpacegroupAnalyzer(structure, symprec=symprec)
#     symops = sga.get_symmetry_operations()
#     isomorphic_rots = [op.rotation_matrix for op in symops]
#     isomorphic_taus = [op.translation_vector for op in symops]
#
#     parity = -np.eye(3)
#
#     if time_reversal:
#         reciprocal_rots = [-np.eye(3)]
#         reciprocal_taus = [[0, 0, 0]]
#     else:
#         reciprocal_rots = [np.eye(3)]
#         reciprocal_taus = [[0, 0, 0]]
#
#     for rot, tau in zip(isomorphic_rots, isomorphic_taus):
#
#         # convert to reciprocal primitive basis
#         rot = np.around(
#             np.dot(frac_real_to_frac_recip, np.dot(rot, frac_recip_to_frac_real)),
#             decimals=2,
#         )
#         # tau = np.dot(tau, frac_recip_to_frac_real)  # TODO: double check
#         tau = np.dot(tau, structure.lattice.reciprocal_lattice.matrix)
#
#         reciprocal_rots.append(rot)
#         reciprocal_taus.append(tau)
#
#         if time_reversal:
#             reciprocal_rots.append(np.dot(parity, rot))
#             reciprocal_taus.append(-tau)
#
#     reciprocal_rots, unique_idxs = np.unique(reciprocal_rots, axis=0, return_index=True)
#     return reciprocal_rots, np.array(reciprocal_taus)[unique_idxs]
