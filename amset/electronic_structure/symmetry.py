import logging

import numpy as np
from pymatgen import Structure, SymmOp
from pymatgen.electronic_structure.bandstructure import BandStructure

from amset.constants import defaults
from amset.electronic_structure.kpoints import (
    get_kpoints_from_bandstructure,
    get_mesh_from_kpoint_diff,
    kpoints_to_first_bz,
    ktol,
)
from amset.log import log_list

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

logger = logging.getLogger(__name__)


def get_symmetry_equivalent_kpoints(
    structure,
    kpoints,
    symprec=defaults["symprec"],
    tol=ktol,
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

    rotation_matrices, _, _ = get_reciprocal_point_group_operations(
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


def expand_kpoints(
    structure,
    kpoints,
    symprec=defaults["symprec"],
    return_mapping=False,
    time_reversal=True,
    verbose=True,
):
    if verbose:
        logger.info("Desymmetrizing k-point mesh")

    kpoints = np.array(kpoints).round(8)

    # due to limited input precision of the k-points, the mesh is returned as a float
    mesh = get_mesh_from_kpoint_diff(kpoints)
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
    if verbose:
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

    # zone boundary consistent with VASP not with spglib
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


def get_reciprocal_point_group_operations(
    structure: Structure,
    symprec: float = defaults["symprec"],
    time_reversal: bool = True,
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

    # put identity first and time-reversal last
    sort_idx = np.argsort(np.abs(rotations - np.eye(3)).sum(axis=(1, 2)) + is_tr * 10)

    return rotations[sort_idx], translations[sort_idx], is_tr[sort_idx]


def expand_bandstructure(
    bandstructure, symprec=defaults["symprec"], time_reversal=True
):
    kpoints = get_kpoints_from_bandstructure(bandstructure)
    full_kpoints, _, _, _, _, kp_mapping = expand_kpoints(
        bandstructure.structure,
        kpoints,
        symprec=symprec,
        time_reversal=time_reversal,
        return_mapping=True,
    )
    return BandStructure(
        full_kpoints,
        {s: b[:, kp_mapping] for s, b in bandstructure.bands.items()},
        bandstructure.structure.lattice.reciprocal_lattice,
        bandstructure.efermi,
        structure=bandstructure.structure,
    )


def rotate_bandstructure(bandstructure: BandStructure, frac_symop: SymmOp):
    """Won't rotate projections..."""
    kpoints = get_kpoints_from_bandstructure(bandstructure)
    recip_rot = frac_symop.rotation_matrix.T
    rot_kpoints = np.dot(recip_rot, kpoints.T).T

    # map to first BZ, use VASP zone boundary convention
    rot_kpoints = kpoints_to_first_bz(rot_kpoints, negative_zone_boundary=False)

    # rotate structure
    structure = bandstructure.structure.copy()
    structure.apply_operation(frac_symop, fractional=True)

    return BandStructure(
        rot_kpoints,
        bandstructure.bands,
        structure.lattice.reciprocal_lattice,
        bandstructure.efermi,
        structure=structure,
    )
