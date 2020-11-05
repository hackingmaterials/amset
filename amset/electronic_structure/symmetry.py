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
    mesh, is_shifted = get_mesh_from_kpoint_diff(kpoints)
    status_info = ["Found initial mesh: {:.3f} x {:.3f} x {:.3f}".format(*mesh)]

    if is_shifted:
        shift = np.array([1, 1, 1])
    else:
        shift = np.array([0, 0, 0])

    # to avoid issues to limited input precision, recalculate the input k-points
    # so that the mesh is integer and the k-points are not truncated
    # to a small precision
    addresses = np.rint((kpoints + shift / (mesh * 2)) * mesh)
    mesh = np.rint(mesh)
    kpoints = addresses / mesh - shift / (mesh * 2)

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
    unique_addresses = (unique_rotated_kpoints + shift / (mesh * 2)) * mesh
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


def rotation_matrix_to_cartesian(rotation_matrix, lattice):
    """Transform rotation matrix from fractional basis to cartesian basis."""
    sb = np.matmul(lattice.reciprocal_lattice_crystallographic.matrix, rotation_matrix)
    return np.matmul(lattice.matrix, sb.T)


def rotation_matrix_to_su2(rotation_matrix, eps=1e-8):
    """Take a rotation matrix and turn it into a matrix that represents the
    same rotation in spin space [SU(2) matrix].

    Adapted from the `find_u` function in Quantum Espresso.

    Args:
        rotation_matrix: A 3x3 rotation matrix.
        eps: Numerical tolerance.

    Returns:
        The rotation matrix in SU(2) form.
    """
    det = np.linalg.det(rotation_matrix)
    rotation_matrix = np.array(rotation_matrix)

    # inversion has no effect in spin space, so improper rotations are multiplied by
    # inversion
    if np.abs(det + 1) < eps:
        saux = -rotation_matrix
    else:
        saux = rotation_matrix

    # Check for identity or inversion
    if np.allclose(saux, np.eye(3), atol=eps):
        return np.array([[1, 0], [0, 1]], dtype=np.complex)

    # Find the rotation axis and the rotation angle
    ax = get_rotation_axis(saux)
    angle = get_rotation_angle(saux)
    angle *= 0.5 * np.pi / 180
    cosa = np.cos(angle)
    sina = np.sin(angle)

    # Set the spin space rotation matrix elements
    u11 = complex(cosa, -ax[2] * sina)
    u12 = complex(-ax[1] * sina, -ax[0] * sina)
    u = np.array([[u11, u12], [u12.conjugate(), u11.conjugate()]], dtype=np.complex)

    # For each 3x3 rotation one can associate two 2x2 rotation matrices in spin
    # space. This function returns the U matrix with positive cosa term
    if cosa < -eps:
        u = -u
    return u


def get_rotation_axis(rotation_matrix, eps=1e-7):
    """
    Find the rotation axis from a rotation matrix.

    The orientation of the axis is with the tip in the hemisphere z >= 0. In the xy
    plane the axis is in the x > 0 region and the positive y axis is taken for z = 0
    and x = 0.

    Adapted from the `versor` subroutine in QE.

    Args:
        rotation_matrix: A 3x3 rotation matrix.
        eps: Numerical tolerance.

    Returns:
        A 3x1 vector of the rotation axis.
    """
    ts = get_symmetry_type(rotation_matrix)
    rotation_matrix = np.array(rotation_matrix)

    if ts not in (3, 4, 6):
        raise ValueError("Transformation matrix is not a proper or improper rotation.")

    a1 = np.zeros(3)
    if ts == 4:
        # Proper rotation of 180 degrees

        # First, the case where the axis is parallel to a coordinate axis
        ax = np.zeros(3)
        for i in range(3):
            if abs(rotation_matrix[i, i] - 1) < eps:
                ax[i] = 1
        norm = np.linalg.norm(ax)

        if abs(norm) > eps:
            return ax

        # Then the general case
        for i in range(3):
            a1[i] = np.sqrt(np.abs(rotation_matrix[i, i] + 1) / 2)

        for i in range(3):
            for j in range(i + 1, 3):
                if np.abs(a1[i] * a1[j]) > eps:
                    a1[i] = 0.5 * rotation_matrix[i, j] / a1[j]
    else:
        # It is not a 180 rotation: compute the rotation axis
        a1[0] = -rotation_matrix[1, 2] + rotation_matrix[2, 1]
        a1[1] = -rotation_matrix[2, 0] + rotation_matrix[0, 2]
        a1[2] = -rotation_matrix[0, 1] + rotation_matrix[1, 0]

    # The direction of the axis is arbitrarily chosen, with positive z. In the
    # xy plane with positive x, and along y with positive y.
    if a1[2] < -eps:
        a1 = -a1
    elif abs(a1[2]) < eps and a1[0] < -eps:
        a1 = -a1
    elif abs(a1[2]) < eps and abs(a1[0]) < eps and a1[1] < -eps:
        a1 = -a1

    norm = np.linalg.norm(a1)
    if norm < eps:
        raise ValueError("Problem with rotation matrix")
    return a1 / norm


def get_rotation_angle(rotation_matrix, eps=1e-7):
    """Get the rotation angle from a rotation matrix

    Args:
        rotation_matrix: A 3x3 rotation matrix.
        eps: Numerical tolerance.

    Returns:
        The rotation angle.
    """
    if get_symmetry_type(rotation_matrix) == 4:
        # proper 180 rotation
        return 180

    rotation_matrix = np.array(rotation_matrix)

    # compute the axis
    a1 = np.zeros(3)
    a1[0] = -rotation_matrix[1, 2] + rotation_matrix[2, 1]
    a1[1] = -rotation_matrix[2, 0] + rotation_matrix[0, 2]
    a1[2] = -rotation_matrix[0, 1] + rotation_matrix[1, 0]

    sint = 0.5 * np.linalg.norm(a1)
    if sint < eps or abs(sint) > 1 + eps:
        raise ValueError("Invalid rotation matrix")

    # small rounding errors that make |sint|>1.0 produce NaN in the next arcsin
    # function, so we remove them
    if abs(sint) > 1:
        sint = np.sign(sint)

    # The direction of the axis is arbitrarily chosen, with positive z. In the
    # xy plane with positive x, and along y with positive y.
    ax = a1.copy()
    if ax[2] < -eps:
        ax = -ax
    elif abs(ax[2]) < eps and ax[1] < -eps:
        ax = -ax
    elif abs(ax[2]) < eps and abs(ax[1]) < eps and ax[0] < -eps:
        ax = -ax

    if abs(a1[0]) > eps:
        sint = abs(sint) * np.sign(a1[0] / ax[0])
    elif abs(a1[1]) > eps:
        sint = abs(sint) * np.sign(a1[1] / ax[1])
    elif abs(a1[2]) > eps:
        sint = abs(sint) * np.sign(a1[2] / ax[2])

    # Compute the cos of the angle
    ax = a1 / (2 * sint)
    cost = 0
    if abs(ax[0] ** 2 - 1) > eps:
        cost = (rotation_matrix[0, 0] - ax[0] ** 2) / (1 - ax[0] ** 2)
    elif abs(ax[1] ** 2 - 1) > eps:
        cost = (rotation_matrix[1, 1] - ax[1] ** 2) / (1 - ax[1] ** 2)
    elif abs(ax[2] ** 2 - 1) > eps:
        cost = (rotation_matrix[2, 2] - ax[2] ** 2) / (1 - ax[2] ** 2)

    if abs(sint ** 2 + cost ** 2 - 1) > eps:
        raise ValueError("Problem calculating rotation angle.")

    angle_rot1 = np.arcsin(sint) * 180 / np.pi
    if angle_rot1 < 0:
        if cost < 0:
            angle_rot1 = -angle_rot1 + 180
        else:
            angle_rot1 = 360 + angle_rot1
    else:
        if cost < 0:
            angle_rot1 = -angle_rot1 + 180

    return angle_rot1


def get_symmetry_type(symmetry_matrix, eps=1e-7):
    """This function receives a 3x3 orthogonal matrix which is a symmetry
    operation of the point group of the crystal written in cartesian
    coordinates and gives as output a code according to the following:

    1. identity
    2. inversion
    3. proper rotation of an angle <> 180 degrees
    4. proper rotation of 180 degrees
    5. mirror symmetry
    6. improper rotation

    Args:
        symmetry_matrix: A 3x3 symmetry matrix.
        eps: Numerical tolerance.

    Returns:
        The symmetry type.
    """

    # check for identity
    if np.allclose(symmetry_matrix, np.eye(3), atol=eps):
        return 1

    # check for inversion
    if np.allclose(symmetry_matrix, -np.eye(3), atol=eps):
        return 2

    det = np.linalg.det(symmetry_matrix)

    # Determinant equal to 1: proper rotation
    if abs(det - 1) < eps:
        # check if an eigenvalue is equal to -1 (180 rotation)
        det1 = np.linalg.det(symmetry_matrix + np.eye(3))
        if abs(det1) < eps:
            # 180 proper rotation
            return 4
        else:
            # proper rotation <> 180
            return 3

    # Determinant equal to -1: mirror symmetry or improper rotation
    if abs(det + 1) < eps:
        # check if an eigenvalue is equal to 1 (mirror symmetry)
        det1 = np.linalg.det(symmetry_matrix - np.eye(3))

        if abs(det1) < eps:
            # mirror symmetry
            return 5
        else:
            # improper rotation
            return 6

    raise ValueError("Unrecognised symmetry")
