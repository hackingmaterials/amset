import numpy as np

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def kpoints_to_first_bz(kpoints: np.ndarray) -> np.ndarray:
    """Translate fractional k-points to the first Brillouin zone.

    I.e. all k-points will have fractional coordinates:
        -0.5 <= fractional coordinates < 0.5

    Args:
        kpoints: The k-points in fractional coordinates.

    Returns:
        The translated k-points.
    """
    kp = kpoints - np.round(kpoints)
    kp[kp == 0.5] = -0.5
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
    # return (addresses + shift) / mesh

    full_kpoints = (addresses + shift) / mesh
    sort_idx = np.lexsort((full_kpoints[:, 2], full_kpoints[:, 2] < 0,
                           full_kpoints[:, 1], full_kpoints[:, 1] < 0,
                           full_kpoints[:, 0], full_kpoints[:, 0] < 0))
    full_kpoints = full_kpoints[sort_idx]
    return full_kpoints


def get_dense_kpoint_mesh(mesh):
    kpts = np.stack(
        np.mgrid[
        0:mesh[0] + 1,
        0:mesh[1] + 1,
        0:mesh[2] + 1],
        axis=-1).reshape(-1, 3).astype(float)

    # remove central point for all even ndim as this will fall
    # exactly the in the centre of the grid and will be on top of
    # the original k-point
    if not any(mesh % 2):
        kpts = np.delete(kpts, int(1 + np.product(mesh + 1) / 2), axis=0)

    kpts /= mesh  # gets frac kpts between 0 and 1
    kpts -= 0.5
    return kpts


def get_symmetry_equivalent_kpoints(structure, kpoints, symprec=0.1, tol=1e-6,
                                    return_inverse=False,
                                    rotation_matrices=None,
                                    time_reversal_symmetry=True):
    round_dp = int(np.log10(1 / tol))

    def shift_and_round(k):
        k = kpoints_to_first_bz(k)
        k = np.round(k, round_dp)
        return list(map(tuple, k))

    kpoints = np.asarray(kpoints)
    round_kpoints = shift_and_round(kpoints)

    if rotation_matrices is None:
        sg = SpacegroupAnalyzer(structure, symprec=symprec)
        symmops = sg.get_symmetry_operations(cartesian=False)
        rotation_matrices = np.array([o.rotation_matrix for o in symmops])

    if time_reversal_symmetry:
        all_rotations = np.concatenate((rotation_matrices, -rotation_matrices))
        rotation_matrices = np.unique(all_rotations, axis=0)

    cart_rotation_matrices = np.array(
        [similarity_transformation(
            structure.lattice.reciprocal_lattice.matrix, r.T)
         for r in rotation_matrices])

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
            new_points = shift_and_round(np.dot(kpoints[i], rotation_matrices))

            equiv_points_mapping.update(zip(new_points, [i] * len(new_points)))
            rotation_matrix_mapping[i] = dict(
                zip(new_points, cart_rotation_matrices))

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


def symmetrize_kpoints(structure, kpoints, symprec=0.1, tol=1e-6,
                       time_reversal_symmetry=True):
    round_dp = int(np.log10(1 / tol))

    def shift_and_round(k):
        k = kpoints_to_first_bz(k)
        k = np.round(k, round_dp)
        return list(map(tuple, k))

    sg = SpacegroupAnalyzer(structure, symprec=symprec)
    symmops = sg.get_symmetry_operations(cartesian=False)
    rotation_matrices = np.array([o.rotation_matrix for o in symmops])

    if time_reversal_symmetry:
        all_rotations = np.concatenate((rotation_matrices, -rotation_matrices))
        rotation_matrices = np.unique(all_rotations, axis=0)

    kpoints = kpoints_to_first_bz(kpoints)

    symmetrized_kpoints = np.concatenate(
        [np.dot(kpoints, rot) for rot in rotation_matrices])
    symmetrized_kpoints = kpoints_to_first_bz(symmetrized_kpoints)
    _, unique_idxs = np.unique(shift_and_round(symmetrized_kpoints),
                               axis=0, return_index=True)

    return symmetrized_kpoints[unique_idxs]
