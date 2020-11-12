import numpy as np
import pytest
from pymatgen.io.ase import AseAtomsAdaptor
from pytest import mark
from spglib import get_ir_reciprocal_mesh

from amset.electronic_structure.symmetry import (
    expand_kpoints,
    get_rotation_angle,
    get_rotation_axis,
    get_symmetry_type,
    rotation_matrix_to_su2,
)


@mark.parametrize(
    "symmetry_matrix,expected",
    [
        pytest.param([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 1, id="identity"),
        pytest.param([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], 2, id="inversion"),
        pytest.param([[1, 0, 0], [0, 0, -1], [0, 1, 0]], 3, id="x-rotation (90)"),
        pytest.param([[1, 0, 0], [0, -1, 0], [0, 0, -1]], 4, id="x-rotation (180)"),
        pytest.param(
            [
                [0.5, 0.5, 0.7071068],
                [0.5, 0.5, -0.7071068],
                [-0.7071068, 0.7071068, 0.0],
            ],
            3,
            id="xy-rotation (90)",
        ),
        pytest.param([[0, 1, 0], [1, 0, 0], [0, 0, -1]], 4, id="xy-rotation (180)"),
        pytest.param([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], 5, id="x-reflection"),
        pytest.param([[0, -1, 0], [-1, 0, 0], [0, 0, 1]], 5, id="xy-reflection"),
        pytest.param(
            [[-1, 0, 0], [0, 0, -1], [0, 1, 0]], 6, id="x-improper rotation (90)"
        ),
        pytest.param(
            [[0, -1, 0], [1, 0, 0], [-1, 0, 0]],
            pytest.raises(ValueError),
            id="malformed matrix",
        ),
    ],
)
def test_get_symmetry_type(symmetry_matrix, expected):
    if not isinstance(expected, int):
        with expected:
            get_symmetry_type(symmetry_matrix)
    else:
        assert get_symmetry_type(symmetry_matrix) == expected


@pytest.mark.parametrize(
    "symmetry_matrix,expected",
    [
        pytest.param(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], pytest.raises(ValueError), id="identity"
        ),
        pytest.param(
            [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            pytest.raises(ValueError),
            id="inversion",
        ),
        pytest.param([[1, 0, 0], [0, 0, -1], [0, 1, 0]], 90, id="x-rotation (90)"),
        pytest.param([[1, 0, 0], [0, -1, 0], [0, 0, -1]], 180, id="x-rotation (180)"),
        pytest.param(
            [
                [0.5, 0.5, 0.7071068],
                [0.5, 0.5, -0.7071068],
                [-0.7071068, 0.7071068, 0.0],
            ],
            90,
            id="xy-rotation (90)",
        ),
        pytest.param([[0, 1, 0], [1, 0, 0], [0, 0, -1]], 180, id="xy-rotation (180)"),
        pytest.param(
            [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
            pytest.raises(ValueError),
            id="x-reflection",
        ),
        pytest.param(
            [[0, -1, 0], [-1, 0, 0], [0, 0, 1]],
            pytest.raises(ValueError),
            id="xy-reflection",
        ),
        pytest.param(
            [[-1, 0, 0], [0, 0, -1], [0, 1, 0]], 90, id="x-improper rotation (90)"
        ),
        pytest.param(
            [[0, -1, 0], [1, 0, 0], [-1, 0, 0]],
            pytest.raises(ValueError),
            id="malformed matrix",
        ),
    ],
)
def test_get_rotation_angle(symmetry_matrix, expected):
    if not isinstance(expected, int):
        with expected:
            get_rotation_angle(symmetry_matrix)
    else:
        assert get_rotation_angle(symmetry_matrix) == expected


@pytest.mark.parametrize(
    "symmetry_matrix,expected",
    [
        pytest.param(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], pytest.raises(ValueError), id="identity"
        ),
        pytest.param(
            [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            pytest.raises(ValueError),
            id="inversion",
        ),
        pytest.param(
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]], [1, 0, 0], id="x-rotation (90)"
        ),
        pytest.param(
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]], [1, 0, 0], id="x-rotation (180)"
        ),
        pytest.param(
            [
                [0.5, 0.5, 0.7071068],
                [0.5, 0.5, -0.7071068],
                [-0.7071068, 0.7071068, 0.0],
            ],
            [0.70710678118, 0.70710678118, 0.0],
            id="xy-rotation (90)",
        ),
        pytest.param(
            [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
            [0.70710678118, 0.70710678118, 0.0],
            id="xy-rotation (180)",
        ),
        pytest.param(
            [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
            pytest.raises(ValueError),
            id="x-reflection",
        ),
        pytest.param(
            [[0, -1, 0], [-1, 0, 0], [0, 0, 1]],
            pytest.raises(ValueError),
            id="xy-reflection",
        ),
        pytest.param(
            [[-1, 0, 0], [0, 0, -1], [0, 1, 0]],
            [1, 0, 0],
            id="x-improper rotation (90)",
        ),
        pytest.param(
            [[0, -1, 0], [1, 0, 0], [-1, 0, 0]],
            pytest.raises(ValueError),
            id="malformed matrix",
        ),
    ],
)
def test_get_rotation_axis(symmetry_matrix, expected):
    if not isinstance(expected, list):
        with expected:
            get_rotation_axis(symmetry_matrix)
    else:
        axis = get_rotation_axis(symmetry_matrix)
        np.testing.assert_array_almost_equal(axis, expected)


@pytest.mark.parametrize(
    "symmetry_matrix,expected",
    [
        pytest.param(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1 + 0j, 0j], [0j, 1 + 0j]],
            id="identity",
        ),
        pytest.param(
            [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            [[1 + 0j, 0j], [0j, 1 + 0j]],
            id="inversion",
        ),
        pytest.param(
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
            [[0.707107 + 0j, -0 - 0.707107j], [0.707107j, 0.707107 + 0j]],
            id="x-rotation (90)",
        ),
        pytest.param(
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
            [[0j, -1j], [1j, 0j]],
            id="x-rotation (180)",
        ),
        pytest.param(
            [
                [0.5, 0.5, 0.7071068],
                [0.5, 0.5, -0.7071068],
                [-0.7071068, 0.7071068, 0.0],
            ],
            [[(0.707107 + 0j), (-0.5 - 0.5j)], [(-0.5 + 0.5j), (0.707107 + 0j)]],
            id="xy-rotation (90)",
        ),
        pytest.param(
            [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
            [[0j, -0.7071068 - 0.707107j], [-0.7071068 + 0.707107j, 0j]],
            id="xy-rotation (180)",
        ),
        pytest.param(
            [[-1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0j, -1j], [1j, 0j]], id="x-reflection"
        ),
        pytest.param(
            [[0, -1, 0], [-1, 0, 0], [0, 0, 1]],
            [[0j, -0.7071068 - 0.707107j], [-0.7071068 + 0.707107j, 0j]],
            id="xy-reflection",
        ),
        pytest.param(
            [[-1, 0, 0], [0, 0, -1], [0, 1, 0]],
            [[0.707107 + 0j, 0.707107j], [-0 - 0.707107j, 0.707107 + 0j]],
            id="x-improper rotation (90)",
        ),
        pytest.param(
            [[0, -1, 0], [1, 0, 0], [-1, 0, 0]],
            pytest.raises(ValueError),
            id="malformed matrix",
        ),
    ],
)
def test_rotation_matrix_to_su2(symmetry_matrix, expected):
    if not isinstance(expected, list):
        with expected:
            rotation_matrix_to_su2(symmetry_matrix)
    else:
        su2 = rotation_matrix_to_su2(symmetry_matrix)
        np.testing.assert_array_almost_equal(su2, expected)


@pytest.fixture(params=[0, 1], ids=["gamma", "shift"])
def shift(request):
    idx = request.param
    return np.array([idx, idx, idx])


@pytest.fixture(
    params=[
        [5, 5, 5],
        [6, 6, 6],
        [21, 21, 21],
        [22, 22, 22],
        [5, 7, 9],
        [6, 8, 10],
        [21, 8, 13],
        [4, 5, 1],
    ],
    ids=[
        "odd (iso)",
        "even (iso)",
        "large odd (iso)",
        "large even (iso)",
        "odd (aniso)",
        "even (aniso)",
        "mixed (aniso)",
        "2D material",
    ],
)
def mesh(request):
    return np.array(request.param)


def test_expand_kpoints(symmetry_structure, shift, mesh):
    def _kpoints_to_first_bz(kp):
        """helper function to map k-points to 1st BZ"""
        kp = np.array(kp)
        kp = kp - np.round(kp)
        kp[kp.round(8) == -0.5] = 0.5
        return kp

    def _sort_kpoints(kp):
        """Helper function to put k-points in a consistent order"""
        kp = kp.round(8)
        sort_idx = np.lexsort((kp[:, 2], kp[:, 1], kp[:, 0]))
        return kp[sort_idx]

    # generate true k-points and IR k-points using spglib
    atoms = AseAtomsAdaptor.get_atoms(symmetry_structure)
    mapping, addresses = get_ir_reciprocal_mesh(mesh, atoms, is_shift=shift)
    true_kpoints = addresses / mesh + shift / (mesh * 2)
    true_kpoints = _kpoints_to_first_bz(true_kpoints)
    true_kpoints_sort = _sort_kpoints(true_kpoints)

    ir_mapping = np.unique(mapping, return_index=False)
    ir_kpoints = true_kpoints[ir_mapping]

    # try to expand the irreducible k-points back  to the full BZ
    full_kpoints, rots, _, _, op_mapping, kp_mapping = expand_kpoints(
        symmetry_structure, ir_kpoints, return_mapping=True
    )
    full_kpoints = _kpoints_to_first_bz(full_kpoints)
    full_kpoints_sort = _sort_kpoints(full_kpoints)

    # assert final k-points match the expected true k-points
    diff = np.linalg.norm(full_kpoints_sort - true_kpoints_sort, axis=1)
    assert np.max(diff) == 0

    # now ensure that the rotation mapping actually works
    rotated_kpoints = []
    for r, k in zip(op_mapping, kp_mapping):
        rotated_kpoints.append(np.dot(rots[r], ir_kpoints[k]))
    rotated_kpoints = _kpoints_to_first_bz(rotated_kpoints)
    rotated_kpoints_sort = _sort_kpoints(rotated_kpoints)

    # assert rotated k-points match the expected true k-points
    diff = np.linalg.norm(rotated_kpoints_sort - true_kpoints_sort, axis=1)
    assert np.max(diff) == 0
