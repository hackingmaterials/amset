import pytest
import numpy as np

from amset.electronic_structure.symmetry import (
    get_symmetry_type,
    get_rotation_angle,
    get_rotation_axis,
    rotation_matrix_to_su2,
)


@pytest.mark.parametrize(
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
