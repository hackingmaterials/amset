from contextlib import contextmanager

import numpy as np
import pytest
from pymatgen.electronic_structure.core import Spin

from amset.util import (
    cast_dict_list,
    cast_dict_ndarray,
    cast_elastic_tensor,
    cast_piezoelectric_tensor,
    cast_tensor,
    get_progress_bar,
    groupby,
    parse_deformation_potential,
    parse_doping,
    parse_ibands,
    parse_temperatures,
    tensor_average,
    validate_settings,
)


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize(
    "tensor,expected",
    [
        pytest.param([[3, 0, 0], [0, 4, 0], [0, 0, 5]], 4, id="diagonal"),
        pytest.param([[0, 3, 3], [3, 3, 4], [3, 4, 3]], 2, id="off-diagonal"),
    ],
)
def test_tensor_average(tensor, expected):
    assert tensor_average(tensor) == expected


@pytest.mark.parametrize(
    "elements, groups, expected",
    [
        pytest.param(
            ["a", "b", "1", "2", "c", "d"],
            [2, 0, 1, 2, 0, 0],
            [["b", "c", "d"], ["1"], ["a", "2"]],
            id="mixed",
        ),
        pytest.param(
            [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
            [2, 2, 1, 0, 0],
            [[[3, 3, 3], [4, 4, 4]], [[2, 2, 2]], [[0, 0, 0], [1, 1, 1]]],
            id="coords",
        ),
    ],
)
def test_groupby(elements, groups, expected):
    elements = ["a", "b", "1", "2", "c", "d"]
    groups = [2, 0, 1, 2, 0, 0]
    expected_output = [["b", "c", "d"], ["1"], ["a", "2"]]
    output = groupby(elements, groups)
    output = [x.tolist() for x in output]
    assert output == expected_output


_expected_elastic = [
    [
        [[3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 1.5, 0.0], [1.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.5], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
    ],
    [
        [[0.0, 1.5, 0.0], [1.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [0.0, 1.5, 0.0]],
    ],
    [
        [[0.0, 0.0, 1.5], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [0.0, 1.5, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 3.0]],
    ],
]
_elastic_voigt = [
    [3, 0, 0, 0, 0, 0],
    [0, 3, 0, 0, 0, 0],
    [0, 0, 3, 0, 0, 0],
    [0, 0, 0, 1.5, 0, 0],
    [0, 0, 0, 0, 1.5, 0],
    [0, 0, 0, 0, 0, 1.5],
]
_expected_piezo = [
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0084], [0.0, 0.0084, 0.0]],
    [[0.0, 0.0, 0.0084], [0.0, 0.0, 0.0], [0.0084, 0.0, 0.0]],
    [[0.0, 0.0084, 0.0], [0.0084, 0.0, 0.0], [0.0, 0.0, 0.0]],
]
_piezo_voigt = [
    [0, 0, 0, 0.0084, 0, 0],
    [0, 0, 0, 0, 0.0084, 0],
    [0, 0, 0, 0, 0, 0.0084],
]


@pytest.mark.parametrize(
    "settings,expected",
    [
        pytest.param(
            {},
            {
                "scattering_type": "auto",
                "temperatures": np.array([300]),
                "calculate_mobility": True,
                "separate_mobility": True,
                "mobility_rates_only": False,
            },
            id="empty",
        ),
        pytest.param(
            {"doping": "1E16:1E20:5", "temperatures": "100:500:5"},
            {
                "doping": np.array([1e16, 1e17, 1e18, 1e19, 1e20]),
                "temperatures": np.array([100, 200, 300, 400, 500]),
            },
            id="doping",
        ),
        pytest.param(
            {"deformation_potential": "1,2"},
            {"deformation_potential": (1, 2)},
            id="deformation (str-1)",
        ),
        pytest.param(
            {"deformation_potential": "1."},
            {"deformation_potential": 1.0},
            id="deformation (str-2)",
        ),
        pytest.param(
            {"deformation_potential": "deformation.h5"},
            {"deformation_potential": "deformation.h5"},
            id="deformation (str-3)",
        ),
        pytest.param(
            {
                "static_dielectric": 3,
                "high_frequency_dielectric": 3,
                "elastic_constant": 3,
            },
            {
                "static_dielectric": np.eye(3) * 3,
                "high_frequency_dielectric": np.eye(3) * 3,
                "elastic_constant": np.array(_expected_elastic),
            },
            id="tensor cast (int)",
        ),
        pytest.param(
            {"static_dielectric": [1, 2, 3], "high_frequency_dielectric": [1, 2, 3]},
            {
                "static_dielectric": np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
                "high_frequency_dielectric": np.array(
                    [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
                ),
            },
            id="tensor cast (list)",
        ),
        pytest.param(
            {
                "static_dielectric": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                "high_frequency_dielectric": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                "elastic_constant": _expected_elastic,
                "piezoelectric_constant": _piezo_voigt,
            },
            {
                "static_dielectric": np.arange(9).reshape(3, 3),
                "high_frequency_dielectric": np.arange(9).reshape(3, 3),
                "elastic_constant": np.array(_expected_elastic),
                "piezoelectric_constant": np.array(_expected_piezo),
            },
            id="tensor cast (array)",
        ),
        pytest.param({"donor_charge": 2}, {"defect_charge": 2}, id="d charge"),
        pytest.param({"acceptor_charge": 2}, {"defect_charge": 2}, id="a charge"),
        pytest.param({"mispelt_parameter": 1}, pytest.raises(ValueError), id="raises"),
    ],
)
def test_validate_settings(settings, expected):
    if not isinstance(expected, dict):
        # ensure errors are thrown correctly
        with expected:
            validate_settings(settings)

    else:
        settings = validate_settings(settings)
        for name, expected_value in expected.items():
            value = settings[name]
            assert isinstance(settings[name], type(expected_value))

            if isinstance(value, np.ndarray):
                expected_value = expected_value.tolist()
                value = value.tolist()

            assert expected_value == value


@pytest.mark.parametrize(
    "value,expected",
    [
        pytest.param(3, np.eye(3) * 3, id="int"),
        pytest.param([0, 1, 2], [[0, 0, 0], [0, 1, 0], [0, 0, 2]], id="list"),
    ],
)
def test_cast_tensor(value, expected):
    np.testing.assert_array_equal(cast_tensor(value), expected)


@pytest.mark.parametrize(
    "value,expected",
    [
        pytest.param(3, _expected_elastic, id="int"),
        pytest.param(_elastic_voigt, _expected_elastic, id="Voigt"),
        pytest.param(_expected_elastic, _expected_elastic, id="3x3x3x3"),
    ],
)
def test_cast_elastic_tensor(value, expected):
    np.testing.assert_array_equal(cast_elastic_tensor(value), expected)


@pytest.mark.parametrize(
    "value,expected",
    [
        pytest.param(_piezo_voigt, _expected_piezo, id="Voigt"),
        pytest.param(_expected_piezo, _expected_piezo, id="3x3x3"),
    ],
)
def test_cast_piezoelectric_tensor(value, expected):
    np.testing.assert_array_equal(cast_piezoelectric_tensor(value), expected)


@pytest.mark.parametrize(
    "value,expected",
    [
        pytest.param(
            {"a": np.array([1, 2, 3]), "c": [1]},
            {"a": [1, 2, 3], "c": [1]},
            id="single",
        ),
        pytest.param(
            {"a": np.array([1, 2, 3]), "c": [1], 123: np.array([[0, 0], [0, 0]])},
            {"a": [1, 2, 3], "c": [1], 123: [[0, 0], [0, 0]]},
            id="double",
        ),
        pytest.param(
            {"a": {"b": np.array([1, 2, 3])}, "c": {"d": np.array([1, 2])}},
            {"a": {"b": [1, 2, 3]}, "c": {"d": [1, 2]}},
            id="nested",
        ),
        pytest.param(
            {"a": {Spin.up: np.array([1, 2, 3])}, "c": {"d": np.array([1, 2])}},
            {"a": {"up": [1, 2, 3]}, "c": {"d": [1, 2]}},
            id="spin",
        ),
    ],
)
def test_cast_dict_list(value, expected):
    assert cast_dict_list(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        pytest.param(
            {"a": [1, 2, 3], "c": 1}, {"a": np.array([1, 2, 3]), "c": 1}, id="single"
        ),
        pytest.param(
            {"a": [1, 2, 3], "c": 1, 123: [[0, 0], [0, 0]]},
            {"a": np.array([1, 2, 3]), "c": 1, 123: np.array([[0, 0], [0, 0]])},
            id="double",
        ),
        pytest.param(
            {"a": {"b": [1, 2, 3]}, "c": {"d": [1, 2]}},
            {"a": {"b": np.array([1, 2, 3])}, "c": {"d": np.array([1, 2])}},
            id="nested",
        ),
        pytest.param(
            {"a": {"up": [1, 2, 3]}, "c": {"d": [1, 2]}},
            {"a": {Spin.up: np.array([1, 2, 3])}, "c": {"d": np.array([1, 2])}},
            id="spin",
        ),
    ],
)
def test_cast_dict_ndarray(value, expected):
    def compare(a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            all_keys = list(a.keys()) + list(b.keys())
            for k in all_keys:
                compare(a[k], b[k])
        elif isinstance(a, dict) or isinstance(b, dict):
            assert False
        else:
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                a = a.tolist()
                b = b.tolist()
            elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                # both not numpy arrays
                assert False

            assert a == b

    compare(cast_dict_ndarray(value), expected)


@pytest.mark.parametrize(
    "value,expected",
    [
        pytest.param("1E16", [1e16], id="single"),
        pytest.param("1E16,1E17", [1e16, 1e17], id="multiple"),
        pytest.param("1E16:1E19:4", [1e16, 1e17, 1e18, 1e19], id="range"),
        pytest.param("1E16:1E19:4:1", pytest.raises(ValueError), id="error"),
    ],
)
def test_parse_doping(value, expected):
    if not isinstance(expected, list):
        with expected:
            parse_doping(value)
    else:
        parsed = parse_doping(value)
        assert type(parsed) == np.ndarray
        assert parsed.tolist() == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        pytest.param("100", [100], id="single"),
        pytest.param("100,200", [100, 200], id="multiple"),
        pytest.param("100:400:4", [100, 200, 300, 400], id="range"),
        pytest.param("100:400:4:1", pytest.raises(ValueError), id="error"),
    ],
)
def test_parse_temperatures(value, expected):
    if not isinstance(expected, list):
        with expected:
            parse_temperatures(value)
    else:
        parsed = parse_temperatures(value)
        assert type(parsed) == np.ndarray
        assert parsed.tolist() == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        pytest.param("deformation.h5", "deformation.h5", id="str"),
        pytest.param("2", 2, id="single"),
        pytest.param("2,5", (2, 5), id="both"),
        pytest.param("100,2,1", pytest.raises(ValueError), id="error"),
    ],
)
def test_parse_deformation_potential(value, expected):
    if not isinstance(expected, (tuple, int, float, str)):
        with expected:
            parse_deformation_potential(value)
    else:
        parsed = parse_deformation_potential(value)
        assert parsed == expected


@pytest.mark.parametrize(
    "iterable,total,error",
    [
        pytest.param([1, 2, 3], None, does_not_raise(), id="iterable"),
        pytest.param(None, 3, does_not_raise(), id="total"),
        pytest.param(None, None, pytest.raises(ValueError), id="error"),
    ],
)
def test_get_progress_bar(iterable, total, error):
    with error:
        pbar = get_progress_bar(iterable=iterable, total=total)
        pbar.close()


@pytest.mark.parametrize(
    "value,expected",
    [
        pytest.param("1", {Spin.up: [0]}, id="single up"),
        pytest.param("1, 2", {Spin.up: [0, 1]}, id="multiple up"),
        pytest.param("1:4", {Spin.up: [0, 1, 2, 3]}, id="range up"),
        pytest.param("1.2", {Spin.up: [0], Spin.down: [1]}, id="single up-down"),
        pytest.param(
            "1, 3, 5.2, 4, 6",
            {Spin.up: [0, 2, 4], Spin.down: [1, 3, 5]},
            id="multiple up-down",
        ),
        pytest.param(
            "1:4.2:5",
            {Spin.up: [0, 1, 2, 3], Spin.down: [1, 2, 3, 4]},
            id="range up-down",
        ),
        pytest.param([1, 2, 3, 4], {Spin.up: [0, 1, 2, 3]}, id="list up"),
        pytest.param(
            ([1, 2, 3, 4], [2, 3, 4, 5]),
            {Spin.up: [0, 1, 2, 3], Spin.down: [1, 2, 3, 4]},
            id="list up-down",
        ),
        pytest.param("100:400:4:1", pytest.raises(ValueError), id="error"),
    ],
)
def test_parse_ibands(value, expected):
    if not isinstance(expected, dict):
        with expected:
            parse_ibands(value)
    else:
        parsed = parse_ibands(value)
        parsed = {s: i.tolist() for s, i in parsed.items()}
        assert parsed == expected
