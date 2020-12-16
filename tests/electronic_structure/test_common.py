import pytest
from pymatgen import Spin

from amset.electronic_structure.common import (
    get_cbm_energy,
    get_efermi,
    get_vb_idx,
    get_vbm_energy,
)


@pytest.mark.parametrize(
    "system,energy_cutoff,expected",
    [
        pytest.param(
            "tricky_sp", 1.5, {Spin.up: 2, Spin.down: -1}, id="default tricky"
        ),
        pytest.param("tricky_sp", 5, {Spin.up: 12, Spin.down: 10}, id="large tricky"),
    ],
)
def test_get_vb_idx_tricky(band_structures, system, energy_cutoff, expected):
    result = get_vb_idx(energy_cutoff, band_structures[system])
    assert result == expected


@pytest.mark.parametrize(
    "system,vb_idx,expected",
    [
        pytest.param(
            "tricky_sp", {Spin.up: 37, Spin.down: 34}, 0.6084, id="both spin tricky"
        ),
        pytest.param(
            "tricky_sp", {Spin.up: 37, Spin.down: -1}, 0.6084, id="up only tricky"
        ),
        pytest.param(
            "tricky_sp", {Spin.up: -1, Spin.down: 34}, -2.9677, id="down only tricky"
        ),
    ],
)
def test_get_vbm_energy(band_structures, system, vb_idx, expected):
    result = get_vbm_energy(band_structures[system].bands, vb_idx)
    assert result == expected


@pytest.mark.parametrize(
    "system,vb_idx,expected",
    [
        pytest.param(
            "tricky_sp", {Spin.up: 38, Spin.down: 35}, 2.3641, id="both spin tricky"
        ),
        pytest.param(
            "tricky_sp", {Spin.up: 38, Spin.down: 79}, 4.434, id="up only tricky"
        ),
        pytest.param(
            "tricky_sp", {Spin.up: 79, Spin.down: 35}, 2.3641, id="down only tricky"
        ),
    ],
)
def test_get_cbm_energy(band_structures, system, vb_idx, expected):
    result = get_cbm_energy(band_structures[system].bands, vb_idx)
    assert result == expected


@pytest.mark.parametrize(
    "system,vb_idx,expected",
    [
        pytest.param(
            "tricky_sp", {Spin.up: 38, Spin.down: 35}, 3.4574, id="both spin tricky"
        ),
        pytest.param(
            "tricky_sp", {Spin.up: 38, Spin.down: 79}, 14.50355, id="up only tricky"
        ),
        pytest.param(
            "tricky_sp", {Spin.up: 79, Spin.down: 35}, 13.4251, id="down only tricky"
        ),
    ],
)
def test_get_efermi(band_structures, system, vb_idx, expected):
    result = get_efermi(band_structures[system].bands, vb_idx)
    assert result == expected
