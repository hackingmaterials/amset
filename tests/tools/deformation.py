from pathlib import Path

import pytest
from click.testing import CliRunner
from pymatgen import Spin

from amset.deformation.io import load_deformation_potentials
from amset.tools.deformation import read


@pytest.mark.parametrize(
    "options,nbands",
    [
        pytest.param([], 5, id="default"),
        pytest.param(["--bands", "2:7"], 6, id="bands"),
    ],
)
def test_read(clean_dir, test_dir, options, nbands):
    deform_dir = test_dir / "deformation"
    folders = [str(deform_dir / f"0{i}") for i in range(4)]

    runner = CliRunner()
    result = runner.invoke(read, folders + options)

    assert result.exit_code == 0
    assert "band:    4  k-point: [ -0.38  -0.38   0.00  ]" in result.output
    assert "[  8.71   0.42   0.00]" in result.output

    deformation_file = Path("deformation.h5")
    assert deformation_file.exists()

    deformation_potentials, _, _ = load_deformation_potentials(deformation_file)
    assert deformation_potentials[Spin.up][4, 0, 0, 0] == pytest.approx(3.25)
    assert deformation_potentials[Spin.up][3, 10, 0, 0] == pytest.approx(7.3)

    assert len(deformation_potentials[Spin.up]) == nbands
