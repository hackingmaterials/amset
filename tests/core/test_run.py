"""
Main integration test for amset calculations

Tests running amset from:
- A vasprun.xml file
- A directory with override settings specified

The following tests are performed:
- don't write mesh, using projections + deformation potential tuple + single elastic
  constant/piezoelectric for Silicon
- write mesh file, using projections + deformation potential tuple + single elastic
  constant/piezoelectric for Silicon
- use wavefunction coefficients + using deformation potential file + full elastic
  constant/piezoelectric for Silicon
- use wavefunction coefficients + using deformation potential file + full elastic
  constant/piezoelectric + no cache for Silicon
- use wavefunction coefficients + using deformation potential file + full elastic
  constant/piezoelectric for Gallium Arsenide
- don't write mesh, using projections + deformation potential tuple + single elastic
  constant/piezoelectric for K2ReF6 (tricky spin polarized system)
"""
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from monty.serialization import dumpfn

from amset.core.run import Runner

si_settings_no_mesh: Dict[str, Any] = {
    "interpolation_factor": 5,
    "doping": [-1e15, -1e16, -1e17],
    "deformation_potential": (6.5, 6.5),
    "elastic_constant": 190,
    "static_dielectric": 13.1,
    "use_projections": True,
    "nworkers": 1,
}
si_settings_mesh = deepcopy(si_settings_no_mesh)
si_settings_mesh.update({"write_mesh": True})
si_settings_wavefunction = deepcopy(si_settings_no_mesh)
si_settings_wavefunction.update(
    {
        "deformation_potential": "deformation.h5",
        "elastic_constant": [
            [144, 53, 53, 0, 0, 0],
            [53, 144, 53, 0, 0, 0],
            [53, 53, 144, 0, 0, 0],
            [0, 0, 0, 75, 0, 0],
            [0, 0, 0, 0, 75, 0],
            [0, 0, 0, 0, 0, 75],
        ],
        "static_dielectric": [[11.7, 0, 0], [0, 11.7, 0], [0, 0, 11.7]],
        "use_projections": False,
    }
)
si_settings_wavefunction_nocache = deepcopy(si_settings_wavefunction)
si_settings_wavefunction_nocache.update({"cache_wavefunction": False})
si_transport_projections = {
    ("mobility", ("overall", (0, 0))): 1443.887767841073,
    ("mobility", ("overall", (-1, 0))): 653.8846044592898,
    ("seebeck", (0, 0)): -1.01e03,
    ("seebeck", (-1, 0)): -7.18e02,
    ("conductivity", (0, 0)): 2.43e01,
    ("conductivity", (-1, 0)): 1.05e03,
    ("electronic_thermal_conductivity", (0, 0)): 0.0012601364875891449,
    ("electronic_thermal_conductivity", (-1, 0)): 0.005528446007774847,
}
si_transport_wavefunction = {
    ("mobility", ("overall", (0, 0))): 1138.346074448407,
    ("mobility", ("overall", (-1, 0))): 538.9257004979358,
    ("seebeck", (0, 0)): -977.8109947263332,
    ("seebeck", (-1, 0)): -718.8554371827041,
    ("conductivity", (0, 0)): 19.473244465428902,
    ("conductivity", (-1, 0)): 863.4587743458269,
    ("electronic_thermal_conductivity", (0, 0)): 0.0015021583389928138,
    ("electronic_thermal_conductivity", (-1, 0)): 0.004664695625665922,
}

gaas_settings_wavefunction = {
    "doping": [-3e13],
    "temperatures": [201, 290, 401, 506, 605, 789, 994],
    "bandgap": 1.33,
    "interpolation_factor": 5,
    "deformation_potential": (1.2, 8.6),
    "elastic_constant": 139.7,
    "donor_charge": 1,
    "acceptor_charge": 1,
    "static_dielectric": 12.18,
    "high_frequency_dielectric": 10.32,
    "pop_frequency": 8.16,
    "nworkers": 1,
}
gaas_transport = {
    ("mobility", ("overall", (0, 0))): 21146.50712288404,
    ("mobility", ("overall", (0, -1))): 2168.9439586020367,
    ("seebeck", (0, 0)): -858.4197653054092,
    ("seebeck", (0, -1)): -608.8106879408261,
    ("conductivity", (0, 0)): 10.164124084492931,
    ("conductivity", (0, -1)): 216.5534973787272,
    ("electronic_thermal_conductivity", (0, -1)): 0.02553675563131928,
}

test_data = [
    pytest.param(
        "Si",
        si_settings_no_mesh,
        si_transport_projections,
        0.001,
        ["transport", "!mesh"],
        ["ADP", "IMP"],
        id="Si (no mesh, projections, simple scats)",
    ),
    pytest.param(
        "Si",
        si_settings_mesh,
        si_transport_projections,
        0.001,
        ["transport", "mesh"],
        ["ADP", "IMP"],
        id="Si (mesh, projections, simple scats)",
    ),
    pytest.param(
        "Si",
        si_settings_wavefunction,
        si_transport_wavefunction,
        0.001,
        ["transport", "!mesh"],
        ["ADP", "IMP"],
        id="Si (wavefunction, best scats)",
    ),
    pytest.param(
        "Si",
        si_settings_wavefunction_nocache,
        si_transport_wavefunction,
        0.001,
        ["transport", "!mesh"],
        ["ADP", "IMP"],
        id="Si (wavefunction, best scats, no cache)",
    ),
    pytest.param(
        "GaAs",
        gaas_settings_wavefunction,
        gaas_transport,
        0.001,
        ["transport", "!mesh"],
        ["ADP", "IMP", "POP"],
        id="GaAs (wavefunction, best scats)",
    ),
]


@pytest.mark.usefixtures("clean_dir")
@pytest.mark.parametrize("system,settings,transport,max_aniso,files,scats", test_data)
def test_run_amset_from_vasprun(
    example_dir, system, settings, transport, max_aniso, files, scats
):
    vasprun, settings = _prep_inputs(example_dir, system, settings)
    runner = Runner.from_vasprun(vasprun, settings)
    amset_data = runner.run()
    _validate_data(amset_data, transport, max_aniso, files, scats)


@pytest.mark.usefixtures("clean_dir")
@pytest.mark.parametrize(
    "system,settings,transport,max_aniso,files,scats", [test_data[0]]
)
def test_run_amset_from_directory(
    example_dir, system, settings, transport, max_aniso, files, scats
):
    vasprun, settings = _prep_inputs(example_dir, system, settings)
    dumpfn(settings, "settings.yaml")
    runner = Runner.from_directory(".", input_file=vasprun, settings_override=settings)
    amset_data = runner.run()
    _validate_data(amset_data, transport, max_aniso, files, scats)


@pytest.mark.usefixtures("clean_dir")
def test_run_tricky_spin_polarized(band_structure_data):
    settings = {
        "interpolation_factor": 2,
        "temperatures": [300],
        "doping": [1e15],
        "deformation_potential": (6.5, 6.5),
        "elastic_constant": 190,
        "use_projections": True,
        "scattering_type": ["ADP"],
        "nworkers": 1,
    }
    files = ["transport", "!mesh"]
    scats = ["ADP"]
    transport = {
        ("mobility", ("overall", (0, 0))): 1.261213643946453,
        ("seebeck", (0, 0)): 1356.1341462605026,
    }

    bs = band_structure_data["tricky_sp"]["band_structure"]
    nelect = band_structure_data["tricky_sp"]["nelect"]
    runner = Runner(bs, nelect, settings)
    amset_data = runner.run()
    _validate_data(amset_data, transport, 1, files, scats)


def _prep_inputs(example_dir, system, settings):
    """Prepare calculation inputs"""

    input_dir = example_dir / system
    vasprun = input_dir / "vasprun.xml.gz"
    settings["wavefunction_coefficients"] = input_dir / "wavefunction.h5"

    ed = settings.get("deformation_potential")
    if isinstance(ed, str):
        settings["deformation_potential"] = input_dir / ed
    return vasprun, settings


def _validate_data(amset_data, transport, max_aniso, files, scats):
    """Validate calculation outputs"""

    # assert results are isotropic (often a warning sign that something is wrong)
    for temp_mobility in amset_data.mobility["overall"][:, 0]:
        eigs = np.linalg.eigvals(temp_mobility)
        assert np.abs(1 - (np.max(eigs) / np.min(eigs))) < max_aniso

    # check correct files are written out
    ls = [p.as_posix() for p in Path(".").glob("*")]
    for file in files:
        if file[0] == "!":
            assert all([file[1:] not in x for x in ls])
        else:
            assert any([file in x for x in ls])

    # check transport properties
    for (prop, loc), expected in transport.items():
        calculated = getattr(amset_data, prop)
        if prop == "mobility":
            value = calculated[loc[0]][loc[1]]
        else:
            value = calculated[loc]

        assert value.shape == (3, 3)

        value = np.average(np.linalg.eigvalsh(value))
        print("('{}', {}): {},".format(prop, loc, value))

        # assert values agree to within 1 %
        assert (
            np.abs(1 - value / expected) < 0.01
        ), f"property: {prop}, loc: {loc}, differs by more than 1%: calculated: {value}, expected: {expected}"

    # check scattering types
    assert set(amset_data.scattering_labels) == set(scats)
