import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
from monty.serialization import dumpfn, loadfn


@pytest.fixture(scope="session")
def test_dir():
    module_dir = Path(__file__).resolve().parent
    test_dir = module_dir / "test_data"
    return test_dir.resolve()


@pytest.fixture(scope="session")
def example_dir():
    root_dir = Path(__file__).resolve().parent.parent
    example_dir = root_dir / "examples"
    return example_dir.resolve()


@pytest.fixture
def log_to_stdout():
    # Set Logging
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    root.addHandler(ch)


@pytest.fixture
def clean_dir():
    old_cwd = os.getcwd()
    newpath = tempfile.mkdtemp()
    os.chdir(newpath)
    yield
    os.chdir(old_cwd)
    shutil.rmtree(newpath)


@pytest.fixture(
    params=[
        "Si_227",
        "Fe_229",
        "S_58",
        "Rb2P3_69",
        "K2Au3_71",
        "LaI3_63",
        "KCeF4_123",
        "RbO2_129",
        "BaN2_15",
        "TiNi_11",
        "CaC2_2",
        "KNO3_160",
        "ZnO_186",
    ],
    ids=[
        "F cubic",
        "I cubic",
        "P orth",
        "I orth",
        "F orth",
        "C orth",
        "P tet",
        "I tet",
        "C mono",
        "P mono",
        "tri",
        "rhom",
        "hex",
    ],
    scope="session",
)
def symmetry_structure(test_dir, request):
    return loadfn(test_dir / "structures" / f"{request.param}.json.gz")


@pytest.fixture(scope="session")
def band_structure_data(test_dir):
    data = {}
    for band_structure_data_file in (test_dir / "band_structures").glob("*json*"):
        key = str(band_structure_data_file.name)
        for replace_str in [".json", ".gz", "band_structure_data"]:
            key = key.replace(replace_str, "")
        key = key.rstrip("_").lstrip("_")

        data[key] = loadfn(band_structure_data_file)
    return data


@pytest.fixture(scope="session")
def band_structures(band_structure_data):
    return {k: v["band_structure"] for k, v in band_structure_data.items()}


if __name__ == "__main__":
    # download test data
    materials = {
        "Si_227": "mp-149",  # F cubic
        "Fe_229": "mp-13",  # I cubic
        "S_58": "mp-558014",  # P orthorhombic
        "Rb2P3_69": "mp-2079",  # I orthorhombic
        "K2Au3_71": "mp-8700",  # F orthorhombic
        "LaI3_63": "mp-27979",  # C orthorhombic
        "KCeF4_123": "mp-1223451",  # P tetragonal
        "RbO2_129": "mp-12105",  # I tetragonal
        "BaN2_15": "mp-1001",  # C monoclinic
        "TiNi_11": "mp-1048",  # P monoclinic
        "CaC2_2": "mp-642822",  # triclinic
        "KNO3_160": "mp-6920",  # rhombohedral
        "ZnO_186": "mp-2133",  # hexagonal
    }
    from pymatgen.ext.matproj import MPRester

    _mpr = MPRester()
    _module_dir = Path(__file__).resolve().parent
    _structure_dir = _module_dir / "test_data" / "structures"

    if not _structure_dir.exists():
        _structure_dir.mkdir()

    for name, mp_id in materials.items():
        _s = _mpr.get_structure_by_material_id(mp_id)
        dumpfn(_s, _structure_dir / f"{name}.json.gz")
