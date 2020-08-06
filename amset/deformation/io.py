from pathlib import Path

import h5py
import numpy as np
from amset.constants import str_to_spin
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.outputs import Poscar


def write_deformation_potentials(
    deformation_potentials, kpoints, structure, filename="deformation.h5"
):
    with h5py.File(filename, "w") as f:
        for spin, spin_deform in deformation_potentials.items():
            name = "deformation_potentials_{}".format(spin.name)
            f.create_dataset(name, data=spin_deform, compression="gzip")
        f["structure"] = np.string_(structure.to_json())
        f["kpoints"] = kpoints


def load_deformation_potentials(filename):
    deformation_potentials = {}
    with h5py.File(filename, "r") as f:
        deform_keys = [k for k in list(f.keys()) if "deformation_potentials" in k]
        for key in deform_keys:
            spin = str_to_spin[key.split("_")[-1]]
            deformation_potentials[spin] = np.array(f[key])

        structure_str = np.string_(np.array(f["structure"])).decode()
        structure = Structure.from_str(structure_str, fmt="json")
        kpoints = np.array(f["kpoints"])

    return deformation_potentials, kpoints, structure


def write_deformed_poscars(deformed_structures, directory="."):
    n_deformations = len(deformed_structures)
    n_digits = int(np.floor(np.log10(n_deformations)) + 2)
    directory = Path(directory)
    for i, deformed_structure in enumerate(deformed_structures):
        # pad with leading zeros so the files are sorted correctly
        filename = "POSCAR-{0:0{1}}".format(i + 1, n_digits)
        deform_poscar = Poscar(deformed_structure)
        deform_poscar.write_file(directory / filename, significant_figures=16)
