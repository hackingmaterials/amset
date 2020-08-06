import numpy as np
import h5py
from pymatgen.core.structure import Structure

from amset.constants import str_to_spin


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
