import h5py
import numpy as np
from pymatgen import Structure

from amset.constants import str_to_spin

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"


def write_coefficients(coeffs, gpoints, kpoints, structure, filename="coeffs.h5"):
    with h5py.File(filename, "w") as f:
        for spin, spin_coeffs in coeffs.items():
            name = "coefficients_{}".format(spin.name)
            dset = f.create_dataset(
                name, spin_coeffs.shape, compression="gzip", dtype=np.complex
            )
            dset[...] = spin_coeffs

        f["structure"] = np.string_(structure.to_json())
        f["kpoints"] = kpoints
        f["gpoints"] = gpoints


def load_coefficients(filename):
    coeffs = {}
    with h5py.File(filename, "r") as f:
        coeff_keys = [k for k in list(f.keys()) if "coefficients" in k]
        for key in coeff_keys:
            spin = str_to_spin[key.split("_")[1]]
            coeffs[spin] = np.array(f[key])

        structure_str = np.string_(np.array(f["structure"])).decode()
        structure = Structure.from_str(structure_str, fmt="json")
        kpoints = np.array(f["kpoints"])
        if "gpoints" in f:
            gpoints = np.array(f["gpoints"])
        else:
            gpoints = None

    return coeffs, gpoints, kpoints, structure
