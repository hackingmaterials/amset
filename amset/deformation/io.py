from pathlib import Path

import h5py
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.vasp.outputs import Outcar, Vasprun

from amset.constants import defaults, str_to_spin
from amset.electronic_structure.common import get_band_structure

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"


def write_deformation_potentials(
    deformation_potentials, kpoints, structure, filename="deformation.h5"
):
    with h5py.File(filename, "w") as f:
        for spin, spin_deform in deformation_potentials.items():
            name = f"deformation_potentials_{spin.name}"
            f.create_dataset(name, data=spin_deform, compression="gzip")
        f["structure"] = np.string_(structure.to_json())
        f["kpoints"] = kpoints
    return filename


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


def write_deformed_poscars(deformed_structures, directory=None):
    if not directory:
        directory = "."
    directory = Path(directory)
    n_deformations = len(deformed_structures)
    n_digits = int(np.floor(np.log10(n_deformations)) + 2)
    for i, deformed_structure in enumerate(deformed_structures):
        # pad with leading zeros so the files are sorted correctly
        filename = "POSCAR-{0:0{1}}".format(i + 1, n_digits)
        deform_poscar = Poscar(deformed_structure)
        deform_poscar.write_file(directory / filename, significant_figures=16)


def parse_calculation(folder, zero_weighted_kpoints=defaults["zero_weighted_kpoints"]):
    vr = Vasprun(get_gzipped_file("vasprun.xml", folder))
    out = Outcar(get_gzipped_file("OUTCAR", folder))
    bs = get_band_structure(vr, zero_weighted=zero_weighted_kpoints)
    reference_level = get_reference_energy(bs, out)
    return {"reference": reference_level, "bandstructure": bs}


def get_gzipped_file(filename, folder):
    folder = Path(folder)
    gz_filename = filename + ".gz"
    if (folder / filename).exists():
        return folder / filename
    elif (folder / gz_filename).exists():
        return folder / gz_filename
    else:
        raise FileNotFoundError(f"Could not find {filename} file in {folder}")


def get_reference_energy(bandstructure, outcar):
    if bandstructure.is_metal():
        return bandstructure.efermi
    else:
        # read the average potential at atomic cores from the OUTCAR; note: if
        # ICORELEVEL = 1, these will be not be written and pot will be an empty list
        pot = outcar.read_avg_core_poten()

        if len(pot) > 0:
            return np.mean(pot[0])

        # read the core level eigenvalues from the OUTCAR; note: if ICORELEVEL
        # is not set these will be not be written and eigen will be a list
        # of empty dictionaries
        eigen = outcar.read_core_state_eigen()
        ref = [x["1s"][0] for x in eigen if "1s" in x]
        if len(ref) > 0:
            return np.mean(ref)
        else:
            raise ValueError(
                "OUTCAR does not contain avg electrostatic potential or eigenvalues"
            )
