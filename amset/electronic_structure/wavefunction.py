import h5py

import numpy as np

from tqdm.auto import tqdm

try:
    from pawpyseed.core.wavefunction import Wavefunction
    from pawpyseed.core.momentum import MomentumMatrix
except ImportError:
    raise ImportError("Pawpyseed is required for extracting wavefunction coefficients\n"
                      "Follow the installation instrucations at "
                      "https://github.com/kylebystrom/pawpyseed")


from pymatgen import Spin, Structure
from amset.constants import numeric_types

str_to_spin = {"up": Spin.up, "down": Spin.down}
spin_to_int = {Spin.up: 0, Spin.down: 1}
int_to_spin = {0: Spin.up, 1: Spin.down}


def get_wavefunction(
    structure="CONTCAR",
    potcar="POTCAR",
    wavecar="WAVECAR",
    vasprun="vasprun.xml",
    directory=None,
):
    # Add symprec option
    if directory:
        wf = Wavefunction.from_directory(path=directory)
    else:
        wf = Wavefunction.from_files(
            struct=structure, wavecar=wavecar, cr=potcar, vr=vasprun
        )
    dwf = wf.desymmetrized_copy()
    return dwf


def get_wavefunction_coefficients(wavefunction, bs, iband=None, encut=600, pbar=True):
    mm = MomentumMatrix(wavefunction, encut=encut)
    if not iband:
        iband = {}

    coeffs = {}
    for spin_idx in range(wavefunction.nspin):
        spin = int_to_spin[spin_idx]
        spin_iband = iband.get(spin, None)

        coeffs[spin] = _get_spin_wavefunction_coefficients(
            mm, bs, spin, iband=spin_iband, pbar=pbar
        )
    return coeffs


def _get_spin_wavefunction_coefficients(mm, bs, spin, iband=None, pbar=True):
    ncoeffs = mm.momentum_grid.shape[0]
    nkpoints = mm.wf.kpts.shape[0]

    if iband is None:
        iband = np.arange(bs.bands[spin].shape[0], dtype=int)

    elif isinstance(iband, numeric_types):
        iband = [iband]

    coeff_shape = (len(iband), nkpoints)
    coeffs = np.zeros(coeff_shape + (ncoeffs,), dtype=complex)
    ns = spin_to_int[spin]
    state_idxs = list(np.ndindex(coeff_shape))

    if pbar:
        state_idxs = tqdm(state_idxs)

    for nb, nk in state_idxs:
        coeffs[nb, nk] = mm.get_reciprocal_fullfw(nb, nk, ns)
        coeffs[nb, nk] /= np.linalg.norm(coeffs[nb, nk])

    return coeffs


def dump_coefficients(coeffs, kpoints, structure, filename="coeffs.h5"):
    with h5py.File(filename, "w") as f:
        for spin, spin_coeffs in coeffs.items():
            name = "coefficients_{}".format(spin.name)
            f[name] = spin_coeffs

        f["structure"] = np.string_(structure.to_json())
        f["kpoints"] = kpoints


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

    return coeffs, kpoints, structure
