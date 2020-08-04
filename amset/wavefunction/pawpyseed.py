import itertools

import h5py
import numpy as np
from monty.dev import requires
from tqdm.auto import tqdm

from amset.constants import numeric_types
from pymatgen import Spin, Structure
from pymatgen.io.vasp import Potcar, Vasprun

try:
    import pawpyseed as pawpy
except ImportError:
    pawpy = None

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

str_to_spin = {"up": Spin.up, "down": Spin.down}
spin_to_int = {Spin.up: 0, Spin.down: 1}
int_to_spin = {0: Spin.up, 1: Spin.down}

pawpy_msg = (
    "Pawpyseed is required for extracting wavefunction coefficients\nFollow the"
    "installation instructions at https://github.com/kylebystrom/pawpyseed"
)


@requires(pawpy, pawpy_msg)
def get_wavefunction(
    potcar="POTCAR",
    wavecar="WAVECAR",
    vasprun="vasprun.xml",
    directory=None,
    symprec=None,
):
    from pawpyseed.core.wavefunction import Wavefunction, CoreRegion
    from pawpyseed.core import pawpyc

    if directory:
        wf = Wavefunction.from_directory(path=directory)
    else:
        if isinstance(vasprun, str):
            vasprun = Vasprun(vasprun)

        if isinstance(potcar, str):
            potcar = Potcar.from_file(potcar)

        ngx = vasprun.parameters["NGX"]
        ngy = vasprun.parameters["NGY"]
        ngz = vasprun.parameters["NGZ"]
        dim = np.array([ngx, ngy, ngz])
        symprec = vasprun.parameters["SYMPREC"]
        structure = vasprun.final_structure

        pwf = pawpyc.PWFPointer(wavecar, vasprun)
        core_region = CoreRegion(potcar)

        wf = Wavefunction(structure, pwf, core_region, dim, symprec, False)

    dwf = wf.desymmetrized_copy(time_reversal_symmetry=False, symprec=symprec)
    return dwf


@requires(pawpy, pawpy_msg)
def get_wavefunction_coefficients(wavefunction, bs, iband=None, encut=600, pbar=True):
    from pawpyseed.core.momentum import MomentumMatrix

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
    return coeffs, mm.momentum_grid


def _get_spin_wavefunction_coefficients(mm, bs, spin, iband=None, pbar=True):
    from amset.constants import output_width

    if iband is None:
        iband = np.arange(bs.bands[spin].shape[0], dtype=int)

    elif isinstance(iband, numeric_types):
        iband = [iband]

    ncoeffs = mm.momentum_grid.shape[0]
    nkpoints = mm.wf.kpts.shape[0]
    ns = spin_to_int[spin]
    coeffs = np.zeros((len(iband), nkpoints, ncoeffs), dtype=complex)

    state_idxs = list(itertools.product(enumerate(iband), range(nkpoints)))
    if pbar:
        state_idxs = tqdm(state_idxs, ncols=output_width)

    for (i, nb), nk in state_idxs:
        coeffs[i, nk] = mm.get_reciprocal_fullfw(nb, nk, ns)
        coeffs[i, nk] /= np.linalg.norm(coeffs[i, nk])

    return coeffs


@requires(pawpy, pawpy_msg)
def get_converged_encut(
    wavefunction, bs, iband=None, max_encut=500, n_samples=1000, std_tol=0.02
):
    from pawpyseed.core.momentum import MomentumMatrix

    mm = MomentumMatrix(wavefunction, encut=max_encut)

    sample_points = sample_random_kpoints(wavefunction, bs, n_samples, iband=iband)
    origin = sample_points[0]
    sample_points = sample_points[1:]

    true_overlaps = get_overlaps(mm, origin, sample_points)

    # filter points to only include these with reasonable overlaps
    mask = true_overlaps > 0.05
    true_overlaps = true_overlaps[mask]
    sample_points = sample_points[mask]

    for encut in np.arange(100, max_encut, 50):
        mm = MomentumMatrix(wavefunction, encut=encut)
        fake_overlaps = get_overlaps(mm, origin, sample_points)
        diff = (true_overlaps / fake_overlaps) - 1
        if diff.std() < std_tol:
            return encut

    return max_encut


def get_overlaps(mm, origin, points):
    coeffs = np.zeros((len(points), mm.momentum_grid.shape[0]), dtype=complex)

    origin = mm.get_reciprocal_fullfw(origin[1], origin[2], origin[0])
    origin /= np.linalg.norm(origin)

    for i, (s_idx, b_idx, k_idx) in enumerate(points):
        coeffs[i] = mm.get_reciprocal_fullfw(b_idx, k_idx, s_idx)
        coeffs[i] /= np.linalg.norm(coeffs[i])

    overlaps = np.abs((np.conj(origin[:, None]) * coeffs.T).sum(axis=0)) ** 2
    return overlaps


def sample_random_kpoints(wavefunction, bs, n_samples, iband=None):
    if not iband:
        iband = {}

    spin_idxs = []
    band_idxs = []
    kpoint_idxs = []

    for spin_idx in range(wavefunction.nspin):
        spin = int_to_spin[spin_idx]
        spin_iband = iband.get(spin, None)

        if spin_iband is None:
            spin_iband = np.arange(bs.bands[spin].shape[0], dtype=int)
        elif isinstance(spin_iband, numeric_types):
            spin_iband = [spin_iband]

        s_idxs = [spin_idx] * n_samples
        b_idxs = np.random.randint(min(spin_iband), max(spin_iband) + 1, n_samples)
        k_idxs = np.random.randint(0, wavefunction.kpts.shape[0], n_samples)

        spin_idxs.append(s_idxs)
        band_idxs.append(b_idxs)
        kpoint_idxs.append(k_idxs)

    spin_idxs = np.concatenate(spin_idxs)
    band_idxs = np.concatenate(band_idxs)
    kpoint_idxs = np.concatenate(kpoint_idxs)
    return np.stack([spin_idxs, band_idxs, kpoint_idxs], axis=1)


def dump_coefficients(coeffs, kpoints, structure, filename="coeffs.h5"):
    with h5py.File(filename, "w") as f:
        for spin, spin_coeffs in coeffs.items():
            name = "coefficients_{}".format(spin.name)
            dset = f.create_dataset(
                name, spin_coeffs.shape, compression="gzip", dtype=np.complex
            )
            dset[...] = spin_coeffs

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
