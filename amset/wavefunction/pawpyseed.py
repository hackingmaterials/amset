import itertools

import numpy as np
from monty.dev import requires
from tqdm.auto import tqdm

from amset.constants import numeric_types
from amset.wavefunction.common import int_to_spin, sample_random_kpoints, spin_to_int
from pymatgen.io.vasp import Potcar, Vasprun

try:
    import pawpyseed as pawpy
except ImportError:
    pawpy = None

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

pawpy_msg = (
    "Pawpyseed is required for extracting wavefunction coefficients\nFollow the"
    "installation instructions at https://github.com/kylebystrom/pawpyseed"
)


@requires(pawpy, pawpy_msg)
def get_wavefunction(
    potcar="POTCAR", wavecar="WAVECAR", vasprun="vasprun.xml", directory=None
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

    return wf


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

    nspins = wavefunction.nspin
    nkpoints = wavefunction.kpts.shape[0]
    if iband is None:
        iband = {s: len(bs.bands[s]) for s in bs.spins}

    sample_points = sample_random_kpoints(nspins, nkpoints, iband, n_samples)
    origin = sample_points[0]
    sample_points = sample_points[1:]

    mm = MomentumMatrix(wavefunction, encut=max_encut)
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
