import warnings
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from amset.constants import int_to_spin, numeric_types, spin_to_int
from amset.wavefunction.common import (
    get_gpoint_indices,
    get_gpoints,
    get_min_gpoints,
    sample_random_kpoints,
)

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"


def get_wavefunction(wavecar="WAVECAR", directory=None):
    from pymatgen.io.vasp import Wavecar

    if directory:
        wavecar = Path(directory) / wavecar

    return Wavecar(wavecar)


def get_wavefunction_coefficients(wavefunction, iband=None, encut=500, pbar=True):
    if encut > wavefunction.encut:
        warnings.warn(
            "Selected encut greater than encut of calculation. "
            "Many coefficients will likely be zero."
        )

    valid_gpoints = get_gpoints(wavefunction.b, wavefunction._nbmax, encut)
    min_gpoint, num_gpoint = get_min_gpoints(wavefunction._nbmax)
    valid_indices = get_gpoint_indices(valid_gpoints, min_gpoint, num_gpoint)

    if not iband:
        iband = {}

    coeffs = {}
    for spin_idx in range(wavefunction.spin):
        spin = int_to_spin[spin_idx]
        spin_iband = iband.get(spin, None)

        coeffs[spin] = _get_spin_wavefunction_coefficients(
            wavefunction,
            spin,
            valid_indices,
            min_gpoint,
            num_gpoint,
            iband=spin_iband,
            pbar=pbar,
        )
    return coeffs, valid_gpoints


def _get_spin_wavefunction_coefficients(
    wavefunction, spin, valid_indices, min_gpoint, num_gpoint, iband=None, pbar=True
):
    from amset.constants import output_width

    set_valid_indices = set(valid_indices)

    # mapping from each valid gpoint index to final gpoint array
    indices_map = np.full(np.max(valid_indices) + 1, -1, dtype=int)
    indices_map[valid_indices] = np.arange(len(valid_indices))

    ispin = spin_to_int[spin]
    if wavefunction.spin == 1:
        original_coeffs = wavefunction.coeffs
    else:
        original_coeffs = wavefunction.coeffs[ispin]

    if iband is None:
        # take all bands
        iband = np.arange(len(original_coeffs[0]), dtype=int)

    elif isinstance(iband, numeric_types):
        iband = [iband]

    ncoeffs = len(valid_indices)
    nkpoints = len(wavefunction.kpoints)
    coeffs = np.zeros((len(iband), nkpoints, ncoeffs), dtype=complex)
    state_idxs = list(range(nkpoints))
    if pbar:
        state_idxs = tqdm(state_idxs, ncols=output_width)

    for nk in state_idxs:
        gpoints_to_keep, indices_to_keep = _get_valid_state_coefficients(
            wavefunction, nk, min_gpoint, num_gpoint, set_valid_indices
        )
        coeff_indices = indices_map[indices_to_keep]

        if np.any(coeff_indices == -1):
            raise RuntimeError("Something went wrong mapping coefficients")

        for i, nb in enumerate(iband):
            coeffs[i, nk, coeff_indices] = original_coeffs[nk][nb][gpoints_to_keep]

    coeffs /= np.linalg.norm(coeffs, axis=2)[..., None]

    return coeffs


def _get_valid_state_coefficients(
    wavefunction, nkpoint, min_gpoint, num_gpoint, set_valid_indices
):
    """
    Returns the index of the gpoints in the Gpoints array and the correspoding
    'index' identifier.
    """
    gpoints = wavefunction.Gpoints[nkpoint]
    indices = get_gpoint_indices(gpoints, min_gpoint, num_gpoint)

    # map the indices to the original gpoint array
    indices_map = np.full(np.max(indices) + 1, -1, dtype=int)
    indices_map[indices] = np.arange(len(indices))

    # find which indices are in the list of valid indices
    set_indices = set(indices)
    indices_to_keep = set_valid_indices.intersection(set_indices)
    indices_to_keep = np.array(list(indices_to_keep))

    to_keep = indices_map[indices_to_keep]
    return to_keep, indices_to_keep


def get_converged_encut(
    wavefunction, iband=None, max_encut=500, n_samples=1000, std_tol=0.002
):
    nspins = wavefunction.spin
    nkpoints = len(wavefunction.kpoints)

    if iband is None:
        # this is a little different to usual iband
        iband = {}
        for ispin in range(wavefunction.spin):
            spin = int_to_spin[ispin]
            if wavefunction.spin == 1:
                iband[spin] = len(wavefunction.coeffs[0])
            else:
                iband[spin] = len(wavefunction.coeffs[ispin][0])

    sample_points = sample_random_kpoints(nspins, nkpoints, iband, n_samples)
    origin = sample_points[0]
    sample_points = sample_points[1:]

    coeffs, _ = get_wavefunction_coefficients(wavefunction, encut=max_encut, pbar=False)
    true_overlaps = get_overlaps(coeffs, origin, sample_points)

    # filter points to only include these with reasonable overlaps
    mask = true_overlaps > 0.05
    true_overlaps = true_overlaps[mask]
    sample_points = sample_points[mask]

    for encut in np.arange(200, max_encut, 50):
        coeffs, _ = get_wavefunction_coefficients(wavefunction, encut=encut, pbar=False)
        fake_overlaps = get_overlaps(coeffs, origin, sample_points)

        diff = (true_overlaps / fake_overlaps) - 1
        if diff.std() < std_tol:
            return encut

    return max_encut


def get_overlaps(coeffs, origin, points):
    ncoeffs = list(coeffs.values())[0].shape[2]
    select_coeffs = np.zeros((len(points), ncoeffs), dtype=complex)

    origin_spin = int_to_spin[origin[0]]
    origin = coeffs[origin_spin][origin[1], origin[2]]

    for i, (s_idx, b_idx, k_idx) in enumerate(points):
        spin = int_to_spin[s_idx]
        select_coeffs[i] = coeffs[spin][b_idx, k_idx]

    overlaps = np.abs((np.conj(origin[:, None]) * select_coeffs.T).sum(axis=0)) ** 2
    return overlaps
