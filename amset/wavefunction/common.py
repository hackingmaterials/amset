import logging
import time
from typing import Dict, List, Union

import numpy as np
from pymatgen import Spin

from amset.constants import int_to_spin, numeric_types
from amset.electronic_structure.symmetry import (
    rotation_matrix_to_cartesian,
    rotation_matrix_to_su2,
)
from amset.log import log_time_taken
from amset.util import get_progress_bar

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

logger = logging.getLogger(__name__)


def sample_random_kpoints(
    nspins: int,
    ikpoints: List[int],
    nband: Dict[Spin, Union[List[int], int]],
    n_samples: int,
):
    spin_idxs = []
    band_idxs = []
    kpoint_idxs = []

    for spin_idx in range(nspins):
        spin = int_to_spin[spin_idx]

        spin_iband = nband[spin]
        if isinstance(spin_iband, numeric_types):
            spin_iband = np.arange(spin_iband, dtype=int)

        s_idxs = [spin_idx] * n_samples
        b_idxs = np.random.randint(min(spin_iband), max(spin_iband) + 1, n_samples)
        k_idxs = np.random.choice(ikpoints, n_samples)

        spin_idxs.append(s_idxs)
        band_idxs.append(b_idxs)
        kpoint_idxs.append(k_idxs)

    spin_idxs = np.concatenate(spin_idxs)
    band_idxs = np.concatenate(band_idxs)
    kpoint_idxs = np.concatenate(kpoint_idxs)
    return np.stack([spin_idxs, band_idxs, kpoint_idxs], axis=1)


def desymmetrize_coefficients(
    coeffs,
    gpoints,
    kpoints,
    structure,
    rotations,
    translations,
    is_tr,
    op_mapping,
    kp_mapping,
    pbar=True,
):
    logger.info("Desymmetrizing wavefunction coefficients")
    t0 = time.perf_counter()

    ncl = is_ncl(coeffs)
    rots = rotations[op_mapping]
    taus = translations[op_mapping]
    trs = is_tr[op_mapping]

    su2s = None
    if ncl:
        # get cartesian rotation matrix
        r_cart = [rotation_matrix_to_cartesian(r, structure.lattice) for r in rotations]

        # calculate SU(2)
        su2_no_dagger = np.array([rotation_matrix_to_su2(r) for r in r_cart])

        # calculate SU(2)^{dagger}
        su2 = np.conjugate(su2_no_dagger).transpose((0, 2, 1))
        su2s = su2[op_mapping]

    g_mesh = (np.abs(gpoints).max(axis=0) + 3) * 2
    g1, g2, g3 = (gpoints + g_mesh / 2).astype(int).T  # indices of g-points to keep

    all_rot_coeffs = {}
    for spin, spin_coeffs in coeffs.items():
        coeff_shape = (len(spin_coeffs), len(rots)) + tuple(g_mesh)
        if ncl:
            coeff_shape += (2,)
        rot_coeffs = np.zeros(coeff_shape, dtype=complex)

        state_idxs = list(range(len(rots)))
        if pbar:
            state_idxs = get_progress_bar(state_idxs, desc="progress")

        for k_idx in state_idxs:
            map_idx = kp_mapping[k_idx]

            rot = rots[k_idx]
            tau = taus[k_idx]
            tr = trs[k_idx]
            kpoint = kpoints[map_idx]

            rot_kpoint = np.dot(rot, kpoint)
            kdiff = np.around(rot_kpoint)
            rot_kpoint -= kdiff

            edges = np.around(rot_kpoint, 5) == -0.5
            rot_kpoint += edges
            kdiff -= edges

            rot_gpoints = np.dot(rot, gpoints.T).T
            rot_gpoints = np.around(rot_gpoints).astype(int)
            rot_gpoints += kdiff.astype(int)

            if tr:
                tau = -tau

            factor = np.exp(-1j * 2 * np.pi * np.dot(rot_gpoints + rot_kpoint, tau))
            rg1, rg2, rg3 = (rot_gpoints + g_mesh / 2).astype(int).T

            if ncl:
                # perform rotation in spin space
                su2 = su2s[k_idx]
                rc = np.zeros_like(spin_coeffs[:, map_idx])
                rc[:, :, 0] = (
                    su2[0, 0] * spin_coeffs[:, map_idx, :, 0]
                    + su2[0, 1] * spin_coeffs[:, map_idx, :, 1]
                )
                rc[:, :, 1] = (
                    su2[1, 0] * spin_coeffs[:, map_idx, :, 0]
                    + su2[1, 1] * spin_coeffs[:, map_idx, :, 1]
                )
                rot_coeffs[:, k_idx, rg1, rg2, rg3] = factor[None, :, None] * rc
            else:
                rot_coeffs[:, k_idx, rg1, rg2, rg3] = spin_coeffs[:, map_idx] * factor

            if tr and not ncl:
                rot_coeffs[:, k_idx] = np.conjugate(rot_coeffs[:, k_idx])

        all_rot_coeffs[spin] = rot_coeffs[:, :, g1, g2, g3]

    log_time_taken(t0)
    return all_rot_coeffs


def get_gpoints(reciprocal_lattice, nbmax, encut, kpoint=(0, 0, 0)):
    all_g = np.array(list(np.ndindex(tuple(2 * nbmax[::-1] + 1))))
    all_g = all_g[:, [2, 1, 0]]  # swap columns

    all_g[all_g[:, 2] > nbmax[2], 2] -= 2 * nbmax[2] + 1
    all_g[all_g[:, 1] > nbmax[1], 1] -= 2 * nbmax[1] + 1
    all_g[all_g[:, 0] > nbmax[0], 0] -= 2 * nbmax[0] + 1

    cart_g = np.dot(all_g + kpoint, reciprocal_lattice)
    norm_g = np.linalg.norm(cart_g, axis=1)
    ener_g = norm_g ** 2 / 0.262465831

    return all_g[ener_g <= encut]


def get_min_gpoints(nbmax):
    min_gpoint = -nbmax
    num_gpoint = nbmax * 2 + 1
    return min_gpoint, num_gpoint


def get_gpoint_indices(gpoints, min_gpoint, num_gpoint):
    shifted_g = gpoints - min_gpoint
    nyz = num_gpoint[1] * num_gpoint[2]
    nz = num_gpoint[2]
    indices = shifted_g[:, 0] * nyz + shifted_g[:, 1] * nz + shifted_g[:, 2]
    return indices.astype(int)


def is_ncl(coefficients):
    return len(list(coefficients.values())[0].shape) == 4


def get_overlap(origin, final):
    if len(origin.shape) == 2:
        # ncl
        return (
            np.abs(
                np.dot(np.conj(origin[:, 0]), final[:, :, 0].T)
                + np.dot(np.conj(origin[:, 1]), final[:, :, 1].T)
            )
            ** 2
        )
    else:
        return np.abs(np.dot(np.conj(origin), final.T)) ** 2
