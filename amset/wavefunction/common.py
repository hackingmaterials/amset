from typing import Dict, List, Union

import numpy as np
from tqdm.auto import tqdm

from amset.constants import numeric_types, output_width
from pymatgen import Spin

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

str_to_spin = {"up": Spin.up, "down": Spin.down}
spin_to_int = {Spin.up: 0, Spin.down: 1}
int_to_spin = {0: Spin.up, 1: Spin.down}


def sample_random_kpoints(
    nspins: int, nkpoints: int, nband: Dict[Spin, Union[List[int], int]], n_samples: int
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
        k_idxs = np.random.randint(0, nkpoints, n_samples)

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
    rotations,
    translations,
    is_tr,
    op_mapping,
    kp_mapping,
    pbar=True,
):
    ops = rotations[op_mapping]
    taus = translations[op_mapping]
    trs = is_tr[op_mapping]

    min_gpoint = gpoints.min(axis=0)
    num_gpoint = gpoints.max(axis=0) - min_gpoint + 1
    valid_indices = get_gpoint_indices(gpoints, min_gpoint, num_gpoint)
    indices_map = np.full(np.max(valid_indices) + 1, -1, dtype=int)
    indices_map[valid_indices] = np.arange(len(valid_indices))

    all_rot_coeffs = {}
    for spin, spin_coeffs in coeffs.items():
        all_rot_coeffs[spin] = np.zeros(
            (len(spin_coeffs), len(ops), len(gpoints)), dtype=np.complex
        )

        state_idxs = list(range(len(ops)))
        if pbar:
            state_idxs = tqdm(state_idxs, ncols=output_width)

        for k_idx in state_idxs:
            map_idx = kp_mapping[k_idx]

            op = ops[k_idx]
            tau = taus[k_idx]
            tr = trs[k_idx]
            kpoint = kpoints[map_idx]

            rot_kpoint = np.dot(op, kpoint)
            kdiff = np.around(rot_kpoint)
            rot_kpoint -= kdiff

            edges = np.around(rot_kpoint, 5) == -0.5
            rot_kpoint += edges
            kdiff -= edges

            rot_gpoints = np.dot(op, gpoints.T).T
            rot_gpoints = np.around(rot_gpoints).astype(int)
            rot_gpoints += kdiff.astype(int)
            rot_indices = get_gpoint_indices(rot_gpoints, min_gpoint, num_gpoint)

            # some g-points may now be outside encut limit so only keep the g-points
            # that are in the original g-point mesh
            rot_map = np.full(np.max(rot_indices) + 1, -1, dtype=int)
            rot_map[rot_indices] = np.arange(len(rot_indices))
            to_keep = np.array(list(set(valid_indices).intersection(set(rot_indices))))
            keep_indices = rot_map[to_keep]

            kgt_factor = np.dot(rot_kpoint + rot_gpoints, -tau)
            exp_factor = np.exp(1j * 2 * np.pi * kgt_factor)
            rot_coeffs = exp_factor[None, :] * spin_coeffs[:, map_idx]

            if tr:
                rot_coeffs = np.conjugate(rot_coeffs)

            order = indices_map[to_keep]
            all_rot_coeffs[spin][:, k_idx, order] = rot_coeffs[:, keep_indices]

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
