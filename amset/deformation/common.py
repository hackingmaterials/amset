import logging
import time

import numpy as np

from amset.electronic_structure.symmetry import similarity_transformation
from amset.log import log_time_taken
from amset.util import get_progress_bar

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

_voigt_idxs = ([0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1])
_fmt_str = (
    "[{0:{space}.{dp}f} {1:{space}.{dp}f} {2:{space}.{dp}f} {3:{space}.{dp}f} "
    "{4:{space}.{dp}f} {5:{space}.{dp}f}]"
)

logger = logging.getLogger(__name__)


def get_formatted_tensors(tensors):
    min_n = np.inf
    for tensor in tensors:
        tensor = tensor.round(5)
        frac_t = tensor - np.round(tensor)
        min_n = min(np.min(np.abs(frac_t[np.nonzero(frac_t)])), min_n)

    # get the number of decimal places we will need
    dp = np.ceil(np.log10(1 / min_n)).astype(int)
    space = dp + 3

    formatted_tensors = []
    for tensor in tensors:
        voigt = np.array(tensor)[_voigt_idxs].round(5)
        voigt[voigt == -0.0] = 0
        formatted_tensors.append(_fmt_str.format(*voigt, space=space, dp=dp))

    return formatted_tensors


def desymmetrize_deformation_potentials(
    deformation_potentials, structure, rotations, op_mapping, kp_mapping, pbar=True
):
    logger.info("Desymmetrizing deformation potentials")
    t0 = time.perf_counter()

    rlat = structure.lattice.reciprocal_lattice.matrix
    sims = np.array([similarity_transformation(rlat, r) for r in rotations])
    inv_sims = np.array([np.linalg.inv(s) for s in sims])

    sims = sims[op_mapping]
    inv_sims = inv_sims[op_mapping]

    all_deformation_potentials = {}
    for spin, spin_deformation_potentials in deformation_potentials.items():
        all_deformation_potentials[spin] = np.zeros(
            (len(spin_deformation_potentials), len(sims), 3, 3)
        )

        state_idxs = list(np.ndindex((len(spin_deformation_potentials), len(sims))))
        if pbar:
            state_idxs = get_progress_bar(state_idxs, desc="progress")

        for b_idx, k_idx in state_idxs:
            map_idx = kp_mapping[k_idx]

            sim = sims[k_idx]
            inv_sim = inv_sims[k_idx]

            inner = np.dot(sim, spin_deformation_potentials[b_idx, map_idx])
            rot_deform = np.abs(np.dot(inner, inv_sim))

            # inner = np.dot(spin_deformation_potentials[b_idx, map_idx], inv_sim)
            # rot_deform = np.abs(np.dot(sim, inner))
            # inner = np.dot(spin_deformation_potentials[b_idx, map_idx], sim)
            # rot_deform = np.abs(np.dot(inv_sim, inner))

            all_deformation_potentials[spin][b_idx, k_idx] = rot_deform

    log_time_taken(t0)
    return all_deformation_potentials
