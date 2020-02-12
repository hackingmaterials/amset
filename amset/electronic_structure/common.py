import numpy as np
from typing import Tuple

from pymatgen.electronic_structure.bandstructure import BandStructure


def get_energy_cutoffs(
    energy_cutoff: float, band_structure: BandStructure
) -> Tuple[float, float]:
    if energy_cutoff and band_structure.is_metal():
        min_e = band_structure.efermi - energy_cutoff
        max_e = band_structure.efermi + energy_cutoff
    elif energy_cutoff:
        min_e = band_structure.get_vbm()["energy"] - energy_cutoff
        max_e = band_structure.get_cbm()["energy"] + energy_cutoff
    else:
        min_e = min(
            [band_structure.bands[spin].min() for spin in band_structure.bands.keys()]
        )
        max_e = max(
            [band_structure.bands[spin].max() for spin in band_structure.bands.keys()]
        )

    return min_e, max_e


def get_ibands(energy_cutoff: float, band_structure: BandStructure, return_idx=True):
    min_e, max_e = get_energy_cutoffs(energy_cutoff, band_structure)

    ibands = {}
    for spin, bands in band_structure.bands.items():
        ibands[spin] = np.any((bands > min_e) & (bands < max_e), axis=1)

        if return_idx:
            ibands[spin] = np.where(ibands[spin])[0]

    return ibands


def get_vb_idx(energy_cutoff: float, band_structure: BandStructure):
    if band_structure.is_metal():
        return None

    ibands = get_ibands(energy_cutoff, band_structure, return_idx=False)

    new_vb_idx = {}
    for spin, bands in band_structure.bands.items():
        spin_ibands = ibands[spin]

        # valence bands are all bands that contain energies less than efermi
        vbs = (bands < band_structure.efermi).any(axis=1)
        vb_idx = np.where(vbs)[0].max()

        # need to know the index of the valence band after discounting
        # bands during the interpolation. As ibands is just a list of
        # True/False, we can count the number of Trues included up to
        # and including the VBM to get the new number of valence bands
        new_vb_idx[spin] = sum(spin_ibands[: vb_idx + 1]) - 1

    return new_vb_idx
