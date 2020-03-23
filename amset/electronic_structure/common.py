from typing import Tuple

import numpy as np

from amset.constants import angstrom_to_bohr, bohr_to_angstrom
from pymatgen import Structure
from pymatgen.electronic_structure.bandstructure import BandStructure

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"


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


def get_velocities_from_outer_product(
    velocities_product, return_norm=False, symmetry_information=None
):
    # not recommended to give symmetry information AND return_norm=False
    # probably raise a warning
    velocities = {}

    for spin, spin_velocities_product in velocities_product.items():

        if symmetry_information:
            ir_kpoints_idx = symmetry_information["ir_kpoints_idx"]
            spin_velocities_product = spin_velocities_product[..., ir_kpoints_idx]

        v = np.sqrt(np.diagonal(spin_velocities_product, axis1=1, axis2=2))

        if return_norm:
            v = np.linalg.norm(v, axis=2)

        if symmetry_information:
            ir_to_full_idx = symmetry_information["ir_to_full_kpoint_mapping"]
            v = spin_velocities_product[..., ir_to_full_idx]

        velocities[spin] = v

    return velocities


def get_atomic_structure(structure):
    return Structure(
        structure.lattice.matrix * angstrom_to_bohr,
        structure.species,
        structure.frac_coords,
    )


def get_angstrom_structure(structure):
    return Structure(
        structure.lattice.matrix * bohr_to_angstrom,
        structure.species,
        structure.frac_coords,
    )
