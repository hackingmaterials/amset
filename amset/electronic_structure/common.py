from typing import Dict, Tuple

import numpy as np
from pymatgen import Spin, Structure
from pymatgen.electronic_structure.bandstructure import BandStructure

from amset.constants import angstrom_to_bohr, bohr_to_angstrom, defaults

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

from pymatgen.io.vasp import Vasprun


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


def get_vbm_energy(energies: Dict[Spin, np.ndarray], vb_idx: Dict[Spin, int]) -> float:
    """Get the valence band maximum energy from energies and valence band indices.

    Args:
        energies: The energies as a dict of `{spin: np.ndarray}`, where the array has
            the shape `(nbands, nkpoints)`.
        vb_idx: The valence band indices for each spin channel as a dictionary of
            `{spin: valence band index}`. If there are no valence band indices for a
            specific spin channel, the index will be `-1`. Note, a valence band index
            equal to the number of bands implies there are no conduction bands for
            that spin channel.

    Returns:
        The valence band maximum energy.
    """
    e_vbm = -np.inf
    for spin, spin_energies in energies.items():
        spin_cbm_idx = vb_idx[spin] + 1
        if spin_cbm_idx > 0:
            # if spin_cbm_idx <= 0 there are no valence bands for this spin channel
            e_vbm = max(e_vbm, np.max(spin_energies[:spin_cbm_idx]))
    return e_vbm


def get_cbm_energy(energies: Dict[Spin, np.ndarray], vb_idx: Dict[Spin, int]) -> float:
    """Get the conduction band minimum energy from energies and valence band indices.

    Args:
        energies: The energies as a dict of `{spin: np.ndarray}`, where the array has
            the shape `(nbands, nkpoints)`.
        vb_idx: The valence band indices for each spin channel as a dictionary of
            `{spin: valence band index}`. If there are no valence band indices for a
            specific spin channel, the index will be `-1`. Note, a valence band index
            equal to the number of bands implies there are no conduction bands for
            that spin channel.

    Returns:
        The conduction band minimum energy.
    """
    e_cbm = np.inf
    for spin, spin_energies in energies.items():
        spin_cbm_idx = vb_idx[spin] + 1
        if spin_cbm_idx < spin_energies.shape[0]:
            # if spin_cbm_idx >= nbands there are no conduction bands for this spin
            e_cbm = min(e_cbm, np.min(spin_energies[spin_cbm_idx:]))
    return e_cbm


def get_efermi(energies: Dict[Spin, np.ndarray], vb_idx: Dict[Spin, int]) -> float:
    """Get the fermi level energy from energies and valence band index indices.

    Args:
        energies: The energies as a dict of `{spin: np.ndarray}`, where the array has
            the shape `(nbands, nkpoints)`.
        vb_idx: The valence band indices for each spin channel as a dictionary of
            `{spin: valence band index}`. If there are no valence band indices for a
            specific spin channel, the index will be `-1`. Note, a valence band index
            equal to the number of bands implies there are no conduction bands for
            that spin channel.

    Returns:
        The Fermi level, set to halfway between the VBM and CBM.
    """
    e_vbm = get_vbm_energy(energies, vb_idx)
    e_cbm = get_cbm_energy(energies, vb_idx)
    return (e_vbm + e_cbm) / 2


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
        site_properties=structure.site_properties,
    )


def get_angstrom_structure(structure):
    return Structure(
        structure.lattice.matrix * bohr_to_angstrom,
        structure.species,
        structure.frac_coords,
        site_properties=structure.site_properties,
    )


def get_band_structure(
    vasprun: Vasprun, zero_weighted: str = defaults["zero_weighted_kpoints"]
) -> BandStructure:
    """
    Get a band structure from a Vasprun object.

    This can ensure that if the calculation contains zero-weighted k-points then the
    weighted k-points will be discarded (helps with hybrid calculations).

    Also ensures that the Fermi level is set correctly.

    Args:
        vasprun: A vasprun object.
        zero_weighted: How to handle zero-weighted k-points if they are present in the
            calculation. Options are:
            - "keep": Keep zero-weighted k-points in the band structure.
            - "drop": Drop zero-weighted k-points, keeping only the weighted k-points.
            - "prefer": Drop weighted-kpoints if zero-weighted k-points are present
              in the calculation (useful for cheap hybrid calculations).

    Returns:
        A band structure.
    """
    # first check if Fermi level crosses a band
    k_idx = get_zero_weighted_kpoint_indices(vasprun, mode=zero_weighted)
    kpoints = np.array(vasprun.actual_kpoints)[k_idx]

    projections = {}
    eigenvalues = {}
    for spin, spin_eigenvalues in vasprun.eigenvalues.items():
        # discard weight and set shape nbands, nkpoints
        eigenvalues[spin] = spin_eigenvalues[k_idx, :, 0].transpose(1, 0)

        if vasprun.projected_eigenvalues:
            # is nkpoints, nbands, nion, norb; we need nbands, nkpoints, norb, nion
            spin_projections = vasprun.projected_eigenvalues[spin]
            projections[spin] = spin_projections[k_idx].transpose(1, 0, 3, 2)

    # finding the Fermi level is quite painful, as VASP can sometimes put it slightly
    # inside a band
    fermi_crosses_band = False
    for spin_eigenvalues in eigenvalues.values():
        eigs_below = np.any(spin_eigenvalues < vasprun.efermi, axis=1)
        eigs_above = np.any(spin_eigenvalues > vasprun.efermi, axis=1)
        if np.any(eigs_above & eigs_below):
            fermi_crosses_band = True

    # if the Fermi level crosses a band, the eigenvalue band properties is a more
    # reliable way to check whether this is a real effect
    bandgap, cbm, vbm, _ = vasprun.eigenvalue_band_properties

    if not fermi_crosses_band:
        # safe to use VASP fermi level
        efermi = vasprun.efermi
    elif fermi_crosses_band and bandgap == 0:
        # it is actually a metal
        efermi = vasprun.efermi
    else:
        # Set Fermi level half way between valence and conduction bands
        efermi = (cbm + vbm) / 2

    return BandStructure(
        kpoints,
        eigenvalues,
        vasprun.final_structure.lattice.reciprocal_lattice,
        efermi,
        structure=vasprun.final_structure,
        projections=projections,
    )


def get_zero_weighted_kpoint_indices(vasprun: Vasprun, mode: str) -> np.ndarray:
    """
    Get zero weighted k-point k-point indices from a vasprun.

    If the calculation doesn't contain zero-weighted k-points, then the indices of
    all the k-points will be returned. Alternatively, if the calculation contains
    a mix of weighted and zero-weighted k-points, then only the indices of the
    zero-weighted k-points will be returned.

    Args:
        vasprun:  A vasprun object.
        mode: How to handle zero-weighted k-points if they are present in the
            calculation. Options are:
            - "keep": Keep zero-weighted k-points in the band structure.
            - "drop": Drop zero-weighted k-points, keeping only the weighted k-points.
            - "prefer": Drop weighted-kpoints if zero-weighted k-points are present
              in the calculation (useful for cheap hybrid calculations).

    Returns:
        The indices of the valid k-points.
    """
    weights = np.array(vasprun.actual_kpoints_weights)
    is_zero_weight = weights == 0

    if mode not in ("prefer", "drop", "keep"):
        raise ValueError(f"Unrecognised zero-weighted k-point mode: {mode}")

    if mode == "prefer" and np.any(is_zero_weight):
        return np.where(is_zero_weight)[0]
    elif mode == "drop" and np.any(~is_zero_weight):
        return np.where(~is_zero_weight)[0]
    else:
        return np.arange(len(weights))
