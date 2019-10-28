import logging
from copy import deepcopy

import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

from amset.data import AmsetData
from amset.constants import e, sqrt2
from amset.misc.log import log_list
from amset.scattering.elastic import calculate_inverse_screening_length_sq

logger = logging.getLogger(__name__)


class AbstractBasicScattering(ABC):

    name: str
    required_properties: Tuple[str]

    def __init__(self, materials_properties: Dict[str, Any], amset_data: AmsetData):
        self.properties = {p: materials_properties[p] for p in self.required_properties}
        self.doping = amset_data.doping
        self.temperatures = amset_data.temperatures
        self.nbands = {s: len(amset_data.energies[s]) for s in amset_data.spins}
        self.spins = amset_data.spins

    @property
    @abstractmethod
    def rates(self):
        pass


class BrooksHerringScattering(AbstractBasicScattering):

    name = "IMP(BH)"
    required_properties = ("acceptor_charge", "donor_charge", "static_dielectric")

    def __init__(self, materials_properties: Dict[str, Any], amset_data: AmsetData):
        # this is similar to the full IMP scattering, except for the prefactor
        # which follows the simplified BH formula
        super().__init__(materials_properties, amset_data)
        logger.debug("Initializing IMP(BK) scattering")

        inverse_screening_length_sq = calculate_inverse_screening_length_sq(
            amset_data, self.properties["static_dielectric"]
        )
        impurity_concentration = np.zeros(amset_data.fermi_levels.shape)

        imp_info = []
        for n, t in np.ndindex(inverse_screening_length_sq.shape):
            n_conc = np.abs(amset_data.electron_conc[n, t])
            p_conc = np.abs(amset_data.hole_conc[n, t])

            impurity_concentration[n, t] = (
                n_conc * self.properties["donor_charge"] ** 2
                + p_conc * self.properties["acceptor_charge"] ** 2
            )
            imp_info.append(
                "{:.2g} cm⁻³ & {} K: β² = {:.4g} nm⁻², Nᵢᵢ = {:.4g}".format(
                    amset_data.doping[n],
                    amset_data.temperatures[t],
                    inverse_screening_length_sq[n, t],
                    impurity_concentration[n, t],
                )
            )

        logger.debug(
            "Inverse screening length (β) and impurity concentration " "(Nᵢᵢ):"
        )
        log_list(imp_info, level=logging.DEBUG)

        prefactor = (
            e ** 4
            * impurity_concentration
            / (16 * sqrt2 * np.pi * self.properties["static_dielectric"] ** 2)
        )

        normalized_energies = get_normalized_energies(amset_data)
        dos_effective_masses = get_dos_effective_masses(amset_data)

        self._rates = {}
        for spin in self.spins:
            num = 8 * dos_effective_masses[spin] * normalized_energies[spin]
            num = np.tile(num, (len(self.doping), len(self.temperatures), 1))

            # b is from the classic Brooks–Herring formula, it has the shape
            # (ndops, ntemps, nbands, nkpoints)
            b = num / inverse_screening_length_sq[:, :, None, None]
            bp1 = 1 + b
            b_factor = np.log(bp1) - b / bp1
            k_factor = np.power(normalized_energies[spin], -3 / 2) / np.sqrt(
                dos_effective_masses[spin]
            )
            self._rates[spin] = prefactor * k_factor * b_factor

    @property
    def rates(self):
        # need to return rates with shape (nspins, ndops, ntemps, nbands, nkpoints)
        return self._rates


def get_normalized_energies(amset_data: AmsetData):
    # normalize the energies, for metals the energies are given as abs(E-Ef) where
    # Ef is the intrinsic Fermi level; for semiconductors, the energies are given as
    # VBM - E for valence band states and E - CBM for conduction band states
    norm_e = deepcopy(amset_data.energies)
    spins = amset_data.spins
    if amset_data.is_metal:
        for spin in spins:
            norm_e[spin][:] = np.abs(norm_e[spin] - amset_data._efermi)

    else:
        vb_idx = amset_data.vb_idx
        vbm_e = np.max([amset_data.energies[s][: vb_idx[s]] for s in spins])
        cbm_e = np.max([amset_data.energies[s][vb_idx[s] + 1] for s in spins])

        for spin in spins:
            norm_e[: vb_idx[spin]][:] = vbm_e - norm_e[: vb_idx[spin]]
            norm_e[vb_idx[spin] + 1:][:] = norm_e[vb_idx[spin] + 1:] - cbm_e

    return norm_e


def get_dos_effective_masses(amset_data: AmsetData):
    dos_effective_masses = {}
    for spin in amset_data.spins:
        masses_abs = np.abs(amset_data.effective_masses[spin])
        masses_eig = np.linalg.eigh(masses_abs)[0]
        dos_effective_masses[spin] = np.power(np.product(masses_eig, axis=2), 1 / 3)
    return dos_effective_masses
