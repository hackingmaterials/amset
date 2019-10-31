import logging
from copy import deepcopy

import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

from BoltzTraP2 import units
from BoltzTraP2.units import BOLTZMANN, Second
from scipy.constants import physical_constants

from amset.data import AmsetData
from amset.constants import e, sqrt2, hbar, small_val, bohr_to_angstrom
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
            "Inverse screening length (β) and impurity concentration (Nᵢᵢ):"
        )
        log_list(imp_info, level=logging.DEBUG)

        inv_cm_to_bohr = 100 * physical_constants["Bohr radius"][0]
        inv_nm_to_bohr = 1e9 * physical_constants["Bohr radius"][0]

        impurity_concentration *= inv_cm_to_bohr ** 3

        # normalized energies has shape (nspins, ndoping, ntemps, nbands, nkpoints)
        normalized_energies = get_normalized_energies(amset_data, broaden=False)

        # dos effective masses has shape (nspins, nbands, nkpoints)
        dos_effective_masses = get_dos_effective_masses(amset_data)

        # screening has shape (ndoping, nbands, 1, 1)
        # screening = inverse_screening_length_sq[..., None, None] * inv_nm_to_bohr ** 2
        screening = inverse_screening_length_sq[..., None, None] * inv_nm_to_bohr ** 2

        prefactor = (
            np.pi
            * impurity_concentration
            / (sqrt2 * self.properties["static_dielectric"] ** 2)
        )

        self._rates = {}
        for spin in self.spins:
            masses = np.tile(dos_effective_masses[spin],
                             (len(self.doping), len(self.temperatures), 1, 1))
            energies = normalized_energies[spin]

            # b is from the classic Brooks–Herring formula, it has the shape
            # (ndops, ntemps, nbands, nkpoints)
            b = (8 * masses * energies) / screening
            b_factor = np.log(b + 1) - b / (b + 1)
            k_factor = np.power(energies, -3 / 2) / np.sqrt(masses)

            self._rates[spin] = (
                    prefactor[:, :, None, None] * b_factor * Second * k_factor
            )

            from pymatgen import Spin
            print(self._rates[Spin.up][0, 0, 4])
            print(b.min())
            print(b.max())

    @property
    def rates(self):
        # need to return rates with shape (nspins, ndops, ntemps, nbands, nkpoints)
        return self._rates


def get_normalized_energies(amset_data: AmsetData, broaden=False):
    # normalize the energies; energies returned as abs(E-Ef) (+ kBT if broaden is True)
    spins = amset_data.spins
    fermi_shape = amset_data.fermi_levels.shape
    energies = {spin: np.empty(fermi_shape + amset_data.energies[spin].shape) for spin in spins}

    if amset_data.is_metal:
        for spin in spins:
            spin_energies = deepcopy(amset_data.energies[spin])
            for n, t in np.ndindex(fermi_shape):
                broadening = BOLTZMANN * amset_data.temperatures[t] if broaden else 0
                energies[spin][n, t] = np.abs(spin_energies - amset_data._efermi) + broadening
    else:
        vb_idx = amset_data.vb_idx
        vbm_e = np.max([amset_data.energies[s][: vb_idx[s] + 1] for s in spins])
        cbm_e = np.min([amset_data.energies[s][vb_idx[s] + 1:] for s in spins])
        for spin in spins:
            spin_energies = deepcopy(amset_data.energies[spin])
            spin_energies[:vb_idx[spin] + 1] = vbm_e - spin_energies[:vb_idx[spin] + 1]
            spin_energies[vb_idx[spin] + 1:] = spin_energies[vb_idx[spin] + 1:] - cbm_e

            for n, t in np.ndindex(fermi_shape):
                broadening = BOLTZMANN * amset_data.temperatures[t] if broaden else small_val
                energies[spin][n, t] = spin_energies + broadening

    # for spin in spins:
    #     spin_energies = amset_data.energies[spin]
    #     for n, t in np.ndindex(fermi_shape):
    #         fermi = amset_data.fermi_levels[n, t]
    #         broadening = BOLTZMANN * amset_data.temperatures[t] if broaden else 0
    #         energies[spin][n, t] = np.abs(spin_energies - fermi)  # + broadening

    return energies


def get_dos_effective_masses(amset_data: AmsetData):
    dos_effective_masses = {}
    for spin in amset_data.spins:
        masses = 1 / amset_data.curvature[spin]
        # masses_eig = np.linalg.eigh(masses)[0]

        masses_eig = np.diagonal(masses, axis1=2, axis2=3)
        masses_abs = np.abs(masses_eig)
        dos_effective_masses[spin] = np.power(np.product(masses_abs, axis=2), 1 / 3)
        # dos_effective_masses[spin] = np.full(dos_effective_masses[spin].shape, 0.55)
    return dos_effective_masses
