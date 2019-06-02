
import logging

from abc import ABC, abstractmethod

import numpy as np

from typing import Dict, Tuple, Any

from scipy.constants import epsilon_0
from scipy.integrate import trapz

from amset.constants import k_B, e, hbar
from amset.data import AmsetData
from amset.log import log_list
from amset.utils.transport import f0

logger = logging.getLogger(__name__)


class AbstractElasticScattering(ABC):

    name: str
    required_properties: Tuple[str]

    def __init__(self,
                 materials_properties: Dict[str, Any],
                 amset_data: AmsetData):
        self.properties = {p: materials_properties[p]
                           for p in self.required_properties}
        self.doping = amset_data.doping
        self.temperatures = amset_data.temperatures
        self.nbands = {s: len(amset_data.energies[s]) for s in amset_data.spins}
        self.spins = amset_data.spins

    @abstractmethod
    def prefactor(self):
        pass

    @abstractmethod
    def factor(self, *args):
        pass


class AcousticDeformationPotentialScattering(AbstractElasticScattering):

    name = "ACD"
    required_properties = ("deformation_potential", "elastic_constant")

    def __init__(self,
                 materials_properties: Dict[str, Any],
                 amset_data: AmsetData):
        super().__init__(materials_properties, amset_data)
        self.vb_idx = amset_data.vb_idx
        self.is_metal = amset_data.is_metal

    def prefactor(self):
        prefactor = {s: (1e18 * e * k_B / (4.0 * np.pi ** 2 * hbar *
                                           self.properties["elastic_constant"]))
                     for s in self.spins}

        deformation_potential = self.properties["deformation_potential"]
        for spin in self.spins:
            prefactor[spin] *= self.temperatures[None, :, None] * np.ones(
                (len(self.doping), len(self.temperatures), self.nbands[spin]))

            if self.is_metal and isinstance(deformation_potential, tuple):
                logger.warning(
                    "System is metallic but deformation potentials for both "
                    "the valence and conduction bands have been set. Using the "
                    "valence band value for all bands")
                prefactor[spin] *= deformation_potential[0]

            elif not self.is_metal and isinstance(deformation_potential, tuple):
                cb_idx = self.vb_idx[spin] + 1
                prefactor[spin][:, :, :cb_idx] *= deformation_potential[0] ** 2
                prefactor[spin][:, :, cb_idx:] *= deformation_potential[1] ** 2

            elif not self.is_metal:
                logger.warning(
                    "System is semiconducting but only one deformation "
                    "potential has been set. Using this value for all bands.")
                prefactor[spin] *= deformation_potential ** 2

            else:
                prefactor[spin] *= deformation_potential ** 2

        return prefactor

    def factor(self, k_diff_sq: np.ndarray):
        return np.ones((len(self.doping), len(self.temperatures),
                        k_diff_sq.shape[0]))


class IonizedImpurityScattering(AbstractElasticScattering):

    name = "IMP"
    required_properties = ("acceptor_charge", "donor_charge",
                           "static_dielectric")

    def __init__(self,
                 materials_properties: Dict[str, Any],
                 amset_data: AmsetData):
        super().__init__(materials_properties, amset_data)
        logger.debug("Initializing IMP scattering")

        self.beta_sq = np.zeros(amset_data.fermi_levels.shape)
        self.impurity_concentration = np.zeros(amset_data.fermi_levels.shape)

        tdos = amset_data.dos.tdos
        energies = amset_data.dos.energies
        fermi_levels = amset_data.fermi_levels
        de = amset_data.dos.de
        v_idx = amset_data.dos.idx_vbm
        c_idx = amset_data.dos.idx_cbm
        vol = amset_data.structure.volume

        # 1e-8 is Angstrom to cm conversion
        conv = 1 / (vol * 1e-8 ** 3)

        imp_info = []
        for n, t in np.ndindex(self.beta_sq.shape):
            ef = fermi_levels[n, t]
            temp = amset_data.temperatures[t]
            f = f0(energies, ef, temp)
            integral = trapz(tdos * f * (1 - f), x=energies)
            self.beta_sq[n, t] = (
                e ** 2 * integral * 1e12 /
                (self.properties["static_dielectric"] * epsilon_0 * k_B *
                 temp * e * vol))

            # calculate impurity concentration
            n_conc = np.abs(conv * np.sum(
                tdos[c_idx:] * f0(energies[c_idx:], ef, temp) * de[c_idx:],
                axis=0))
            p_conc = np.abs(conv * np.sum(
                tdos[:v_idx + 1] * (1 - f0(energies[:v_idx + 1], ef, temp))
                * de[:v_idx + 1], axis=0))

            self.impurity_concentration[n, t] = (
                    n_conc * self.properties["donor_charge"] ** 2 +
                    p_conc * self.properties["acceptor_charge"] ** 2)
            imp_info.append(
                "{:.2g} cm⁻³ & {} K: β² = {:.4g}, Nᵢᵢ = {:.4g}".format(
                    amset_data.doping[n], temp, self.beta_sq[n, t],
                    self.impurity_concentration[n, t]))

        logger.debug("Inverse screening length (β) and impurity concentration "
                     "(Nᵢᵢ):")
        log_list(imp_info, level=logging.DEBUG)

    def prefactor(self):
        prefactor = ((1e-3 / (e ** 2)) * e ** 4 * self.impurity_concentration /
                     (4.0 * np.pi ** 2 * epsilon_0 ** 2 * hbar *
                      self.properties["static_dielectric"] ** 2))

        # need to return prefactor with shape (nspins, ndops, ntemps, nbands)
        # currently it has shape (ndops, ntemps)
        return {s: np.repeat(prefactor[:, :, None], self.nbands[s], axis=-1)
                for s in self.spins}

    def factor(self, k_diff_sq: np.ndarray):
        # tile k_diff_sq to make it commensurate with the dimensions of beta
        return 1 / (np.tile(k_diff_sq, (len(self.doping),
                                        len(self.temperatures), 1)) +
                    self.beta_sq[..., None]) ** 2


class PiezoelectricScattering(AbstractElasticScattering):

    name = "PIE"
    required_properties = ("piezoelectric_coefficient", "static_dielectric")

    def __init__(self,
                 materials_properties: Dict[str, Any],
                 amset_data: AmsetData):
        super().__init__(materials_properties, amset_data)

    def prefactor(self):
        unit_conversion = 1e9 / e
        factor = (
            unit_conversion * e ** 2 * k_B *
            self.properties["piezoelectric_coefficient"] ** 2 /
            (4.0 * np.pi ** 2 * hbar * epsilon_0 *
             self.properties["static_dielectric"]))

        # need to return prefactor with shape (nspins, ndops, ntemps, nbands)
        return {
            s: self.temperatures[None, :, None] * factor * np.ones(
                (len(self.doping), len(self.temperatures), self.nbands[s]))
            for s in self.spins}

    def factor(self, k_diff_sq: np.ndarray):
        # factor should have shape (ndops, ntemps, nkpts)
        return 1 / np.tile(k_diff_sq, (len(self.doping),
                                       len(self.temperatures), 1))

