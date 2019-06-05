
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
from pymatgen import Spin

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
    def prefactor(self, spin: Spin, b_idx: int):
        pass

    @abstractmethod
    def factor(self, k_diff_sq: np.ndarray):
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
        self._prefactor = (1e18 * e * k_B / (
                4.0 * np.pi ** 2 * hbar * self.properties["elastic_constant"]))

        self.deformation_potential = self.properties["deformation_potential"]
        if self.is_metal and isinstance(self.deformation_potential, tuple):
            logger.warning(
                "System is metallic but deformation potentials for both "
                "the valence and conduction bands have been set... using the "
                "valence band potential for all bands")
            self.deformation_potential = self.deformation_potential[0]

        elif not self.is_metal and not isinstance(
                self.deformation_potential, tuple):
            logger.warning(
                "System is semiconducting but only one deformation "
                "potential has been set... using this potential for all bands.")
            self.deformation_potential = (self.deformation_potential,
                                          self.deformation_potential)

    def prefactor(self, spin: Spin, b_idx: int):
        prefactor = self._prefactor * self.temperatures[None, :] * np.ones(
            (len(self.doping), len(self.temperatures)))

        if self.is_metal:
            prefactor *= self.properties["deformation_potential"] ** 2

        else:
            def_idx = 1 if b_idx > self.vb_idx[spin] else 0
            prefactor *= self.properties["deformation_potential"][def_idx] ** 2

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
        vol = amset_data.structure.volume

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

            n_conc = np.abs(amset_data.electron_conc[n, t])
            p_conc = np.abs(amset_data.hole_conc[n, t])

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

        self._prefactor = (
                (1e-3 / (e ** 2)) * e ** 4 * self.impurity_concentration /
                (4.0 * np.pi ** 2 * epsilon_0 ** 2 * hbar *
                 self.properties["static_dielectric"] ** 2))

    def prefactor(self, spin: Spin, b_idx: int):
        # need to return prefactor with shape (nspins, ndops, ntemps, nbands)
        return self._prefactor

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
        unit_conversion = 1e9 / e
        self._prefactor = (unit_conversion * e ** 2 * k_B *
                           self.properties["piezoelectric_coefficient"] ** 2 /
                           (4.0 * np.pi ** 2 * hbar * epsilon_0 *
                            self.properties["static_dielectric"]))

    def prefactor(self, spin: Spin, b_idx: int):
        # need to return prefactor with shape (ndops, ntemps)
        return self._prefactor * self.temperatures[None, :] * np.ones(
                (len(self.doping), len(self.temperatures)))

    def factor(self, k_diff_sq: np.ndarray):
        # factor should have shape (ndops, ntemps, nkpts)
        return 1 / np.tile(k_diff_sq, (len(self.doping),
                                       len(self.temperatures), 1))
