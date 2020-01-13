import logging


from abc import ABC, abstractmethod

import numpy as np

from typing import Dict, Tuple, Any

from BoltzTraP2 import units
from BoltzTraP2.fd import FD
from BoltzTraP2.units import BOLTZMANN
from scipy.constants import epsilon_0

from amset.constants import k_B, e, hbar, gpa_to_au
from amset.data import AmsetData
from pymatgen import Spin

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"
__date__ = "June 21, 2019"

logger = logging.getLogger(__name__)


class AbstractElasticScattering(ABC):

    name: str
    required_properties: Tuple[str]

    def __init__(self, materials_properties: Dict[str, Any], amset_data: AmsetData):
        self.properties = {p: materials_properties[p] for p in self.required_properties}
        self.doping = amset_data.doping
        self.temperatures = amset_data.temperatures
        self.nbands = {s: len(amset_data.energies[s]) for s in amset_data.spins}
        self.spins = amset_data.spins

    @abstractmethod
    def prefactor(self, spin: Spin, b_idx: int):
        pass

    @abstractmethod
    def factor(self):
        pass


class AcousticDeformationPotentialScattering(AbstractElasticScattering):

    name = "ACD"
    required_properties = ("deformation_potential", "elastic_constant")

    def __init__(self, materials_properties: Dict[str, Any], amset_data: AmsetData):
        super().__init__(materials_properties, amset_data)
        self.vb_idx = amset_data.vb_idx
        self.is_metal = amset_data.is_metal
        self._prefactor = (
            BOLTZMANN
            * units.Second
            / (4.0 * np.pi ** 2 * self.properties["elastic_constant"] * gpa_to_au)
        )

        self.deformation_potential = self.properties["deformation_potential"]
        if self.is_metal and isinstance(self.deformation_potential, tuple):
            logger.warning(
                "System is metallic but deformation potentials for both "
                "the valence and conduction bands have been set... using the "
                "valence band potential for all bands"
            )
            self.deformation_potential = self.deformation_potential[0] * units.eV

        elif self.is_metal:
            self.deformation_potential = self.deformation_potential * units.eV

        elif not self.is_metal and not isinstance(self.deformation_potential, tuple):
            logger.warning(
                "System is semiconducting but only one deformation "
                "potential has been set... using this potential for all bands."
            )
            self.deformation_potential = (
                self.deformation_potential * units.eV,
                self.deformation_potential * units.eV,
            )

        else:
            self.deformation_potential = (
                self.deformation_potential[0] * units.eV,
                self.deformation_potential[1] * units.eV,
            )

    def prefactor(self, spin: Spin, b_idx: int):
        prefactor = (
            self._prefactor
            * self.temperatures[None, :]
            * np.ones((len(self.doping), len(self.temperatures)))
        )

        if self.is_metal:
            prefactor *= self.deformation_potential ** 2

        else:
            def_idx = 1 if b_idx > self.vb_idx[spin] else 0
            prefactor *= self.deformation_potential[def_idx] ** 2

        return prefactor

    def factor(self):
        def acd_function_generator(_):
            return lambda x: np.ones_like(x[0])

        return acd_function_generator


# class IonizedImpurityScattering(AbstractElasticScattering):
#
#     name = "IMP"
#     required_properties = ("acceptor_charge", "donor_charge", "static_dielectric")
#
#     def __init__(self, materials_properties: Dict[str, Any], amset_data: AmsetData):
#         super().__init__(materials_properties, amset_data)
#         logger.debug("Initializing IMP scattering")
#
#         self.inverse_screening_length_sq = calculate_inverse_screening_length_sq(
#             amset_data, self.properties["static_dielectric"]
#         )
#         self.impurity_concentration = np.zeros(amset_data.fermi_levels.shape)
#
#         imp_info = []
#         for n, t in np.ndindex(self.inverse_screening_length_sq.shape):
#             n_conc = np.abs(amset_data.electron_conc[n, t])
#             p_conc = np.abs(amset_data.hole_conc[n, t])
#
#             self.impurity_concentration[n, t] = (
#                 n_conc * self.properties["donor_charge"] ** 2
#                 + p_conc * self.properties["acceptor_charge"] ** 2
#             )
#             imp_info.append(
#                 "{:.2g} cm⁻³ & {} K: β² = {:.4g} a₀⁻², Nᵢᵢ = {:.4g} cm⁻³".format(
#                     amset_data.doping[n] * (1 / bohr_to_cm) ** 3,
#                     amset_data.temperatures[t],
#                     self.inverse_screening_length_sq[n, t],
#                     self.impurity_concentration[n, t] * (1 / bohr_to_cm) ** 3,
#                 )
#             )
#
#         logger.debug(
#             "Inverse screening length (β) and impurity concentration " "(Nᵢᵢ):"
#         )
#         log_list(imp_info, level=logging.DEBUG)
#
#         self._prefactor = (
#             self.impurity_concentration
#             * 4
#             * units.Second
#             / self.properties["static_dielectric"] ** 2
#         )
#
#     def prefactor(self, spin: Spin, b_idx: int):
#         # need to return prefactor with shape (nspins, ndops, ntemps, nbands)
#         return self._prefactor
#
#     # def factor(self, k_diff_sq: np.ndarray):
#     def factor(self):
#         def imp_function_generator(z_coords_sq: np.ndarray):
#             return (
#                 lambda x: 1
#                 / (
#                     (
#                         x[0][None, None, :, :] ** 2
#                         + x[1][None, None, :, :] ** 2
#                         + z_coords_sq[None, None, :, None]
#                         + self.inverse_screening_length_sq[:, :, None, None]
#                     )
#                 )
#                 ** 2
#             )
#
#         return imp_function_generator


class PiezoelectricScattering(AbstractElasticScattering):

    name = "PIE"
    required_properties = ("piezoelectric_coefficient", "static_dielectric")

    def __init__(self, materials_properties: Dict[str, Any], amset_data: AmsetData):
        super().__init__(materials_properties, amset_data)
        unit_conversion = 1e9 / e
        self._prefactor = (
            unit_conversion
            * e ** 2
            * k_B
            * self.properties["piezoelectric_coefficient"] ** 2
            / (
                4.0
                * np.pi ** 2
                * hbar
                * epsilon_0
                * self.properties["static_dielectric"]
            )
        )

    def prefactor(self, spin: Spin, b_idx: int):
        # need to return prefactor with shape (ndops, ntemps)
        return (
            self._prefactor
            * self.temperatures[None, :]
            * np.ones((len(self.doping), len(self.temperatures)))
        )

    def factor(self, k_diff_sq: np.ndarray):
        # factor should have shape (ndops, ntemps, nkpts)
        return 1 / np.tile(k_diff_sq, (len(self.doping), len(self.temperatures), 1))


def calculate_inverse_screening_length_sq(amset_data, static_dielectric):
    inverse_screening_length_sq = np.zeros(amset_data.fermi_levels.shape)

    tdos = amset_data.dos.tdos
    energies = amset_data.dos.energies
    fermi_levels = amset_data.fermi_levels
    vol = amset_data.structure.volume

    for n, t in np.ndindex(inverse_screening_length_sq.shape):
        ef = fermi_levels[n, t]
        temp = amset_data.temperatures[t]
        f = FD(energies, ef, temp * units.BOLTZMANN)
        integral = np.trapz(tdos * f * (1 - f), x=energies)
        inverse_screening_length_sq[n, t] = (
            integral * 4 * np.pi / (static_dielectric * BOLTZMANN * temp * vol)
        )

    return inverse_screening_length_sq


# def calculate_inverse_screening_length_sq(amset_data: AmsetData, static_dielectric):
#     inverse_screening_length_sq = np.zeros(amset_data.fermi_levels.shape)
#
#     fermi_levels = amset_data.fermi_levels
#     k_norm = np.linalg.norm(np.dot(amset_data.full_kpoints, amset_data.structure.lattice.reciprocal_lattice.matrix), axis=1)
#     k_sq = (k_norm / np.pi) ** 2
#
#     for n, t in np.ndindex(inverse_screening_length_sq.shape):
#
#         ef = fermi_levels[n, t]
#         temp = amset_data.temperatures[t]
#         integral = 0
#         for spin, spin_energies in amset_data.energies.items():
#             for band_energies in spin_energies:
#                 f = FD(band_energies, ef, temp * units.BOLTZMANN)
#                 integral += np.sum(k_sq * f * (1 - f))
#
#         integral /= len(amset_data.full_kpoints)
#
#         inverse_screening_length_sq[n, t] = (
#                 integral * 4 * np.pi / (static_dielectric * BOLTZMANN * temp)
#         )
#
#     return inverse_screening_length_sq
#
