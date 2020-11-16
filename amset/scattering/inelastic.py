import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
from pymatgen import Spin

from amset.constants import boltzmann_au, hbar, s_to_au
from amset.core.data import AmsetData
from amset.log import log_list
from amset.scattering.common import calculate_inverse_screening_length_sq

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

logger = logging.getLogger(__name__)


class AbstractInelasticScattering(ABC):

    name: str
    required_properties: Tuple[str]

    def __init__(self, properties, doping, temperatures, nbands):
        self.properties = properties
        self.doping = doping
        self.temperatures = temperatures
        self.nbands = nbands
        self.spins = list(nbands.keys())

    @classmethod
    def from_amset_data(
        cls, materials_properties: Dict[str, Any], amset_data: AmsetData
    ):
        return cls(
            cls.get_properties(materials_properties),
            amset_data.doping,
            amset_data.temperatures,
            cls.get_nbands(amset_data),
        )

    @abstractmethod
    def prefactor(self, spin: Spin, b_idx: int):
        pass

    @abstractmethod
    def factor(self, unit_q, norm_q_sq, emission, f):
        pass

    @classmethod
    def get_properties(cls, materials_properties):
        return {p: materials_properties[p] for p in cls.required_properties}

    @staticmethod
    def get_nbands(amset_data):
        return {s: len(amset_data.energies[s]) for s in amset_data.spins}


class PolarOpticalScattering(AbstractInelasticScattering):

    name = "POP"
    required_properties = (
        "pop_frequency",
        "static_dielectric",
        "high_frequency_dielectric",
        "free_carrier_screening",
    )

    def __init__(
        self,
        properties,
        doping,
        temperatures,
        nbands,
        pop_frequency,
        n_po,
        inverse_screening_length_sq,
    ):
        super().__init__(properties, doping, temperatures, nbands)
        self.pop_frequency = pop_frequency
        self.n_po = n_po
        self.inverse_screening_length_sq = inverse_screening_length_sq
        self._prefactor = s_to_au * pop_frequency / 2

    @classmethod
    def from_amset_data(
        cls, materials_properties: Dict[str, Any], amset_data: AmsetData
    ):
        logger.info("Initializing POP scattering")

        # convert from THz to angular frequency in Hz
        pop_frequency = (
            materials_properties["pop_frequency"] * 1e12 * 2 * np.pi / s_to_au
        )

        if materials_properties["free_carrier_screening"]:
            # use high-frequency diel for screening length
            avg_diel = np.linalg.eigvalsh(
                materials_properties["high_frequency_dielectric"]
            ).mean()
            inverse_screening_length_sq = calculate_inverse_screening_length_sq(
                amset_data, avg_diel
            )
        else:
            inverse_screening_length_sq = np.zeros_like(amset_data.fermi_levels)

        # n_po (phonon concentration) has shape (ntemps, )
        n_po = 1 / (
            np.exp(pop_frequency / (boltzmann_au * amset_data.temperatures)) - 1
        )

        n_po = n_po[None, :]

        log_list(
            [
                "average N_po: {:.4f}".format(np.mean(n_po)),
                "ω_po: {:.4g} 2π THz".format(
                    materials_properties["pop_frequency"] * 2 * np.pi
                ),
                "ħω: {:.4f} eV".format(pop_frequency * hbar * s_to_au),
            ]
        )

        # want to store two intermediate properties for:
        #             emission      and        absorption
        # (1-f)(N_po + 1) + f(N_po) and (1-f)N_po + f(N_po + 1)
        # note that these are defined for the scattering rate S(k', k).
        # For the rate S(k, k') the definitions are reversed.

        # self.emission_f_out = {
        #     s: n_po + 1 - amset_data.f[s]
        #     for s in amset_data.spins}
        # self.absorption_f_out = {
        #     s: n_po + amset_data.f[s]
        #     for s in amset_data.spins}
        #
        # self.emission_f_in = {
        #     s: n_po + amset_data.f[s]
        #     for s in amset_data.spins}
        # self.absorption_f_in = {
        #     s: n_po + 1 - amset_data.f[s]
        #     for s in amset_data.spins}
        return cls(
            cls.get_properties(materials_properties),
            amset_data.doping,
            amset_data.temperatures,
            cls.get_nbands(amset_data),
            pop_frequency,
            n_po,
            inverse_screening_length_sq,
        )

    def prefactor(self, spin: Spin, b_idx: int):
        # need to return prefactor with shape (nspins, ndops, ntemps, nbands)
        return self._prefactor * np.ones((len(self.doping), len(self.temperatures)))

    def factor(self, unit_q: np.ndarray, norm_q_sq: np.ndarray, emission, f):
        # presuming that this is out scattering
        if emission:
            factor = self.n_po + 1 - f
        else:
            factor = self.n_po + f

        high_t = self.properties["high_frequency_dielectric"]
        static_t = self.properties["static_dielectric"]
        high_freq_diel = np.einsum("ij,ij->i", unit_q, np.dot(high_t, unit_q.T).T)
        static_diel = np.einsum("ij,ij->i", unit_q, np.dot(static_t, unit_q.T).T)

        dielectric_term = 4 * np.pi * (1 / high_freq_diel - 1 / static_diel)

        return (
            factor[..., None]
            * dielectric_term[None, None]
            / (norm_q_sq[None, None] + self.inverse_screening_length_sq[..., None])
        )

        # # factor should have shape (ndops, ntemps, nkpts)
        # factor = 1 / np.tile(k_diff_sq, (len(self.doping),
        #                                  len(self.temperatures), 1))
        # if emission:
        #     if out:
        #         return factor * self.emission_f_out[spin][:, :, b_idx, k_idx]
        #     else:
        #         return factor * self.emission_f_in[spin][:, :, b_idx, k_idx]
        # else:
        #     if out:
        #         return factor * self.absorption_f_out[spin][:, :, b_idx, k_idx]
        #     else:
        #         return factor * self.absorption_f_in[spin][:, :, b_idx, k_idx]
