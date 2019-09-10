import logging
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

import numpy as np
from scipy.constants import epsilon_0

from amset.misc.constants import hbar, k_B, e
from amset.data import AmsetData
from pymatgen import Spin

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"
__date__ = "June 21, 2019"

logger = logging.getLogger(__name__)


class AbstractInelasticScattering(ABC):

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
    def factor(self, spin, b_idx, k_idx, k_diff_sq: np.ndarray, emission,
               out):
        pass


class PolarOpticalScattering(AbstractInelasticScattering):

    name = "POP"
    required_properties = ("pop_frequency", "static_dielectric",
                           "high_frequency_dielectric")

    def __init__(self,
                 materials_properties: Dict[str, Any],
                 amset_data: AmsetData):
        super().__init__(materials_properties, amset_data)
        logger.debug("Initializing POP scattering")

        # convert from THz to angular frequency in Hz
        self.pop_frequency = self.properties["pop_frequency"] * 1e12 * 2 * np.pi

        # n_po (phonon concentration) has shape (ntemps, )
        n_po = 1 / (np.exp(hbar * self.pop_frequency /
                    (k_B * amset_data.temperatures)) - 1)

        n_po = n_po[None, :, None, None]

        logger.debug("average N_po = {}".format(np.mean(n_po)))

        # want to store two intermediate properties for:
        #             emission      and        absorption
        # (1-f)(N_po + 1) + f(N_po) and (1-f)N_po + f(N_po + 1)
        # note that these are defined for the scattering rate S(k', k).
        # For the rate S(k, k') the definitions are reversed.

        self.emission_f_out = {
            s: n_po + 1 - amset_data.f[s]
            for s in amset_data.spins}
        self.absorption_f_out = {
            s: n_po + amset_data.f[s]
            for s in amset_data.spins}

        self.emission_f_in = {
            s: n_po + amset_data.f[s]
            for s in amset_data.spins}
        self.absorption_f_in = {
            s: n_po + 1 - amset_data.f[s]
            for s in amset_data.spins}

        unit_conversion = 1e9 / e
        self._prefactor = unit_conversion * (
                e ** 2 * self.pop_frequency / (8 * np.pi ** 2) *
                (1 / self.properties["high_frequency_dielectric"] -
                 1 / self.properties["static_dielectric"]) / epsilon_0)

    def prefactor(self, spin: Spin, b_idx: int):
        # need to return prefactor with shape (nspins, ndops, ntemps, nbands)
        return self._prefactor * np.ones(
            (len(self.doping), len(self.temperatures)))

    def factor(self, spin, b_idx, k_idx, k_diff_sq: np.ndarray, emission,
               out):
        # factor should have shape (ndops, ntemps, nkpts)
        factor = 1 / np.tile(k_diff_sq, (len(self.doping),
                                         len(self.temperatures), 1))
        if emission:
            if out:
                return factor * self.emission_f_out[spin][:, :, b_idx, k_idx]
            else:
                return factor * self.emission_f_in[spin][:, :, b_idx, k_idx]
        else:
            if out:
                return factor * self.absorption_f_out[spin][:, :, b_idx, k_idx]
            else:
                return factor * self.absorption_f_in[spin][:, :, b_idx, k_idx]
