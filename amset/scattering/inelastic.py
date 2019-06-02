from abc import ABC

import numpy as np
from scipy.constants import epsilon_0

from BoltzTraP2 import units
from BoltzTraP2.fd import FD
from amset.constants import hbar, k_B, e
from amset.data import AmsetData
from amset.scattering.elastic import AbstractElasticScattering


class AbstractElasticScattering(ABC):

    name: str
    required_properties: Tuple[str]

    def initialize(self, amset_data):
        pass

    @abstractmethod
    def prefactor(self, amset_data: AmsetData):
        pass

    @abstractmethod
    def factor(self, *args):
        pass


class PolarOpticalScattering(AbstractElasticScattering):

    name = "POP"
    required_properties = ("pop_frequency", "static_dielectric",
                           "high_frequency_dielectric")
    inelastic = True

    def __init__(self,
                 pop_frequency: float,
                 static_dielectric: float,
                 high_frequency_dielectric: float):
        self.pop_frequency = pop_frequency
        self.static_dielectric = static_dielectric
        self.high_frequency_dielectric = high_frequency_dielectric
        self.emission_f = None
        self.absorption_f = None
        self.g = None

    def initialize(self, amset_data):
        # g has shape (spin, ndops, ntemps, nbands, nkpts)
        self.g = {s: np.ones(amset_data.fermi_levels.shape +
                             amset_data.energies[s].shape)
                  for s in amset_data.spins}

        # f has shape (spin, ndops, ntemps, nbands, nkpts)
        f = {s: np.ones(amset_data.fermi_levels.shape +
                        amset_data.energies[s].shape)
             for s in amset_data.spins}

        for s, (n, t) in zip(amset_data.spins,
                             np.ndindex(amset_data.fermi_levels.shape)):
            f[s][n, t] = FD(
                amset_data.energies[s],
                amset_data.fermi_levels[n, t] * units.eV,
                amset_data.temperatures[t])

        # n_po (phonon concentration) has shape (ntemps, )
        n_po = 1 / np.exp(hbar * self.pop_frequency /
                          k_B * amset_data.temperatures) - 1
        n_po.reshape(1, len(amset_data.temperatures), 1, 1)

        # want to store two intermediate properties for:
        #             emission      and        absorption
        # (1-f)(N_po + 1) + f(N_po) and (1-f)N_po + f(N_po + 1)
        self.emission_f = {s: (1 - f[s]) * (n_po + 1) + f[s] * n_po
                           for s in amset_data.spins}
        self.absorption_f = {s: (1 - f[s]) * n_po + f[s] * (n_po + 1)
                             for s in amset_data.spins}

    def prefactor(self, amset_data: AmsetData):
        prefactor = (e ** 2 * self.pop_frequency / (4 * np.pi * hbar) *
                     (1 / self.high_frequency_dielectric -
                      1 / self.static_dielectric) / epsilon_0 * 100 / e)

        # need to return prefactor with shape (nspins, ndops, ntemps, nbands)
        return {s: np.ones(amset_data.fermi_levels.shape +
                           (amset_data.energies.shape[0], )) * prefactor
                for s in amset_data.spins}

    def factor(self, doping: np.ndarray, temperatures: np.ndarray,
               k_diff_sq: np.ndarray, k_angles: np.ndarray, k_p_idx):
        return 1
