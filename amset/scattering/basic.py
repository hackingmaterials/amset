import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np

from amset.constants import nm_to_bohr, s_to_au
from amset.core.data import AmsetData

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

logger = logging.getLogger(__name__)


class AbstractBasicScattering(ABC):

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

    @property
    @abstractmethod
    def rates(self):
        pass

    @classmethod
    def get_properties(cls, materials_properties):
        return {p: materials_properties[p] for p in cls.required_properties}

    @staticmethod
    def get_nbands(amset_data):
        return {s: len(amset_data.energies[s]) for s in amset_data.spins}


class ConstantRelaxationTime(AbstractBasicScattering):

    name = "CRT"
    required_properties = ("constant_relaxation_time",)

    def __init__(self, properties, doping, temperatures, nbands, shape):
        super().__init__(properties, doping, temperatures, nbands)
        rate = 1 / self.properties["constant_relaxation_time"]
        self._rates = {s: np.full(shape[s], rate) for s in self.spins}

    @classmethod
    def from_amset_data(
        cls, materials_properties: Dict[str, Any], amset_data: AmsetData
    ):
        shape = {
            s: amset_data.fermi_levels.shape + amset_data.energies[s].shape
            for s in amset_data.spins
        }
        return cls(
            cls.get_properties(materials_properties),
            amset_data.doping,
            amset_data.temperatures,
            cls.get_nbands(amset_data),
            shape,
        )

    @property
    def rates(self):
        # need to return rates with shape (nspins, ndops, ntemps, nbands, nkpoints)
        return self._rates


class MeanFreePathScattering(AbstractBasicScattering):

    name = "MFP"
    required_properties = ("mean_free_path",)

    def __init__(self, properties, doping, temperatures, nbands, rates):
        super().__init__(properties, doping, temperatures, nbands)
        self._rates = rates

    @classmethod
    def from_amset_data(
        cls, materials_properties: Dict[str, Any], amset_data: AmsetData
    ):
        rates = {}
        mfp = materials_properties["mean_free_path"] * nm_to_bohr
        ir_kpoints_idx = amset_data.ir_kpoints_idx
        for spin in amset_data.spins:
            vvelocities = amset_data.velocities_product[spin][..., ir_kpoints_idx]
            v = np.sqrt(np.diagonal(vvelocities, axis1=1, axis2=2))
            v = np.linalg.norm(v, axis=2)
            v[v < 0.005] = 0.005  # handle very small velocities
            velocities = np.tile(
                v, (len(amset_data.doping), len(amset_data.temperatures), 1, 1)
            )

            spin_rates = velocities * s_to_au / mfp
            rates[spin] = spin_rates[..., amset_data.ir_to_full_kpoint_mapping]
        return cls(
            cls.get_properties(materials_properties),
            amset_data.doping,
            amset_data.temperatures,
            cls.get_nbands(amset_data),
            rates,
        )

    @property
    def rates(self):
        # need to return rates with shape (nspins, ndops, ntemps, nbands, nkpoints)
        return self._rates


class ScaledRelaxationTime(AbstractBasicScattering):

    name = "SRT"
    required_properties = ("base_relaxation_time",)

    def __init__(self, properties, doping, temperatures, nbands, shape):
        super().__init__(properties, doping, temperatures, nbands)
        self._rates = {}
        for spin in self.spins:
            spin_rates = np.zeros(shape[spin])
            for i, temperature in enumerate(temperatures):
                lifetime = self.properties["base_relaxation_time"] * 300 / temperature
                spin_rates[:, i] = 1 / lifetime

            self._rates[spin] = spin_rates

    @classmethod
    def from_amset_data(
        cls, materials_properties: Dict[str, Any], amset_data: AmsetData
    ):
        shape = {
            s: amset_data.fermi_levels.shape + amset_data.energies[s].shape
            for s in amset_data.spins
        }
        return cls(
            cls.get_properties(materials_properties),
            amset_data.doping,
            amset_data.temperatures,
            cls.get_nbands(amset_data),
            shape,
        )

    @property
    def rates(self):
        # need to return rates with shape (nspins, ndops, ntemps, nbands, nkpoints)
        return self._rates
