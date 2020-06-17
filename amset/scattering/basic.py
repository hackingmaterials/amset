import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
from BoltzTraP2.units import Second

from amset.constants import nm_to_bohr
from amset.core.data import AmsetData

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

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


class ConstantRelaxationTime(AbstractBasicScattering):

    name = "CRT"
    required_properties = ("constant_relaxation_time",)

    def __init__(self, materials_properties: Dict[str, Any], amset_data: AmsetData):
        super().__init__(materials_properties, amset_data)
        rate = 1 / self.properties["constant_relaxation_time"]
        shape = {
            s: amset_data.fermi_levels.shape + amset_data.energies[s].shape
            for s in self.spins
        }
        self._rates = {s: np.full(shape[s], rate) for s in self.spins}

    @property
    def rates(self):
        # need to return rates with shape (nspins, ndops, ntemps, nbands, nkpoints)
        return self._rates


class MeanFreePathScattering(AbstractBasicScattering):

    name = "MFP"
    required_properties = ("mean_free_path",)

    def __init__(self, materials_properties: Dict[str, Any], amset_data: AmsetData):
        super().__init__(materials_properties, amset_data)

        # convert mean free path from nm to bohr
        mfp = self.properties["mean_free_path"] * nm_to_bohr

        self._rates = {}
        ir_kpoints_idx = amset_data.ir_kpoints_idx
        for spin in self.spins:
            vvelocities = amset_data.velocities_product[spin][..., ir_kpoints_idx]
            v = np.sqrt(np.diagonal(vvelocities, axis1=1, axis2=2))
            v = np.linalg.norm(v, axis=2)
            v[v < 0.005] = 0.005  # handle very small velocities
            velocities = np.tile(v, (len(self.doping), len(self.temperatures), 1, 1))

            rates = velocities * Second / mfp
            self._rates[spin] = rates[..., amset_data.ir_to_full_kpoint_mapping]

    @property
    def rates(self):
        # need to return rates with shape (nspins, ndops, ntemps, nbands, nkpoints)
        return self._rates
