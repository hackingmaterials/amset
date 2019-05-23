"""
TODO: add from dict and to dict methods which help load/save scattering and
      doping info
"""

import itertools
import logging
from typing import Optional, Dict

import numpy as np

from monty.json import MSONable

from pymatgen import Spin
from pymatgen.electronic_structure.dos import FermiDos

logger = logging.getLogger(__name__)


class ElectronicStructure(MSONable):

    def __init__(self,
                 energies: Dict[Spin, np.ndarray],
                 vvelocities_product: Dict[Spin, np.ndarray],
                 projections: Dict[Spin, Dict[str, np.ndarray]],
                 kpoint_mesh: np.ndarray,
                 full_kpoints: np.ndarray,
                 ir_kpoints: np.ndarray,
                 ir_kpoint_weights: np.ndarray,
                 ir_kpoints_idx: np.ndarray,
                 ir_to_full_kpoint_mapping: np.ndarray,
                 dos: FermiDos,
                 dos_weight: int):
        self.energies = energies
        self.velocities_product = vvelocities_product
        self.kpoint_mesh = kpoint_mesh
        self.full_kpoints = full_kpoints
        self.ir_kpoints = ir_kpoints
        self.ir_kpoint_weights = ir_kpoint_weights
        self.ir_kpoints_idx = ir_kpoints_idx
        self.ir_to_full_kpoint_mapping = ir_to_full_kpoint_mapping
        self.structure = dos.structure
        self._projections = projections
        self.dos = dos
        self.dos_weight = dos_weight
        self.spins = self.energies.keys()
        self.kpoint_norms = np.linalg.norm(full_kpoints, axis=1)

        self.a_factor = {}
        self.c_factor = {}

        for spin in self.spins:
            self.a_factor[spin] = (
                projections[spin]["s"] / (projections[spin]["s"] ** 2 +
                                          projections[spin]["p"] ** 2) ** 0.5)
            self.c_factor[spin] = (1 - self.a_factor[spin] ** 2) ** 0.5

        self.scattering_rates = None
        self.doping = None
        self.temperatures = None
        self.fermi_levels = None

    def set_doping_and_temperatures(self,
                                    doping: np.ndarray,
                                    temperatures: np.ndarray):
        self.doping = doping
        self.temperatures = temperatures

        self.fermi_levels = np.zeros((len(doping), len(temperatures)))
        for c, t in itertools.product(doping, temperatures):
            self.fermi_levels[c, t] = self.dos.get_fermi(c, t)

    def set_scattering_rates(self,
                             scattering_rates: Dict[Spin, np.ndarray]):
        for spin in self.spins:
            expected_shape = ((len(self.doping), len(self.temperatures)) +
                              self.energies[spin].shape)
            if scattering_rates[spin].shape != expected_shape:
                raise ValueError(
                    "Shape of scattering rates array does not match the number"
                    " of dopings, temperatures, bands, or kpoints!")

        self.scattering_rates = scattering_rates


