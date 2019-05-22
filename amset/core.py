from typing import Optional, Dict

import numpy as np

from monty.json import MSONable

from pymatgen import Spin
from pymatgen.electronic_structure.dos import FermiDos


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
                 scattering_rates: Optional[Dict[Spin, np.ndarray]] = None):
        self.energies = energies
        self.velocities_product = vvelocities_product
        self.kpoint_mesh = kpoint_mesh
        self.full_kpoints = full_kpoints
        self.ir_kpoints = ir_kpoints
        self.ir_kpoint_weights = ir_kpoint_weights
        self.ir_kpoints_idx = ir_kpoints_idx
        self.ir_to_full_kpoint_mapping = ir_to_full_kpoint_mapping
        self.scattering_rates = scattering_rates
        self.structure = dos.structure
        self._projections = projections
        self._dos = dos
        self.spins = self.energies.keys()
        self.kpoint_norms = np.linalg.norm(full_kpoints, axis=1)

        self.a_factor = {}
        self.c_factor = {}

        for spin in self.spins:
            self.a_factor[spin] = (
                projections[spin]["s"] / (projections[spin]["s"] ** 2 +
                                          projections[spin]["p"] ** 2) ** 0.5)
            self.c_factor[spin] = (1 - self.a_factor[spin] ** 2) ** 0.5
