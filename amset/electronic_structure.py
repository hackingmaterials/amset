from typing import Optional, Dict

import numpy as np

from monty.json import MSONable

from pymatgen import Spin
from pymatgen.electronic_structure.dos import FermiDos


class ElectronicStructure(MSONable):

    def __init__(self,
                 energies: Dict[Spin, np.ndarray],
                 vvelocities_product: Dict[Spin, np.ndarray],
                 projections: Dict[str, Dict[Spin, np.ndarray]],
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
        self._structure = dos.structure
        self._projections = projections
        self._dos = dos
