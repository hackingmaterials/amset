"""
TODO: add from dict and to dict methods which help load/save scattering_type and
      doping info
"""

import logging
import scipy
from typing import Optional, Dict, List

import numpy as np

from monty.json import MSONable
from scipy.ndimage import gaussian_filter1d

from BoltzTraP2 import units
from BoltzTraP2.bandlib import DOS
from amset.util import log_list, groupby
from pymatgen import Spin, Structure
from pymatgen.electronic_structure.dos import FermiDos, Dos

logger = logging.getLogger(__name__)


class AmsetData(MSONable):

    def __init__(self,
                 structure: Structure,
                 energies: Dict[Spin, np.ndarray],
                 vvelocities_product: Dict[Spin, np.ndarray],
                 projections: Dict[Spin, Dict[str, np.ndarray]],
                 kpoint_mesh: np.ndarray,
                 full_kpoints: np.ndarray,
                 ir_kpoints: np.ndarray,
                 ir_kpoint_weights: np.ndarray,
                 ir_kpoints_idx: np.ndarray,
                 ir_to_full_kpoint_mapping: np.ndarray,
                 efermi: float,
                 is_metal: bool,
                 soc: bool,
                 dos: Optional[FermiDos] = None,
                 dos_weight: Optional[int] = None,
                 vb_idx: Optional[Dict[Spin, int]] = None
                 ):
        self.structure = structure
        self.energies = energies
        self.velocities_product = vvelocities_product
        self.kpoint_mesh = kpoint_mesh
        self.full_kpoints = full_kpoints
        self.ir_kpoints = ir_kpoints
        self.ir_kpoint_weights = ir_kpoint_weights
        self.ir_kpoints_idx = ir_kpoints_idx
        self.ir_to_full_kpoint_mapping = ir_to_full_kpoint_mapping
        self._projections = projections
        self._efermi = efermi
        self._soc = soc

        self.dos = dos
        self.dos_weight = dos_weight
        self.is_metal = is_metal
        self.vb_idx = None if is_metal else vb_idx
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
        self.scattering_labels = None
        self.doping = None
        self.temperatures = None
        self.fermi_levels = None

        self.grouped_ir_to_full = groupby(
            np.arange(len(full_kpoints)), ir_to_full_kpoint_mapping)

    def calculate_dos(self, dos_estep: float = 0.01, dos_width: float = 0.01):
        """

        Args:
            dos_estep: The DOS energy step, where smaller numbers give more
                accuracy but are more expensive.
            dos_width: The DOS gaussian smearing width in eV.
        """

        all_energies = np.vstack([self.energies[spin] for spin in self.spins])
        all_energies /= units.eV  # convert from Hartree to eV for DOS

        # add a few multiples of dos_width to emin and emax to account for tails
        dos_emin = np.min(all_energies) - dos_width * 5
        dos_emax = np.max(all_energies) + dos_width * 5
        npts = int(round((dos_emax - dos_emin) / dos_estep))

        logger.debug("DOS parameters:")
        log_list(["emin: {:.2f} eV".format(dos_emin),
                  "emax: {:.2f} eV".format(dos_emax),
                  "broadening width: {} eV".format(dos_width)])

        emesh, densities = DOS(all_energies.T, erange=(dos_emin, dos_emax),
                               npts=npts)

        densities = gaussian_filter1d(densities, dos_width /
                                      (emesh[1] - emesh[0]))

        self.dos_weight = 1 if self._soc or len(self.spins) == 2 else 2
        densities *= self.dos_weight
        dos = Dos(self._efermi, emesh, {Spin.up: densities})

        # integrate up to Fermi level to get number of electrons
        energy_mask = emesh <= (self._efermi + dos_width)
        nelect = scipy.trapz(densities[energy_mask], emesh[energy_mask])

        logger.debug("DOS contains {:.3f} electrons".format(nelect))
        self.dos = FermiDos(dos, structure=self.structure, nelecs=nelect)

    def set_doping_and_temperatures(self,
                                    doping: np.ndarray,
                                    temperatures: np.ndarray):
        if not self.dos:
            raise RuntimeError(
                "The DOS should be calculated (AmsetData.calculate_dos) before "
                "setting doping levels.")

        self.doping = doping
        self.temperatures = temperatures

        self.fermi_levels = np.zeros((len(doping), len(temperatures)))

        logger.info("Calculated Fermi levels:")

        fermi_level_info = []
        for c, t in np.ndindex(self.fermi_levels.shape):
            # do minus -c as FermiDos treats negative concentrations as electron
            # doping and +ve as hole doping (the opposite to amset).
            self.fermi_levels[c, t] = self.dos.get_fermi(
                -doping[c], temperatures[t])
            fermi_level_info.append("{:.2g} cm⁻³ & {} K: {:.4f} eV".format(
                doping[c], temperatures[t], self.fermi_levels[c, t]))

        log_list(fermi_level_info)

    def set_scattering_rates(self,
                             scattering_rates: Dict[Spin, np.ndarray],
                             scattering_labels: List[str]):
        for spin in self.spins:
            expected_shape = ((len(self.doping), len(self.temperatures)) +
                              self.energies[spin].shape)
            if scattering_rates[spin].shape[1:] != expected_shape:
                raise ValueError(
                    "Shape of scattering_type rates array does not match the "
                    "number of dopings, temperatures, bands, or kpoints")

            if scattering_rates[spin].shape[0] != len(scattering_labels):
                raise ValueError(
                    "Number of scattering_type rates does not match number of "
                    "scattering_type labels")

        self.scattering_rates = scattering_rates
        self.scattering_labels = scattering_labels
