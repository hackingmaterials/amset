import numpy as np

from abc import ABC, abstractmethod
from logging import Logger
from typing import Optional, Union, Tuple, List

from monty.json import MSONable
from amset.logging import LoggableMixin
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


class AbstractInterpolater(MSONable, LoggableMixin, ABC):

    def __init__(self, band_structure: BandStructure, num_electrons: int,
                 calc_dir: str = '.', logger: Union[bool, Logger] = True,
                 log_level: Optional[int] = None, symprec: float = 0.01):
        self._band_structure = band_structure
        self._num_electrons = num_electrons
        self._calc_dir = calc_dir
        self._logger = self.get_logger(logger, level=log_level)
        self._symprec = symprec
        self._sga = SpacegroupAnalyzer(band_structure.structure,
                                       symprec=symprec)

    @abstractmethod
    def initialize(self):
        """Initialise the interpolater."""
        pass

    @abstractmethod
    def get_energy(self, kpoint: np.ndarray, iband: int,
                   return_velocity: bool = False,
                   return_effective_mass: bool = False
                   ) -> Union[float, Tuple[Union[float, np.ndarray], ...]]:
        """Gets the interpolated energy for a specific k-point and band.

        Args:
            kpoint: The k-point fractional coordinates.
            iband: The band index (0-indexed).
            return_velocity: Whether to return the band velocity.
            return_effective_mass: Whether to return the band effective mass.

        Returns:
            The band energies as a numpy array. If ``return_velocity`` or
            ``return_effective_mass`` are ``True`` a tuple is returned,
            formatted as::

               (energy, Optional[velocity], Optional[effecitve_mass])

            The velocity and effective mass are given as the 1x3 trace and
            full 3x3 tensor, respectively (along cartesian directions).
        """

    @abstractmethod
    def get_energies(self, kpoints: np.ndarray,
                     iband: Optional[Union[int, List[int]]] = None,
                     scissor: float = 0.0,
                     return_velocity: bool = False,
                     return_effective_mass: bool = False
                     ) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """Gets the interpolated energies for multiple k-points in a band.

        Args:
            kpoints: The k-points in fractional coordinates.
            iband: A band index or list of band indicies for which to get the
                energies. Band indices are 0-indexed. If ``None``, the energies
                for all available bands will be returned.
            scissor: The amount by which the band gap is scissored.
            return_velocity: Whether to return the band velocities.
            return_effective_mass: Whether to return the band effective masses.

        Returns:
            The band energies as a numpy array. If ``return_velocity`` or
            ``return_effective_mass`` are ``True`` a tuple is returned,
            formatted as::

                (energies, Optional[velocities], Optional[effective_masses])

            The velocities and effective masses are given as the 1x3 trace and
            full 3x3 tensor, respectively (along cartesian directions).
        """
        pass

    def get_dos(self, kpoint_mesh: List[int], emin: float = None,
                emax: float = None, estep: float = 0.001,
                width: float = 0.2, scissor: float = 0.0,
                normalize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the density of states using the interpolated bands.

        Args:
            kpoint_mesh: The k-point mesh as a 1x3 array. E.g.,``[6, 6, 6]``.
            emin: The minium energy. If ``None`` it will be calculated
                automatically from the band energies.
            emax: The maximum energy. If ``None`` it will be calculated
                automatically from the band energies.
            estep: The energy step, where smaller numbers give more
                accuracy but are more expensive.
            width: The gaussian smearing width.
            scissor: The amount by which the band gap is scissored.
            normalize: Whether to normalize the DOS.

        Returns:
            The density of states data, formatted as::

                (energies, densities)

        """
        mesh_data = np.array(self._sga.get_ir_reciprocal_mesh(kpoint_mesh))
        ir_kpts = mesh_data[:, 0]
        weights = mesh_data[:, 1] / mesh_data[:, 1].sum()

        energies = self.get_energies(ir_kpts, scissor=scissor)
        nbands = energies.shape[0]

        emin = emin if emin else np.min(energies)
        emax = emax if emax else np.max(energies)

        height = 1.0 / (width * np.sqrt(2 * np.pi))
        epoints = int(round((emax - emin) / estep))
        emesh = np.linspace(emin, emax, num=epoints, endpoint=True)
        dos = np.zeros(len(emesh))

        for ik, w in enumerate(weights):
            for b in range(nbands):
                g = height * np.exp(
                    -((emesh - energies[b, ik]) / width) ** 2 / 2.)
                dos += w * g

        if normalize:
            normalization_factor = nbands * (1 if self._soc else 2)


        return emesh, dos
