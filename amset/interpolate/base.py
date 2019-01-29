from abc import ABC, abstractmethod
from monty.json import MSONable
from amset.logging import LoggableMixin


class AbstractInterpolater(MSONable, LoggableMixin, ABC):

    def __init__(self, band_structure, num_electrons, calc_dir='.'):
        self._band_structure = band_structure
        self._num_electrons = num_electrons
        self._calc_dir = calc_dir

    @abstractmethod
    def initialize(self):
        """Initialise the interpolater."""
        pass

    def get_energy(self, kpoint, iband, return_velocity=True,
                   return_effective_mass=False):
        """Gets the interpolated energy for a specific k-point and band.

        Args:
            kpoint (np.ndarray): The k-point fractional coordinates.
            iband (int): The band index (1-indexed).
            return_velocity (bool, optional): Whether to return the band
                velocity.
            return_effective_mass (bool, optional): Whether to return the band
                effective mass.

        Returns:
            (tuple[int or np.ndarray]): A tuple containing the band energy, and
            optionally the velocity and effective mass if asked for. The
            velocities and effective masses are given as the full 3x3 tensors
            along cartesian directions.
        """
        pass

    @abstractmethod
    def get_energies(self, kpoints, iband, scissor=0.0, is_cb=None,
                     return_effective_mass=True):
        """Gets the interpolated energies for multiple k-points in a band.

        Args:
            kpoints (np.ndarray): The k-points in fractional coordinates.
            iband (int): The band index (1-indexed).
            scissor (float, optional): The amount by which the band gap is
                scissored.
            is_cb (bool, optional): Whether the band of interest is a conduction
                band. Ignored if ``scissor == 0``.
            return_effective_mass (bool, optional): Whether to return the band
                effective masses.

        Returns:
            (tuple[np.ndarray]): A tuple containing the band energies,
             velocities and, optionally, the effective masses if asked for, for
             each k-point. The velocities and effective masses are given as
             the full 3x3 tensors along cartesian directions.
        """
        pass
