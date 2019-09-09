"""
Customised implementation of FermiDos. Will Move back to pymatgen at some point.
"""
import logging

import numpy as np

from typing import Optional, Union, Tuple, Dict

from BoltzTraP2 import units
from BoltzTraP2.fd import FD
from monty.json import MSONable

from pymatgen import Structure, Spin
from pymatgen.electronic_structure.dos import Dos

logger = logging.getLogger(__name__)


class FermiDos(Dos, MSONable):
    """
    This wrapper class helps relate the density of states, doping levels
    (i.e. carrier concentrations) and corresponding fermi levels. A negative
    doping concentration indicates the majority carriers are electrons
    (n-type doping); a positive doping concentration indicates holes are the
    majority carriers (p-type doping).

    Args:
        efermi: The Fermi level energy in Hartree.
        energies: A sequences of energies in Hartree.
        densities ({Spin: np.array}): representing the density of states
            for each Spin.
        structure: A structure. If not provided, the structure
            of the dos object will be used. If the dos does not have an
            associated structure object, an error will be thrown.
        dos_weight: The weighting for the dos. Defaults to 2 for non-spin
            polarized calculations and 1 for spin-polarized calculations.
    """

    def __init__(self,
                 efermi: float,
                 energies: np.ndarray,
                 densities: Dict[Spin, np.ndarray],
                 structure: Structure,
                 dos_weight: Optional[float] = None):

        super().__init__(efermi, energies, densities)
        self.structure = structure

        if not dos_weight:
            dos_weight = 2 if len(self.densities) == 1 else 1
        self.dos_weight = dos_weight
        self.tdos = np.array(self.get_densities()) * self.dos_weight
        self.de = self.energies[1] - self.energies[0]

        # N_electrons to concentration (in 1/cm^3) conversion; 1e-8 is
        # conversion from Angstrom to cm
        self._conv = self.structure.volume * 1e-8 ** 3

        # integrate up to Fermi level to get number of electrons
        self.nelect = self.tdos[self.energies <= self.efermi].sum() * self.de

        logger.debug("Intrinsic DOS Fermi level: {:.4f} eV".format(
            self.efermi / units.eV))
        logger.debug("DOS contains {:.3f} electrons".format(self.nelect))

    def get_doping(self,
                   fermi_level: float,
                   temperature: float,
                   return_electron_hole_conc: bool = False
                   ) -> Union[float, Tuple[float, float, float]]:
        """
        Calculate the doping (majority carrier concentration) at a given
        fermi level  and temperature. A simple Left Riemann sum is used for
        integrating the density of states over energy & equilibrium Fermi-Dirac
        distribution.

        Args:
            fermi_level: The fermi_level level in Hartree.
            temperature: The temperature in Kelvin.
            return_electron_hole_conc: Whether to also return the separate
                electron and hole concentrations at the doping level.

        Returns:
            If return_electron_hole_conc is False: the doping concentration in
            units of 1/cm^3. Negative values indicate that the majority carriers
            are electrons (n-type doping) whereas positive values indicates the
            majority carriers are holes (p-type doping).

            If return_electron_hole_conc is True: the doping concentration,
            electron concentration and hole concentration as a tuple.
        """
        if temperature == 0.:
            occ = np.where(self.energies < fermi_level, 1., 0.)
            occ[self.energies == fermi_level] = .5
        else:
            kbt = temperature * units.BOLTZMANN
            occ = FD(self.energies, fermi_level, kbt)

        wdos = self.tdos * occ
        num_electrons = wdos.sum() * self.de
        conc = (num_electrons - self.nelect) / self._conv

        if return_electron_hole_conc:
            cb_conc = wdos[self.energies > self.efermi].sum() * self.de
            vb_conc = wdos[self.energies <= self.efermi].sum() * self.de
            cb_conc = cb_conc / self._conv
            vb_conc = (self.nelect - vb_conc) / self._conv
            return conc, cb_conc, vb_conc

        else:
            return conc

    def get_fermi(self,
                  concentration: float,
                  temperature: float,
                  tol: float = 0.01,
                  nstep: int = 50,
                  step: float = 0.1,
                  precision: int = 10,
                  return_electron_hole_conc=False):
        """
        Finds the fermi level at which the doping concentration at the given
        temperature (T) is equal to concentration. A greedy algorithm is used
        where the relative error is minimized by calculating the doping at a
        grid which continually becomes finer.

        Args:
            concentration: The doping concentration in 1/cm^3. Negative values
                represent n-type doping and positive values represent p-type
                doping.
            temperature: The temperature in Kelvin.
            return_electron_hole_conc: Whether to also return the separate
                electron and hole concentrations at the doping level.

        Returns:
            If return_electron_hole_conc is False: The Fermi level in eV. Note
            that this is different from the default dos.efermi.

            If return_electron_hole_conc is True: the Fermi level, electron
            concentration and hole concentration at the Fermi level as a tuple.
            The electron and hole concentrations are in cm^-3.

        """
        fermi = self.efermi  # initialize target fermi
        relative_error = float("inf")
        for _ in range(precision):
            frange = np.arange(-nstep, nstep + 1) * step + fermi
            calc_doping = np.array([self.get_doping(f, temperature)
                                    for f in frange])
            relative_error = abs(calc_doping / concentration - 1.0)
            fermi = frange[np.argmin(relative_error)]
            step /= 10.0

        if min(relative_error) > tol:
            raise ValueError(
                "Could not find fermi within {}% of concentration={}".format(
                    tol * 100, concentration))

        if return_electron_hole_conc:
            _, n_elec, n_hole = self.get_doping(
                fermi, temperature, return_electron_hole_conc=True)
            return fermi, n_elec, n_hole
        else:
            return fermi
