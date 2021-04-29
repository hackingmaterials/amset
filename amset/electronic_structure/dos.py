"""
Customised implementation of FermiDos. Will Move back to pymatgen at some point.
"""
import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.dos import Dos

from amset.constants import boltzmann_au, ev_to_hartree, hartree_to_ev
from amset.electronic_structure.fd import fd

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

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
        atomic_units: Whether energies are given in eV or Hartree.
        num_electrons: The number of electrons in the system. If None, this will be
            calculated by integrating up to the intrinsic Fermi level.
    """

    def __init__(
        self,
        efermi: float,
        energies: np.ndarray,
        densities: Dict[Spin, np.ndarray],
        structure: Structure,
        dos_weight: Optional[float] = None,
        atomic_units: bool = True,
        num_electrons: Optional[float] = None,
    ):
        # structure should be atomic structure
        super().__init__(efermi, energies, densities)
        self.structure = structure
        self.atomic_units = atomic_units

        if not dos_weight:
            dos_weight = 2 if len(self.densities) == 1 else 1

        self.dos_weight = dos_weight
        self.tdos = np.array(self.get_densities()) * self.dos_weight
        self.de = self.energies[1] - self.energies[0]
        self._num_electrons = num_electrons  # this is just for msonability

        if num_electrons is None:
            # integrate up to Fermi level to get number of electrons
            self.nelect = self.tdos[self.energies <= self.efermi].sum() * self.de
        else:
            self.nelect = num_electrons

        logger.info(
            "Intrinsic DOS Fermi level: {:.4f} eV".format(
                self.efermi * hartree_to_ev if atomic_units else self.efermi
            )
        )
        logger.info(f"DOS contains {self.nelect:.3f} electrons")

    def get_doping(
        self,
        fermi_level: float,
        temperature: float,
        return_electron_hole_conc: bool = False,
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
            units of 1/Bohr^3. Negative values indicate that the majority carriers
            are electrons (n-type doping) whereas positive values indicates the
            majority carriers are holes (p-type doping).

            If return_electron_hole_conc is True: the doping concentration,
            electron concentration and hole concentration as a tuple.
        """
        wdos = _get_weighted_dos(
            self.energies,
            self.tdos,
            fermi_level,
            temperature,
            atomic_units=self.atomic_units,
        )

        num_electrons = wdos.sum() * self.de
        conc = (self.nelect - num_electrons) / self.structure.volume

        if return_electron_hole_conc:
            cb_conc = wdos[self.energies > self.efermi].sum() * self.de
            vb_conc = wdos[self.energies <= self.efermi].sum() * self.de
            cb_conc = cb_conc / self.structure.volume
            vb_conc = (self.nelect - vb_conc) / self.structure.volume
            return conc, cb_conc, vb_conc

        else:
            return conc

    def get_num_electrons(self, fermi_level: float, temperature: float) -> float:
        """
        Calculate the number of electrons at a given fermi level and temperature.
        A simple Left Riemann sum is used for integrating the density of states over
        energy & equilibrium Fermi-Dirac distribution.

        Args:
            fermi_level: The fermi_level level in Hartree.
            temperature: The temperature in Kelvin.

        Returns:
            The number of electrons.
        """
        wdos = _get_weighted_dos(
            self.energies,
            self.tdos,
            fermi_level,
            temperature,
            atomic_units=self.atomic_units,
        )

        num_electrons = wdos.sum() * self.de
        return num_electrons

    def get_fermi_from_num_electrons(
        self,
        num_electrons: float,
        temperature: float,
        tol: float = 0.01,
        nstep: int = 50,
        step: float = 0.1,
        precision: int = 10,
    ):
        # this is finding the Fermi level of metals
        fermi = self.efermi  # initialize target fermi
        relative_error = float("inf")
        for _ in range(precision):
            frange = np.arange(-nstep, nstep + 1) * step + fermi
            calc_nelectrons = [self.get_num_electrons(f, temperature) for f in frange]
            relative_error = abs(np.array(calc_nelectrons) / num_electrons - 1.0)
            fermi = frange[np.argmin(relative_error)]
            step /= 10.0

        if min(relative_error) > tol:
            raise ValueError(
                "Could not find fermi within {}% of num electrons={}".format(
                    tol * 100, num_electrons
                )
            )

        return fermi

    def get_fermi(
        self,
        concentration: float,
        temperature: float,
        tol: float = 0.01,
        nstep: int = 50,
        step: float = 0.1,
        precision: int = 10,
        return_electron_hole_conc=False,
    ):
        """
        Finds the fermi level at which the doping concentration at the given
        temperature (T) is equal to concentration. A greedy algorithm is used
        where the relative error is minimized by calculating the doping at a
        grid which continually becomes finer.

        Args:
            concentration: The doping concentration in 1/Bohr^3. Negative values
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
            The electron and hole concentrations are in Bohr^-3.
        """
        fermi = self.efermi  # initialize target fermi
        relative_error = float("inf")
        for _ in range(precision):
            frange = np.arange(-nstep, nstep + 1) * step + fermi
            calc_doping = np.array([self.get_doping(f, temperature) for f in frange])
            relative_error = abs(calc_doping / concentration - 1.0)
            fermi = frange[np.argmin(relative_error)]
            step /= 10.0

        if min(relative_error) > tol:
            raise ValueError(
                "Could not find fermi within {}% of concentration={}".format(
                    tol * 100, concentration
                )
            )

        if return_electron_hole_conc:
            _, n_elec, n_hole = self.get_doping(
                fermi, temperature, return_electron_hole_conc=True
            )
            return fermi, n_elec, n_hole
        else:
            return fermi


def _get_weighted_dos(energies, dos, fermi_level, temperature, atomic_units=True):
    if temperature == 0.0:
        occ = np.where(energies < fermi_level, 1.0, 0.0)
        occ[energies == fermi_level] = 0.5
    else:
        kbt = temperature * boltzmann_au
        if atomic_units:
            occ = fd(energies, fermi_level, kbt)
        else:
            occ = fd(energies * ev_to_hartree, fermi_level * ev_to_hartree, kbt)

    wdos = dos * occ
    return wdos
