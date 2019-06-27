"""
Customised implementation of FermiDos. Will Move back to pymatgen at some point.
"""
import warnings

import numpy as np

from typing import Optional, Union, Tuple

from monty.json import MSONable

from pymatgen import Structure, Spin
from pymatgen.electronic_structure.dos import Dos, f0


class FermiDos(Dos, MSONable):
    """
    This wrapper class helps relate the density of states, doping levels
    (i.e. carrier concentrations) and corresponding fermi levels. A negative
    doping concentration indicates the majority carriers are electrons
    (n-type doping); a positive doping concentration indicates holes are the
    majority carriers (p-type doping).

    Args:
        dos: Pymatgen Dos object.
        structure: A structure. If not provided, the structure
            of the dos object will be used. If the dos does not have an
            associated structure object, an error will be thrown.
        dos_weight: The weighting for the dos. Defaults to 2 for non-spin
            polarized calculations and 1 for spin-polarized calculations.
        nelecs: The number of electrons included in the energy range of
            dos used for normalizing the densities. If None, the densities
            will not be normalized.
        bandgap: If set, the energy values are scissored so that the electronic
            band gap matches this value.
    """

    def __init__(self, dos: Dos, structure: Structure = None,
                 dos_weight: Optional[float] = None,
                 nelecs: Optional[float] = None,
                 bandgap: Optional[float] = None):
        super().__init__(dos.efermi, energies=dos.energies,
                         densities={k: np.array(d) for k, d in
                                    dos.densities.items()})

        if structure is None:
            if hasattr(dos, "structure"):
                structure = dos.structure
            else:
                raise ValueError("Structure object is not provided and not "
                                 "present in dos")

        self.structure = structure
        self.nelecs = nelecs or self.structure.composition.total_electrons

        self.volume = self.structure.volume
        self.energies = np.array(dos.energies)
        self.de = np.hstack(
            (self.energies[1:], self.energies[-1])) - self.energies

        if not dos_weight:
            dos_weight = 2 if len(self.densities) == 1 else 1
        self.dos_weight = dos_weight

        self.tdos = np.array(self.get_densities()) * self.dos_weight

        nelect_integ = (self.tdos * self.de)[self.energies <= self.efermi].sum()
        if nelecs:
            # normalize total density of states based on integral at 0K
            self.tdos = self.tdos * self.nelecs / nelect_integ
        else:
            self.nelecs = nelect_integ

        self.idx_fermi = np.argmin(abs(self.energies - self.efermi))

        if bandgap:
            ecbm, evbm = self.get_cbm_vbm()
            if evbm < self.efermi < ecbm:
                eref = self.efermi
            else:
                eref = (evbm + ecbm) / 2.0

            idx_fermi = np.argmin(abs(self.energies - eref))

            self.energies[:idx_fermi] -= (bandgap - (ecbm - evbm)) / 2.0
            self.energies[idx_fermi:] += (bandgap - (ecbm - evbm)) / 2.0

    def get_doping(self, fermi_level: float, temperature: float,
                   return_electron_hole_conc: bool = False
                   ) -> Union[float, Tuple[float, float, float]]:
        """
        Calculate the doping (majority carrier concentration) at a given
        fermi level  and temperature. A simple Left Riemann sum is used for
        integrating the density of states over energy & equilibrium Fermi-Dirac
        distribution.

        Args:
            fermi_level: The fermi_level level in eV.
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
        cb_integral = np.sum(
            self.tdos[self.idx_fermi:]
            * f0(self.energies[self.idx_fermi:], fermi_level, temperature)
            * self.de[self.idx_fermi:], axis=0)
        vb_integral = np.sum(
            self.tdos[:self.idx_fermi] *
            (1 - f0(self.energies[:self.idx_fermi], fermi_level, temperature))
            * self.de[:self.idx_fermi], axis=0)

        # 1e-8 is conversion from Angstrom to cm
        conv = self.volume * 1e-8 ** 3
        doping = (vb_integral - cb_integral) / conv

        if return_electron_hole_conc:
            return doping, cb_integral / conv, vb_integral / conv
        else:
            return doping

    def get_fermi_interextrapolated(self, concentration: float,
                                    temperature: float, warn: bool = True,
                                    c_ref: float = 1e10, **kwargs) -> float:
        """
        Similar to get_fermi except that when get_fermi fails to converge,
        an interpolated or extrapolated fermi is returned with the assumption
        that the fermi level changes linearly with log(abs(concentration)).

        Args:
            concentration: The doping concentration in 1/cm^3. Negative values
                represent n-type doping and positive values represent p-type
                doping.
            temperature: The temperature in Kelvin.
            warn: Whether to give a warning the first time the fermi cannot be
                found.
            c_ref: A doping concentration where get_fermi returns a
                value without error for both c_ref and -c_ref.
            **kwargs: Keyword arguments passed to the get_fermi function.

        Returns:
            The Fermi level. Note, the value is possibly interpolated or
            extrapolated and must be used with caution.
        """
        try:
            return self.get_fermi(concentration, temperature, **kwargs)
        except ValueError as e:
            if warn:
                warnings.warn(str(e))

            if abs(concentration) < c_ref:
                if abs(concentration) < 1e-10:
                    concentration = 1e-10

                # max(10, ) is to avoid log(0<x<1) and log(1+x) both of which
                # are slow
                f2 = self.get_fermi_interextrapolated(
                    max(10, abs(concentration) * 10.), temperature, warn=False,
                    **kwargs)
                f1 = self.get_fermi_interextrapolated(
                    -max(10, abs(concentration) * 10.), temperature, warn=False,
                    **kwargs)
                c2 = np.log(abs(1 + self.get_doping(f2, temperature)))
                c1 = -np.log(abs(1 + self.get_doping(f1, temperature)))
                slope = (f2 - f1) / (c2 - c1)
                return f2 + slope * (np.sign(concentration) *
                                     np.log(abs(1 + concentration)) - c2)

            else:
                f_ref = self.get_fermi_interextrapolated(
                    np.sign(concentration) * c_ref, temperature, warn=False,
                    **kwargs)
                f_new = self.get_fermi_interextrapolated(
                    concentration / 10., temperature, warn=False, **kwargs)
                clog = np.sign(concentration) * np.log(abs(concentration))
                c_newlog = np.sign(concentration) * np.log(
                    abs(self.get_doping(f_new, temperature)))
                slope = (f_new - f_ref) / (c_newlog - np.sign(concentration)
                                           * 10.)
                return f_new + slope * (clog - c_newlog)

    def get_fermi(self, concentration: float, temperature: float,
                  rtol: float = 0.01, nstep: int = 50, step: float = 0.1,
                  precision: int = 8, return_electron_hole_conc=False):
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
            rtol: The maximum acceptable relative error.
            nstep: THe number of steps checked around a given fermi level.
            step: Initial step in energy when searching for the Fermi level.
            precision: Essentially the decimal places of calculated Fermi level.

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

        if min(relative_error) > rtol:
            raise ValueError(
                "Could not find fermi within {}% of concentration={}".format(
                    rtol * 100, concentration))

        if return_electron_hole_conc:
            _, n_elec, n_hole = self.get_doping(
                fermi, temperature, return_electron_hole_conc=True)
            return fermi, n_elec, n_hole
        else:
            return fermi

    @classmethod
    def from_dict(cls, d):
        """
        Returns Dos object from dict representation of Dos.
        """
        dos = Dos(d["efermi"], d["energies"],
                  {Spin(int(k)): v for k, v in d["densities"].items()})
        return FermiDos(dos, structure=Structure.from_dict(d["structure"]),
                        nelecs=d["nelecs"], dos_weight=d["dos_weight"])

    def as_dict(self):
        """
        Json-serializable dict representation of Dos.
        """
        return {"@module": self.__class__.__module__,
                "@class": self.__class__.__name__, "efermi": self.efermi,
                "energies": list(self.energies),
                "densities": {str(spin): list(dens)
                              for spin, dens in self.densities.items()},
                "structure": self.structure, "nelecs": self.nelecs,
                "dos_weight": self.dos_weight}
