import logging
from typing import Union, List

import numpy as np
from BoltzTraP2 import units
from BoltzTraP2.bandlib import fermiintegrals, BTPDOS, calc_Onsager_coefficients
from monty.json import MSONable

from amset.constants import e
from amset.core import ElectronicStructure

logger = logging.getLogger(__name__)


class BTESolver(MSONable):

    def __init__(self,
                 calculate_mobility: bool = True,
                 separate_scattering_mobilities: bool = False):
        self.separate_scattering_mobilities = separate_scattering_mobilities
        self.calculate_mobility = calculate_mobility

    def solve_bte(self, electronic_structure: ElectronicStructure):
        if not all([electronic_structure.doping,
                    electronic_structure.temperatures,
                    electronic_structure.scattering_rates]):
            raise ValueError("Electronic structure must contain dopings "
                             "temperatures and scattering_type rates")

        sigma, seebeck, kappa, hall = _calculate_transport_properties(
            electronic_structure)

        if not self.calculate_mobility:
            return sigma, seebeck, kappa, hall

        if electronic_structure.is_metal:
            logger.info("System is metallic, refusing to calculate carrier "
                        "mobility")
            return sigma, seebeck, kappa, hall, None

        # solve mobility
        mobility = {"total": _calculate_mobility(
            electronic_structure,
            list(range(len(electronic_structure.scattering_labels))))}

        if self.separate_scattering_mobilities:
            for rate_idx, name in enumerate(
                    electronic_structure.scattering_labels):
                mobility[name] = _calculate_mobility(
                    electronic_structure, rate_idx)

        return sigma, seebeck, kappa, hall, mobility


def _calculate_mobility(electronic_structure: ElectronicStructure,
                        rate_idx: Union[int, List[int], np.ndarray]):
    if isinstance(rate_idx, int):
        rate_idx = [rate_idx]

    n_t_size = (len(electronic_structure.doping),
                len(electronic_structure.temperatures))
    all_rates = electronic_structure.scattering_rates
    all_vv = electronic_structure.velocities_product
    all_energies = electronic_structure.energies

    mobility = np.zeros(n_t_size + (3,))
    for n, t in np.ndindex(n_t_size):
        energies = []
        vv = []
        rates = []
        for spin in electronic_structure.spins:
            cb_idx = electronic_structure.vb_idx[spin] + 1
            if n > 0:
                # electrons
                energies.append(all_energies[spin][:cb_idx])
                vv.append(all_vv[spin][:cb_idx])
                rates.append(np.sum(all_rates[spin][rate_idx, :cb_idx], axis=0))
            else:
                # holes
                energies.append(all_energies[spin][cb_idx:])
                vv.append(all_vv[spin][cb_idx:])
                rates.append(np.sum(all_rates[spin][rate_idx, cb_idx:], axis=0))

        energies = np.vstack(energies)
        vv = np.vstack(vv)
        lifetimes = 1 / np.vstack(rates)

        # obtain the Fermi integrals the temperature and doping
        epsilon, dos, vvdos, cdos = BTPDOS(
            energies, vv, scattering_model=lifetimes,
            npts=len(electronic_structure.dos.energies))

        # todo, this properly
        fermi_level = [5.940700 * units.eV]
        temp = np.array([300.])

        carriers, l0, l1, l2, lm11 = fermiintegrals(
            epsilon, dos, vvdos, mur=fermi_level, Tr=temp,
            dosweight=electronic_structure.dos_weight)

        volume = (electronic_structure.structure.lattice.volume *
                  units.Angstrom ** 3)

        # Rescale the carrier count into a volumetric density in cm**(-3)
        carriers = ((-carriers[0, ...] - electronic_structure.dos.nelecs) /
                    (volume / (units.Meter / 100.) ** 3))

        # Compute the Onsager coefficients from those Fermi integrals
        sigma, _, _, _ = calc_Onsager_coefficients(
            l0, l1, l2, fermi_level, temp, volume)

        # todo: Store sigma in correct array
        sigma = sigma[0, ...]
        mobility[n, t] = (sigma * 0.01 / (e * carriers[0]))[0]

    return mobility


def _calculate_transport_properties(electronic_structure):
    energies = np.vstack([electronic_structure.energies[spin]
                          for spin in electronic_structure.spins])
    vv = np.vstack([electronic_structure.velocities_product[spin]
                    for spin in electronic_structure.spins])

    n_t_size = (len(electronic_structure.doping),
                len(electronic_structure.temperatures))

    sigma = np.zeros(n_t_size + (3,))
    seebeck = np.zeros(n_t_size + (3,))
    kappa = np.zeros(n_t_size + (3,))
    hall = np.zeros(n_t_size + (3,))

    # solve sigma, seebeck, kappa and hall using information from all bands
    for n, t in np.ndindex(n_t_size):
        sum_rates = [np.sum(
            electronic_structure.scattering_rates[s][:, n, t], axis=0)
            for s in electronic_structure.spins]
        lifetimes = 1 / np.vstack(sum_rates)

        # obtain the Fermi integrals the temperature and doping
        epsilon, dos, vvdos, cdos = BTPDOS(
            energies, vv, scattering_model=lifetimes,
            npts=len(electronic_structure.dos.energies))

        # todo, this properly
        fermi_level = [5.940700 * units.eV]
        temp = np.array([300.])

        carriers, l0, l1, l2, lm11 = fermiintegrals(
            epsilon, dos, vvdos, mur=fermi_level, Tr=temp,
            dosweight=electronic_structure.dos_weight)

        volume = (electronic_structure.structure.lattice.volume *
                  units.Angstrom ** 3)

        # Rescale the carrier count into a volumetric density in cm**(-3)
        carriers = ((-carriers[0, ...] - electronic_structure.dos.nelecs) /
                    (volume / (units.Meter / 100.) ** 3))

        # Compute the Onsager coefficients from those Fermi integrals
        sigma, seebeck, kappa, hall = calc_Onsager_coefficients(
            l0, l1, l2, fermi_level, temp, volume)

        # todo: Store sigma in correct array

        sigma = sigma[0, ...]
        mobility = sigma * 0.01 / (e * carriers[0])
        print("mobility: {}".format(mobility[0]))

    return sigma, seebeck, kappa, hall
