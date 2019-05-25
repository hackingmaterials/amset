import logging
import time
from typing import Union, List

import numpy as np
from BoltzTraP2 import units
from BoltzTraP2.bandlib import fermiintegrals, BTPDOS, calc_Onsager_coefficients
from monty.json import MSONable

from amset.constants import e
from amset.data import AmsetData
from amset.util import log_time_taken

logger = logging.getLogger(__name__)


class BTESolver(MSONable):

    def __init__(self,
                 calculate_mobility: bool = True,
                 separate_scattering_mobilities: bool = False):
        self.separate_scattering_mobilities = separate_scattering_mobilities
        self.calculate_mobility = calculate_mobility

    def solve_bte(self, amset_data: AmsetData):
        if not all([amset_data.doping is not None,
                    amset_data.temperatures is not None,
                    amset_data.scattering_rates is not None]):
            raise ValueError("Electronic structure must contain dopings "
                             "temperatures and scattering rates")

        logger.info("Calculating conductivity, Seebeck, and electronic thermal "
                    "conductivity tensors.")
        t0 = time.perf_counter()
        sigma, seebeck, kappa = _calculate_transport_properties(
            amset_data)
        log_time_taken(t0)

        if not self.calculate_mobility:
            return sigma, seebeck, kappa

        if amset_data.is_metal:
            logger.info("System is metallic, refusing to calculate carrier "
                        "mobility")
            return sigma, seebeck, kappa, None

        logger.info("Calculating overall mobility")
        t0 = time.perf_counter()
        mobility = {"overall": _calculate_mobility(
            amset_data,
            list(range(len(amset_data.scattering_labels))))}
        log_time_taken(t0)

        if self.separate_scattering_mobilities:
            logger.info("Calculating individual scattering rate mobilities")
            t0 = time.perf_counter()
            for rate_idx, name in enumerate(
                    amset_data.scattering_labels):
                mobility[name] = _calculate_mobility(
                    amset_data, rate_idx)
            log_time_taken(t0)

        return sigma, seebeck, kappa, mobility


def _calculate_mobility(amset_data: AmsetData,
                        rate_idx: Union[int, List[int], np.ndarray]):
    if isinstance(rate_idx, int):
        rate_idx = [rate_idx]

    n_t_size = (len(amset_data.doping),
                len(amset_data.temperatures))
    all_rates = amset_data.scattering_rates
    all_vv = amset_data.velocities_product
    all_energies = amset_data.energies

    mobility = np.zeros(n_t_size + (3, 3))
    for n, t in np.ndindex(n_t_size):
        energies = []
        vv = []
        rates = []
        for spin in amset_data.spins:
            cb_idx = amset_data.vb_idx[spin] + 1
            if n > 0:
                # electrons
                energies.append(all_energies[spin][cb_idx:])
                vv.append(all_vv[spin][cb_idx:])
                rates.append(np.sum(all_rates[spin][rate_idx, n, t, cb_idx:],
                                    axis=0))
            else:
                # holes
                energies.append(all_energies[spin][:cb_idx])
                vv.append(all_vv[spin][:cb_idx])
                rates.append(np.sum(all_rates[spin][rate_idx, n, t, :cb_idx],
                                    axis=0))

        energies = np.vstack(energies)
        vv = np.vstack(vv)
        lifetimes = 1 / np.vstack(rates)

        # Nones are required as BoltzTraP2 expects the Fermi and temp as arrays
        fermi = amset_data.fermi_levels[n, t][None] * units.eV
        temp = amset_data.temperatures[t][None]

        # obtain the Fermi integrals the temperature and doping
        epsilon, dos, vvdos, cdos = BTPDOS(
            energies, vv, scattering_model=lifetimes,
            npts=len(amset_data.dos.energies))

        carriers, l0, l1, l2, lm11 = fermiintegrals(
            epsilon, dos, vvdos, mur=fermi, Tr=temp,
            dosweight=amset_data.dos_weight)

        volume = (amset_data.structure.lattice.volume *
                  units.Angstrom ** 3)

        # Rescale the carrier count into a volumetric density in cm**(-3)
        carriers = ((-carriers[0, ...] - amset_data.dos.nelecs) /
                    (volume / (units.Meter / 100.) ** 3))

        # Compute the Onsager coefficients from Fermi integrals
        sigma, _, _, _ = calc_Onsager_coefficients(
            l0, l1, l2, amset_data.doping[[n]],
            amset_data.temperatures[[t]], volume)

        # convert mobility to cm^2/V.s
        mobility[n, t] = (sigma[0, ...] * 0.01 / (e * carriers[0]))

    return mobility


def _calculate_transport_properties(amset_data):
    energies = np.vstack([amset_data.energies[spin]
                          for spin in amset_data.spins])
    vv = np.vstack([amset_data.velocities_product[spin]
                    for spin in amset_data.spins])

    n_t_size = (len(amset_data.doping),
                len(amset_data.temperatures))

    sigma = np.zeros(n_t_size + (3, 3))
    seebeck = np.zeros(n_t_size + (3, 3))
    kappa = np.zeros(n_t_size + (3, 3))

    # solve sigma, seebeck, kappa and hall using information from all bands
    for n, t in np.ndindex(n_t_size):
        sum_rates = [np.sum(
            amset_data.scattering_rates[s][:, n, t], axis=0)
            for s in amset_data.spins]
        lifetimes = 1 / np.vstack(sum_rates)

        # Nones are required as BoltzTraP2 expects the Fermi and temp as arrays
        fermi = amset_data.fermi_levels[n, t][None] * units.eV
        temp = amset_data.temperatures[t][None]

        # obtain the Fermi integrals
        epsilon, dos, vvdos, cdos = BTPDOS(
            energies, vv, scattering_model=lifetimes,
            npts=len(amset_data.dos.energies))

        _, l0, l1, l2, lm11 = fermiintegrals(
            epsilon, dos, vvdos, mur=fermi, Tr=temp,
            dosweight=amset_data.dos_weight)

        volume = (amset_data.structure.lattice.volume *
                  units.Angstrom ** 3)

        # Compute the Onsager coefficients from Fermi integrals
        # Don't store the Hall coefficient as we don't have the curvature
        # information.
        # TODO: Fix Hall coefficient
        sigma[n, t], seebeck[n, t], kappa[n, t], _ = \
            calc_Onsager_coefficients(l0, l1, l2, fermi, temp, volume)

        # convert seebeck to µV/K
        seebeck *= 1e6

    return sigma, seebeck, kappa
