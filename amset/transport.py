import logging
import time
from typing import Union, List

import numpy as np
from BoltzTraP2 import units
from BoltzTraP2.bandlib import fermiintegrals, calc_Onsager_coefficients, DOS, \
    lambda_to_tau
from monty.json import MSONable

from amset.dos import ADOS
from amset.misc.constants import e
from amset.data import AmsetData
from amset.misc.log import log_time_taken

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"
__date__ = "June 21, 2019"

logger = logging.getLogger(__name__)


class TransportCalculator(MSONable):

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
        sigma, seebeck, kappa = _calculate_transport_properties(amset_data)
        log_time_taken(t0)

        if not self.calculate_mobility:
            return sigma, seebeck, kappa, None

        if amset_data.is_metal:
            logger.info("System is metallic, refusing to calculate carrier "
                        "mobility")
            return sigma, seebeck, kappa, None

        logger.info("Calculating overall mobility")
        t0 = time.perf_counter()
        mobility = {"overall": _calculate_mobility(
            amset_data, list(range(len(amset_data.scattering_labels))))}
        log_time_taken(t0)

        if self.separate_scattering_mobilities:
            logger.info("Calculating individual scattering rate mobilities")
            t0 = time.perf_counter()
            for rate_idx, name in enumerate(amset_data.scattering_labels):
                mobility[name] = _calculate_mobility(amset_data, rate_idx)
            log_time_taken(t0)

        return sigma, seebeck, kappa, mobility


def _calculate_mobility(amset_data: AmsetData,
                        rate_idx: Union[int, List[int], np.ndarray]):
    if isinstance(rate_idx, int):
        rate_idx = [rate_idx]

    n_t_size = (len(amset_data.doping), len(amset_data.temperatures))
    all_rates = amset_data.scattering_rates
    all_vv = amset_data.velocities_product
    all_energies = amset_data.energies
    kmask = amset_data.transport_mask

    mobility = np.zeros(n_t_size + (3, 3))
    for n, t in np.ndindex(n_t_size):
        energies = []
        vv = []
        rates = []
        for spin in amset_data.spins:
            cb_idx = amset_data.vb_idx[spin] + 1
            if amset_data.doping[n] > 0:
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

        # print("equiv_k", amset_data.full_kpoints[equiv_points])
        # print("lifetimes", lifetimes[0, equiv_points])
        # print("weights", amset_data.kpoint_weights[equiv_points])
        # print("velocities", vv[0, :, :, equiv_points])
        # equiv_points = amset_data.grouped_ir_to_full[-10]
        # print("equiv_k", amset_data.full_kpoints[equiv_points])
        # print("lifetimes", lifetimes[0, equiv_points])
        # print("weights", amset_data.kpoint_weights[equiv_points])
        # print("velocities", vv[0, :, :, equiv_points])

        # equiv_points = amset_data.grouped_ir_to_full[-1]
        # print("equiv_k", amset_data.full_kpoints[equiv_points])
        # print("lifetimes", lifetimes[0, equiv_points])
        # print("weights", amset_data.kpoint_weights[equiv_points])

        # mask data to remove extra k-points that are not part of the regular
        # k-point mesh (necessary as BoltzTraP2 doesn't support custom k-point
        # weights.
        # energies = energies[:, kmask]
        # vv = vv[..., kmask]
        # lifetimes = lifetimes[:, kmask]

        # Nones are required as BoltzTraP2 expects the Fermi and temp as arrays
        fermi = amset_data.fermi_levels[n, t][None]
        temp = amset_data.temperatures[t][None]

        # obtain the Fermi integrals for the temperature and doping
        # epsilon, dos, vvdos, cdos = BTPDOS(
        #     energies, vv, scattering_model=lifetimes,
        #     npts=len(amset_data.dos.energies))
        epsilon, dos, vvdos, cdos = AMSETDOS(
            energies, vv, scattering_model=lifetimes,
            npts=len(amset_data.dos.energies),
            kpoint_weights=amset_data.kpoint_weights)

        _, l0, l1, l2, lm11 = fermiintegrals(
            epsilon, dos, vvdos, mur=fermi, Tr=temp,
            dosweight=amset_data.dos.dos_weight)

        # Compute the Onsager coefficients from Fermi integrals
        volume = (amset_data.structure.lattice.volume * units.Angstrom ** 3)
        sigma, _, _, _ = calc_Onsager_coefficients(
            l0, l1, l2, fermi, temp, volume)

        if amset_data.doping[n] > 0:
            carrier_conc = amset_data.electron_conc[n, t]
        else:
            carrier_conc = amset_data.hole_conc[n, t]

        # convert mobility to cm^2/V.s
        mobility[n, t] = sigma[0, ...] * 0.01 / (e * carrier_conc)

    return mobility


def _calculate_transport_properties(amset_data):
    kmask = amset_data.transport_mask
    energies = np.vstack([amset_data.energies[spin]
                          for spin in amset_data.spins])
    vv = np.vstack([amset_data.velocities_product[spin]
                    for spin in amset_data.spins])

    # mask data to remove extra k-points that are not part of the regular
    # k-point mesh (necessary as BoltzTraP2 doesn't support custom k-point
    # weights.
    # energies = energies[:, kmask]
    # vv = vv[..., kmask]

    n_t_size = (len(amset_data.doping), len(amset_data.temperatures))

    sigma = np.zeros(n_t_size + (3, 3))
    seebeck = np.zeros(n_t_size + (3, 3))
    kappa = np.zeros(n_t_size + (3, 3))

    # solve sigma, seebeck, kappa and hall using information from all bands
    for n, t in np.ndindex(n_t_size):
        sum_rates = [np.sum(amset_data.scattering_rates[s][:, n, t], axis=0)
                     for s in amset_data.spins]
        lifetimes = 1 / np.vstack(sum_rates)
        # lifetimes = lifetimes[:, kmask]
        # print(lifetimes.min())

        # Nones are required as BoltzTraP2 expects the Fermi and temp as arrays
        fermi = amset_data.fermi_levels[n, t][None]
        temp = amset_data.temperatures[t][None]

        # obtain the Fermi integrals
        # epsilon, dos, vvdos, cdos = BTPDOS(
        #     energies, vv, scattering_model=lifetimes,
        #     npts=len(amset_data.dos.energies))
        epsilon, dos, vvdos, cdos = AMSETDOS(
            energies, vv, scattering_model=lifetimes,
            npts=len(amset_data.dos.energies),
            kpoint_weights=amset_data.kpoint_weights)

        _, l0, l1, l2, lm11 = fermiintegrals(
            epsilon, dos, vvdos, mur=fermi, Tr=temp,
            dosweight=amset_data.dos.dos_weight)

        volume = (amset_data.structure.lattice.volume * units.Angstrom ** 3)

        # Compute the Onsager coefficients from Fermi integrals
        # Don't store the Hall coefficient as we don't have the curvature
        # information.
        # TODO: Fix Hall coefficient
        sigma[n, t], seebeck[n, t], kappa[n, t], _ = \
            calc_Onsager_coefficients(l0, l1, l2, fermi, temp, volume)

    # convert seebeck to µV/K
    seebeck *= 1e6

    return sigma, seebeck, kappa


def AMSETDOS(eband,
             vvband,
             cband=None,
             erange=None,
             npts=None,
             scattering_model="uniform_tau",
             kpoint_weights=None,
             ):
    """Compute the DOS, transport DOS and "curvature DOS".

    The transport DOS is weighted by the outer product of the group velocity
    with itself, and by the relaxation time. The "curvature DOS" is weighted by
    the curvature and the relaxation time.

    In order to use a custom mean free path instead of a custom lifetime, use
    the auxiliary function lambda_to_tau().

    Args:
        eband: (nbands, nkpoints) array with the band energies
        vvband: (nbands, 3, 3, nkpoints) array with the outer product of each
            group velocity with itself.
        cband: None or (nbands, 3, 3, 3, nkpoints) array with the curvatures
        erange: range of energies for the DOS. If not provided, it will be
            automatically determined by the DOS function.
        npts: number of bins in the histogram used to determine the DOS. If not
            provided, the DOS function will take a conservative guess.
        scattering_model: model to be used for the electron lifetimes. The
            following choices are available:

                - "uniform_tau": uniform lifetime for all carriers
                - "uniform_lambda": uniform mean free path for all carriers
                - A 2d array with the same shape as eband, with a scattering
                  rate for each electron mode.

    Returns:
        Four arrays containing the bin energies, the DOS, the transport dos and
        the "curvature DOS". If cband is none, the last element of the returned
        value will also be none. The sizes of the returned arrays are (npts, ),
        (npts,), (3, 3, npts) and (3, 3, 3, npts).
    """
    # kpoint_weights = kpoint_weights * len(kpoint_weights)
    kpoint_weights = np.tile(kpoint_weights, (len(eband), 1))
    dos = ADOS(eband.T, erange=erange, npts=npts, weights=kpoint_weights.T)
    # dos = ADOS(eband.T, erange=erange, npts=npts)
    npts = dos[0].size
    iu0 = np.array(np.triu_indices(3)).T
    vvdos = np.zeros((3, 3, npts))
    multpl = np.ones_like(eband)

    if isinstance(scattering_model, str) and scattering_model == "uniform_tau":
        pass

    elif isinstance(scattering_model,
                    str) and scattering_model == "uniform_lambda":
        multpl = lambda_to_tau(vvband, multpl)

    elif isinstance(scattering_model, np.ndarray):
        if scattering_model.shape != eband.shape:
            raise ValueError(
                "scattering_model and ebands must have the same shape")

        multpl = scattering_model

    else:
        raise ValueError("unknown scattering model")

    for i, j in iu0:
        weights = vvband[:, i, j, :] * multpl * kpoint_weights
        vvdos[i, j] = ADOS(
            eband.T, weights=weights.T, erange=erange, npts=npts)[1]
    il1 = np.tril_indices(3, -1)
    iu1 = np.triu_indices(3, 1)
    vvdos[il1[0], il1[1]] = vvdos[iu1[0], iu1[1]]

    if cband is None:
        cdos = None
    else:
        cdos = np.zeros((3, 3, 3, npts))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    weights = cband[:, i, j, k, :] * multpl * multpl
                    cdos[i, j, k] = DOS(
                        eband.T, weights=weights.T, erange=erange,
                        npts=npts)[1]
    return dos[0], dos[1], vvdos, cdos


