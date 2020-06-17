import logging
import time
from typing import List, Union

import numpy as np
from BoltzTraP2.bandlib import calc_Onsager_coefficients

from amset.constants import bohr_to_cm, defaults, e
from amset.core.data import AmsetData
from amset.electronic_structure.boltztrap import fermiintegrals
from amset.log import log_time_taken
from amset.util import get_progress_bar

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

logger = logging.getLogger(__name__)

_e_str = "Electronic structure must contain dopings temperatures and scattering rates"


def solve_boltzman_transport_equation(
    amset_data: AmsetData,
    calculate_mobility: bool = defaults["calculate_mobility"],
    separate_mobility: bool = defaults["separate_mobility"],
    progress_bar: bool = defaults["print_log"],
):
    has_doping = amset_data.doping is not None
    has_temps = amset_data.temperatures is not None
    has_rates = amset_data.scattering_rates is not None
    if not (has_doping and has_temps and has_rates):
        raise ValueError(_e_str)

    logger.info(
        "Calculating conductivity, Seebeck, and electronic thermal conductivity"
    )
    t0 = time.perf_counter()
    sigma, seebeck, kappa = _calculate_transport_properties(
        amset_data, progress_bar=progress_bar
    )
    log_time_taken(t0)

    if not calculate_mobility:
        return sigma, seebeck, kappa, None

    if amset_data.is_metal:
        logger.info("System is metallic, refusing to calculate carrier mobility")
        return sigma, seebeck, kappa, None

    n_scats = len(amset_data.scattering_labels)

    logger.info("Calculating overall mobility")
    t0 = time.perf_counter()
    overall = _calculate_mobility(
        amset_data, np.arange(n_scats), pbar_label="mobility" if progress_bar else None
    )
    mobility = {"overall": overall}
    log_time_taken(t0)

    if separate_mobility:
        logger.info("Calculating individual scattering rate mobilities")
        t0 = time.perf_counter()
        for rate_idx, name in enumerate(amset_data.scattering_labels):
            mobility[name] = _calculate_mobility(
                amset_data, rate_idx, pbar_label=name if progress_bar else None
            )
        log_time_taken(t0)

    return sigma, seebeck, kappa, mobility


def _calculate_mobility(
    amset_data: AmsetData,
    rate_idx: Union[int, List[int], np.ndarray],
    pbar_label: str = "mobility",
):
    if isinstance(rate_idx, int):
        rate_idx = [rate_idx]

    volume = amset_data.structure.volume
    mobility = np.zeros(amset_data.fermi_levels.shape + (3, 3))

    epsilon, dos = amset_data.tetrahedral_band_structure.get_density_of_states(
        amset_data.dos.energies, sum_spins=True, use_cached_weights=True
    )

    if pbar_label is not None:
        pbar = get_progress_bar(
            iterable=list(np.ndindex(amset_data.fermi_levels.shape)), desc=pbar_label
        )
    else:
        pbar = list(np.ndindex(amset_data.fermi_levels.shape))

    for n, t in pbar:
        br = {s: np.arange(len(amset_data.energies[s])) for s in amset_data.spins}
        cb_idx = {s: amset_data.vb_idx[s] + 1 for s in amset_data.spins}

        if amset_data.doping[n] < 0:
            band_idx = {s: br[s][cb_idx[s] :] for s in amset_data.spins}
        else:
            band_idx = {s: br[s][: cb_idx[s]] for s in amset_data.spins}

        lifetimes = {
            s: 1 / np.sum(amset_data.scattering_rates[s][rate_idx, n, t], axis=0)
            for s in amset_data.spins
        }

        # Nones are required as BoltzTraP2 expects the Fermi and temp as arrays
        fermi = amset_data.fermi_levels[n, t][None]
        temp = amset_data.temperatures[t][None]

        # obtain the Fermi integrals for the temperature and doping
        vvdos = get_transport_dos(
            amset_data.tetrahedral_band_structure,
            amset_data.velocities_product,
            lifetimes,
            amset_data.dos.energies,
            band_idx=band_idx,
        )

        c, l0, l1, l2, lm11 = fermiintegrals(
            epsilon, dos, vvdos, mur=fermi, Tr=temp, dosweight=amset_data.dos.dos_weight
        )

        # Compute the Onsager coefficients from Fermi integrals
        sigma, _, _, _ = calc_Onsager_coefficients(l0, l1, l2, fermi, temp, volume)

        if amset_data.doping[n] < 0:
            carrier_conc = amset_data.electron_conc[n, t]
        else:
            carrier_conc = amset_data.hole_conc[n, t]

        # don't use c as we don't use the correct DOS each time
        # c = -c[0, ...] / (volume / (Meter / 100.)**3)

        # convert mobility to cm^2/V.s
        uc = 0.01 / (e * carrier_conc * (1 / bohr_to_cm) ** 3)
        mobility[n, t] = sigma[0, ...] * uc

    return mobility


def _calculate_transport_properties(
    amset_data, progress_bar: bool = defaults["print_log"]
):
    n_t_size = (len(amset_data.doping), len(amset_data.temperatures))

    sigma = np.zeros(n_t_size + (3, 3))
    seebeck = np.zeros(n_t_size + (3, 3))
    kappa = np.zeros(n_t_size + (3, 3))
    volume = amset_data.structure.volume

    epsilon, dos = amset_data.tetrahedral_band_structure.get_density_of_states(
        amset_data.dos.energies, sum_spins=True, use_cached_weights=True
    )

    iterable = list(np.ndindex(n_t_size))
    if progress_bar:
        pbar = get_progress_bar(iterable=iterable, desc="transport")
    else:
        pbar = iterable

    # solve sigma, seebeck, kappa and hall using information from all bands
    for n, t in pbar:
        lifetimes = {
            s: 1 / np.sum(amset_data.scattering_rates[s][:, n, t], axis=0)
            for s in amset_data.spins
        }

        # Nones are required as BoltzTraP2 expects the Fermi and temp as arrays
        fermi = amset_data.fermi_levels[n, t][None]
        temp = amset_data.temperatures[t][None]

        # obtain the Fermi integrals
        vvdos = get_transport_dos(
            amset_data.tetrahedral_band_structure,
            amset_data.velocities_product,
            lifetimes,
            amset_data.dos.energies,
        )

        _, l0, l1, l2, lm11 = fermiintegrals(
            epsilon, dos, vvdos, mur=fermi, Tr=temp, dosweight=amset_data.dos.dos_weight
        )

        # Compute the Onsager coefficients from Fermi integrals
        # Don't store the Hall coefficient as we don't have the curvature
        # information.
        sigma[n, t], seebeck[n, t], kappa[n, t], _ = calc_Onsager_coefficients(
            l0, l1, l2, fermi, temp, volume
        )

    # convert seebeck to µV/K
    seebeck *= 1e6

    return sigma, seebeck, kappa


def get_transport_dos(
    tetrahedron_band_structure, vvband, lifetimes, energies, band_idx=None
):
    """Compute the transport DOS

    The transport DOS is weighted by the outer product of the group velocity
    with itself, and by the relaxation time.

    Args:
        vvband: (nbands, 3, 3, nkpoints) array with the outer product of each
            group velocity with itself.

    Returns:
        The transport dos with the same (3, 3, npts).
    """
    # vvband is nb, 3, 3, nk it should be nb, nk, 3, 3
    vvband = {s: v.transpose(0, 3, 1, 2) for s, v in vvband.items()}
    weights = {s: vvband[s] * lifetimes[s][:, :, None, None] for s in lifetimes}

    _, vvdos = tetrahedron_band_structure.get_density_of_states(
        energies,
        integrand=weights,
        sum_spins=True,
        band_idx=band_idx,
        use_cached_weights=True,
    )

    # vvdos is npts, 3, 3 it should be 3, 3, npts
    vvdos = vvdos.transpose(1, 2, 0)

    return vvdos
