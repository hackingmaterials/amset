import logging
import sys
import time
from typing import Union, List

import numpy as np
from BoltzTraP2.bandlib import fermiintegrals, calc_Onsager_coefficients
from monty.json import MSONable
from tqdm import tqdm

from amset.constants import e, bohr_to_cm, output_width
from amset.data import AmsetData
from amset.misc.log import log_time_taken

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"
__date__ = "June 21, 2019"

logger = logging.getLogger(__name__)

_e_str = "Electronic structure must contain dopings temperatures and scattering rates"


class TransportCalculator(MSONable):
    def __init__(
        self,
        calculate_mobility: bool = True,
        separate_scattering_mobilities: bool = False,
    ):
        self.separate_scattering_mobilities = separate_scattering_mobilities
        self.calculate_mobility = calculate_mobility

    def solve_bte(self, amset_data: AmsetData):
        has_doping = amset_data.doping is not None
        has_temps = amset_data.temperatures is not None
        has_rates = amset_data.scattering_rates is not None
        if not (has_doping and has_temps and has_rates):
            raise ValueError(_e_str)

        logger.info(
            "Calculating conductivity, Seebeck, and electronic thermal conductivity"
        )
        t0 = time.perf_counter()
        sigma, seebeck, kappa = _calculate_transport_properties(amset_data)
        log_time_taken(t0)

        if not self.calculate_mobility:
            return sigma, seebeck, kappa, None

        if amset_data.is_metal:
            logger.info("System is metallic, refusing to calculate carrier mobility")
            return sigma, seebeck, kappa, None

        n_scats = len(amset_data.scattering_labels)

        logger.info("Calculating overall mobility")
        t0 = time.perf_counter()
        mobility = {"overall": _calculate_mobility(amset_data, np.arange(n_scats))}
        log_time_taken(t0)

        if self.separate_scattering_mobilities:
            logger.info("Calculating individual scattering rate mobilities")
            t0 = time.perf_counter()
            for rate_idx, name in enumerate(amset_data.scattering_labels):
                mobility[name] = _calculate_mobility(
                    amset_data, rate_idx, pbar_label="mobility ({})".format(name)
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

    pbar = tqdm(
        list(np.ndindex(amset_data.fermi_levels.shape)),
        ncols=output_width,
        desc="    ├── {}".format(pbar_label),
        bar_format="{l_bar}{bar}| {elapsed}<{remaining}{postfix}",
        file=sys.stdout,
    )
    for n, t in pbar:
        br = {s: np.arange(len(amset_data.energies[s])) for s in amset_data.spins}
        cb_idx = {s: amset_data.vb_idx[s] + 1 for s in amset_data.spins}

        if amset_data.doping[n] > 0:
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
        epsilon, dos, vvdos = get_transport_dos(
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

        if amset_data.doping[n] > 0:
            carrier_conc = amset_data.electron_conc[n, t]
        else:
            carrier_conc = amset_data.hole_conc[n, t]

        # c = -c[0, ...] / (volume / (Meter / 100.)**3)

        # convert mobility to cm^2/V.s
        uc = 0.01 / (e * carrier_conc * (1 / bohr_to_cm) ** 3)
        mobility[n, t] = sigma[0, ...] * uc

    return mobility


def _calculate_transport_properties(amset_data, pbar_label="transport"):
    n_t_size = (len(amset_data.doping), len(amset_data.temperatures))

    sigma = np.zeros(n_t_size + (3, 3))
    seebeck = np.zeros(n_t_size + (3, 3))
    kappa = np.zeros(n_t_size + (3, 3))
    volume = amset_data.structure.volume

    pbar = tqdm(
        list(np.ndindex(n_t_size)),
        ncols=output_width,
        desc="    ├── {}".format(pbar_label),
        bar_format="{l_bar}{bar}| {elapsed}<{remaining}{postfix}",
        file=sys.stdout,
    )
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
        epsilon, dos, vvdos = get_transport_dos(
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

    emesh, dos = tetrahedron_band_structure.get_density_of_states(
        energies, sum_spins=True, band_idx=band_idx, use_cached_weights=True
    )

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

    return emesh, dos, vvdos
