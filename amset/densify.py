import logging
import math
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import cauchy

from BoltzTraP2 import units
from BoltzTraP2.bandlib import DOS, dFDde
from amset import amset_defaults as defaults
from amset.data import AmsetData
from amset.interpolate import Interpolater
from amset.log import log_list
from amset.util import kpoints_to_first_bz
from amset.voronoi import PeriodicVoronoi

logger = logging.getLogger(__name__)
pdefaults = defaults["performance"]


class BandDensifier(object):

    def __init__(self,
                 interpolater: Interpolater,
                 amset_data: AmsetData,
                 dos_estep: float = pdefaults["dos_estep"],
                 energy_cutoff: Optional[float] = None,
                 scissor: float = None,
                 bandgap: float = None):
        if amset_data.fermi_levels is None:
            raise RuntimeError(
                "amset_data doesn't contain Fermi level information")

        self._interpolater = interpolater
        self._amset_data = amset_data
        self._dos_estep = dos_estep
        self._energy_cutoff = energy_cutoff
        self._bandgap = bandgap
        self._scissor = scissor

        logger.info("Generating densification weights")

        self._densification_weights = {s: np.zeros(amset_data.energies[s].shape)
                                       for s in amset_data.spins}
        all_energies = {s: [] for s in amset_data.spins}
        all_weights = {s: [] for s in amset_data.spins}
        max_weight = 0
        for spin in amset_data.spins:
            for b_idx, band_energies in enumerate(amset_data.energies[spin]):
                # calculate the DOS and sum of Fermi integrals with doping and
                # temperature on a band-by-band basis. Note: if we add interband
                # scattering at a later stage this should be for the overall
                # band structure not the band DOS etc.
                dos_min = band_energies.min() - self._dos_estep
                dos_max = band_energies.max() + self._dos_estep
                npts = math.ceil((dos_max - dos_min) / self._dos_estep) * 10
                energies, dos = DOS(band_energies[:, None], npts=npts,
                                    erange=(dos_min, dos_max))
                # dos = smoothen_DOS(energies, dos, 10)

                # three fermi integrals govern transport properties:
                #   1. df/de controls conductivity and mobility
                #   2. (e-u) * df/de controls Seebeck
                #   3. (e-u)^2 df/de controls electronic thermal cond
                # take the absolute sum of the integrals across all doping and
                # temperatures. this gives us which energies are important for
                # transport for ever Fermi level and temperature

                integral_sum = np.zeros(dos.shape)

                for n, t in np.ndindex(amset_data.fermi_levels.shape):
                    ef = amset_data.fermi_levels[n, t] * units.eV
                    kbt = amset_data.temperatures[t] * units.BOLTZMANN
                    band_dfde = dFDde(energies, ef, kbt)
                    sigma_int = np.abs(band_dfde)
                    seeb_int = np.abs((energies - ef) * band_dfde)
                    ke_int = np.abs((energies - ef) ** 2 * band_dfde)

                    # normalize the transport integrals and sum
                    integral_sum += sigma_int / sigma_int.max()
                    integral_sum += seeb_int / seeb_int.max()
                    integral_sum += ke_int / ke_int.max()

                # weights for densification are the sum of Fermi integrals / DOS
                # I.e. regions with larger Fermi integrals are prioritized
                # and regions with low DOS are prioritized.
                weights = integral_sum / dos
                weights[np.isnan(weights) | np.isinf(weights)] = 0
                all_energies[spin].append(energies)
                all_weights[spin].append(weights)

                max_weight = max(max_weight, weights.max())

        for spin in amset_data.spins:
            for i in range(len(all_weights[spin])):
                # normalize weightings so they are consistent across all bands
                weights = all_weights[spin][i] / max_weight

                # interpolate the weights to all kpoints
                interp_weights = interp1d(all_energies[spin][i], weights)(
                    amset_data.energies[spin][i])

                interp_weights[interp_weights < 0.05] = 0

                self._densification_weights[spin][i] = interp_weights

        self._sum_weights = sum([np.sum(self._densification_weights[s])
                                 for s in amset_data.spins])

    def densify(self, n_extra_kpoints: float = pdefaults["n_extra_kpoints"]):
        logger.info("Densifying band structure around Fermi integrals")

        # add additional k-points around the k-points in the existing mesh
        # the number of additional k-points is proportional to the
        # densification weight for that k-point
        factor = n_extra_kpoints / self._sum_weights
        extra_kpoint_counts = {
            s: np.ceil(self._densification_weights[s] * factor).astype(int)
            for s in self._densification_weights}

        # recalculate the number of points to take into account rounding
        # generally means the number of additional k-points is
        total_points = sum([np.sum(extra_kpoint_counts[s])
                            for s in extra_kpoint_counts])

        # add additional points according to a Lorentz distribution:
        # P = 1/(1 + |k-k0|^2/γ^2)
        # we set gamma to be 1/20 * the average inter k-point spacing for the
        # x, y and z directions. Note that this only works well if the original
        # k-point density (not lattice) is isotropic, which will be the case if
        # using BoltzTraP interpolation.
        rlat = self._amset_data.structure.lattice.reciprocal_lattice

        gamma_sq = (np.average(1 / self._amset_data.kpoint_mesh) / 5) ** 2

        log_list(["# extra kpoints: {}".format(total_points),
                  "Densification γ: {:.5f} Å⁻¹".format(
                      np.average(rlat.abc / self._amset_data.kpoint_mesh)/20)])

        lorentz_points = cauchy.rvs(loc=0, scale=gamma_sq,
                                    size=(total_points, 3))

        # flattened_kpoints is a list of kpoints, where each point is present
        # as many times as the number of extra points surrounding it.
        # the total number of flattened_kpoints is therefore equal to the total
        # number of extra points. The order of flattened k-points does
        # not matter, as to keep the number of kpoints the same for each band,
        # each band will be interpolated at all extra k-points
        flattened_kpoints = np.concatenate([
            np.repeat(self._amset_data.full_kpoints,
                      n_points_per_kpoint, axis=0)
            for spin in self._amset_data.spins
            for n_points_per_kpoint in extra_kpoint_counts[spin]])

        # flattened_kpoints now contains the new k-points to interpolate
        flattened_kpoints += lorentz_points
        flattened_kpoints = kpoints_to_first_bz(flattened_kpoints)

        # TODO: Test get_energies and get amset_data return same values
        #   for the same k-point mesh.
        energies, velocities, projections = self._interpolater.get_energies(
            flattened_kpoints, energy_cutoff=self._energy_cutoff,
            bandgap=self._bandgap, scissor=self._scissor,
            return_velocity=True, return_effective_mass=False,
            return_projections=True, atomic_units=True)

        # finally, calculate k-point weights as the volume of each cell in the
        # Voronoi decomposition.
        all_kpoints = np.concatenate(
            (self._amset_data.full_kpoints, flattened_kpoints))

        voronoi = PeriodicVoronoi(
            all_kpoints, original_mesh=self._amset_data.kpoint_mesh)
        volumes = voronoi.compute_volumes()

        sum_volumes = volumes.sum()
        vol_diff = abs((np.linalg.det(rlat.matrix) / sum_volumes) - 1)

        if vol_diff > 0.01:
            logger.warning("Sum of Voronoi volumes differs from reciprocal "
                           "lattice volume by {:.1f}%".format(vol_diff * 100))

        kpoint_weights = volumes / sum_volumes

        # note k-point weights is for all k-points, whereas the other properties
        # are just for the additional k-points
        return (flattened_kpoints, energies, velocities, projections,
                kpoint_weights)
