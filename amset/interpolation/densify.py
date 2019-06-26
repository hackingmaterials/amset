import itertools
import logging
import math
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d

from BoltzTraP2 import units
from BoltzTraP2.bandlib import DOS, dFDde
from amset import amset_defaults as defaults
from amset.data import AmsetData
from amset.interpolation.interpolate import Interpolater
from amset.misc.log import log_list
from amset.misc.util import kpoints_to_first_bz
from amset.interpolation.voronoi import PeriodicVoronoi

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"
__date__ = "June 21, 2019"

logger = logging.getLogger(__name__)
gdefaults = defaults["general"]
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
                weights = np.log1p(integral_sum) / np.log1p(dos)
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

    def densify(self,
                num_extra_kpoints: float = gdefaults["num_extra_kpoints"]):
        logger.info("Densifying band structure around Fermi integrals")

        # add additional k-points around the k-points in the existing mesh
        # the number of additional k-points is proportional to the
        # densification weight for that k-point
        factor = num_extra_kpoints / self._sum_weights
        extra_kpoint_counts = {
            s: np.ceil(self._densification_weights[s] * factor).astype(int)
            for s in self._densification_weights}

        # get the number of extra points for each kpoint. Shape of
        # n_points_per_kpoint is (nkpoints, )
        n_points_per_kpoint = np.sum(np.concatenate([
            extra_kpoint_counts[s] for s in extra_kpoint_counts]), axis=0)

        # recalculate the number of points to take into account rounding
        # generally means the number of additional k-points is larger than
        # specified by the user
        total_points = np.sum(n_points_per_kpoint)

        # max distance is just under half the distance to the nearest k-point.
        # Note, the averaging means that this only works well if the original
        # k-point density is isotropic, which will be the case if using
        # BoltzTraP interpolation.
        max_dist = np.average(1 / self._amset_data.kpoint_mesh) / 2.1

        # get the kpoints which will be densified
        k_mask = n_points_per_kpoint > 0
        k_coords = self._amset_data.full_kpoints[k_mask]
        n_points_per_kpoint = n_points_per_kpoint[k_mask]
        print(sum(k_mask))
        print(n_points_per_kpoint)

        # add additional points in concenctric spheres around the k-points
        #
        extra_kpoints = np.concatenate(
            [_generate_points(n_extra, max_dist) + kpoint
             for kpoint, n_extra in zip(k_coords, n_points_per_kpoint)])
        log_list(["# extra kpoints: {}".format(total_points),
                  "max frac k-distance: {:.5f}".format(max_dist)])

        from monty.serialization import dumpfn
        dumpfn(extra_kpoints, "extra_kpoints.json")

        extra_kpoints = kpoints_to_first_bz(extra_kpoints)

        skip = 5 / self._interpolater.interpolation_factor
        energies, vvelocities, projections = self._interpolater.get_energies(
            extra_kpoints, energy_cutoff=self._energy_cutoff,
            bandgap=self._bandgap, scissor=self._scissor,
            return_velocity=True, return_effective_mass=False,
            return_projections=True, atomic_units=True,
            return_vel_outer_prod=True, skip_coefficients=skip)

        # finally, calculate k-point weights as the volume of each cell in the
        # Voronoi decomposition.
        all_kpoints = np.concatenate(
            (self._amset_data.full_kpoints, extra_kpoints))

        voronoi = PeriodicVoronoi(
            all_kpoints, original_mesh=self._amset_data.kpoint_mesh)
        kpoint_weights = voronoi.compute_volumes()

        # note k-point weights is for all k-points, whereas the other properties
        # are just for the additional k-points
        return (extra_kpoints, energies, vvelocities, projections,
                kpoint_weights)


def _generate_points_cube(n_extra_kpoints: int, max_distance: float):
    n_p = math.ceil(np.cbrt(n_extra_kpoints) / 2)
    # forward_p = np.geomspace(max_distance/20, max_distance, n_p)
    forward_p = np.linspace(max_distance/20, max_distance, n_p)
    all_p = np.concatenate((-forward_p, forward_p))
    return np.array(list(itertools.product(all_p, all_p, all_p)))


def _generate_points(n_extra_kpoints: int, max_distance: float,
                     min_n_points_per_sphere: int = 32,
                     max_n_points_per_sphere: int = 256,
                     default_n_spheres: int = 5):

    # each sphere must contain at least min_n_points_per_sphere
    actual_n_spheres = min(
        default_n_spheres, math.ceil(n_extra_kpoints / min_n_points_per_sphere))

    actual_n_spheres = max(
        actual_n_spheres, math.ceil(n_extra_kpoints / max_n_points_per_sphere))

    # generate radii for spheres, getting logarithmically further away, up to
    # max_distance. Note, not all spheres may be used
    if actual_n_spheres > default_n_spheres:
        radii = np.geomspace(max_distance/10, max_distance, actual_n_spheres)
    else:
        radii = np.geomspace(max_distance/10, max_distance, default_n_spheres)

    n_points_per_sphere = math.ceil(n_extra_kpoints / actual_n_spheres)

    sphere_points = np.concatenate([
        sunflower_sphere(radius=r, samples=n)
        for r, n in zip(radii, [n_points_per_sphere] * n_points_per_sphere)])

    return sphere_points[:n_extra_kpoints]


def fibonacci_sphere(radius=1, samples=1, randomize=False):
    rnd = 1.
    if randomize:
        rnd = np.random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x, y, z])

    return np.array(points) * radius


def sunflower_sphere(radius=1, samples=1):
    indices = np.arange(0, samples) + 0.5

    phi = np.arccos(1 - 2 * indices / samples)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    return np.stack((x, y, z), axis=1) * radius
