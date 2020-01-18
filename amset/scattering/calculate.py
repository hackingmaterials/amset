"""
This module implements methods to calculate electron scattering based on an
AmsetData object.
"""

import logging
import time
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional, Union

import numpy as np
from BoltzTraP2 import units
from BoltzTraP2.fd import FD
from scipy.interpolate import griddata

from amset.constants import hbar, small_val, spin_name
from amset.data import AmsetData
from amset.log import log_list, log_time_taken
from amset.scattering.basic import AbstractBasicScattering
from amset.scattering.elastic import AbstractElasticScattering
from amset.scattering.inelastic import AbstractInelasticScattering
from amset.tetrahedron import (get_cross_section_values, get_projected_intersections,
                               integrate_function_over_cross_section)
from amset.util import get_progress_bar
from pymatgen import Spin
from pymatgen.util.coord import pbc_diff

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

logger = logging.getLogger(__name__)

_all_scatterers = (
    AbstractElasticScattering.__subclasses__()
    + AbstractInelasticScattering.__subclasses__()
    + AbstractBasicScattering.__subclasses__()
)
_scattering_mechanisms = {m.name: m for m in _all_scatterers}


class ScatteringCalculator(object):
    def __init__(
        self,
        settings: Dict[str, float],
        amset_data: AmsetData,
        scattering_type: Union[str, List[str], float] = "auto",
        use_symmetry: bool = True,
        nworkers: int = -1,
    ):
        if amset_data.temperatures is None or amset_data.doping is None:
            raise RuntimeError(
                "AmsetData doesn't contain doping levels or temperatures"
            )

        self.scattering_type = scattering_type
        self.settings = settings
        self.nworkers = nworkers if nworkers != -1 else cpu_count()
        self.use_symmetry = use_symmetry
        self.scatterers = self.get_scatterers(scattering_type, settings, amset_data)
        self.amset_data = amset_data

        if self.amset_data.fd_cutoffs:
            self.scattering_energy_cutoffs = self.amset_data.fd_cutoffs
        else:
            self.scattering_energy_cutoffs = (
                min(self.amset_data.dos.energies),
                max(self.amset_data.dos.energies),
            )

    @property
    def basic_scatterers(self):
        return [s for s in self.scatterers if isinstance(s, AbstractBasicScattering)]

    @property
    def inelastic_scatterers(self):
        return [
            s for s in self.scatterers if isinstance(s, AbstractInelasticScattering)
        ]

    @property
    def elastic_scatterers(self):
        return [s for s in self.scatterers if isinstance(s, AbstractElasticScattering)]

    @property
    def scatterer_labels(self):
        basic_names = [s.name for s in self.basic_scatterers]
        elastic_names = [s.name for s in self.elastic_scatterers]
        inelastic_names = [s.name for s in self.inelastic_scatterers]

        return basic_names + elastic_names + inelastic_names

    @staticmethod
    def get_scatterers(
        scattering_type: Union[str, List[str], float],
        settings: Dict[str, Any],
        amset_data: AmsetData,
    ) -> List[Union[AbstractElasticScattering, AbstractInelasticScattering]]:
        if scattering_type == "auto":
            # dynamically determine the available scattering mechanism subclasses
            logger.info(
                "Examining material properties to determine possible "
                "scattering mechanisms"
            )

            scattering_type = []
            for name, mechanism in _scattering_mechanisms.items():
                if all([settings.get(x, False) for x in mechanism.required_properties]):
                    scattering_type.append(name)

            if not scattering_type:
                raise ValueError(
                    "No scattering mechanisms possible with " "material properties"
                )

        else:
            for name in scattering_type:
                missing_properties = [
                    p
                    for p in _scattering_mechanisms[name].required_properties
                    if not settings.get(p, False)
                ]

                if missing_properties:
                    str_missing = ", ".join(missing_properties)
                    raise ValueError(
                        "{} scattering mechanism specified but the following "
                        "material properties are missing: {}".format(name, str_missing)
                    )

        str_scats = ", ".join(scattering_type)
        logger.info("Scattering mechanisms to be calculated: {}".format(str_scats))

        return [
            _scattering_mechanisms[name](settings, amset_data)
            for name in scattering_type
        ]

    def calculate_scattering_rates(self):
        spins = self.amset_data.spins
        full_kpoints = self.amset_data.full_kpoints
        f_shape = self.amset_data.fermi_levels.shape
        scattering_shape = (len(self.scatterer_labels),) + f_shape

        # rates has shape (spin, nscatterers, ndoping, ntemp, nbands, nkpoints)
        rates = {
            s: np.zeros(scattering_shape + self.amset_data.energies[s].shape)
            for s in spins
        }
        masks = {
            s: np.full(scattering_shape + self.amset_data.energies[s].shape, True)
            for s in spins
        }

        if self.use_symmetry:
            nkpoints = len(self.amset_data.ir_kpoints_idx)
        else:
            nkpoints = len(full_kpoints)

        logger.info("Scattering information:")
        log_list(["# k-points: {}".format(nkpoints)])

        for spin in spins:
            for b_idx in range(len(self.amset_data.energies[spin])):
                str_b = "Calculating rates for {} band {}"
                logger.info(str_b.format(spin_name[spin], b_idx + 1))

                t0 = time.perf_counter()
                rates[spin][..., b_idx, :], masks[spin][
                    ..., b_idx, :
                ] = self.calculate_band_rates(spin, b_idx)

                info = [
                    "max rate: {:.4g}".format(rates[spin][..., b_idx, :].max()),
                    "min rate: {:.4g}".format(rates[spin][..., b_idx, :].min()),
                    "time: {:.4f} s".format(time.perf_counter() - t0),
                ]
                log_list(info)

            # fill in k-points outside Fermi-Dirac cutoffs with a default value
            rates[spin][masks[spin]] = 1e14

        # if the k-point density is low, some k-points may not have other k-points
        # within the energy tolerance leading to zero rates
        rates = _interpolate_zero_rates(rates, full_kpoints, masks)

        return rates

    def calculate_band_rates(self, spin: Spin, b_idx: int):
        conversion = self.amset_data.structure.lattice.reciprocal_lattice.volume

        kpoints_idx = self.amset_data.ir_kpoints_idx
        nkpoints = len(kpoints_idx)

        buf = 0.01 * 5 * units.eV
        band_energies = self.amset_data.energies[spin][b_idx, kpoints_idx]
        mask = (band_energies < self.scattering_energy_cutoffs[0] - buf) | (
            band_energies > self.scattering_energy_cutoffs[1] + buf
        )
        fill_mask = mask[self.amset_data.ir_to_full_kpoint_mapping]

        n = np.sum(~fill_mask)
        logger.debug("  ├── # k-points within Fermi–Dirac cut-offs: {}".format(n))

        # get k-point indexes of k-points within FD cutoffs (faster than np.where)
        k_idx_in_cutoff = np.arange(nkpoints)[~mask]

        to_stack = []
        if len(self.basic_scatterers) > 0:
            basic_rates = np.array(
                [m.rates[spin][:, :, b_idx, kpoints_idx] for m in self.basic_scatterers]
            )
            to_stack.append(basic_rates)

        if len(self.elastic_scatterers) > 0:
            elastic_prefactors = conversion * np.array(
                [m.prefactor(spin, b_idx) for m in self.elastic_scatterers]
            )
            elastic_rates = np.zeros(elastic_prefactors.shape + (nkpoints,))

            if len(k_idx_in_cutoff) > 0:
                pbar = get_progress_bar(k_idx_in_cutoff, desc="elastic")
                for k_idx in pbar:
                    elastic_rates[..., k_idx] = self.calculate_rate(spin, b_idx, k_idx)

            elastic_rates *= elastic_prefactors[..., None]
            to_stack.append(elastic_rates)

        if len(self.inelastic_scatterers) > 0:
            inelastic_prefactors = conversion * np.array(
                [m.prefactor(spin, b_idx) for m in self.inelastic_scatterers]
            )
            inelastic_rates = np.zeros(inelastic_prefactors.shape + (nkpoints,))
            f_pop = self.settings["pop_frequency"]
            energy_diff = f_pop * 1e12 * 2 * np.pi * hbar * units.eV

            if len(k_idx_in_cutoff) > 0:
                pbar = get_progress_bar(k_idx_in_cutoff, desc="inelastic")
                inelastic_rates[:, :, :, k_idx_in_cutoff] = 0
                for k_idx in pbar:
                    for ediff in [energy_diff, -energy_diff]:
                        inelastic_rates[:, :, :, k_idx] += self.calculate_rate(
                            spin, b_idx, k_idx, energy_diff=ediff
                        )

            inelastic_rates *= inelastic_prefactors[..., None]
            to_stack.append(inelastic_rates)

        all_band_rates = np.vstack(to_stack)

        return all_band_rates[..., self.amset_data.ir_to_full_kpoint_mapping], fill_mask

    def calculate_rate(self, spin, b_idx, k_idx, energy_diff=None):
        rlat = self.amset_data.structure.lattice.reciprocal_lattice.matrix
        ir_kpoints_idx = self.amset_data.ir_kpoints_idx
        energy = self.amset_data.energies[spin][b_idx, ir_kpoints_idx][k_idx]

        if energy_diff:
            energy += energy_diff

        tbs = self.amset_data.tetrahedral_band_structure

        tet_dos, tet_mask, cs_weights, tet_contributions = tbs.get_tetrahedra_density_of_states(
            spin,
            energy,
            return_contributions=True,
            symmetry_reduce=False,
            # band_idx=b_idx,
        )

        if len(tet_dos) == 0:
            return 0

        # next, get k-point indices and band_indices
        property_mask, band_kpoint_mask, band_mask, kpoint_mask = tbs.get_masks(
            spin, tet_mask
        )

        k = self.amset_data.ir_kpoints[k_idx]
        k_primes = self.amset_data.full_kpoints[kpoint_mask]

        overlap = self.amset_data.overlap_calculator.get_overlap(
            spin, b_idx, k, band_mask, k_primes
        )

        # put overlap back in array with shape (nbands, nkpoints)
        all_overlap = np.zeros(self.amset_data.energies[spin].shape)
        all_overlap[band_kpoint_mask] = overlap

        # now select the properties at the tetrahedron vertices
        vert_overlap = all_overlap[property_mask]

        # get interpolated overlap and k-point at centre of tetrahedra cross sections
        tet_overlap = get_cross_section_values(vert_overlap, *tet_contributions)
        tetrahedra = tbs.tetrahedra[spin][tet_mask]

        # have to deal with the case where the tetrahedron cross section crosses the
        # zone boundary. This is a slight inaccuracy but we just treat the
        # cross section as if it is on one side of the boundary
        tet_kpoints = self.amset_data.full_kpoints[tetrahedra]
        base_kpoints = tet_kpoints[:, 0][:, None, :]
        k_diff = pbc_diff(tet_kpoints, base_kpoints) + pbc_diff(base_kpoints, k)
        k_diff = np.dot(k_diff, rlat)

        intersections = get_cross_section_values(
            k_diff, *tet_contributions, average=False
        )
        projected_intersections = get_projected_intersections(intersections)

        if energy_diff:
            f = np.zeros(self.amset_data.fermi_levels.shape)

            for n, t in np.ndindex(self.amset_data.fermi_levels.shape):
                f[n, t] = FD(
                    energy,
                    self.amset_data.fermi_levels[n, t],
                    self.amset_data.temperatures[t] * units.BOLTZMANN,
                )

            functions = np.array(
                [m.factor(energy_diff <= 0, f) for m in self.inelastic_scatterers]
            )
        else:
            functions = np.array([m.factor() for m in self.elastic_scatterers])

        rates = np.array(
            [
                integrate_function_over_cross_section(
                    f,
                    projected_intersections,
                    *tet_contributions[0:3],
                    return_shape=self.amset_data.fermi_levels.shape,
                    cross_section_weights=cs_weights
                )
                for f in functions
            ]
        )

        # sometimes the projected intersections can be nan when the density of states
        # contribution is infinitesimally small; this catches those errors
        rates[np.isnan(rates)] = 0

        rates /= self.amset_data.structure.lattice.reciprocal_lattice.volume
        rates *= tet_overlap

        return np.sum(rates, axis=-1)


def _interpolate_zero_rates(rates, kpoints, masks: Optional = None):
    # loop over all scattering types, doping, temps, and bands and interpolate
    # zero scattering rates based on the nearest k-point
    logger.info("Interpolating missing scattering rates")
    n_rates = sum([np.product(rates[spin].shape[:-1]) for spin in rates])
    pbar = get_progress_bar(total=n_rates, desc="progress")

    t0 = time.perf_counter()
    k_idx = np.arange(len(kpoints))
    for spin in rates:
        for s, d, t, b in np.ndindex(rates[spin].shape[:-1]):

            if masks is not None:
                mask = np.invert(masks[spin][s, d, t, b])
            else:
                mask = [True] * len(rates[spin][s, d, t, b])

            non_zero_rates = rates[spin][s, d, t, b, mask] > 1e7
            # non_zero_rates = rates[spin][s, d, t, b, mask] != 0
            zero_rate_idx = k_idx[mask][~non_zero_rates]
            non_zero_rate_idx = k_idx[mask][non_zero_rates]

            if not np.any(non_zero_rates):
                # all scattering rates are zero so cannot interpolate
                # generally this means the scattering prefactor is zero. E.g.
                # for POP when studying non polar materials
                rates[spin][s, d, t, b, mask] += small_val

            elif np.sum(non_zero_rates) != np.sum(mask):
                # interpolation seems to work best when all the kpoints are +ve
                # therefore add 0.5
                # Todo: Use cartesian coordinates (will be more robust to
                #  oddly shaped cells)
                rates[spin][s, d, t, b, zero_rate_idx] = griddata(
                    points=kpoints[non_zero_rate_idx] + 0.5,
                    values=rates[spin][s, d, t, b, non_zero_rate_idx],
                    xi=kpoints[zero_rate_idx] + 0.5,
                    method="nearest",
                )

            pbar.update()
    pbar.close()
    log_time_taken(t0)

    return rates
