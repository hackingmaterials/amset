"""
This module implements methods to calculate electron scattering based on an
AmsetData object.

Todo:
 - Interband scattering
"""

import copy
import logging
import math
import sys
import time
from multiprocessing import cpu_count, Process, Queue
from typing import Dict, Union, List, Any, Optional

import numpy as np
from monty.json import MSONable
from scipy.interpolate import griddata
from sklearn.neighbors.ball_tree import BallTree
from tqdm import tqdm

from BoltzTraP2 import units

from amset.scattering.basic import AbstractBasicScattering
from pymatgen import Spin

from amset.constants import A_to_nm, output_width, over_sqrt_pi, hbar, \
    small_val, default_small_e, spin_name
from amset.data import AmsetData
from amset.misc.log import log_list, log_time_taken
from amset.scattering.elastic import AbstractElasticScattering
from amset.scattering.inelastic import AbstractInelasticScattering
from amset.misc.util import create_shared_array, gen_even_slices
from pymatgen.util.coord import pbc_diff

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"
__date__ = "June 21, 2019"

logger = logging.getLogger(__name__)

_all_scatterers = (
        AbstractElasticScattering.__subclasses__() +
        AbstractInelasticScattering.__subclasses__() +
        AbstractBasicScattering.__subclasses__()
)
_scattering_mechanisms = {m.name: m for m in _all_scatterers}


class ScatteringCalculator(MSONable):

    def __init__(self,
                 materials_properties: Dict[str, float],
                 amset_data: AmsetData,
                 scattering_type: Union[str, List[str], float] = "auto",
                 gauss_width: float = 0.001,
                 g_tol: float = 0.01,
                 max_g_iter: int = 8,
                 use_symmetry: bool = True,
                 nworkers: int = -1):
        if amset_data.temperatures is None or amset_data.doping is None:
            raise RuntimeError(
                "AmsetData doesn't contain doping levels or temperatures")

        self.scattering_type = scattering_type
        self.materials_properties = materials_properties
        self.gauss_width = gauss_width
        self.g_tol = g_tol
        self.max_g_iter = max_g_iter
        self.nworkers = nworkers if nworkers != -1 else cpu_count()
        self.use_symmetry = use_symmetry
        self.scatterers = self.get_scatterers(
            scattering_type, materials_properties, amset_data)
        self.amset_data = amset_data

        if self.amset_data.fd_cutoffs:
            self.scattering_energy_cutoffs = self.amset_data.fd_cutoffs
        else:
            self.scattering_energy_cutoffs = (min(self.amset_data.dos.energies),
                                              max(self.amset_data.dos.energies))

    @property
    def basic_scatterers(self):
        return [s for s in self.scatterers
                if isinstance(s, AbstractBasicScattering)]

    @property
    def inelastic_scatterers(self):
        return [s for s in self.scatterers
                if isinstance(s, AbstractInelasticScattering)]

    @property
    def elastic_scatterers(self):
        return [s for s in self.scatterers
                if isinstance(s, AbstractElasticScattering)]

    @property
    def scatterer_labels(self):
        elastic_names = [s.name for s in self.elastic_scatterers]

        if self.max_g_iter != 1:
            # have in and out scattering rates
            inelastic_names = ["{}_{}".format(s.name, d) for s in
                               self.inelastic_scatterers for d in ["in", "out"]]
        else:
            inelastic_names = [s.name for s in self.inelastic_scatterers]

        return elastic_names + inelastic_names

    @staticmethod
    def get_scatterers(scattering_type: Union[str, List[str], float],
                       materials_properties: Dict[str, Any],
                       amset_data: AmsetData,
                       prefer_full_imp: bool = True,
                       ) -> List[Union[AbstractElasticScattering,
                                       AbstractInelasticScattering,
                                       AbstractBasicScattering]]:
        # dynamically determine the available scattering mechanism subclasses
        if scattering_type == "auto":
            logger.info("Examining material properties to determine possible "
                        "scattering mechanisms")

            scattering_type = [
                name for name, mechanism in _scattering_mechanisms.items()
                if all([materials_properties.get(x, False) for x in
                        mechanism.required_properties])]

            if not scattering_type:
                raise ValueError("No scattering mechanisms possible with "
                                 "material properties")

            if prefer_full_imp:
                avoid = "IMP(BH)"
            else:
                avoid = "IMP"
            scattering_type = [s for s in scattering_type if s != avoid]

        else:
            for name in scattering_type:
                missing_properties = [
                    p for p in _scattering_mechanisms[name].required_properties
                    if not materials_properties.get(p, False)]

                if missing_properties:
                    raise ValueError(
                        "{} scattering mechanism specified but the following "
                        "material properties are missing: {}".format(
                            name, ", ".join(missing_properties)))

        logger.info("The following scattering mechanisms will be "
                    "calculated: {}".format(", ".join(scattering_type)))

        return [_scattering_mechanisms[name](materials_properties, amset_data)
                for name in scattering_type]

    def calculate_scattering_rates(self):
        spins = self.amset_data.spins
        full_kpoints = self.amset_data.full_kpoints

        # rates has shape (spin, nscatterers, ndoping, ntemp, nbands, nkpoints)
        rates = {s: np.zeros((len(self.scatterer_labels),
                              len(self.amset_data.doping),
                              len(self.amset_data.temperatures)) +
                             self.amset_data.energies[s].shape)
                 for s in spins}
        masks = {s: np.full((len(self.scatterer_labels),
                             len(self.amset_data.doping),
                             len(self.amset_data.temperatures)) +
                            self.amset_data.energies[s].shape, True)
                 for s in spins}

        if self.use_symmetry:
            nkpoints = len(self.amset_data.ir_kpoints_idx)
        else:
            nkpoints = len(full_kpoints)

        if len(self.amset_data.full_kpoints) > 1.5e6:
            batch_size = 30
        else:
            batch_size = 100

        nsplits = math.ceil(nkpoints/batch_size)
        logger.info("Scattering information:")
        log_list(["energy tolerance: {} eV".format(self.gauss_width),
                  "# k-points: {}".format(nkpoints),
                  "batch size: {}".format(batch_size)])

        for spin in spins:
            for b_idx in range(len(self.amset_data.energies[spin])):
                logger.info("Calculating rates for {} band {}".format(
                    spin_name[spin], b_idx + 1))

                t0 = time.perf_counter()
                rates[spin][:, :, :, b_idx, :], masks[spin][:, :, :, b_idx, :] \
                    = self.calculate_band_rates(spin, b_idx, nsplits)

                log_list([
                    "max rate: {:.4g}".format(rates[spin][..., b_idx, :].max()),
                    "min rate: {:.4g}".format(rates[spin][..., b_idx, :].min()),
                    "time: {:.4f} s".format(time.perf_counter() - t0)])

        # if the k-point density is low, some k-points may not have
        # other k-points within the energy tolerance leading to zero rates
        rates = _interpolate_zero_rates(rates, full_kpoints, masks)

        return rates

    def calculate_band_rates(self,
                             spin: Spin,
                             b_idx: int,
                             nsplits: int):
        integral_conversion = (
                (2 * np.pi) ** 3
                / (self.amset_data.structure.lattice.volume * A_to_nm ** 3)
                / self.gauss_width)

        # prefactors have shape [nscatterers, ndoping, ntemp)
        elastic_prefactors = integral_conversion * np.array(
            [m.prefactor(spin, b_idx) for m in self.elastic_scatterers])
        inelastic_prefactors = integral_conversion * np.array(
            [m.prefactor(spin, b_idx) for m in self.inelastic_scatterers])

        if self.use_symmetry:
            kpoints_idx = self.amset_data.ir_kpoints_idx
        else:
            kpoints_idx = np.arange(len(self.amset_data.full_kpoints))

        nkpoints = len(kpoints_idx)

        band_energies = self.amset_data.energies[spin][b_idx, kpoints_idx]

        # filter_points far from band edge where Fermi integrals are very small
        buf = self.gauss_width * 5 * units.eV
        ball_band_energies = copy.deepcopy(band_energies)
        mask = (band_energies < self.scattering_energy_cutoffs[0] - buf) | (
                band_energies > self.scattering_energy_cutoffs[1] + buf)
        if self.use_symmetry:
            fill_mask = mask[self.amset_data.ir_to_full_kpoint_mapping]
        else:
            fill_mask = mask

        logger.debug("  ├── # k-points within Fermi–Dirac cut-offs: {}".format(
            np.sum(~fill_mask)))

        # set the energies out of range to infinite so that they will not be
        # included in the scattering rate calculations
        ball_band_energies[mask] *= float("inf")

        ball_tree = BallTree(ball_band_energies[:, None], leaf_size=100)
        g = np.zeros(self.amset_data.fermi_levels.shape +
                     (len(self.amset_data.energies[spin][b_idx]), ))

        s_g, g = create_shared_array(g, return_buffer=True)

        s_energies = create_shared_array(band_energies)
        s_kpoints = create_shared_array(self.amset_data.full_kpoints)
        s_k_norms = create_shared_array(self.amset_data.kpoint_norms)
        s_k_weights = create_shared_array(self.amset_data.kpoint_weights)
        s_a_factor = create_shared_array(
            self.amset_data.a_factor[spin][b_idx, kpoints_idx])
        s_c_factor = create_shared_array(
            self.amset_data.c_factor[spin][b_idx, kpoints_idx])

        rlat = self.amset_data.structure.lattice.reciprocal_lattice.matrix

        # spawn as many worker processes as needed, put all bands in the queue,
        # and let them work until all the required rates have been computed.
        workers = []
        iqueue = Queue()
        oqueue = Queue()

        for i in range(self.nworkers):
            args = (self.scatterers, ball_tree, spin, b_idx,
                    self.gauss_width * units.eV,
                    s_g, s_energies, s_kpoints, s_k_norms, s_k_weights,
                    s_a_factor, s_c_factor, len(band_energies), rlat, iqueue,
                    oqueue)
            if self.use_symmetry:
                kwargs = {
                    "grouped_ir_to_full": self.amset_data.grouped_ir_to_full,
                    "ir_kpoints_idx": self.amset_data.ir_kpoints_idx}

                workers.append(Process(target=scattering_worker, args=args,
                                       kwargs=kwargs))
            else:
                workers.append(Process(target=scattering_worker, args=args))

        slices = list(gen_even_slices(nkpoints, nsplits))

        for w in workers:
            w.start()

        elastic_rates = None
        if self.basic_scatterers:
            basic_rates = np.array([s.rates for s in self.basic_scatters])

        if self.elastic_scatterers:
            elastic_rates = self._fill_workers(
                nkpoints, slices, iqueue, oqueue, desc="elastic")
            elastic_rates *= elastic_prefactors[..., None]

            if mask is not None:
                # these k-points are outside the range of the FD distribution
                # and therefore will not contribute to the scattering
                elastic_rates[:, :, :, fill_mask] = 1e14

        if self.inelastic_scatterers:
            # currently only supports one inelastic scattering energy difference
            # convert frequency to THz and get energy in Rydberg
            energy_diff = (self.materials_properties["pop_frequency"] * 1e12
                           * 2 * np.pi * hbar * units.eV)

            n_inelastic = len(self.inelastic_scatterers)
            shape = (n_inelastic, len(self.amset_data.doping),
                     len(self.amset_data.temperatures),
                     len(self.amset_data.full_kpoints))
            in_rates = np.zeros(shape)
            out_rates = np.zeros(shape)

            # in 1/s
            force = (self.amset_data.dfdk[spin][:, :, b_idx] *
                     default_small_e / hbar)
            # force = np.zeros(self.amset_data.dfde[spin][:, :, b_idx].shape)

            # if max_iter == 1 then RTA, don't calculate in rates
            calculate_in_rate = self.max_g_iter != 1
            calculate_out_rate = True

            for _ in range(self.max_g_iter):

                # rates are formatted as s1_i, s1_i, s2_o, s2_o etc
                inelastic_rates = self._fill_workers(
                    nkpoints, slices, iqueue, oqueue, energy_diff=energy_diff,
                    desc="inelastic", calculate_out_rate=calculate_out_rate,
                    calculate_in_rate=calculate_in_rate)

                if calculate_in_rate:
                    # in rates always returned first
                    in_rates = inelastic_rates[:n_inelastic]
                    in_rates *= inelastic_prefactors[..., None]

                    if fill_mask is not None:
                        in_rates[:, :, :, fill_mask] = 1e14

                if calculate_out_rate:
                    # out rate independent of g so only need calculate it once
                    idx = n_inelastic if calculate_in_rate else 0
                    out_rates = inelastic_rates[idx:]
                    out_rates *= inelastic_prefactors[..., None]
                    calculate_out_rate = False

                    if fill_mask is not None:
                        out_rates[:, :, :, fill_mask] = 1e14

                if self.max_g_iter != 1:
                    new_g = calculate_g(
                        out_rates, in_rates, elastic_rates, force)
                    if new_g[:, :, ~fill_mask].shape[-1] == 0:
                        g_diff = 0
                    else:
                        g_diff = np.abs(np.average(new_g[:, :, ~fill_mask] -
                                                   g[:, :, ~fill_mask]))
                        # print(new_g[0, 0, 0:4])
                        # print(np.max(new_g[:, :, ~fill_mask]))
                        # print(np.min(new_g[:, :, ~fill_mask]))

                    logger.debug("  ├── difference in g value: {:.4g}".format(
                        g_diff))
                    if g_diff < self.g_tol:
                        break

                    # update the shared buffer
                    g[:] = new_g[:]

            to_stack = [elastic_rates] if elastic_rates is not None else []
            if calculate_in_rate:
                to_stack.append(in_rates)
            to_stack.append(out_rates)
            all_band_rates = np.vstack(to_stack)

        else:
            all_band_rates = elastic_rates

        # The "None"s at the end of the queue signal the workers that there are
        # no more jobs left and they must therefore exit.
        for i in range(self.nworkers):
            iqueue.put(None)

        for w in workers:
            w.join()
            w.terminate()

        return all_band_rates, fill_mask

    def _fill_workers(self, nkpoints, slices, iqueue, oqueue, desc="progress",
                      energy_diff=0., calculate_out_rate=False,
                      calculate_in_rate=False):
        if energy_diff:
            # inelastic scattering
            n_scat = len(self.inelastic_scatterers)
            n_scat *= 2 if (calculate_out_rate and calculate_in_rate) else 1
        else:
            # elastic scattering
            n_scat = len(self.elastic_scatterers)

        band_rates = np.zeros(
            (n_scat, len(self.amset_data.doping),
             len(self.amset_data.temperatures), nkpoints))

        for s in slices:
            iqueue.put((s, energy_diff, calculate_out_rate, calculate_in_rate))

        # The results are processed as soon as they are ready.
        desc = "    ├── {}".format(desc)
        pbar = tqdm(total=nkpoints, ncols=output_width, desc=desc,
                    bar_format='{l_bar}{bar}| {elapsed}<{remaining}{postfix}',
                    file=sys.stdout)
        for _ in range(len(slices)):
            s, band_rates[:, :, :, s] = oqueue.get()
            pbar.update(s.stop - s.start)
        pbar.close()

        if self.use_symmetry:
            return band_rates[
                   :, :, :, self.amset_data.ir_to_full_kpoint_mapping]

        else:
            return band_rates


def scattering_worker(scatterers, ball_tree, spin, b_idx, gauss_width, s_g,
                      s_energies, s_kpoints, s_kpoint_norms, s_k_weights,
                      s_a_factor, s_c_factor, nkpoints,
                      reciprocal_lattice_matrix, iqueue, oqueue,
                      ir_kpoints_idx=None, grouped_ir_to_full=None):
    ntemps = scatterers[0].temperatures.shape[0]
    ndoping = scatterers[0].doping.shape[0]
    energies = np.frombuffer(s_energies).reshape(nkpoints)
    kpoints = np.frombuffer(s_kpoints).reshape(-1, 3)
    kpoint_weights = np.frombuffer(s_k_weights)
    kpoint_norms = np.frombuffer(s_kpoint_norms)
    a_factor = np.frombuffer(s_a_factor).reshape(nkpoints)
    c_factor = np.frombuffer(s_c_factor).reshape(nkpoints)
    g = np.frombuffer(s_g).reshape(ndoping, ntemps, -1)

    elastic_scatterers = [s for s in scatterers
                          if isinstance(s, AbstractElasticScattering)]
    inelastic_scatterers = [s for s in scatterers
                            if isinstance(s, AbstractInelasticScattering)]

    while True:
        job = iqueue.get()

        if job is None:
            break
        s, energy_diff, calculate_out_rate, calculate_in_rate = job

        if energy_diff == 0:
            # elastic scattering
            k_p_idx, ediff = ball_tree.query_radius(
                energies[s, None], gauss_width * 5, return_distance=True)

            k_idx = np.repeat(np.arange(len(k_p_idx)),
                              [len(a) for a in k_p_idx])
            k_p_idx = np.concatenate(k_p_idx)

            ediff = np.concatenate(ediff)
            if ediff.size == 0:
                band_rates = np.zeros(
                    (len(elastic_scatterers), ndoping, ntemps,
                     s.stop - s.start))
            else:
                k_idx += s.start

                if nkpoints != kpoints.shape[0]:
                    # working with symmetry reduced k-points
                    band_rates = get_ir_band_rates(
                        spin, b_idx, elastic_scatterers, ediff, gauss_width, s,
                        k_idx, k_p_idx,
                        kpoints, kpoint_norms, kpoint_weights, a_factor,
                        c_factor, reciprocal_lattice_matrix, ir_kpoints_idx,
                        grouped_ir_to_full)
                else:
                    # no symmetry, use the full BZ mesh
                    band_rates = get_band_rates(
                        spin, b_idx, elastic_scatterers, ediff, gauss_width, s,
                        k_idx, k_p_idx,
                        kpoints, kpoint_norms, kpoint_weights, a_factor,
                        c_factor, reciprocal_lattice_matrix)

        else:
            # sum scattering in and out rates over emission and absorption
            # *2 because have s_in and s_out for each scatterer
            n = 2 if calculate_in_rate and calculate_out_rate else 1
            band_rates = np.zeros(
                (len(inelastic_scatterers) * n, ndoping, ntemps,
                 s.stop - s.start))

            # inelastic scattering
            for diff in [energy_diff, -energy_diff]:
                k_p_idx, ediff = ball_tree.query_radius(
                    energies[s, None] + diff, gauss_width * 5,
                    return_distance=True)

                k_idx = np.repeat(np.arange(len(k_p_idx)),
                                  [len(a) for a in k_p_idx])
                k_p_idx = np.concatenate(k_p_idx)

                ediff = np.concatenate(ediff)
                if ediff.size != 0:

                    k_idx += s.start

                    if nkpoints != kpoints.shape[0]:
                        # working with symmetry reduced k-points
                        # now format of band rates is (s1_i, s2_i, s1_o, s2_o..)
                        band_rates += get_ir_band_rates(
                            spin, b_idx, inelastic_scatterers, ediff,
                            gauss_width, s, k_idx, k_p_idx, kpoints,
                            kpoint_norms, kpoint_weights, a_factor,
                            c_factor, reciprocal_lattice_matrix, ir_kpoints_idx,
                            grouped_ir_to_full, g=g, emission=diff <= 0,
                            calculate_out_rate=calculate_out_rate,
                            calculate_in_rate=calculate_in_rate)
                    else:
                        # no symmetry, use the full BZ mesh
                        band_rates += get_band_rates(
                            spin, b_idx, inelastic_scatterers, ediff,
                            gauss_width, s, k_idx,  k_p_idx, kpoints,
                            kpoint_norms, kpoint_weights, a_factor, c_factor,
                            reciprocal_lattice_matrix, g=g, emission=diff >= 0,
                            calculate_out_rate=calculate_out_rate,
                            calculate_in_rate=calculate_in_rate)

        oqueue.put((s, band_rates))


def get_ir_band_rates(spin, b_idx, scatterers, ediff, gauss_width, s, k_idx,
                      k_p_idx, kpoints, kpoint_norms, kpoint_weights, a_factor,
                      c_factor, reciprocal_lattice_matrix, ir_kpoints_idx,
                      grouped_ir_to_full, g=None, emission=False,
                      calculate_out_rate=True,
                      calculate_in_rate=True):
    from numpy.core.umath_tests import inner1d

    # k_idx and k_p_idx are currently their reduced form. E.g., span 0 to
    # n_ir_kpoints-1; find actual columns of k_p_idx in the full Brillouin zone
    # by lookup
    full_k_p_idx_grouped = grouped_ir_to_full[k_p_idx]

    # get the reduced k_idx including duplicate k_idx for the full k_prime
    repeated_k_idx = np.repeat(k_idx, [len(g) for g in full_k_p_idx_grouped])
    expand_k_idx = np.repeat(np.arange(
        len(k_idx)), [len(g) for g in full_k_p_idx_grouped])

    # flatten the list of mapped columns
    full_k_p_idx = np.concatenate(full_k_p_idx_grouped)

    # get the indices of the k_idx in the full Brillouin zone
    full_k_idx = ir_kpoints_idx[repeated_k_idx]

    mask = full_k_idx != full_k_p_idx
    full_k_idx = full_k_idx[mask]
    full_k_p_idx = full_k_p_idx[mask]
    expand_k_idx = expand_k_idx[mask]

    k_dot = inner1d(kpoints[full_k_idx], kpoints[full_k_p_idx])
    # k_angles is the cosine of the angles
    k_angles = k_dot / (kpoint_norms[full_k_idx] * kpoint_norms[full_k_p_idx])
    k_angles[np.isnan(k_angles)] = 1.

    a_vals = a_factor[k_idx] * a_factor[k_p_idx]
    c_vals = c_factor[k_idx] * c_factor[k_p_idx]

    overlap = (a_vals[expand_k_idx] + c_vals[expand_k_idx] * k_angles) ** 2
    weighted_overlap = (w0gauss(ediff / gauss_width)[expand_k_idx] * overlap *
                        kpoint_weights[full_k_p_idx])

    # norm of k difference squared in 1/nm
    k_diff_sq = np.linalg.norm(np.dot(
        pbc_diff(kpoints[full_k_idx], kpoints[full_k_p_idx]),
        reciprocal_lattice_matrix) / 0.1, axis=1) ** 2

    if isinstance(scatterers[0], AbstractElasticScattering):
        # factors has shape: (n_scatterers, n_doping, n_temperatures, n_kpts)
        factors = np.array([m.factor(k_diff_sq) for m in scatterers])
    else:
        # factors has shape: (n_scatterers, n_doping, n_temperatures, n_kpts)
        # first calculate scattering in rate
        to_stack = []
        if calculate_in_rate:
            in_factors = np.array([
                m.factor(spin, b_idx, full_k_idx, k_diff_sq, not emission,
                         out=False)
                for m in scatterers])
            in_factors *= (g[None, :, :, full_k_p_idx] *
                           k_angles[None, None, None, :])
            to_stack.append(in_factors)

        if calculate_out_rate:
            out_factors = np.array([
                m.factor(spin, b_idx, full_k_p_idx, k_diff_sq, emission,
                         out=True)
                for m in scatterers])
            to_stack.append(out_factors)

        factors = np.vstack(to_stack)

    rates = weighted_overlap * factors

    repeated_k_idx = repeated_k_idx[mask] - s.start
    unique_rows = np.unique(repeated_k_idx)

    # band rates has shape (n_scatterers, n_doping, n_temperatures, n_kpoints)
    band_rates = np.zeros((list(factors.shape[:-1]) + [s.stop - s.start]))

    # could vectorize this using numpy apply along axis if need be
    for s, n, t in np.ndindex(band_rates.shape[:-1]):
        band_rates[s, n, t, unique_rows] = np.bincount(
            repeated_k_idx, weights=rates[s, n, t])[unique_rows]

    return band_rates


def get_band_rates(spin, b_idx, scatterers, ediff, gauss_width, s, k_idx,
                   k_p_idx, kpoints, kpoint_norms, a_factor, c_factor,
                   reciprocal_lattice_matrix):
    from numpy.core.umath_tests import inner1d

    mask = k_idx != k_p_idx  # no self-scattering
    k_idx = k_idx[mask]
    k_p_idx = k_p_idx[mask]
    ediff = ediff[mask]

    k_dot = inner1d(kpoints[k_idx], kpoints[k_p_idx])
    k_angles = k_dot / (kpoint_norms[k_idx] * kpoint_norms[k_p_idx])
    k_angles[np.isnan(k_angles)] = 1.

    overlap = (a_factor[k_idx] * a_factor[k_p_idx] +
               c_factor[k_idx] * c_factor[k_p_idx] * k_angles) ** 2

    weighted_overlap = w0gauss(ediff / gauss_width) * overlap

    # norm of k difference squared in 1/nm
    k_diff_sq = np.linalg.norm(np.dot(
        kpoints[k_idx] - kpoints[k_p_idx],
        reciprocal_lattice_matrix) / 0.1, axis=1) ** 2

    # factors is array with shape:
    # (n_scatterers, n_doping, n_temperatures, n_k_diff_sq)
    factors = np.array([m.factor(k_diff_sq) for m in scatterers])

    rates = weighted_overlap * factors

    k_idx -= s.start
    unique_rows = np.unique(k_idx)

    # band rates has shape (n_scatterers, n_doping, n_temperatures, n_kpoints)
    band_rates = np.zeros((list(factors.shape[:-1]) + [s.stop - s.start]))

    # could vectorize this using numpy apply along axis if need be
    for s, n, t in np.ndindex(band_rates.shape[:-1]):
        band_rates[s, n, t, unique_rows] = np.bincount(
            k_idx, weights=rates[s, n, t])[unique_rows]

    return band_rates


def w0gauss(x):
    return np.exp(-(x**2)) * over_sqrt_pi


def calculate_g(out_rates, in_rates, elastic_rates, force):
    if elastic_rates is not None:
        out_rates = np.vstack((out_rates, elastic_rates))
    return (np.sum(in_rates, axis=0) - force) / (
        np.sum(out_rates, axis=0) + small_val)


def _interpolate_zero_rates(rates, kpoints, masks: Optional = None):
    # loop over all scattering types, doping, temps, and bands and interpolate
    # zero scattering rates based on the nearest k-point
    logger.info("Interpolating missing scattering rates")
    n_rates = sum([np.product(rates[spin].shape[:-1]) for spin in rates])
    pbar = tqdm(total=n_rates, ncols=output_width, desc="    ├── progress",
                bar_format='{l_bar}{bar}| {elapsed}<{remaining}{postfix}',
                file=sys.stdout)

    n_zero_rates = 0
    n_lots = 0
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
                n_lots += 1
                # interpolation seems to work best when all the kpoints are +ve
                # therefore add 0.5
                # Todo: Use cartesian coordinates (will be more robust to
                # oddly shaped cells)
                rates[spin][s, d, t, b, zero_rate_idx] = griddata(
                    points=kpoints[non_zero_rate_idx] + 0.5,
                    values=rates[spin][s, d, t, b, non_zero_rate_idx],
                    xi=kpoints[zero_rate_idx] + 0.5, method='nearest')

            pbar.update()
    pbar.close()
    log_time_taken(t0)

    if n_zero_rates > 0:
        logger.warning("WARNING: N zero rates: {:.0f}".format(
            n_zero_rates/n_lots))

    return rates
