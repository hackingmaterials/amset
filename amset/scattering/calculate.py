"""
This module implements methods to calculate electron scattering based on an
AmsetData object.
"""
import logging
import math
import sys
import time
from multiprocessing import cpu_count, Process, Queue
from typing import Dict, Union, List, Any

import numpy as np
from monty.json import MSONable
from scipy.interpolate import griddata
from sklearn.neighbors.ball_tree import BallTree
from tqdm import tqdm

from BoltzTraP2 import units
from pymatgen import Spin

from amset.constants import A_to_nm, output_width, over_sqrt_pi, hbar, \
    default_small_e, small_val, e
from amset.data import AmsetData
from amset.log import log_list
from amset.scattering.elastic import AbstractElasticScattering
from amset.scattering.inelastic import AbstractInelasticScattering
from amset.util import spin_name, create_shared_array, gen_even_slices

logger = logging.getLogger(__name__)


class ScatteringCalculator(MSONable):

    def __init__(self,
                 materials_properties: Dict[str, float],
                 amset_data: AmsetData,
                 scattering_type: Union[str, List[str], float] = "auto",
                 energy_tol: float = 0.001,
                 g_tol: float = 0.01,
                 max_g_iter: int = 8,
                 use_symmetry: bool = True,
                 nworkers: int = -1):
        if amset_data.temperatures is None or amset_data.doping is None:
            raise RuntimeError(
                "AmsetData doesn't contain doping levels or temperatures")

        self.scattering_type = scattering_type
        self.materials_properties = materials_properties
        self.energy_tol = energy_tol
        self.g_tol = g_tol
        self.max_g_iter = max_g_iter
        self.nworkers = nworkers if nworkers != -1 else cpu_count()
        self.use_symmetry = use_symmetry
        self.scatterers = self.get_scatterers(
            scattering_type, materials_properties, amset_data)
        self.amset_data = amset_data

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
        inelastic_names = ["{}_{}".format(s.name, d) for s in
                           self.inelastic_scatterers for d in ["in", "out"]]
        return elastic_names + inelastic_names

    @staticmethod
    def get_scatterers(scattering_type: Union[str, List[str], float],
                       materials_properties: Dict[str, Any],
                       amset_data: AmsetData
                       ) -> List[Union[AbstractElasticScattering,
                                       AbstractInelasticScattering]]:
        # dynamically determine the available scattering mechanism subclasses
        scattering_mechanisms = {
            m.name: m for m in
            AbstractElasticScattering.__subclasses__() +
            AbstractInelasticScattering.__subclasses__()}

        if scattering_type == "auto":
            logger.info("Examining material properties to determine possible "
                        "scattering mechanisms")

            scattering_type = [
                name for name, mechanism in scattering_mechanisms.items()
                if all([materials_properties.get(x, False) for x in
                        mechanism.required_properties])]

            if not scattering_type:
                raise ValueError("No scattering mechanisms possible with "
                                 "material properties")

        else:
            for name in scattering_type:
                missing_properties = [
                    p for p in scattering_mechanisms[name].required_properties
                    if not materials_properties.get(p, False)]

                if missing_properties:
                    raise ValueError(
                        "{} scattering mechanism specified but the following "
                        "material properties are missing: {}".format(
                            name, ", ".join(missing_properties)))

        logger.info("The following scattering mechanisms will be "
                    "calculated: {}".format(", ".join(scattering_type)))

        return [scattering_mechanisms[name](materials_properties, amset_data)
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

        if self.use_symmetry:
            nkpoints = len(self.amset_data.ir_kpoints_idx)
        else:
            nkpoints = len(full_kpoints)

        batch_size = min(500., 1 / (self.energy_tol * 2))
        nsplits = math.ceil(nkpoints/batch_size)
        logger.info("Scattering information:")
        log_list(["energy tolerance: {} eV".format(self.energy_tol),
                  "# k-points: {}".format(nkpoints),
                  "batch size: {}".format(batch_size)])

        for spin in spins:
            for b_idx in range(len(self.amset_data.energies[spin])):
                logger.info("Calculating rates for {} band {}".format(
                    spin_name[spin], b_idx + 1))

                t0 = time.perf_counter()
                rates[spin][:, :, :, b_idx, :] = self.calculate_band_rates(
                    spin, b_idx, nsplits)

                log_list([
                    "max rate: {:.4g}".format(rates[spin][..., b_idx, :].max()),
                    "min rate: {:.4g}".format(rates[spin][..., b_idx, :].min()),
                    "time: {:.4f} s".format(time.perf_counter() - t0)])

        # if the k-point density is low, some k-points may not have
        # other k-points within the energy tolerance leading to zero rates
        rates = _interpolate_zero_rates(rates, full_kpoints)

        return rates

    def calculate_band_rates(self,
                             spin: Spin,
                             b_idx: int,
                             nsplits: int):
        integral_conversion = (
                (2 * np.pi) ** 3
                / (self.amset_data.structure.lattice.volume * A_to_nm ** 3)
                / (len(self.amset_data.full_kpoints) * self.energy_tol))

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
        ball_tree = BallTree(band_energies[:, None], leaf_size=100)
        g = np.ones(self.amset_data.fermi_levels.shape +
                    (len(self.amset_data.energies[spin][b_idx]), )) * small_val

        s_g, g = create_shared_array(g, return_buffer=True)

        s_energies = create_shared_array(band_energies)
        s_kpoints = create_shared_array(self.amset_data.full_kpoints)
        s_k_norms = create_shared_array(self.amset_data.kpoint_norms)
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
                    self.energy_tol * units.eV,
                    s_g, s_energies, s_kpoints, s_k_norms, s_a_factor,
                    s_c_factor, len(band_energies), rlat, iqueue, oqueue)
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
        if self.elastic_scatterers:
            elastic_rates = self._fill_workers(
                nkpoints, slices, 0., False, iqueue, oqueue, desc="elastic")
            elastic_rates *= elastic_prefactors[..., None]

        if self.inelastic_scatterers:
            # currently only supports one inelastic scattering energy difference
            # convert frequency to THz and get energy in Rydberg
            energy_diff = (self.materials_properties["pop_frequency"] * 1e12
                           * 2 * np.pi * hbar * units.eV)

            first_run = True
            n_inelastic = len(self.inelastic_scatterers)
            shape = (n_inelastic, len(self.amset_data.doping),
                     len(self.amset_data.temperatures), nkpoints)
            in_rates = np.zeros(shape)
            out_rates = np.zeros(shape)

            # in 1/s
            force = (self.amset_data.dfdk[spin][:, :, b_idx] *
                     default_small_e / hbar)
            # force = 0
            force = np.zeros(force.shape)
            # force = small_val * 2

            for _ in range(self.max_g_iter):
                # rates are formatted as s1_i, s1_i, s2_o, s2_o etc
                inelastic_rates = self._fill_workers(
                    nkpoints, slices, energy_diff, first_run, iqueue, oqueue,
                    desc="inelastic")
                in_rates = inelastic_rates[:n_inelastic]
                in_rates *= inelastic_prefactors[..., None]
                # print("overall {} & are zero".format(
                #     sum(in_rates[0, 0, 7] == 0) * 100 / len(in_rates[0, 0, 7])))
                # print("in rates", in_rates.mean())
                in_rates = _interpolate_band_rates(
                    in_rates, self.amset_data.full_kpoints)

                if first_run:
                    # in rate is indepent of g so only need to calculate it once
                    # rates include both in and out scattering
                    out_rates = inelastic_rates[n_inelastic:]
                    out_rates *= inelastic_prefactors[..., None]
                    # print("out rates", out_rates.mean())
                    first_run = False
                    out_rates = _interpolate_band_rates(
                        out_rates, self.amset_data.full_kpoints)

                new_g = calculate_g(out_rates, in_rates, elastic_rates, force)
                g_diff = np.abs(np.average(new_g-g))
                logger.debug("  ├── difference in g value: {:.2g}".format(
                    g_diff))
                if g_diff < self.g_tol:
                    break

                # print("new_g", new_g[0, 0, 2:5])
                # print("in_rates", in_rates[0, 0, 0, 2:5])
                # print("out_rates", out_rates[0, 0, 0, 2:5])
                # print("f0", self.amset_data.f[spin][0, 0, b_idx, 2:5])
                # print("prefactor", inelastic_prefactors[0, 0, 0])
                # print("emission", self.inelastic_scatterers[0].emission_f[spin][0, 0, b_idx, 2:5])
                # print("absorption", self.inelastic_scatterers[0].absorption_f[spin][0, 0, b_idx, 2:5])
                # print("force", force[0, 0, 2:5])
                # # print(new_g.shape)
                # print("new_g", new_g[0, 7, 2:5])
                print("in_rates", in_rates[0, 0, 7, 2:5])
                # print("out_rates", out_rates[0, 0, 7, 2:5])
                # print("f0", self.amset_data.f[spin][0, 7, b_idx, 2:5])
                # print("prefactor", inelastic_prefactors[0, 0, 7])
                # print("emission", self.inelastic_scatterers[0].emission_f[spin][0, 0, b_idx, 2:5])
                # print("absorption", self.inelastic_scatterers[0].absorption_f[spin][0, 7, b_idx, 2:5])
                # print("force", force[0, 7, 2:5])
                # sys.exit()

                # print("f0", self.amset_data.f[spin][0, 7, b_idx, 0:100])

                # update the shared buffer
                g[:] = new_g[:]
                # print(g)

            # print("average(g!=0)", np.average(g[g!=0]))
            if elastic_rates is not None:
                # print("el shape", elastic_rates.shape)
                # print("out shape", out_rates.shape)
                # print("in shape", in_rates.shape)
                all_band_rates = np.vstack((elastic_rates, in_rates, out_rates))
            else:
                all_band_rates = np.vstack((in_rates, out_rates))
        else:
            all_band_rates = elastic_rates

        # The "None"s at the end of the queue signal the workers that there are
        # no more jobs left and they must therefore exit.
        for i in range(self.nworkers):
            iqueue.put(None)

        for w in workers:
            w.join()
            w.terminate()

        return all_band_rates

    def _fill_workers(self, nkpoints, slices, energy_diff,
                      calculate_out_rate, iqueue, oqueue, desc="progress"):
        if energy_diff:
            # inelastic scattering
            n_scat = len(self.inelastic_scatterers)
            n_scat *= 2 if calculate_out_rate else 1
        else:
            # elastic scattering
            n_scat = len(self.elastic_scatterers)

        band_rates = np.zeros(
            (n_scat, len(self.amset_data.doping),
             len(self.amset_data.temperatures), nkpoints))

        for s in slices:
            iqueue.put((s, energy_diff, calculate_out_rate))

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


def scattering_worker(scatterers, ball_tree, spin, b_idx, energy_tol, s_g,
                      senergies, skpoints, skpoint_norms, sa_factor, sc_factor,
                      nkpoints, reciprocal_lattice_matrix, iqueue, oqueue,
                      ir_kpoints_idx=None, grouped_ir_to_full=None):
    ntemps = scatterers[0].temperatures.shape[0]
    ndoping = scatterers[0].doping.shape[0]
    energies = np.frombuffer(senergies).reshape(nkpoints)
    kpoints = np.frombuffer(skpoints).reshape(-1, 3)
    kpoint_norms = np.frombuffer(skpoint_norms)
    a_factor = np.frombuffer(sa_factor).reshape(nkpoints)
    c_factor = np.frombuffer(sc_factor).reshape(nkpoints)
    g = np.frombuffer(s_g).reshape(ndoping, ntemps, -1)

    elastic_scatterers = [s for s in scatterers
                          if isinstance(s, AbstractElasticScattering)]
    inelastic_scatterers = [s for s in scatterers
                            if isinstance(s, AbstractInelasticScattering)]

    while True:
        job = iqueue.get()

        if job is None:
            break
        s, energy_diff, calculate_out_rate = job

        if energy_diff == 0:
            # elastic scattering
            k_p_idx, ediff = ball_tree.query_radius(
                energies[s, None], energy_tol * 5, return_distance=True)

            k_idx = np.repeat(np.arange(len(k_p_idx)),
                              [len(a) for a in k_p_idx])
            k_p_idx = np.concatenate(k_p_idx)

            ediff = np.concatenate(ediff)

            k_idx += s.start

            if nkpoints != kpoints.shape[0]:
                # working with symmetry reduced k-points
                band_rates = get_ir_band_rates(
                    spin, b_idx, elastic_scatterers, ediff, energy_tol, s,
                    k_idx, k_p_idx,
                    kpoints, kpoint_norms, a_factor, c_factor,
                    reciprocal_lattice_matrix, ir_kpoints_idx,
                    grouped_ir_to_full)
            else:
                # no symmetry, use the full BZ mesh
                band_rates = get_band_rates(
                    spin, b_idx, elastic_scatterers, ediff, energy_tol, s,
                    k_idx, k_p_idx,
                    kpoints, kpoint_norms, a_factor, c_factor,
                    reciprocal_lattice_matrix)

        else:
            # sum scattering in and out rates over emission and absorption

            if calculate_out_rate:
                # should have shape ((nscatters * 2), ndops, ntemps, nkpts)
                # *2 because have s_in and s_out for each scatterer
                band_rates = np.zeros(
                    (len(inelastic_scatterers) * 2, ndoping, ntemps,
                     s.stop - s.start))
            else:
                # only have in rate
                band_rates = np.zeros(
                    (len(inelastic_scatterers), ndoping, ntemps,
                     s.stop-s.start))

            # inelastic scattering
            for diff in [energy_diff, -energy_diff]:
                k_p_idx, ediff = ball_tree.query_radius(
                    energies[s, None] - diff, energy_tol * 5,
                    return_distance=True)
                # print("{} % have no scattering patner: ".format(
                #      100 * len([x for x in k_p_idx if len(x) == 0]) / len(k_p_idx)))

                k_idx = np.repeat(np.arange(len(k_p_idx)),
                                  [len(a) for a in k_p_idx])
                k_p_idx = np.concatenate(k_p_idx)

                ediff = np.concatenate(ediff)
                k_idx += s.start

                if nkpoints != kpoints.shape[0]:
                    # working with symmetry reduced k-points
                    # now format of band rates is (s1_i, s2_i, s1_o, s2_o...)
                    band_rates += get_ir_band_rates(
                        spin, b_idx, inelastic_scatterers, ediff, energy_tol, s,
                        k_idx, k_p_idx, kpoints, kpoint_norms, a_factor,
                        c_factor, reciprocal_lattice_matrix, ir_kpoints_idx,
                        grouped_ir_to_full, g=g, emission=diff <= 0,
                        calculate_out_rate=calculate_out_rate)
                else:
                    # no symmetry, use the full BZ mesh
                    band_rates += get_band_rates(
                        spin, b_idx, inelastic_scatterers, ediff, energy_tol, s,
                        k_idx,  k_p_idx, kpoints, kpoint_norms, a_factor,
                        c_factor, reciprocal_lattice_matrix, g=g,
                        emission=diff >= 0,
                        calculate_out_rate=calculate_out_rate)

        oqueue.put((s, band_rates))


def get_ir_band_rates(spin, b_idx, scatterers, ediff, energy_tol, s, k_idx,
                      k_p_idx, kpoints,
                      kpoint_norms, a_factor, c_factor,
                      reciprocal_lattice_matrix, ir_kpoints_idx,
                      grouped_ir_to_full, g=None, emission=False,
                      calculate_out_rate=True):
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
    k_angles = k_dot / (kpoint_norms[full_k_idx] * kpoint_norms[full_k_p_idx])
    k_angles[np.isnan(k_angles)] = 1.

    a_vals = a_factor[k_idx] * a_factor[k_p_idx]
    c_vals = c_factor[k_idx] * c_factor[k_p_idx]

    overlap = (a_vals[expand_k_idx] + c_vals[expand_k_idx] * k_angles) ** 2
    weighted_overlap = w0gauss(ediff / energy_tol)[expand_k_idx] * overlap

    # norm of k difference squared in 1/nm
    k_diff_sq = np.linalg.norm(np.dot(
        kpoints[full_k_idx] - kpoints[full_k_p_idx],
        reciprocal_lattice_matrix) / 0.1, axis=1) ** 2

    if isinstance(scatterers[0], AbstractElasticScattering):
        # factors has shape: (n_scatterers, n_doping, n_temperatures, n_kpts)
        factors = np.array([m.factor(k_diff_sq) for m in scatterers])
    else:
        # factors has shape: (n_scatterers, n_doping, n_temperatures, n_kpts)
        # first calculate scattering in rate
        factors = np.array([
            m.factor(spin, b_idx, full_k_idx, k_diff_sq, not emission)
            for m in scatterers])
        factors *= g[None, :, :, full_k_p_idx] * k_angles[None, None, None, :]

        if calculate_out_rate:
            factors_out = np.array([
                m.factor(spin, b_idx, full_k_p_idx, k_diff_sq, emission)
                for m in scatterers])

            # factors now has shape: (nscat*2, n_doping, n_temperatures, n_kpts)
            factors = np.vstack((factors, factors_out))

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


def get_band_rates(spin, b_idx, scatterers, ediff, energy_tol, s, k_idx,
                   k_p_idx, kpoints, kpoint_norms, a_factor, c_factor,
                   reciprocal_lattice_matrix):
    from numpy.core.umath_tests import inner1d

    mask = k_idx != k_p_idx
    k_idx = k_idx[mask]
    k_p_idx = k_p_idx[mask]
    ediff = ediff[mask]

    k_dot = inner1d(kpoints[k_idx], kpoints[k_p_idx])
    k_angles = k_dot / (kpoint_norms[k_idx] * kpoint_norms[k_p_idx])
    k_angles[np.isnan(k_angles)] = 1.

    overlap = (a_factor[k_idx] * a_factor[k_p_idx] +
               c_factor[k_idx] * c_factor[k_p_idx] * k_angles) ** 2

    weighted_overlap = w0gauss(ediff / energy_tol) * overlap

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
    return (np.sum(in_rates, axis=0) - force + small_val) / (
        np.sum(out_rates, axis=0) + small_val)


def _interpolate_zero_rates(rates, kpoints):
    # loop over all scattering types, doping, temps, and bands and interpolate
    # zero scattering rates based on the nearest k-point
    # for spin in rates:
    #     rates[spin][rates[spin] == 0] = 1e14

    # for spin in rates:
    #     rates[spin][rates[spin] == 0] += small_val

    nzero_rates = 0
    total_kpoints = 0
    for spin in rates:
        for s, d, t, b in np.ndindex(rates[spin].shape[:-1]):
            non_zero_rates = rates[spin][s, d, t, b] == 0.

            if any(non_zero_rates):
                nzero_rates += sum(non_zero_rates)
                rates[spin][s, d, t, b] = griddata(
                    points=kpoints[~non_zero_rates],
                    values=rates[spin][s, d, t, b, ~non_zero_rates],
                    xi=kpoints, method='nearest')

        total_kpoints += np.prod(rates[spin].shape)

    if nzero_rates > 0:
        logger.warning("Warning: {:.4f} % of k-points had zero scattering rate."
                       " Increase interpolation_factor and check results for "
                       "convergence".format(nzero_rates * 100 / total_kpoints))

    return rates

def _interpolate_band_rates(rates, kpoints):
    # loop over all scattering types, doping, temps, and bands and interpolate
    # zero scattering rates based on the nearest k-point
    # for spin in rates:
    #     rates[spin][rates[spin] == 0] = 1e14
    # print(kpoints.shape)

    for s, d, t in np.ndindex(rates.shape[:-1]):
        non_zero_rates = rates[s, d, t] == 0.
        # print(len(non_zero_rates))

        if any(non_zero_rates):
            rates[s, d, t] = griddata(
                points=kpoints[~non_zero_rates],
                values=rates[s, d, t, ~non_zero_rates],
                xi=kpoints, method='nearest')
    #
    # if nzero_rates > 0:
    #     logger.warning("Warning: {:.4f} % of k-points had zero scattering rate."
    #                    " Increase interpolation_factor and check results for "
    #                    "convergence".format(nzero_rates * 100 / total_kpoints))

    return rates
