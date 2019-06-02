"""
This module implements methods to calculate electron scattering based on an
AmsetData object.
"""

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

from amset.constants import A_to_nm, output_width, over_sqrt_pi
from amset.data import AmsetData
from amset.log import log_list
from amset.scattering.elastic import logger, AbstractElasticScattering
from amset.util import spin_name, create_shared_array, gen_even_slices


class ScatteringCalculator(MSONable):

    def __init__(self,
                 materials_properties: Dict[str, float],
                 amset_data: AmsetData,
                 scattering_type: Union[str, List[str], float] = "auto",
                 energy_tol: float = 0.001,
                 g_tol: float = 0.01,
                 use_symmetry: bool = True,
                 nworkers: int = -1):
        if amset_data.temperatures is None or amset_data.doping is None:
            raise RuntimeError(
                "AmsetData doesn't contain doping levels or temperatures")

        self.scattering_type = scattering_type
        self.materials_properties = materials_properties
        self.energy_tol = energy_tol
        self.g_tol = g_tol
        self.nworkers = nworkers if nworkers != -1 else cpu_count()
        self.use_symmetry = use_symmetry
        self.scatterers = self.get_scatterers(
            scattering_type, materials_properties, amset_data)
        self.amset_data = amset_data

    @staticmethod
    def get_scatterers(scattering_type: Union[str, List[str], float],
                       materials_properties: Dict[str, Any],
                       amset_data: AmsetData
                       ) -> List[AbstractElasticScattering]:
        # dynamically determine the available scattering mechanism subclasses
        scattering_mechanisms = {
            m.name: m for m in AbstractElasticScattering.__subclasses__()}

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

        # prefactors now has shape (spin, nscatterers, ndoping, ntemp, nbands)
        prefactors = [m.prefactor() for m in self.scatterers]
        prefactors = {s: np.array([prefactor[s] for prefactor in prefactors])
                      for s in spins}

        # rates has shape (spin, nscatterers, ndoping, ntemp, nbands, nkpoints)
        rates = {s: np.zeros(prefactors[s].shape + (len(full_kpoints), ))
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

        integral_conversion = (2 * np.pi) ** 3 / (
            self.amset_data.structure.lattice.volume *
            A_to_nm ** 3) / (len(full_kpoints) * self.energy_tol)
        integral_conversion = {s: integral_conversion * prefactors[s]
                               for s in spins}

        for spin in spins:
            for b_idx in range(len(self.amset_data.energies[spin])):
                logger.info("Calculating rates for {} band {}".format(
                    spin_name[spin], b_idx + 1))

                t0 = time.perf_counter()
                rates[spin][:, :, :, b_idx, :] = self.calculate_band_rates(
                    spin, b_idx, nsplits)

                log_list([
                    "max rate: {:.4g}".format(
                        (rates[spin][:, :, :, b_idx] *
                         integral_conversion[
                             spin][:, :, :, b_idx, None]).max()),
                    "min rate: {:.4g}".format(
                        (rates[spin][:, :, :, b_idx] *
                         integral_conversion[
                             spin][:, :, :, b_idx, None]).min()),
                    "time: {:.4f} s".format(time.perf_counter() - t0)])

            rates[spin] *= integral_conversion[spin][:, :, :, :, None]

        # if the k-point density is low, some k-points may not have
        # other k-points within the energy tolerance leading to zero rates
        rates = _interpolate_zero_rates(rates, full_kpoints)

        return rates

    def calculate_band_rates(self,
                             spin: Spin,
                             b_idx: int,
                             nsplits: int,
                             energy_diff: float = 0):
        if self.use_symmetry:
            kpoints_idx = self.amset_data.ir_kpoints_idx
        else:
            kpoints_idx = np.arange(len(self.amset_data.full_kpoints))

        nkpoints = len(kpoints_idx)

        band_energies = self.amset_data.energies[spin][b_idx, kpoints_idx]
        ball_tree = BallTree(band_energies[:, None], leaf_size=100)

        s_energies = create_shared_array(band_energies)
        s_kpoints = create_shared_array(self.amset_data.full_kpoints)
        s_k_norms = create_shared_array(self.amset_data.kpoint_norms)
        s_a_factor = create_shared_array(
            self.amset_data.a_factor[spin][b_idx, kpoints_idx])
        s_c_factor = create_shared_array(
            self.amset_data.c_factor[spin][b_idx, kpoints_idx])

        red_band_rates = np.zeros(
            (len(self.scatterers), len(self.amset_data.doping),
             len(self.amset_data.temperatures), nkpoints))

        rlat = self.amset_data.structure.lattice.reciprocal_lattice.matrix

        # spawn as many worker processes as needed, put all bands in the queue,
        # and let them work until all the required rates have been computed.
        workers = []
        iqueue = Queue()
        oqueue = Queue()
        slices = list(gen_even_slices(nkpoints, nsplits))

        for s in slices:
            iqueue.put(s)

        # The "None"s at the end of the queue signal the workers that there are
        # no more jobs left and they must therefore exit.
        for i in range(self.nworkers):
            iqueue.put(None)

        for i in range(self.nworkers):
            args = (self.scatterers, ball_tree, self.energy_tol * units.eV,
                    energy_diff, s_energies, s_kpoints, s_k_norms, s_a_factor,
                    s_c_factor, len(band_energies), rlat, iqueue, oqueue)
            if self.use_symmetry:
                kwargs = {
                    "grouped_ir_to_full": self.amset_data.grouped_ir_to_full,
                    "ir_kpoints_idx": self.amset_data.ir_kpoints_idx}

                workers.append(Process(target=scattering_worker, args=args,
                                       kwargs=kwargs))
            else:
                workers.append(Process(target=scattering_worker, args=args))

        for w in workers:
            w.start()

        # The results are processed as soon as they are ready.

        desc = "    ├── progress".format(spin_name[spin], b_idx + 1)
        pbar = tqdm(total=nkpoints, ncols=output_width, desc=desc,
                    bar_format='{l_bar}{bar}| {elapsed}<{remaining}{postfix}',
                    file=sys.stdout)
        for _ in range(nsplits):
            s, red_band_rates[:, :, :, s] = oqueue.get()
            pbar.update(s.stop - s.start)
        pbar.close()

        for w in workers:
            w.join()

        if self.use_symmetry:
            all_band_rates = red_band_rates[
                :, :, :, self.amset_data.ir_to_full_kpoint_mapping]

        else:
            all_band_rates = red_band_rates

        return all_band_rates


def scattering_worker(scatterers, ball_tree, energy_tol, energy_diff,
                      senergies, skpoints, skpoint_norms, sa_factor, sc_factor,
                      nkpoints, reciprocal_lattice_matrix, iqueue, oqueue,
                      ir_kpoints_idx=None, grouped_ir_to_full=None):
    energies = np.frombuffer(senergies).reshape(nkpoints)
    kpoints = np.frombuffer(skpoints).reshape(-1, 3)
    kpoint_norms = np.frombuffer(skpoint_norms)
    a_factor = np.frombuffer(sa_factor).reshape(nkpoints)
    c_factor = np.frombuffer(sc_factor).reshape(nkpoints)

    while True:
        s = iqueue.get()

        if s is None:
            break

        k_p_idx, ediff = ball_tree.query_radius(
            energies[s, None], energy_tol * 5, return_distance=True)

        k_idx = np.repeat(np.arange(len(k_p_idx)), [len(a) for a in k_p_idx])
        k_p_idx = np.concatenate(k_p_idx)

        ediff = np.concatenate(ediff)

        k_idx += s.start

        if nkpoints != kpoints.shape[0]:
            # working with symmetry reduced k-points
            band_rates = get_ir_band_rates(
                scatterers, ediff, energy_tol, s, k_idx,  k_p_idx,
                kpoints, kpoint_norms, a_factor, c_factor,
                reciprocal_lattice_matrix, ir_kpoints_idx, grouped_ir_to_full)
        else:
            # no symmetry, use the full BZ mesh
            band_rates = get_band_rates(
                scatterers, ediff, energy_tol, s, k_idx,  k_p_idx,
                kpoints, kpoint_norms, a_factor, c_factor,
                reciprocal_lattice_matrix)

        oqueue.put((s, band_rates))


def get_ir_band_rates(scatterers, ediff, energy_tol, s, k_idx, k_p_idx, kpoints,
                      kpoint_norms, a_factor, c_factor,
                      reciprocal_lattice_matrix, ir_kpoints_idx,
                      grouped_ir_to_full):
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

    # factors has shape: (n_scatterers, n_doping, n_temperatures, n_k_diff_sq)
    factors = np.array([m.factor(k_diff_sq) for m in scatterers])

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


def get_band_rates(scatterers, ediff, energy_tol, s, k_idx, k_p_idx,
                   kpoints, kpoint_norms, a_factor, c_factor,
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


def _interpolate_zero_rates(rates, kpoints):
    # loop over all scattering types, doping, temps, and bands and interpolate
    # zero scattering rates based on the nearest k-point
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



