"""
This module implements methods to calculate electron scattering based on an
AmsetData object.
"""

import inspect
import logging
import math
import sys
import time

from abc import ABC, abstractmethod
from multiprocessing import Queue, Process, cpu_count

import numpy as np

from typing import Union, List, Dict, Tuple, Any

from BoltzTraP2 import units
from monty.json import MSONable
from scipy.constants import epsilon_0
from scipy.interpolate import griddata
from scipy.integrate import trapz
from sklearn.neighbors.ball_tree import BallTree
from sklearn.utils import gen_even_slices
from tqdm import tqdm

from amset.constants import k_B, e, hbar, over_sqrt_pi, A_to_nm, output_width
from amset.data import AmsetData
from amset.util import create_shared_array, spin_name
from amset.log import log_list
from amset.utils.transport import f0
from pymatgen import Spin
from pymatgen.util.coord import pbc_diff

logger = logging.getLogger(__name__)


class ScatteringCalculator(MSONable):

    def __init__(self,
                 materials_properties: Dict[str, float],
                 scattering_type: Union[str, List[str], float] = "auto",
                 energy_tol: float = 0.001,
                 g_tol: float = 0.01,
                 use_symmetry: bool = True,
                 nworkers: int = -1):
        self.scattering_type = scattering_type
        self.materials_properties = materials_properties
        self.energy_tol = energy_tol
        self.g_tol = g_tol
        self.nworkers = nworkers if nworkers != -1 else cpu_count()
        self.use_symmetry = use_symmetry
        self.scatterers = get_scatterers(scattering_type, materials_properties)

    def calculate_scattering_rates(self,
                                   amset_data: AmsetData,
                                   ):
        # initialize scattering mechanims
        [m.initialize(amset_data) for m in self.scatterers]
        prefactors = [m.prefactor(amset_data) for m in self.scatterers]

        # prefactors now has shape (spin, nscatterers, ndoping, ntemp, nbands)
        prefactors = {s: np.array([prefactor[s] for prefactor in prefactors])
                      for s in amset_data.spins}

        # rates has shape (spin, nscatterers, ndoping, ntemp, nbands, nkpoints)
        rates = {
            s: np.zeros(prefactors[s].shape + (len(amset_data.full_kpoints), ))
            for s in amset_data.spins}

        if self.use_symmetry:
            nkpoints = len(amset_data.ir_kpoints_idx)
        else:
            nkpoints = len(amset_data.full_kpoints)

        batch_size = min(500., 1 / (self.energy_tol * 2))
        nsplits = math.ceil(nkpoints/batch_size)
        logger.info("Scattering information:")
        log_list(["energy tolerance: {} eV".format(self.energy_tol),
                  "# k-points: {}".format(nkpoints),
                  "batch size: {}".format(batch_size)])

        integral_conversion = (2 * np.pi) ** 3 / (
            amset_data.structure.lattice.volume *
            A_to_nm ** 3) / (len(amset_data.full_kpoints) * self.energy_tol)
        integral_conversion = {s: integral_conversion * prefactors[s]
                               for s in amset_data.spins}

        for spin in amset_data.spins:
            for b_idx in range(len(amset_data.energies[spin])):
                logger.info("Calculating rates for {} band {}".format(
                    spin_name[spin], b_idx + 1))

                t0 = time.perf_counter()
                rates[spin][:, :, :, b_idx, :] = self.calculate_band_rates(
                    spin, b_idx, nsplits, amset_data)

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
        rates = _interpolate_zero_rates(rates, amset_data.full_kpoints)

        return rates

    def calculate_band_rates(self,
                             spin: Spin,
                             b_idx: int,
                             nsplits: int,
                             amset_data: AmsetData,
                             energy_diff: float = 0):
        if self.use_symmetry:
            kpoints_idx = amset_data.ir_kpoints_idx
        else:
            kpoints_idx = np.arange(len(amset_data.full_kpoints))

        nkpoints = len(kpoints_idx)

        band_energies = amset_data.energies[spin][b_idx, kpoints_idx]
        ball_tree = BallTree(band_energies[:, None], leaf_size=100)

        s_energies = create_shared_array(band_energies)
        s_kpoints = create_shared_array(amset_data.full_kpoints)
        s_k_norms = create_shared_array(amset_data.kpoint_norms)
        s_a_factor = create_shared_array(
            amset_data.a_factor[spin][b_idx, kpoints_idx])
        s_c_factor = create_shared_array(
            amset_data.c_factor[spin][b_idx, kpoints_idx])

        red_band_rates = np.zeros(
            (len(self.scatterers), len(amset_data.doping),
             len(amset_data.temperatures), nkpoints))

        rlat = amset_data.structure.lattice.reciprocal_lattice.matrix

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
                    s_c_factor, len(band_energies), rlat, amset_data.doping,
                    amset_data.temperatures, iqueue, oqueue)
            if self.use_symmetry:
                kwargs = {"grouped_ir_to_full": amset_data.grouped_ir_to_full,
                          "ir_kpoints_idx": amset_data.ir_kpoints_idx}
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
                :, :, :, amset_data.ir_to_full_kpoint_mapping]

        else:
            all_band_rates = red_band_rates

        return all_band_rates


class AbstractScatteringMechanism(ABC):

    name: str
    required_properties: Tuple[str]
    inelastic = False

    def initialize(self, amset_data):
        pass

    @abstractmethod
    def prefactor(self, amset_data: AmsetData):
        pass

    @abstractmethod
    def factor(self, *args):
        pass


class AcousticDeformationPotentialScattering(AbstractScatteringMechanism):

    name = "ACD"
    required_properties = ("deformation_potential", "elastic_constant")

    def __init__(self,
                 deformation_potential: Union[Tuple[float, float], float],
                 elastic_constant: float):
        self.deformation_potential: Union[Dict, float] = deformation_potential
        self.elastic_constant = elastic_constant

    def prefactor(self, amset_data: AmsetData):
        is_metal = amset_data.is_metal
        prefactor = {s: (1e18 * e * k_B / (4.0 * np.pi ** 2 * hbar *
                                           self.elastic_constant))
                     for s in amset_data.spins}

        for spin in amset_data.spins:
            prefactor[spin] *= np.ones(
                (len(amset_data.doping), len(amset_data.temperatures),
                 len(amset_data.energies[spin]))
                ) * amset_data.temperatures[None, :, None]

            if is_metal and isinstance(self.deformation_potential, tuple):
                logger.warning(
                    "System is metallic but deformation potentials for both "
                    "the valence and conduction bands have been set. Using the "
                    "valence band value for all bands")
                prefactor[spin] *= self.deformation_potential[0]

            elif not is_metal and isinstance(self.deformation_potential, tuple):
                cb_idx = amset_data.vb_idx[spin] + 1
                prefactor[spin][:, :, :cb_idx] *= \
                    self.deformation_potential[0] ** 2
                prefactor[spin][:, :, cb_idx:] *= \
                    self.deformation_potential[1] ** 2

            elif not is_metal:
                logger.warning(
                    "System is semiconducting but only one deformation "
                    "potential has been set. Using this value for all bands.")
                prefactor[spin] *= self.deformation_potential ** 2

            else:
                prefactor[spin] *= self.deformation_potential ** 2

        return prefactor

    def factor(self, doping: np.ndarray, temperatures: np.ndarray,
               k_diff_sq: np.ndarray):
        return np.ones((len(doping), len(temperatures), k_diff_sq.shape[0]))


class IonizedImpurityScattering(AbstractScatteringMechanism):

    name = "IMP"
    required_properties = ("acceptor_charge", "donor_charge",
                           "static_dielectric")

    def __init__(self,
                 acceptor_charge: int,
                 donor_charge: int,
                 static_dielectric: float):
        self.acceptor_charge = acceptor_charge
        self.donor_charge = donor_charge
        self.static_dielectric = static_dielectric
        self.beta_sq = None
        self.impurity_concentration = None

    def initialize(self, amset_data):
        logger.debug("Initializing IMP scattering")

        self.beta_sq = np.zeros(amset_data.fermi_levels.shape)
        self.impurity_concentration = np.zeros(amset_data.fermi_levels.shape)

        tdos = amset_data.dos.tdos
        energies = amset_data.dos.energies
        fermi_levels = amset_data.fermi_levels
        de = amset_data.dos.de
        v_idx = amset_data.dos.idx_vbm
        c_idx = amset_data.dos.idx_cbm
        vol = amset_data.structure.volume

        # 1e-8 is Angstrom to cm conversion
        conv = 1 / (vol * 1e-8 ** 3)

        imp_info = []
        for n, t in np.ndindex(self.beta_sq.shape):
            ef = fermi_levels[n, t]
            temp = amset_data.temperatures[t]
            f = f0(energies, ef, temp)
            integral = trapz(tdos * f * (1 - f), x=energies)
            self.beta_sq[n, t] = (
                e ** 2 * integral * 1e12 /
                (self.static_dielectric * epsilon_0 * k_B * temp * e * vol))

            # calculate impurity concentration
            n_conc = np.abs(conv * np.sum(
                tdos[c_idx:] * f0(energies[c_idx:], ef, temp) * de[c_idx:],
                axis=0))
            p_conc = np.abs(conv * np.sum(
                tdos[:v_idx + 1] * (1 - f0(energies[:v_idx + 1], ef, temp))
                * de[:v_idx + 1], axis=0))

            self.impurity_concentration[n, t] = (
                    n_conc * self.donor_charge ** 2 +
                    p_conc * self.acceptor_charge ** 2)
            imp_info.append(
                "{:.2g} cm⁻³ & {} K: β² = {:.4g}, Nᵢᵢ = {:.4g}".format(
                    amset_data.doping[n], temp, self.beta_sq[n, t],
                    self.impurity_concentration[n, t]))

        logger.debug("Inverse screening length (β) and impurity concentration "
                     "(Nᵢᵢ):")
        log_list(imp_info, level=logging.DEBUG)

    def prefactor(self, amset_data: AmsetData):
        prefactor = ((1e-3 / (e ** 2)) * e ** 4 * self.impurity_concentration /
                     (4.0 * np.pi ** 2 * self.static_dielectric ** 2 *
                      epsilon_0 ** 2 * hbar))

        # need to return prefactor with shape (nspins, ndops, ntemps, nbands)
        # currently it has shape (ndops, ntemps)
        return {s: np.repeat(prefactor[:, :, None], len(amset_data.energies[s]),
                             axis=-1)
                for s in amset_data.spins}

    def factor(self, doping: np.ndarray, temperatures: np.ndarray,
               k_diff_sq: np.ndarray):
        # tile k_diff_sq to make it commensurate with the dimensions of beta
        return 1 / (np.tile(k_diff_sq, (len(doping), len(temperatures), 1)) +
                    self.beta_sq[..., None]) ** 2


class PiezoelectricScattering(AbstractScatteringMechanism):

    name = "PIE"
    temperature_dependent = True
    required_properties = ("piezoelectric_coefficient", "static_dielectric")

    def __init__(self,
                 piezoelectric_coefficient: float,
                 static_dielectric: float,
                 *args):
        super().__init__(*args)
        self.piezoelectric_coefficient = piezoelectric_coefficient
        self.static_dielectric = static_dielectric

    @property
    def prefactor(self, amset_data: AmsetData):
        unit_conversion = 1e9 / e
        prefactor = {s: np.ones(amset_data.fermi_levels.shape +
                                amset_data.energies[s].shape[0])
                     for s in amset_data.spins}

        prefactor = (unit_conversion * e ** 2 * k_B * amset_data.temperatures *
            self.piezoelectric_coefficient ** 2 /
            (4.0 * np.pi ** 2 * hbar * epsilon_0 * self.static_dielectric))

        prefactor = ((1e-3 / (e ** 2)) * e ** 4 * self.impurity_concentration /
                     (4.0 * np.pi ** 2 * self.static_dielectric ** 2 *
                      epsilon_0 ** 2 * hbar))

        # need to return prefactor with shape (nspins, ndops, ntemps, nbands)
        # currently it has shape (ndops, ntemps)
        return {s: np.repeat(prefactor[:, :, None], len(amset_data.energies[s]),
                             axis=-1)
                for s in amset_data.spins}

    def factor(self, k_diff_sq):
        return 1


class PolarOpticalScattering(AbstractScatteringMechanism):

    name = "POP"
    temperature_dependent = False
    required_properties = ("pop_frequency", "static_dielectric",
                           "high_frequency_dielectric")
    inelastic = True

    def __init__(self,
                 pop_frequency: float,
                 static_dielectric: float,
                 high_frequency_dielectric: float,
                 *args):
        super().__init__(*args)
        self.pop_frequency = pop_frequency
        self.static_dielectric = static_dielectric
        self.high_frequency_dielectric = high_frequency_dielectric

    @property
    def prefactor(self):
        return 1

    def factor(self, k_diff_sq):
        return 1


def get_scatterers(scatttering_type: Union[str, List[str], float],
                   materials_properties: Dict[str, Any],
                   ) -> List[AbstractScatteringMechanism]:
    # dynamically determine the available scattering mechanism subclasses
    scattering_mechanisms = {
        obj.name: obj for _, obj in inspect.getmembers(sys.modules[__name__])
        if inspect.isclass(obj) and
        obj is not AbstractScatteringMechanism and
        issubclass(obj, AbstractScatteringMechanism)}

    if scatttering_type == "auto":
        logger.info("Examining material properties to determine possible "
                    "scattering mechanisms")

        scatttering_type = [
            name for name, mechanism in scattering_mechanisms.items()
            if all([materials_properties.get(x, False) for x in
                    mechanism.required_properties])]

        if not scatttering_type:
            raise ValueError(
                "No scattering mechanisms possible with material properties")

    else:
        for name in scatttering_type:
            missing_properties = [
                p for p in scattering_mechanisms[name].required_properties
                if not materials_properties.get(p, False)]

            if missing_properties:
                raise ValueError(
                    "{} scattering mechanism specified but the following "
                    "material properties are missing: {}".format(
                        name, ", ".join(missing_properties)))

    logger.info("The following scattering mechanisms will be "
                "calculated: {}".format(", ".join(scatttering_type)))

    scatterers = []
    for name in scatttering_type:
        mechanism = scattering_mechanisms[name]
        scatterers.append(mechanism(
            *[materials_properties[p] for p in mechanism.required_properties]))
    return scatterers


def scattering_worker(scatterers, ball_tree, energy_tol, energy_diff,
                      senergies, skpoints, skpoint_norms, sa_factor, sc_factor,
                      nkpoints, reciprocal_lattice_matrix, doping, temperatures,
                      iqueue, oqueue, ir_kpoints_idx=None,
                      grouped_ir_to_full=None):
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
                reciprocal_lattice_matrix, doping, temperatures,
                ir_kpoints_idx, grouped_ir_to_full)
        else:
            # no symmetry, use the full BZ mesh
            band_rates = get_band_rates(
                scatterers, ediff, energy_tol, s, k_idx,  k_p_idx,
                kpoints, kpoint_norms, a_factor, c_factor,
                reciprocal_lattice_matrix, doping, temperatures)

        oqueue.put((s, band_rates))


def get_ir_band_rates(scatterers, ediff, energy_tol, s, k_idx, k_p_idx, kpoints,
                      kpoint_norms, a_factor, c_factor,
                      reciprocal_lattice_matrix, doping, temperatures,
                      ir_kpoints_idx, grouped_ir_to_full):
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
    factors = np.array([m.factor(doping, temperatures, k_diff_sq)
                        for m in scatterers])

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
                   reciprocal_lattice_matrix, doping, temperatures):
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
    factors = np.array([m.factor(doping, temperatures, k_diff_sq)
                        for m in scatterers])

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
