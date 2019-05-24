"""
This module implements methods to calculate electron scattering based on an
ElectronStructure object.
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
from sklearn.neighbors.ball_tree import BallTree
from sklearn.utils import gen_even_slices
from tqdm import tqdm

from amset.constants import k_B, e, hbar, over_sqrt_pi, A_to_nm, output_width
from amset.core import ElectronicStructure
from amset.util import create_shared_array, spin_name, log_list
from pymatgen import Spin
from pymatgen.util.coord import pbc_diff

logger = logging.getLogger(__name__)


class ScatteringCalculator(MSONable):

    def __init__(self,
                 materials_properties: Dict[str, float],
                 doping: np.ndarray,
                 temperatures: np.ndarray,
                 scattering_type: Union[str, List[str], float] = "auto",
                 energy_tol: float = 0.001,
                 g_tol: float = 0.01,
                 use_symmetry: bool = True,
                 nworkers: int = -1):
        self.temperatures = temperatures
        self.doping = doping
        self.scattering_type = scattering_type
        self.materials_properties = materials_properties
        self.energy_tol = energy_tol
        self.g_tol = g_tol
        self.nworkers = nworkers if nworkers != -1 else cpu_count()
        self.use_symmetry = use_symmetry
        self.scatterers = get_scatterers(
            scattering_type, materials_properties, doping, temperatures)

    def calculate_scattering_rates(self,
                                   electronic_structure: ElectronicStructure,
                                   ):
        prefactors = np.array([m.prefactor for m in self.scatterers])

        # rates has shape (nscatterers, ndoping, ntemp, nbands, nkpoints)
        rates = {s: np.zeros(prefactors.shape +
                             electronic_structure.energies[s].shape)
                 for s in electronic_structure.spins}

        if self.use_symmetry:
            kpoints_idx = electronic_structure.ir_kpoints_idx
        else:
            kpoints_idx = np.arange(len(electronic_structure.full_kpoints))
        nkpoints = len(kpoints_idx)

        batch_size = min(500., 1 / (self.energy_tol * 2))
        nsplits = math.ceil(nkpoints/batch_size)
        logger.info("Scattering information:")
        log_list(["energy tolerance: {} eV".format(self.energy_tol),
                  "# k-points: {}".format(nkpoints),
                  "batch size: {}".format(batch_size)])

        integral_conversion = (2 * np.pi) ** 3 / (
            electronic_structure.structure.lattice.volume *
            A_to_nm ** 3) / (nkpoints * self.energy_tol)
        integral_conversion *= prefactors

        for spin in electronic_structure.spins:
            for b_idx in range(len(electronic_structure.energies[spin])):
                logger.info("Calculating rates for {} band {}".format(
                    spin_name[spin], b_idx + 1))

                t0 = time.perf_counter()
                rates[spin][:, :, :, b_idx, :] = self.calculate_band_rates(
                    spin, b_idx, kpoints_idx, nsplits, electronic_structure)

                log_list([
                    "max rate: {:.4g}".format(
                        (rates[spin][:, :, :, b_idx] *
                         integral_conversion[:, :, :, None, None]).max()),
                    "min rate: {:.4g}".format(
                        (rates[spin][:, :, :, b_idx] *
                         integral_conversion[:, :, :, None, None]).min()),
                    "time: {:.4f} s".format(time.perf_counter() - t0)])

            rates[spin] *= integral_conversion[:, :, :, None, None]

        # if the k-point density is low, some k-points may not have
        # other k-points within the energy tolerance leading to zero rates
        rates = _interpolate_zero_rates(
            rates, electronic_structure.full_kpoints)

        return rates

    def calculate_band_rates(self,
                             spin: Spin,
                             b_idx: int,
                             kpoints_idx: np.ndarray,
                             nsplits: int,
                             electronic_structure: ElectronicStructure,
                             energy_diff: float = 0):
        nkpoints = len(kpoints_idx)

        band_energies = electronic_structure.energies[spin][b_idx, kpoints_idx]
        ball_tree = BallTree(band_energies[:, None], leaf_size=100)

        s_energies = create_shared_array(band_energies)
        s_kpoints = create_shared_array(electronic_structure.full_kpoints)
        s_k_norms = create_shared_array(electronic_structure.kpoint_norms)
        s_a_factor = create_shared_array(
            electronic_structure.a_factor[spin][b_idx, kpoints_idx])
        s_c_factor = create_shared_array(
            electronic_structure.c_factor[spin][b_idx, kpoints_idx])

        red_band_rates = np.zeros(
            (len(self.scatterers), len(electronic_structure.doping),
             len(electronic_structure.temperatures), nkpoints))

        rlat = electronic_structure.structure.lattice.reciprocal_lattice.matrix

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
            workers.append(Process(
                target=scattering_worker,
                args=(self.scatterers, ball_tree, self.energy_tol * units.eV,
                      energy_diff, s_energies, s_kpoints, s_k_norms, s_a_factor,
                      s_c_factor, len(band_energies), rlat, iqueue, oqueue)))

        for w in workers:
            w.start()

        # The results are processed as soon as they are ready.

        desc = "    ├── Progress".format(
            spin_name[spin], b_idx + 1)
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
                :, :, :, electronic_structure.ir_to_full_kpoint_mapping]

        else:
            all_band_rates = red_band_rates

        return all_band_rates


class AbstractScatteringMechanism(ABC):

    name: str
    required_properties: Tuple[str]
    inelastic = False

    def __init__(self,
                 temperature_dependent: bool,
                 prefactor: float,
                 doping: np.ndarray,
                 temperatures: np.ndarray):
        self._doping = doping
        self._temperatures = temperatures
        self._temperature_dependent = temperature_dependent
        self._prefactor = prefactor

    @property
    def prefactor(self):
        if self._temperature_dependent:
            return (np.ones((len(self._doping), len(self._temperatures)))
                    * self._prefactor * self._temperatures[None, :])
        else:
            return (np.ones((len(self._doping), len(self._temperatures)))
                    * self._prefactor)

    @abstractmethod
    def factor(self, **kwargs):
        pass


class AcousticDeformationPotentialScattering(AbstractScatteringMechanism):

    name = "ACD"
    required_properties = ("deformation_potential", "elastic_constant")

    def __init__(self,
                 deformation_potential: float,
                 elastic_constant: float,
                 doping: np.ndarray,
                 temperatures: np.ndarray):
        self.deformation_potential = deformation_potential
        self.elastic_constant = elastic_constant
        prefactor = (1e18 * e * k_B * self.deformation_potential ** 2
                     / (4.0 * np.pi ** 2 * hbar * self.elastic_constant))
        super().__init__(True, prefactor, doping, temperatures)

    def factor(self, k_diff_sq):
        return np.ones((len(self._doping), len(self._temperatures),
                        k_diff_sq.shape[0]))


class IonizedImpurityScattering(AbstractScatteringMechanism):

    name = "IMP"
    required_properties = ("acceptor_charge", "donor_charge",
                           "static_dielectric")

    def __init__(self,
                 acceptor_charge: int,
                 donor_charge: int,
                 static_dielectric: float,
                 electronic_structure: ElectronicStructure,
                 doping: np.ndarray,
                 temperatures: np.ndarray):
        self.acceptor_charge = acceptor_charge
        self.donor_charge = donor_charge
        self.static_dielectric = static_dielectric
        self.beta = 1  # TODO: Calculate beta for doping and temperatures
        self.impurity_concentration = 1  # TODO: Calculate impurity conc at n, T

        prefactor = ((0.001 / e) ** 2 * e ** 4 /
                     (4.0 * np.pi ** 2 * self.static_dielectric ** 2 *
                      epsilon_0 ** 2 * hbar))
        prefactor = self.impurity_concentration * prefactor
        super().__init__(False, prefactor, doping, temperatures)

    def factor(self, k_diff_sq):
        # tile k_diff_sq to make it commensurate with the dimensions of beta
        return 1 / (np.tile(k_diff_sq, (
            len(self._doping), len(self._temperatures))) + self.beta ** 2) ** 2


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
    def prefactor(self):
        return 1

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
                   doping: np.ndarray,
                   temperatures: np.ndarray
                   ) -> List[AbstractScatteringMechanism]:
    # dynamically determine the available scattering mechanism subclasses
    scattering_mechanisms = {
        obj.name: obj for _, obj in inspect.getmembers(sys.modules[__name__])
        if inspect.isclass(obj) and
        obj is not AbstractScatteringMechanism and
        issubclass(obj, AbstractScatteringMechanism)}

    if scatttering_type == "auto":
        logger.info("Examining materials properties to determine possible "
                    "scattering_type mechanisms")

        scatttering_type = [
            name for name, mechanism in scattering_mechanisms.items()
            if all([materials_properties.get(x, False) for x in
                    mechanism.required_properties])]

        if not scatttering_type:
            raise ValueError("No scattering mechanisms possible with set of "
                             "materials properties provided")

    else:
        for name in scatttering_type:
            missing_properties = [
                p for p in scattering_mechanisms[name].required_properties
                if not materials_properties.get(p, False)]

            if missing_properties:
                raise ValueError(
                    "{} scattering mechanism specified but the following "
                    "materials properties are missing: {}".format(
                        name, ", ".join(missing_properties)))

    logger.info("The following scattering mechanisms will be "
                "calculated: {}".format(", ".join(scatttering_type)))

    scatterers = []
    for name in scatttering_type:
        mechanism = scattering_mechanisms[name]
        scatterers.append(mechanism(
            *[materials_properties[p] for p in mechanism.required_properties] +
            [doping, temperatures]))
    return scatterers


def scattering_worker(scatterers, ball_tree, energy_tol, energy_diff,
                      senergies, skpoints, skpoint_norms, sa_factor, sc_factor,
                      nkpoints, reciprocal_lattice_matrix, iqueue,
                      oqueue):
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
            raise ValueError("symmetry not yet supported")
            # band_rates = get_ir_band_rates(
            #     b_idx, ediff, energy_tol, s, k_idx,  k_p_idx, kpoints,
            #     kpoint_norms, a_factor, c_factor, ir_kpoints_idx,
            #     grouped_ir_to_full)
        else:
            # no symmetry, use the full BZ mesh
            band_rates = get_band_rates(
                scatterers, ediff, energy_tol, s, k_idx,  k_p_idx,
                kpoints, kpoint_norms, a_factor, c_factor,
                reciprocal_lattice_matrix)

        oqueue.put((s, band_rates))


def get_ir_band_rates(iband, ediff, energy_tol, s, k_idx, k_p_idx, kpoints,
                      kpoint_norms,
                      a_factor, c_factor, ir_kpoints_idx, grouped_ir_to_full):
    from numpy.core.umath_tests import inner1d

    # k_idx and k_p_idx are currently their reduced form. E.g., span 0 to
    # n_ir_kpoints-1; find actual columns of k_p_idx in the full Brillouin zone
    # by lookup
    full_cols_grouped = grouped_ir_to_full[k_p_idx]

    # get the reduced k_idx including duplicate k_idx for the full k_prime
    repeated_rows = np.repeat(k_idx, [len(g) for g in full_cols_grouped])
    expand_rows = np.repeat(np.arange(
        len(k_idx)), [len(g) for g in full_cols_grouped])

    # flatten the list of mapped columns
    full_cols = np.concatenate(full_cols_grouped)

    # get the indices of the k_idx in the full Brillouin zone
    full_rows = ir_kpoints_idx[repeated_rows]

    mask = full_rows != full_cols
    full_rows = full_rows[mask]
    full_cols = full_cols[mask]
    expand_rows = expand_rows[mask]

    k_dot = inner1d(kpoints[full_rows],
                    kpoints[full_cols])
    k_angles = k_dot / (kpoint_norms[full_rows] * kpoint_norms[full_cols])
    k_angles[np.isnan(k_angles)] = 1.

    a_vals = a_factor[iband, k_idx] * a_factor[iband, k_p_idx]
    c_vals = c_factor[iband, k_idx] * c_factor[iband, k_p_idx]

    overlap = (a_vals[expand_rows] + c_vals[expand_rows]
               * k_angles) ** 2
    weights = w0gauss(ediff / energy_tol)

    k_idx -= s.start
    repeated_rows = repeated_rows[mask] - s.start
    r = overlap * prefactor * weights[expand_rows]

    unique_rows = np.unique(k_idx)
    band_rates = np.zeros(s.stop - s.start)
    band_rates[unique_rows] = np.bincount(repeated_rows,
                                          weights=r)[unique_rows]

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

    k_diff_sq = np.dot(pbc_diff(kpoints[k_idx], kpoints[k_p_idx]),
                       reciprocal_lattice_matrix) ** 2

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

    # sleep(1000)

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
