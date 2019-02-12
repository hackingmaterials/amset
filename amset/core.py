# coding: utf-8
from __future__ import absolute_import

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import cProfile
import json
import os
import time
import warnings
from collections import OrderedDict
from copy import deepcopy
from itertools import count
from math import log, pi
from multiprocessing import cpu_count
from os.path import join as path_join
from pprint import pformat
from pstats import Stats
from sys import stdout

import numpy as np
from monty.json import MontyEncoder, MSONable, MontyDecoder
from monty.serialization import dumpfn, loadfn
from scipy.interpolate import griddata

from amset.logging import LoggableMixin
from amset.scattering.elastic import (
    IonizedImpurityScattering, AcousticDeformationScattering,
    PiezoelectricScattering, DislocationScattering)
from amset.interpolate import get_interpolater
from amset.utils.band_structure import get_bindex_bspin, \
    remove_duplicate_kpoints, \
    get_closest_k, generate_adaptive_kmesh, get_band_orbital_contributions
from amset.utils.constants import hbar, m_e, A_to_m, m_to_cm, A_to_nm, e, k_B, \
    epsilon_0, default_small_E, dTdz, sq3
from amset.utils.general import norm, cos_angle, remove_from_grid, get_angle, \
    sort_angles, AmsetError, get_tp
from amset.utils.k_integration import generate_k_mesh_axes, create_grid, \
    array_to_kgrid, normalize_array
from amset.plotting import get_amset_plots
from amset.utils.transport import f0, df0de, fermi_integral, calculate_sio, \
    free_e_dos
from amset.valley import Valley
from pymatgen.io.vasp import Vasprun, Spin, Kpoints
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

try:
    from BoltzTraP2 import sphere, fite
    from pymatgen.electronic_structure.boltztrap2 import BandstructureLoader
except ImportError:
    warnings.warn(
        'BoltzTraP2 not imported; "boltztrap2" interpolation not available.')

__author__ = "Alireza Faghaninia, Alex Ganose, Jason Frost, Anubhav Jain"
__copyright__ = "Copyright 2017, HackingMaterials"
__version__ = "0.1.0"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"
__status__ = "Development"

_doping_names = {"n": "conduction band(s)", "p": "valence band(s)"}
_inst_args = ['kgrid0', 'egrid0', 'kgrid_tp', 'cbm_vbm', 'mobility',
              'seebeck', 'elastic_scats', 'inelastic_scats', 'Efrequency0']


class Amset(MSONable, LoggableMixin):
    """ Runs Amset on a pymatgen from a VASP run (i.e. vasprun.xml). Amset is
    an ab initio model for calculating the mobility and Seebeck coefficient
    using Bolƒtzmann transport equation (BTE). The band structure in the
    Brilluin zone (BZ) is extracted from vasprun.xml to calculate the group
    velocity and transport properties in presence of various scattering
    mechanisms.

    Currently the following scattering mechanisms with their corresponding
    three-letter abbreviations implemented are: ionized impurity scattering (
    IMP), acoustic phonon deformation potential (ACD), piezoelectric (PIE),
    and charged dislocation scattering (DIS). Also, longitudinal polar
    optical phonon (POP) in implemented as an inelastic scattering mechanism
    that can alter the electronic distribution (the reason BTE has to be
    solved explicitly; for more information, see references [R, A]).

    You can control the level of theory via various inputs. For example,
    by assuming that the band structure is isotropic at the surrounding point
    of each k-point (i.e. bs_is_isotropic == True), one can significantly
    reduce the computational effort otherwise needed for accurate numerical
    integration of the scatterings.

    A small comment on the structure of this code: the calculations are done
    and stored in two main dictionary type variable called kgrid and egrid.
    kgrid contains all calculations that are done in k-space meaning that for
    each k-point and each band that is included there is a
    number/vector/property stored. On the other hand, the egrid is everything
    in energy scale hence we have number/vector/property stored at each
    energy point.

    References:
        [R]: D. L. Rode, Low-Field Electron Transport, Elsevier, 1975, vol. 10.,
            DOI: 10.1016/S0080-8784(08)60331-2
        [A]: A. Faghaninia, C. S. Lo and J. W. Ager, Phys. Rev. B, "Ab initio
            electronic transport model with explicit solution to the linearized
            Boltzmann transport equation" 2015, 91(23), 5100.,
            DOI: 10.1103/PhysRevB.91.235123
        [Q]: B. K. Ridley, Quantum Processes in Semiconductors, oxford
            university press, Oxford, 5th edn., 2013.
            DOI: 10.1093/acprof:oso/9780199677214.001.0001

    Args:
        calc_dir (str): Where Amset will be running.
        material_params (dict, required): material parameters; current options:

            "epsilon_s" (float>0, required): static dielectric constant
            "epsilon_inf" (float>0): high-frequency dielectric constant
            "C_el" (float>0): average elastic constant in GPa only used by ACD
            "P_PIE" (float>0): piezoelectric coefficient only used by PIE
            "E_D" (float>0 or {"n": float, "p": float}): CBM/VBM acoustic phonon
                deformation potential constant in eV; only used by ACD
            "N_dis" (float>0): charged linear discloation concentration only
                used by DIS. Units: 1/cm2 (# of dislocations in unit thickness)
            "user_bandgap" (float): the target band gap set artificially.
            "scissor" (float): amount of artificial change to the electronic
                band gap in eV; ignored if user_bandgap is set.
            "donor_charge" (int): the charge of the shallow donors (e.g. 2)
                defaults to 1
            "acceptor_charge" (int): the charge of the shallow acceptors
                defaults to 1
            "dislocations_charge" (int): absolute value of the charge of the
                linear dislocations; defaults to 1
            "important_points" (dict): the important band extrema dominating the
                transport; defaults to None to automatically find those points.
                examples: None or {'n': [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
                                   'p': [[0.0, 0.0, 0.0]]}
            "W_POP" (float): Longitudinal polar optical phonon frequency in THz
                at the k-point where the CBM/VBM is located

            example of material_params: {"epsilon_s": 10.3, "user_bandgap": 1.}
        interpolation (str): The band structure interpolation method.
              Options are "boltztrap1", "boltztrap2" (default), and "parabolic".
        parabolic_bands (None or list): None is recommended; otherwise
              set to simulate a band structure with one or multiple parabolic
              bands; For example, [[[[0.0, 0.0, 0.0], [0.0, 0.09]]]] denotes
              a single parabolic band, with a single extremum at
              Gamma ([0, 0, 0]) that is 0.0 eV above/below the CBM/VBM and
              has an effective mass of 0.09. Coordinates are fractional. For
              more information see the docs for get_parabolic_energy function.
        model_params (dict): parameters related to the model and the
            formulation:

            - "bs_is_isotropic" (bool): whether to use isotropic band
              formulation note that True is recommended as it is much faster
              than the anisotropic formulation while still captures some
              anisotropy. However, some details of anisotropic band may be lost
              in which case set to False.
            - "elastic_scats" ([str]): list of elastic scattering mechanisms to
              be include; for example: ["ACD", "IMP", "PIE"]
            - "inelastic_scats" ([str]): list of inelastic scattering mechanisms
               to be included; for example: ["POP"]

        performance_params (dict): parameters related to convergence, speed,
            etc; the options are:

            - "dE_min" (float): minimum energy difference differentiated in the
              energy grid (egrid); essentially the resolution of the egrid
            - "Ecut" (float or {"n": float, "p": float}): energy cutoff from the
              CBM/VBM beyond which the band structure is ignored.
            - "dos_bwidth" (float): the bandwidth (in eV) used for calculating
              the density of states (DOS)
            - "dos_kdensity" (int > 100): the uniform k-point density for DOS
            - "BTE_iters" (int>3): the number of iterations in solving the
              linearized Boltzmann Transport Equation (BTE)
            - "max_nbands" (None, int>=1): the maximum number of bands included;
              set None for autmatic determination bands on the Ecut.
            - "max_normk0" (float): the cutoff in reciprocal space from a given
              extremum in units of 1/nm
            - "max_nvalleys (None or int>=1): the maximum number of valleys
              included in each band. Set to None for automatic setting.
            - "n_jobs" (int>=1): the number of jobs in parallelization.
              Currently, it is only relevant to interpolation method of
              "boltztrap1"

        dopings ([float]): list of input carrier concentrations; c<0 for
            electrons and c>0 for holes
        temperatures ([float]): input temperatures (T) in Kelvin.
        integration (str): 'e' or 'k'. Currently only e or integration of
            properties in the energy-scale is supported
        logger (Logger, bool): A custom logger object to use for logging.
            Alternatively, if set to True, the default automatminer logger will
            be used. If set to False, then no logging will occur.
        log_level (int): e.g. logging.DEBUG; set logging.ERROR to turn off
            the logging
        timeout_hours (float): timeout_hours in hours. If Amset takes longer
            than this, the calculations will stop. However, if transport
            properties are already calculated, the postprocessing (e.g. write to
            file) might violate this timeout_hours.

    Returns:
        (None): results are accessible through various methods such as the
        logged result (on the screen or the logfile), or through the following
        methods::

            Amset.to_csv(): transport properties at different c&T
            Amset.as_dict(): most commonly used attributes as python dictionary
            Amset.grids_to_json(): details of relaxation time, scattering rates,
                group velocities, etc in both kgrid and egrid
            Amset.to_file(): writes everything to a json file to be used again
                later (via Amset.from_file() e.g. for plotting). to_file is
                more complete than grids_to_json and is readable by from_file
            Amset.plot(): convenience function to plot the mobilities, energy,
                perturbation function, scattering rates, etc.
     """

    def __init__(self, band_structure, num_electrons, material_params,
                 interpolation: str = 'boltztrap2', coeff_file=None,
                 parabolic_bands=None, calc_dir=None,
                 model_params=None, performance_params=None,
                 dopings=None, temperatures=None, integration='e', logger=True,
                 log_level=None, timeout_hours=48):

        self._logger = self.get_logger(logger, level=log_level)
        self._log_level = log_level

        if integration == 'k':
            self.logger.warning(
                "k-integration is not fully implemented! The results may be"
                "unreliable, especially for non-cubic systems. 'e'-integration "
                "is recommended.")
        elif integration != 'e':
            self.log_raise(ValueError, 'Only "e" integration supported.')

        if band_structure.is_metal():
            self.log_raise(ValueError, 'Band structure is metallic. This is not'
                                       'supported by AMSET.')

        self.band_structure = band_structure
        self.num_electrons = num_electrons

        # TODO use these rather than set material params etc.
        # also remove deepcopy
        self._material_params = deepcopy(material_params)
        self._model_params = deepcopy(model_params)
        self._performance_params = deepcopy(performance_params)
        self._interpolation = interpolation
        self._coeff_file = coeff_file
        self._parabolic_bands = parabolic_bands

        self.structure = band_structure.structure
        self.calc_dir = calc_dir if calc_dir else '.'
        self.dopings = list(map(int, dopings)) if dopings else [-1e20, 1e20]
        self.all_types = list(set([get_tp(c) for c in self.dopings]))
        self.temperatures = list(map(int, temperatures)) if temperatures else \
            [300, 600]
        self.integration = integration
        self.timeout_hours = timeout_hours * 3600.0
        self.kpoints = [k.frac_coords for k in band_structure.kpoints]
        self.cartesian_kpoints = np.array([self.get_cartesian_coords(k)
                                           for k in self.kpoints]) / A_to_nm
        self.rotations = SpacegroupAnalyzer(
            self.structure).get_symmetry_dataset()['rotations']

        self.set_model_params(model_params)
        self.set_material_params(material_params)
        self.set_performance_params(performance_params)
        self.interpolater = get_interpolater(
            band_structure, num_electrons, coeff_file=coeff_file,
            parabolic_band_parameters=parabolic_bands,
            interpolation_type=interpolation)

        cbm = band_structure.get_cbm()
        vbm = band_structure.get_vbm()
        eigs = np.array([self.band_structure.bands[s] for s in Spin
                         if s in self.band_structure.bands])
        self.dos_emin = np.min(eigs)
        self.dos_emax = np.max(eigs)

        self.dft_gap = cbm["energy"] - vbm["energy"]
        self.offset_from_vrun = {'n': 0.0, 'p': 0.0}

        if self.user_bandgap:
            if self.scissor != 0.0:
                self.logger.warning("user_bandgap and scissor are both set, "
                                    "overriding scissor.")
            self.scissor = self.user_bandgap - self.dft_gap
            self.logger.info('scissor is set to {}'.format(self.scissor))

        cbm_vbm = {"n": {"kpoint": self.kpoints[cbm["kpoint_index"][0]],
                         "energy": cbm["energy"],
                         "bidx": get_bindex_bspin(vbm, is_cbm=False)[0] + 1,
                         "included": 0,
                         "eff_mass_xx": [0.0, 0.0, 0.0]},
                   "p": {"kpoint": self.kpoints[vbm["kpoint_index"][0]],
                         "energy": vbm["energy"],
                         "bidx": get_bindex_bspin(vbm, is_cbm=False)[0],
                         "included": 0,
                         "eff_mass_xx": [0.0, 0.0, 0.0]}}

        if self.parabolic_bands is None:
            # normalise the band energies to the band edge
            max_cb = np.min(eigs[0], axis=1) - cbm['energy']
            min_vb = np.max(eigs[0], axis=1) - vbm['energy']

            # count the number of bands within cut-off from the band edges
            cbm_vbm['n']['included'] = len(
                max_cb[(max_cb >= 0) & (max_cb <= self.Ecut["n"])])
            cbm_vbm['p']['included'] = len(
                max_cb[(min_vb <= 0) & (min_vb >= -self.Ecut["p"])])
        else:
            cbm_vbm["n"]["included"] = len(self.parabolic_bands)
            cbm_vbm["p"]["included"] = len(self.parabolic_bands)

        self.init_nbands = {'n': cbm_vbm['n']['included'],
                            'p': cbm_vbm['p']['included']}

        self.cbm_vbm = cbm_vbm
        self.cbm_vbm0 = deepcopy(cbm_vbm)
        self.valleys = {tp: {'band {}'.format(i): OrderedDict()
                             for i in range(self.cbm_vbm0[tp]['included'])}
                        for tp in ['p', 'n']}
        self.num_bands = {tp: self.cbm_vbm[tp]["included"] for tp in ['n', 'p']}
        self.kgrid_tp = None
        self.seebeck = {'n': None, 'p': None}
        self.start_time = None

    @staticmethod
    def from_vasprun(vasprun, material_params, **kwargs):
        if isinstance(vasprun, str):
            vasprun = Vasprun(vasprun, parse_projected_eigen=True)

        band_structure = vasprun.get_band_structure()
        num_electrons = vasprun.parameters['NELECT']
        return Amset(band_structure, num_electrons, material_params, **kwargs)

    def run_profiled(self, kgrid_tp="coarse", nfuncs=15):
        """
        Very similar to the run method except that it is time-profiled: it
            shows the total and per-call time elapsed in running each function.

        Args:
            see args (coeff_file, kgrid_tp) for run() method
            nfuncs (int): only print the nfuncs most time-consuming functions
        """
        profiler = cProfile.Profile()
        profiler.runcall(lambda: self.run(kgrid_tp=kgrid_tp))
        stats = Stats(profiler, stream=stdout)
        stats.strip_dirs()
        stats.sort_stats('cumulative')
        stats.print_stats(nfuncs)

    def _check_timeout_hours(self):
        if time.time() - self.start_time > self.timeout_hours:
            self.log_raise(RuntimeError,
                           'The calculations exceeded the timeout_hours of '
                           '{:.4f} hours'.format(self.timeout_hours / 3600.0))

    def run(self, kgrid_tp="coarse"):
        """
        Function to run Amset and log the main outputs, populate the two main
            grid variables: k-points grid (kgrid) and energy grid (egrid) and
            class attributes such as Amset.mobility and Amset.seebeck.

        Args:
            kgrid_tp (str): define the density of k-point mesh.
                options: 'very coarse', 'coarse', 'fine'

        Returns (None):
            many instance variables get updated with calculated properties.
        """
        self.start_time = time.time()

        self.logger.info('Running Amset on {}'.format(
            self.structure.composition.reduced_formula))
        self.logger.info('integration: {}'.format(self.integration))
        self.logger.info("number of cpu used (n_jobs): {}".format(self.n_jobs))
        self.logger.debug('direct lattice matrix:\n{}'.format(
            self.structure.lattice.matrix))
        self.logger.info("cell volume = {} A**3".format(self.structure.volume))
        self.logger.info("original cbm_vbm:\n {}".format(self.cbm_vbm))
        self.kgrid_tp = kgrid_tp
        self.logger.info(
            'Running on "{}" mesh for each valley'.format(kgrid_tp))
        self.logger.info(
            'band interpolation="{}" method'.format(self._interpolation))
        self.logger.info('max_nbands={}'.format(self.max_nbands))
        self.logger.info('max_nvalleys={}'.format(self.max_nvalleys))
        self.logger.info('max_normk={}'.format(self.max_normk))

        self._initialize_transport_vars()
        self._update_cbm_vbm()

        # with ibands_tuple, for each couple of conduction/valence bands we
        # only use 1 band together (i.e. always ib==0)
        for tp in ['p', 'n']:
            self.cbm_vbm[tp]['included'] = 1

        self._check_timeout_hours()
        self.dos = self._calculate_dos()

        if self.integration == 'k':
            kpts = self.generate_kmesh(
                important_points={'n': [[0.0, 0.0, 0.0]],
                                  'p': [[0.0, 0.0, 0.0]]},
                kgrid_tp=kgrid_tp)

            # the purpose of the following line is just to generate
            # self.energy_array that find_fermi_k function uses
            self.get_energy_array(kpts, return_energies=True,
                                  num_bands=self.init_nbands)
            self.fermi_level = self.find_fermi_k(num_bands=self.init_nbands)

        elif self.integration == 'e':
            self.fermi_level = {c: {T: self.find_fermi(c, T)
                                    for T in self.temperatures}
                                for c in self.dopings}

        self.logger.info('fermi level = {}'.format(self.fermi_level))
        self.logger.info('initial number of bands: {}'.format(self.init_nbands))
        self.logger.debug('calculated dopings: {}'.format(self.calc_doping))

        vibands = list(range(self.init_nbands['p']))
        cibands = list(range(self.init_nbands['n']))

        if len(vibands) > len(cibands):
            ibands_tuple = list(zip(vibands, cibands + [cibands[0]] * (
                    len(vibands) - len(cibands))))
            for ivt in range(len(cibands), len(vibands)):
                self.count_mobility[ivt]['n'] = False
        else:
            ibands_tuple = list(
                zip(vibands + [vibands[0]] * (len(cibands) - len(vibands)),
                    cibands))
            for ivt in range(len(vibands), len(cibands)):
                self.count_mobility[ivt]['p'] = False
        self.ibands_tuple = ibands_tuple
        self.count_mobility0 = deepcopy(self.count_mobility)
        # each time num_bands will be {'n': 1, 'p': 1} but w/ different band idx
        if self.max_nbands:
            ibands_tuple = ibands_tuple[
                           :min(len(ibands_tuple), self.max_nbands)]

        self.logger.debug('here ibands_tuple: [(val. band #, cond. band #)]')
        self.logger.debug(ibands_tuple)
        self.logger.debug('here whether to count bands')
        self.logger.debug(self.count_mobility)

        self.denominator = {
            c: {T: {'p': 0.0, 'n': 0.0} for T in self.temperatures} for c in
            self.dopings}
        self.seeb_denom = {
            c: {T: {'p': 0.0, 'n': 0.0} for T in self.temperatures} for c in
            self.dopings}

        for self.ibrun, (self.nbelow_vbm, self.nabove_cbm) in enumerate(
                ibands_tuple):
            self._check_timeout_hours()
            self.logger.info(
                'going over conduction and valence # {}'.format(self.ibrun))
            self.all_important_pts = self.find_all_important_points(
                self.nbelow_vbm, nabove_cbm=self.nabove_cbm)

            if self.max_nvalleys['n'] is None and self.max_nvalleys[
                'p'] is None:
                max_nvalleys = max(len(self.important_pts['n']),
                                   len(self.important_pts['p']))
            else:
                max_nvalleys = max(min(len(self.important_pts['n']),
                                       self.max_nvalleys["n"] or 1000),
                                   min(len(self.important_pts['p']),
                                       self.max_nvalleys["p"] or 1000))

            for ivalley in range(max_nvalleys):
                self._check_timeout_hours()
                self.count_mobility[self.ibrun] = self.count_mobility0[
                    self.ibrun]
                important_points = {'n': None, 'p': None}
                for tp in ['p', 'n']:
                    try:
                        important_points[tp] = [self.important_pts[tp][ivalley]]
                    except:
                        important_points[tp] = [self.important_pts[tp][0]]
                        self.count_mobility[self.ibrun][tp] = False

                if self.max_normk0 is None:
                    for tp in ['n', 'p']:
                        min_dist = 100.0  # in 1/nm
                        # This for loop not only checks the distance between
                        # different valleys but among symmetrically
                        # equivalent k-points of the same valley so the
                        # cutoff would make sense
                        for k in self.band_structure.get_sym_eq_kpoints(
                                important_points[tp][0], cartesian=False):
                            kdiff = get_closest_k(
                                k, self.all_important_pts[tp], return_diff=True,
                                exclude_self=True)
                            new_dist = 1 / A_to_nm * norm(
                                self.get_cartesian_coords(kdiff))
                            # to avoid self-counting, 0.1 criterion added:
                            if new_dist < min_dist and new_dist > 0.01:
                                min_dist = new_dist
                        self.max_normk[tp] = min_dist / 2.0
                        if self.max_normk[tp] < 0.05:
                            raise AmsetError(self.logger, "max_normk['{}']={} "
                                                          "is way too low! Please check your input and if "
                                                          "necessary manually enter the band extrema "
                                                          "coordinates".format(
                                tp, self.max_normk[tp]))
                for tp in ['p', 'n']:
                    if self.max_nvalleys[tp] == 1 and self.max_normk0 is None:
                        self.max_normk[tp] = 5.0
                        self.logger.warn(
                            'max_normk["{}"] -> {} avoiding unlrealistic'
                            ' scattering'.format(tp, self.max_normk[tp]))
                self.logger.info(
                    'at valence band #{} and conduction band #{}'.format(
                        self.nbelow_vbm, self.nabove_cbm))
                self.logger.info(
                    'Current valleys:\n{}'.format(important_points))
                self.logger.info('Whether to count valleys: {}'.format(
                    self.count_mobility[self.ibrun]))
                self.logger.info('max_normk:\n{}'.format(self.max_normk))
                self.logger.info('important points for this band:\n{}'.format(
                    important_points))

                if not self.count_mobility[self.ibrun]['n'] and not \
                        self.count_mobility[self.ibrun]['p']:
                    self.logger.info(
                        'skipping this valley as it is unimportant for both n and p type...')
                    continue
                kpts = self.generate_kmesh(important_points=important_points,
                                           kgrid_tp=kgrid_tp)
                kpts, energies = self.get_energy_array(kpts,
                                                       return_energies=True,
                                                       num_bands={'p': 1,
                                                                  'n': 1})
                self._check_timeout_hours()

                if min(energies['n']) - self.cbm_vbm['n']['energy'] > self.Ecut[
                    'n']:
                    self.logger.info('not counting conduction band {} valley\
                     {} due to off enery...'.format(self.ibrun,
                                                    important_points['n'][0]))
                    self.count_mobility[self.ibrun]['n'] = False
                if self.cbm_vbm['p']['energy'] - max(energies['p']) > self.Ecut[
                    'p']:
                    self.logger.info('not counting valence band {} valley {} \
                    due to off enery...'.format(self.ibrun,
                                                important_points['p'][0]))
                    self.count_mobility[self.ibrun]['p'] = False

                if not self.count_mobility[self.ibrun]['n'] and \
                        not self.count_mobility[self.ibrun]['p']:
                    self.logger.info("skipping this iband as it's unimportant"
                                     " or energies are off:\n{}".format(
                        important_points))
                    continue

                corrupt_tps = self.init_kgrid(kpts, important_points)
                for tp in corrupt_tps:
                    self.count_mobility[self.ibrun][tp] = False

                if not self.count_mobility[self.ibrun]['n'] and not \
                        self.count_mobility[self.ibrun]['p']:
                    self.logger.info(
                        'skipping this valley as it is unimportant or its energies are way off...')
                    continue
                corrupt_tps = self.init_egrid()
                for tp in corrupt_tps:
                    self.count_mobility[self.ibrun][tp] = False
                if not self.count_mobility[self.ibrun]['n'] and not \
                        self.count_mobility[self.ibrun]['p']:
                    self.logger.info(
                        'skipping this valley as it is unimportant or its energies are way off...')
                    continue

                self.bandgap = min(self.egrid["n"]["all_en_flat"]) \
                               - max(self.egrid["p"]["all_en_flat"])
                if abs(self.bandgap - (self.cbm_vbm["n"]["energy"] \
                                       - self.cbm_vbm["p"]["energy"] \
                                       + self.scissor)) > k_B * 300:
                    warnings.warn("The band gaps do NOT match! \
                    The selected k-mesh is probably too coarse.")
                self.map_to_egrid(prop_name="g",
                                  c_and_T_idx=True,
                                  prop_type="vector")
                self.map_to_egrid(prop_name="velocity",
                                  c_and_T_idx=False,
                                  prop_type="vector")

                # find the indexes of equal energy or those with ±hbar*W_POP for scattering via phonon emission and absorption
                if not self.bs_is_isotropic or "POP" in self.inelastic_scats:
                    self.generate_angles_and_indexes_for_integration()

                # calculate all elastic scattering rates in kgrid and then map it to egrid:
                for sname in self.elastic_scats:
                    self.s_elastic(sname=sname)
                    self.map_to_egrid(prop_name=sname)

                self.map_to_egrid(prop_name="relaxation time")

                for c in self.dopings:
                    for T in self.temperatures:
                        seeb_integ = \
                            self.egrid["Seebeck_integral_numerator"][c][T][tp] / \
                            self.egrid["Seebeck_integral_denominator"][c][T][tp]
                        fermi = self.fermi_level[c][T]
                        for tp in ["n", "p"]:
                            fermi_norm = fermi - self.cbm_vbm[tp]["energy"]
                            for ib in range(len(self.kgrid[tp]["energy"])):
                                for ik in range(
                                        len(self.kgrid[tp]["kpoints"][ib])):
                                    E = self.kgrid[tp]["energy"][ib][ik]
                                    v = self.kgrid[tp]["velocity"][ib][ik]
                                    self.kgrid[tp]["f0"][c][T][ib][ik] = f0(E,
                                                                            fermi,
                                                                            T) * 1.0
                                    self.kgrid[tp]["df0dk"][c][T][ib][
                                        ik] = hbar * df0de(E, fermi,
                                                           T) * v  # in cm
                                    self.kgrid[tp]["electric force"][c][T][ib][
                                        ik] = \
                                        -1 * self.kgrid[tp]["df0dk"][c][T][ib][
                                            ik] * \
                                        default_small_E / hbar  # in 1/s
                                    E_norm = E - self.cbm_vbm[tp]["energy"]
                                    self.kgrid[tp]["thermal force"][c][T][ib][
                                        ik] = \
                                        -v * f0(E_norm, fermi_norm, T) * (
                                                1 - f0(
                                            E_norm, fermi_norm,
                                            T)) * dTdz / T * (
                                                E_norm / (k_B * T) - seeb_integ
                                        )
                        # - norm(v)/sq3 * f0(E_norm, fermi_norm, T) * (1 - f0(
                        if self.integration == 'k':
                            dop_tp = get_tp(c)
                            f0_all = 1 / (np.exp((self.energy_array[dop_tp] -
                                                  self.fermi_level[c][T]) / (
                                                         k_B * T)) + 1)
                            if c < 0:
                                electrons = self.integrate_over_states(f0_all,
                                                                       dop_tp)
                                self.logger.info(
                                    'k-integral of f0 above band gap at c={:.2e}, T={}: {}'.format(
                                        c, T, electrons))
                            if c > 0:
                                holes = self.integrate_over_states(1 - f0_all,
                                                                   dop_tp)
                                self.logger.info(
                                    'k-integral of 1-f0 below band gap at c={:.2e}, T={}: {}'.format(
                                        c, T, holes))

                self.map_to_egrid(prop_name="f0", c_and_T_idx=True,
                                  prop_type="vector")
                self.map_to_egrid(prop_name="df0dk", c_and_T_idx=True,
                                  prop_type="vector")

                # solve BTE in presence of electric and thermal driving force to get perturbation to Fermi-Dirac: g
                self.solve_BTE_iteratively()
                if self.integration == 'k':
                    test_k_anisotropic = False  # for k-integration
                    valley_transport = self.calculate_transport_properties_with_k(
                        test_k_anisotropic, important_points)
                elif self.integration == 'e':
                    if len(self.Efrequency['n']) <= 1 or len(
                            self.Efrequency['p']) <= 1:
                        raise AmsetError(self.logger,
                                         'The egrid is too small for n- or p-'
                                         'type for e-integration of transport')
                    valley_transport = self.calculate_transport_properties_with_E()
                else:
                    raise AmsetError(self.logger, 'Unsupported integration '
                                                  'method: {}'.format(
                        self.integration))
                self.logger.info('finished calculating the mobility of the '
                                 'valley {} and band (p, n) {}'.format(
                    important_points, self.ibands_tuple[self.ibrun]))
                self.logger.info('count_mobility: {}'.format(
                    self.count_mobility[self.ibrun]))
                self.logger.info('transport properties of the current valley'
                                 '\n{}'.format(pformat(valley_transport)))

                if self.ibrun == 0 and ivalley == 0:  # 1-valley only since it's SPB
                    self.calculate_spb_transport()

                self.logger.info('Mobility Labels: {}'.format(self.mo_labels))
                for tp in ['p', 'n']:
                    valley_ndegen = self.band_structure.get_kpoint_degeneracy(
                        important_points[tp][0])
                    self.logger.debug(
                        'valley_ndegen = {} for {}'.format(valley_ndegen,
                                                           important_points[tp][
                                                               0]))
                    for c in self.dopings:
                        for T in self.temperatures:
                            self.kgrid[tp]["relaxation time"][c][T][ib] = \
                                1 / (self.kgrid[tp]["_all_elastic"][c][T][ib] \
                                     + self.kgrid[tp]["S_o"][c][T][ib] \
                                     + self.kgrid[tp]["S_i"][c][T][ib])

                            if self.count_mobility[self.ibrun][tp]:
                                if self.integration == 'k':
                                    f0_all = 1. / (np.exp((self.energy_array[
                                                               'n'] -
                                                           self.fermi_level[c][
                                                               T]) / (
                                                                  k_B * T)) + 1.)
                                    f0p_all = 1. / (np.exp((self.energy_array[
                                                                'p'] -
                                                            self.fermi_level[c][
                                                                T]) / (
                                                                   k_B * T)) + 1.)
                                    finteg = f0_all if tp == 'n' else 1 - f0p_all
                                    self.denominator[c][T][
                                        tp] += 3 * default_small_E * self.integrate_over_states(
                                        finteg, tp) + 1e-10
                                    self.seeb_denom[c][T][
                                        tp] += self.integrate_over_states(
                                        finteg * (1 - finteg), tp)
                                elif self.integration == 'e':
                                    finteg = "f0" if tp == "n" else "1 - f0"
                                    self.denominator[c][T][
                                        tp] += 3 * default_small_E * self.integrate_over_E(
                                        props=[finteg], tp=tp, c=c, T=T,
                                        xDOS=False, xvel=False) * valley_ndegen
                                    self.seeb_denom[c][T][tp] += \
                                        self.egrid[
                                            "Seebeck_integral_denominator"][
                                            c][T][tp] * valley_ndegen
                                for mu in self.mo_labels + ["J_th"]:
                                    self.mobility[tp][mu][c][T] += \
                                        valley_transport[tp][mu][c][
                                            T] * valley_ndegen
                                self.mobility[tp]['seebeck'][c][T] += \
                                    valley_transport[tp]['seebeck'][c][
                                        T]  # seeb is multiplied by DOS so no need for degeneracy

                self.map_to_egrid(prop_name="relaxation time")

                if self.parabolic_bands0 is None:
                    for tp in ['p', 'n']:
                        if self.count_mobility[self.ibrun][tp]:
                            k = important_points[tp][0]
                            for i in range(3):
                                if abs(k[i]) < 1e-4:
                                    k[i] = 0.0
                                elif abs(abs(k[i]) - 0.5) < 1e-4:
                                    k[i] = round(k[i], 1)
                                elif round(k[i], 1) == round(k[i], 2):
                                    k[i] = round(k[i], 1)
                                elif k[i] != round(k[i], 2):
                                    k[i] = round(k[i], 2)
                            # TODO: here the numerator of band/valleys is initiated by w/o valley_ndegen which results
                            # in valleys.json mobility values being lower; write a function that goes over all values of
                            # valley transport and multiplies them by valley_ndegen = self.band_structure.get_kpoint_degeneracy(k)
                            self.valleys[tp]['band {}'.format(self.ibrun)][
                                '{};{};{}'.format(k[0], k[1], k[2])] = \
                                valley_transport[tp]

                kgrid_rm_list = ["f_th", "S_i_th", "S_o_th"]
                self.kgrid = remove_from_grid(self.kgrid, kgrid_rm_list)
                if ivalley == 0 and self.ibrun == 0:
                    # TODO: make it possible for the user to choose which valley(s) to plot
                    self.kgrid0 = deepcopy(self.kgrid)
                    self.egrid0 = deepcopy(self.egrid)
                    self.Efrequency0 = deepcopy(self.Efrequency)
        self.logger.debug('here denominator:\n{}'.format(self.denominator))

        for tp in ['p', 'n']:
            for c in self.dopings:
                for T in self.temperatures:
                    for mu in self.mo_labels + ["J_th"]:
                        self.mobility[tp][mu][c][T] /= self.denominator[c][T][
                            tp]
                        for band in list(self.valleys[tp].keys()):
                            for valley_k in list(self.valleys[tp][band].keys()):
                                self.valleys[tp][band][valley_k][mu][c][T] /= \
                                    self.denominator[c][T][tp]
                    self.mobility[tp]["seebeck"][c][T] /= self.seeb_denom[c][T][
                        tp]
                    for band in list(self.valleys[tp].keys()):
                        for valley_k in list(self.valleys[tp][band].keys()):
                            self.valleys[tp][band][valley_k]["seebeck"][c][T] /= \
                                self.seeb_denom[c][T][tp]

        # finalize Seebeck values:
        sigma = {
            tp: {c: {T: 0.0 for T in self.temperatures} for c in self.dopings}
            for
            tp in ['p', 'n']}
        for tp in ['p', 'n']:
            for c in self.dopings:
                for T in self.temperatures:
                    self.logger.debug(
                        '3 terms of {0}-type seebeck at c={1:.2e}, T={2}'.format(
                            tp, c, T))
                    self.logger.debug('seebeck integral term: {}'.format(
                        str(self.mobility[tp]['seebeck'][c][T] * (-1e6) * k_B)))
                    self.mobility[tp]['seebeck'][c][T] -= \
                        (self.fermi_level[c][T] - self.cbm_vbm[tp]["energy"]) \
                        / (k_B * T)
                    self.logger.debug(
                        'seebeck term Fermi level w.r.t. the CBM/VBM: {}'.format(
                            str((self.fermi_level[c][T] - self.cbm_vbm[tp][
                                "energy"]) / (k_B * T) * (-1e6) * k_B)))
                    self.mobility[tp]['seebeck'][c][T] *= (-1e6) * k_B
                    self.mobility[tp]["seebeck"][c][
                        T] -= 0  # TODO: J_th term is too large, see why (e.g. in SnS)
                    # self.mobility[tp]["seebeck"][c][T] += 1e6 * self.mobility[tp]["J_th"][c][T]\
                    #     /(self.mobility[tp]["overall"][c][T]*e*float(1+abs(self.calc_doping[c][T][tp])))/dTdz
                    self.logger.debug('seebeck term J_th: {}'.format(str(
                        1e6 * self.mobility[tp]["J_th"][c][T] / (
                                self.mobility[tp]["overall"][c][
                                    T] * e * float(
                            1 + abs(self.calc_doping[c][T][tp]))) / dTdz)))
                    for band in list(self.valleys[tp].keys()):
                        for valley_k in list(self.valleys[tp][band].keys()):
                            self.valleys[tp][band][valley_k]["seebeck"][c][
                                T] -= (self.fermi_level[c][T] -
                                       self.cbm_vbm[tp]["energy"]) / (k_B * T)
                            self.valleys[tp][band][valley_k]["seebeck"][c][
                                T] *= (-1e6) * k_B
            self.seebeck[tp] = self.mobility[tp].pop('seebeck')
            sigma[tp][c][T] = sum(
                self.mobility[tp]['overall'][c][T]) / 3. * e * abs(
                self.calc_doping[c][T][tp])

        self.logger.debug('here is the conductivity, sigma:\n{}'.format(sigma))
        # seebeck = deepcopy(self.seebeck)
        # for c in self.dopings:
        #     for T in self.temperatures:
        #         self.seebeck['n'][c][T] = (sigma['n'][c][T]*seebeck['n'][c][T] - sigma['p'][c][T]*seebeck['p'][c][T])/(sigma['n'][c][T]+sigma['p'][c][T])
        self.logger.info('run finished.')
        self.logger.info('\nfinal mobility:\n{}'.format(pformat(self.mobility)))
        self.logger.info('\nfinal Seebeck:\n{}'.format(pformat(self.seebeck)))

    def _initialize_transport_vars(self):
        """
        Variables related to transport such as cbm_vbm, mobility, etc. This
        internal method is supposed to be called after read_vrun.
        """
        if self.parabolic_bands0 is not None:
            self.cbm_vbm["n"]["kpoint"] = self.cbm_vbm["p"]["kpoint"] = \
                self.parabolic_bands[0][0][0]

        # initialize transport variables
        self.mo_labels = self.elastic_scats + self.inelastic_scats + ['overall',
                                                                      'average']
        self.spb_labels = ['SPB_ACD']
        self.transport_labels = self.mo_labels + self.spb_labels + ["seebeck",
                                                                    "J_th"]
        self.mobility = {
            tp: {el_mech: {c: {T: np.array([0., 0., 0.], dtype='float') \
                               for T in self.temperatures} \
                           for c in self.dopings} \
                 for el_mech in self.transport_labels} \
            for tp in ["n", "p"]}
        self.calc_doping = {c: {T: {'n': None, 'p': None} \
                                for T in self.temperatures} \
                            for c in self.dopings}
        self.ibrun = 0  # counter of the ibands_tuple (band-valley walker)
        self.count_mobility = [{'n': True, 'p': True} \
                               for _ in range(max(self.init_nbands['p'],
                                                  self.init_nbands['n']))]

    def calculate_spb_transport(self):
        """
        Using single parabolic band (SPB), calculates some elastic scattering-
        limited mobility values named with an "SPB_*" prefix. For example,
        mobility limited by acoustic phonon deformation potential (SPB_ACD)

        Returns: None; results are saved inside self.mobility
        """
        for tp in ['p', 'n']:
            for c in self.dopings:
                for T in self.temperatures:
                    fermi = self.fermi_level[c][T]

                    # TODO: note that now I only calculate one mobility, define a relative energy term for both n- and p-SPB later
                    energy = self.cbm_vbm[get_tp(c)]["energy"]

                    # ACD mobility based on single parabolic band extracted from Thermoelectric Nanomaterials,
                    # chapter 1, page 12: "Material Design Considerations Based on Thermoelectric Quality Factor"
                    self.mobility[tp]["SPB_ACD"][c][
                        T] = 2 ** 0.5 * pi * hbar ** 4 * e * self.C_el * 1e9 / (
                            3 * (abs(self.cbm_vbm[tp][
                                         "eff_mass_xx"] + self.gs) * m_e) ** 2.5 * (
                                    k_B * T) ** 1.5 * self.E_D[tp] ** 2) \
                             * fermi_integral(0, fermi, T, energy) \
                             / fermi_integral(0.5, fermi, T,
                                              energy) * e ** 0.5 * 1e4  # to cm2/V.s

    def generate_kmesh(self, important_points, kgrid_tp='coarse'):
        """
        List of kpoints surrounding important points. This adaptive mesh is
        finer closer to the "important" points or band extrema and they get
        coarser further away.

        Args:
            important_points ({"n": [3x1 array], "p": [3x1 array]}): list of
                important k-points for conduction ("n") and valence ("p") bands
            kgrid_tp (str): type of kgrid or how dense it is. options are:
                "very coarse", "coarse", "fine", "very fine", "super fine"
        Returns ({"n": [3x1 array], "p": [3x1 array]}): list of k-points for
            both n and p type.

        """
        self.kgrid_array = {}
        self.kgrid_array_cartesian = {}
        self.k_hat_array = {}
        self.k_hat_array_cartesian = {}
        self.dv_grid = {}
        kpts = {}
        if self.integration == 'e':
            kpts = generate_adaptive_kmesh(self.band_structure,
                                           important_points, kgrid_tp)
        for tp in ['n', 'p']:
            points_1d = generate_k_mesh_axes(important_points[tp], kgrid_tp,
                                             one_list=True)
            self.kgrid_array[tp] = create_grid(points_1d)
            if self.integration == 'k':
                kpts[tp] = array_to_kgrid(self.kgrid_array[tp])
            N = self.kgrid_array[tp].shape
            self.kgrid_array_cartesian[tp] = np.zeros((N[0], N[1], N[2], 3))
            for ii in range(N[0]):
                for jj in range(N[1]):
                    for kk in range(N[2]):
                        self.kgrid_array_cartesian[tp][ii, jj, kk,
                        :] = self.get_cartesian_coords(
                            self.kgrid_array[tp][ii, jj, kk])  # 1/A

            # generate a normalized numpy array of vectors pointing in the direction of k
            self.k_hat_array[tp] = normalize_array(self.kgrid_array[tp])
            self.k_hat_array_cartesian[tp] = normalize_array(
                self.kgrid_array_cartesian[tp])
            self.dv_grid[tp] = self.find_dv(self.kgrid_array[tp])
        return kpts

    def _update_cbm_vbm(self):
        """Updates the energy of the CBM/VBM using the interpolated energies."""
        for tp in ("p", "n"):
            energy, effective_mass = self.interpolater.get_energies(
                [self.cbm_vbm0[tp]['kpoint']], self.cbm_vbm0[tp]['bidx'],
                scissor=self.scissor, return_effective_mass=True)

            self.offset_from_vrun[tp] = energy[0] - self.cbm_vbm0[tp]["energy"]
            self.logger.debug(
                "offset from vasprun energy values for {}-type = {} eV".format(
                    tp, self.offset_from_vrun[tp]))
            self.cbm_vbm0[tp].update({
                'energy': energy[0],
                'eff_mass_xx': effective_mass[0].diagonal()})
            self.cbm_vbm[tp].update({
                'energy': energy[0],
                'eff_mass_xx': effective_mass[0].diagonal()})

        self.logger.debug("cbm_vbm after updating:\n {}".format(self.cbm_vbm))

        new_gap = self.cbm_vbm["n"]["energy"] - self.cbm_vbm["p"]["energy"]
        old_gap = self.dft_gap + self.scissor
        if abs(old_gap - new_gap) > 0.1:
            self.log_raise(RuntimeError,
                           'The interpolated band gap is very different to the'
                           'the input band gap! {}!={}'.format(
                               new_gap, old_gap))

    def get_energy_array(self, kpoints, return_energies=False, num_bands=None):
        """
        Multi-purpose function to populate energy_array instance variable as
        well as the dos and dos_* instance variables when the function is called
        for the first time. These are all calculated for a given k-point mesh
        (kpts) that may be different for n-type and p-type.

        Args:
            kpoints ({"n": [3x1 array], "p": [3x1 array]}): k-point meshes
            return_energies (bool): whether to return sorted energy values
            num_bands ({"n": int, "p": int}): number of bands for n- and p-type

        Returns:
            depending on return_energies only returns back the kpts with k-point
            coordinates sorted based on their energy value or that and also the
            sorted energy values.
        """
        num_bands = num_bands or self.num_bands

        energies = {"n": [0.0] * len(kpoints['n']),
                    "p": [0.0] * len(kpoints['p'])}
        energies_sorted = deepcopy(energies)

        # both pos_idx and energy_array are only needed by k-integration methods
        # TODO: Refactor these into their own methods
        self.pos_idx = {'n': [], 'p': []}
        self.energy_array = {'n': [], 'p': []}

        start_time = time.time()
        if not return_energies:
            return kpoints

        for i, tp in enumerate(["p", "n"]):
            for ib in range(num_bands[tp]):
                iband = (ib + self.cbm_vbm[tp]["bidx"]
                         + (i - 1) * self.cbm_vbm[tp]["included"])
                iband = 8 if tp == "p" else 9
                energies[tp] = self.interpolater.get_energies(
                    kpoints[tp], iband, scissor=self.scissor)
                self._check_timeout_hours()

                if self.integration == 'k':
                    self.energy_array[tp].append(
                        self.grid_from_ordered_list(energies[tp], tp,
                                                    none_missing=True))

                # we only need the 1st band energies to order k-points:
                if ib == 0:
                    e_sort_idx = np.array(
                        energies[tp]).argsort() if tp == "n" else np.array(
                        energies[tp]).argsort()[::-1]
                    energies_sorted[tp] = [energies[tp][ie] for ie in
                                           e_sort_idx]
                    energies[tp] = [energies[tp][ie] for ie in e_sort_idx]
                    self.pos_idx[tp] = np.array(range(len(e_sort_idx)))[
                        e_sort_idx].argsort()
                    kpoints[tp] = [kpoints[tp][ie] for ie in e_sort_idx]

        self.energy_array = {tp: np.array(self.energy_array[tp])
                             for tp in ('p', 'n')}
        self.logger.debug("time to calculate ibz energy: {}".format(
            time.time() - start_time))

        return kpoints, energies_sorted

    def _calculate_dos(self) -> np.ndarray:
        """Calculate the density of states.

        Returns:
            The density of states data, formatted as::

                (energies, densities)
        """
        kpoint_mesh = Kpoints.automatic_density_by_vol(
            self.structure, kppvol=self.dos_kdensity).kpts[0]
        self.logger.info('kmesh used for dos: {}'.format(kpoint_mesh))

        dos = self.interpolater.get_dos(
            kpoint_mesh, estep=max(self.dE_min, 0.0001),
            width=self.dos_bwidth, scissor=self.scissor, normalize=True,
            vbm_e=self.cbm_vbm['p']['energy'], cbm_e=self.cbm_vbm['n']['energy'])

        self.logger.debug("dos_emin = {} and dos_emax= {}".format(
            dos[:, 0].min(), dos[:, 0].max()))

        return dos

    def find_all_important_points(self, nbelow_vbm=0, nabove_cbm=0, **kwargs):
        """
        As the name suggests, given some pre-populated variables such as band_structure and
        cbm_vbm and the rest of input args, it calculates the k-coordinates of
        the band structure extrema (valence band maxima or conduction band
        minima) at the specified bands which are the "important" points. Note
        that the symmetrically equivalent k-points and those points that are
        too close to each other are excluded.

        Args:
            nbelow_vbm (int): sets how many bands below the VBM is the target
                valence band
            nabove_cbm (int): sets how many bands above the CBM is the target
                conduction band
            **kwargs: other keyword arguments related to get_bs_extrema such
                as line_density

        Returns ({"n": [3x1 array], "p":[3x1 array]}):
            list of band extrema (i.e. important k-points) for the selected
            conduction ("n") and valence ("p") band.
        """
        if self.important_pts is None or nbelow_vbm + nabove_cbm > 0:
            extrema_vb, new_vb = self.interpolater.get_extrema(
                self.cbm_vbm['p']['bidx'] - nbelow_vbm,
                e_cut=self.Ecut['p'], return_global_extrema=True,
                scissor=self.scissor, **kwargs)
            extrema_cb, new_cb = self.interpolater.get_extrema(
                self.cbm_vbm['n']['bidx'] + nabove_cbm,
                e_cut=self.Ecut['n'], return_global_extrema=True,
                scissor=self.scissor, **kwargs)

            self.important_pts = {'p': extrema_vb,
                                  'n': extrema_cb}

            if new_cb['energy'] < self.cbm_vbm['n']['energy']:
                self.cbm_vbm['n']['energy'] = new_cb['energy']
                self.cbm_vbm['n']['kpoint'] = new_cb['kpoint']

            if new_vb['energy'] > self.cbm_vbm['p']['energy']:
                self.cbm_vbm['p']['energy'] = new_vb['energy']
                self.cbm_vbm['p']['kpoint'] = new_vb['kpoint']

        self.logger.info('Here all the initial extrema (valleys):\n{}'.format(
            self.important_pts))
        return deepcopy(self.important_pts)

    def write_input_files(self, path=None, dir_name="run_data"):
        """
        Writes all 3 types of inputs in json files for example to conveniently
        track what inputs had been used later or read inputs from files (see
        from_files method)

        Args:
            path (str): the final path to write the input files
            dir_name (str): the name of the folder where the input files are
                written; ignored if path is set.

        Returns (None):
        """
        path = os.path.join(path or self.calc_dir, dir_name)
        if not os.path.exists(path):
            os.makedirs(name=path)

        material_params = self.material_params.copy()
        if self.W_POP:
            material_params["W_POP"] = self.W_POP / (1e12 * 2 * pi)
        model_params = self.model_params
        performance_params = self.performance_params

        with open(os.path.join(path, "material_params.json"), "w") as fp:
            json.dump(material_params, fp, sort_keys=True, indent=4,
                      ensure_ascii=False, cls=MontyEncoder)
        with open(os.path.join(path, "model_params.json"), "w") as fp:
            json.dump(model_params, fp, sort_keys=True, indent=4,
                      ensure_ascii=False, cls=MontyEncoder)
        with open(os.path.join(path, "performance_params.json"), "w") as fp:
            json.dump(performance_params, fp, sort_keys=True, indent=4,
                      ensure_ascii=False, cls=MontyEncoder)

    def set_material_params(self, params):
        """
        Set (or retrieve from input parameters) the material's parameters.
        This method is meant to be called after set_model_parameters as it may
        modify self.model_parameters.

        Args:
            params (dict): some parameters such as "epsilon_s" are mandatory.
                for example params={} is not acceptable but the following is:
                params = {'epsilon_s': 11.111}

        Returns (None):
            Sets the corresponding material parameters and material_params
                instance variables.
        """
        self.epsilon_s = params["epsilon_s"]
        self.P_PIE = params.get("P_PIE", None) or 0.15  # unitless
        if self.P_PIE is None and 'PIE' in self.elastic_scats:
            self.logger.warning('"PIE" removed from the list. It cannot be '
                                'modeled w/o P_PIE in material_params')
            self.elastic_scats.pop(self.elastic_scats.index('PIE'))
        E_D = params.get("E_D", None)
        self.C_el = params.get("C_el", None)
        if (E_D is None or self.C_el is None) and 'ACD' in self.elastic_scats:
            self.logger.warning('"ACD" removed from the list. It cannot be '
                                'modeled w/o C_el or E_D in material_params')
            self.elastic_scats.pop(self.elastic_scats.index('ACD'))
        if isinstance(E_D, dict):
            if 'n' not in E_D and 'p' not in E_D:
                raise ValueError('Neither "n" nor "p" keys not found in E_D')
            self.E_D = E_D
        elif E_D:
            self.E_D = {'n': E_D, 'p': E_D}
        else:
            self.E_D = E_D

        self.epsilon_inf = params.get("epsilon_inf", None)
        self.W_POP = params.get("W_POP", None)
        if self.W_POP:
            self.W_POP *= 1e12 * 2 * pi  # convert to THz
        if 'POP' in self.inelastic_scats:
            if self.epsilon_inf is None or self.W_POP is None:
                warnings.warn(
                    'POP cannot be calculated w/o epsilon_inf and W_POP')
                self.inelastic_scats.pop(self.inelastic_scats.index('POP'))

        self.N_dis = params.get("N_dis", 0.1)  # in 1/cm**2
        self.scissor = params.get("scissor", 0.0)
        self.user_bandgap = params.get("user_bandgap", None)

        donor_charge = params.get("donor_charge", 1.0)
        acceptor_charge = params.get("acceptor_charge", 1.0)
        dislocations_charge = params.get("dislocations_charge", 1.0)
        self.charge = {"n": donor_charge, "p": acceptor_charge,
                       "dislocations": dislocations_charge}
        self.important_pts = params.get('important_points', None)
        self.material_params = {
            "epsilon_s": self.epsilon_s,
            "epsilon_inf": self.epsilon_inf,
            "C_el": self.C_el,
            "P_PIE": self.P_PIE,
            "E_D": self.E_D,
            "N_dis": self.N_dis,
            "scissor": self.scissor,
            "donor_charge": self.charge["n"],
            "acceptor_charge": self.charge["p"],
            "dislocations_charge": self.charge["dislocations"],
            "important_points": self.important_pts,
            "user_bandgap": self.user_bandgap,
            "W_POP": self.W_POP
        }

    def set_model_params(self, params=None):
        """
        Set (or retrieve from input parameters) instance variables related to
        the model and the level of the theory; these are set based on params
        (dict) set by the user or their default values

        Args:
            params (dict):

        Returns (None):
        """
        params = params or {}
        self.bs_is_isotropic = params.get("bs_is_isotropic", True)
        self.elastic_scats = params.get("elastic_scats", ["ACD", "IMP", "PIE"])
        self.inelastic_scats = params.get("inelastic_scats", ["POP"])
        self.parabolic_bands0 = params.get("parabolic_bands", None)
        self.parabolic_bands = self.parabolic_bands0
        self.soc = params.get("soc", False)
        self.logger.info("bs_is_isotropic: {}".format(self.bs_is_isotropic))
        self.model_params = {
            "bs_is_isotropic": self.bs_is_isotropic,
            "elastic_scats": self.elastic_scats,
            "inelastic_scats": self.inelastic_scats,
            "parabolic_bands": self.parabolic_bands
        }

    def set_performance_params(self, params=None):
        """
        Set (or retrieve from input parameters) that are related to running
        performance and speed and store them as corresponding instance variables

        Args:
            params (dict): must be at least {} to invoke all the default values
                examples are {} or {'Ecut': 1.0, "max_nvalleys": 3}

        Returns (None):
        """
        params = params or {}
        self.dE_min = params.get("dE_min", 0.0001)
        self.nE_min = params.get("nE_min", 5)
        c_factor = max(1., max(
            [log(abs(ci) / 1e19) for ci in self.dopings] + [1.]) ** 0.5)
        Ecut = params.get("Ecut",
                          c_factor * 5 * k_B * max(self.temperatures + [600]))
        self.max_Ecut = params.get("Ecut",
                                   1.5)  # TODO-AF: set this default Encut based on maximum energy range that the current BS covers between
        Ecut = min(Ecut, self.max_Ecut)
        self.Ecut = {tp: Ecut if tp in self.all_types else Ecut * 2. / 3. for tp
                     in ["n", "p"]}
        for tp in ["n", "p"]:
            self.logger.debug("{}-Ecut: {} eV \n".format(tp, self.Ecut[tp]))
        self.dos_bwidth = params.get("dos_bwidth", 0.075)
        self.dos_kdensity = params.get("dos_kdensity", 5500)
        # self.dos_kdensity = params.get("dos_kdensity", 1500) # just for rapid testing
        self.v_min = 1000
        self.gs = float(
            1e-32)  # small value (e.g. used for an initial non-zero val)
        self.gl = float(1e32)  # global large value
        self.BTE_iters = params.get("BTE_iters", 5)
        self.n_jobs = params.get("n_jobs", -1)
        if self.n_jobs == -1:
            self.n_jobs = min(cpu_count(), 16)
        self.max_nbands = params.get("max_nbands", None)
        self.max_normk0 = params.get("max_normk", None)
        self.max_normk = {'n': self.max_normk0, 'p': self.max_normk0}
        self.max_nvalleys = params.get("max_nvalleys", None)
        if not isinstance(self.max_nvalleys, dict):
            self.max_nvalleys = {
                tp: self.max_nvalleys if tp in self.all_types else 1 for tp in
                ['p', 'n']}
        self.performance_params = {
            "dE_min": self.dE_min,
            "Ecut": self.Ecut,
            "max_Ecut": self.max_Ecut,
            "dos_bwidth": self.dos_bwidth,
            "dos_kdensity": self.dos_kdensity,
            "BTE_iters": self.BTE_iters,
            "max_nbands": self.max_nbands,
            "max_normk0": self.max_normk0,
            "max_nvalleys": self.max_nvalleys,
            "n_jobs": self.n_jobs,
        }

    def __getitem__(self, key):
        if key == "kgrid":
            return self.kgrid
        elif key == "egrid":
            return self.egrid
        else:
            raise KeyError

    def get_cartesian_coords(self, frac_k, reciprocal=True):
        """
        Transformation from fractional too cartesian. Note that this is
        different form get_cartesian_coords method available in
        self.structure.lattice.reciprocal_lattice, that one does NOT work with
        BolzTraP outputs

        Args:
            frac_k (np.ndarray): a 3-D vector in fractional (unitless)
            coordinates or a list of such coordinates
            reciprocal (bool): whether the cartesian output is in real (Angstrom)
                or reciprocal space (1/Angstrom).

        Returns (np.ndarray): frac_k ransformed into cartesian coordinates
        """
        if reciprocal:
            return np.dot(self.structure.lattice.reciprocal_lattice.matrix,
                          np.array(frac_k))
        else:
            return np.dot(self.structure.lattice.matrix, np.array(frac_k))

    def seeb_int_num(self, c, T):
        """
        Returns the numerator of the integral term in eq (52) of [R] for
        calculation of the Seebeck coefficient.

        *This is a wrapper function used as an input to calculate_property.*

        Args:
            c (int): the carrier concentration <0 for electrons, >0 for holes
            T (int): temperature in Kelvin

        Returns (dict: {"n": float, "p": float}): Seebeck integral numerator
            integrated over the energy scale (egrid).
        """
        fn = lambda E, fermi, T: f0(E, fermi, T) * (1 - f0(E, fermi, T)) * E / (
                k_B * T)
        return {t: self.integrate_func_over_E(func=fn, tp=t, T=T,
                                              fermi=self.fermi_level[c][T],
                                              normalize_energy=True,
                                              xDOS=True) for t in ["n", "p"]}

    def seeb_int_denom(self, c, T):
        """
        Returns the denominator of the integral term in eq (52) of [R] for
        calculation of the Seebeck coefficient.

        *This is a wrapper function used as an input to calculate_property.*

        Args:
            c (int): the carrier concentration <0 for electrons, >0 for holes
            T (int): temperature in Kelvin

        Returns:
            (dict: {"n": float, "p": float}): Seebeck integral denominator
            integrated over the energy scale (egrid).
        """
        return {t: self.gs + self.integrate_over_E(
            props=["f0x1-f0"], tp=t, c=c, T=T, xDOS=True) for t in ["n", "p"]}

    def calculate_property(self, prop_name, prop_func, for_all_E=False):
        """
        Calculates the propery at all concentrations and temperatures using
        the given function and insert it into self.egrid

        Args:
            prop_name (str): the name of the property
            prop_func (obj): the given function MUST takes c and T as required
                inputs in this order.

        Returns (None):
            populates the prop_name in egrid.
        """
        if for_all_E:
            for tp in ["n", "p"]:
                self.egrid[tp][prop_name] = \
                    {c: {T: [self.gs] * len(self.egrid[tp]["energy"]) \
                         for T in self.temperatures} for c in self.dopings}
        else:
            self.egrid[prop_name] = {c: {T: self.gs for T in self.temperatures
                                         } for c in self.dopings}
        for c in self.dopings:
            for T in self.temperatures:
                if for_all_E:
                    fermi = self.fermi_level[c][T]
                    for tp in ["n", "p"]:
                        for ie, E in enumerate(self.egrid[tp]["energy"]):
                            self.egrid[tp][prop_name][c][T][ie] = prop_func(E,
                                                                            fermi,
                                                                            T)
                else:
                    self.egrid[prop_name][c][T] = prop_func(c, T)

    def calculate_N_II(self, c, T):
        """
        Returns the total simulated ionoized impurity concentration.

        Args:
            c (int): the carrier concentration
            T (int): the temperature in kelvin

        Returns (float): inoized impurity (IMP) scattering concentration (N_II)
        """
        N_II = abs(self.calc_doping[c][T]["n"]) * self.charge["n"] ** 2 + \
               abs(self.calc_doping[c][T]["p"]) * self.charge["p"] ** 2 + \
               self.N_dis / self.structure.volume ** (1 / 3) * 1e8 * \
               self.charge["dislocations"] ** 2
        # N_dis is a given 2D concentration of charged dislocations in 1/cm**2
        return N_II

    def pre_init_egrid(self):
        """
        Just to initialize the energy grid (egrid) and the energy values as
        opposed to all the variables defined in the final egrid.

        Returns ([str]): keep track of which types failed; "n" or "p" type or
            neither. Cause of failure could be that too few k-points are left
            for that type.
        """
        min_nE = 2
        corrupt_tps = []
        self.egrid = {
            "n": {"energy": [], "DOS": [], "all_en_flat": [],
                  "all_ks_flat": []},
            "p": {"energy": [], "DOS": [], "all_en_flat": [],
                  "all_ks_flat": []},
        }
        self.kgrid_to_egrid_idx = {"n": [], "p": []}
        self.Efrequency = {"n": [], "p": []}
        self.sym_freq = {"n": [], "p": []}
        E_idx = {"n": [], "p": []}
        for tp in ["n", "p"]:
            for ib, en_vec in enumerate(self.kgrid[tp]["energy"]):
                self.egrid[tp]["all_en_flat"] += list(en_vec)
                self.egrid[tp]["all_ks_flat"] += list(
                    self.kgrid[tp]["kpoints"][ib])
                E_idx[tp] += [(ib, iek) for iek in range(len(en_vec))]
            ieidxs = np.argsort(self.egrid[tp]["all_en_flat"])
            self.egrid[tp]["all_en_flat"] = [self.egrid[tp]["all_en_flat"][ie]
                                             for ie in ieidxs]
            self.egrid[tp]["all_ks_flat"] = [self.egrid[tp]["all_ks_flat"][ie]
                                             for ie in ieidxs]
            E_idx[tp] = [E_idx[tp][ie] for ie in ieidxs]

        # setting up energy grid and DOS:
        for tp in ["n", "p"]:
            energy_counter = []
            i = 0
            last_is_counted = False
            while i < len(self.egrid[tp]["all_en_flat"]) - 1:
                sum_E = self.egrid[tp]["all_en_flat"][i]
                sum_nksym = len(remove_duplicate_kpoints(
                    self.band_structure.get_sym_eq_kpoints(
                        self.egrid[tp]["all_ks_flat"][i])))
                counter = 1.0  # because the ith member is already included in sum_E
                current_ib_ie_idx = [E_idx[tp][i]]
                j = i
                while j < len(self.egrid[tp]["all_en_flat"]) - 1 and \
                        abs(self.egrid[tp]["all_en_flat"][i] -
                            self.egrid[tp]["all_en_flat"][j + 1]) < self.dE_min:
                    counter += 1
                    current_ib_ie_idx.append(E_idx[tp][j + 1])
                    sum_E += self.egrid[tp]["all_en_flat"][j + 1]
                    sum_nksym += len(remove_duplicate_kpoints(
                        self.band_structure.get_sym_eq_kpoints(
                            self.egrid[tp]["all_ks_flat"][i + 1])))

                    if j + 1 == len(self.egrid[tp]["all_en_flat"]) - 1:
                        last_is_counted = True
                    j += 1
                self.egrid[tp]["energy"].append(sum_E / counter)
                self.kgrid_to_egrid_idx[tp].append(current_ib_ie_idx)
                self.sym_freq[tp].append(sum_nksym / counter)
                energy_counter.append(counter)
                self.egrid[tp]["DOS"].append(
                    self.dos[self.get_Eidx_in_dos(sum_E / counter)][1])
                i = j + 1

            if not last_is_counted:
                self.egrid[tp]["energy"].append(
                    self.egrid[tp]["all_en_flat"][-1])
                self.kgrid_to_egrid_idx[tp].append([E_idx[tp][-1]])
                self.egrid[tp]["DOS"].append(self.dos[self.get_Eidx_in_dos(
                    self.egrid[tp]["energy"][-1])][1])
            self.egrid[tp]["size"] = len(self.egrid[tp]["energy"])
            self.Efrequency[tp] = [len(Es) for Es in
                                   self.kgrid_to_egrid_idx[tp]]
            if len(self.Efrequency[tp]) < min_nE:
                warnings.warn(
                    "The final {}-egrid have fewer than {} energy values".format(
                        tp, min_nE))
                corrupt_tps.append(tp)
        return corrupt_tps

    def init_egrid(self):
        """
        Initializes the self.egrid dict containing energy grid and relevant
        properties such as "DOS". This must be called after pre_init_egrid so
        that the energy values are already populated.

        Returns ([str]): keep track of which types failed; "n" or "p" type or
            neither. Cause of failure could be that too few k-points are left
            for that type.
        """
        corrupt_tps = self.pre_init_egrid()
        if "n" in corrupt_tps and "p" in corrupt_tps:
            return corrupt_tps
        for tp in ['n', 'p']:
            self.egrid[tp]["relaxation time constant"] = {
                c: {T: 0.0 for T in self.temperatures} for c in self.dopings}

        self.calculate_property(prop_name="f0", prop_func=f0, for_all_E=True)
        self.calculate_property(prop_name="f", prop_func=f0, for_all_E=True)
        self.calculate_property(prop_name="f_th", prop_func=f0, for_all_E=True)

        for prop in ["f", "f_th"]:
            self.map_to_egrid(prop_name=prop, c_and_T_idx=True)

        self.calculate_property(prop_name="f0x1-f0", prop_func=lambda E, fermi,
                                                                      T: f0(E,
                                                                            fermi,
                                                                            T) * (
                                                                                 1 - f0(
                                                                             E,
                                                                             fermi,
                                                                             T)),
                                for_all_E=True)
        for c in self.dopings:
            for T in self.temperatures:
                fermi = self.fermi_level[c][T]
                for tp in ["n", "p"]:
                    for ib in range(len(self.kgrid[tp]["energy"])):
                        for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                            E = self.kgrid[tp]["energy"][ib][ik]
                            self.kgrid[tp]["f0"][c][T][ib][ik] = f0(E, fermi, T)

        self.calculate_property(prop_name="beta",
                                prop_func=self.inverse_screening_length)
        self.logger.debug('inverse screening length, beta is \n{}'.format(
            self.egrid["beta"]))
        self.calculate_property(prop_name="N_II",
                                prop_func=self.calculate_N_II)
        self.calculate_property(prop_name="Seebeck_integral_numerator",
                                prop_func=self.seeb_int_num)
        self.calculate_property(prop_name="Seebeck_integral_denominator",
                                prop_func=self.seeb_int_denom)
        return corrupt_tps

    def get_Eidx_in_dos(self, E, Estep=None):
        """
        After the density of states (DOS) is initialized this function returns
        the index of a given energy level
        Args:
            E (float): the energy level in eV
            Estep (float): optional, the energy step used in calculating DOS
        Returns (int): index of the energy level in the DOS
        """
        if not Estep:
            Estep = max(self.dE_min, 0.0001)
        calculated_index = int(round((E - self.dos[:, 0].min()) / Estep))
        return min(calculated_index, len(self.dos[:, 0]) - 1)

    def G(self, tp, ib, ik, ib_prm, ik_prm, X):
        """
        The overlap integral betweek vectors k and k'

        Args:
            ik (int): index of vector k in kgrid
            ik_prm (int): index of vector k' in kgrid
            X (float): cosine of the angle between vectors k and k'

        Returns (float): the overlap integral
        """
        a = self.kgrid[tp]["a"][ib][ik]
        c = self.kgrid[tp]["c"][ib][ik]
        return (a * self.kgrid[tp]["a"][ib_prm][ik_prm] + \
                X * c * self.kgrid[tp]["c"][ib_prm][ik_prm]) ** 2

    def remove_indexes(self, rm_idx_list, rearranged_props):
        """
        The k-points with velocity < 1 cm/s (either in valence or conduction band) are taken out as those are
            troublesome later with extreme values (e.g. too high elastic scattering rates)

        Args:
            rm_idx_list ([int]): the kpoint indexes that need to be removed for each property
            rearranged_props ([str]): list of properties for which some indexes need to be removed
        """
        for i, tp in enumerate(["n", "p"]):
            for ib in range(self.cbm_vbm[tp]["included"]):
                rm_idx_list_ib = list(set(rm_idx_list[tp][ib]))
                rm_idx_list_ib.sort(reverse=True)
                rm_idx_list[tp][ib] = rm_idx_list_ib
                self.logger.debug(
                    "# of {}-type kpoints indexes with low velocity or off-energy: {}".format(
                        tp, len(rm_idx_list_ib)))
            for prop in rearranged_props:
                self.kgrid[tp][prop] = \
                    np.array([np.delete(self.kgrid[tp][prop][ib],
                                        rm_idx_list[tp][ib], axis=0) \
                              for ib in range(self.cbm_vbm[tp]["included"])])

    def initialize_var(self, grid, names, val_type="scalar", initval=0.0,
                       is_nparray=True, c_T_idx=False):
        """
        Initializes a variable/key within the self.kgrid variable

        Args:
            grid (str): options are "kgrid" or "egrid": whether to initialize vars in self.kgrid or self.egrid
            names (list): list of the names of the variables
            val_type (str): options are "scalar", "vector", "matrix" or "tensor"
            initval (float): the initial value (e.g. if val_type=="vector", each of the vector's elements==init_val)
            is_nparray (bool): whether the final initial content is an numpy.array or not.
            c_T_idx (bool): whether to define the variable at each concentration, c, and temperature, T.
        """
        if not isinstance(names, list):
            names = [names]
        if val_type.lower() in ["scalar"]:
            initial_val = initval
        elif val_type.lower() in ["vector"]:
            initial_val = [initval, initval, initval]
        elif val_type.lower() in ["tensor", "matrix"]:
            initial_val = [[initval for _ in range(3)] for _ in range(3)]

        for name in names:
            for tp in ["n", "p"]:
                self[grid][tp][name] = 0.0
                if grid == "kgrid":
                    init_content = [[initial_val for _ in
                                     range(len(self[grid][tp]["kpoints"][j]))]
                                    for j in
                                    range(self.cbm_vbm[tp]["included"])]
                elif grid == "egrid":
                    init_content = [initial_val for _ in
                                    self[grid][tp]["energy"]]
                else:
                    raise TypeError(
                        'The argument "grid" must be set to either "kgrid" or "egrid"')
                if is_nparray:
                    if not c_T_idx:
                        self[grid][tp][name] = np.array(init_content)
                    else:
                        self[grid][tp][name] = {
                            c: {T: np.array(init_content) for T in
                                self.temperatures} for c in
                            self.dopings}
                else:
                    # TODO: if not is_nparray both temperature values will be equal probably because both are equal to init_content that are a list and FOREVER they will change together. Keep is_nparray as True as it makes a copy, otherwise you are doomed! See if you can fix this later
                    if val_type not in ["scalar"] and c_T_idx:
                        raise ValueError(
                            "For now keep using is_nparray=True to see why for not is_nparray everything becomes equal at all temepratures (lists are not copied but they are all the same)")
                    else:
                        if not c_T_idx:
                            self[grid][tp][name] = init_content
                        else:
                            self[grid][tp][name] = {
                                c: {T: init_content for T in self.temperatures}
                                for
                                c in
                                self.dopings}

    def init_kgrid(self, kpts, important_points, delete_off_points=True):
        """
        Initializes the kgrid dict that contain various properties such as
        energy, velocity, cartesian_kpoints, etc for both "n"-type (conduction
        bands) and "p"-type (valence bands) based on the k-point mesh, kpts,
        surrounding the important_points that are the important extrema.

        Args:
            kpts ([3x1 arrays]): list of k-points fractional coordinates
            important_points ({"n": [3x1 arrays], "p": [3x1 arrays]}):
            delete_off_points (bool): whether to delete points that have too
                low velocity or energy values too off from CBM/VBM, etc. True
                is recommended, set False only for testing

        Returns ([str]): keep track of which types failed; "n" or "p" type or
            neither. Cause of failure could be that too few k-points are left
            for that type.
        """
        corrupt_tps = []
        self.kgrid = {
            "n": {},
            "p": {}}
        self.num_bands = {"n": 1, "p": 1}
        for tp in ["n", "p"]:
            self.num_bands[tp] = self.cbm_vbm[tp]["included"]
            self.kgrid[tp]["kpoints"] = [kpts[tp] for ib in
                                         range(self.num_bands[tp])]

        self.initialize_var("kgrid", ["energy", "a", "c", "norm(v)", "norm(k)"],
                            "scalar", 0.0, is_nparray=False)
        self.initialize_var("kgrid", ["velocity"], "vector", 0.0,
                            is_nparray=False)

        start_time = time.time()
        rm_idx_list = {"n": [[] for i in range(self.cbm_vbm["n"]["included"])],
                       "p": [[] for i in range(self.cbm_vbm["p"]["included"])]}
        self.initialize_var("kgrid",
                            ["old cartesian kpoints", "cartesian kpoints"],
                            "vector", 0.0, is_nparray=False, c_T_idx=False)
        self.initialize_var("kgrid", ["norm(k)"], "scalar", 0.0,
                            is_nparray=False, c_T_idx=False)
        self.logger.debug(
            "The DFT gap right before calculating final energy values: {}".format(
                self.dft_gap))

        for i, tp in enumerate(["p", "n"]):
            self.cbm_vbm[tp]["cartesian k"] = self.get_cartesian_coords(
                self.cbm_vbm[tp]["kpoint"]) / A_to_nm
            self.cbm_vbm[tp]["all cartesian k"] = remove_duplicate_kpoints(
                self.band_structure.get_sym_eq_kpoints(
                    self.cbm_vbm[tp]["cartesian k"],
                    cartesian=True))
            sgn = (-1) ** i
            for ib in range(self.cbm_vbm[tp]["included"]):
                for ik, k in enumerate(self.kgrid[tp]['kpoints'][ib]):
                    self.kgrid[tp]["old cartesian kpoints"][ib][ik] = \
                        self.get_cartesian_coords(
                            self.kgrid[tp]["kpoints"][ib][ik]) / A_to_nm

                s_orbital, p_orbital = get_band_orbital_contributions(
                    self.band_structure,
                    self.cbm_vbm[tp]["bidx"] - 1 - sgn * ib)
                orbitals = {"s": s_orbital, "p": p_orbital}
                fit_orbs = {
                    orb: griddata(points=np.array(self.cartesian_kpoints),
                                  values=np.array(orbitals[orb]),
                                  xi=np.array(
                                      self.kgrid[tp]["old cartesian kpoints"][
                                          ib]),
                                  method='nearest') for orb in orbitals.keys()}

                iband = self.cbm_vbm[tp]["bidx"] + (i - 1) * \
                        self.cbm_vbm["p"]["included"] + ib
                iband = 8 if tp == "p" else 9
                print("tp: {}\tib: {}".format(tp, ib))
                print(self.kgrid[tp]["kpoints"][ib])
                self.kgrid[tp]["energy"][ib], self.kgrid[tp]["velocity"][ib] = \
                    self.interpolater.get_energies(
                        self.kgrid[tp]["kpoints"][ib], iband,
                        scissor=self.scissor, return_velocity=True)
                dumpfn(self.kgrid[tp]["energy"][ib], "new_energy_{}.json.gz".format(ib))
                dumpfn(self.kgrid[tp]["velocity"][ib], "new_velocity_{}.json.gz".format(ib))

                self.kgrid[tp]["cartesian kpoints"][ib] = np.array(
                    self.kgrid[tp]["old cartesian kpoints"][ib])  # made a copy

                # compare to a for-loop this map reduce time and memory usage:
                self.kgrid[tp]["cartesian kpoints"][ib] = list(
                    map(lambda k: self.get_cartesian_coords(get_closest_k(
                        k, self.band_structure.get_sym_eq_kpoints(
                            important_points[tp][0]),
                        return_diff=True)) / A_to_nm,
                        self.kgrid[tp]["kpoints"][ib]))

                self.kgrid[tp]["norm(k)"][ib] = np.linalg.norm(
                    self.kgrid[tp]["cartesian kpoints"][ib], axis=1)

                for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                    # self.kgrid[tp]["norm(v)"][ib][ik] = norm(self.kgrid[tp]["velocity"][ib][ik])
                    if (len(rm_idx_list[tp][ib]) + 20 < len(
                            self.kgrid[tp]['kpoints'][ib])) and (
                            (self.kgrid[tp]["velocity"][ib][
                                 ik] < self.v_min).all()
                            # if all members are small, that point should be removed otherwise scattering blows up and I get nan mobilities
                            or \
                            (abs(self.kgrid[tp]["energy"][ib][ik] -
                                 self.cbm_vbm[tp]["energy"]) > self.Ecut[tp]) \
                            or \
                            ((self.max_normk[tp]) and (
                                    self.kgrid[tp]["norm(k)"][ib][ik] >
                                    self.max_normk[tp]) and (
                                     self.parabolic_bands0 is None))
                            or \
                            (self.kgrid[tp]["norm(k)"][ib][ik] < 1e-3)
                    ):
                        rm_idx_list[tp][ib].append(ik)
                    if self.parabolic_bands is None:
                        self.kgrid[tp]["a"][ib][ik] = fit_orbs["s"][ik] / (
                                fit_orbs["s"][ik] ** 2 + fit_orbs["p"][
                            ik] ** 2) ** 0.5
                        if np.isnan(self.kgrid[tp]["a"][ib][ik]):
                            self.kgrid[tp]["a"][ib][ik] = np.mean(
                                self.kgrid[tp]["a"][ib][:ik])
                        self.kgrid[tp]["c"][ib][ik] = (1 -
                                                       self.kgrid[tp]["a"][ib][
                                                           ik] ** 2) ** 0.5
                    else:
                        self.kgrid[tp]["a"][ib][ik] = 1.0  # parabolic: s-only
                        self.kgrid[tp]["c"][ib][ik] = 0.0
                self.kgrid[tp]["norm(v)"][ib] = np.linalg.norm(
                    self.kgrid[tp]["velocity"][ib], axis=1)
            self.logger.debug(
                "average of the {}-type group velocity in kgrid:\n {}".format(
                    tp, np.mean(self.kgrid[tp]["velocity"][0], axis=0)))

        rearranged_props = ["velocity", "energy", "a", "c", "kpoints",
                            "cartesian kpoints", "old cartesian kpoints",
                            "norm(v)", "norm(k)"]

        self.logger.debug(
            "time to calculate E, v, m_eff at all k-points: \n {}".format(
                time.time() - start_time))
        start_time = time.time()

        for tp in ["n", "p"]:
            rm_idx_list[tp] = [rm_idx_list[tp][0] for _ in
                               range(self.cbm_vbm[tp]["included"])]
        self.rm_idx_list = deepcopy(
            rm_idx_list)  # format: [tp][ib][ik]; instance variable only used in k-integration
        if delete_off_points:
            self.remove_indexes(rm_idx_list, rearranged_props=rearranged_props)
        del rm_idx_list
        for tp in ['p', 'n']:
            self.logger.debug(
                "average of the {}-type group velocity in kgrid after removing points:\n {}".format(
                    tp, np.mean(self.kgrid[tp]["velocity"][0], axis=0)))

        self.logger.debug('current cbm_vbm:\n{}'.format(self.cbm_vbm))
        for tp in ["n", "p"]:
            for ib in range(len(self.kgrid[tp]["energy"])):
                self.logger.info(
                    "Final # of {}-kpts in band #{}: {}".format(tp, ib, len(
                        self.kgrid[tp]["kpoints"][ib])))

            if len(self.kgrid[tp]["kpoints"][0]) < 5:
                corrupt_tps.append(tp)
        self.logger.debug(
            "time to calculate energy, velocity, m* for all: {} seconds".format(
                time.time() - start_time))

        # sort "energy", "kpoints", etc based on energy in ascending order and keep track of old indexes
        e_sort_idx_2 = self.sort_vars_based_on_energy(args=rearranged_props,
                                                      ascending=True)

        # k-integration stuff:
        self.pos_idx_2 = deepcopy(e_sort_idx_2)
        for tp in ['n', 'p']:
            for ib in range(self.num_bands[tp]):
                self.pos_idx_2[tp][ib] = \
                    np.array(range(len(e_sort_idx_2[tp][ib])))[
                        e_sort_idx_2[tp][ib]].argsort()

        for tp in ["n", "p"]:
            self.kgrid[tp]["size"] = [len(self.kgrid[tp]["kpoints"][ib]) \
                                      for ib in
                                      range(len(self.kgrid[tp]["kpoints"]))]

        self.initialize_var("kgrid", ["W_POP"], "scalar", 0.0, c_T_idx=False)
        self.initialize_var("kgrid", ["N_POP"], "scalar", 0.0, c_T_idx=True)
        self.initialize_var(grid="kgrid", names=[
            "_all_elastic", "S_i", "S_i_th", "S_o", "S_o_th", "g", "g_th",
            "g_POP", "f0", "f", "f_th", "relaxation time", "df0dk",
            "electric force", "thermal force"], val_type="vector",
                            initval=self.gs, is_nparray=True, c_T_idx=True)

        for tp in ["n", "p"]:
            for ib in range(self.cbm_vbm[tp]["included"]):
                # We define W_POP in the grid this way for future W_POP(k)
                self.kgrid[tp]["W_POP"][ib] = \
                    [self.W_POP] * len(self.kgrid[tp]["kpoints"][ib])
                for c in self.dopings:
                    for T in self.temperatures:
                        self.kgrid[tp]["N_POP"][c][T][ib] = np.array(
                            [1 / (np.exp(hbar * W_POP / (k_B * T)) - 1) for
                             W_POP in self.kgrid[tp]["W_POP"][ib]])
        return corrupt_tps

    def sort_vars_based_on_energy(self, args, ascending=True):
        """
        Sorts the list of variables specified by args based on their "energy"
        values in each band for both "n"- and "p"-type bands and in ascending
        order by default.

        Args:
            args ([str]): list of arguments in self.kgrid.
            ascending (bool): whether to sort in ascending order (default)

        Returns ([int]):
            after actually modifying the args to be sorted based on energy, it
                also returns the indexes in the order that results in sorted
                energy.
        """
        ikidxs = {'n': {ib: [] for ib in range(self.num_bands['n'])},
                  'p': {ib: [] for ib in range(self.num_bands['p'])}}
        for tp in ["n", "p"]:
            for ib in range(self.cbm_vbm[tp]["included"]):
                ikidxs[tp][ib] = np.argsort(self.kgrid[tp]["energy"][ib])
                if not ascending:
                    ikidxs[tp][ib].reverse()
                for arg in args:
                    self.kgrid[tp][arg][ib] = np.array(
                        [self.kgrid[tp][arg][ib][ik] for ik in ikidxs[tp][ib]])
        return ikidxs

    def generate_angles_and_indexes_for_integration(self,
                                                    avg_Ediff_tolerance=0.02):
        """
        Generates the indexes of k' points that have the same energy (for elastic scattering) as E(k) or
        have energy equal to E(k) plus or minus of the energy of the optical phonon for inelastic scattering.
        Also, generated and stored the cosine of the angles between such points and a given input k-point

        Args:
            avg_Ediff_tolerance (float): in eV the average allowed energy difference between the target E(k') and
                what it actially is (e.g. to prevent/identify large energy differences if enforced scattering)
        """
        self.initialize_var("kgrid", ["X_E_ik", "X_Eplus_ik", "X_Eminus_ik"],
                            "scalar", [], is_nparray=False,
                            c_T_idx=False)

        # elastic scattering
        for tp in ["n", "p"]:
            for ib in range(len(self.kgrid[tp]["energy"])):
                self.nforced_scat = {"n": 0.0, "p": 0.0}
                self.ediff_scat = {"n": [], "p": []}
                for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                    self.kgrid[tp]["X_E_ik"][ib][
                        ik] = self.get_X_ib_ik_near_new_E(tp, ib, ik,
                                                          E_change=0.0,
                                                          forced_min_npoints=self.nE_min,
                                                          tolerance=self.dE_min)
                enforced_ratio = self.nforced_scat[tp] / sum(
                    [len(points) for points in self.kgrid[tp]["X_E_ik"][ib]])
                self.logger.info(
                    "enforced scattering ratio for {}-type elastic scattering at band {}:\n {}".format(
                        tp, ib, enforced_ratio))
                if enforced_ratio > 0.9:
                    warnings.warn(
                        "the k-grid is too coarse for an acceptable simulation of elastic scattering in {};"
                            .format(_doping_names[tp]))

                avg_Ediff = sum(self.ediff_scat[tp]) / max(
                    len(self.ediff_scat[tp]), 1)
                if avg_Ediff > avg_Ediff_tolerance:
                    warnings.warn(
                        "{}-type average energy difference of the enforced scattered k-points is more than"
                        " {}, try running with a more dense k-point mesh".format(
                            tp, avg_Ediff_tolerance))

        # inelastic scattering
        if "POP" in self.inelastic_scats:
            for tp in ["n", "p"]:
                sgn = (-1) ** (["n", "p"].index(tp))
                for ib in range(len(self.kgrid[tp]["energy"])):
                    self.nforced_scat = {"n": 0.0, "p": 0.0}
                    self.ediff_scat = {"n": [], "p": []}
                    for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                        self.kgrid[tp]["X_Eplus_ik"][ib][
                            ik] = self.get_X_ib_ik_near_new_E(tp, ib, ik,
                                                              E_change=+ hbar *
                                                                       self.kgrid[
                                                                           tp][
                                                                           "W_POP"][
                                                                           ib][
                                                                           ik],
                                                              forced_min_npoints=self.nE_min,
                                                              tolerance=None)
                        self.kgrid[tp]["X_Eminus_ik"][ib][
                            ik] = self.get_X_ib_ik_near_new_E(tp, ib, ik,
                                                              E_change=- hbar *
                                                                       self.kgrid[
                                                                           tp][
                                                                           "W_POP"][
                                                                           ib][
                                                                           ik],
                                                              forced_min_npoints=self.nE_min,
                                                              tolerance=None)
                    enforced_ratio = self.nforced_scat[tp] / (
                            sum([len(points) for points in
                                 self.kgrid[tp]["X_Eplus_ik"][ib]]) + \
                            sum([len(points) for points in
                                 self.kgrid[tp]["X_Eminus_ik"][ib]]))
                    self.logger.info(
                        "enforced scattering ratio: {}-type inelastic at band {}:\n{}".format(
                            tp, ib, enforced_ratio))

                    if enforced_ratio > 0.9:
                        # TODO: this should be an exception but for now I turned to warning for testing.
                        warnings.warn(
                            "the k-grid is too coarse for an acceptable simulation of POP scattering in {};"
                            " you can try this k-point grid but without POP as an inelastic scattering.".format(
                                _doping_names[tp]))

                    avg_Ediff = sum(self.ediff_scat[tp]) / max(
                        len(self.ediff_scat[tp]), 1)
                    if avg_Ediff > avg_Ediff_tolerance:
                        # TODO: this should be an exception but for now I turned to warning for testing.
                        warnings.warn(
                            "{}-type average energy difference of the enforced scattered k-points is more than"
                            " {}, try running with a more dense k-point mesh".format(
                                tp, avg_Ediff_tolerance))

    def unique_X_ib_ik_symmetrically_equivalent(self, tp, ib, ik):
        """
        Given indexes of a k-point in "n"-tp or "p"-tp grid, it returns a list
            of unique and symmetricall equivalent k-points.

        Args:
            tp (str): options are "n" for conduction bands and "p" for valence
            ib (int): band index (0 for the last valence/first conduction band)
            ik (int): k-point index in the kgrid

        Returns ([1x3 numpy.ndarray]):
            a list of k-points symmetrically equivalent to the original sorted
                on the cosine of the angle between them and the original point
        """
        frac_k = self.kgrid[tp]["kpoints"][ib][ik]
        fractional_ks = np.dot(frac_k, self.rotations)
        k = self.kgrid[tp]["kpoints"][ib][ik]
        seks = [self.get_cartesian_coords(frac_k) / A_to_nm for frac_k in
                fractional_ks]

        all_Xs = []
        new_X_ib_ik = []
        for sek in seks:
            X = cos_angle(k, sek)
            if X in all_Xs:
                continue
            else:
                new_X_ib_ik.append((X, ib, ik, sek))
                all_Xs.append(X)
        all_Xs.sort()
        return new_X_ib_ik

    def get_X_ib_ik_near_new_E(self, tp, ib, ik, E_change, forced_min_npoints=0,
                               tolerance=None):
        """
        Returns the sorted (based on angle, X) list of angle and band and
        k-point indexes of all the points that are within tolerance of E + E_change

        **Attention! this function assumes self.kgrid is sorted based on the
        energy in ascending order.**

        Args:
            tp (str): type of the band; options: "n" or "p"
            ib (int): the band index
            ik (int): the k-point index
            E_change (float): the difference between E(k') and E(k)
            forced_min_npoints (int): the number of k-points that are forcefully included in
                scattering if not enough points are found
            tolerance (float): the energy tolerance for finding the k' points that are within E_change energy of E(k)
        """
        tolerance = tolerance or self.dE_min
        E = self.kgrid[tp]["energy"][ib][ik]
        E_prm = E + E_change  # E_prm is E prime, the new energy
        k = self.kgrid[tp]["cartesian kpoints"][ib][ik]
        # we count the point itself; it does not result in self-scattering (due to 1-X term); however, it is necessary
        # to avoid zero scattering as in the integration each term is (X[i+1]-X[i])*(integrand[i]+integrand[i+1)/2
        result = [(1, ib, ik)]

        nk = len(self.kgrid[tp]["kpoints"][ib])

        for ib_prm in range(self.cbm_vbm[tp]["included"]):
            ik_closest_E = np.abs(
                self.kgrid[tp]["energy"][ib_prm] - E_prm).argmin()

            for step, start in [(1, 0), (-1, -1)]:
                ik_prm = ik_closest_E + start  # go up from ik_closest_E, down from ik_closest_E - 1
                while ik_prm >= 0 and ik_prm < nk and abs(
                        self.kgrid[tp]["energy"][ib_prm][
                            ik_prm] - E_prm) < tolerance:
                    k_prm = self.kgrid[tp]["cartesian kpoints"][ib_prm][ik_prm]
                    X_ib_ik = (cos_angle(k, k_prm), ib_prm, ik_prm)
                    if norm(self.kgrid[tp]["old cartesian kpoints"][ib_prm][
                                ik_prm] -
                            self.kgrid[tp]["old cartesian kpoints"][ib][
                                ik]) < 2 * self.max_normk[tp]:
                        result.append(X_ib_ik)
                    ik_prm += step

        if E_change != 0.0:
            ib_prm = ib
            ik_closest_E = np.abs(
                self.kgrid[tp]["energy"][ib_prm] - E_prm).argmin()

            for step, start in [(1, 0), (-1, -1)]:
                # step -1 is in case we reached the end (ik_prm == nk - 1) when
                #  we choose from the lower energy k-points
                ik_prm = ik_closest_E + start  # go up from ik_closest_E
                while ik_prm >= 0 and ik_prm < nk and len(
                        result) - 1 < forced_min_npoints:
                    # add all the k-points that have the same energy as E_prime E(k_pm); these values are stored in X_E_ik
                    for X_ib_ik in self.kgrid[tp]["X_E_ik"][ib_prm][ik_prm]:
                        X, ib_pmpm, ik_pmpm = X_ib_ik
                        X_ib_ik_new = (
                            cos_angle(k, self.kgrid[tp]["cartesian kpoints"][
                                ib_pmpm][ik_pmpm]), ib_pmpm, ik_pmpm)
                        if (X_ib_ik_new[1], X_ib_ik_new[2]) not in [
                            (entry[1], entry[2]) for entry in result]:
                            result.append(X_ib_ik_new)
                        self.nforced_scat[tp] += 1

                    self.ediff_scat[tp].append(
                        self.kgrid[tp]["energy"][ib][ik] -
                        self.kgrid[tp]["energy"][ib_prm][ik_prm])
                    ik_prm += step

        result.sort(key=lambda x: x[0])
        return result

    def s_el_eq(self, sname, tp, c, T, k, k_prm):
        """
        Returns the scattering rate at wave vector k at a certain concentration and temperature
        for a specific elastic scattering mechanisms determined by sname

        Args:
            sname (string): abbreviation of the name of the elastic scatteirng mechanisms; options: IMP, ADE, PIE, DIS
            c (int): carrier concentration
            T (int): the temperature
            k (list): list containing fractional coordinates of the k vector
            k_prm (list): list containing fractional coordinates of the k prime vector
        """
        norm_diff_k = norm(
            k - k_prm)  # the slope for PIE and IMP don't match with bs_is_isotropic
        if norm_diff_k == 0.0:
            warnings.warn(
                "WARNING!!! same k and k' vectors as input of the elastic scattering equation")
            return 0.  # 0

        if sname.upper() in ["IMP"]:  # ionized impurity scattering
            unit_conversion = 0.001 / e ** 2
            return unit_conversion * e ** 4 * self.egrid["N_II"][c][T] / \
                   (4.0 * pi ** 2 * self.epsilon_s ** 2 * epsilon_0 ** 2 * hbar) \
                   / ((norm_diff_k ** 2 + self.egrid["beta"][c][T][
                tp] ** 2) ** 2)

        elif sname.upper() in [
            "ACD"]:  # acoustic deformation potential scattering
            unit_conversion = 1e18 * e
            return unit_conversion * k_B * T * self.E_D[tp] ** 2 / (
                    4.0 * pi ** 2 * hbar * self.C_el)

        elif sname.upper() in ["PIE"]:  # piezoelectric scattering
            unit_conversion = 1e9 / e
            return unit_conversion * e ** 2 * k_B * T * self.P_PIE ** 2 \
                   / (
                           norm_diff_k ** 2 * 4.0 * pi ** 2 * hbar * epsilon_0 * self.epsilon_s)

        elif sname.upper() in ["DIS"]:
            return self.gs

        else:
            raise ValueError(
                "The elastic scattering name {} is not supported!".format(
                    sname))

    def integrate_func_over_E(self, func, tp, fermi, T,
                              interpolation_nsteps=None, xDOS=True,
                              normalize_energy=False):
        """
        Integrates a single function (func) over the egrid.

        Args:
            func (object): a function object
            tp (str): options are "n" or "p"
            fermi (float): the fermi level
            T (int): the temperature in Kelvin
            interpolation_nsteps (None or int): the resolution of integration
            xDOS (bool): whether to multiply each energy value by its density
                of states (DOS)
            normalize_energy (bool): whether to set the CBM/VBM as the reference

        Returns:
            (float): The integral value of func, integrated over the egrid.
        """
        if not interpolation_nsteps:
            interpolation_nsteps = max(200, int(
                500.0 / len(self.egrid[tp]["energy"])))
        integral = 0.0
        if normalize_energy:
            fermi -= self.cbm_vbm[tp]["energy"]
        for ie in range(len(self.egrid[tp]["energy"]) - 1):
            E = self.egrid[tp]["energy"][ie]
            dE = (self.egrid[tp]["energy"][ie + 1] - E) / interpolation_nsteps
            if normalize_energy:
                E -= self.cbm_vbm[tp]["energy"]
            if xDOS:
                dS = (self.egrid[tp]["DOS"][ie + 1] - self.egrid[tp]["DOS"][
                    ie]) / interpolation_nsteps
                for i in range(interpolation_nsteps):
                    integral += dE * (
                            self.egrid[tp]["DOS"][ie] + i * dS) * func(
                        E + i * dE, fermi, T)
            else:
                for i in range(interpolation_nsteps):
                    integral += dE * func(E + i * dE, fermi, T)
        return integral

    def find_dv(self, grid):
        dv = np.zeros(grid[:, :, :, 0].shape)
        # N is a vector of the number of x, y, and z points
        N = grid.shape

        for i in range(N[0]):
            for j in range(N[1]):
                for k in range(N[2]):
                    if i > 0:
                        dx1 = (grid[i, j, k, 0] - grid[i - 1, j, k, 0]) / 2
                    else:
                        dx1 = grid[i, j, k, 0] - (-0.5)
                    if i < N[0] - 1:
                        dx2 = (grid[i + 1, j, k, 0] - grid[i, j, k, 0]) / 2
                    else:
                        dx2 = 0.5 - grid[i, j, k, 0]

                    if j > 0:
                        dy1 = (grid[i, j, k, 1] - grid[i, j - 1, k, 1]) / 2
                    else:
                        dy1 = grid[i, j, k, 1] - (-0.5)
                    if j < N[1] - 1:
                        dy2 = (grid[i, j + 1, k, 1] - grid[i, j, k, 1]) / 2
                    else:
                        dy2 = 0.5 - grid[i, j, k, 1]

                    if k > 0:
                        dz1 = (grid[i, j, k, 2] - grid[i, j, k - 1, 2]) / 2
                    else:
                        dz1 = grid[i, j, k, 2] - (-0.5)
                    if k < N[2] - 1:
                        dz2 = (grid[i, j, k + 1, 2] - grid[i, j, k, 2]) / 2
                    else:
                        dz2 = 0.5 - grid[i, j, k, 2]
                    # find fractional volume
                    dv[i, j, k] = (dx1 + dx2) * (dy1 + dy2) * (dz1 + dz2)

        # convert from fractional to cartesian (k space) volume
        dv *= self.structure.lattice.reciprocal_lattice.volume / (
                A_to_m * m_to_cm) ** 3
        return dv

    def integrate_over_k(self, func_grid, tp):
        """
        Takes a coordinate grid in the form of a numpy array (CANNOT have
        missing points) and a function to integrate and finds the integral
        using finite differences; missing points should be input as 0 in the
        .function

        Args:
            func_grid: in the interest of not prematurely optimizing, func_grid
                must be a  perfect grid: the only deviation from
                the cartesian coordinate system can be uniform stretches,
                distance between adjacent planes of points as in the
                can be any value, but no points can be missing from the next
                plane in this case the format of fractional_grid is a 4d grid
                the last dimension is a vector of the k point fractional coordinates
                the dv grid is 3d and the indexes correspond to those of func_grid

        Returns:
        """
        if func_grid.ndim == 3:
            return np.sum(func_grid * self.dv_grid[tp])
        return [np.sum(func_grid[:, :, :, i] * self.dv_grid[tp]) for i in
                range(func_grid.shape[3])]

    def integrate_over_E(self, props, tp, c, T, xDOS=False, xvel=False,
                         interpolation_nsteps=None):
        """
        Integrates the multiplication of props in the egrid over dE where
        E stands for energy.

        Args:
            props ([str]): list of property names. These properties must
                be available in egrid such as "g" or "df0dk" or "ACD". Note
                that some simple math operations of properties are permitted.
                examples::

                    "1 - f0"
                    "f0x1-f0"
                    "/ACD"

            tp (str): "p" (valence bands) or "n" (conduction bands) type
            c (int): the carrier concentration
            T (int): the absolute temperature in Kelvin
            xDOS (bool): whether to multiply the integrand by density of states
            xvel (bool): whether to multiply the integrand by the group velocity
            interpolation_nsteps (int): number of steps (dE) between each
                energy levels in egrid; the higher the more accurate but slower

        Returns (float): the result of the integrations
        """
        imax_occ = len(self.Efrequency[tp][:-1])

        if not interpolation_nsteps:
            interpolation_nsteps = max(200, int(
                500.0 / len(self.egrid[tp]["energy"])))
        diff = [0.0 for _ in props]
        integral = self.gs
        for ie in range(imax_occ):
            E = self.egrid[tp]["energy"][ie]
            dE = abs(
                self.egrid[tp]["energy"][ie + 1] - E) / interpolation_nsteps
            if xDOS:
                dS = (self.egrid[tp]["DOS"][ie + 1] - self.egrid[tp]["DOS"][
                    ie]) / interpolation_nsteps
            if xvel:
                dv = (self.egrid[tp]["velocity"][ie + 1] -
                      self.egrid[tp]["velocity"][ie]) / interpolation_nsteps
            for j, p in enumerate(props):
                if "/" in p:
                    diff[j] = (self.egrid[tp][p.split("/")[-1]][c][T][ie + 1] -
                               self.egrid[tp][p.split("/")[-1]][c][T][
                                   ie]) / interpolation_nsteps
                elif "1 -" in p:
                    diff[j] = (1 - self.egrid[tp][
                        p.split("-")[-1].replace(" ", "")][c][T][ie + 1] - (1 - \
                                                                            self.egrid[
                                                                                tp][
                                                                                p.split(
                                                                                    "-")[
                                                                                    -1].replace(
                                                                                    " ",
                                                                                    "")][
                                                                                c][
                                                                                T][
                                                                                ie])) / interpolation_nsteps
                else:
                    diff[j] = (self.egrid[tp][p][c][T][ie + 1] -
                               self.egrid[tp][p][c][T][
                                   ie]) / interpolation_nsteps
            for i in range(interpolation_nsteps):
                multi = dE
                for j, p in enumerate(props):
                    if p[0] == "/":
                        multi /= self.egrid[tp][p.split("/")[-1]][c][T][ie] + \
                                 diff[j] * i
                    elif "1 -" in p:
                        multi *= 1 - self.egrid[tp][
                            p.split("-")[-1].replace(" ", "")][c][T][ie] + diff[
                                     j] * i
                    else:
                        multi *= self.egrid[tp][p][c][T][ie] + diff[j] * i
                if xDOS:
                    multi *= self.egrid[tp]["DOS"][ie] + dS * i
                if xvel:
                    multi *= self.egrid[tp]["velocity"][ie] + dv * i
                integral += multi
        return np.array(integral)

    def integrate_over_X(self, tp, X_E_index, integrand, ib, ik, c, T,
                         sname=None, g_suffix=""):
        """
        integrate numerically with a simple trapezoidal algorithm.

        Args:
            tp (str): 'n' or 'p' type
            X_E_index ([[[(float, int, int)]]]): list of (X, ib', ik') for each
                k-point at each band
            integrand (func): the integrand function; options: el_integrand_X
                or inel_integrand_X for elastic and inelastic respectively

            ib (int): the band index
            ik (int): the k-point index
            c (float): the carrier concentration
            T (float): the temperature
            sname (str): the scattering name (see options in the documentation
                of el_integrand_X and inel_integrand_X functions
            g_suffix:

        Returns (float or numpy.array): the integrated value/vector
        """
        summation = 0.0
        if len(X_E_index[ib][ik]) == 0:
            raise ValueError(
                "enforcing scattering points did NOT work, {}[{}][{}] is empty".format(
                    X_E_index, ib, ik))
        X, ib_prm, ik_prm = X_E_index[ib][ik][0]
        current_integrand = integrand(tp, c, T, ib, ik, ib_prm, ik_prm, X,
                                      sname=sname, g_suffix=g_suffix)

        ikp = 0
        while ikp < len(X_E_index[ib][ik]) - 1:
            DeltaX = X_E_index[ib][ik][ikp + 1][0] - X_E_index[ib][ik][ikp][0]
            same_X_ks = [self.kgrid[tp]['cartesian kpoints'][ib_prm][ik_prm]]
            same_X_ks_integrands = [current_integrand]
            loop_found = False
            while DeltaX < 0.01 and ikp < len(X_E_index[ib][ik]) - 2:
                ikp += 1
                loop_found = True
                X, ib_prm, ik_prm = X_E_index[ib][ik][ikp]
                same_X_ks.append(
                    self.kgrid[tp]['cartesian kpoints'][ib_prm][ik_prm])
                same_X_ks_integrands.append(
                    integrand(tp, c, T, ib, ik, ib_prm, ik_prm, X, sname=sname,
                              g_suffix=g_suffix))
                DeltaX = X_E_index[ib][ik][ikp + 1][0] - X_E_index[ib][ik][ikp][
                    0]

            if len(same_X_ks) > 1:
                m = np.sum(same_X_ks, axis=0) / len(same_X_ks)
                same_X_ks = np.array(same_X_ks) - m
                same_X_ks_sort, ks_indexes = sort_angles(same_X_ks)
                same_X_ks_sort = np.vstack((same_X_ks_sort, same_X_ks_sort[0]))
                ks_indexes.append(ks_indexes[0])
                sm = 0.0
                for j in range(len(ks_indexes) - 1):
                    angle = get_angle(same_X_ks_sort[j + 1], same_X_ks_sort[j])
                    sm += (same_X_ks_integrands[ks_indexes[j + 1]] + \
                           same_X_ks_integrands[ks_indexes[j]]) / 2.0 * angle
                dum = sm / (2 * pi) / 2.0
                ikp += 1

            if not loop_found:
                dum = current_integrand / 2.0
                ikp += 1

            X, ib_prm, ik_prm = X_E_index[ib][ik][ikp]
            current_integrand = integrand(tp, c, T, ib, ik, ib_prm, ik_prm, X,
                                          sname=sname, g_suffix=g_suffix)
            if np.sum(current_integrand) == 0.0:
                dum *= 2
            elif np.sum(dum) == 0.0:
                dum = current_integrand
            else:
                dum += current_integrand / 2.0
            summation += dum * DeltaX  # In case of two points with the same X, DeltaX==0 so no duplicates
        return summation

    def el_integrand_X(self, tp, c, T, ib, ik, ib_prm, ik_prm, X, sname=None,
                       g_suffix=""):
        """
        Returns the evaluated (float) expression inside the elastic equations
            to be integrated over angles, dX.

        Args:
            tp (str): "n" or "p" type
            c (float): carrier concentration/doping in cm**-3
            T (float): the temperature
            ib (int): the band index starting from 0 (CBM/VBM)
            ik (int): the k-point index
            ib_prm (int): ib' (band index for k' state)
            ik_prm (int): ik' (k-index for k' state)
            X (float): the angle between k and k'
            sname (str): elastic scattering name: 'ACD', 'PIE', 'IMP'
            g_suffix (str): '' or '_th' (th for thermal)

        Returns (float): the integrand for elastic scattering integration
        """
        k = self.kgrid[tp]["cartesian kpoints"][ib][ik]
        k_prm = self.kgrid[tp]["cartesian kpoints"][ib_prm][ik_prm]
        if k[0] == k_prm[0] and k[1] == k_prm[1] and k[2] == k_prm[2]:
            return np.array(
                [0.0, 0.0,
                 0.0])  # self-scattering is not defined;regardless, the returned integrand must be a vector

        return (1 - X) * self.kgrid[tp]["norm(k)"][ib_prm][
            ik_prm] ** 2 * self.s_el_eq(sname, tp, c, T, k, k_prm) \
               * self.G(tp, ib, ik, ib_prm, ik_prm, X) / (
                       self.kgrid[tp]["norm(v)"][ib_prm][ik_prm] / sq3)

    def inel_integrand_X(self, tp, c, T, ib, ik, ib_prm, ik_prm, X, sname=None,
                         g_suffix=""):
        """
        Returns the evaluated (float) expression of the S_o & S_i(g) integrals.

        Args:
            tp (str): "n" or "p" type
            c (float): carrier concentration/doping in cm**-3
            T (float): the temperature
            ib (int): the band index starting from 0 (CBM/VBM)
            ik (int): the k-point index
            ib_prm (int): ib' (band index for k' state)
            ik_prm (int): ik' (k-index for k' state)
            X (float): the angle between k and k'
            sname (str): scattering name: 'S_oX_Eplus_ik', 'S_oX_Eminus_ik',
                'S_iX_Eplus_ik' or 'S_iX_Eminus_ik'
            g_suffix (str): '' or '_th' (th for thermal)

        Returns (float): the integrand for POP scattering (to be integrated
            over X)
        """
        if tp == "n" and 'minus' in sname and self.kgrid[tp]["energy"][ib][
            ik] - hbar * self.kgrid[tp]["W_POP"][ib][ik] < self.cbm_vbm[tp][
            "energy"]:
            return 0.0
        if tp == "p" and 'plus' in sname and self.kgrid[tp]["energy"][ib][
            ik] + hbar * self.kgrid[tp]["W_POP"][ib][ik] > self.cbm_vbm[tp][
            "energy"]:
            return 0.0
        if abs(self.kgrid[tp]['energy'][ib_prm][ik_prm] - \
               self.kgrid[tp]['energy'][ib][ik]) < \
                hbar * self.kgrid[tp]["W_POP"][ib][ik] / 2:
            return 0.0

        k = self.kgrid[tp]["cartesian kpoints"][ib][ik]
        k_prm = self.kgrid[tp]["cartesian kpoints"][ib_prm][ik_prm]
        if tp == "n":
            f = self.kgrid[tp]["f"][c][T][ib][ik]
            f_prm = self.kgrid[tp]["f"][c][T][ib_prm][ik_prm]
        else:
            f = 1 - self.kgrid[tp]["f"][c][T][ib][ik]
            f_prm = 1 - self.kgrid[tp]["f"][c][T][ib_prm][ik_prm]

        if k[0] == k_prm[0] and k[1] == k_prm[1] and k[2] == k_prm[2]:
            return 0.0
        N_POP = self.kgrid[tp]["N_POP"][c][T][ib][ik]
        norm_diff = norm(k - k_prm)
        if norm_diff < 1e-4:
            return 0.0

        integ = self.kgrid[tp]["norm(k)"][ib_prm][ik_prm] ** 2 * self.G(tp, ib,
                                                                        ik,
                                                                        ib_prm,
                                                                        ik_prm,
                                                                        X) / \
                (self.kgrid[tp]["norm(v)"][ib_prm][
                     ik_prm] * norm_diff ** 2 / sq3)

        if "S_i" in sname:
            integ *= X * self.kgrid[tp]["g" + g_suffix][c][T][ib_prm][ik_prm]
            if "minus" in sname:
                integ *= (1 - f) * N_POP + f * (1 + N_POP)
            elif "plus" in sname:
                integ *= (1 - f) * (1 + N_POP) + f * N_POP
            else:
                raise ValueError(
                    '"plus" or "minus" must be in sname for phonon absorption and emission respectively')
        elif "S_o" in sname:
            if "minus" in sname:
                integ *= (1 - f_prm) * (
                        1 + N_POP) + f_prm * N_POP  # interestingly f or f_prm does NOT make any difference (maybe close energies?)
            elif "plus" in sname:
                integ *= (1 - f_prm) * N_POP + f_prm * (1 + N_POP)
            else:
                raise ValueError(
                    '"plus" or "minus" must be in sname for phonon absorption and emission respectively')
        else:
            raise ValueError(
                'Unsupported inelastic scattering name: {}'.format(sname))
        return integ

    def s_inel_eq_isotropic(self, once_called=False):
        """
        calclates the inelastic S_i and S_o scattering rates in the kgrid based
            on the isotropic formulation (integrated equations from Rode)

        Args:
            once_called (bool): since scattering out, S_o, needs to be
                calculated only once (not a function of g), we use this flag

        Returns:
            updates values of S_i and S_o (np.array at each k-point) in kgrid
        """
        for tp in ["n", "p"]:
            for c in self.dopings:
                for T in self.temperatures:
                    for ib in range(len(self.kgrid[tp]["kpoints"])):
                        results = [calculate_sio(tp, c, T, ib, ik,
                                                 once_called, self.kgrid,
                                                 self.cbm_vbm,
                                                 self.epsilon_s,
                                                 self.epsilon_inf) for ik in
                                   range(len(self.kgrid[tp]["kpoints"][ib]))]
                        rlts = np.mean(np.array(results), axis=0)
                        if len(list(rlts)) != 4:
                            print(rlts)
                            raise ValueError

                        for ik, res in enumerate(results):
                            self.kgrid[tp]["S_i"][c][T][ib][ik] = res[0]
                            self.kgrid[tp]["S_i_th"][c][T][ib][ik] = res[1]
                            if not once_called:
                                if (res[2] < 0.1).any():
                                    # for the even of ill-defined scattering to avoid POP mobility blowing up
                                    self.kgrid[tp]["S_o"][c][T][ib][ik] = rlts[
                                        2]
                                else:
                                    self.kgrid[tp]["S_o"][c][T][ib][ik] = res[2]

                                if (res[3] < 0.1).any():
                                    self.kgrid[tp]["S_o_th"][c][T][ib][ik] = \
                                        rlts[3]
                                else:
                                    self.kgrid[tp]["S_o_th"][c][T][ib][ik] = \
                                        res[3]

    def s_inelastic(self, sname=None, g_suffix=""):
        """
        For the anisotropic formulation, sets the inelastic scattering rates,
        S_o and S_i, in the kgrid.

        Args:
            sname (str): scattering name; current options are "S_o" and "S_i"
            g_suffix (str): this suffix determines which phenomena is the
                perturbation to the electronic distribution (g) come from.
                Current options are "" (overall, electric) and "_th" (thermal)

        Returns (None):
        """
        for tp in ["n", "p"]:
            for c in self.dopings:
                for T in self.temperatures:
                    for ib in range(len(self.kgrid[tp]["energy"])):
                        for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                            summation = np.array([0.0, 0.0, 0.0])
                            for X_E_index_name in ["X_Eplus_ik", "X_Eminus_ik"]:
                                summation += self.integrate_over_X(tp,
                                                                   self.kgrid[
                                                                       tp][
                                                                       X_E_index_name],
                                                                   self.inel_integrand_X,
                                                                   ib=ib, ik=ik,
                                                                   c=c, T=T,
                                                                   sname=sname + X_E_index_name,
                                                                   g_suffix=g_suffix)
                            self.kgrid[tp][sname][c][T][ib][
                                ik] = summation * e ** 2 * \
                                      self.kgrid[tp]["W_POP"][ib][ik] / (
                                              4 * pi * hbar) * (
                                              1 / self.epsilon_inf - 1 / self.epsilon_s) / epsilon_0 * 100 / e

    def s_elastic(self, sname):
        """
        The scattering rate equation for each elastic scattering name (sname)

        Args:
            sname (str): elastic scattering name: 'IMP', 'ADE', 'PIE', 'DIS'

        Returns:
            it directly calculates the scattering rate at each k-point at each
                c and T (self.kgrid[tp][sname][c][T][ib][ik])
        """
        sname = sname.upper()
        self.initialize_var("egrid", sname, "vector", 0.0, c_T_idx=True)
        self.initialize_var("kgrid", sname, "vector", 0.0, c_T_idx=True)

        for tp in ["n", "p"]:
            for c in self.dopings:
                for T in self.temperatures:
                    valley = Valley(
                        self.kgrid[tp]["cartesian kpoints"][0],
                        self.kgrid[tp]['norm(k)'][0],
                        self.kgrid[tp]['velocity'][0],
                        self.kgrid[tp]['norm(v)'][0],
                        self.kgrid[tp]['a'][0],
                        self.kgrid[tp]['c'][0],
                        angle_k_prime_mapping=self.kgrid[tp]["X_E_ik"][0]
                    )

                    if sname == 'IMP':
                        scats = IonizedImpurityScattering(
                            self.bs_is_isotropic, valley, self.epsilon_s,
                            self.egrid["N_II"][c][T],
                            self.egrid["beta"][c][T][tp])
                    elif sname == 'ACD':
                        scats = AcousticDeformationScattering(
                            self.bs_is_isotropic, valley, self.C_el,
                            self.E_D[tp], T)
                    elif sname == 'PIE':
                        scats = PiezoelectricScattering(
                            self.bs_is_isotropic, valley, self.epsilon_s,
                            self.P_PIE, T)
                    elif sname == 'DIS':
                        scats = DislocationScattering(
                            self.bs_is_isotropic, valley, self.epsilon_s,
                            self.egrid["beta"][c][T][tp], self.N_dis,
                            self.structure.lattice.c)
                    else:
                        raise ValueError("Unknown scattering type: {}".format(
                            sname))

                    rates = scats.calculate_scattering()
                    self.kgrid[tp][sname][c][T][0] = rates
                    self.kgrid[tp]["_all_elastic"][c][T][0] += rates

    def map_to_egrid(self, prop_name, c_and_T_idx=True, prop_type="vector"):
        """
        Maps a propery from kgrid to egrid conserving the nomenclature.
        The mapped property w/ format: kgrid[tp][prop_name][c][T][ib][ik]
        will have the format: egrid[tp][prop_name][c][T][ie]

        Args:
            prop_name (string): the name of the property to be mapped. It must
                be available in the kgrid.
            c_and_T_idx (bool): if True, the propetry will be calculated and
                mapped at each concentration, c, and T
            prop_type (str): options are "scalar", "vector", "tensor"

        Returns (None): sets egrid[tp][prop_name][c][T] as floats or np.ndarray
        """
        if not c_and_T_idx:
            self.initialize_var("egrid", prop_name, prop_type, initval=self.gs,
                                is_nparray=True, c_T_idx=False)
            for tp in ["n", "p"]:
                for ie, en in enumerate(self.egrid[tp]["energy"]):
                    for ib, ik in self.kgrid_to_egrid_idx[tp][ie]:
                        self.egrid[tp][prop_name][ie] += \
                            self.kgrid[tp][prop_name][ib][ik]
                    self.egrid[tp][prop_name][ie] /= \
                        len(self.kgrid_to_egrid_idx[tp][ie])
        else:
            self.initialize_var("egrid", prop_name, prop_type, initval=self.gs,
                                is_nparray=True, c_T_idx=True)
            for tp in ["n", "p"]:
                for c in self.dopings:
                    for T in self.temperatures:
                        for ie, en in enumerate(self.egrid[tp]["energy"]):
                            for ib, ik in self.kgrid_to_egrid_idx[tp][ie]:
                                self.egrid[tp][prop_name][c][T][ie] += \
                                    self.kgrid[tp][prop_name][c][T][ib][ik]
                            self.egrid[tp][prop_name][c][T][ie] /= \
                                len(self.kgrid_to_egrid_idx[tp][ie])

    def find_fermi(self, c, T, rtol=0.01, rtol_loose=0.03, step=0.1, nstep=50):
        """
        Finds the Fermi level at a given c and T at egrid (i.e. DOS)

        Args:
            c (float): The doping concentration;
                c < 0 indicate n-type (i.e. electrons) and c > 0 for p-type
            T (float): The temperature.
            rtol (0<float<1): convergance threshold for relative error
            rtol_loose (0<float<1): maximum relative error allowed between the
                calculated and input c
            step (float): initial step size in eV. (-step*nstep,step*nstep) is
                the initial energy range around the fermi level initial guess
            nstep (int): number of steps to check before and after a given
                fermi level

        Returns (float in eV):
            The fitted/calculated Fermi level
        """
        relative_error = self.gl
        typ = get_tp(c)
        fermi = self.cbm_vbm[typ]["energy"] + 0.01  # initialize fermi non-zero

        conversion = 1.0 / (self.structure.volume * (A_to_m * m_to_cm) ** 3)

        dos_e = np.array([d[0] for d in self.dos])
        dos_de = np.array([self.dos[i + 1][0] - self.dos[i][0] \
                           for i, _ in enumerate(self.dos[:-1])] + [0.0])

        # left is faster, trapezoidal makes no difference
        dos_dos = self.dos[:, 1]

        # fix energy, energy diff. and dos for integration at all fermi levels
        es = np.repeat(dos_e.reshape((len(dos_e), 1)), 2 * nstep + 1, axis=1)
        de = np.repeat(dos_de.reshape((len(dos_de), 1)), 2 * nstep + 1, axis=1)
        tdos = np.repeat(dos_dos.reshape((len(dos_dos), 1)), 2 * nstep + 1,
                         axis=1)

        print(self.cbm_vbm)
        self.vbm_dos_idx = self.get_Eidx_in_dos(self.cbm_vbm["p"]["energy"])
        self.cbm_dos_idx = self.get_Eidx_in_dos(self.cbm_vbm["n"]["energy"])
        dumpfn(self.dos, "new_dos.json.gz")

        ## debug plot to make sure CBM/VBM recognized correctly in DOS
        PLOT_CALCULATED_DOS = True
        if PLOT_CALCULATED_DOS:
            ## if not commented, make sure run_type='BOLTZ' in BoltztrapRunner
            self.logger.debug('PLOT_CALCULATED_DOS is set to True, if '
                              'interpolation=boltztrap1, make sure to set '
                              'run_type="BOLTZ" when running BoltztrapRunner')
            start_idx = self.get_Eidx_in_dos(self.cbm_vbm["p"]["energy"] - 1.)
            end_idx = self.get_Eidx_in_dos(self.cbm_vbm["n"]["energy"] + 1.)
            plt.plot(dos_e[start_idx: end_idx], dos_dos[start_idx: end_idx])
            plt.scatter(dos_e[self.cbm_dos_idx], 0, s=100, marker='s')
            plt.scatter(dos_e[self.vbm_dos_idx], 0, s=100, marker='s')
            plt.show()
            # plt.savefig(os.path.join(self.calc_dir, 'test_bsdos.png'))

        self.logger.debug("Calculating the fermi level at T={} K".format(T))
        midgap_idx = int((self.vbm_dos_idx + self.cbm_dos_idx) / 2)
        for i in range(20):
            fermi_range = np.linspace(fermi - nstep * step,
                                      fermi + nstep * step, 2 * nstep + 1)
            n_dopings = -conversion \
                        * np.sum(tdos[midgap_idx:] \
                                 * f0(es[midgap_idx:], fermi_range, T) \
                                 * de[midgap_idx:], axis=0)
            p_dopings = conversion \
                        * np.sum(tdos[:midgap_idx + 1] \
                                 * (1 - f0(es[:midgap_idx + 1], fermi_range, T)) \
                                 * de[:midgap_idx + 1], axis=0)
            relative_error = abs((n_dopings + p_dopings) / c - 1.0)
            fermi_idx = np.argmin(relative_error)
            fermi = fermi_range[fermi_idx]
            self.calc_doping[c][T]['n'] = n_dopings[fermi_idx]
            self.calc_doping[c][T]['p'] = p_dopings[fermi_idx]
            for tp in ['p', 'n']:
                if np.isnan(self.calc_doping[c][T][tp]):
                    self.calc_doping[c][T][tp] = 10
            if relative_error[fermi_idx] < rtol:
                self.logger.info(
                    "fermi at {0:.2e} 1/cm3 and {1} K: {2:.6f}".format(c, T,
                                                                       fermi))
                return fermi
            step /= 10.0

        if relative_error[fermi_idx] > rtol_loose:
            raise AmsetError(self.logger,
                             'The calculated concentration is not within {0:.2f}% of'
                             ' the given value ({1:.2e}) at T={2}'.format(
                                 rtol_loose * 100, c, T))
        elif relative_error[fermi_idx] > rtol:
            self.logger.warning(
                'Fermi calculated with a loose tolerance of {0:.2f}%'
                ' at c={1:.2e}, T={2}K'.format(rtol_loose, c, T))
        return fermi

    def inverse_screening_length(self, c, T):
        """
        Calculates the inverse charge screening length (beta) based on Eq. 70
        of reference [R]. Beta is used in calculation of ionized impurity (IMP)
        scattering.

        Args:
            c (float): the carrier concentration (to get the fermi level)
            T (float): the temperature

        Returns (float): the inverse screening length (beta) in 1/nm units
        """
        beta = {}
        for tp in ["n", "p"]:
            integral = self.integrate_over_E(props=["f0", "1 - f0"],
                                             tp=tp, c=c, T=T, xDOS=True)
            beta[tp] = (e ** 2 / (self.epsilon_s * epsilon_0 * k_B * T) \
                        * integral / self.structure.volume * 1e12 / e) ** 0.5
        return beta

    def solve_BTE_iteratively(self):
        """
        Iteratively solve linearized/low-field Boltzmann Transport Equation
        See equation (43) page 20 of the reference [R]

        Returns (None): the results are stored in "g*" keys in kgrid and egrid
        """
        if "POP" in self.inelastic_scats and not self.bs_is_isotropic:
            # calculating S_o scattering rate which is NOT a function of g
            for g_suffix in ["", "_th"]:
                self.s_inelastic(sname="S_o" + g_suffix, g_suffix=g_suffix)

        # solve BTE to calculate S_i scattering rate and perturbation (g) in an iterative manner
        for iter in range(self.BTE_iters):
            self._check_timeout_hours()
            self.logger.info("Performing iteration # {}".format(iter))
            if "POP" in self.inelastic_scats:
                if self.bs_is_isotropic:
                    if iter == 0:
                        self.s_inel_eq_isotropic(once_called=False)
                    else:
                        self.s_inel_eq_isotropic(once_called=True)
                else:
                    for g_suffix in ["", "_th"]:
                        self.s_inelastic(sname="S_i" + g_suffix,
                                         g_suffix=g_suffix)
            for c in self.dopings:
                for T in self.temperatures:
                    for tp in ["n", "p"]:
                        g_old = np.array(self.kgrid[tp]["g"][c][T][0])
                        for ib in range(self.cbm_vbm[tp]["included"]):
                            self.kgrid[tp]["g_POP"][c][T][ib] = (self.kgrid[tp][
                                                                     "S_i"][c][
                                                                     T][ib] +
                                                                 self.kgrid[tp][
                                                                     "electric force"][
                                                                     c][T][
                                                                     ib]) / (
                                                                        self.kgrid[
                                                                            tp][
                                                                            "S_o"][
                                                                            c][
                                                                            T][
                                                                            ib] + self.gs + 1.0)

                            self.kgrid[tp]["g"][c][T][ib] = (self.kgrid[tp][
                                                                 "S_i"][c][T][
                                                                 ib] +
                                                             self.kgrid[tp][
                                                                 "electric force"][
                                                                 c][
                                                                 T][ib]) / (
                                                                    self.kgrid[
                                                                        tp][
                                                                        "S_o"][
                                                                        c][
                                                                        T][
                                                                        ib] +
                                                                    self.kgrid[
                                                                        tp][
                                                                        "_all_elastic"][
                                                                        c][
                                                                        T][
                                                                        ib])

                            self.kgrid[tp]["g_th"][c][T][ib] = (self.kgrid[tp][
                                                                    "S_i_th"][
                                                                    c][T][ib] +
                                                                self.kgrid[tp][
                                                                    "thermal force"][
                                                                    c][T][
                                                                    ib]) / (
                                                                       self.kgrid[
                                                                           tp][
                                                                           "S_o_th"][
                                                                           c][
                                                                           T][
                                                                           ib] +
                                                                       self.kgrid[
                                                                           tp][
                                                                           "_all_elastic"][
                                                                           c][
                                                                           T][
                                                                           ib])

                            self.kgrid[tp]["f"][c][T][ib] = \
                                self.kgrid[tp]["f0"][c][T][ib] + \
                                self.kgrid[tp]["g"][c][T][ib]
                            self.kgrid[tp]["f_th"][c][T][ib] = \
                                self.kgrid[tp]["f0"][c][T][ib] + \
                                self.kgrid[tp]["g_th"][c][T][ib]

                        avg_g_diff = np.mean(
                            [abs(g_old[ik] - self.kgrid[tp]["g"][c][T][0][ik])
                             for ik in range(len(g_old))])
                        self.logger.info(
                            "Average difference in {0}-type g term at c={1:.2e} and T={2}: {3}".format(
                                tp, c, T, avg_g_diff))

        for prop in ["electric force", "thermal force", "g", "g_POP", "g_th",
                     "S_i", "S_o", "S_i_th", "S_o_th"]:
            self.map_to_egrid(prop_name=prop, c_and_T_idx=True)

        for tp in ["n", "p"]:
            for c in self.dopings:
                for T in self.temperatures:
                    for ie in range(len(self.egrid[tp]["g_POP"][c][T])):
                        if norm(self.egrid[tp]["g_POP"][c][T][ie]) > 1:
                            self.egrid[tp]["g_POP"][c][T][ie] = [1e-5, 1e-5,
                                                                 1e-5]

    def calculate_transport_properties_with_E(self):
        """
        Mobility and Seebeck coefficient are calculated by integrating the
        perturbation of electron distribution and group velocity over the energy
            """
        valley_transport = {tp: {
            el_mech: {c: {T: np.array([0., 0., 0.]) for T in self.temperatures}
                      for
                      c in
                      self.dopings} for el_mech in self.transport_labels} for tp
            in
            ["n", "p"]}

        for c in self.dopings:
            for T in self.temperatures:
                for j, tp in enumerate(["p", "n"]):
                    # mobility numerators
                    for mu_el in self.elastic_scats:
                        valley_transport[tp][mu_el][c][T] = (
                                                                -1) * default_small_E / hbar * \
                                                            self.integrate_over_E(
                                                                props=[
                                                                    "/" + mu_el,
                                                                    "df0dk"],
                                                                tp=tp, c=c, T=T,
                                                                xDOS=False,
                                                                xvel=True)

                    for mu_inel in self.inelastic_scats:
                        valley_transport[tp][mu_inel][c][
                            T] = self.integrate_over_E(props=[
                            "g_" + mu_inel], tp=tp, c=c, T=T, xDOS=False,
                            xvel=True)
                        mu_overall_valley = self.integrate_over_E(props=["g"],
                                                                  tp=tp, c=c,
                                                                  T=T,
                                                                  xDOS=False,
                                                                  xvel=True)

                    # TODO: make sure that units of J_th is correct and at the end (after we divide by the denominator), we arrive at A/cm2
                    valley_transport[tp]["J_th"][c][T] = self.integrate_over_E(
                        props=["g_th"],
                        tp=tp, c=c, T=T,
                        xDOS=False, xvel=True)

                    faulty_overall_mobility = False
                    temp_avg = np.array([0.0, 0.0, 0.0])
                    for transport in self.elastic_scats + self.inelastic_scats:
                        temp_avg += 1 / valley_transport[tp][transport][c][T]
                        if norm(mu_overall_valley) > norm(
                                valley_transport[tp][transport][c][T]):
                            faulty_overall_mobility = True  # because the overall mobility should be lower than all
                    valley_transport[tp]['average'][c][T] = 1 / temp_avg

                    if norm(
                            mu_overall_valley) == 0.0 or faulty_overall_mobility:
                        valley_transport[tp]['overall'][c][T] = \
                            valley_transport[tp]['average'][c][T]
                    else:
                        valley_transport[tp]["overall"][c][
                            T] = mu_overall_valley
                    self.egrid[tp]["relaxation time constant"][c][T] = \
                        self.mobility[tp]["overall"][c][T] \
                        * 1e-4 * m_e * self.cbm_vbm[tp][
                            "eff_mass_xx"] / e  # 1e-4 to convert cm2/V.s to m2/V.s
                    valley_transport[tp]["seebeck"][c][T] = \
                        self.egrid["Seebeck_integral_numerator"][c][T][tp]
        return valley_transport

    def as_dict(self):
        """
        Mobility, input parameters, etc as a serializable python dictionary.

        Returns (dict):
        """
        out_d = super(Amset, self).as_dict()
        out_d['logger'] = True  # cannot be converted to JSON

        # TODO: Fix this so that the non-private attributes are not overwritten
        out_d['performance_params'] = self._performance_params
        out_d['model_params'] = self._model_params
        out_d['material_params'] = self._material_params

        for arg in _inst_args:
            if hasattr(self, arg):
                out_d[arg] = getattr(self, arg)

        return out_d

    @classmethod
    def from_dict(cls, d):
        decoded = {k: MontyDecoder().process_decoded(v) for k, v in d.items()
                   if not k.startswith("@")}
        amset = cls(**{k: v for k, v in decoded.items() if k not in _inst_args})

        for k in _inst_args:
            if k in decoded:
                setattr(amset, k, decoded[k])
        return amset

    def to_file(self, filename='amset.json.gz', directory='run_data',
                overwrite=True):
        """Serialize the AMSET run to a JSON file.

        Can be loaded by the ``Amset.from_file()`` method for post-processing
        (e.g. plotting).

        Args:
            filename (str): The JSON file name.
            directory (str): The directory in which to save the file.
                Folders will be created if they do not already exist.
                The directory should be relative to Amset.calc_dir.
            overwrite (bool): Whether to overwrite the JSON file if it exists.
                If set to False, a new file with the suffix '_n' is created,
                where n is 1, 2, 3, etc.

        Returns:
            (str): The full path to saved file.
        """
        directory = path_join(self.calc_dir, directory)

        if not os.path.exists(directory):
            os.makedirs(directory)

        filepath = path_join(directory, filename)
        if not overwrite and os.path.exists(filepath):
            self.logger.info("{} already exists, refusing to overwrite".format(
                filepath))
            pre = filename.split('.')[0]  # filename with no extension
            suf = ".".join(filename.split('.')[1:])  # file extension
            # gets the first filename that doesn't exist from an generator
            file_gen = (path_join(directory, '{}_{}.{}'.format(pre, i, suf))
                        for i in count())
            filepath = next(f for f in file_gen if not os.path.exists(f))

        self.logger.info("Writing AMSET data to {}".format(filepath))
        dumpfn(self, filepath)
        return filepath

    @staticmethod
    def from_file(filepath):
        """Load an AMSET run from a JSON file.

        The AMSET json can file written using the ``Amset.to_file()``
        method.

        Args:
            filepath (str): Full path to the AMSET json file.

        Returns:
            (Amset): An Amset object.
        """
        return loadfn(filepath)

    def grids_to_json(self, kgrid=True, trimmed=False, max_ndata=None, n0=0,
                      valleys=True, path=None, dir_name="run_data"):
        """
        Writes the kgrid and egird to json files

        Args:
            kgrid (bool): whether to also write kgrid to kgrid.json
            trimmed (bool): if trimmed some properties (dict keys) will be
                removed to save space
            max_ndata (int): the maximum index from the CBM/VBM written to file
            n0 (int): the initial list index of a property written to file

        Returns: egrid.json and (optional) kgrid.json file(s)
        """
        path = os.path.join(path or self.calc_dir, dir_name)
        if not os.path.exists(path):
            os.makedirs(name=path)

        if not max_ndata:
            max_ndata = int(self.gl)
        egrid = deepcopy(self.egrid)
        if trimmed:
            nmax = int(min([max_ndata + 1, min([len(egrid["n"]["energy"]),
                                                len(egrid["p"]["energy"])])]))
            for tp in ["n", "p"]:
                for key in egrid[tp]:
                    if key in ['size', 'J_th', 'relaxation time constant',
                               'conductivity', 'seebeck', 'TE_power_factor']:
                        continue
                    try:
                        for c in self.dopings:
                            for T in self.temperatures:
                                if tp == "n":
                                    egrid[tp][key][c][T] = \
                                        self.egrid[tp][key][c][T][n0:n0 + nmax]
                                else:
                                    egrid[tp][key][c][T] = \
                                        self.egrid[tp][key][c][T][::-1][
                                        n0:n0 + nmax]
                    except:
                        try:
                            if tp == "n":
                                egrid[tp][key] = self.egrid[tp][key][
                                                 n0:n0 + nmax]
                            else:
                                egrid[tp][key] = self.egrid[tp][key][::-1][
                                                 n0:n0 + nmax]
                        except:
                            if key not in ['mobility']:
                                self.logger.warning(
                                    'in grids_to_json: cutting {} '
                                    'in egrid failed!'.format(
                                        key))

        with open(os.path.join(path, "egrid.json"), 'w') as fp:
            json.dump(egrid, fp, sort_keys=True, indent=4, ensure_ascii=False,
                      cls=MontyEncoder)

        # self.kgrid trimming
        if kgrid:
            kgrid = deepcopy(self.kgrid)
            if trimmed:
                nmax = min([max_ndata + 1, min([len(kgrid["n"]["kpoints"][0]),
                                                len(kgrid["p"]["kpoints"][
                                                        0])])])
                for tp in ["n", "p"]:
                    for key in kgrid[tp]:
                        if key in ["size"]:
                            continue
                        try:
                            for c in self.dopings:
                                for T in self.temperatures:
                                    if tp == "n":
                                        kgrid[tp][key][c][T] = [
                                            self.kgrid[tp][key][c][T][b][
                                            n0:n0 + nmax]
                                            for b in range(
                                                self.cbm_vbm[tp]["included"])]
                                    else:
                                        kgrid[tp][key][c][T] = [
                                            self.kgrid[tp][key][c][T][b][::-1][
                                            n0:n0 + nmax]
                                            for b in range(
                                                self.cbm_vbm[tp]["included"])]
                        except:
                            try:
                                if tp == "n":
                                    kgrid[tp][key] = [
                                        self.kgrid[tp][key][b][n0:n0 + nmax]
                                        for b in
                                        range(self.cbm_vbm[tp]["included"])]
                                else:
                                    kgrid[tp][key] = [
                                        self.kgrid[tp][key][b][::-1][
                                        n0:n0 + nmax]
                                        for b in
                                        range(self.cbm_vbm[tp]["included"])]
                            except:
                                if key not in ['mobility']:
                                    self.logger.warning(
                                        'in grids_to_json: cutting {} '
                                        'in kgrid failed!'.format(key))

            with open(os.path.join(path, "kgrid.json"), 'w') as fp:
                json.dump(kgrid, fp, sort_keys=True, indent=4,
                          ensure_ascii=False, cls=MontyEncoder)
        if valleys:
            with open(os.path.join(path, "valleys.json"), 'w') as fp:
                json.dump(self.valleys, fp,
                          sort_keys=True, indent=4,
                          ensure_ascii=False, cls=MontyEncoder)

    def to_csv(self, path=None, dir_name="run_data",
               csv_filename='amset_results.csv'):
        """
        Writes the calculated transport properties to a csv file. These
        properties are overall mobility and those values calculated assuming
        only one scattering mechanism is limiting the mobility. Seebeck
        coefficient is also reported all at any given combination of carrier
        concentration, c, or temperature, T.

        Args:
            path (str): full path to the folder where the .csv file is saved
            dir_name (str): folder where .csv is saved; ignored if path is set
            csv_filename (str): the name of the .csv file

        Returns (None):
        """
        import csv
        path = os.path.join(path or self.calc_dir, dir_name)
        if not os.path.exists(path):
            os.makedirs(name=path)
        with open(os.path.join(path, csv_filename), 'w') as csvfile:
            fieldnames = ['type', 'c(cm-3)', 'T(K)', 'overall', 'average'] + \
                         self.elastic_scats + self.inelastic_scats + [
                             'Seebeck (uV/K)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for c in self.dopings:
                tp = get_tp(c)
                for T in self.temperatures:
                    row = {'type': tp, 'c(cm-3)': abs(c), 'T(K)': T}
                    for p in ['overall',
                              'average'] + self.elastic_scats + self.inelastic_scats:
                        row[p] = sum(self.mobility[tp][p][c][T]) / 3
                    row['Seebeck (uV/K)'] = sum(self.seebeck[tp][c][T]) / 3
                    writer.writerow(row)

    def plot(self, **kwargs):
        """
        Plots the given k_plots and e_plots properties.

        See docstring for amset.plotting.get_amset_plots for kewyword arguments.

        """
        get_amset_plots(amset=self, **kwargs)

    def find_fermi_k(self, tolerance=0.001, num_bands=None):
        """
        **method used only by "k"-integration method.**

        Args:
            tolerance:
            num_bands:

        Returns:

        """
        num_bands = num_bands or self.num_bands
        closest_energy = {c: {T: None for T in self.temperatures} for c in
                          self.dopings}
        self.f0_array = {c: {T: {tp: list(range(num_bands[tp])) \
                                 for tp in ['n', 'p']} \
                             for T in self.temperatures} \
                         for c in self.dopings}
        for c in self.dopings:
            tp = get_tp(c)
            tol = tolerance * abs(c)
            for T in self.temperatures:
                step = 0.1
                range_of_energies = np.arange(self.cbm_vbm[tp]['energy'] - 2,
                                              self.cbm_vbm[tp]['energy'] + 2.1,
                                              step)
                diff = 1000.0 * abs(c)
                while (diff > tol):
                    # try a number for fermi level
                    diffs = {}
                    for e_f in range_of_energies:
                        # calculate distribution in both conduction and valence bands
                        f_con = 1 / (np.exp(
                            (self.energy_array['n'] - e_f) / (k_B * T)) + 1)
                        f_val = 1 / (np.exp(
                            (self.energy_array['p'] - e_f) / (k_B * T)) + 1)
                        # density of states in k space is V/8pi^3 per spin, but total states per real volume per k volume is 2/8pi^3
                        dens_of_states = 1 / (4 * np.pi ** 3)
                        # see if it is close to concentration
                        n_concentration = \
                            self.integrate_over_states(f_con * dens_of_states,
                                                       'n')[
                                0]
                        p_concentration = \
                            self.integrate_over_states(
                                (1 - f_val) * dens_of_states,
                                'p')[0]
                        diffs[e_f] = abs(
                            (p_concentration - n_concentration) - c)
                    # compare all the numbers and zoom in on the closest
                    closest_energy[c][T] = min(diffs, key=diffs.get)
                    range_of_energies = np.arange(closest_energy[c][T] - step,
                                                  closest_energy[c][T] + step,
                                                  step / 10)
                    step /= 10
                    diff = diffs[closest_energy[c][T]]
                # find the calculated concentrations (dopings) of each type at the determined fermi level
                e_f = closest_energy[c][T]
                for j, tp in enumerate(['n', 'p']):
                    for ib in list(range(num_bands[tp])):
                        self.f0_array[c][T][tp][ib] = 1 / (np.exp(
                            (self.energy_array[tp][ib][:, :, :, 0] - e_f) / (
                                    k_B * T)) + 1)
                    self.calc_doping[c][T][tp] = self.integrate_over_states(
                        j - np.array(self.f0_array[c][T][tp]), tp)
        return closest_energy

    def get_scalar_output(self, vec, dir):
        """
        **method used only by "k"-integration method.**

        As the name suggests, it returns a scalar off a vector for plotting.

        Args:
            vec (3x1 array or list): the input vector.
            dir (str): the direction; options are "x", "y", "z" and "avg"

        Returns (float):
        """
        if dir == 'x':
            return vec[0]
        if dir == 'y':
            return vec[1]
        if dir == 'z':
            return vec[2]
        if dir == 'avg':
            return sum(vec) / 3

    def calc_v_vec(self, tp):
        """
        **method used only by "k"-integration method.**

        Args:
            tp:

        Returns:

        """
        v_vec_all_bands = []
        v_norm_all_bands = []
        for ib in range(self.num_bands[tp]):
            v_norm_k_ordered = (v_vec_k_ordered[:, 0] ** 2 + v_vec_k_ordered[:,
                                                             1] ** 2 + v_vec_k_ordered[
                                                                       :,
                                                                       2] ** 2) ** 0.5
            v_vec_all_bands.append(
                self.grid_from_ordered_list(v_vec_k_ordered, tp,
                                            none_missing=True))
            v_norm_all_bands.append(
                self.grid_from_ordered_list(v_norm_k_ordered, tp,
                                            none_missing=True, scalar=True))
        return np.array(v_vec_all_bands), np.array(v_norm_all_bands)

    def array_from_kgrid(self, prop_name, tp, c=None, T=None, denom=False,
                         none_missing=False, fill=None):
        """
        **method used only by "k"-integration method.**

        turns a kgrid property into a list of grid arrays of that property for k integration

        Args:
            prop_name:
            tp:
            c:
            T:
            denom:
            none_missing:
            fill:

        Returns:

        """
        if c:
            return np.array([self.grid_from_energy_list(
                self.kgrid[tp][prop_name][c][T][ib], tp, ib, denom=denom,
                none_missing=none_missing, fill=fill) for ib in
                range(self.num_bands[tp])])
        else:
            return np.array([self.grid_from_energy_list(
                self.kgrid[tp][prop_name][ib], tp, ib, denom=denom,
                none_missing=none_missing, fill=fill) for ib in
                range(self.num_bands[tp])])

    def grid_from_energy_list(self, props, tp, ib, denom=False,
                              none_missing=False, fill=None):
        """
        **method used only by "k"-integration method.**

        Args:
            props: a list that is sorted by energy and missing removed points
            tp:
            ib:
            denom:
            none_missing:
            fill:

        Returns:

        """
        if not fill:
            if not denom:
                fill = 0
            if denom:
                fill = 1
        adjusted_props = list(props)
        # step 0 is reverse second sort
        adjusted_props = np.array(adjusted_props)[self.pos_idx_2[tp][ib]]
        adjusted_props = [adjusted_props[i] for i in
                          range(adjusted_props.shape[0])]

        # reverse what has been done: step 1 is add new points back
        if not none_missing:
            insert_list = False
            if type(adjusted_props[0]) == np.ndarray or type(
                    adjusted_props[0]) == list:
                if len(adjusted_props[0]) == 3:
                    insert_list = True
            for ik in self.rm_idx_list[tp][ib]:
                adjusted_props.insert(ik,
                                      fill) if not insert_list else adjusted_props.insert(
                    ik, [fill, fill, fill])

        # step 2 is reorder based on first sort
        adjusted_props = np.array(adjusted_props)[self.pos_idx[tp]]
        # then call grid_from_ordered_list
        return self.grid_from_ordered_list(adjusted_props, tp, denom=denom,
                                           none_missing=True)

    def grid_from_ordered_list(self, props, tp, denom=False, none_missing=False,
                               scalar=False):
        """
        **method used only by "k"-integration method.**

        Args:
            props:
            tp:
            denom:
            none_missing:
            scalar:

        Returns:  a grid of the (x,y,z) k points in the proper grid
        """
        N = list(self.kgrid_array[tp].shape)
        if scalar:
            N[-1] = 1
        grid = np.zeros(N)
        adjusted_props = list(props)

        # put zeros back into spots of missing indexes
        # self.rm_idx_list format: [tp][ib][ik]
        if not none_missing:
            for ib in range(self.num_bands[tp]):
                for ik in self.rm_idx_list[tp][ib]:
                    if not denom:
                        adjusted_props.insert(ik, 0)
                    if denom:
                        adjusted_props.insert(ik, 1)

        for i in range(N[0]):
            for j in range(N[1]):
                for k in range(N[2]):
                    grid[i, j, k] = adjusted_props[
                        i * N[1] * N[2] + j * N[2] + k]
        return grid

    def integrate_over_states(self, integrand_grid, tp='all'):
        """
        **method used only by "k"-integration method.**

        Args:
            integrand_grid: list or array of array grids
            tp:

        Returns:

        """
        integrand_grid = np.array(integrand_grid)
        if type(integrand_grid[0][0, 0, 0]) == list or type(
                integrand_grid[0][0, 0, 0]) == np.ndarray:
            result = np.zeros(3)
        else:
            result = 0
        num_bands = integrand_grid.shape[0]
        for ib in range(num_bands):
            result += self.integrate_over_k(integrand_grid[ib], tp)
        return result

    def calculate_transport_properties_with_k(self, test_anisotropic,
                                              important_points):
        """
        **method used only by "k"-integration method.**

        Calculates transport properties for isotropic materials with integration
        over the k-points rather than energy

        Args:
            test_anisotropic:
            important_points:

        Returns:

        """
        # calculate mobility by averaging velocity per electric field strength
        mu_num = {tp: {
            el_mech: {c: {T: [0, 0, 0] for T in self.temperatures} for c in
                      self.dopings} for el_mech in self.elastic_scats} for tp in
            ["n", "p"]}
        valley_transport = {tp: {
            el_mech: {c: {T: np.array([0., 0., 0.]) for T in self.temperatures}
                      for
                      c in
                      self.dopings} for el_mech in self.transport_labels} for tp
            in
            ["n", "p"]}

        for c in self.dopings:
            for T in self.temperatures:
                for j, tp in enumerate(["n", "p"]):
                    E_array = self.array_from_kgrid('energy', tp)
                    if not self.count_mobility[self.ibrun][tp]:
                        continue
                    N = self.kgrid_array[tp].shape

                    # get quantities that are independent of mechanism
                    num_k = [len(self.kgrid[tp]["energy"][ib]) for ib in
                             range(self.num_bands[tp])]
                    df0dk = self.array_from_kgrid('df0dk', tp, c, T)
                    v = self.array_from_kgrid('velocity', tp)
                    f0_removed = self.array_from_kgrid('f0', tp, c, T)
                    f0_all = 1 / (np.exp(
                        (self.energy_array[tp] - self.fermi_level[c][T]) / (
                                k_B * T)) + 1)

                    np.set_printoptions(precision=3)

                    # TODO: the anisotropic case is not correct right now
                    if not self.bs_is_isotropic or test_anisotropic:

                        v_vec, v_norm = self.calc_v_vec(tp)

                        # TODO: get f through solving the BTE anisotropically
                        # k_hat = np.array([self.k_hat_array[tp] for ib in range(self.num_bands[tp])])
                        k_hat_cartesian = np.array(
                            [self.k_hat_array_cartesian[tp] for ib in
                             range(self.num_bands[tp])])
                        g = self.array_from_kgrid("g", tp, c, T)
                        # x is the cosine of the angle between the force and k, or negative the cosine of the angle
                        # between the electric fields and k
                        x = -k_hat_cartesian
                        f_T = f0_all + x * g

                        if tp == 'n':
                            print('v')
                            print(v_vec.shape)
                            print(v_vec[0, (N[0] - 1) / 2, (N[1] - 1) / 2, :])
                            print('k_hat_cartesian')
                            print(k_hat_cartesian[0, (N[0] - 1) / 2,
                                  (N[1] - 1) / 2, :])
                            print('g')
                            print(g[0, (N[0] - 1) / 2, (N[1] - 1) / 2, :])

                            print('v*f0_all')
                            print((v_vec * f0_all)[0, (N[0] - 1) / 2,
                                  (N[1] - 1) / 2, :])
                            print('v*f_T')
                            print(
                                (v_vec * f_T)[0, (N[0] - 1) / 2, (N[1] - 1) / 2,
                                :])
                            print('v*(f_T-f0_all)')
                            print(
                                (v_vec * k_hat_cartesian * g)[0, (N[0] - 1) / 2,
                                (N[1] - 1) / 2, :])
                            print('v*f_T*d3k')
                            print(((v_vec * f_T)[0] * self.dv_grid['n'][:, :, :,
                                                      np.newaxis])[
                                  (N[0] - 1) / 2, (N[1] - 1) / 2, :])
                            print("sum")
                            print(np.sum(((v_vec * f_T)[0] * self.dv_grid['n'][
                                                             :, :, :,
                                                             np.newaxis]),
                                         axis=(0, 1, 2)))

                        # from equation 44 in Rode, overall
                        # nu_el = self.array_from_kgrid('_all_elastic', tp, c, T, denom=True)
                        # numerator = -self.integrate_over_states(v * self.k_hat_array[tp] * (-1 / hbar) * df0dk / nu_el, tp)
                        # denominator = self.integrate_over_states(f0, tp) * hbar * default_small_E
                        numerator = self.integrate_over_states(
                            v_vec / default_small_E * f_T, tp)
                        numerator2 = self.integrate_over_states(
                            v_vec / default_small_E * f0_all, tp)
                        numerator3 = self.integrate_over_states(
                            v_vec / default_small_E * (f_T - f0_all), tp)
                        numerator4 = self.integrate_over_states(
                            v_vec / default_small_E * (x * g), tp)
                        numerator5 = self.integrate_over_states(
                            v_norm * x * x * g, tp) / default_small_E
                        numerator6 = self.integrate_over_states(
                            v_norm * x ** 2 * g, tp) / default_small_E
                        denominator = self.integrate_over_states(
                            j + ((-1) ** j) * f_T, tp)
                        self.mobility[tp]['overall'][c][
                            T] = numerator / denominator

                        if tp == 'n':
                            print(
                                'ANISOTROPIC numerator, numerator without g, and denominator:')
                            print(numerator)
                            print(numerator2)
                            print(numerator3)
                            print(numerator4)
                            print(numerator5)
                            print(numerator6)
                            print(denominator)

                        if tp == 'n':
                            denominator_iso = self.integrate_over_states(f0_all,
                                                                         tp)
                        if tp == 'p':
                            denominator_iso = self.integrate_over_states(
                                1 - f0_all, tp)
                        numerator_iso = self.integrate_over_states(g * v_norm,
                                                                   tp) / 3 / default_small_E

                        if tp == 'n':
                            print('v')
                            print(v[0, (N[0] - 1) / 2, (N[1] - 1) / 2, :])
                            print('g*v')
                            print((g * v)[0, (N[0] - 1) / 2, (N[1] - 1) / 2, :])
                            print('ISOTROPIC numerator and denominator:')
                            print(numerator_iso)
                            print(denominator_iso)

                        k_norm = np.sqrt(
                            self.kgrid_array_cartesian[tp][:, :, :, 0] ** 2 +
                            self.kgrid_array_cartesian[tp][:, :, :, 1] ** 2 +
                            self.kgrid_array_cartesian[tp][:, :, :, 2] ** 2) / (
                                         A_to_m * m_to_cm)
                        print('norm(k)')
                        print(k_norm[(N[0] - 1) / 2, (N[1] - 1) / 2, :])  # 1/cm
                        print(
                            self.structure.lattice.reciprocal_lattice.volume)  # in 1/A^3
                        k_0 = (
                                  self.structure.lattice.reciprocal_lattice.volume) ** (
                                      1. / 3) / (A_to_m * m_to_cm)
                        print('k_0')
                        print(k_0)  # in 1/cm
                        print('test integral of e^(-r) * cos^2(theta)')
                        aa = 10 / (k_0 / 2)  # in cm
                        print(self.integrate_over_k(
                            np.exp(-aa * k_norm) * k_hat_cartesian[0, :, :, :,
                                                   2] ** 2 * aa ** 3, tp))

                    if self.bs_is_isotropic and not test_anisotropic:
                        if tp == get_tp(c):
                            self.logger.info(
                                'calculating mobility by integrating over'
                                ' k-grid and isotropic BS assumption...')
                            self.logger.debug('current valley is at {}'.format(
                                important_points))
                            self.logger.debug('the denominator is:\n{}'.format(
                                self.denominator))

                        for el_mech in self.elastic_scats:
                            nu_el = self.array_from_kgrid(el_mech, tp, c, T,
                                                          denom=True)
                            # this line should have -e / hbar except that hbar is in units of eV*s so in those units e=1
                            g = -1 / hbar * df0dk / nu_el
                            valley_transport[tp][el_mech][c][
                                T] = self.integrate_over_states(g * v, tp)
                            # from equation 45 in Rode, inelastic mechanisms
                        for inel_mech in self.inelastic_scats:
                            g = self.array_from_kgrid("g_" + inel_mech, tp, c,
                                                      T)
                            valley_transport[tp][inel_mech][c][
                                T] = self.integrate_over_states(g * v, tp)

                        # from equation 45 in Rode, overall
                        g = self.array_from_kgrid("g", tp, c, T)
                        valley_transport[tp]['overall'][c][
                            T] = self.integrate_over_states(g * v, tp)
                        g_th = self.array_from_kgrid("g", tp, c, T)
                        valley_transport[tp]["J_th"][c][
                            T] = self.integrate_over_states(g_th * v, tp)
                        valley_transport[tp]["seebeck"][c][
                            T] = self.integrate_over_states(
                            f0_all * (1 - f0_all) * E_array, tp) / (k_B * T)

                    # figure out average mobility
                    faulty_overall_mobility = False
                    mu_overrall_norm = norm(
                        valley_transport[tp]['overall'][c][T])
                    mu_average = np.array([0.0, 0.0, 0.0])
                    for transport in self.elastic_scats + self.inelastic_scats:
                        # averaging all mobility values via Matthiessen's rule
                        mu_average += 1 / (np.array(
                            valley_transport[tp][transport][c][T]) + 1e-32)
                        if mu_overrall_norm > norm(
                                valley_transport[tp][transport][c][T]):
                            faulty_overall_mobility = True  # because the overall mobility should be lower than all
                        valley_transport[tp]["average"][c][T] = 1 / mu_average

                    # Decide if the overall mobility make sense or it should be equal to average (e.g. when POP is off)
                    if (
                            mu_overrall_norm == 0.0 or faulty_overall_mobility) and not test_anisotropic:
                        self.logger.warning(
                            'There may be a problem with overall '
                            'mobility; setting it to average...')
                        valley_transport[tp]['overall'][c][T] = \
                            valley_transport[tp]["average"][c][T]

        return valley_transport

# if __name__ == "__main__":
#
#     # inputs
#     mass = 0.25
#     use_parabolic_bands = False
#     model_params = {'bs_is_isotropic': True,
#                     'elastic_scats': ['ACD', 'IMP', 'PIE'],
#                     'inelastic_scats': ['POP']
#                     }
#     if use_parabolic_bands:
#         model_params["parabolic_bands"] = [[
#             [[0.0, 0.0, 0.0], [0.0, mass]],
#         ]]
#     performance_params = {"dE_min": 0.0001, "nE_min": 5,
#                           "BTE_iters": 5,
#                           "max_nbands": 1,
#                           "max_normk": None,
#                           "n_jobs": -1,
#                           "max_nvalleys": 1,
#                           "interpolation": "boltztrap1",
#                           "max_Ecut": 1.0,
#                           "dos_kdensity": 50
#                           }
#     material_params = {"epsilon_s": 12.9, "epsilon_inf": 10.9, "W_POP": 8.73,
#                        # experimental from [R]
#                        "C_el": 139.7, "E_D": {"n": 8.6, "p": 8.6},
#                        "P_PIE": 0.052,
#                        "user_bandgap": 1.54,
#                        # "important_points": {'n': [[0. , 0.5, 0. ]],
#                        #                      'p': [[0. , 0.0, 0. ]]},
#                        }
#     input_dir = "../test_files/GaAs_mp-2534"
#     # coeff_file = None
#     coeff_file = os.path.join(input_dir, "fort.123")
#
#     # instantiate and run AMSET:
#     amset = Amset.from_vasprun(
#         os.path.join(input_dir, "vasprun.xml"),
#         material_params=material_params, calc_dir='.',
#         model_params=model_params,
#         performance_params=performance_params,
#         dopings=[-3e13],
#         temperatures=[300, 600],
#         integration='e',
#     )
#     amset.run_profiled(coeff_file, kgrid_tp='very fine')

# amset.write_input_files()
# amset.to_csv()
# amset.as_dict()
# amset.to_file()
# amset.plot(k_plots=['energy'], e_plots='all', mode='offline',
#            carrier_types=amset.all_types)
#
# amset.grids_to_json(kgrid=True, trimmed=True, max_ndata=100, n0=0)
