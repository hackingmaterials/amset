# coding: utf-8
from __future__ import absolute_import
import gzip
import warnings
import time
import json
from collections import OrderedDict
from multiprocessing import cpu_count

from numpy import dot
from pstats import Stats
from random import random
from scipy.interpolate import griddata
from pprint import pprint
import os
from sys import stdout as STDOUT

import numpy as np
from math import log, pi
from pymatgen.electronic_structure.boltztrap import BoltztrapRunner
from pymatgen.io.vasp import Vasprun, Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from monty.json import MontyEncoder, MontyDecoder
import cProfile
from copy import deepcopy

from amset.utils.analytical_band_from_BZT import Analytical_bands, get_dos_from_poly_bands, get_poly_energy

from amset.utils.tools import norm, generate_k_mesh_axes, \
    create_grid, array_to_kgrid, normalize_array, f0, df0dE, cos_angle, \
    fermi_integral, calculate_Sio, remove_from_grid, get_tp, \
    remove_duplicate_kpoints, get_angle, sort_angles, get_closest_k, \
    get_energy_args, get_bindex_bspin, get_bs_extrema, \
    AmsetError, kpts_to_first_BZ, get_dos_boltztrap2, \
    setup_custom_logger, insert_intermediate_kpoints, interpolate_bs, \
    get_dft_orbitals, generate_adaptive_kmesh, create_plots

from amset.utils.constants import hbar, m_e, A_to_m, m_to_cm, \
    A_to_nm, e, k_B, \
    epsilon_0, default_small_E, dTdz, sq3

try:
    import BoltzTraP2
    import BoltzTraP2.dft
    from BoltzTraP2 import sphere, fite
    from amset.utils.pymatgen_loader_for_bzt2 import PymatgenLoader
except ImportError:
    warnings.warn('BoltzTraP2 not imported, "boltztrap2" interpolation not available.')

__author__ = "Alireza Faghaninia, Jason Frost, Anubhav Jain"
__copyright__ = "Copyright 2017, HackingMaterials"
__version__ = "0.1.0"
__maintainer__ = "Alireza Faghaninia"
__email__ = "alireza.faghaninia@gmail.com"
__status__ = "Development"



class Amset(object):
    """ This class is used to run Amset on a pymatgen from a VASP run (i.e. vasprun.xml). Amset is an ab initio model
    for calculating the mobility and Seebeck coefficient using Bolƒtzmann transport equation (BTE). The band structure
    in the Brilluin zone (BZ) is extracted from vasprun.xml to calculate the group velocity and transport properties
    in presence of various scattering mechanisms.

     Currently the following scattering mechanisms with their corresponding three-letter abbreviations implemented are:
     ionized impurity scattering (IMP), acoustic phonon deformation potential (ACD), piezoelectric (PIE), and charged
     dislocation scattering (DIS). Also, longitudinal polar optical phonon (POP) in implemented as an inelastic
     scattering mechanism that can alter the electronic distribution (the reason BTE has to be solved explicitly; for
     more information, see references [R, A]).

     you can control the level of theory via various inputs. For example, by assuming that the band structure is
     isotropic at the surrounding point of each k-point (i.e. bs_is_isotropic == True), once can significantly reduce
     the computational effort needed for accurate numerical integration of the scatterings.

    * a small comment on the structure of this code: the calculations are done and stred in two main dictionary type
    variable called kgrid and egrid. kgrid contains all calculations that are done in k-space meaning that for each
    k-point and each band that is included there is a number/vector/property stored. On the other hand, the egrid
    is everything in energy scale hence we have number/vector/property stored at each energy point.

     References:
         [R]: D. L. Rode, Low-Field Electron Transport, Elsevier, 1975, vol. 10., DOI: 10.1016/S0080-8784(08)60331-2
         [A]: A. Faghaninia, C. S. Lo and J. W. Ager, Phys. Rev. B, "Ab initio electronic transport model with explicit
          solution to the linearized Boltzmann transport equation" 2015, 91(23), 5100., DOI: 10.1103/PhysRevB.91.235123
         [Q]: B. K. Ridley, Quantum Processes in Semiconductors, oxford university press, Oxford, 5th edn., 2013.
          DOI: 10.1093/acprof:oso/9780199677214.001.0001

     """
    def __init__(self, calc_dir, material_params, vasprun_file=None,
                 model_params=None, performance_params=None,
                 dopings=None, temperatures=None, integration='e', loglevel=None):
        """
        Args:
            calc_dir (str): path to the vasprun.xml (a required argument)
            material_params (dict): parameters related to the material (a required argument)
            model_params (dict): parameters related to the model used and the level of theory
            performance_params (dict): parameters related to convergence, speed, etc.
            dopings ([float]): list of input carrier concentrations; c<0 for electrons and c>0 for holes
            temperatures ([float]): list of input temperatures
            loglevel (int): e.g. logging.DEBUG
        """

        self.logger = setup_custom_logger('amset_logger', calc_dir, 'amset.log',
                                     level=loglevel)
        self.calc_dir = calc_dir
        self.vasprun_file = vasprun_file
        self.dopings = dopings or [-1e20, 1e20]
        self.all_types = list(set([get_tp(c) for c in self.dopings]))
        self.tp_title = {"n": "conduction band(s)", "p": "valence band(s)"}
        self.temperatures = temperatures or [300.0, 600.0]
        self.set_model_params(model_params)
        self.logger.info('independent_valleys: {}'.format(self.independent_valleys))
        self.set_material_params(material_params)
        self.set_performance_params(performance_params)
        self.integration = integration
        self.logger.info('integration: {}'.format(self.integration))
        if self.integration=="k":
            self.logger.warning(
                "k-integration is not fully implemented! The results may be "
                "unreliable. It also only works for structures with a symmetric"
                " lattic constant! 'e'-integration is recommended.")
        elif self.integration!="e":
            raise AmsetError(self.logger, "Unsupported integration method: "
                                          "'{}'".format(self.integration))
        self.logger.info("number of cpu used (n_jobs): {}".format(self.n_jobs))
        self.counter = 0 # a global counter just for debugging
        self.offset_from_vrun = {'n': 0.0, 'p': 0.0}


    def run_profiled(self, coeff_file=None, kgrid_tp="coarse",
                     write_outputs=True, nfuncs=15):
        """
        Very similar to the run method except that it is time-profiled.

        Args:
            see args (coeff_file, kgrid_tp and write_outputs) for run() method
            nfuncs (int): only print the nfuncs most time-consuming functions
        """
        profiler = cProfile.Profile()
        profiler.runcall(lambda: self.run(coeff_file, kgrid_tp=kgrid_tp,
                                               write_outputs=write_outputs))
        stats = Stats(profiler, stream=STDOUT)
        stats.strip_dirs()
        stats.sort_stats('cumulative')
        stats.print_stats(nfuncs)


    def run(self, coeff_file=None, kgrid_tp="coarse",
            write_outputs=True, test_k_anisotropic=False):
        """
        Function to run Amset and generate the main outputs.

        Args:
            coeff_file: the path to fort.123* file containing the coefficients of
                the interpolated band structure generated by a modified version of
                BoltzTraP. If None, BoltzTraP will run to generate the file.
            kgrid_tp (str): define the density of k-point mesh.
                options: 'very coarse', 'coarse', 'fine'

        Returns (None):
            many instance variables get updated with calculated properties.
        """
        self.logger.info('Running on "{}" mesh for each valley'.format(kgrid_tp))
        self.read_vrun(vasprun_file=self.vasprun_file, filename="vasprun.xml")
        self._initialize_transport_vars(coeff_file=coeff_file)
        # make the reference energy consistent w/ interpolation rather than DFT
        self.update_cbm_vbm_dos(coeff_file=coeff_file)
        #TODO: if we use ibands_tuple, then for each couple of conduction/valence bands we only use 1 band together (i.e. always ib==0)
        for tp in ['p', 'n']:
            self.cbm_vbm[tp]['included'] = 1
        self.logger.debug("cbm_vbm after updating:\n {}".format(self.cbm_vbm))
        if self.integration == 'k':
            kpts = self.generate_kmesh(important_points={'n': [[0.0, 0.0, 0.0]], 'p': [[0.0, 0.0, 0.0]]}, kgrid_tp=kgrid_tp)
            # the purpose of the following line is just to generate self.energy_array that find_fermi_k function uses
            _, _ = self.get_energy_array(coeff_file, kpts, once_called=False, return_energies=True, num_bands=self.initial_num_bands, nbelow_vbm=0, nabove_cbm=0)
            self.fermi_level = self.find_fermi_k(num_bands=self.initial_num_bands)
        elif self.integration == 'e':
            # method 1: good for k-integration but limited to symmetric lattice vectors
            kpts = self.generate_kmesh(important_points={'n': [[0.0, 0.0, 0.0]], 'p': [[0.0, 0.0, 0.0]]}, kgrid_tp='very coarse')
            self.get_energy_array(coeff_file, kpts, once_called=False, return_energies=False, num_bands=self.initial_num_bands, nbelow_vbm=0, nabove_cbm=0)
            self.fermi_level = {c: {T: None for T in self.temperatures} for c in self.dopings}
            for c in self.dopings:
                for T in self.temperatures:
                    self.fermi_level[c][T] = self.find_fermi(c, T)
        self.logger.info('fermi level = {}'.format(self.fermi_level))
        self.logger.info('here initial number of bands:\n{}'.format(self.initial_num_bands))
        vibands = list(range(self.initial_num_bands['p']))
        cibands = list(range(self.initial_num_bands['n']))

        if len(vibands) > len(cibands):
            ibands_tuple = list(zip(vibands, cibands+[cibands[0]]*(len(vibands)-len(cibands))))
            for ivt in range(len(cibands), len(vibands)):
                self.count_mobility[ivt]['n'] = False
        else:
            ibands_tuple = list(zip(vibands+[vibands[0]]*(len(cibands)-len(vibands)), cibands))
            for ivt in range(len(vibands), len(cibands)):
                self.count_mobility[ivt]['p'] = False
        self.ibands_tuple = ibands_tuple
        self.count_mobility0 = deepcopy(self.count_mobility)
        #each time num_bands will be {'n': 1, 'p': 1} but w/ different band idx
        if self.max_nbands:
            ibands_tuple = ibands_tuple[:min(len(ibands_tuple), self.max_nbands)]

        self.logger.debug('here ibands_tuple: [(val. band #, cond. band #)]')
        self.logger.debug(ibands_tuple)
        self.logger.debug('here whether to count bands')
        self.logger.debug(self.count_mobility)

        self.denominator = {c: {T: {'p': 0.0, 'n': 0.0} for T in self.temperatures} for c in self.dopings}
        self.seeb_denom = {c: {T: {'p': 0.0, 'n': 0.0} for T in self.temperatures} for c in self.dopings}
        for self.ibrun, (self.nbelow_vbm, self.nabove_cbm) in enumerate(ibands_tuple):
            self.logger.info('going over conduction and valence # {}'.format(self.ibrun))
            self.find_all_important_points(coeff_file,
                                           nbelow_vbm=self.nbelow_vbm,
                                           nabove_cbm=self.nabove_cbm,
                                           interpolation=self.interpolation)
            max_nvalleys = max(len(self.important_pts['n']),
                               len(self.important_pts['p']))
            if self.max_nvalleys is not None:
                max_nvalleys = min(max_nvalleys, self.max_nvalleys)
            for ivalley in range(max_nvalleys):
                self.count_mobility[self.ibrun] = self.count_mobility0[self.ibrun]
                once_called = True
                important_points = {'n': None, 'p': None}
                if ivalley == 0 and self.ibrun==0:
                    once_called = False
                for tp in ['p', 'n']:
                    try:
                        important_points[tp] = [self.important_pts[tp][ivalley]]
                    except:
                        important_points[tp] = [self.important_pts[tp][0]]
                        self.count_mobility[self.ibrun][tp] = False

                if self.max_normk0 is None:
                    for tp in ['n', 'p']:
                        min_dist = 100.0 # in 1/nm
                        kc = self.get_cartesian_coords(important_points[tp][0])
                        for k in self.bs.get_sym_eq_kpoints(kc, cartesian=True):
                            kc_carts = [self.get_cartesian_coords(kp) for \
                                        kp in self.all_important_pts[tp]]
                            new_dist = 1/A_to_nm * norm(get_closest_k(
                                k, kc_carts, return_diff=True, exclude_self=True))
                            # to avoid self-counting, 0.01 criterion added:
                            if new_dist < min_dist and new_dist > 0.01:
                                min_dist = new_dist
                        self.max_normk[tp] = min_dist/2.0
                if self.max_nvalleys and self.max_nvalleys==1:
                    kmax = self.max_normk0 or 5.0
                    self.max_normk = {'n': kmax, 'p': kmax}
                    if self.max_normk0 is None:
                        self.logger.warn('max_normk set to {} to avoid'
                                         'unlrealistic scattering'.format(kmax))

                self.logger.info('at valence band #{} and conduction band #{}'.format(self.nbelow_vbm, self.nabove_cbm))
                self.logger.info('Current valleys:\n{}'.format(important_points))
                self.logger.info('Whether to count valleys: {}'.format(self.count_mobility[self.ibrun]))
                self.logger.info('max_normk:\n{}'.format(self.max_normk))
                self.logger.info('important points for this band:\n{}'.format(important_points))

                if not self.count_mobility[self.ibrun]['n'] and not self.count_mobility[self.ibrun]['p']:
                    self.logger.info('skipping this valley as it is unimportant for both n and p type...')
                    continue
                kpts = self.generate_kmesh(important_points=important_points, kgrid_tp=kgrid_tp)
                kpts, energies=self.get_energy_array(coeff_file, kpts,
                                                     once_called=once_called,
                                                     return_energies=True,
                                                     nbelow_vbm=self.nbelow_vbm,
                                                     nabove_cbm=self.nabove_cbm,
                                                     num_bands={'p': 1, 'n': 1})

                if min(energies['n']) - self.cbm_vbm['n']['energy'] > self.Ecut['n']:
                    self.logger.info('not counting conduction band {} valley\
                     {} due to off enery...'.format(self.ibrun, important_points['n'][0]))
                    self.count_mobility[self.ibrun]['n'] = False
                if self.cbm_vbm['p']['energy'] - max(energies['p']) > self.Ecut['p']:
                    self.logger.info('not counting valence band {} valley {} \
                    due to off enery...'.format(self.ibrun, important_points['p'][0]))
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

                if not self.count_mobility[self.ibrun]['n'] and not self.count_mobility[self.ibrun]['p']:
                    self.logger.info('skipping this valley as it is unimportant or its energies are way off...')
                    continue

                # for now, I keep once_called as False in init_egrid until I get rid of egrid mobilities
                corrupt_tps = self.init_egrid(once_called=False)
                for tp in corrupt_tps:
                    self.count_mobility[self.ibrun][tp] = False
                if not self.count_mobility[self.ibrun]['n'] and not self.count_mobility[self.ibrun]['p']:
                    self.logger.info('skipping this valley as it is unimportant or its energies are way off...')
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

                if self.independent_valleys:
                    for c in self.dopings:
                        for T in self.temperatures:
                            if self.integration=='k':
                                f0_all = 1 / (np.exp((self.energy_array['n'] - self.fermi_level[c][T]) / (k_B * T)) + 1)
                                f0p_all = 1 / (np.exp((self.energy_array['p'] - self.fermi_level[c][T]) / (k_B * T)) + 1)
                                self.denominator[c][T]['n'] = (3 * default_small_E * self.integrate_over_states(f0_all, 'n') + 1e-10)
                                self.denominator[c][T]['p'] = (3 * default_small_E * self.integrate_over_states(1-f0p_all, 'p') + 1e-10)
                            elif self.integration=='e':
                                self.denominator[c][T]['n'] = 3 * default_small_E * self.integrate_over_E(prop_list=["f0"], tp='n', c=c, T=T, xDOS=False, xvel=False)
                                self.denominator[c][T]['p'] = 3 * default_small_E * self.integrate_over_E(prop_list=["1 - f0"], tp='p', c=c, T=T, xDOS=False, xvel=False)
                                for tp in ['n', 'p']:
                                    self.seeb_denom[c][T][tp] = self.egrid["Seebeck_integral_denominator"][c][T][tp]

                # find the indexes of equal energy or those with ±hbar*W_POP for scattering via phonon emission and absorption
                if not self.bs_is_isotropic or "POP" in self.inelastic_scats:
                    self.generate_angles_and_indexes_for_integration()

                # calculate all elastic scattering rates in kgrid and then map it to egrid:
                for sname in self.elastic_scats:
                    self.s_elastic(sname=sname)
                    self.map_to_egrid(prop_name=sname)

                self.map_to_egrid(prop_name="_all_elastic")
                self.map_to_egrid(prop_name="relaxation time")

                for c in self.dopings:
                    for T in self.temperatures:
                        fermi = self.fermi_level[c][T]
                        for tp in ["n", "p"]:
                            fermi_norm = fermi - self.cbm_vbm[tp]["energy"]
                            for ib in range(len(self.kgrid[tp]["energy"])):
                                for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                                    E = self.kgrid[tp]["energy"][ib][ik]
                                    v = self.kgrid[tp]["velocity"][ib][ik]
                                    self.kgrid[tp]["f0"][c][T][ib][ik] = f0(E, fermi, T) * 1.0
                                    self.kgrid[tp]["df0dk"][c][T][ib][ik] = hbar * df0dE(E, fermi, T) * v  # in cm
                                    self.kgrid[tp]["electric force"][c][T][ib][ik] = \
                                        -1 * self.kgrid[tp]["df0dk"][c][T][ib][ik] * \
                                        default_small_E / hbar  # in 1/s

                                    E_norm = E - self.cbm_vbm[tp]["energy"]
                                    self.kgrid[tp]["thermal force"][c][T][ib][ik] = \
                                        - v * f0(E_norm, fermi_norm, T) * (1 - f0(
                                        E_norm, fermi_norm, T)) * (E_norm / (k_B * T)-\
                                        self.egrid["Seebeck_integral_numerator"][c][
                                            T][tp] / self.egrid[
                                                "Seebeck_integral_denominator"][c][
                                                T][tp]) * dTdz / T
                        if self.integration=='k':
                            dop_tp = get_tp(c)
                            f0_all = 1 / (np.exp((self.energy_array[dop_tp] - self.fermi_level[c][T]) / (k_B * T)) + 1)
                            if c < 0:
                                electrons = self.integrate_over_states(f0_all, dop_tp)
                                self.logger.info('k-integral of f0 above band gap at c={}, T={}: {}'.format(c, T, electrons))
                            if c > 0:
                                holes = self.integrate_over_states(1-f0_all, dop_tp)
                                self.logger.info('k-integral of 1-f0 below band gap at c={}, T={}: {}'.format(c, T, holes))

                self.map_to_egrid(prop_name="f0", c_and_T_idx=True, prop_type="vector")
                self.map_to_egrid(prop_name="df0dk", c_and_T_idx=True, prop_type="vector")

                # solve BTE in presence of electric and thermal driving force to get perturbation to Fermi-Dirac: g
                self.solve_BTE_iteratively()
                if self.integration=='k':
                    valley_transport = self.calculate_transport_properties_with_k(test_k_anisotropic, important_points)
                elif self.integration=='e':
                    valley_transport = self.calculate_transport_properties_with_E(important_points)
                else:
                    raise AmsetError('Unsupported integration method: {}'.format(self.integration))
                self.logger.info('mobility of the valley {} and band (p, n) {}'.format(important_points, self.ibands_tuple[self.ibrun]))
                self.logger.info('count_mobility: {}'.format(self.count_mobility[self.ibrun]))
                pprint(valley_transport)

                if self.ibrun==0 and ivalley==0: # 1-valley only since it's SPB
                    self.calculate_spb_transport()


                self.logger.info('Mobility Labels: {}'.format(self.mo_labels))
                for c in self.dopings:
                    for T in self.temperatures:
                        for tp in ['p', 'n']:
                            valley_ndegen = self.bs.get_kpoint_degeneracy(important_points[tp][0])
                            if self.count_mobility[self.ibrun][tp]:
                                if not self.independent_valleys:
                                    if self.integration=='k':
                                        f0_all = 1. / (np.exp((self.energy_array['n'] - self.fermi_level[c][T]) / (k_B * T)) + 1.)
                                        f0p_all = 1. / (np.exp((self.energy_array['p'] - self.fermi_level[c][T]) / (k_B * T)) + 1.)
                                        finteg = f0_all if tp == 'n' else 1-f0p_all
                                        self.denominator[c][T][tp] += 3 * default_small_E * self.integrate_over_states(finteg, tp) + 1e-10
                                        self.seeb_denom[c][T][tp] += self.integrate_over_states(finteg*(1-finteg), tp)
                                    elif self.integration=='e':
                                        finteg = "f0" if tp=="n" else "1 - f0"
                                        self.denominator[c][T][tp] += 3 * default_small_E * self.integrate_over_E(prop_list=[finteg], tp=tp, c=c, T=T, xDOS=False, xvel=False)  * valley_ndegen
                                        self.seeb_denom[c][T][tp] += self.egrid["Seebeck_integral_denominator"][c][T][tp] * valley_ndegen
                                    for mu in self.mo_labels+["J_th"]:
                                        self.mobility[tp][mu][c][T] += valley_transport[tp][mu][c][T] * valley_ndegen
                                    self.mobility[tp]['seebeck'][c][T] += valley_transport[tp]['seebeck'][c][T] # seeb is multiplied by DOS so no need for degeneracy
                                else:
                                    for mu in self.mo_labels+["J_th"]:
                                        self.mobility[tp][mu][c][T] += valley_transport[tp][mu][c][T] * valley_ndegen
                                    self.mobility[tp]["seebeck"][c][T] += valley_transport[tp]["seebeck"][c][T] * valley_ndegen
                                    self.mobility[tp]["J_th"][c][T] += valley_transport[tp]["J_th"][c][T]


                if self.poly_bands0 is None:
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
                            self.valleys[tp]['band {}'.format(self.ibrun)]['{};{};{}'.format(k[0], k[1], k[2])] = valley_transport[tp]

                kgrid_rm_list = ["effective mass", "kweights",
                                 "f_th", "S_i_th", "S_o_th"]
                self.kgrid = remove_from_grid(self.kgrid, kgrid_rm_list)
                if ivalley==0 and self.ibrun==0:
                    # TODO: make it possible for the user to choose which valley(s) to plot
                    self.kgrid0 = deepcopy(self.kgrid)
                    self.egrid0 = deepcopy(self.egrid)
                    self.Efrequency0 = deepcopy(self.Efrequency)
        self.logger.debug('here denominator:\n{}'.format(self.denominator))

        if not self.independent_valleys:
            for tp in ['p', 'n']:
                for c in self.dopings:
                    for T in self.temperatures:
                        for mu in self.mo_labels+["J_th"]:
                            self.mobility[tp][mu][c][T] /= self.denominator[c][T][tp]
                            for band in list(self.valleys[tp].keys()):
                                for valley_k in list(self.valleys[tp][band].keys()):
                                    self.valleys[tp][band][valley_k][mu][c][T] /= self.denominator[c][T][tp]
                        self.mobility[tp]["seebeck"][c][T] /= self.seeb_denom[c][T][tp]
                        for band in list(self.valleys[tp].keys()):
                                for valley_k in list(self.valleys[tp][band].keys()):
                                    self.valleys[tp][band][valley_k]["seebeck"][c][T] /= self.seeb_denom[c][T][tp]

        # finalize Seebeck values:
        for tp in ['p', 'n']:
            for c in self.dopings:
                for T in self.temperatures:
                    self.mobility[tp]['seebeck'][c][T] -= (self.fermi_level[c][T] - self.cbm_vbm[tp]["energy"]) / (k_B * T)
                    self.mobility[tp]['seebeck'][c][T] *= -1e6 * k_B
                    self.mobility[tp]["seebeck"][c][T] += 1e6 * self.mobility[tp]["J_th"][c][T]/(self.mobility[tp]["overall"][c][T]*e*abs(c))/dTdz
                    for band in list(self.valleys[tp].keys()):
                        for valley_k in list(self.valleys[tp][band].keys()):
                            self.valleys[tp][band][valley_k]["seebeck"][c][T] -= (self.fermi_level[c][T] - self.cbm_vbm[tp]["energy"]) / (k_B * T)
                            self.valleys[tp][band][valley_k]["seebeck"][c][T] *= -1e6 * k_B

        print('\nFinal Transport Values:')
        pprint(self.mobility)
        if write_outputs:
            self.to_file()


    def _initialize_transport_vars(self, coeff_file):
        """
        Variables related to transport such as cbm_vbm, mobility, etc. This
        internal method is supposed to be called after read_vrun.

        Args:
            coeff_file: the path to fort.123* file containing the coefficients of
                the interpolated band structure generated by a modified version of
                BoltzTraP. If None, BoltzTraP will run to generate the file.

        Returns (None): instance variables get updated/created.
        """
        if self.poly_bands0 is not None:
            self.cbm_vbm["n"]["energy"] = self.dft_gap
            self.cbm_vbm["p"]["energy"] = 0.0
            self.cbm_vbm["n"]["kpoint"] = self.cbm_vbm["p"]["kpoint"] = \
                self.poly_bands0[0][0][0]
        if not coeff_file and self.interpolation == "boltztrap1":
            self.logger.warning(
                '\nRunning BoltzTraP to generate the cube file...')
            boltztrap_runner = BoltztrapRunner(bs=self.bs, nelec=self.nelec,
                                               doping=[1e20],
                                               tmax=max(self.temperatures))
            boltztrap_runner.run(path_dir=self.calc_dir)
            coeff_file = os.path.join(self.calc_dir, 'boltztrap', 'fort.123')
            self.logger.warning(
                'BoltzTraP run finished, I suggest to set the following '
                'to skip this step next time:\n{}="{}"'.format(
                    "coeff_file",
                    os.path.join(self.calc_dir, 'boltztrap', 'fort.123')
                ))
            if not os.path.exists(coeff_file):
                raise AmsetError(self.logger,
                                 '{} does not exist! generating the cube file '
                                 '(i.e. fort.123) requires a modified version of BoltzTraP. '
                                 'Contact {}'.format(coeff_file, __email__))
        # initialize transport variables
        self.mo_labels = self.elastic_scats + self.inelastic_scats + ['overall', 'average']
        self.spb_labels = ['SPB_ACD']
        self.transport_labels = self.mo_labels + self.spb_labels + ["seebeck","J_th"]
        self.mobility = {tp: {el_mech: {c: {T: np.array([0., 0., 0.]) \
                                            for T in self.temperatures} \
                                        for c in self.dopings} \
                              for el_mech in self.transport_labels} \
                         for tp in ["n", "p"]}
        self.calc_doping = {c: {T: {'n': None, 'p': None} \
                                for T in self.temperatures} \
                            for c in self.dopings}
        self.ibrun = 0 # counter of the ibands_tuple (band-valley walker)
        self.count_mobility = [{'n': True, 'p': True} \
                               for _ in range(max(self.initial_num_bands['p'],
                                                  self.initial_num_bands['n']))]


    def calculate_spb_transport(self):
        """
        Using single parabolic band (SPB), calculates some elastic scattering-
        limited mobility values named with an SPB_ prefix. For example,
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
                    self.mobility[tp]["SPB_ACD"][c][T] = 2 ** 0.5 * pi * hbar ** 4 * e * self.C_el * 1e9 / (
                        3 * (self.cbm_vbm[tp]["eff_mass_xx"] * m_e) ** 2.5 * (k_B * T) ** 1.5 * self.E_D[tp] ** 2) \
                                * fermi_integral(0, fermi, T, energy) \
                                / fermi_integral(0.5, fermi, T, energy) * e ** 0.5 * 1e4  # to cm2/V.s


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
            kpts = generate_adaptive_kmesh(self.bs, important_points, kgrid_tp)
        for tp in ['n', 'p']:
            points_1d = generate_k_mesh_axes(important_points[tp], kgrid_tp, one_list=True)
            self.kgrid_array[tp] = create_grid(points_1d)
            if self.integration == 'k':
                kpts[tp] = array_to_kgrid(self.kgrid_array[tp])
            N = self.kgrid_array[tp].shape
            self.kgrid_array_cartesian[tp] = np.zeros((N[0], N[1], N[2], 3))
            for ii in range(N[0]):
                for jj in range(N[1]):
                    for kk in range(N[2]):
                        self.kgrid_array_cartesian[tp][ii,jj,kk,:] = self.get_cartesian_coords(self.kgrid_array[tp][ii,jj,kk])   # 1/A

            # generate a normalized numpy array of vectors pointing in the direction of k
            self.k_hat_array[tp] = normalize_array(self.kgrid_array[tp])
            self.k_hat_array_cartesian[tp] = normalize_array(self.kgrid_array_cartesian[tp])
            self.dv_grid[tp] = self.find_dv(self.kgrid_array[tp])
        return kpts


    def update_cbm_vbm_dos(self, coeff_file):
        if self.poly_bands0 is None:
            if self.interpolation=="boltztrap1":
                self.logger.debug(
                    "start interpolating bands from {}".format(coeff_file))
            self.all_ibands = []
            for i, tp in enumerate(["p", "n"]):
                sgn = (-1) ** (i + 1)
                for ib in range(self.cbm_vbm0[tp]["included"]):
                    self.all_ibands.append(self.cbm_vbm0[tp]["bidx"] + sgn * ib)

            self.all_ibands.sort()
            self.logger.debug("all_ibands: {}".format(self.all_ibands))
            if self.interpolation == "boltztrap1":
                self.interp_params = get_energy_args(coeff_file, self.all_ibands)
        else:
            self.poly_bands = np.array(self.poly_bands0)
            for ib in range(len(self.poly_bands0)):
                for valley in range(len(self.poly_bands0[ib])):
                    self.poly_bands[ib][valley][
                        0] = remove_duplicate_kpoints(
                        self.bs.get_sym_eq_kpoints(
                            self.poly_bands0[ib][valley][0],
                            cartesian=True))


        for i, tp in enumerate(["p", "n"]):
            sgn = (-1) ** i
            iband = i*self.cbm_vbm0["p"]["included"] if self.interpolation=="boltztrap1" else self.cbm_vbm[tp]["bidx"]
            if self.poly_bands is not None:
                energy, velocity, effective_m = self.calc_poly_energy(self.cbm_vbm0[tp]["kpoint"], tp, 0)
            else:
                energies, _, masses = interpolate_bs([self.cbm_vbm0[tp]["kpoint"]], self.interp_params, iband=iband, sgn=sgn, method=self.interpolation, scissor=self.scissor, matrix=self._vrun.lattice.matrix)
                energy = energies[0]
                effective_m = masses[0]

            self.offset_from_vrun[tp] = energy - self.cbm_vbm0[tp]["energy"]
            self.logger.debug("offset from vasprun energy values for {}-type = {} eV".format(tp, self.offset_from_vrun[tp]))
            self.cbm_vbm0[tp]["energy"] = energy
            self.cbm_vbm0[tp]["eff_mass_xx"] = effective_m.diagonal()

        if self.poly_bands is None:
            self.dos_emax += self.offset_from_vrun['n']
            self.dos_emin += self.offset_from_vrun['p']
        for tp in ['p', 'n']:
            self.cbm_vbm[tp]['energy'] = self.cbm_vbm0[tp]['energy']
            self.cbm_vbm[tp]['eff_mass_xx'] = self.cbm_vbm0[tp]['eff_mass_xx']
        self._avg_eff_mass = {tp: abs(np.mean(self.cbm_vbm0[tp]["eff_mass_xx"])) for tp in ["n", "p"]}


    def get_energy_array(self, coeff_file, kpts, once_called=False,
                         return_energies=False, num_bands=None,
                         nbelow_vbm=0, nabove_cbm=0):
        num_bands = num_bands or self.num_bands
        start_time = time.time()
        self.logger.info("self.nkibz = {}".format(self.nkibz))
        if self.poly_bands0 is None:
            if self.interpolation == 'boltztrap1':
                self.logger.debug("start interpolating bands from {}".format(coeff_file))
                analytical_bands = Analytical_bands(coeff_file=coeff_file)
                self.all_ibands = []
                for ib in range(num_bands['p']):
                    self.all_ibands.append(self.cbm_vbm0['p']["bidx"] - nbelow_vbm - ib)
                for ib in range(num_bands['n']):
                    self.all_ibands.append(self.cbm_vbm0['n']["bidx"] + nabove_cbm + ib)
                self.all_ibands.sort()
                self.logger.debug("all_ibands: {}".format(self.all_ibands))
                self.interp_params = get_energy_args(coeff_file, self.all_ibands)
            elif self.interpolation != 'boltztrap2':
                raise ValueError('Unsupported interpolation method: "{}"'.format(self.interpolation))
        else:
            # first modify the self.poly_bands to include all symmetrically equivalent k-points (k_i)
            # these points will be used later to generate energy based on the minimum norm(k-k_i)
            self.poly_bands = np.array(self.poly_bands0)
            for ib in range(len(self.poly_bands0)):
                for valley in range(len(self.poly_bands0[ib])):
                    self.poly_bands[ib][valley][0] = remove_duplicate_kpoints(
                        self.bs.get_sym_eq_kpoints(self.poly_bands0[ib][valley][0], cartesian=True))

        self.logger.debug("time to get engre and calculate the outvec2: {} seconds".format(time.time() - start_time))
        start_time = time.time()
        energies = {"n": [0.0 for k in kpts['n']], "p": [0.0 for k in kpts['p']]}
        energies_sorted = {"n": [0.0 for k in kpts['n']], "p": [0.0 for k in kpts['p']]}
        velocities = {"n": [[0.0, 0.0, 0.0] for k in kpts['n']], "p": [[0.0, 0.0, 0.0] for k in kpts['p']]}
        self.pos_idx = {'n': [], 'p': []}
        self.energy_array = {'n': [], 'p': []}

        if return_energies:
            for i, tp in enumerate(["p", "n"]):
                sgn = (-1) ** i
                for ib in range(num_bands[tp]):
                    if self.poly_bands is not None:
                        for ik in range(len(kpts[tp])):
                            energies[tp][ik], _, _ = self.calc_poly_energy(kpts[tp][ik], tp, ib)
                    else:
                        iband = i * num_bands['p'] + ib if self.interpolation=="boltztrap1" else self.cbm_vbm[tp]["bidx"] + (i - 1) * self.cbm_vbm["p"]["included"] + ib
                        energies[tp], velocities[tp], _ = interpolate_bs(
                                kpts[tp], interp_params=self.interp_params,
                                iband=iband, sgn=sgn, method=self.interpolation,
                                scissor=self.scissor,
                                matrix=self._vrun.lattice.matrix, n_jobs=self.n_jobs)
                    if self.integration == 'k':
                        self.energy_array[tp].append(self.grid_from_ordered_list(energies[tp], tp, none_missing=True))

                    # we only need the 1st band energies to order k-points:
                    if ib == 0:
                        e_sort_idx = np.array(energies[tp]).argsort() if tp == "n" else np.array(energies[tp]).argsort()[::-1]
                        energies_sorted[tp] = [energies[tp][ie] for ie in e_sort_idx]
                        energies[tp] = [energies[tp][ie] for ie in e_sort_idx]
                        self.pos_idx[tp] = np.array(range(len(e_sort_idx)))[e_sort_idx].argsort()
                        kpts[tp] = [kpts[tp][ie] for ie in e_sort_idx]

            self.logger.debug("time to calculate ibz energy, velocity info and store them to variables: \n {}".format(time.time()-start_time))
            if self.poly_bands is not None:
                all_bands_energies = {"n": [], "p": []}
                for tp in ["p", "n"]:
                    all_bands_energies[tp] = energies[tp]
                    for ib in range(1, len(self.poly_bands)):
                        for ik in range(len(kpts[tp])):
                            energy, velocity, effective_m = get_poly_energy(
                                self.get_cartesian_coords(kpts[ik]) / A_to_nm,
                                poly_bands=self.poly_bands, type=tp, ib=ib, bandgap=self.dft_gap + self.scissor)
                            all_bands_energies[tp].append(energy)
                if not once_called:
                    self.dos_emin = min(all_bands_energies["p"])
                    self.dos_emax = max(all_bands_energies["n"])

            del e_sort_idx
            self.energy_array = {tp: np.array(self.energy_array[tp]) for tp in
                                 ['p', 'n']}

        if not once_called:
            if self.poly_bands is None:
                if self.interpolation=="boltztrap1":
                    emesh, dos, dos_nbands, bmin=analytical_bands.get_dos_from_scratch(
                            self._vrun.final_structure, [
                            self.nkdos, self.nkdos, self.nkdos],self.dos_emin,
                            self.dos_emax,int(round((self.dos_emax - self.dos_emin) \
                            / max(self.dE_min, 0.0001))), width=self.dos_bwidth,
                            scissor=self.scissor, vbmidx=self.cbm_vbm["p"]["bidx"])
                    self.logger.debug("dos_nbands: {} \n".format(dos_nbands))
                    self.dos_start = min(self._vrun.get_band_structure().as_dict()["bands"]["1"][bmin]) \
                                     + self.offset_from_vrun['p']
                    self.dos_end = max(self._vrun.get_band_structure().as_dict()["bands"]["1"][bmin+dos_nbands]) \
                                   + self.offset_from_vrun['n']
                elif self.interpolation=="boltztrap2":
                    emesh, dos, dos_nbands = get_dos_boltztrap2(self.interp_params,
                                            self._vrun.final_structure,
                            mesh=[self.nkdos, self.nkdos, self.nkdos],
                            estep=max(self.dE_min, 0.0001), vbmidx = self.cbm_vbm["p"]["bidx"]-1,
                                width=self.dos_bwidth, scissor=self.scissor)
                    self.dos_start = emesh[0]
                    self.dos_emin = emesh[0]
                    self.dos_end = emesh[-1]
                    self.dos_emax = emesh[-1]
                else:
                    raise ValueError('Unsupported interpolation: "{}"'.format(self.interpolation))
                self.dos_normalization_factor = dos_nbands if self.soc else dos_nbands * 2
            else:
                self.logger.debug("here self.poly_bands: \n {}".format(self.poly_bands))
                emesh, dos = get_dos_from_poly_bands(self._vrun.final_structure, self._rec_lattice,
                        [self.nkdos, self.nkdos, self.nkdos], self.dos_emin,
                        self.dos_emax, int(round((self.dos_emax - self.dos_emin) \
                        / max(self.dE_min, 0.0001))),poly_bands=self.poly_bands,
                        bandgap=self.cbm_vbm["n"]["energy"] - self.cbm_vbm["p"][
                        "energy"], width=self.dos_bwidth, SPB_DOS=False)
                self.dos_normalization_factor = len(self.poly_bands) * 2 * 2
                self.dos_start = self.dos_emin
                self.dos_end = self.dos_emax

            self.logger.debug("DOS normalization factor: {}".format(
                                            self.dos_normalization_factor))
            integ = 0.0
            self.dos_start = abs(emesh - self.dos_start).argmin()
            self.dos_end = abs(emesh - self.dos_end).argmin()
            for idos in range(self.dos_start, self.dos_end):
                integ += (dos[idos + 1] + dos[idos]) / 2 * (emesh[idos + 1] - emesh[idos])
            self.logger.debug("dos integral from {} index to {}: {}".format(self.dos_start,  self.dos_end, integ))
            dos = [g / integ * self.dos_normalization_factor for g in dos]
            self.dos = zip(emesh, dos)
            self.dos_emesh = np.array(emesh)
            self.vbm_dos_idx = self.get_Eidx_in_dos(self.cbm_vbm["p"]["energy"])
            self.cbm_dos_idx = self.get_Eidx_in_dos(self.cbm_vbm["n"]["energy"])
            self.dos = [list(a) for a in self.dos]

        if return_energies:
            return kpts, energies_sorted
        else:
            return kpts


    def find_all_important_points(self, coeff_file, nbelow_vbm=0, nabove_cbm=0, interpolation="boltztrap1", ext_height=0.25):
        if interpolation=="boltztrap1":
            ibands = [self.cbm_vbm['p']['bidx']-nbelow_vbm,
                      self.cbm_vbm['n']['bidx']+nabove_cbm]
            self.interp_params = get_energy_args(coeff_file, ibands)
        if self.important_pts is None or nbelow_vbm+nabove_cbm>0:
            if self.poly_bands0 is None:
                eref = {typ: self.cbm_vbm[typ]['energy'] for typ in ['p', 'n']}
            else:
                eref = None
            Ecut = {tp: min(2.0, max(self.Ecut[tp]-ext_height, ext_height)) for tp in ['p', 'n']}
            self.important_pts, new_cbm_vbm = get_bs_extrema(self.bs, coeff_file,
                    interp_params=self.interp_params, method=interpolation,
                    Ecut=Ecut, eref=eref, return_global=True, n_jobs=self.n_jobs,
                    nbelow_vbm= nbelow_vbm, nabove_cbm=nabove_cbm, scissor=self.scissor)
            if new_cbm_vbm['n']['energy'] < self.cbm_vbm['n']['energy'] and self.poly_bands0 is None:
                self.cbm_vbm['n']['energy'] = new_cbm_vbm['n']['energy']
                self.cbm_vbm['n']['kpoint'] = new_cbm_vbm['n']['kpoint']
            if new_cbm_vbm['p']['energy'] > self.cbm_vbm['p']['energy'] and self.poly_bands0 is None:
                self.cbm_vbm['p']['energy'] = new_cbm_vbm['p']['energy']
                self.cbm_vbm['p']['kpoint'] = new_cbm_vbm['p']['kpoint']
        self.all_important_pts = deepcopy(self.important_pts)
        self.logger.info('Here all the initial extrema (valleys):\n{}'.format(
                self.important_pts))


    def write_input_files(self, path=None, dir_name="run_data"):
        """
        Writes all 3 types of inputs in json files for example to
        conveniently track what inputs had been used later or read
        inputs from files (see from_files method)
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
            json.dump(material_params, fp, sort_keys=True, indent=4, ensure_ascii=False, cls=MontyEncoder)
        with open(os.path.join(path, "model_params.json"), "w") as fp:
            json.dump(model_params, fp, sort_keys=True, indent=4, ensure_ascii=False, cls=MontyEncoder)
        with open(os.path.join(path, "performance_params.json"), "w") as fp:
            json.dump(performance_params, fp, sort_keys=True, indent=4, ensure_ascii=False, cls=MontyEncoder)


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
        """
        self.epsilon_s = params["epsilon_s"]
        self.P_PIE = params.get("P_PIE", None) or 0.15  # unitless
        E_D = params.get("E_D", None)
        self.C_el = params.get("C_el", None)
        if (E_D is None or self.C_el is None) and 'ACD' in self.elastic_scats:
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
            self.W_POP *= 1e12 * 2 * pi # convert to THz
        if 'POP' in self.inelastic_scats:
            if self.epsilon_inf is None or self.W_POP is None:
                warnings.warn('POP cannot be calculated w/o epsilon_inf and W_POP')
                self.inelastic_scats.pop(self.inelastic_scats.index('POP'))

        self.N_dis = params.get("N_dis", None) or 0.1  # in 1/cm**2
        self.scissor = params.get("scissor", None) or 0.0
        self.user_bandgap = params.get("user_bandgap", None)

        donor_charge = params.get("donor_charge", 1.0)
        acceptor_charge = params.get("acceptor_charge", 1.0)
        dislocations_charge = params.get("dislocations_charge", 1.0)
        self.charge = {"n": donor_charge, "p": acceptor_charge, "dislocations": dislocations_charge}
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
        self.poly_bands0 = params.get("poly_bands", None)
        self.poly_bands = self.poly_bands0
        self.soc = params.get("soc", False)
        self.logger.info("bs_is_isotropic: {}".format(self.bs_is_isotropic))
        self.independent_valleys = params.get('independent_valleys', False)
        self.model_params = {
            "bs_is_isotropic": self.bs_is_isotropic,
            "elastic_scats": self.elastic_scats,
            "inelastic_scats": self.inelastic_scats,
            "poly_bands": self.poly_bands
        }


    def set_performance_params(self, params=None):
        """
        Set (or retrieve from input parameters) that are related to running
        performance and speed and store them as corresponding instance variables

        Args:
            params (dict): must be at least {} to invoke all the default values
                examples are {} or {'Ecut': 1.0, 'nkdos': 25}

        Returns (None):
        """
        params = params or {}
        self.nkibz = params.get("nkibz", 40)
        self.dE_min = params.get("dE_min", 0.0001)
        self.nE_min = params.get("nE_min", 5)
        c_factor = max(1, 3 * abs(max([log(abs(ci)/float(1e19)) for ci in self.dopings]))**0.25)
        Ecut = params.get("Ecut", c_factor * 5 * k_B * max(self.temperatures + [300]))
        self.Ecut_max = params.get("Ecut", 1.5) #TODO-AF: set this default Encut based on maximum energy range that the current BS covers between
        Ecut = min(Ecut, self.Ecut_max)
        self.Ecut = {tp: Ecut if tp in self.all_types else Ecut/2.0 for tp in ["n", "p"]}
        for tp in ["n", "p"]:
            self.logger.debug("{}-Ecut: {} eV \n".format(tp, self.Ecut[tp]))
        self.dos_bwidth = params.get("dos_bwidth", 0.1)
        self.nkdos = params.get("nkdos", 29)
        self.v_min = 1000
        self.gs = 1e-32  # small value (e.g. used for an initial non-zero val)
        self.gl = 1e32  # global large value
        self.BTE_iters = params.get("BTE_iters", 5)
        self.n_jobs = params.get("n_jobs", -1)
        if self.n_jobs == -1:
            self.n_jobs = cpu_count()
        self.max_nbands = params.get("max_nbands", None)
        self.max_normk0 = params.get("max_normk", None)
        self.max_normk = {'n': self.max_normk0, 'p': self.max_normk0}
        self.max_nvalleys = params.get("max_nvalleys", None)
        self.interpolation = params.get("interpolation", "boltztrap1")
        if self.interpolation == "boltztrap2":
            try:
                import BoltzTraP2
            except ImportError:
                self.logger.error('Failed to import BoltzTraP2! '
                                  '"boltztrap2" interpolation not available.')
        self.performance_params = {
            "nkibz": self.nkibz,
            "dE_min": self.dE_min,
            "Ecut": self.Ecut,
            "Ecut_max": self.Ecut_max,
            "dos_bwidth": self.dos_bwidth,
            "nkdos": self.nkdos,
            "BTE_iters": self.BTE_iters,
            "max_nbands": self.max_nbands,
            "max_normk0": self.max_normk0,
            "max_nvalleys": self.max_nvalleys,
            "n_jobs": self.n_jobs,
            "interpolation": self.interpolation
        }


    def __getitem__(self, key):
        if key == "kgrid":
            return self.kgrid
        elif key == "egrid":
            return self.egrid
        else:
            raise KeyError


    def read_vrun(self, vasprun_file=None, filename="vasprun.xml"):
        """
        Reads the vasprun file and populates some instance variables such as
        volume, interp_params, _vrun, etc to be used by other methods.

        Args:
            vasprun_file (str): full path to the vasprun file. If provided,
                the filename will be ignored.
            filename (str): if vasprun_file is None, this filename will be
                looked for inside calc_dir to be used as the vasprun file

        Returns (None):
        """
        vasprun_file = vasprun_file or os.path.join(self.calc_dir, filename)
        self._vrun = Vasprun(vasprun_file, parse_projected_eigen=True)
        self.interp_params = None
        if self.interpolation == "boltztrap2":
            bz2_data = PymatgenLoader(self._vrun)
            equivalences = sphere.get_equivalences(bz2_data.atoms,
                                                   len(bz2_data.kpoints) * 5)
            lattvec = bz2_data.get_lattvec()
            coeffs = fite.fitde3D(bz2_data, equivalences)
            self.interp_params = (equivalences, lattvec, coeffs)
        self.volume = self._vrun.final_structure.volume
        self.logger.info("unitcell volume = {} A**3".format(self.volume))
        self.density = self._vrun.final_structure.density
        self._rec_lattice = self._vrun.final_structure.lattice.reciprocal_lattice

        sg = SpacegroupAnalyzer(self._vrun.final_structure)
        self.rotations, _ = sg._get_symmetry()
        self.bs = self._vrun.get_band_structure()
        self.bs.structure = self._vrun.final_structure
        self.nbands = self.bs.nb_bands
        self.lorbit = 11 if len(sum(self._vrun.projected_eigenvalues[Spin.up][0][10])) > 5 else 10

        self.DFT_cartesian_kpts = np.array([self.get_cartesian_coords(k) for k in self._vrun.actual_kpoints])/ A_to_nm

        cbm_vbm = {"n": {"kpoint": [], "energy": 0.0, "bidx": 0, "included": 0, "eff_mass_xx": [0.0, 0.0, 0.0]},
                   "p": {"kpoint": [], "energy": 0.0, "bidx": 0, "included": 0, "eff_mass_xx": [0.0, 0.0, 0.0]}}
        cbm = self.bs.get_cbm()
        vbm = self.bs.get_vbm()
        self.logger.info("total number of bands: {}".format(self._vrun.get_band_structure().nb_bands))
        cbm_vbm["n"]["energy"] = cbm["energy"]
        cbm_vbm["n"]["bidx"], _ = get_bindex_bspin(cbm, is_cbm=True)
        cbm_vbm["n"]["kpoint"] = self.bs.kpoints[cbm["kpoint_index"][0]].frac_coords
        cbm_vbm["p"]["energy"] = vbm["energy"]
        cbm_vbm["p"]["bidx"], _ = get_bindex_bspin(vbm, is_cbm=False)
        cbm_vbm["p"]["kpoint"] = self.bs.kpoints[vbm["kpoint_index"][0]].frac_coords

        self.dft_gap = cbm["energy"] - vbm["energy"]
        self.logger.debug("DFT gap from vasprun.xml : {} eV".format(self.dft_gap))
        if self.user_bandgap:
            if self.scissor != 0.0:
                self.logger.warning('"user_bandgap" is set hence previously set '
                    '"scissor" is ignored. Continuing with scissor={}'.format(
                        self.user_bandgap - self.dft_gap))
            self.scissor = self.user_bandgap - self.dft_gap
        if self.soc:
            self.nelec = cbm_vbm["p"]["bidx"] + 1
        else:
            self.nelec = (cbm_vbm["p"]["bidx"] + 1) * 2
        self.logger.debug("total number of electrons nelec: {}".format(self.nelec))
        bsd = self.bs.as_dict()
        if bsd["is_spin_polarized"]:
            self.dos_emin = min(min(bsd["bands"]["1"][0]), min(bsd["bands"]["-1"][0]))
            self.dos_emax = max(max(bsd["bands"]["1"][-1]), max(bsd["bands"]["-1"][-1]))
        else:
            self.dos_emin = min(bsd["bands"]["1"][0])
            self.dos_emax = max(bsd["bands"]["1"][-1])
        self.initial_num_bands = {'n': None, 'p': None}
        if self.poly_bands0 is None:
            for i, tp in enumerate(["n", "p"]):
                Ecut = self.Ecut[tp]
                sgn = (-1) ** i
                while abs(min(sgn * np.array(bsd["bands"]["1"][cbm_vbm[tp]["bidx"] + sgn * cbm_vbm[tp]["included"]])) -
                                          sgn * cbm_vbm[tp]["energy"]) < Ecut:
                    cbm_vbm[tp]["included"] += 1
                self.initial_num_bands[tp] = cbm_vbm[tp]["included"]
        else:
            cbm_vbm["n"]["included"] = cbm_vbm["p"]["included"] = len(self.poly_bands0)
            self.initial_num_bands['n'] = self.initial_num_bands['p'] = len(self.poly_bands0)
        cbm_vbm["p"]["bidx"] += 1
        cbm_vbm["n"]["bidx"] = cbm_vbm["p"]["bidx"] + 1
        self.cbm_vbm = cbm_vbm
        self.cbm_vbm0 = deepcopy(cbm_vbm)
        self.valleys = {tp: {'band {}'.format(i): OrderedDict() for i in range(self.cbm_vbm0[tp]['included']) } for tp in ['p', 'n']}
        self.logger.info("original cbm_vbm:\n {}".format(cbm_vbm))
        self.num_bands = {tp: self.cbm_vbm[tp]["included"] for tp in ['n', 'p']}


    def get_cartesian_coords(self, frac_k, reciprocal=True):
        """
        Transformation from fractional too cartesian. Note that this is different
        form get_cartesian_coords method available in self._rec_lattice, that
        one does NOT work with BolzTraP outputs

        Args:
            frac_k (np.ndarray): a 3-D vector in fractional (unitless)
            coordinates or a list of such coordinates
            reciprocal (bool): whether the cartesian output is in real (Angstrom)
                or reciprocal space (1/Angstrom).

        Returns (np.ndarray): frac_k ransformed into cartesian coordinates
        """
        if reciprocal:
            return np.dot(self._rec_lattice.matrix, np.array(frac_k))
        else:
            return np.dot(self._vrun.lattice.matrix, np.array(frac_k))


    def seeb_int_num(self, c, T):
        """
        Wrapper function to do an integration taking only the concentration,
        c, and the temperature, T, as inputs
        """
        fn = lambda E, fermi, T: f0(E,fermi,T) * (1-f0(E,fermi,T)) * E/(k_B*T)
        return {t: self.integrate_func_over_E(func=fn,
                                              tp=t,
                                              fermi=self.fermi_level[c][T],
                                              T=T,
                                              normalize_energy=True,
                                              xDOS=False) \
                for t in ["n", "p"]}


    def seeb_int_denom(self, c, T):
        """
        Wrapper function to do an integration taking only the concentration,
        c, and the temperature, T, as inputs
        """
        return {t: self.gs + self.integrate_over_E(prop_list=["f0x1-f0"],
                                                   tp=t, c=c, T=T, xDOS=False)\
                for t in["n", "p"]}


    def calculate_property(self, prop_name, prop_func, for_all_E=False):
        """
        Calculates the propery at all concentrations and temperatures using
        the given function and insert it into self.egrid

        Args:
            prop_name (str): the name of the property
            prop_func (obj): the given function MUST takes c and T as required inputs in this order.
        """
        if for_all_E:
            for tp in ["n", "p"]:
                self.egrid[tp][prop_name] = {
                c: {T: [self.gs for E in self.egrid[tp]["energy"]] for T in self.temperatures}
                for c in self.dopings}
        else:
            self.egrid[prop_name] = {c: {T: self.gs for T in self.temperatures} for c in self.dopings}
        for c in self.dopings:
            for T in self.temperatures:
                if for_all_E:
                    fermi = self.fermi_level[c][T]
                    for tp in ["n", "p"]:
                        for ie, E in enumerate(self.egrid[tp]["energy"]):
                            self.egrid[tp][prop_name][c][T][ie] = prop_func(E, fermi, T)
                else:
                    self.egrid[prop_name][c][T] = prop_func(c, T)


    def calculate_N_II(self, c, T):
        """
        Args:
            c (float): the carrier concentration
            T (float): the temperature in kelvin

        Returns (float): inoized impurity (IMP) scattering concentration (N_II)
        """
        N_II = abs(self.calc_doping[c][T]["n"]) * self.charge["n"] ** 2 + \
               abs(self.calc_doping[c][T]["p"]) * self.charge["p"] ** 2 + \
               self.N_dis / self.volume ** (1 / 3) * 1e8 * self.charge["dislocations"] ** 2
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
                  "all_ks_flat": [], "mobility": {}},
            "p": {"energy": [], "DOS": [], "all_en_flat": [],
                  "all_ks_flat": [], "mobility": {}},
        }
        self.kgrid_to_egrid_idx = {"n": [], "p": []}
        self.Efrequency = {"n": [], "p": []}
        self.sym_freq = {"n": [], "p":[]}
        E_idx = {"n": [], "p": []}
        for tp in ["n", "p"]:
            for ib, en_vec in enumerate(self.kgrid[tp]["energy"]):
                self.egrid[tp]["all_en_flat"] += list(en_vec)
                self.egrid[tp]["all_ks_flat"] += list(self.kgrid[tp]["kpoints"][ib])
                E_idx[tp] += [(ib, iek) for iek in range(len(en_vec))]
            ieidxs = np.argsort(self.egrid[tp]["all_en_flat"])
            self.egrid[tp]["all_en_flat"] = [self.egrid[tp]["all_en_flat"][ie] for ie in ieidxs]
            self.egrid[tp]["all_ks_flat"] = [self.egrid[tp]["all_ks_flat"][ie] for ie in ieidxs]
            E_idx[tp] = [E_idx[tp][ie] for ie in ieidxs]

        # setting up energy grid and DOS:
        for tp in ["n", "p"]:
            energy_counter = []
            i = 0
            last_is_counted = False
            while i < len(self.egrid[tp]["all_en_flat"]) - 1:
                sum_E = self.egrid[tp]["all_en_flat"][i]
                sum_nksym = len(remove_duplicate_kpoints(self.bs.get_sym_eq_kpoints(self.egrid[tp]["all_ks_flat"][i])))
                counter = 1.0  # because the ith member is already included in sum_E
                current_ib_ie_idx = [E_idx[tp][i]]
                j = i
                while j < len(self.egrid[tp]["all_en_flat"]) - 1 and \
                        abs(self.egrid[tp]["all_en_flat"][i] - self.egrid[tp]["all_en_flat"][j + 1]) < self.dE_min:
                    counter += 1
                    current_ib_ie_idx.append(E_idx[tp][j + 1])
                    sum_E += self.egrid[tp]["all_en_flat"][j + 1]
                    sum_nksym += len(remove_duplicate_kpoints(self.bs.get_sym_eq_kpoints(self.egrid[tp]["all_ks_flat"][i+1])))

                    if j + 1 == len(self.egrid[tp]["all_en_flat"]) - 1:
                        last_is_counted = True
                    j += 1
                self.egrid[tp]["energy"].append(sum_E / counter)
                self.kgrid_to_egrid_idx[tp].append(current_ib_ie_idx)
                self.sym_freq[tp].append(sum_nksym / counter)
                energy_counter.append(counter)
                self.egrid[tp]["DOS"].append(self.dos[self.get_Eidx_in_dos(sum_E / counter)][1])
                i = j + 1

            if not last_is_counted:
                self.egrid[tp]["energy"].append(self.egrid[tp]["all_en_flat"][-1])
                self.kgrid_to_egrid_idx[tp].append([E_idx[tp][-1]])
                self.egrid[tp]["DOS"].append(self.dos[self.get_Eidx_in_dos(self.egrid[tp]["energy"][-1])][1])
            self.egrid[tp]["size"] = len(self.egrid[tp]["energy"])
            self.Efrequency[tp] = [len(Es) for Es in self.kgrid_to_egrid_idx[tp]]
            if len(self.Efrequency[tp]) < min_nE:
                warnings.warn("The final {}-egrid have fewer than {} energy values".format(tp, min_nE))
                corrupt_tps.append(tp)
        # if len(self.Efrequency["n"]) < min_nE or len(self.Efrequency["p"]) < min_nE:
            # raise ValueError("The final egrid have fewer than {} energy values, Amset stops now".format(min_nE))
        return corrupt_tps


    def init_egrid(self, once_called):
        """
        Initializes the self.egrid dict containing energy grid and relevant
        properties such as "DOS". This must be called after pre_init_egrid so
        that the energy values are already populated.

        Args:
            once_called (bool): whether init_egrid was called once before or
                not. It is used internally for caching.

        Returns ([str]): keep track of which types failed; "n" or "p" type or
            neither. Cause of failure could be that too few k-points are left
            for that type.
        """
        corrupt_tps = self.pre_init_egrid()
        if "n" in corrupt_tps and "p" in corrupt_tps:
            return corrupt_tps
        if not once_called:
            self.egrid["calc_doping"] = {c: {T: {"n": 0.0, "p": 0.0} for T in self.temperatures} for c in self.dopings}
            for transport in ["conductivity", "J_th", "seebeck", "TE_power_factor", "relaxation time constant"]:
                for tp in ['n', 'p']:
                    self.egrid[tp][transport] = {c: {T: 0.0 for T in\
                            self.temperatures} for c in self.dopings}

            # populate the egrid at all c and T with properties; they can be called via self.egrid[prop_name][c][T] later
            if self.integration == 'k':
                self.egrid["calc_doping"] = self.calc_doping
            elif self.integration == 'e':
                self.egrid["calc_doping"] = self.calc_doping
                self.egrid["fermi"] = self.fermi_level

        self.calculate_property(prop_name="f0", prop_func=f0, for_all_E=True)
        self.calculate_property(prop_name="f", prop_func=f0, for_all_E=True)
        self.calculate_property(prop_name="f_th", prop_func=f0, for_all_E=True)

        for prop in ["f", "f_th"]:
            self.map_to_egrid(prop_name=prop, c_and_T_idx=True)

        self.calculate_property(prop_name="f0x1-f0", prop_func=lambda E, fermi,
                T: f0(E, fermi, T) * (1 - f0(E, fermi, T)), for_all_E=True)
        for c in self.dopings:
            for T in self.temperatures:
                fermi = self.fermi_level[c][T]
                for tp in ["n", "p"]:
                    for ib in range(len(self.kgrid[tp]["energy"])):
                        for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                            E = self.kgrid[tp]["energy"][ib][ik]
                            self.kgrid[tp]["f0"][c][T][ib][ik] = f0(E,fermi,T)

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
        return int(round((E - self.dos_emin) / Estep))


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
                self.logger.debug("# of {}-type kpoints indexes with low velocity or off-energy: {}".format(tp,len(rm_idx_list_ib)))
            for prop in rearranged_props:
                self.kgrid[tp][prop] = np.array([np.delete(self.kgrid[tp][prop][ib], rm_idx_list[tp][ib], axis=0) \
                                                 for ib in range(self.cbm_vbm[tp]["included"])])


    def initialize_var(self, grid, names, val_type="scalar", initval=0.0, is_nparray=True, c_T_idx=False):
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
                    init_content = [[initial_val for _ in range(len(self[grid][tp]["kpoints"][j]))]
                                    for j in range(self.cbm_vbm[tp]["included"])]
                elif grid == "egrid":
                    init_content = [initial_val for _ in self[grid][tp]["energy"]]
                else:
                    raise TypeError('The argument "grid" must be set to either "kgrid" or "egrid"')
                if is_nparray:
                    if not c_T_idx:
                        self[grid][tp][name] = np.array(init_content)
                    else:
                        self[grid][tp][name] = {c: {T: np.array(init_content) for T in self.temperatures} for c in
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
                            self[grid][tp][name] = {c: {T: init_content for T in self.temperatures} for c in
                                                    self.dopings}


    def get_intermediate_kpoints(self, k1, k2, nsteps):
        """
        Returns a list nsteps number of k-points between k1 & k2 excluding
        k1 & k2 themselves. k1 & k2 are nparray
        """
        dkii = (k2 - k1) / float(nsteps + 1)
        return [k1 + i * dkii for i in range(1, nsteps + 1)]


    def get_intermediate_kpoints_list(self, k1, k2, nsteps):
        """
        Returns a list nsteps number of k-points between k1 & k2 excluding
        k1 & k2 themselves. k1 & k2 are lists
        """
        if nsteps < 1:
            return []
        dk = [(k2[i] - k1[i]) / float(nsteps + 1) for i in range(len(k1))]
        return [[k1[i] + n * dk[i] for i in range(len(k1))] for n in range(1, nsteps + 1)]


    def calc_poly_energy(self, xkpt, tp, ib):
        """
        Calculates parabolic or other polynomial bands at given k-point & band

        Args:
            xkpt (3x1 array or list): fractional coordinates of a given k-point
            tp (str): 'n' or 'p' type
            ib (int): the band index

        Returns:
            (energy(eV), velocity (cm/s), effective mass) from a parabolic band
        """
        energy, velocity, effective_m = get_poly_energy(
            self.get_cartesian_coords(xkpt)/A_to_nm, poly_bands=self.poly_bands,
            type=tp, ib=ib, bandgap=self.dft_gap + self.scissor)
        return energy, velocity, effective_m


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
        # TODO: remove anything with "weight" later if ended up not using weights at all!
        kweights = {tp: [1.0 for i in kpts[tp]] for tp in ["n", "p"]}

        # actual initiation of the kgrid
        self.kgrid = {
            "n": {},
            "p": {}}
        self.num_bands = {"n": 1, "p": 1}
        # self.logger.debug('here the n-type kgrid :\n{}'.format(kpts['n']))
        for tp in ["n", "p"]:
            self.num_bands[tp] = self.cbm_vbm[tp]["included"]
            self.kgrid[tp]["kpoints"] = [kpts[tp] for ib in range(self.num_bands[tp])]
            self.kgrid[tp]["kweights"] = [kweights[tp] for ib in range(self.num_bands[tp])]

        self.initialize_var("kgrid", ["energy", "a", "c", "norm(v)", "norm(k)"], "scalar", 0.0, is_nparray=False)
        self.initialize_var("kgrid", ["velocity"], "vector", 0.0, is_nparray=False)
        self.velocity_signed = {tp: np.array([[[0,0,0] for ik in range(len(kpts[tp]))] for ib in range(self.num_bands[tp])]) for tp in ['n', 'p']}
        self.initialize_var("kgrid", ["effective mass"], "tensor", 0.0, is_nparray=False)

        start_time = time.time()
        rm_idx_list = {"n": [[] for i in range(self.cbm_vbm["n"]["included"])],
                       "p": [[] for i in range(self.cbm_vbm["p"]["included"])]}
        self.initialize_var("kgrid", ["old cartesian kpoints", "cartesian kpoints"], "vector", 0.0, is_nparray=False, c_T_idx=False)
        self.initialize_var("kgrid", ["norm(k)", "norm(actual_k)"], "scalar", 0.0, is_nparray=False, c_T_idx=False)
        self.logger.debug("The DFT gap right before calculating final energy values: {}".format(self.dft_gap))

        for i, tp in enumerate(["p", "n"]):
            self.cbm_vbm[tp]["cartesian k"] = self.get_cartesian_coords(self.cbm_vbm[tp]["kpoint"])/A_to_nm
            self.cbm_vbm[tp]["all cartesian k"] = remove_duplicate_kpoints(
                self.bs.get_sym_eq_kpoints(self.cbm_vbm[tp]["cartesian k"],
                                           cartesian=True))
            sgn = (-1) ** i
            for ib in range(self.cbm_vbm[tp]["included"]):
                for ik, k in enumerate(self.kgrid[tp]['kpoints'][ib]):
                    self.kgrid[tp]["old cartesian kpoints"][ib][ik] = \
                        self.get_cartesian_coords(self.kgrid[tp]["kpoints"][ib][ik]) / A_to_nm

                s_orbital, p_orbital = get_dft_orbitals(
                    vasprun=self._vrun,
                    bidx=self.cbm_vbm[tp]["bidx"] - 1 - sgn * ib,
                    lorbit=self.lorbit)
                orbitals = {"s": s_orbital, "p": p_orbital}
                fit_orbs = {orb: griddata(points=np.array(self.DFT_cartesian_kpts), values=np.array(orbitals[orb]),
                    xi=np.array(self.kgrid[tp]["old cartesian kpoints"][ib]), method='nearest') for orb in orbitals.keys()}

                iband = i * self.cbm_vbm["p"]["included"] + ib if self.interpolation=="boltztrap1" else self.cbm_vbm[tp]["bidx"] + (i-1)*self.cbm_vbm["p"]["included"]+ib
                self.kgrid[tp]["energy"][ib], \
                        self.kgrid[tp]["velocity"][ib], \
                        self.kgrid[tp]["effective mass"][ib] = \
                    interpolate_bs(self.kgrid[tp]["kpoints"][ib], self.interp_params, iband=iband, sgn=sgn, method=self.interpolation, scissor=self.scissor, matrix=self._vrun.lattice.matrix, n_jobs=self.n_jobs)

                self.kgrid[tp]["cartesian kpoints"][ib] = np.array(
                    self.kgrid[tp]["old cartesian kpoints"][ib]) # made a copy
                for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                    self.kgrid[tp]["cartesian kpoints"][ib][ik] = self.get_cartesian_coords(get_closest_k(
                            self.kgrid[tp]["kpoints"][ib][ik], self.bs.get_sym_eq_kpoints(important_points[tp][0]), return_diff=True)) / A_to_nm
                    self.kgrid[tp]["norm(k)"][ib][ik] = norm(self.kgrid[tp]["cartesian kpoints"][ib][ik])
                    self.kgrid[tp]["norm(actual_k)"][ib][ik] = norm(self.kgrid[tp]["old cartesian kpoints"][ib][ik])
                    if self.poly_bands is not None:
                        self.kgrid[tp]["energy"][ib][ik], \
                                self.kgrid[tp]["velocity"][ib][ik], \
                                self.kgrid[tp]["effective mass"][ib][ik] = \
                                get_poly_energy(
                                    self.kgrid[tp]["cartesian kpoints"][ib][ik],
                                   poly_bands=self.poly_bands, type=tp, ib=ib,
                                        bandgap=self.dft_gap + self.scissor)

                    self.kgrid[tp]["norm(v)"][ib][ik] = norm(self.kgrid[tp]["velocity"][ib][ik])
                    if (len(rm_idx_list[tp][ib]) + 20 < len(self.kgrid[tp]['kpoints'][ib])) and (
                            (self.kgrid[tp]["velocity"][ib][ik] < self.v_min).all()
                            # if all members are small, that point should be removed otherwise scattering blows up and I get nan mobilities
                        or \
                            (abs(self.kgrid[tp]["energy"][ib][ik] - self.cbm_vbm[tp]["energy"]) > self.Ecut[tp]) \
                        or \
                            ((self.max_normk[tp]) and (self.kgrid[tp]["norm(k)"][ib][ik] > self.max_normk[tp]) and (self.poly_bands0 is None))
                        or \
                            (self.kgrid[tp]["norm(k)"][ib][ik] < 1e-3)
                    ):
                        rm_idx_list[tp][ib].append(ik)
                    if self.poly_bands is None:
                        self.kgrid[tp]["a"][ib][ik] = fit_orbs["s"][ik]/ (fit_orbs["s"][ik]**2 + fit_orbs["p"][ik]**2)**0.5
                        self.kgrid[tp]["c"][ib][ik] = (1 - self.kgrid[tp]["a"][ib][ik]**2)**0.5
                    else:
                        self.kgrid[tp]["a"][ib][ik] = 1.0  # parabolic: s-only
                        self.kgrid[tp]["c"][ib][ik] = 0.0
            self.logger.debug("average of the {}-type group velocity in kgrid:\n {}".format(
                        tp, np.mean(self.kgrid[tp]["velocity"][0], axis=0)))

        rearranged_props = ["velocity",  "effective mass", "energy", "a", "c",
                            "kpoints", "cartesian kpoints",
                            "old cartesian kpoints", "kweights",
                            "norm(v)", "norm(k)", "norm(actual_k)"]

        self.logger.debug("time to calculate E, v, m_eff at all k-points: \n {}".format(time.time()-start_time))
        start_time = time.time()

        for tp in ["n", "p"]:
            rm_idx_list[tp] = [rm_idx_list[tp][0] for _ in range(self.cbm_vbm[tp]["included"])]
        self.rm_idx_list = deepcopy(rm_idx_list)   # format: [tp][ib][ik]
        if delete_off_points:
            self.remove_indexes(rm_idx_list, rearranged_props=rearranged_props)
        for tp in ['p', 'n']:
            self.logger.debug(
                "average of the {}-type group velocity in kgrid after removing points:\n {}".format(
                    tp, np.mean(self.kgrid[tp]["velocity"][0], axis=0)))
        self.logger.debug("dos_emin = {} and dos_emax= {}".format(self.dos_emin, self.dos_emax))

        self.logger.debug('current cbm_vbm:\n{}'.format(self.cbm_vbm))
        for tp in ["n", "p"]:
            for ib in range(len(self.kgrid[tp]["energy"])):
                self.logger.info("Final # of {}-kpts in band #{}: {}".format(tp, ib, len(self.kgrid[tp]["kpoints"][ib])))

            if len(self.kgrid[tp]["kpoints"][0]) < 5:
                corrupt_tps.append(tp)
        self.logger.debug("time to calculate energy, velocity, m* for all: {} seconds".format(time.time() - start_time))

        # sort "energy", "kpoints", "kweights", etc based on energy in ascending order and keep track of old indexes
        e_sort_idx_2 = self.sort_vars_based_on_energy(args=rearranged_props, ascending=True)
        self.pos_idx_2 = deepcopy(e_sort_idx_2)
        for tp in ['n', 'p']:
            for ib in range(self.num_bands[tp]):
                self.pos_idx_2[tp][ib] = np.array(range(len(e_sort_idx_2[tp][ib])))[e_sort_idx_2[tp][ib]].argsort()

        # to save memory avoiding storage of variables that we don't need down the line
        for tp in ["n", "p"]:
            self.kgrid[tp].pop("effective mass", None)
            self.kgrid[tp].pop("kweights", None)
            self.kgrid[tp]["size"] = [len(self.kgrid[tp]["kpoints"][ib]) \
                                      for ib in range(len(self.kgrid[tp]["kpoints"]))]
        self.initialize_var("kgrid", ["W_POP"], "scalar", 0.0, is_nparray=True, c_T_idx=False)
        self.initialize_var("kgrid", ["N_POP"], "scalar", 0.0, is_nparray=True, c_T_idx=True)

        for tp in ["n", "p"]:
            for ib in range(self.cbm_vbm[tp]["included"]):
                # We define W_POP in the grid this way for future W_POP(k)
                self.kgrid[tp]["W_POP"][ib] = \
                    [self.W_POP]*len(self.kgrid[tp]["kpoints"][ib])
                for c in self.dopings:
                    for T in self.temperatures:
                        self.kgrid[tp]["N_POP"][c][T][ib] = np.array(
                            [1 / (np.exp(hbar * W_POP / (k_B * T)) - 1) for W_POP in self.kgrid[tp]["W_POP"][ib]])

        self.initialize_var(grid="kgrid", names=[
                "_all_elastic", "S_i", "S_i_th", "S_o", "S_o_th", "g", "g_th",
                "g_POP", "f", "f_th", "relaxation time", "df0dk",
                "electric force","thermal force"], val_type="vector",
                            initval=self.gs, is_nparray=True, c_T_idx=True)

        self.initialize_var("kgrid", ["f0", "f_plus", "f_minus", "g_plus", "g_minus"], "vector", self.gs,
                            is_nparray=True, c_T_idx=True)
        return corrupt_tps


    def sort_vars_based_on_energy(self, args, ascending=True):
        """
        Sorts the list of variables specified by "args" (type: [str]) in self.kgrid based on the "energy" values
        in each band for both "n"- and "p"-type bands and in ascending order by default.
        """
        ikidxs = {'n': {ib: [] for ib in range(self.num_bands['n'])}, 'p': {ib: [] for ib in range(self.num_bands['p'])}}
        for tp in ["n", "p"]:
            for ib in range(self.cbm_vbm[tp]["included"]):
                ikidxs[tp][ib] = np.argsort(self.kgrid[tp]["energy"][ib])
                if not ascending:
                    ikidxs[tp][ib].reverse()
                for arg in args:
                    self.kgrid[tp][arg][ib] = np.array([self.kgrid[tp][arg][ib][ik] for ik in ikidxs[tp][ib]])
        return ikidxs


    def generate_angles_and_indexes_for_integration(self, avg_Ediff_tolerance=0.02):
        """
        Generates the indexes of k' points that have the same energy (for elastic scattering) as E(k) or
        have energy equal to E(k) plus or minus of the energy of the optical phonon for inelastic scattering.
        Also, generated and stored the cosine of the angles between such points and a given input k-point

        Args:
            avg_Ediff_tolerance (float): in eV the average allowed energy difference between the target E(k') and
                what it actially is (e.g. to prevent/identify large energy differences if enforced scattering)
        """
        self.initialize_var("kgrid", ["X_E_ik", "X_Eplus_ik", "X_Eminus_ik"], "scalar", [], is_nparray=False,
                            c_T_idx=False)

        # elastic scattering
        for tp in ["n", "p"]:
            for ib in range(len(self.kgrid[tp]["energy"])):
                self.nforced_scat = {"n": 0.0, "p": 0.0}
                self.ediff_scat = {"n": [], "p": []}
                for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                    self.kgrid[tp]["X_E_ik"][ib][ik] = self.get_X_ib_ik_near_new_E(tp, ib, ik,
                            E_change=0.0, forced_min_npoints=self.nE_min, tolerance=self.dE_min)
                enforced_ratio = self.nforced_scat[tp] / sum([len(points) for points in self.kgrid[tp]["X_E_ik"][ib]])
                self.logger.info("enforced scattering ratio for {}-type elastic scattering at band {}:\n {}".format(
                        tp, ib, enforced_ratio))
                if enforced_ratio > 0.9:
                    warnings.warn("the k-grid is too coarse for an acceptable simulation of elastic scattering in {};"
                        .format(self.tp_title[tp]))

                avg_Ediff = sum(self.ediff_scat[tp]) / max(len(self.ediff_scat[tp]), 1)
                if avg_Ediff > avg_Ediff_tolerance:
                    warnings.warn("{}-type average energy difference of the enforced scattered k-points is more than"
                                  " {}, try running with a more dense k-point mesh".format(tp, avg_Ediff_tolerance))

        # inelastic scattering
        if "POP" in self.inelastic_scats:
            for tp in ["n", "p"]:
                sgn = (-1) ** (["n", "p"].index(tp))
                for ib in range(len(self.kgrid[tp]["energy"])):
                    self.nforced_scat = {"n": 0.0, "p": 0.0}
                    self.ediff_scat = {"n": [], "p": []}
                    for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                        self.kgrid[tp]["X_Eplus_ik"][ib][ik] = self.get_X_ib_ik_near_new_E(tp, ib, ik,
                                E_change= + hbar * self.kgrid[tp]["W_POP"][ib][ik],forced_min_npoints=self.nE_min, tolerance=None)
                        self.kgrid[tp]["X_Eminus_ik"][ib][ik] = self.get_X_ib_ik_near_new_E(tp, ib, ik,
                                E_change= - hbar * self.kgrid[tp]["W_POP"][ib][ik],forced_min_npoints=self.nE_min, tolerance=None)
                    enforced_ratio = self.nforced_scat[tp] / (
                        sum([len(points) for points in self.kgrid[tp]["X_Eplus_ik"][ib]]) + \
                        sum([len(points) for points in self.kgrid[tp]["X_Eminus_ik"][ib]]))
                    self.logger.info(
                        "enforced scattering ratio: {}-type inelastic at band {}:\n{}".format(tp, ib, enforced_ratio))

                    if enforced_ratio > 0.9:
                        # TODO: this should be an exception but for now I turned to warning for testing.
                        warnings.warn(
                            "the k-grid is too coarse for an acceptable simulation of POP scattering in {};"
                            " you can try this k-point grid but without POP as an inelastic scattering.".format(
                                self.tp_title[tp]))

                    avg_Ediff = sum(self.ediff_scat[tp]) / max(len(self.ediff_scat[tp]), 1)
                    if avg_Ediff > avg_Ediff_tolerance:
                        # TODO: this should be an exception but for now I turned to warning for testing.
                        warnings.warn(
                            "{}-type average energy difference of the enforced scattered k-points is more than"
                            " {}, try running with a more dense k-point mesh".format(tp, avg_Ediff_tolerance))


    def unique_X_ib_ik_symmetrically_equivalent(self, tp, ib, ik):
        frac_k = self.kgrid[tp]["kpoints"][ib][ik]
        fractional_ks = np.dot(frac_k, self.rotations)
        k = self.kgrid[tp]["kpoints"][ib][ik]
        seks = [self.get_cartesian_coords(frac_k) / A_to_nm for frac_k in fractional_ks]

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


    def get_X_ib_ik_near_new_E(self, tp, ib, ik, E_change, forced_min_npoints=0, tolerance=None):
        """
        Returns the sorted (based on angle, X) list of angle and band and
        k-point indexes of all the points that are within tolerance of E + E_change
        Attention!!! this function assumes self.kgrid is sorted based on the energy in ascending order.

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
            ik_closest_E = np.abs(self.kgrid[tp]["energy"][ib_prm] - E_prm).argmin()

            for step, start in [(1, 0), (-1, -1)]:
                ik_prm = ik_closest_E + start  # go up from ik_closest_E, down from ik_closest_E - 1
                while ik_prm >= 0 and ik_prm < nk and abs(self.kgrid[tp]["energy"][ib_prm][ik_prm] - E_prm) < tolerance:
                    k_prm = self.kgrid[tp]["cartesian kpoints"][ib_prm][ik_prm]
                    X_ib_ik = (cos_angle(k, k_prm), ib_prm, ik_prm)
                    if norm(self.kgrid[tp]["old cartesian kpoints"][ib_prm][ik_prm] - self.kgrid[tp]["old cartesian kpoints"][ib][ik]) < 2*self.max_normk[tp]:
                        result.append(X_ib_ik)
                    ik_prm += step

        if E_change != 0.0:
            ib_prm = ib
            ik_closest_E = np.abs(self.kgrid[tp]["energy"][ib_prm] - E_prm).argmin()

            for step, start in [(1, 0), (-1, -1)]:
                # step -1 is in case we reached the end (ik_prm == nk - 1) when
                #  we choose from the lower energy k-points
                ik_prm = ik_closest_E + start  # go up from ik_closest_E
                while ik_prm >= 0 and ik_prm < nk and len(result) - 1 < forced_min_npoints:
                    # add all the k-points that have the same energy as E_prime E(k_pm); these values are stored in X_E_ik
                    for X_ib_ik in self.kgrid[tp]["X_E_ik"][ib_prm][ik_prm]:
                        X, ib_pmpm, ik_pmpm = X_ib_ik
                        X_ib_ik_new = (
                        cos_angle(k, self.kgrid[tp]["cartesian kpoints"][ib_pmpm][ik_pmpm]), ib_pmpm, ik_pmpm)
                        if (X_ib_ik_new[1], X_ib_ik_new[2]) not in [(entry[1], entry[2]) for entry in result]:
                            result.append(X_ib_ik_new)
                        self.nforced_scat[tp] += 1

                    self.ediff_scat[tp].append(
                        self.kgrid[tp]["energy"][ib][ik] - self.kgrid[tp]["energy"][ib_prm][ik_prm])
                    ik_prm += step

        result.sort(key=lambda x: x[0])
        return result


    def s_el_eq(self, sname, tp, c, T, k, k_prm):
        """
        Returns the scattering rate at wave vector k at a certain concentration and temperature
        for a specific elastic scattering mechanisms determined by sname

        Args:
            sname (string): abbreviation of the name of the elastic scatteirng mechanisms; options: IMP, ADE, PIE, DIS
            c (float): carrier concentration
            T (float): the temperature
            k (list): list containing fractional coordinates of the k vector
            k_prm (list): list containing fractional coordinates of the k prime vector
        """
        norm_diff_k = norm(k - k_prm)  # the slope for PIE and IMP don't match with bs_is_isotropic
        if norm_diff_k == 0.0:
            warnings.warn("WARNING!!! same k and k' vectors as input of the elastic scattering equation")
            return 0.0

        if sname.upper() in ["IMP"]:  # ionized impurity scattering
            unit_conversion = 0.001 / e ** 2
            return unit_conversion * e ** 4 * self.egrid["N_II"][c][T] / \
                   (4.0 * pi ** 2 * self.epsilon_s ** 2 * epsilon_0 ** 2 * hbar) \
                   / ((norm_diff_k ** 2 + self.egrid["beta"][c][T][tp] ** 2) ** 2)

        elif sname.upper() in ["ACD"]:  # acoustic deformation potential scattering
            unit_conversion = 1e18 * e
            return unit_conversion * k_B * T * self.E_D[tp] ** 2 / (4.0 * pi ** 2 * hbar * self.C_el)

        elif sname.upper() in ["PIE"]:  # piezoelectric scattering
            unit_conversion = 1e9 / e
            return unit_conversion * e ** 2 * k_B * T * self.P_PIE ** 2 \
                   / (norm_diff_k ** 2 * 4.0 * pi ** 2 * hbar * epsilon_0 * self.epsilon_s)

        elif sname.upper() in ["DIS"]:
            return self.gs

        else:
            raise ValueError("The elastic scattering name {} is not supported!".format(sname))


    def integrate_func_over_E(self, func, tp, fermi, T, interpolation_nsteps=None, xDOS=True, normalize_energy=False):
        if not interpolation_nsteps:
            interpolation_nsteps = max(200, int(500.0 / len(self.egrid[tp]["energy"])))
        integral = 0.0
        for ie in range(len(self.egrid[tp]["energy"]) - 1):
            E = self.egrid[tp]["energy"][ie]
            dE = (self.egrid[tp]["energy"][ie + 1] - E) / interpolation_nsteps
            if normalize_energy:
                E -= self.cbm_vbm[tp]["energy"]
                fermi -= self.cbm_vbm[tp]["energy"]
            if xDOS:
                dS = (self.egrid[tp]["DOS"][ie + 1] - self.egrid[tp]["DOS"][ie]) / interpolation_nsteps
                for i in range(interpolation_nsteps):
                    integral += dE * (self.egrid[tp]["DOS"][ie] + i * dS) * func(E + i * dE, fermi, T)
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
                        dx1 = (grid[i,j,k,0] - grid[i-1,j,k,0]) / 2
                    else:
                        dx1 = grid[i,j,k,0] - (-0.5)
                    if i < N[0] - 1:
                        dx2 = (grid[i+1,j,k,0] - grid[i,j,k,0]) / 2
                    else:
                        dx2 = 0.5 - grid[i,j,k,0]

                    if j > 0:
                        dy1 = (grid[i,j,k,1] - grid[i,j-1,k,1]) / 2
                    else:
                        dy1 = grid[i,j,k,1] - (-0.5)
                    if j < N[1] - 1:
                        dy2 = (grid[i,j+1,k,1] - grid[i,j,k,1]) / 2
                    else:
                        dy2 = 0.5 - grid[i,j,k,1]

                    if k > 0:
                        dz1 = (grid[i,j,k,2] - grid[i,j,k-1,2]) / 2
                    else:
                        dz1 = grid[i,j,k,2] - (-0.5)
                    if k < N[2] - 1:
                        dz2 = (grid[i,j,k+1,2] - grid[i,j,k,2]) / 2
                    else:
                        dz2 = 0.5 - grid[i,j,k,2]
                    # find fractional volume
                    dv[i,j,k] = (dx1 + dx2) * (dy1 + dy2) * (dz1 + dz2)

        # convert from fractional to cartesian (k space) volume
        dv *= self._rec_lattice.volume / (A_to_m * m_to_cm) ** 3
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
        return [np.sum(func_grid[:,:,:,i] * self.dv_grid[tp]) for i in range(func_grid.shape[3])]


    def integrate_over_normk(self, prop_list, tp, c, T, xDOS):
        normk_tp = "norm(k)"
        for ib in [0]:
            normk_sorted_idx = np.argsort(self.kgrid[tp][normk_tp][ib])
            normk_vec = np.array(self.kgrid[tp][normk_tp][ib])
            dk_vec = np.array([
                self.kgrid[tp][normk_tp][ib][normk_sorted_idx[j+1]] - \
                self.kgrid[tp][normk_tp][ib][normk_sorted_idx[j]]
                        for j in range(len(normk_sorted_idx)-1)] + [0.0])
            integral_vec = normk_vec * dk_vec
            if xDOS:
                integral_vec *= normk_vec**2/pi
            for j, p in enumerate(prop_list):
                if p[0] == "/":
                    vec = np.array(self.kgrid[tp][p.split("/")[-1]][c][T][ib])
                elif "1 -" in p:
                    vec = np.array(self.kgrid[tp][p.split("-")[-1].replace(" ", "")][c][T][ib])
                else:
                    vec = np.array(self.kgrid[tp][p][c][T][ib])
                if len(vec.shape) > 1:
                    integral_vec *= np.mean(vec, axis=-1)
                else:
                    integral_vec *= vec
            integral = np.sum(integral_vec)
        return integral


    def integrate_over_E(self, prop_list, tp, c, T, xDOS=False, xvel=False, interpolation_nsteps=None):
        imax_occ = len(self.Efrequency[tp][:-1])

        if not interpolation_nsteps:
            interpolation_nsteps = max(200, int(500.0 / len(self.egrid[tp]["energy"])))
        diff = [0.0 for prop in prop_list]
        integral = self.gs
        for ie in range(imax_occ):
            E = self.egrid[tp]["energy"][ie]
            dE = abs(self.egrid[tp]["energy"][ie + 1] - E) / interpolation_nsteps
            if xDOS:
                dS = (self.egrid[tp]["DOS"][ie + 1] - self.egrid[tp]["DOS"][ie]) / interpolation_nsteps
            if xvel:
                dv = (self.egrid[tp]["velocity"][ie + 1] - self.egrid[tp]["velocity"][ie]) / interpolation_nsteps
            for j, p in enumerate(prop_list):
                if "/" in p:
                    diff[j] = (self.egrid[tp][p.split("/")[-1]][c][T][ie + 1] -
                               self.egrid[tp][p.split("/")[-1]][c][T][ie]) / interpolation_nsteps
                elif "1 -" in p:
                    diff[j] = (1 - self.egrid[tp][p.split("-")[-1].replace(" ", "")][c][T][ie + 1] - (1- \
                               self.egrid[tp][p.split("-")[-1].replace(" ", "")][c][T][ie])) / interpolation_nsteps
                else:
                    diff[j] = (self.egrid[tp][p][c][T][ie + 1] - self.egrid[tp][p][c][T][ie]) / interpolation_nsteps
            for i in range(interpolation_nsteps):
                multi = dE
                for j, p in enumerate(prop_list):
                    if p[0] == "/":
                        multi /= self.egrid[tp][p.split("/")[-1]][c][T][ie] + diff[j] * i
                    elif "1 -" in p:
                        multi *= 1 - self.egrid[tp][p.split("-")[-1].replace(" ", "")][c][T][ie] + diff[j] * i
                    else:
                        multi *= self.egrid[tp][p][c][T][ie] + diff[j] * i
                if xDOS:
                    multi *= self.egrid[tp]["DOS"][ie] + dS * i
                if xvel:
                    multi *= self.egrid[tp]["velocity"][ie] + dv * i
                integral += multi
        return np.array(integral)



    def integrate_over_X(self, tp, X_E_index, integrand, ib, ik, c, T, sname=None, g_suffix=""):
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
            raise ValueError("enforcing scattering points did NOT work, {}[{}][{}] is empty".format(X_E_index, ib, ik))
        X, ib_prm, ik_prm = X_E_index[ib][ik][0]
        current_integrand = integrand(tp, c, T, ib, ik, ib_prm, ik_prm, X, sname=sname, g_suffix=g_suffix)

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
                same_X_ks.append(self.kgrid[tp]['cartesian kpoints'][ib_prm][ik_prm])
                same_X_ks_integrands.append(integrand(tp, c, T, ib, ik, ib_prm, ik_prm, X, sname=sname, g_suffix=g_suffix))
                DeltaX = X_E_index[ib][ik][ikp + 1][0] - X_E_index[ib][ik][ikp][0]

            if len(same_X_ks) > 1:
                m = np.sum(same_X_ks, axis=0)/len(same_X_ks)
                same_X_ks = np.array(same_X_ks) - m
                same_X_ks_sort, ks_indexes = sort_angles(same_X_ks)
                same_X_ks_sort = np.vstack((same_X_ks_sort, same_X_ks_sort[0]))
                ks_indexes.append(ks_indexes[0])
                sm = 0.0
                for j in range(len(ks_indexes) - 1):
                    angle = get_angle(same_X_ks_sort[j+1], same_X_ks_sort[j])
                    sm += (same_X_ks_integrands[ks_indexes[j+1]] + \
                           same_X_ks_integrands[ks_indexes[j]])/2.0 * angle
                dum = sm/(2*pi)/2.0
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



    def el_integrand_X(self, tp, c, T, ib, ik, ib_prm, ik_prm, X, sname=None, g_suffix=""):
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
                [0.0, 0.0, 0.0])  # self-scattering is not defined;regardless, the returned integrand must be a vector


        return (1 - X) * self.kgrid[tp]["norm(k)"][ib_prm][ik_prm] ** 2 * self.s_el_eq(sname, tp, c, T, k, k_prm) \
               * self.G(tp, ib, ik, ib_prm, ik_prm, X) / (self.kgrid[tp]["norm(v)"][ib_prm][ik_prm] / sq3)


    def inel_integrand_X(self, tp, c, T, ib, ik, ib_prm, ik_prm, X, sname=None, g_suffix=""):
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
        if tp == "n" and 'minus' in sname and self.kgrid[tp]["energy"][ib][ik]-hbar*self.kgrid[tp]["W_POP"][ib][ik]<self.cbm_vbm[tp]["energy"]:
            return 0.0
        if tp == "p" and 'plus' in sname and self.kgrid[tp]["energy"][ib][ik]+hbar*self.kgrid[tp]["W_POP"][ib][ik]>self.cbm_vbm[tp]["energy"]:
            return 0.0
        if abs(self.kgrid[tp]['energy'][ib_prm][ik_prm] - \
                self.kgrid[tp]['energy'][ib][ik]) < \
                                hbar*self.kgrid[tp]["W_POP"][ib][ik]/2:
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

        integ = self.kgrid[tp]["norm(k)"][ib_prm][ik_prm]**2*self.G(tp, ib, ik, ib_prm, ik_prm, X)/\
                (self.kgrid[tp]["norm(v)"][ib_prm][ik_prm]*norm_diff**2/sq3)

        if "S_i" in sname:
            integ *= X * self.kgrid[tp]["g" + g_suffix][c][T][ib_prm][ik_prm]
            if "minus" in sname:
                integ *= (1 - f) * N_POP + f * (1 + N_POP)
            elif "plus" in sname:
                integ *= (1 - f) * (1 + N_POP) + f * N_POP
            else:
                raise ValueError('"plus" or "minus" must be in sname for phonon absorption and emission respectively')
        elif "S_o" in sname:
            if "minus" in sname:
                integ *= (1 - f_prm) * (1 + N_POP) + f_prm * N_POP # interestingly f or f_prm does NOT make any difference (maybe close energies?)
            elif "plus" in sname:
                integ *= (1 - f_prm) * N_POP + f_prm * (1 + N_POP)
            else:
                raise ValueError('"plus" or "minus" must be in sname for phonon absorption and emission respectively')
        else:
            raise ValueError('Unsupported inelastic scattering name: {}'.format(sname))
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
                        results = [calculate_Sio(tp, c, T, ib, ik,
                                    once_called, self.kgrid, self.cbm_vbm,
                                    self.epsilon_s, self.epsilon_inf) for ik in
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
                                    self.kgrid[tp]["S_o"][c][T][ib][ik] = rlts[2]
                                else:
                                    self.kgrid[tp]["S_o"][c][T][ib][ik] = res[2]

                                if (res[3] < 0.1).any():
                                    self.kgrid[tp]["S_o_th"][c][T][ib][ik] = rlts[3]
                                else:
                                    self.kgrid[tp]["S_o_th"][c][T][ib][ik] = res[3]


    def s_inelastic(self, sname=None, g_suffix=""):
        """
        Calculates the inelastic/POP scattering rate (with correct units)
        by integrating over dX (X being the angle between k and k' states) for
        all band-kpoint pair.

        Args:
            sname (str): scattering name: 'S_oX_Eplus_ik', 'S_oX_Eminus_ik',
                'S_iX_Eplus_ik' or 'S_iX_Eminus_ik'
            g_suffix (str): perturbation name; options: "", "_POP" or "_th"
        """
        for tp in ["n", "p"]:
            for c in self.dopings:
                for T in self.temperatures:
                    for ib in range(len(self.kgrid[tp]["energy"])):
                        cumulative = 0.0
                        for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                            summation = np.array([0.0, 0.0, 0.0])
                            for X_E_index_name in ["X_Eplus_ik", "X_Eminus_ik"]:
                                summation += self.integrate_over_X(tp, self.kgrid[tp][X_E_index_name],
                                    self.inel_integrand_X, ib=ib, ik=ik, c=c,
                                    T=T, sname=sname + X_E_index_name, g_suffix=g_suffix)
                            cumulative += summation
                            if "S_o" in sname and np.min(summation) < 0.1:
                                summation = cumulative / (ik+1) # set small S_o rates to average rate (so far) to avoid inelastic scattering blow up (division by S_o ~ 0 in POP)
                            self.kgrid[tp][sname][c][T][ib][ik] = summation * e ** 2 * self.kgrid[tp]["W_POP"][ib][ik] / (4 * pi * hbar) * (1 / self.epsilon_inf - 1 / self.epsilon_s) / epsilon_0 * 100 / e


    def s_el_eq_isotropic(self, sname, tp, c, T, ib, ik):
        """
        Returns elastic scattering rate (a numpy vector) at given point
        (i.e. k-point, c, T) in isotropic formulation (i.e. if
        self.bs_is_isotropic==True). This assumption significantly simplifies
        the model and the integrated rates at each k/energy directly extracted
        from the literature can be used here.

        Args:
            sname (str): elastic scattering name: 'ACD', 'IMP', 'PIE', 'DIS'
            tp (str): 'n' or 'p' type respectively for conduction and valence
            c (float): carrier concentration
            T (float): temperature
            ib (int): band index starting from 0 (0 for CBM/VBM bands)
            ik (int): k-point index

        Returns (float): scalar (since assumed isotropic) scattering rate.
        """
        v = self.kgrid[tp]["norm(v)"][ib][ik] / sq3  # because of isotropic assumption, we treat the BS as 1D
        # v = self.kgrid[tp]["velocity"][ib][ik]  # because of isotropic assumption, we treat the BS as 1D
        knrm = self.kgrid[tp]["norm(k)"][ib][ik]
        par_c = self.kgrid[tp]["c"][ib][ik]

        if sname.upper() == "ACD":
            # The following two lines are from Rode's chapter (page 38)
            return (k_B * T * self.E_D[tp] ** 2 * knrm ** 2) / (3 * pi * hbar ** 2 * self.C_el * 1e9 * v) \
                   * (3 - 8 * par_c ** 2 + 6 * par_c ** 4) * e * 1e20

        elif sname.upper() == "IMP":
            # double-checked the units and equation on 5/12/2017
            # The following is a variation of Dingle's theory available in [R]
            beta = self.egrid["beta"][c][T][tp]
            B_II = (4 * knrm ** 2 / beta ** 2) / (1 + 4 * knrm ** 2 / beta ** 2) + 8 * (beta ** 2 + 2 * knrm ** 2) / (
            beta ** 2 + 4 * knrm ** 2) * par_c ** 2 + \
                   (3 * beta ** 4 + 6 * beta ** 2 * knrm ** 2 - 8 * knrm ** 4) / (
                   (beta ** 2 + 4 * knrm ** 2) * knrm ** 2) * par_c ** 4
            D_II = 1 + 2 * beta ** 2 * par_c ** 2 / knrm ** 2 + 3 * beta ** 4 * par_c ** 4 / (4 * knrm ** 4)

            return abs((e ** 4 * abs(self.egrid["N_II"][c][T])) / (
                8 * pi * v * self.epsilon_s ** 2 * epsilon_0 ** 2 * hbar ** 2 *
                knrm ** 2) * (D_II * log(1 + 4 * knrm ** 2 / beta ** 2) - B_II) * 3.89564386e27)

        elif sname.upper() == "PIE":
            return (e ** 2 * k_B * T * self.P_PIE ** 2) / (
                6 * pi * hbar ** 2 * self.epsilon_s * epsilon_0 * v) * (
                       3 - 6 * par_c ** 2 + 4 * par_c ** 4) * 100 / e

        elif sname.upper() == "DIS":
            return (self.N_dis * e ** 4 * knrm) / (
            hbar ** 2 * epsilon_0 ** 2 * self.epsilon_s ** 2 * (self._vrun.lattice.c * A_to_nm) ** 2 * v) \
                   / (self.egrid["beta"][c][T][tp] ** 4 * (
            1 + (4 * knrm ** 2) / (self.egrid["beta"][c][T][tp] ** 2)) ** 1.5) \
                   * 2.43146974985767e42 * 1.60217657 / 1e8;
        else:
            raise ValueError('The elastic scattering name "{}" is NOT supported.'.format(sname))


    def s_elastic(self, sname):
        """
        The scattering rate equation for each elastic scattering name (sname)

        Args:
            sname (st): elastic scattering name: 'IMP', 'ADE', 'PIE', 'DIS'

        Returns:
            it directly calculates the scattering rate at each k-point at each
                c and T (self.kgrid[tp][sname][c][T][ib][ik])
        """
        sname = sname.upper()
        for tp in ["n", "p"]:
            self.egrid[tp][sname] = {c: {T: np.array([[0.0, 0.0, 0.0] for i in
                    range(len(self.egrid[tp]["energy"]))]) for T in
                    self.temperatures} for c in self.dopings}
            self.kgrid[tp][sname] = {c: {T: np.array([[[0.0, 0.0, 0.0] for i in
                    range(len(self.kgrid[tp]["kpoints"][j]))]
                    for j in range(self.cbm_vbm[tp]["included"])]) for T in
                    self.temperatures} for c in self.dopings}
            for c in self.dopings:
                for T in self.temperatures:
                    for ib in range(len(self.kgrid[tp]["energy"])):
                        for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                            if self.bs_is_isotropic:
                                self.kgrid[tp][sname][c][T][ib][ik] = self.s_el_eq_isotropic(sname, tp, c, T, ib, ik)
                            else:
                                summation = self.integrate_over_X(tp, X_E_index=self.kgrid[tp]["X_E_ik"],
                                                                  integrand=self.el_integrand_X,
                                                                  ib=ib, ik=ik, c=c, T=T, sname=sname, g_suffix="")
                                self.kgrid[tp][sname][c][T][ib][ik] = abs(summation) * 2e-7 * pi / hbar
                                if norm(self.kgrid[tp][sname][c][T][ib][ik]) < 100 and sname not in ["DIS"]:
                                    self.logger.warning("Here {} rate < 1.\nX_E_ik:\n{}".format(sname, self.kgrid[tp]["X_E_ik"][ib][ik]))
                                    self.kgrid[tp][sname][c][T][ib][ik] = [1e10, 1e10, 1e10]

                                if norm(self.kgrid[tp][sname][c][T][ib][ik]) > 1e20:
                                    self.logger.warning('too large rate for {} at k={}, v={}:'.format(
                                        sname, self.kgrid[tp]['kpoints'][ib][ik], self.kgrid[tp]['velocity'][ib][ik]))
                            self.kgrid[tp]["_all_elastic"][c][T][ib][ik] += self.kgrid[tp][sname][c][T][ib][ik]
                        self.kgrid[tp]["relaxation time"][c][T][ib] = 1 / self.kgrid[tp]["_all_elastic"][c][T][ib]



    def map_to_egrid(self, prop_name, c_and_T_idx=True, prop_type="vector"):
        """
        Maps a propery from kgrid to egrid conserving the nomenclature.
            The mapped property w/ format: kgrid[tp][prop_name][c][T][ib][ik]
            will have the format: egrid[tp][prop_name][c][T][ie]

        Args:
            prop_name (string): the name of the property to be mapped. It must be available in the kgrid.
            c_and_T_idx (bool): if True, the propetry will be calculated and maped at each concentration, c, and T
            prop_type (str): options are "scalar", "vector", "tensor"

        Returns (float or numpy.array): egrid[tp][prop_name][c][T][ie]
        """
        if not c_and_T_idx:
            self.initialize_var("egrid", prop_name, prop_type, initval=self.gs, is_nparray=True, c_T_idx=False)
            for tp in ["n", "p"]:
                for ie, en in enumerate(self.egrid[tp]["energy"]):
                    for ib, ik in self.kgrid_to_egrid_idx[tp][ie]:
                        self.egrid[tp][prop_name][ie] += self.kgrid[tp][prop_name][ib][ik]
                    self.egrid[tp][prop_name][ie] /= len(self.kgrid_to_egrid_idx[tp][ie])
        else:
            self.initialize_var("egrid", prop_name, prop_type, initval=self.gs, is_nparray=True, c_T_idx=True)
            for tp in ["n", "p"]:
                for c in self.dopings:
                    for T in self.temperatures:
                        for ie, en in enumerate(self.egrid[tp]["energy"]):
                            for ib, ik in self.kgrid_to_egrid_idx[tp][ie]:
                                self.egrid[tp][prop_name][c][T][ie] += self.kgrid[tp][prop_name][c][T][ib][ik]
                            self.egrid[tp][prop_name][c][T][ie] /= len(self.kgrid_to_egrid_idx[tp][ie])


    def find_fermi_k(self, tolerance=0.001, num_bands = None):
        num_bands = num_bands or self.num_bands
        closest_energy = {c: {T: None for T in self.temperatures} for c in self.dopings}
        self.f0_array = {c: {T: {tp: list(range(num_bands[tp])) for tp in ['n', 'p']} for T in self.temperatures} for c in self.dopings}
        for c in self.dopings:
            tp = get_tp(c)
            tol = tolerance * abs(c)
            for T in self.temperatures:
                step = 0.1
                range_of_energies = np.arange(self.cbm_vbm[tp]['energy'] - 2, self.cbm_vbm[tp]['energy'] + 2.1, step)
                diff = 1000 * abs(c)
                while(diff > tol):
                    # try a number for fermi level
                    diffs = {}
                    for e_f in range_of_energies:
                        # calculate distribution in both conduction and valence bands
                        f_con = 1 / (np.exp((self.energy_array['n'] - e_f) / (k_B * T)) + 1)
                        f_val = 1 / (np.exp((self.energy_array['p'] - e_f) / (k_B * T)) + 1)
                        # density of states in k space is V/8pi^3 per spin, but total states per real volume per k volume is 2/8pi^3
                        dens_of_states = 1 / (4*np.pi**3)
                        # see if it is close to concentration
                        n_concentration = self.integrate_over_states(f_con * dens_of_states, 'n')[0]
                        p_concentration = self.integrate_over_states((1 - f_val) * dens_of_states, 'p')[0]
                        diffs[e_f] = abs((p_concentration - n_concentration) - c)
                    # compare all the numbers and zoom in on the closest
                    closest_energy[c][T] = min(diffs, key=diffs.get)
                    range_of_energies = np.arange(closest_energy[c][T] - step, closest_energy[c][T] + step, step / 10)
                    step /= 10
                    diff = diffs[closest_energy[c][T]]
                # find the calculated concentrations (dopings) of each type at the determined fermi level
                e_f = closest_energy[c][T]
                for j, tp in enumerate(['n', 'p']):
                    for ib in list(range(num_bands[tp])):
                        self.f0_array[c][T][tp][ib] = 1 / (np.exp((self.energy_array[tp][ib][:,:,:,0] - e_f) / (k_B * T)) + 1)
                    self.calc_doping[c][T][tp] = self.integrate_over_states(j - np.array(self.f0_array[c][T][tp]), tp)
        return closest_energy


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
            nstep (int): number of steps to check before and after a given
                fermi level

        Returns (float in eV):
            The fitted/calculated Fermi level
        """
        relative_error = self.gl
        typ = get_tp(c)
        fermi = self.cbm_vbm[typ]["energy"] + 0.01 # initialize fermi non-zero
        conversion = 1.0 / (self.volume * (A_to_m * m_to_cm) ** 3)

        dos_e = np.array([d[0] for d in self.dos])
        dos_de = np.array([self.dos[i + 1][0] - self.dos[i][0] for i, _ in enumerate(self.dos[:-1])] + [0.0])
        dos_dos = np.array([d[1] for d in self.dos])

        # fix energy, energy diff. and dos for integration at all fermi levels
        es = np.repeat(dos_e.reshape((len(dos_e), 1)), 2*nstep+1, axis=1)
        de = np.repeat(dos_de.reshape((len(dos_de), 1)), 2*nstep+1, axis=1)
        tdos = np.repeat(dos_dos.reshape((len(dos_dos), 1)), 2*nstep+1, axis=1)

        self.logger.debug("Calculating the fermi level at T={} K".format(T))
        for i in range(20):
            fermi_range = np.linspace(fermi-nstep*step, fermi+nstep*step, 2*nstep+1)
            n_dopings = -conversion * np.sum(tdos[self.cbm_dos_idx:] * f0(es[self.cbm_dos_idx:], fermi_range, T) * de[self.cbm_dos_idx:], axis=0)
            p_dopings = conversion * np.sum(tdos[:self.vbm_dos_idx+1] * (1 - f0(es[:self.vbm_dos_idx+1], fermi_range, T)) * de[:self.vbm_dos_idx+1], axis=0)
            relative_error = abs((n_dopings+p_dopings)/c - 1.0)
            fermi_idx = np.argmin(relative_error)
            fermi = fermi_range[fermi_idx]
            self.calc_doping[c][T]['n'] = n_dopings[fermi_idx]
            self.calc_doping[c][T]['p'] = p_dopings[fermi_idx]
            if relative_error[fermi_idx] < rtol:
                self.logger.info("fermi at {} 1/cm3 and {} K: {}".format(c, T, fermi))
                return fermi
            step /= 10.0

        if relative_error[fermi_idx] > rtol_loose:
            raise AmsetError('The calculated concentration is not within {}% of'
            ' the given value ({}) at T={}'.format(rtol_loose*100, c, T))
        elif relative_error[fermi_idx] > rtol:
            self.logger.warning('Fermi calculated with a loose tolerance of {}%'
                                ' at c={}, T={}K'.format(rtol_loose, c, T))
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
            integral = self.integrate_over_E(prop_list=["f0","1 - f0"], tp=tp, c=c, T=T, xDOS=True)
            beta[tp] = (e**2 / (self.epsilon_s * epsilon_0 * k_B * T) * integral / self.volume * 1e12 / e) ** 0.5
        return beta


    def to_file(self, path=None, dir_name='run_data', fname='amsetrun',
                force_write=True):
        path = os.path.join(path or self.calc_dir, dir_name)
        if not os.path.exists(path):
            os.makedirs(name=path)
        if not force_write:
            n = 1
            fname0 = fname
            while os.path.exists(os.path.join(path, '{}.json.gz'.format(fname))):
                warnings.warn('The file, {} exists. Amset outputs will be '
                        'written in {}'.format(fname, fname0+'_'+str(n)))
                fname = fname0 + '_' + str(n)
                n += 1
        out_d = self.as_dict()

        # write the output dict to file
        with gzip.GzipFile(os.path.join(path, '{}.json.gz'.format(fname)),
                           mode='w') as fp:
            json_str = json.dumps(out_d, cls=MontyEncoder)
            json_bytes = json_str.encode('utf-8')
            fp.write(json_bytes)


    def as_dict(self):
        """
        Returns the Amset onbject and its selected variables as python
            dictionary (dict)

        Returns (dict):
        """
        out_d = {'kgrid0': self.kgrid0,
                 'egrid0': self.egrid0,
                 'cbm_vbm': self.cbm_vbm,
                 'mobility': self.mobility,
                 'elastic_scats': self.elastic_scats,
                 'inelastic_scats': self.inelastic_scats,
                 'Efrequency0': self.Efrequency0,
                 'dopings': self.dopings,
                 'temperatures': self.temperatures,
                 'material_params': self.material_params,
                 'performance_params': self.performance_params,
                 'model_params': self.model_params,
                 'all_types': self.all_types,
                 }
        return out_d


    @staticmethod
    def from_file(path=None, dir_name="run_data", filename="amsetrun.json.gz"):
        #TODO: make this better, maybe organize these class attributes a bit?
        if not path:
            path = os.path.join(os.getcwd(), dir_name)

        with gzip.GzipFile(os.path.join(path, filename), mode='r') as fp:
            d = json.load(fp, cls=MontyDecoder)
        amset = Amset(calc_dir=path, material_params={'epsilon_s': d['epsilon_s']})
        amset.kgrid0 = d['kgrid0']
        amset.egrid0 = d['egrid0']
        amset.cbm_vbm = d['cbm_vbm']
        amset.mobility = d['mobility']
        amset.elastic_scats = d['elastic_scats']
        amset.inelastic_scats = d['inelastic_scats']
        amset.Efrequency0 = d['Efrequency0']
        amset.dopings = [float(dope) for dope in d['dopings']]
        amset.temperatures = [float(T) for T in d['temperatures']]
        amset.material_params = d['material_params']
        amset.performance_params = d['performance_params']
        amset.model_params = d['model_params']
        amset.all_types = list(set([get_tp(c) for c in amset.dopings]))
        return amset


    def to_json(self, kgrid=True, trimmed=False, max_ndata=None, nstart=0,
                valleys=True, path=None, dir_name="run_data"):
        """
        Writes the kgrid and egird to json files

        Args:
            kgrid (bool): whether to also write kgrid to kgrid.json
            trimmed (bool): if trimmed some properties (dict keys) will be
                removed to save space
            max_ndata (int): the maximum index from the CBM/VBM written to file
            nstart (int): the initial list index of a property written to file

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
                                    egrid[tp][key][c][T] = self.egrid[tp][key][c][T][nstart:nstart + nmax]
                                else:
                                    egrid[tp][key][c][T] = self.egrid[tp][key][c][T][::-1][nstart:nstart + nmax]
                    except:
                        try:
                            if tp == "n":
                                egrid[tp][key] = self.egrid[tp][key][nstart:nstart + nmax]
                            else:
                                egrid[tp][key] = self.egrid[tp][key][::-1][nstart:nstart + nmax]
                        except:
                            if key not in ['mobility']:
                                self.logger.warning('in to_json: cutting {} '
                                                'in egrid failed!'.format(key))

        with open(os.path.join(path, "egrid.json"), 'w') as fp:
            json.dump(egrid, fp, sort_keys=True, indent=4, ensure_ascii=False, cls=MontyEncoder)

        # self.kgrid trimming
        if kgrid:
            kgrid = deepcopy(self.kgrid)
            if trimmed:
                nmax = min([max_ndata + 1, min([len(kgrid["n"]["kpoints"][0]), len(kgrid["p"]["kpoints"][0])])])
                for tp in ["n", "p"]:
                    for key in kgrid[tp]:
                        if key in ["size"]:
                            continue
                        try:
                            for c in self.dopings:
                                for T in self.temperatures:
                                    if tp == "n":
                                        kgrid[tp][key][c][T] = [self.kgrid[tp][key][c][T][b][nstart:nstart + nmax]
                                                            for b in range(self.cbm_vbm[tp]["included"])]
                                    else:
                                        kgrid[tp][key][c][T] = [self.kgrid[tp][key][c][T][b][::-1][nstart:nstart + nmax]
                                                                for b in range(self.cbm_vbm[tp]["included"])]
                        except:
                            try:
                                if tp == "n":
                                    kgrid[tp][key] = [self.kgrid[tp][key][b][nstart:nstart + nmax]
                                                  for b in range(self.cbm_vbm[tp]["included"])]
                                else:
                                    kgrid[tp][key] = [self.kgrid[tp][key][b][::-1][nstart:nstart + nmax]
                                                      for b in range(self.cbm_vbm[tp]["included"])]
                            except:
                                if key not in ['mobility']:
                                    self.logger.warning('in to_json: cutting {} '
                                        'in kgrid failed!'.format(key))

            with open(os.path.join(path, "kgrid.json"), 'w') as fp:
                json.dump(kgrid, fp, sort_keys=True, indent=4, ensure_ascii=False, cls=MontyEncoder)
        if valleys:
            with open(os.path.join(path, "valleys.json"), 'w') as fp:
                json.dump(self.valleys, fp,
                          sort_keys=True, indent=4,
                          ensure_ascii=False, cls=MontyEncoder)


    def solve_BTE_iteratively(self):
        """
        Iteratively solve linearized/low-field Boltzmann Transport Equation
        See equation (43) page 20 of the reference [R]

        Returns (None): the results are stored in "g*" keys in kgrid and egrid
        """
        # calculating S_o scattering rate which is not a function of g
        if "POP" in self.inelastic_scats and not self.bs_is_isotropic:
            for g_suffix in ["", "_th"]:
                self.s_inelastic(sname="S_o" + g_suffix, g_suffix=g_suffix)

        # solve BTE to calculate S_i scattering rate and perturbation (g) in an iterative manner
        for iter in range(self.BTE_iters):
            self.logger.info("Performing iteration # {}".format(iter))
            if "POP" in self.inelastic_scats:
                if self.bs_is_isotropic:
                    if iter == 0:
                        self.s_inel_eq_isotropic(once_called=False)
                    else:
                        self.s_inel_eq_isotropic(once_called=True)
                else:
                    for g_suffix in ["", "_th"]:
                        self.s_inelastic(sname="S_i" + g_suffix, g_suffix=g_suffix)
            for c in self.dopings:
                for T in self.temperatures:
                    for tp in ["n", "p"]:
                        g_old = np.array(self.kgrid[tp]["g"][c][T][0])
                        for ib in range(self.cbm_vbm[tp]["included"]):
                            self.kgrid[tp]["g_POP"][c][T][ib] = (self.kgrid[tp]["S_i"][c][T][ib] +
                                    self.kgrid[tp]["electric force"][c][T][ib]) / (
                                    self.kgrid[tp]["S_o"][c][T][ib] + self.gs + 1.0)

                            self.kgrid[tp]["g"][c][T][ib] = (self.kgrid[tp]["S_i"][c][T][ib] +
                                    self.kgrid[tp]["electric force"][c][
                                    T][ib]) / (self.kgrid[tp]["S_o"][c][T][ib] +
                                    self.kgrid[tp]["_all_elastic"][c][T][ib])

                            self.kgrid[tp]["g_th"][c][T][ib] = (self.kgrid[tp]["S_i_th"][c][T][ib] +
                                    self.kgrid[tp]["thermal force"][c][T][ib]) / (
                                    self.kgrid[tp]["S_o_th"][c][T][ib] + self.kgrid[tp]["_all_elastic"][c][T][ib])

                            # TODO: correct these lines to reflect that f = f0 + x*g
                            self.kgrid[tp]["f"][c][T][ib] = self.kgrid[tp]["f0"][c][T][ib] + self.kgrid[tp]["g"][c][T][ib]
                            self.kgrid[tp]["f_th"][c][T][ib] = self.kgrid[tp]["f0"][c][T][ib] + self.kgrid[tp]["g_th"][c][T][ib]

                        avg_g_diff = np.mean([abs(g_old[ik] - self.kgrid[tp]["g"][c][T][0][ik]) for ik in range(len(g_old))])
                        self.logger.info("Average difference in {}-type g term at c={} and T={}: {}".format(tp, c, T, avg_g_diff))

        for prop in ["electric force", "thermal force", "g", "g_POP", "g_th", "S_i", "S_o", "S_i_th", "S_o_th"]:
            self.map_to_egrid(prop_name=prop, c_and_T_idx=True)

        for tp in ["n", "p"]:
            for c in self.dopings:
                for T in self.temperatures:
                    for ie in range(len(self.egrid[tp]["g_POP"][c][T])):
                        if norm(self.egrid[tp]["g_POP"][c][T][ie]) > 1:
                            self.egrid[tp]["g_POP"][c][T][ie] = [1e-5, 1e-5, 1e-5]

    def calc_v_vec(self, tp):
        v_vec_all_bands = []
        v_norm_all_bands = []
        for ib in range(self.num_bands[tp]):
            v_vec_k_ordered = self.velocity_signed[tp][ib][self.pos_idx[tp]]
            v_norm_k_ordered = (v_vec_k_ordered[:,0]**2 + v_vec_k_ordered[:,1]**2 + v_vec_k_ordered[:,2]**2)**0.5
            v_vec_all_bands.append(self.grid_from_ordered_list(v_vec_k_ordered, tp, none_missing=True))
            v_norm_all_bands.append(self.grid_from_ordered_list(v_norm_k_ordered, tp, none_missing=True, scalar=True))
        return np.array(v_vec_all_bands), np.array(v_norm_all_bands)


    def array_from_kgrid(self, prop_name, tp, c=None, T=None, denom=False, none_missing=False, fill=None):
        """
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
            return np.array([self.grid_from_energy_list(self.kgrid[tp][prop_name][c][T][ib], tp, ib, denom=denom, none_missing=none_missing, fill=fill) for ib in range(self.num_bands[tp])])
        else:
            return np.array([self.grid_from_energy_list(self.kgrid[tp][prop_name][ib], tp, ib, denom=denom, none_missing=none_missing, fill=fill) for ib in range(self.num_bands[tp])])


    def grid_from_energy_list(self, prop_list, tp, ib, denom=False, none_missing=False, fill=None):
        """

        Args:
            prop_list: a list that is sorted by energy and missing removed points
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
        adjusted_prop_list = list(prop_list)
        # step 0 is reverse second sort
        adjusted_prop_list = np.array(adjusted_prop_list)[self.pos_idx_2[tp][ib]]
        adjusted_prop_list = [adjusted_prop_list[i] for i in range(adjusted_prop_list.shape[0])]

        # reverse what has been done: step 1 is add new points back
        if not none_missing:
            insert_list = False
            if type(adjusted_prop_list[0]) == np.ndarray or type(adjusted_prop_list[0]) == list:
                if len(adjusted_prop_list[0]) == 3:
                    insert_list = True
            for ik in self.rm_idx_list[tp][ib]:
                adjusted_prop_list.insert(ik, fill) if not insert_list else adjusted_prop_list.insert(ik, [fill,fill,fill])

        # step 2 is reorder based on first sort
        adjusted_prop_list = np.array(adjusted_prop_list)[self.pos_idx[tp]]
        # then call grid_from_ordered_list
        return self.grid_from_ordered_list(adjusted_prop_list, tp, denom=denom, none_missing=True)


    def grid_from_ordered_list(self, prop_list, tp, denom=False, none_missing=False, scalar=False):
        """
        Args:
            prop_list:
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
        adjusted_prop_list = list(prop_list)

        # put zeros back into spots of missing indexes
        # self.rm_idx_list format: [tp][ib][ik]
        if not none_missing:
            for ib in range(self.num_bands[tp]):
                for ik in self.rm_idx_list[tp][ib]:
                    if not denom:
                        adjusted_prop_list.insert(ik, 0)
                    if denom:
                        adjusted_prop_list.insert(ik, 1)

        for i in range(N[0]):
            for j in range(N[1]):
                for k in range(N[2]):
                    grid[i,j,k] = adjusted_prop_list[i*N[1]*N[2] + j*N[2] + k]
        return grid


    def integrate_over_states(self, integrand_grid, tp='all'):
        """

        Args:
            integrand_grid: list or array of array grids
            tp:

        Returns:

        """
        integrand_grid = np.array(integrand_grid)
        if type(integrand_grid[0][0,0,0]) == list or type(integrand_grid[0][0,0,0]) == np.ndarray:
            result = np.zeros(3)
        else:
            result = 0
        num_bands = integrand_grid.shape[0]
        for ib in range(num_bands):
            result += self.integrate_over_k(integrand_grid[ib], tp)
        return result


    # calculates transport properties for isotropic materials
    def calculate_transport_properties_with_k(self, test_anisotropic, important_points):
        # calculate mobility by averaging velocity per electric field strength
        mu_num = {tp: {el_mech: {c: {T: [0, 0, 0] for T in self.temperatures} for c in self.dopings} for el_mech in self.elastic_scats} for tp in ["n", "p"]}
        valley_transport = {tp: {el_mech: {c: {T: np.array([0., 0., 0.]) for T in self.temperatures} for c in
                  self.dopings} for el_mech in self.transport_labels} for tp in ["n", "p"]}

        for c in self.dopings:
            for T in self.temperatures:
                for j, tp in enumerate(["n", "p"]):
                    E_array = self.array_from_kgrid('energy', tp)
                    if not self.count_mobility[self.ibrun][tp]:
                        continue
                    N = self.kgrid_array[tp].shape

                    # get quantities that are independent of mechanism
                    num_k = [len(self.kgrid[tp]["energy"][ib]) for ib in range(self.num_bands[tp])]
                    df0dk = self.array_from_kgrid('df0dk', tp, c, T)
                    v = self.array_from_kgrid('velocity', tp)
                    f0_removed = self.array_from_kgrid('f0', tp, c, T)
                    f0_all = 1 / (np.exp((self.energy_array[tp] - self.fermi_level[c][T]) / (k_B * T)) + 1)

                    np.set_printoptions(precision=3)

                    # TODO: the anisotropic case is not correct right now
                    if not self.bs_is_isotropic or test_anisotropic:

                        v_vec, v_norm = self.calc_v_vec(tp)

                        #TODO: get f through solving the BTE anisotropically
                        #k_hat = np.array([self.k_hat_array[tp] for ib in range(self.num_bands[tp])])
                        k_hat_cartesian = np.array([self.k_hat_array_cartesian[tp] for ib in range(self.num_bands[tp])])
                        g = self.array_from_kgrid("g", tp, c, T)
                        # x is the cosine of the angle between the force and k, or negative the cosine of the angle
                        # between the electric fields and k
                        x = -k_hat_cartesian
                        f_T = f0_all + x * g

                        if tp == 'n':
                            print('v')
                            print(v_vec.shape)
                            print(v_vec[0, (N[0]-1)/2, (N[1]-1)/2, :])
                            print('k_hat_cartesian')
                            print(k_hat_cartesian[0, (N[0]-1)/2, (N[1]-1)/2, :])
                            print('g')
                            print(g[0, (N[0]-1)/2, (N[1]-1)/2, :])

                            print('v*f0_all')
                            print((v_vec * f0_all)[0, (N[0]-1)/2, (N[1]-1)/2, :])
                            print('v*f_T')
                            print((v_vec * f_T)[0, (N[0] - 1) / 2, (N[1] - 1) / 2, :])
                            print('v*(f_T-f0_all)')
                            print((v_vec * k_hat_cartesian * g)[0, (N[0] - 1) / 2, (N[1] - 1) / 2, :])
                            print('v*f_T*d3k')
                            print(((v_vec * f_T)[0] * self.dv_grid['n'][:,:,:,np.newaxis])[(N[0] - 1) / 2, (N[1] - 1) / 2, :])
                            print("sum")
                            print(np.sum(((v_vec * f_T)[0] * self.dv_grid['n'][:,:,:,np.newaxis]), axis=(0,1,2)))

                        # from equation 44 in Rode, overall
                        #nu_el = self.array_from_kgrid('_all_elastic', tp, c, T, denom=True)
                        # numerator = -self.integrate_over_states(v * self.k_hat_array[tp] * (-1 / hbar) * df0dk / nu_el, tp)
                        # denominator = self.integrate_over_states(f0, tp) * hbar * default_small_E
                        numerator = self.integrate_over_states(v_vec / default_small_E * f_T, tp)
                        numerator2 = self.integrate_over_states(v_vec / default_small_E * f0_all, tp)
                        numerator3 = self.integrate_over_states(v_vec / default_small_E * (f_T - f0_all), tp)
                        numerator4 = self.integrate_over_states(v_vec / default_small_E * (x * g), tp)
                        numerator5 = self.integrate_over_states(v_norm * x * x * g, tp) / default_small_E
                        numerator6 = self.integrate_over_states(v_norm * x**2 * g, tp) / default_small_E
                        denominator = self.integrate_over_states(j + ((-1) ** j) * f_T, tp)
                        self.mobility[tp]['overall'][c][T] = numerator / denominator

                        if tp == 'n':
                            print('ANISOTROPIC numerator, numerator without g, and denominator:')
                            print(numerator)
                            print(numerator2)
                            print(numerator3)
                            print(numerator4)
                            print(numerator5)
                            print(numerator6)
                            print(denominator)

                        if tp == 'n':
                            denominator_iso = self.integrate_over_states(f0_all, tp)
                        if tp == 'p':
                            denominator_iso = self.integrate_over_states(1-f0_all, tp)
                        numerator_iso = self.integrate_over_states(g * v_norm, tp) / 3 / default_small_E

                        if tp == 'n':
                            print('v')
                            print(v[0, (N[0] - 1) / 2, (N[1] - 1) / 2, :])
                            print('g*v')
                            print((g*v)[0, (N[0] - 1) / 2, (N[1] - 1) / 2, :])
                            print('ISOTROPIC numerator and denominator:')
                            print(numerator_iso)
                            print(denominator_iso)

                        k_norm = np.sqrt(self.kgrid_array_cartesian[tp][:,:,:,0]**2 + self.kgrid_array_cartesian[tp][:,:,:,1]**2 + self.kgrid_array_cartesian[tp][:,:,:,2]**2) / (A_to_m * m_to_cm)
                        print('norm(k)')
                        print(k_norm[(N[0] - 1) / 2, (N[1] - 1) / 2, :])   # 1/cm
                        print(self._rec_lattice.volume)   # in 1/A^3
                        k_0 = (self._rec_lattice.volume)**(1./3) / (A_to_m * m_to_cm)
                        #vol = self._rec_lattice.volume / (A_to_m * m_to_cm)**3
                        print('k_0')
                        print(k_0)   # in 1/cm
                        print('test integral of e^(-r) * cos^2(theta)')
                        aa = 10 / (k_0 / 2)   # in cm
                        print(self.integrate_over_k(np.exp(-aa * k_norm) * k_hat_cartesian[0, :, :, :, 2]**2 * aa**3, tp))

                    if self.bs_is_isotropic and not test_anisotropic:
                        if tp == get_tp(c):
                            self.logger.info('calculating mobility by integrating over'
                                         ' k-grid and isotropic BS assumption...')
                            self.logger.debug('current valley is at {}'.format(important_points))
                            self.logger.debug('the denominator is:\n{}'.format(self.denominator))

                        for el_mech in self.elastic_scats:
                            nu_el = self.array_from_kgrid(el_mech, tp, c, T, denom=True)
                            # this line should have -e / hbar except that hbar is in units of eV*s so in those units e=1
                            g = -1 / hbar * df0dk / nu_el
                            valley_transport[tp][el_mech][c][T] = self.integrate_over_states(g * v, tp)
                            # from equation 45 in Rode, inelastic mechanisms
                        for inel_mech in self.inelastic_scats:
                            g = self.array_from_kgrid("g_"+inel_mech, tp, c, T)
                            valley_transport[tp][inel_mech][c][T] = self.integrate_over_states(g * v, tp)

                        # from equation 45 in Rode, overall
                        g = self.array_from_kgrid("g", tp, c, T)
                        valley_transport[tp]['overall'][c][T] = self.integrate_over_states(g * v, tp)
                        g_th = self.array_from_kgrid("g", tp, c, T)
                        valley_transport[tp]["J_th"][c][T] = self.integrate_over_states(g_th * v, tp)
                        valley_transport[tp]["seebeck"][c][T] = self.integrate_over_states(f0_all*(1-f0_all)*E_array, tp)/(k_B*T)

                    # figure out average mobility
                    faulty_overall_mobility = False
                    mu_overrall_norm = norm(valley_transport[tp]['overall'][c][T])
                    mu_average = np.array([0.0, 0.0, 0.0])
                    for transport in self.elastic_scats + self.inelastic_scats:
                        # averaging all mobility values via Matthiessen's rule
                        mu_average += 1 / (np.array(valley_transport[tp][transport][c][T]) + 1e-32)
                        if mu_overrall_norm > norm(valley_transport[tp][transport][c][T]):
                            faulty_overall_mobility = True  # because the overall mobility should be lower than all
                        valley_transport[tp]["average"][c][T] = 1 / mu_average

                    # Decide if the overall mobility make sense or it should be equal to average (e.g. when POP is off)
                    if (mu_overrall_norm == 0.0 or faulty_overall_mobility) and not test_anisotropic:
                        self.logger.warning('There may be a problem with overall '
                                        'mobility; setting it to average...')
                        valley_transport[tp]['overall'][c][T] = valley_transport[tp]["average"][c][T]

                    if self.independent_valleys:
                        for mu in self.mo_labels+["J_th"]:
                            valley_transport[tp][mu][c][T] /= self.denominator[c][T][tp]
                        valley_transport[tp]['seebeck'][c][T] /= self.integrate_over_states(f0_all*(1-f0_all), tp)
        return valley_transport


    def calculate_transport_properties_with_E(self, important_points):
        """
        Mobility and Seebeck coefficient are calculated by integrating the
        perturbation of electron distribution and group velocity over the energy
        """
        valley_transport = {tp: {el_mech: {c: {T: np.array([0., 0., 0.]) for T in self.temperatures} for c in
                  self.dopings} for el_mech in self.transport_labels} for tp in ["n", "p"]}

        for c in self.dopings:
            for T in self.temperatures:
                for j, tp in enumerate(["p", "n"]):
                    # mobility numerators
                    for mu_el in self.elastic_scats:
                        valley_transport[tp][mu_el][c][T] = (-1) * default_small_E / hbar * \
                            self.integrate_over_E(prop_list=["/" + mu_el, "df0dk"], tp=tp, c=c, T=T,
                                        xDOS=False, xvel=True)

                    for mu_inel in self.inelastic_scats:
                        valley_transport[tp][mu_inel][c][T] = self.integrate_over_E(prop_list=[
                                "g_" + mu_inel], tp=tp, c=c, T=T, xDOS=False, xvel=True)
                        mu_overall_valley = self.integrate_over_E(prop_list=["g"],
                               tp=tp, c=c, T=T, xDOS=False, xvel=True)

                    valley_transport[tp]["J_th"][c][T] = (self.integrate_over_E(prop_list=["g_th"], tp=tp, c=c, T=T,
                            xDOS=False, xvel=True)) * e * abs(c)  # in units of A/cm2

                    faulty_overall_mobility = False
                    temp_avg = np.array([0.0, 0.0, 0.0])
                    for transport in self.elastic_scats + self.inelastic_scats:
                        temp_avg += 1/ valley_transport[tp][transport][c][T]
                        if norm(mu_overall_valley) > norm(valley_transport[tp][transport][c][T]):
                            faulty_overall_mobility = True  # because the overall mobility should be lower than all
                    valley_transport[tp]['average'][c][T] = 1 / temp_avg

                    if norm(mu_overall_valley) == 0.0 or faulty_overall_mobility:
                        valley_transport[tp]['overall'][c][T] = valley_transport[tp]['average'][c][T]
                    else:
                        valley_transport[tp]["overall"][c][T] = mu_overall_valley
                    self.egrid[tp]["relaxation time constant"][c][T] = self.mobility[tp]["overall"][c][T] \
                            * 1e-4 * m_e * self.cbm_vbm[tp]["eff_mass_xx"] / e  # 1e-4 to convert cm2/V.s to m2/V.s
                    if self.independent_valleys:
                        for mu in self.mo_labels+["J_th"]:
                            valley_transport[tp][mu][c][T] /= self.denominator[c][T][tp]
                        valley_transport[tp]["seebeck"][c][T] /= self.seeb_denom[c][T][tp]
        return valley_transport


    # for plotting
    def get_scalar_output(self, vec, dir):
        if dir == 'x':
            return vec[0]
        if dir == 'y':
            return vec[1]
        if dir == 'z':
            return vec[2]
        if dir == 'avg':
            return sum(vec) / 3


    def plot(self, k_plots=[], E_plots=[], mobility=True, concentrations='all', carrier_types=['n', 'p'],
             direction=['avg'], show_interactive=True, save_format=None, fontsize=30, ticksize=25, path=None, dir_name="plots",
             margins=100, fontfamily="serif"):
        """
        Plots the given k_plots and E_plots properties.

        Args:
            k_plots: (list of strings) the names of the quantities to be plotted against norm(k)
                options: 'energy', 'df0dk', 'velocity', or just string 'all' (not in a list) to plot everything
            E_plots: (list of strings) the names of the quantities to be plotted against E
                options: 'frequency', 'relaxation time', '_all_elastic', 'df0dk', 'velocity', 'ACD', 'IMP', 'PIE', 'g',
                'g_POP', 'g_th', 'S_i', 'S_o', or just string 'all' (not in a list) to plot everything
            mobility: (boolean) if True, create a mobility against temperature plot
            concentrations: (list of strings) a list of carrier concentrations, or the string 'all' to plot the
                results of calculations done with all input concentrations
            carrier_types: (list of strings) select carrier types to plot data for - ['n'], ['p'], or ['n', 'p']
            direction: (list of strings) options to include in list are 'x', 'y', 'z', 'avg'; determines which
                components of vector quantities are plotted
            show_interactive: (boolean) if True creates and shows interactive html plots
            save_format: (str) format for saving plots; options are 'png', 'jpeg', 'svg', 'pdf', None (None does not
                save the plots). NOTE: plotly credentials are needed, see figrecipes documentation
            fontsize: (int) size of title and axis label text
            ticksize: (int) size of axis tick label text
            path: (string) location to save plots
            margins: (int) figrecipes plotly margins
            fontfamily: (string) plotly font
        """
        path = os.path.join(path or self.calc_dir, dir_name)
        if not os.path.exists(path):
            os.makedirs(name=path)
        supported_k_plots = ['energy', 'df0dk', 'velocity'] + self.elastic_scats
        supported_E_plots = ['frequency', 'relaxation time', 'df0dk', 'velocity'] + self.elastic_scats
        if "POP" in self.inelastic_scats:
            supported_E_plots += ['g', 'g_POP', 'S_i', 'S_o']
            supported_k_plots += ['g', 'g_POP', 'S_i', 'S_o']
        if k_plots == 'all':
            k_plots = supported_k_plots
        if E_plots == 'all':
            E_plots = supported_E_plots
        if concentrations == 'all':
            concentrations = self.dopings

        # make copies of mutable arguments
        k_plots = list(k_plots)
        E_plots = list(E_plots)
        concentrations = list(concentrations)
        carrier_types = list(carrier_types)
        direction = list(direction)

        mu_list = ["overall", "average"] + self.elastic_scats + self.inelastic_scats

        # separate temperature dependent and independent properties
        all_temp_independent_k_props = ['energy', 'velocity']
        all_temp_independent_E_props = ['frequency', 'velocity']
        temp_independent_k_props = []
        temp_independent_E_props = []
        temp_dependent_k_props = []
        for prop in k_plots:
            if prop not in supported_k_plots:
                raise AmsetError(self.logger,
                                 'No support for {} vs. k plot!'.format(prop))
            if prop in all_temp_independent_k_props:
                temp_independent_k_props.append(prop)
            else:
                temp_dependent_k_props.append(prop)
        temp_dependent_E_props = []
        for prop in E_plots:
            if prop not in supported_E_plots:
                raise AmsetError(self.logger,
                                 'No support for {} vs. E plot!'.format(prop))
            if prop in all_temp_independent_E_props:
                temp_independent_E_props.append(prop)
            else:
                temp_dependent_E_props.append(prop)

        vec = {'energy': False,
               'velocity': True,
               'frequency': False}

        for tp in carrier_types:
            x_data = {'k': self.kgrid0[tp]["norm(k)"][0],
                      'E': [E - self.cbm_vbm[tp]["energy"] for E in self.egrid0[tp]["energy"]]}
            x_axis_label = {'k': 'norm(k)', 'E': 'energy (eV)'}

            for c in concentrations:

                # plots of scalar properties first
                tp_c = tp + '_' + str(c)
                for x_value, y_values in [('k', temp_independent_k_props), ('E', temp_independent_E_props)]:
                    y_data_temp_independent = {'k': {'energy': self.kgrid0[tp]['energy'][0],
                                                     'velocity': self.kgrid0[tp]["norm(v)"][0]},
                                               'E': {'frequency': self.Efrequency0[tp]}}
                    for y_value in y_values:
                        if not vec[y_value]:
                            title = None
                            if y_value == 'frequency':
                                title = 'Energy Histogram for {}, c={}'.format(self.tp_title[tp], c)
                            create_plots(x_axis_label[x_value], y_value, show_interactive, save_format, c, tp, tp_c,
                                              fontsize, ticksize, path, margins, fontfamily, plot_data=[(x_data[x_value], y_data_temp_independent[x_value][y_value])],
                                              x_label_short=x_value, title=title)


                for dir in direction:
                    y_data_temp_independent = {'k': {'energy': self.kgrid0[tp]['energy'][0],
                                                     'velocity': self.kgrid0[tp]["norm(v)"][0]},
                                               'E': {'frequency': self.Efrequency0[tp],
                                                     'velocity': [self.get_scalar_output(p, dir) for p in self.egrid0[tp]['velocity']]}}

                    tp_c_dir = tp_c + '_' + dir

                    # temperature independent k and E plots: energy(k), velocity(k), histogram(E), velocity(E)
                    for x_value, y_values in [('k', temp_independent_k_props), ('E', temp_independent_E_props)]:
                        for y_value in y_values:
                            if vec[y_value]:
                                create_plots(x_axis_label[x_value], y_value, show_interactive,
                                                  save_format, c, tp, tp_c_dir,
                                                  fontsize, ticksize, path, margins, fontfamily, plot_data=(x_data[x_value], y_data_temp_independent[x_value][y_value]), x_label_short=x_value)

                    # want variable of the form: y_data_temp_dependent[k or E][prop][temp] (the following lines reorganize
                    try:
                        y_data_temp_dependent = {'k': {prop: {T: [self.get_scalar_output(p, dir) for p in self.kgrid0[tp][prop][c][T][0]]
                                                                for T in self.temperatures} for prop in temp_dependent_k_props},
                                                'E': {prop: {T: [self.get_scalar_output(p, dir) for p in self.egrid0[tp][prop][c][T]]
                                                                for T in self.temperatures} for prop in temp_dependent_E_props}}
                    except KeyError: # for when from_file is called
                        y_data_temp_dependent = {'k': {prop: {T: [self.get_scalar_output(p, dir) for p in self.kgrid0[tp][prop][str(c)][str(int(T))][0]]
                                                                for T in self.temperatures} for prop in temp_dependent_k_props},
                                                'E': {prop: {T: [self.get_scalar_output(p, dir) for p in self.egrid0[tp][prop][str(c)][str(int(T))]]
                                                                for T in self.temperatures} for prop in temp_dependent_E_props}}

                    # temperature dependent k and E plots
                    for x_value, y_values in [('k', temp_dependent_k_props), ('E', temp_dependent_E_props)]:
                        for y_value in y_values:
                            plot_data = []
                            names = []
                            for T in self.temperatures:
                                plot_data.append((x_data[x_value], y_data_temp_dependent[x_value][y_value][T]))
                                names.append(str(T) + ' K')
                            create_plots(x_axis_label[x_value], y_value, show_interactive,
                                              save_format, c, tp, tp_c_dir,
                                              fontsize, ticksize, path, margins, fontfamily, plot_data=plot_data,
                                              x_label_short=x_value, names=names)

                    # mobility plots as a function of temperature (the only plot that does not have k or E on the x axis)
                    if mobility:
                        plot_data = []
                        names = []
                        for mo in mu_list:
                            try:
                                mo_values = [self.mobility[tp][mo][c][T] for T in self.temperatures]
                            except KeyError: # for when from_file is called
                                mo_values = [self.mobility[tp][mo][str(c)][str(int(T))] for T in self.temperatures]
                            plot_data.append((self.temperatures, [self.get_scalar_output(mo_value,
                                    dir) for mo_value in mo_values]))
                            names.append(mo)

                        create_plots("Temperature (K)",
                                "Mobility (cm2/V.s)", show_interactive,
                                save_format, c, tp, tp_c_dir, fontsize-5,
                                ticksize-5, path, margins,
                                fontfamily, plot_data=plot_data, names=names, mode='lines+markers',
                                y_label_short="mobility", y_axis_type='log')



    def to_csv(self, path=None, dir_name="run_data", csv_filename='amset_results.csv'):
        """
        Writes the calculated transport properties to a csv file.

        Args:
            csv_filename (str):

        Returns (.csv file)
        """
        import csv
        path = os.path.join(path or self.calc_dir, dir_name)
        if not os.path.exists(path):
            os.makedirs(name=path)

        with open(os.path.join(path, csv_filename), 'w') as csvfile:
            fieldnames = ['type', 'c(cm-3)', 'T(K)', 'overall', 'average'] + \
                         self.elastic_scats + self.inelastic_scats + ['seebeck']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for c in self.dopings:
                tp = get_tp(c)
                for T in self.temperatures:
                    row = {'type': tp, 'c(cm-3)': abs(c), 'T(K)': T}
                    for p in ['overall', 'average'] + self.elastic_scats + self.inelastic_scats + ["seebeck"]:
                        row[p] = sum(self.mobility[tp][p][c][T])/3
                    writer.writerow(row)


    def test_run(self):
        important_pts = [self.cbm_vbm["n"]["kpoint"]]
        if (np.array(self.cbm_vbm["p"]["kpoint"]) != np.array(self.cbm_vbm["n"]["kpoint"])).any():
            important_pts.append(self.cbm_vbm["p"]["kpoint"])

        points_1d = generate_k_mesh_axes(important_pts, kgrid_tp='very coarse')
        self.kgrid_array = create_grid(points_1d)
        kpts = array_to_kgrid(self.kgrid_array)

        self.k_hat_array = normalize_array(self.kgrid_array)

        self.dv_grid = self.find_dv(self.kgrid_array)

        k_x = self.kgrid_array[:, :, :, 0]
        k_y = self.kgrid_array[:, :, :, 1]
        k_z = self.kgrid_array[:, :, :, 2]
        result = self.integrate_over_k(np.cos(k_x))
        print(result)


if __name__ == "__main__":
    mass = 0.25
    use_poly_bands = False

    model_params = {'bs_is_isotropic': True,
                    'elastic_scats': ['ACD', 'IMP', 'PIE'],
                    'inelastic_scats': ['POP']
        , 'independent_valleys': False
                    }
    if use_poly_bands:
        model_params["poly_bands"] = [[
            [[0.0, 0.0, 0.0], [0.0, mass]],
        ]]

    performance_params = {"dE_min": 0.0001, "nE_min": 5,
                          "BTE_iters": 5,
                          "max_nbands": 1,
                          "max_normk": None,
                          "n_jobs": -1,
                          "max_nvalleys": 1,
                          "interpolation": "boltztrap2",
                          "Ecut_max": 1.0
                          }

    # material_params = {"epsilon_s": 12.9, "epsilon_inf": 10.9, "W_POP": 8.73, # experimental
    material_params = {"epsilon_s": 12.18, "epsilon_inf": 10.32, "W_POP": 8.16, # ab initio (lower overall mobility)
            "C_el": 139.7, "E_D": {"n": 8.6, "p": 8.6}, "P_PIE": 0.052
            , "user_bandgap": 1.54,
            # "important_points": {'n': [[0. , 0.5, 0. ]], 'p': [[0. , 0.0, 0. ]]},
                       }
    input_dir = "../test_files/GaAs/nscf-uniform"
    coeff_file = os.path.join(input_dir, "fort.123")

    amset = Amset(calc_dir='.',
                  vasprun_file=os.path.join(input_dir, "vasprun.xml"),
                  material_params=material_params,
                  model_params=model_params,
                  performance_params=performance_params,
                  dopings = [-3e13],
                  # dopings = [5.10E+18, 7.10E+18, 1.30E+19, 2.80E+19, 6.30E+19],
                  temperatures = [300],
                  # temperatures = [300, 600, 1000],
                  integration='k',
                  )
    amset.run_profiled(coeff_file, kgrid_tp='very coarse', write_outputs=True)

    amset.write_input_files()
    amset.to_csv()
    # amset.to_file()
    amset.plot(k_plots=['energy', 'S_o', 'S_i']
               , E_plots=['velocity', 'df0dk', 'ACD'], show_interactive=True
               , carrier_types=amset.all_types
               , save_format=None)

    amset.to_json(kgrid=True, trimmed=True, max_ndata=100, nstart=0)
