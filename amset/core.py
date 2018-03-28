# coding: utf-8
from __future__ import absolute_import
import gzip
import warnings
import time
import json
from pstats import Stats
from random import random

from pymatgen.symmetry.bandstructure import HighSymmKpath
from scipy.interpolate import griddata
from pprint import pprint
import os
from sys import stdout as STDOUT

import numpy as np
from math import log, pi
from pymatgen.electronic_structure.boltztrap import BoltztrapRunner, \
    BoltztrapAnalyzer
from pymatgen.io.vasp import Vasprun, Spin, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from monty.json import MontyEncoder, MontyDecoder
import cProfile
from copy import deepcopy
import multiprocessing
from joblib import Parallel, delayed

from amset.utils.analytical_band_from_BZT import Analytical_bands, outer, get_dos_from_poly_bands, get_energy, get_poly_energy

from amset.utils.tools import norm, grid_norm, generate_k_mesh_axes, \
    create_grid, array_to_kgrid, normalize_array, f0, df0dE, cos_angle, \
    fermi_integral, GB, calculate_Sio, calculate_Sio_list, remove_from_grid, \
    get_tp, \
    remove_duplicate_kpoints, get_angle, sort_angles, get_closest_k, \
    get_energy_args, get_bindex_bspin, \
    AmsetError, kpts_to_first_BZ, get_dos_boltztrap2, \
    setup_custom_logger, insert_intermediate_kpoints

from amset.utils.constants import hbar, m_e, Ry_to_eV, A_to_m, m_to_cm, \
    A_to_nm, e, k_B, \
    epsilon_0, default_small_E, dTdz, sq3, Hartree_to_eV

try:
    import BoltzTraP2
    import BoltzTraP2.dft
    from BoltzTraP2 import sphere, fite
except ImportError:
    warnings.warn('BoltzTraP2 not imported, "boltztrap2" interpolation not available.')

__author__ = "Alireza Faghaninia, Jason Frost, Anubhav Jain"
__copyright__ = "Copyright 2017, HackingMaterials"
__version__ = "0.1.0"
__maintainer__ = "Alireza Faghaninia"
__email__ = "alireza.faghaninia@gmail.com"
__status__ = "Development"



class AMSET(object):
    """ This class is used to run AMSET on a pymatgen from a VASP run (i.e. vasprun.xml). AMSET is an ab initio model
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
    def __init__(self, calc_dir, material_params, model_params={}, performance_params={},
                 dopings=None, temperatures=None, k_integration=True, e_integration=False, fermi_type='k', loglevel=None):
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
        self.dopings = dopings or [-1e20, 1e20]
        self.all_types = list(set([get_tp(c) for c in self.dopings]))
        self.tp_title = {"n": "conduction band(s)", "p": "valence band(s)"}
        self.temperatures = temperatures or [300.0, 600.0]
        self.set_model_params(model_params)
        self.logger.info('independent_valleys: {}'.format(self.independent_valleys))
        self.set_material_params(material_params)
        self.set_performance_params(performance_params)
        self.k_integration = k_integration
        self.e_integration = e_integration
        assert(self.k_integration != self.e_integration), "AMSET can do either k_integration or e_integration"
        self.logger.info('k_integration: {}'.format(self.k_integration))
        self.logger.info('e_integration: {}'.format(self.e_integration))
        self.fermi_calc_type = fermi_type

        self.num_cores = max(int(multiprocessing.cpu_count()/4), self.max_ncpu)
        if self.parallel:
            self.logger.info("number of cpu used in parallel mode: {}".format(self.num_cores))
        self.counter = 0 # a global counter just for debugging
        self.offset_from_vrun = {'n': 0.0, 'p': 0.0}

    def run_profiled(self, coeff_file=None, kgrid_tp="coarse", write_outputs=True):
        profiler = cProfile.Profile()
        profiler.runcall(lambda: self.run(coeff_file, kgrid_tp=kgrid_tp,
                                           write_outputs=write_outputs))
        stats = Stats(profiler, stream=STDOUT)
        stats.strip_dirs()
        stats.sort_stats('cumulative')
        stats.print_stats(15)  # only print the top 10 (10 slowest functions)


    def run(self, coeff_file=None, kgrid_tp="coarse", write_outputs=True, test_k_anisotropic=False):
        """
        Function to run AMSET and generate the main outputs.

        Args:
        coeff_file: the path to fort.123* file containing the coefficients of
            the interpolated band structure generated by a modified version of
            BoltzTraP. If None, BoltzTraP will run to generate the file.
        kgrid_tp (str): define the density of k-point mesh.
            options: 'very coarse', 'coarse', 'fine'
        """
        self.logger.info('Running on "{}" mesh for each valley'.format(kgrid_tp))
        self.read_vrun(calc_dir=self.calc_dir, filename="vasprun.xml")
        if self.poly_bands0 is not None:
            self.cbm_vbm["n"]["energy"] = self.dft_gap
            self.cbm_vbm["p"]["energy"] = 0.0
            self.cbm_vbm["n"]["kpoint"] = self.cbm_vbm["p"]["kpoint"] = \
            self.poly_bands0[0][0][0]

        if not coeff_file:
            self.logger.warning('\nRunning BoltzTraP to generate the cube file...')
            boltztrap_runner = BoltztrapRunner(bs=self.bs, nelec=self.nelec,
                    # doping=list(set([abs(d) for d in self.dopings])),
                    doping=[1e20], tmax=max(self.temperatures))
            boltztrap_runner.run(path_dir=self.calc_dir)
            # BoltztrapRunner().run(path_dir=self.calc_dir)
            coeff_file = os.path.join(self.calc_dir, 'boltztrap', 'fort.123')
            self.logger.warning('BoltzTraP run finished, I suggest to set the following '
                            'to skip this step next time:\n{}="{}"'.format(
                "coeff_file", os.path.join(self.calc_dir, 'boltztrap', 'fort.123')
            ))
            if not os.path.exists(coeff_file):
                raise AmsetError(self.logger,
                        '{} does not exist! generating the cube file '
                '(i.e. fort.123) requires a modified version of BoltzTraP. '
                             'Contact {}'.format(coeff_file, __email__))

        self.mo_labels = self.elastic_scatterings + self.inelastic_scatterings + ['overall', 'average']
        self.spb_labels = ['SPB_ACD']
        self.mobility = {tp: {el_mech: {c: {T: np.array([0., 0., 0.]) for T in self.temperatures} for c in
                  self.dopings} for el_mech in self.mo_labels+self.spb_labels} for tp in ["n", "p"]}
        self.calc_doping = {c: {T: {'n': None, 'p': None} for T in self.temperatures} for c in self.dopings}

        # make the reference energy consistent w/ interpolation rather than DFT
        self.update_cbm_vbm_dos(coeff_file=coeff_file)

        #TODO: if we use ibands_tuple, then for each couple of conduction/valence bands we only use 1 band together (i.e. always ib==0)
        for tp in ['p', 'n']:
            self.cbm_vbm[tp]['included'] = 1

        self.logger.debug("cbm_vbm after recalculating their energy values:\n {}".format(self.cbm_vbm))

        self.ibrun = 0 # initialize as may be called in init_kgrid as debug
        self.count_mobility = [{'n': True, 'p': True} for _ in range(max(self.initial_num_bands['p'], self.initial_num_bands['n']))]

        if self.pre_determined_fermi is None:
            if self.fermi_calc_type == 'k':
                kpts = self.generate_kmesh(important_points={'n': [[0.0, 0.0, 0.0]], 'p': [[0.0, 0.0, 0.0]]}, kgrid_tp=self.fermi_kgrid_tp)
                # the purpose of the following line is just to generate self.energy_array that find_fermi_k function uses
                analytical_band_tuple, kpts, energies = self.get_energy_array(coeff_file, kpts, once_called=False, return_energies=True, num_bands=self.initial_num_bands, nbelow_vbm=0, nabove_cbm=0)
                self.fermi_level = self.find_fermi_k(num_bands=self.initial_num_bands)
            elif self.fermi_calc_type == 'e':
                kpts = self.generate_kmesh(important_points={'n': [[0.0, 0.0, 0.0]], 'p': [[0.0, 0.0, 0.0]]}, kgrid_tp='very coarse')
                analytical_band_tuple, kpts= self.get_energy_array(coeff_file, kpts, once_called=False, return_energies=False, num_bands=self.initial_num_bands, nbelow_vbm=0, nabove_cbm=0)
                # self.init_kgrid(kpts, important_points={'n': [[0.0, 0.0, 0.0]], 'p': [[0.0, 0.0, 0.0]]}, analytical_band_tuple=analytical_band_tuple, delete_off_points=False)
                # self.pre_init_egrid(once_called=False, dos_tp='standard')
                self.fermi_level = {c: {T: None for T in self.temperatures} for c in self.dopings}
                for c in self.dopings:
                    for T in self.temperatures:
                        self.fermi_level[c][T] = self.find_fermi(c, T)
        else:
        ## uncomment the following only for quick testing if fermi_levels are known
            self.fermi_level = self.pre_determined_fermi
            self.calc_doping = {doping: {T: {'n': 0.0, 'p': 0.0} for T in list(self.fermi_level[doping].keys())} for doping in list(self.fermi_level.keys())}
            for doping in list(self.fermi_level.keys()):
                for T in list(self.fermi_level[doping].keys()):
                    if doping > 0:
                        self.calc_doping[doping][T]['p'] = doping
                    else:
                        self.calc_doping[doping][T]['n'] = doping
        self.logger.info('fermi level = {}'.format(self.fermi_level))

        # self.find_fermi_boltztrap()

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
        #TODO: this ibands_tuple is to treat each band (and valleys with) independently (so each time num_bands will be {'n': 1, 'p': 1} but with different band indexes
        if self.max_nbands:
            ibands_tuple = ibands_tuple[:min(len(ibands_tuple), self.max_nbands)]

        self.logger.debug('here ibands_tuple')
        self.logger.debug(ibands_tuple)

        self.logger.debug('here whether to count bands')
        self.logger.debug(self.count_mobility)


        self.denominator = {c: {T: {'p': 0.0, 'n': 0.0} for T in self.temperatures} for c in self.dopings}
        for self.ibrun, (self.nbelow_vbm, self.nabove_cbm) in enumerate(ibands_tuple):
            self.logger.info('going over conduction and valence # {}'.format(self.ibrun))
            self.find_all_important_points(coeff_file, nbelow_vbm=self.nbelow_vbm, nabove_cbm=self.nabove_cbm)

            # once_called = False
            max_nvalleys = max(len(self.important_pts['n']), len(self.important_pts['p']))
            if self.max_nvalleys is not None:
                max_nvalleys = min(max_nvalleys, self.max_nvalleys)

            for ivalley in range(max_nvalleys):
                self.count_mobility[self.ibrun] = self.count_mobility0[self.ibrun]
                once_called = True
                important_points = {'n': None, 'p': None}
                if ivalley == 0 and self.ibrun==0 and self.pre_determined_fermi is not None:
                    once_called = False
                for tp in ['p', 'n']:
                    try:
                        important_points[tp] = [self.important_pts[tp][ivalley]]
                    except:
                        important_points[tp] = [self.important_pts[tp][0]]
                        self.count_mobility[self.ibrun][tp] = False

                if self.max_normk0 is None:
                    for tp in ['n', 'p']:
                        min_dist = 20.0
                        for k in self.bs.get_sym_eq_kpoints(important_points[tp][0]): # we use the one and only k inside important_points[tp] since bs.get_sym_eq_kpoints return a list by itself
                            new_dist = norm(self.get_cartesian_coords(get_closest_k(k, self.important_pts[tp], return_diff=True, threshold=0.01)) /A_to_nm )
                            # print('here dist')
                            # print get_closest_k(k, self.important_pts[tp][0], return_diff=True)
                            # print(self.important_pts[tp][0])
                            # print(new_dist)
                            if new_dist < min_dist and new_dist > 0.01: # to avoid self-counting, 0.01 criterion added
                                min_dist = new_dist
                        self.max_normk[tp] = min_dist/2.0
                if self.max_nvalleys and self.max_nvalleys==1:
                    # this ignores max_normk0 because if only a single valley, we don't want it to go over the whole BZ
                    self.max_normk = {'n': self.max_normk0 or 2,
                                      'p': self.max_normk0 or 2}
                self.logger.info('at valence band #{} and conduction band #{}'.format(self.nbelow_vbm, self.nabove_cbm))
                self.logger.info('Current valleys:\n{}'.format(important_points))
                self.logger.info('Whether to count valleys: {}'.format(self.count_mobility[self.ibrun]))
                self.logger.info('max_normk:\n{}'.format(self.max_normk))
                self.logger.info('important points for this band:\n{}'.format(self.important_pts))

                if not self.count_mobility[self.ibrun]['n'] and not self.count_mobility[self.ibrun]['p']:
                    self.logger.info('skipping this valley as it is unimportant for both n and p type...')
                    continue

                kpts = self.generate_kmesh(important_points=important_points, kgrid_tp=kgrid_tp)
                analytical_band_tuple, kpts, energies = self.get_energy_array(coeff_file, kpts, once_called=once_called, return_energies=True, nbelow_vbm=self.nbelow_vbm, nabove_cbm=self.nabove_cbm, num_bands={'p': 1, 'n': 1})

                if min(energies['n']) - self.cbm_vbm['n']['energy'] > self.Ecut['n']:
                    self.logger.debug('not counting conduction band {} valley {} due to off enery...'.format(self.ibrun, important_points['n'][0]))
                    # print('here debug')
                    # print(min(energies['n']))
                    # print(self.cbm_vbm['n']['energy'])
                    self.count_mobility[self.ibrun]['n'] = False
                if self.cbm_vbm['p']['energy'] - max(energies['p']) > self.Ecut['p']:
                    self.logger.debug('not counting valence band {} valley {} due to off enery...'.format(self.ibrun, important_points['p'][0]))
                    # print('here debug')
                    # print(max(energies['p']))
                    # print(self.cbm_vbm['p']['energy'])
                    self.count_mobility[self.ibrun]['p'] = False

                if not self.count_mobility[self.ibrun]['n'] and not self.count_mobility[self.ibrun]['p']:
                    self.logger.info('skipping this valley as it is unimportant or its energies are way off...')
                    continue

                corrupt_tps = self.init_kgrid(kpts, important_points, analytical_band_tuple, once_called=once_called)
                # self.logger.debug('here new energy_arrays:\n{}'.format(self.energy_array['n']))
                for tp in corrupt_tps:
                    self.count_mobility[self.ibrun][tp] = False

                if not self.count_mobility[self.ibrun]['n'] and not self.count_mobility[self.ibrun]['p']:
                    self.logger.info('skipping this valley as it is unimportant or its energies are way off...')
                    continue

                # for now, I keep once_called as False in init_egrid until I get rid of egrid mobilities
                self.init_egrid(once_called=False, dos_tp="standard")
                self.bandgap = min(self.egrid["n"]["all_en_flat"]) - max(self.egrid["p"]["all_en_flat"])
                if abs(self.bandgap - (self.cbm_vbm["n"]["energy"] - self.cbm_vbm["p"]["energy"] + self.scissor)) > k_B * 300:
                    warnings.warn("The band gaps do NOT match! The selected k-mesh is probably too coarse.")
                    # raise ValueError("The band gaps do NOT match! The selected k-mesh is probably too coarse.")

                # initialize g in the egrid
                self.map_to_egrid("g", c_and_T_idx=True, prop_type="vector")
                self.map_to_egrid(prop_name="velocity", c_and_T_idx=False, prop_type="vector")


                if self.independent_valleys:
                    for c in self.dopings:
                        for T in self.temperatures:
                            if self.k_integration:
                                f0_all = 1 / (np.exp((self.energy_array['n'] - self.fermi_level[c][T]) / (k_B * T)) + 1)
                                f0p_all = 1 / (np.exp((self.energy_array['p'] - self.fermi_level[c][T]) / (k_B * T)) + 1)
                                self.denominator[c][T]['n'] = (3 * default_small_E * self.integrate_over_states(f0_all, 'n') + 1e-10)
                                self.denominator[c][T]['p'] = (3 * default_small_E * self.integrate_over_states(1-f0p_all, 'p') + 1e-10)
                            elif self.e_integration:
                                self.denominator[c][T]['n'] = 3 * default_small_E * self.integrate_over_E(prop_list=["f0"], tp='n', c=c, T=T, xDOS=False, xvel=False, weighted=False)
                                self.denominator[c][T]['p'] = 3 * default_small_E * self.integrate_over_E(prop_list=["1 - f0"], tp='p', c=c, T=T, xDOS=False, xvel=False, weighted=False)

                # find the indexes of equal energy or those with ±hbar*W_POP for scattering via phonon emission and absorption
                if not self.bs_is_isotropic or "POP" in self.inelastic_scatterings:
                    self.generate_angles_and_indexes_for_integration()

                # calculate all elastic scattering rates in kgrid and then map it to egrid:
                for sname in self.elastic_scatterings:
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
                        dop_tp = get_tp(c)
                        f0_removed = self.array_from_kgrid('f0', dop_tp, c, T)
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

                if self.k_integration:
                    valley_mobility = self.calculate_transport_properties_with_k(test_k_anisotropic, important_points)
                if self.e_integration:
                    valley_mobility = self.calculate_transport_properties_with_E(important_points)
                print('mobility of the valley {} and band (p, n) {}'.format(important_points, self.ibands_tuple[self.ibrun]))
                print('count_mobility: {}'.format(self.count_mobility[self.ibrun]))
                pprint(valley_mobility)

                self.calculate_spb_transport()

                self.logger.info('Mobility Labels: {}'.format(self.mo_labels))
                for c in self.dopings:
                    for T in self.temperatures:
                        if self.count_mobility[self.ibrun][tp]:
                            if not self.independent_valleys:
                                if self.k_integration:
                                    f0_all = 1 / (np.exp((self.energy_array['n'] - self.fermi_level[c][T]) / (k_B * T)) + 1)
                                    f0p_all = 1 / (np.exp((self.energy_array['p'] - self.fermi_level[c][T]) / (k_B * T)) + 1)
                                    # if denominator is defined as a single common denominator, += if specific to each valley, self.denominator[c][T][tp] = ...
                                    self.denominator[c][T]['n'] += 3 * default_small_E * self.integrate_over_states(f0_all, 'n') + 1e-10
                                    self.denominator[c][T]['p'] += 3 * default_small_E * self.integrate_over_states(1-f0p_all, 'p') + 1e-10
                                elif self.e_integration:
                                    self.denominator[c][T]['n'] += 3 * default_small_E * self.integrate_over_E(prop_list=["f0"], tp='n', c=c, T=T, xDOS=False, xvel=False, weighted=False)  * self.bs.get_kpoint_degeneracy(important_points['n'][0])
                                    self.denominator[c][T]['p'] += 3 * default_small_E * self.integrate_over_E(prop_list=["1 - f0"], tp='p', c=c, T=T, xDOS=False, xvel=False, weighted=False) * self.bs.get_kpoint_degeneracy(important_points['p'][0])

                                ## with degeneracy multiplied (I think this is relevant only if all valleys are considered!
                                # if self.k_integration:
                                #     f0_all = 1 / (np.exp((self.energy_array['n'] - self.fermi_level[c][T]) / (k_B * T)) + 1)
                                #     f0p_all = 1 / (np.exp((self.energy_array['p'] - self.fermi_level[c][T]) / (k_B * T)) + 1)
                                #     # if denominator is defined as a single common denominator, += if specific to each valley, self.denominator[c][T][tp] = ...
                                #     self.denominator[c][T]['n'] += (3 * default_small_E * self.integrate_over_states(f0_all, 'n') + 1e-10) * self.bs.get_kpoint_degeneracy(important_points['n'][0])
                                #     self.denominator[c][T]['p'] += (3 * default_small_E * self.integrate_over_states(1-f0p_all, 'p') + 1e-10) *self.bs.get_kpoint_degeneracy(important_points['p'][0])
                                # elif self.e_integration:
                                #     self.denominator[c][T]['n'] += 3 * default_small_E * self.integrate_over_E(prop_list=["f0"], tp='n', c=c, T=T, xDOS=False, xvel=False, weighted=False) * self.bs.get_kpoint_degeneracy(important_points['n'][0])
                                #     self.denominator[c][T]['p'] += 3 * default_small_E * self.integrate_over_E(prop_list=["1 - f0"], tp='p', c=c, T=T, xDOS=False, xvel=False, weighted=False)* self.bs.get_kpoint_degeneracy(important_points['p'][0])


                                for mu in self.mo_labels:
                                    for tp in ['p', 'n']:
                                        self.mobility[tp][mu][c][T] += valley_mobility[tp][mu][c][T] * self.bs.get_kpoint_degeneracy(important_points[tp][0])
                            else:
                                for mu in self.mo_labels:
                                    for tp in ['p', 'n']:
                                        self.mobility[tp][mu][c][T] += valley_mobility[tp][mu][c][T]

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
                            self.valleys[tp]['band {}'.format(self.ibrun)]['{};{};{}'.format(k[0], k[1], k[2])] = valley_mobility[tp]

                kgrid_rm_list = ["effective mass", "kweights",
                                 "f_th", "S_i_th", "S_o_th"]
                self.kgrid = remove_from_grid(self.kgrid, kgrid_rm_list)
                if ivalley==0 and self.ibrun==0:
                    # TODO: make it possible for the user to choose which valley(s) to plot
                    self.kgrid0 = deepcopy(self.kgrid)
                    self.egrid0 = deepcopy(self.egrid)
                    self.Efrequency0 = deepcopy(self.Efrequency)


        # print('here debug mobility')
        # print(self.mobility['n'])
        # print()
        self.logger.debug('here denominator:\n{}'.format(self.denominator))

        if not self.independent_valleys:
            for tp in ['p', 'n']:
                for mu in self.mo_labels:
                    for c in self.dopings:
                        for T in self.temperatures:
                            self.mobility[tp][mu][c][T] /= self.denominator[c][T][tp]
                            for band in list(self.valleys[tp].keys()):
                                for valley_k in list(self.valleys[tp][band].keys()):
                                    self.valleys[tp][band][valley_k][mu][c][T] /= self.denominator[c][T][tp]


        print('\nFinal Mobility Values:')
        pprint(self.mobility)

        if write_outputs:
            self.to_file()


    def calc_analytical_energy(self, kpt, engre, nwave, nsym, nstv, vec, vec2,
                               out_vec2,
                               br_dir, sgn, scissor=0.0):
        """
        Args:
            kpt ([1x3 array]): fractional coordinates of the k-point
            engre, nwave, nsym, stv, vec, vec2, out_vec2, br_dir: all obtained via
                get_energy_args
            sgn (int): options are +1 for valence band and -1 for conduction bands
                sgn is basically ignored (doesn't matter) if scissor==0.0
            scissor (float): the amount by which the band gap is modified/scissored
        Returns:

        """
        energy, de, dde = get_energy(kpt, engre, nwave, nsym, nstv, vec, vec2,
                                     out_vec2, br_dir=br_dir)
        energy = energy * Ry_to_eV - sgn * scissor / 2.0
        # velocity = abs(self.get_cartesian_coords(de, reciprocal=False) / hbar * A_to_m * m_to_cm * Ry_to_eV)
        velocity = abs(self.get_cartesian_coords(de, reciprocal=False)) / (hbar * 2 * pi) / 0.52917721067 * A_to_m * m_to_cm * Ry_to_eV

        # effective_m = hbar ** 2 / (dde * 4 * pi ** 2) / m_e / A_to_m ** 2 * e * Ry_to_eV
        effective_m = 1/(dde/ 0.52917721067) * e / Ry_to_eV / A_to_m**2 * (hbar*2*np.pi)**2 / m_e

        return energy, velocity, effective_m


    def get_bs_extrema(self, bs, coeff_file=None, interp_params=None,
                       interpolation="boltztrap1",
                       nk_ibz=17, v_cut=1e4, min_normdiff=0.05,
                       Ecut=None, nex_max=0, return_global=False, niter=5,
                       nbelow_vbm=0, nabove_cbm=0, scissor=0.0):
        """
        returns a dictionary of p-type (valence) and n-type (conduction) band
            extrema k-points by looking at the 1st and 2nd derivatives of the bands
        Args:
            bs (pymatgen BandStructure object): must containt Structure and have
                the same number of valence electrons and settings as the vasprun.xml
                from which coeff_file is generated.
            coeff_file (str): path to the cube file from BoltzTraP run
            nk_ibz (int): maximum number of k-points in one direction in IBZ
            v_cut (float): threshold under which the derivative is assumed 0 [cm/s]
            min_normdiff (float): the minimum allowed distance norm(fractional k)
                in extrema; this is important to avoid numerical instability errors
            Ecut (float or dict): max energy difference with CBM/VBM allowed for
                extrema
            nex_max (int): max number of low-velocity kpts tested for being extrema
            return_global (bool): in addition to the extrema, return the actual
                CBM (global minimum) and VBM (global maximum) w/ their k-point
            niter (int): number of iterations in basinhoopping for finding the
                global extremum
            nbelow_vbm (int): # of bands below the last valence band
            nabove_vbm (int): # of bands above the first conduction band
            scissor (float): the amount by which the band gap is altered/scissored.
        Returns (dict): {'n': list of extrema fractional coordinates, 'p': same}
        """
        # TODO: MAJOR cleanup needed in this function; also look into parallelizing get_analytical_energy at all kpts if it's time consuming
        # TODO: if decided to only include one of many symmetrically equivalent extrema, write a method to keep only one of symmetrically equivalent extrema as a representative
        Ecut = Ecut or 10 * k_B * 300
        if not isinstance(Ecut, dict):
            Ecut = {'n': Ecut, 'p': Ecut}
        actual_cbm_vbm = {'n': {}, 'p': {}}
        vbm_idx, _ = get_bindex_bspin(bs.get_vbm(), is_cbm=False)
        # vbm_idx = bs.get_vbm()['band_index'][Spin.up][0]
        ibands = [1 - nbelow_vbm,
                  2 + nabove_cbm]  # in this notation, 1 is the last valence band
        ibands = [i + vbm_idx for i in ibands]
        ibz = HighSymmKpath(bs.structure)
        sg = SpacegroupAnalyzer(bs.structure)
        kmesh = sg.get_ir_reciprocal_mesh(mesh=(nk_ibz, nk_ibz, nk_ibz))
        kpts = [k_n_w[0] for k_n_w in kmesh]
        kpts.extend(
            insert_intermediate_kpoints(list(ibz.kpath['kpoints'].values()),
                                        n=10))

        cbmk = np.array(bs.get_cbm()['kpoint'].frac_coords)
        vbmk = np.array(bs.get_cbm()['kpoint'].frac_coords)
        kpts.append(cbmk)
        kpts.append(vbmk)

        # grid = {'energy': [], 'velocity': [], 'mass': [], 'normv': []}
        extrema = {'n': [], 'p': []}

        if interpolation == "boltztrap1":
            engre, nwave, nsym, nstv, vec, vec2, out_vec2, br_dir = get_energy_args(
                coeff_file=coeff_file, ibands=ibands)

        # TODO-AF: for now, I removed the following that only works with boltztrap1; if there is enough value, I will add support for boltztrap2 as well
        # bounds = [(-0.5,0.5), (-0.5,0.5), (-0.5,0.5)]
        # func = lambda x: calc_analytical_energy(x, engre[1], nwave,
        #         nsym, nstv, vec, vec2, out_vec2, br_dir, sgn=-1, scissor=0)[0]
        # opt = basinhopping(func, x0=cbmk, niter=niter, T=0.1, minimizer_kwargs={'bounds': bounds})
        # kpts.append(opt.x)
        #
        # func = lambda x: -calc_analytical_energy(x, engre[0], nwave,
        #         nsym, nstv, vec, vec2, out_vec2, br_dir, sgn=+1, scissor=0)[0]
        # opt = basinhopping(func, x0=vbmk, niter=niter, T=0.1, minimizer_kwargs={'bounds': bounds})
        # kpts.append(opt.x)

        for iband in range(len(ibands)):
            is_cb = [False, True][iband]
            tp = ['p', 'n'][iband]
            if is_cb:
                sgn = -1.0
            else:
                sgn = 1.0

            if interpolation == "boltztrap1":
                energies = []
                velocities = []
                normv = []
                masses = []
                for ik, kpt in enumerate(kpts):
                    en, v, mass = self.calc_analytical_energy(kpt, engre[iband],
                                                         nwave,
                                                         nsym, nstv, vec, vec2,
                                                         out_vec2, br_dir,
                                                         sgn=sgn,
                                                         scissor=scissor)
                    energies.append(en)
                    velocities.append(abs(v))
                    normv.append(norm(v))
                    masses.append(mass.trace() / 3)
            elif interpolation == "boltztrap2":
                fitted = fite.getBands(np.array(kpts), *interp_params)
                energies = fitted[0][ibands[iband] - 1] * Hartree_to_eV - sgn * scissor / 2.
                velocities = fitted[1][:, :, ibands[iband] - 1].T * Hartree_to_eV / hbar * A_to_m * m_to_cm
                normv = [norm(v) for v in velocities]
                masses = [1/(m.trace()/ 3.)* e / Hartree_to_eV / A_to_m**2 * hbar**2/m_e \
                          for m in fitted[2][:, :, :, ibands[iband] - 1].T]
            else:
                raise ValueError(
                    'Unsupported interpolation: "{}"'.format(interpolation))
            indexes = np.argsort(normv)
            energies = [energies[i] for i in indexes]
            normv = [normv[i] for i in indexes]
            velocities = [velocities[i] for i in indexes]
            masses = [masses[i] for i in indexes]
            kpts = [np.array(kpts[i]) for i in indexes]

            # print('here')
            # cbmk = np.array([ 0.44,  0.44,  0.  ])
            # print(np.vstack((bs.get_sym_eq_kpoints(cbmk),bs.get_sym_eq_kpoints(-cbmk))))
            # cbmk = np.array([ 0.5,  0. ,  0.5])
            # print(np.vstack((bs.get_sym_eq_kpoints(cbmk),bs.get_sym_eq_kpoints(-cbmk))))

            # print('here values')
            # print energies[:10]
            # print normv[:10]
            # print kpts[:10]
            # print masses[:10]
            if is_cb:
                iextrem = np.argmin(energies)
                extremum0 = energies[
                    iextrem]  # extremum0 is numerical CBM here
                actual_cbm_vbm[tp]['energy'] = extremum0
                actual_cbm_vbm[tp]['kpoint'] = kpts[iextrem]
                # The following is in case CBM doesn't have a zero numerical norm(v)
                closest_cbm = get_closest_k(kpts[iextrem], np.vstack((
                                                                     bs.get_sym_eq_kpoints(
                                                                         cbmk),
                                                                     bs.get_sym_eq_kpoints(
                                                                         -cbmk))))
                if norm(np.array(kpts[iextrem]) - closest_cbm) < min_normdiff:
                    # and abs(bs.get_cbm()['energy']-extremum0) < 0.05: #TODO: this is not correct unless the fitted energy is calculated at cbmk (bs.get_cbm()['energy'] is dft different reference from interpolation method)
                    extrema['n'].append(cbmk)
                else:
                    extrema['n'].append(kpts[iextrem])
            else:
                iextrem = np.argmax(energies)
                extremum0 = energies[iextrem]
                actual_cbm_vbm[tp]['energy'] = extremum0
                actual_cbm_vbm[tp]['kpoint'] = kpts[iextrem]
                closest_vbm = get_closest_k(kpts[iextrem], np.vstack((
                                                                     bs.get_sym_eq_kpoints(
                                                                         vbmk),
                                                                     bs.get_sym_eq_kpoints(
                                                                         -vbmk))))
                if norm(np.array(kpts[iextrem]) - closest_vbm) < min_normdiff:
                    # and abs(bs.get_vbm()['energy']-extremum0) < 0.05:
                    extrema['p'].append(vbmk)
                else:
                    extrema['p'].append(kpts[iextrem])

            if normv[0] > v_cut:
                raise ValueError(
                    'No extremum point (v<{}) found!'.format(v_cut))
            for i in range(0, len(kpts[:nex_max])):
                # if (velocities[i] > v_cut).all() :
                if normv[i] > v_cut:
                    break
                else:
                    far_enough = True
                    for k in extrema[tp]:
                        if norm(get_closest_k(kpts[i], np.vstack((
                                                                 bs.get_sym_eq_kpoints(
                                                                         k),
                                                                 bs.get_sym_eq_kpoints(
                                                                         -k))),
                                              return_diff=True)) <= min_normdiff:
                            # if norm(kpts[i] - k) <= min_normdiff:
                            far_enough = False
                    if far_enough \
                            and abs(energies[i] - extremum0) < Ecut[tp] \
                            and masses[i] * ((-1) ** (int(is_cb) + 1)) >= 0:
                        extrema[tp].append(kpts[i])
        if not return_global:
            return extrema
        else:
            return extrema, actual_cbm_vbm


    def calculate_spb_transport(self):
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
                                * fermi_integral(0, fermi, T, energy, wordy=False) \
                                / fermi_integral(0.5, fermi, T, energy, wordy=False) * e ** 0.5 * 1e4  # to cm2/V.s


    def find_fermi_boltztrap(self):
        self.logger.warning('\nRunning BoltzTraP to calculate the Fermi levels...')
        boltztrap_runner = BoltztrapRunner(bs=self.bs, nelec=self.nelec,
                doping=list(set([abs(d) for d in self.dopings])),
                        tmax=max(self.temperatures))
        boltztrap_runner.run(path_dir=self.calc_dir)
        # coeff_file = os.path.join(self.calc_dir, 'boltztrap', 'fort.123')
        an = BoltztrapAnalyzer.from_files(os.path.join(self.calc_dir, 'boltztrap'))

        # for c in self.dopings:
        #     for T in self.temperatures:
        #         for c_b in an.doping:


    def generate_kmesh(self, important_points, kgrid_tp='coarse'):
        self.kgrid_array = {}
        self.kgrid_array_cartesian = {}
        self.k_hat_array = {}
        self.k_hat_array_cartesian = {}
        self.dv_grid = {}
        kpts = {}
        for tp in ['n', 'p']:
            points_1d = generate_k_mesh_axes(important_points[tp], kgrid_tp, one_list=True)
            self.kgrid_array[tp] = create_grid(points_1d)
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

            # self.logger.info("number of original ibz {}-type k-points: {}".format(tp, len(kpts[tp])))
            # self.logger.debug("time to get the ibz k-mesh: \n {}".format(time.time()-start_time))
            # start_time = time.time()
        return kpts

    def update_cbm_vbm_dos(self, coeff_file):
        analytical_band_tuple = None
        if self.poly_bands0 is None:
            if self.interpolation=="boltztrap1":
                self.logger.debug(
                    "start interpolating bands from {}".format(coeff_file))
            # analytical_bands = Analytical_bands(coeff_file=coeff_file)

            self.all_ibands = []
            for i, tp in enumerate(["p", "n"]):
                sgn = (-1) ** (i + 1)
                for ib in range(self.cbm_vbm0[tp]["included"]):
                    self.all_ibands.append(self.cbm_vbm0[tp]["bidx"] + sgn * ib)


            self.logger.debug("all_ibands: {}".format(self.all_ibands))

            # # @albalu what are all of these variables (in the next 5 lines)? I don't know but maybe we can lump them together
            # engre, latt_points, nwave, nsym, nsymop, symop, br_dir = analytical_bands.get_engre(iband=all_ibands)
            # nstv, vec, vec2 = analytical_bands.get_star_functions(latt_points, nsym, symop, nwave, br_dir=br_dir)
            # out_vec2 = np.zeros((nwave, max(nstv), 3, 3))
            # for nw in xrange(nwave):
            #     for i in xrange(nstv[nw]):
            #         out_vec2[nw, i] = outer(vec2[nw, i], vec2[nw, i])
            if self.interpolation == "boltztrap1":
                engre, nwave, nsym, nstv, vec, vec2, out_vec2, br_dir = \
                    get_energy_args(coeff_file, self.all_ibands)
            # analytical_band_tuple = (
            # analytical_bands, engre, nwave, nsym, nstv, vec, vec2,
            # out_vec2, br_dir)
        # if using poly bands, remove duplicate k points (@albalu I'm not really sure what this is doing)
        else:
            # first modify the self.poly_bands to include all symmetrically equivalent k-points (k_i)
            # these points will be used later to generate energy based on the minimum norm(k-k_i)

            self.poly_bands = np.array(self.poly_bands0)
            for ib in range(len(self.poly_bands0)):
                for valley in range(len(self.poly_bands0[ib])):
                    self.poly_bands[ib][valley][
                        0] = remove_duplicate_kpoints(
                        self.get_sym_eq_ks_in_first_BZ(
                            self.poly_bands0[ib][valley][0],
                            cartesian=True))


        for i, tp in enumerate(["p", "n"]):
            sgn = (-1) ** i

            if self.poly_bands is not None:
                energy, velocity, effective_m = self.calc_poly_energy(
                    self.cbm_vbm0[tp]["kpoint"], tp, 0)
            elif self.interpolation=="boltztrap1":
                energy, velocity, effective_m = self.calc_analytical_energy(
                        self.cbm_vbm0[tp]["kpoint"],engre[i * self.cbm_vbm0["p"][
                        "included"]],nwave, nsym, nstv, vec, vec2, out_vec2,
                        br_dir, sgn, scissor=self.scissor)
            elif self.interpolation=="boltztrap2":
                fitted = fite.getBands(np.array([self.cbm_vbm0[tp]["kpoint"]]), *self.interp_params)
                energy = fitted[0][self.cbm_vbm[tp]["bidx"]-1][0]*Hartree_to_eV - sgn*self.scissor/2.
                effective_m = 1/fitted[2][:, :, 0, self.cbm_vbm0[tp]["bidx"]-1].T * e / Hartree_to_eV / A_to_m**2 * hbar**2/m_e
            self.offset_from_vrun[tp] = energy - self.cbm_vbm0[tp]["energy"]


            self.logger.debug("offset from vasprun energy values for {}-type = {} eV".format(tp, self.offset_from_vrun[tp]))
            self.cbm_vbm0[tp]["energy"] = energy
            self.cbm_vbm0[tp]["eff_mass_xx"] = effective_m.diagonal()

        if self.poly_bands is None:
            self.dos_emax += self.offset_from_vrun['n']
            self.dos_emin += self.offset_from_vrun['p']

        # print('here debug')
        # print(self.cbm_vbm0)
        # print(len(engre))
        # print(self.all_ibands)


        for tp in ['p', 'n']:
            self.cbm_vbm[tp]['energy'] = self.cbm_vbm0[tp]['energy']
            self.cbm_vbm[tp]['eff_mass_xx'] = self.cbm_vbm0[tp]['eff_mass_xx']
        self._avg_eff_mass = {tp: abs(np.mean(self.cbm_vbm0[tp]["eff_mass_xx"])) for tp in ["n", "p"]}


    def get_energy_array(self, coeff_file, kpts, once_called=False,
                         return_energies=False, num_bands=None,
                         nbelow_vbm=0, nabove_cbm=0):
        num_bands = num_bands or self.num_bands
        start_time = time.time()
        # if not once_called:
        self.logger.info("self.nkibz = {}".format(self.nkibz))


        # the first part is just to update the cbm_vbm once!
        analytical_band_tuple = None
        # TODO for now, I get these parameters everytime which is wasteful but I only need to run this part once
        # if not once_called:
        if True: # We only need to set all_bands once
            # TODO-JF: this if setup energy calculation for SPB and actual BS it would be nice to do this in two separate functions
            # if using analytical bands: create the object, determine list of band indices, and get energy info
            if self.poly_bands0 is None:
                if self.interpolation == 'boltztrap1':
                    self.logger.debug("start interpolating bands from {}".format(coeff_file))
                    analytical_bands = Analytical_bands(coeff_file=coeff_file)
                    # all_ibands supposed to start with index of last valence band then
                    # VBM-1 ... and then index of CBM then CBM+1 ...

                    self.all_ibands = []
                    # for i, tp in enumerate(["p", "n"]):
                    #     # sgn = (-1) ** (i + 1)
                    #     for ib in range(num_bands[tp]):
                    #         self.all_ibands.append(self.cbm_vbm0[tp]["bidx"] + sgn * ib)

                    for ib in range(num_bands['p']):
                        self.all_ibands.append(self.cbm_vbm0['p']["bidx"] - nbelow_vbm - ib)
                    for ib in range(num_bands['n']):
                        self.all_ibands.append(self.cbm_vbm0['n']["bidx"] + nabove_cbm + ib)
                    self.logger.debug("all_ibands: {}".format(self.all_ibands))
                    # # @albalu what are all of these variables (in the next 5 lines)? I don't know but maybe we can lump them together
                    # engre, latt_points, nwave, nsym, nsymop, symop, br_dir = analytical_bands.get_engre(iband=all_ibands)
                    # nstv, vec, vec2 = analytical_bands.get_star_functions(latt_points, nsym, symop, nwave, br_dir=br_dir)
                    # out_vec2 = np.zeros((nwave, max(nstv), 3, 3))
                    # for nw in xrange(nwave):
                    #     for i in xrange(nstv[nw]):
                    #         out_vec2[nw, i] = outer(vec2[nw, i], vec2[nw, i])
                    engre, nwave, nsym, nstv, vec, vec2, out_vec2, br_dir = \
                        get_energy_args(coeff_file, self.all_ibands)
                    analytical_band_tuple = (analytical_bands, engre, nwave, nsym, nstv, vec, vec2, out_vec2, br_dir)
                elif self.interpolation == 'boltztrap2':
                    analytical_band_tuple = self.interp_params
                else:
                    raise ValueError('Unsupported interpolation method: "{}"'.format(self.interpolation))
            else:
                # first modify the self.poly_bands to include all symmetrically equivalent k-points (k_i)
                # these points will be used later to generate energy based on the minimum norm(k-k_i)

                self.poly_bands = np.array(self.poly_bands0)
                for ib in range(len(self.poly_bands0)):
                    for valley in range(len(self.poly_bands0[ib])):
                        self.poly_bands[ib][valley][0] = remove_duplicate_kpoints(
                            self.get_sym_eq_ks_in_first_BZ(self.poly_bands0[ib][valley][0], cartesian=True))

            self.logger.debug("time to get engre and calculate the outvec2: {} seconds".format(time.time() - start_time))

        # calculate only the CBM and VBM energy values - @albalu why is this separate from the other energy value calculations?
        # here we assume that the cbm and vbm k-point coordinates read from vasprun.xml are correct:


        # calculate the energy at initial ibz k-points and look at the first band to decide on additional/adaptive ks
        start_time = time.time()
        energies = {"n": [0.0 for ik in kpts['n']], "p": [0.0 for ik in kpts['p']]}
        energies_sorted = {"n": [0.0 for ik in kpts['n']], "p": [0.0 for ik in kpts['p']]}
        velocities = {"n": [[0.0, 0.0, 0.0] for ik in kpts['n']], "p": [[0.0, 0.0, 0.0] for ik in kpts['p']]}

        self.pos_idx = {'n': [], 'p': []}
        self.energy_array = {'n': [], 'p': []}

        if return_energies:
            # calculate energies
            for i, tp in enumerate(["p", "n"]):
                sgn = (-1) ** i
                for ib in range(num_bands[tp]):
                    if self.poly_bands is not None:
                        for ik in range(len(kpts[tp])):
                            energies[tp][ik], _, _ = self.calc_poly_energy(kpts[tp][ik], tp, ib)
                    elif self.interpolation == "boltztrap1":
                        if not self.parallel:
                            for ik in range(len(kpts[tp])):
                                energy, velocities[tp][ik], effective_m = self.calc_analytical_energy(kpts[tp][ik],engre[i * num_bands['p'] + ib],nwave, nsym, nstv, vec, vec2,out_vec2, br_dir, sgn, scissor=self.scissor)
                                energies[tp][ik] = energy
                        else:
                            results = Parallel(n_jobs=self.num_cores)(delayed(get_energy)(kpts[tp][ik],engre[i * num_bands['p'] + ib], nwave, nsym, nstv, vec, vec2, out_vec2, br_dir) for ik in range(len(kpts[tp])))
                            for ik, res in enumerate(results):
                                energies[tp][ik] = res[0] * Ry_to_eV - sgn * self.scissor / 2.0
                    elif self.interpolation == "boltztrap2":
                        fitted = fite.getBands(np.array(kpts[tp]), *self.interp_params)
                        energies[tp] = fitted[0][self.cbm_vbm['p']['bidx']-1+ i * num_bands['p'], :]*Hartree_to_eV - sgn*self.scissor/2.
                    else:
                        raise ValueError('Unsupported interpolation: "{}"'.format(self.interpolation))

                    self.energy_array[tp].append(self.grid_from_ordered_list(energies[tp], tp, none_missing=True))

                    if ib == 0:      # we only include the first band to decide on order of ibz k-points
                        e_sort_idx = np.array(energies[tp]).argsort() if tp == "n" else np.array(energies[tp]).argsort()[::-1]
                        energies_sorted[tp] = [energies[tp][ie] for ie in e_sort_idx]
                        energies[tp] = [energies[tp][ie] for ie in e_sort_idx]
                        # velocities[tp] = [velocities[tp][ie] for ie in e_sort_idx]
                        self.pos_idx[tp] = np.array(range(len(e_sort_idx)))[e_sort_idx].argsort()
                        kpts[tp] = [kpts[tp][ie] for ie in e_sort_idx]

            N_n = self.kgrid_array['n'].shape

            # for ib in range(self.num_bands['n']):
            #     self.logger.debug('energy (type n, band {}):'.format(ib))
            #     self.logger.debug(self.energy_array['n'][ib][(N_n[0] - 1) / 2, (N_n[1] - 1) / 2, :])
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


        # calculation of the density of states (DOS)
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
                # self.dos_normalization_factor = dos_nbands # not a big change in either mobility values
            else:
                self.logger.debug("here self.poly_bands: \n {}".format(self.poly_bands))
                emesh, dos = get_dos_from_poly_bands(self._vrun.final_structure, self._rec_lattice,
                        [self.nkdos, self.nkdos, self.nkdos], self.dos_emin,
                        self.dos_emax, int(round((self.dos_emax - self.dos_emin) \
                        / max(self.dE_min, 0.0001))),poly_bands=self.poly_bands,
                        bandgap=self.cbm_vbm["n"]["energy"] - self.cbm_vbm["p"][
                        "energy"], width=self.dos_bwidth, SPB_DOS=False)
                self.dos_normalization_factor = len(self.poly_bands) * 2 * 2
                # it is *2 elec/band & *2 because DOS repeats in valence/conduction
                self.dos_start = self.dos_emin
                self.dos_end = self.dos_emax


            self.logger.info("DOS normalization factor: {}".format(self.dos_normalization_factor))

            integ = 0.0
            self.dos_start = abs(emesh - self.dos_start).argmin()
            self.dos_end = abs(emesh - self.dos_end).argmin()
            for idos in range(self.dos_start, self.dos_end):
                # if emesh[idos] > self.cbm_vbm["n"]["energy"]: # we assume anything below CBM as 0 occupation
                #     break
                integ += (dos[idos + 1] + dos[idos]) / 2 * (emesh[idos + 1] - emesh[idos])

            print("dos integral from {} index to {}: {}".format(self.dos_start,  self.dos_end, integ))

            # self.logger.debug("dos before normalization: \n {}".format(zip(emesh, dos)))
            dos = [g / integ * self.dos_normalization_factor for g in dos]
            # self.logger.debug("integral of dos: {} stoped at index {} and energy {}".format(integ, idos, emesh[idos]))

            self.dos = zip(emesh, dos)
            self.dos_emesh = np.array(emesh)
            self.vbm_dos_idx = self.get_Eidx_in_dos(self.cbm_vbm["p"]["energy"])
            self.cbm_dos_idx = self.get_Eidx_in_dos(self.cbm_vbm["n"]["energy"])

            self.logger.info("vbm and cbm DOS index")
            self.logger.info(self.vbm_dos_idx)
            self.logger.info(self.cbm_dos_idx)
            # self.logger.debug("full dos after normalization: \n {}".format(self.dos))
            # self.logger.debug("dos after normalization from vbm idx to cbm idx: \n {}".format(self.dos[self.vbm_dos_idx-10:self.cbm_dos_idx+10]))

            self.dos = [list(a) for a in self.dos]

        if return_energies:
            return analytical_band_tuple, kpts, energies_sorted
        else:
            return analytical_band_tuple, kpts



    def find_all_important_points(self, coeff_file, nbelow_vbm=0, nabove_cbm=0):
        # generate the k mesh in two forms: numpy array for k-integration and list for e-integration
        if self.important_pts is None or nbelow_vbm+nabove_cbm>0:
            self.important_pts, new_cbm_vbm = self.get_bs_extrema(self.bs, coeff_file,
                    interp_params=self.interp_params, interpolation=self.interpolation,
                    nk_ibz=self.nkdos, v_cut=self.v_min, min_normdiff=0.1,
                    Ecut=self.Ecut, nex_max=20, return_global=True, niter=5,
                          nbelow_vbm= nbelow_vbm, nabove_cbm=nabove_cbm, scissor=self.scissor)
            # self.important_pts = {'n': [self.cbm_vbm["n"]["kpoint"]], 'p': [self.cbm_vbm["p"]["kpoint"]]}
            if new_cbm_vbm['n']['energy'] < self.cbm_vbm['n']['energy']:
                # self.cbm_vbm['n']['energy'] = new_cbm_vbm['n']['energy'] + self.scissor/2.0
                self.cbm_vbm['n']['energy'] = new_cbm_vbm['n']['energy']
                self.cbm_vbm['n']['kpoint'] = new_cbm_vbm['n']['kpoint']
            if new_cbm_vbm['p']['energy'] > self.cbm_vbm['p']['energy']:
                # self.cbm_vbm['p']['energy'] = new_cbm_vbm['p']['energy'] - self.scissor/2.0
                self.cbm_vbm['p']['energy'] = new_cbm_vbm['p']['energy']
                self.cbm_vbm['p']['kpoint'] = new_cbm_vbm['p']['kpoint']
            # for tp in ['p', 'n']:
            #     self.cbm_vbm[tp]['energy'] = new_cbm_vbm[tp]['energy']
            #     self.cbm_vbm[tp]['kpoint'] = new_cbm_vbm[tp]['kpoint']

        self.logger.info(('here initial important_pts'))
        self.logger.info((self.important_pts)) # for some reason the nscf uniform GaAs fitted coeffs return positive mass for valence at Gamma!

        # for tp in ['n', 'p']:
        #     self.important_pts[tp].append(self.cbm_vbm[tp]["kpoint"])
        #     self.important_pts[tp].extend(self.add_extrema[tp])
        # for tp in ['n', 'p']:
        #     all_important_ks = []
        #     for k in self.important_pts[tp]:
        #         all_important_ks +=  list(self.bs.get_sym_eq_kpoints(k))
        #     self.important_pts[tp] = remove_duplicate_kpoints(all_important_ks)


        ################TEST FOR Si##########################################
        # cbm_k = np.array([ 0.48648649,  0.48648649,  0.        ])
        # cbm_k = np.array([ 0.42105263,  0.42105263,  0.        ])

        # cbm_k = np.array([ 0.43105263 , 0.43105263,  0.001     ])
        # self.important_pts = {
        #     # 'n': [[ 0.42905263 , 0.42905263 , 0.001     ]],
        #     'n': [[ -4.26549744e-01,  -4.26549744e-01,  -1.35782351e-08]],
        #     # 'n': [[ 0.47058824,  0.47058824,  0.        ]],
        #     'p': [[0.0, 0.0, 0.0]]}
        #     'n': np.vstack((self.bs.get_sym_eq_kpoints(cbm_k),self.bs.get_sym_eq_kpoints(-cbm_k)))
        # }
        #################################################################

        ################TEST FOR GaAs-L ##########################################
        # self.important_pts = {'p': [np.array([ 0.,  0.,  0.])],
        #                       # 'n': [np.array([-0.5,  0. ,  0. ])]}
        #                       # 'n': [np.array([ 0.,  0.,  0.]), np.array([0.5,  0.5 ,  0.5 ]), np.array([ 0.5,  0.,  0.5])]}
        #                     'n': [np.array([ 0.,  0.,  0.]), np.array([0.5,  0.5 ,  0.5 ])]}

                              # 'n': [np.array([ 0.5,  0.5,  0.5]),
                              #       np.array([-0.5,  0. ,  0. ]), np.array([ 0. , -0.5,  0. ]), np.array([ 0. ,  0. , -0.5])
                              #       , np.array([0.5,  0. ,  0. ]), np.array([ 0. , 0.5,  0. ]), np.array([ 0. ,  0. , 0.5])
                              #       ]}



        #################################################################
        self.logger.info('Here are the final extrema considered: \n {}'.format(self.important_pts))


    def write_input_files(self, path=None, dir_name="run_data"):
        """writes all 3 types of inputs in json files for example to
        conveniently track what inputs had been used later or read
        inputs from files (see from_files method)"""
        if not path:
            path = os.path.join(os.getcwd(), dir_name)
            if not os.path.exists(path):
                os.makedirs(name=path)

        material_params = {
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
            "user_bandgap": self.user_bandgap
        }
        if self.W_POP:
            material_params["W_POP"] = self.W_POP / (1e12 * 2 * pi)
        else:
            material_params["W_POP"] = self.W_POP

        model_params = {
            "bs_is_isotropic": self.bs_is_isotropic,
            "elastic_scatterings": self.elastic_scatterings,
            "inelastic_scatterings": self.inelastic_scatterings,
            "poly_bands": self.poly_bands
        }

        performance_params = {
            "nkibz": self.nkibz,
            "dE_min": self.dE_min,
            "Ecut": self.Ecut,
            "adaptive_mesh": self.adaptive_mesh,
            "dos_bwidth": self.dos_bwidth,
            "nkdos": self.nkdos,
            "BTE_iters": self.BTE_iters,
            "max_nbands": self.max_nbands,
            "max_normk0": self.max_normk0,
            "max_nvalleys": self.max_nvalleys,
            "max_ncpu": self.max_ncpu,
            "pre_determined_fermi": self.pre_determined_fermi,
            "interpolation": self.interpolation
        }

        with open(os.path.join(path, "material_params.json"), "w") as fp:
            json.dump(material_params, fp, sort_keys=True, indent=4, ensure_ascii=False, cls=MontyEncoder)
        with open(os.path.join(path, "model_params.json"), "w") as fp:
            json.dump(model_params, fp, sort_keys=True, indent=4, ensure_ascii=False, cls=MontyEncoder)
        with open(os.path.join(path, "performance_params.json"), "w") as fp:
            json.dump(performance_params, fp, sort_keys=True, indent=4, ensure_ascii=False, cls=MontyEncoder)



    def set_material_params(self, params):
        """
        set materials parameters. This function is meant to be called after
            set_model_parameters as it may modify self.model_parameters.
        Args:
            params (dict):
        Returns:

        """

        self.epsilon_s = params["epsilon_s"]
        self.P_PIE = params.get("P_PIE", None) or 0.15  # unitless
        E_D = params.get("E_D", None)
        self.C_el = params.get("C_el", None)
        if (E_D is None or self.C_el is None) and 'ACD' in self.elastic_scatterings:
            self.elastic_scatterings.pop(self.elastic_scatterings.index('ACD'))
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
        if 'POP' in self.inelastic_scatterings:
            if self.epsilon_inf is None or self.W_POP is None:
                warnings.warn('POP cannot be calculated w/o epsilon_inf and W_POP')
                self.inelastic_scatterings.pop(self.inelastic_scatterings.index('POP'))

        self.N_dis = params.get("N_dis", None) or 0.1  # in 1/cm**2
        self.scissor = params.get("scissor", None) or 0.0
        self.user_bandgap = params.get("user_bandgap", None)

        donor_charge = params.get("donor_charge", 1.0)
        acceptor_charge = params.get("acceptor_charge", 1.0)
        dislocations_charge = params.get("dislocations_charge", 1.0)
        self.charge = {"n": donor_charge, "p": acceptor_charge, "dislocations": dislocations_charge}
        self.add_extrema = params.get('add_extrema', None)
        self.add_extrema = self.add_extrema or {'n': [], 'p':[]}
        self.important_pts = params.get('important_points', None)


    def set_model_params(self, params):
        """function to set instant variables related to the model and the level of the theory;
        these are set based on params (dict) set by the user or their default values"""

        self.bs_is_isotropic = params.get("bs_is_isotropic", True)
        self.elastic_scatterings = params.get("elastic_scatterings", ["ACD", "IMP", "PIE"])
        self.inelastic_scatterings = params.get("inelastic_scatterings", ["POP"])

        self.poly_bands0 = params.get("poly_bands", None)
        self.poly_bands = self.poly_bands0

        # TODO: self.gaussian_broadening is designed only for development version and must be False, remove it later.
        # because if self.gaussian_broadening the mapping to egrid will be done with the help of Gaussian broadening
        # and that changes the actual values
        self.gaussian_broadening = False
        self.soc = params.get("soc", False)
        self.logger.info("bs_is_isotropic: {}".format(self.bs_is_isotropic))
        self.independent_valleys = params.get('independent_valleys', False)


    def set_performance_params(self, params):
        self.nkibz = params.get("nkibz", 40)
        self.dE_min = params.get("dE_min", 0.0001)
        self.nE_min = params.get("nE_min", 2)
        c_factor = max(1, 3 * abs(max([log(abs(ci)/float(1e19)) for ci in self.dopings]))**0.25)
        Ecut = params.get("Ecut", c_factor * 5 * k_B * max(self.temperatures + [300]))
        self.Ecut = {tp: Ecut if tp in self.all_types else Ecut/2.0 for tp in ["n", "p"]}
        # self.Ecut = {tp: Ecut for tp in ["n", "p"]}
        for tp in ["n", "p"]:
            self.logger.debug("{}-Ecut: {} eV \n".format(tp, self.Ecut[tp]))
        self.adaptive_mesh = params.get("adaptive_mesh", False)

        self.dos_bwidth = params.get("dos_bwidth",
                                     0.05)  # in eV the bandwidth used for calculation of the total DOS (over all bands & IBZ k-points)
        self.nkdos = params.get("nkdos", 29)
        self.v_min = 1000
        self.gs = 1e-32  # a global small value (generally used for an initial non-zero value)
        self.gl = 1e32  # a global large value

        # TODO: some of the current global constants should be omitted, taken as functions inputs or changed!
        self.BTE_iters = params.get("BTE_iters", 5)
        self.parallel = params.get("parallel", True)
        self.max_ncpu = params.get("max_ncpu", 8)
        self.logger.info("parallel: {}".format(self.parallel))
        self.max_nbands = params.get("max_nbands", None)
        self.max_normk0 = params.get("max_normk", None)
        self.max_normk = {'n': self.max_normk0, 'p': self.max_normk0}
        self.max_nvalleys = params.get("max_nvalleys", None)
        self.fermi_kgrid_tp = params.get("fermi_kgrid_tp", "uniform")
        self.pre_determined_fermi = params.get("pre_determined_fermi")
        self.interpolation = params.get("interpolation", "boltztrap1")


    def __getitem__(self, key):
        if key == "kgrid":
            return self.kgrid
        elif key == "egrid":
            return self.egrid
        else:
            raise KeyError



    def get_dft_orbitals(self, bidx):
        """
        the contribution from s and p orbitals at a given band for kpoints
            that were used in the DFT run (from which vasprun.xml is read)
        Args:
            bidx (idx): band index
        Returns:
            ([float], [float]) two lists: s&p orbital scores at the band # bidx
        """
        projected = self._vrun.projected_eigenvalues
        # print len(projected[Spin.up][0][10])
        # projected indexes : Spin; kidx; bidx; s,py,pz,px,dxy,dyz,dz2,dxz,dx2

        s_orbital = [0.0 for k in self.DFT_cartesian_kpts]
        p_orbital = [0.0 for k in self.DFT_cartesian_kpts]
        for ik in range(len(self.DFT_cartesian_kpts)):
            s_orbital[ik] = sum(projected[Spin.up][ik][bidx])[0]
            if self.lorbit == 10:
                p_orbital[ik] = sum(projected[Spin.up][ik][bidx])[1]
            elif self.lorbit == 11:
                p_orbital[ik] = sum(sum(projected[Spin.up][ik][bidx])[1:4])
        return s_orbital, p_orbital



    def read_vrun(self, calc_dir=".", filename="vasprun.xml"):
        self._vrun = Vasprun(os.path.join(calc_dir, filename), parse_projected_eigen=True)
        self.interp_params = None
        if self.interpolation == "boltztrap2":
            bz2_data = BoltzTraP2.dft.DFTData(calc_dir, derivatives=False)
            equivalences = sphere.get_equivalences(bz2_data.atoms,
                                            len(bz2_data.kpoints) * 10)
            lattvec = bz2_data.get_lattvec()
            coeffs = fite.fitde3D(bz2_data, equivalences)
            self.interp_params = (equivalences, lattvec, coeffs)

        self.volume = self._vrun.final_structure.volume
        self.logger.info("unitcell volume = {} A**3".format(self.volume))
        self.density = self._vrun.final_structure.density
        self._rec_lattice = self._vrun.final_structure.lattice.reciprocal_lattice

        # (self._vrun.lattice.matrix @ self._rec_lattice.matrix) == 2*pi*np.eye(3)

        # print(np.array(self._rec_lattice.matrix)/2/pi)
        # print(np.linalg.norm(self._rec_lattice.matrix))
        # print(self._rec_lattice.matrix)
        # print(np.linalg.inv(self._rec_lattice.matrix))
        #
        # quit()

        # print(norm(self.get_cartesian_coords([0.5, 0.5, 0.5])/A_to_nm )/2  )
        # print(norm(self.get_cartesian_coords([0.5, 0.0, 0.5])/A_to_nm )/2  )

        sg = SpacegroupAnalyzer(self._vrun.final_structure)
        self.rotations, _ = sg._get_symmetry()
        self.bs = self._vrun.get_band_structure()
        self.bs.structure = self._vrun.final_structure
        self.nbands = self.bs.nb_bands
        self.lorbit = 11 if len(sum(self._vrun.projected_eigenvalues[Spin.up][0][10])) > 5 else 10

        self.DFT_cartesian_kpts = np.array(
                [self.get_cartesian_coords(k) for k in self._vrun.actual_kpoints])/ A_to_nm


        # Remember that python band index starts from 0 so bidx==9 refers to the 10th band in VASP
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

        # if self.max_nbands:
        #     for tp in ['p', 'n']:
        #         cbm_vbm[tp]["included"] = self.max_nbands
        #         self.initial_num_bands[tp] = self.max_nbands


        cbm_vbm["p"]["bidx"] += 1
        cbm_vbm["n"]["bidx"] = cbm_vbm["p"]["bidx"] + 1

        self.cbm_vbm = cbm_vbm
        self.cbm_vbm0 = deepcopy(cbm_vbm)
        self.valleys = {tp: {'band {}'.format(i): {} for i in range(self.cbm_vbm0[tp]['included']) } for tp in ['p', 'n']}
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
            return np.dot(self._rec_lattice.matrix, np.array(frac_k).T).T
        else:
            return np.dot(self._vrun.lattice.matrix, np.array(frac_k).T).T


    def seeb_int_num(self, c, T):
        """wrapper function to do an integration taking only the concentration, c, and the temperature, T, as inputs"""
        fn = lambda E, fermi, T: f0(E, fermi, T) * (1 - f0(E, fermi, T)) * E / (k_B * T)
        return {
            t: self.integrate_over_DOSxE_dE(func=fn, tp=t, fermi=self.fermi_level[c][T], T=T, normalize_energy=True)
        for
            t in ["n", "p"]}



    def seeb_int_denom(self, c, T):
        """wrapper function to do an integration taking only the concentration, c, and the temperature, T, as inputs"""
        return {t: self.gs + self.integrate_over_E(prop_list=["f0x1-f0"], tp=t, c=c, T=T, xDOS=True) for t in
                ["n", "p"]}



    def calculate_property(self, prop_name, prop_func, for_all_E=False):
        """
        calculate the propery at all concentrations and Ts using the given function and insert it into self.egrid
        :param prop_name:
        :param prop_func (obj): the given function MUST takes c and T as required inputs in this order.
        :return:
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
                    #fermi = self.egrid["fermi"][c][T]
                    fermi = self.fermi_level[c][T]
                    for tp in ["n", "p"]:
                        for ie, E in enumerate(self.egrid[tp]["energy"]):
                            self.egrid[tp][prop_name][c][T][ie] = prop_func(E, fermi, T)
                else:
                    self.egrid[prop_name][c][T] = prop_func(c, T)



    def calculate_N_II(self, c, T):
        """
        self.N_dis is a given observed 2D concentration of charged dislocations in 1/cm**2
        :param c:
        :param T:
        :return:
        """
        # N_II = abs(self.egrid["calc_doping"][c][T]["n"]) * self.charge["n"] ** 2 + \
        #        abs(self.egrid["calc_doping"][c][T]["p"]) * self.charge["p"] ** 2 + \
        #        self.N_dis / self.volume ** (1 / 3) * 1e8 * self.charge["dislocations"] ** 2

        N_II = abs(self.calc_doping[c][T]["n"]) * self.charge["n"] ** 2 + \
               abs(self.calc_doping[c][T]["p"]) * self.charge["p"] ** 2 + \
               self.N_dis / self.volume ** (1 / 3) * 1e8 * self.charge["dislocations"] ** 2
        return N_II


    def pre_init_egrid(self, once_called=False, dos_tp="standard"):
        self.egrid = {
            "n": {"energy": [], "DOS": [], "all_en_flat": [],
                  "all_ks_flat": [], "mobility": {}},
            "p": {"energy": [], "DOS": [], "all_en_flat": [],
                  "all_ks_flat": [], "mobility": {}},
        }
        self.kgrid_to_egrid_idx = {"n": [],
                                   "p": []}  # list of band and k index that are mapped to each memeber of egrid
        self.Efrequency = {"n": [], "p": []}
        self.sym_freq = {"n": [], "p":[]}
        E_idx = {"n": [], "p": []}
        for tp in ["n", "p"]:
            for ib, en_vec in enumerate(self.kgrid[tp]["energy"]):
                self.egrid[tp]["all_en_flat"] += list(en_vec)
                self.egrid[tp]["all_ks_flat"] += list(self.kgrid[tp]["kpoints"][ib])
                E_idx[tp] += [(ib, iek) for iek in range(len(en_vec))]

            # get the indexes of sorted flattened energy
            ieidxs = np.argsort(self.egrid[tp]["all_en_flat"])
            self.egrid[tp]["all_en_flat"] = [self.egrid[tp]["all_en_flat"][ie] for ie in ieidxs]
            self.egrid[tp]["all_ks_flat"] = [self.egrid[tp]["all_ks_flat"][ie] for ie in ieidxs]

            # sort the tuples of band and energy based on their energy
            E_idx[tp] = [E_idx[tp][ie] for ie in ieidxs]

        # setting up energy grid and DOS:
        for tp in ["n", "p"]:
            energy_counter = []
            i = 0
            last_is_counted = False
            while i < len(self.egrid[tp]["all_en_flat"]) - 1:
                sum_E = self.egrid[tp]["all_en_flat"][i]
                sum_nksym = len(remove_duplicate_kpoints(self.get_sym_eq_ks_in_first_BZ(self.egrid[tp]["all_ks_flat"][i])))
                counter = 1.0  # because the ith member is already included in sum_E
                current_ib_ie_idx = [E_idx[tp][i]]
                j = i
                while j < len(self.egrid[tp]["all_en_flat"]) - 1 and \
                        abs(self.egrid[tp]["all_en_flat"][i] - self.egrid[tp]["all_en_flat"][j + 1]) < self.dE_min:
                    counter += 1
                    current_ib_ie_idx.append(E_idx[tp][j + 1])
                    sum_E += self.egrid[tp]["all_en_flat"][j + 1]
                    sum_nksym += len(remove_duplicate_kpoints(self.get_sym_eq_ks_in_first_BZ(self.egrid[tp]["all_ks_flat"][i+1])))

                    if j + 1 == len(self.egrid[tp]["all_en_flat"]) - 1:
                        last_is_counted = True
                    j += 1
                self.egrid[tp]["energy"].append(sum_E / counter)
                self.kgrid_to_egrid_idx[tp].append(current_ib_ie_idx)
                self.sym_freq[tp].append(sum_nksym / counter)
                energy_counter.append(counter)

                if dos_tp.lower() == "simple":
                    self.egrid[tp]["DOS"].append(counter / len(self.egrid[tp]["all_en_flat"]))
                elif dos_tp.lower() == "standard":
                    # print('here debug')
                    # print(sum_E / counter)
                    # print(self.dos_emin)
                    # print(self.dos_emax)
                    # print(len(self.dos))
                    # print(self.get_Eidx_in_dos(sum_E / counter))
                    # print('end debug')
                    self.egrid[tp]["DOS"].append(self.dos[self.get_Eidx_in_dos(sum_E / counter)][1])
                i = j + 1

            if not last_is_counted:
                self.egrid[tp]["energy"].append(self.egrid[tp]["all_en_flat"][-1])
                self.kgrid_to_egrid_idx[tp].append([E_idx[tp][-1]])
                if dos_tp.lower() == "simple":
                    self.egrid[tp]["DOS"].append(self.nelec / len(self.egrid[tp]["all_en_flat"]))
                elif dos_tp.lower() == "standard":
                    self.egrid[tp]["DOS"].append(self.dos[self.get_Eidx_in_dos(self.egrid[tp]["energy"][-1])][1])

            self.egrid[tp]["size"] = len(self.egrid[tp]["energy"])

        for tp in ["n", "p"]:
            self.Efrequency[tp] = [len(Es) for Es in self.kgrid_to_egrid_idx[tp]]

        min_nE = 2

        if len(self.Efrequency["n"]) < min_nE or len(self.Efrequency["p"]) < min_nE:
            raise ValueError("The final egrid have fewer than {} energy values, AMSET stops now".format(min_nE))



    def init_egrid(self, once_called, dos_tp="standard"):
        """
        :param
            dos_tp (string): options are "simple", ...

        :return: an updated grid that contains the field DOS
        """
        self.pre_init_egrid(once_called=once_called, dos_tp=dos_tp)

        # initialize some fileds/properties
        if not once_called:
            self.egrid["calc_doping"] = {c: {T: {"n": 0.0, "p": 0.0} for T in self.temperatures} for c in self.dopings}
            # for sn in self.elastic_scatterings + self.inelastic_scatterings + ["overall", "average"]:
            #     for tp in ['n', 'p']:
            #         self.egrid[tp]['mobility'][sn] = {c: {T: [0.0, 0.0, 0.0] for T in\
            #                 self.temperatures} for c in self.dopings}
            for transport in ["conductivity", "J_th", "seebeck", "TE_power_factor", "relaxation time constant"]:
                for tp in ['n', 'p']:
                    self.egrid[tp][transport] = {c: {T: 0.0 for T in\
                            self.temperatures} for c in self.dopings}

            # populate the egrid at all c and T with properties; they can be called via self.egrid[prop_name][c][T] later
            if self.fermi_calc_type == 'k':
                self.egrid["calc_doping"] = self.calc_doping
            if self.fermi_calc_type == 'e':
                # self.calc_doping = self.egrid["calc_doping"]
                self.egrid["calc_doping"] = self.calc_doping
                # self.calculate_property(prop_name="fermi", prop_func=self.find_fermi)
                # self.fermi_level = self.egrid["fermi"]
                self.egrid["fermi"] = self.fermi_level

        #TODO: comment out these 3 lines and test, these were commented out in master 9/27/2017
        self.calculate_property(prop_name="f0", prop_func=f0, for_all_E=True)
        self.calculate_property(prop_name="f", prop_func=f0, for_all_E=True)
        self.calculate_property(prop_name="f_th", prop_func=f0, for_all_E=True)

        for prop in ["f", "f_th"]:
            self.map_to_egrid(prop_name=prop, c_and_T_idx=True)

        self.calculate_property(prop_name="f0x1-f0", prop_func=lambda E, fermi,
                T: f0(E, fermi, T) * (1 - f0(E, fermi, T)), for_all_E=True)

        for c in self.dopings:
            for T in self.temperatures:
                #fermi = self.egrid["fermi"][c][T]
                fermi = self.fermi_level[c][T]
                for tp in ["n", "p"]:
                    #fermi_norm = fermi - self.cbm_vbm[tp]["energy"]
                    for ib in range(len(self.kgrid[tp]["energy"])):
                        for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                            E = self.kgrid[tp]["energy"][ib][ik]
                            v = self.kgrid[tp]["velocity"][ib][ik]
                            self.kgrid[tp]["f0"][c][T][ib][ik] = f0(E, fermi, T) * 1.0

        self.calculate_property(prop_name="beta", prop_func=self.inverse_screening_length)
        self.logger.debug('inverse screening length, beta is \n{}'.format(
            self.egrid["beta"]))
        self.calculate_property(prop_name="N_II", prop_func=self.calculate_N_II)
        self.calculate_property(prop_name="Seebeck_integral_numerator", prop_func=self.seeb_int_num)
        self.calculate_property(prop_name="Seebeck_integral_denominator", prop_func=self.seeb_int_denom)



    def get_Eidx_in_dos(self, E, Estep=None):
        if not Estep:
            Estep = max(self.dE_min, 0.0001)
        return int(round((E - self.dos_emin) / Estep)) # ~faster than argmin
        # return abs(self.dos_emesh - E).argmin()



    def G(self, tp, ib, ik, ib_prm, ik_prm, X):
        """
        The overlap integral betweek vectors k and k'
        :param ik (int): index of vector k in kgrid
        :param ik_prm (int): index of vector k' in kgrid
        :param X (float): cosine of the angle between vectors k and k'
        :return: overlap integral
        """
        a = self.kgrid[tp]["a"][ib][ik]
        c = self.kgrid[tp]["c"][ib][ik]
        return (a * self.kgrid[tp]["a"][ib_prm][ik_prm] + \
                X * c * self.kgrid[tp]["c"][ib_prm][ik_prm]) ** 2



    def remove_indexes(self, rm_idx_list, rearranged_props):
        """
        The k-points with velocity < 1 cm/s (either in valence or conduction band) are taken out as those are
            troublesome later with extreme values (e.g. too high elastic scattering rates)
        :param rm_idx_list ([int]): the kpoint indexes that need to be removed for each property
        :param rearranged_props ([str]): list of properties for which some indexes need to be removed
        :return:
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
        initializes a variable/key within the self.kgrid variable
        :param grid (str): options are "kgrid" or "egrid": whether to initialize vars in self.kgrid or self.egrid
        :param names (list): list of the names of the variables
        :param val_type (str): options are "scalar", "vector", "matrix" or "tensor"
        :param initval (float): the initial value (e.g. if val_type=="vector", each of the vector's elements==init_val)
        :param is_nparray (bool): whether the final initial content is an numpy.array or not.
        :param c_T_idx (bool): whether to define the variable at each concentration, c, and temperature, T.
        :return:
        """
        if not isinstance(names, list):
            names = [names]

        if val_type.lower() in ["scalar"]:
            initial_val = initval
        elif val_type.lower() in ["vector"]:
            initial_val = [initval, initval, initval]
        elif val_type.lower() in ["tensor", "matrix"]:
            # initial_val = [ [initval, initval, initval], [initval, initval, initval], [initval, initval, initval] ]
            initial_val = [[initval for i in range(3)] for i in range(3)]

        for name in names:
            for tp in ["n", "p"]:
                self[grid][tp][name] = 0.0
                if grid in ["kgrid"]:
                    init_content = [[initial_val for i in range(len(self[grid][tp]["kpoints"][j]))]
                                    for j in range(self.cbm_vbm[tp]["included"])]
                elif grid in ["egrid"]:
                    init_content = [initial_val for i in range(len(self[grid][tp]["energy"]))]
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
        """return a list nsteps number of k-points between k1 & k2 excluding k1 & k2 themselves. k1 & k2 are nparray"""
        dkii = (k2 - k1) / float(nsteps + 1)
        return [k1 + i * dkii for i in range(1, nsteps + 1)]



    def get_intermediate_kpoints_list(self, k1, k2, nsteps):
        """return a list nsteps number of k-points between k1 & k2 excluding k1 & k2 themselves. k1 & k2 are lists"""
        # dkii = (k2 - k1) / float(nsteps + 1)
        if nsteps < 1:
            return []
        dk = [(k2[i] - k1[i]) / float(nsteps + 1) for i in range(len(k1))]
        # return [k1 + i * dkii for i in range(1, nsteps + 1)]
        return [[k1[i] + n * dk[i] for i in range(len(k1))] for n in range(1, nsteps + 1)]



    @staticmethod
    def get_perturbed_ks(k):
        all_perturbed_ks = []
        # for p in [0.01, 0.03, 0.05]:
        for p in [0.05, 0.1]:
            all_perturbed_ks.append([k_i + p * np.sign(random() - 0.5) for k_i in k])
        return all_perturbed_ks



    def get_ks_with_intermediate_energy(self, kpts, energies, max_Ediff=None, target_Ediff=None):
        final_kpts_added = []
        target_Ediff = target_Ediff or self.dE_min
        for tp in ["n", "p"]:
            max_Ediff = max_Ediff or min(self.Ecut[tp], 10 * k_B * max(self.temperatures))
            if tp not in self.all_types:
                continue
            ies_sorted = list(np.argsort(energies[tp]))
            if tp == "p":
                ies_sorted.reverse()
            for idx, ie in enumerate(ies_sorted[:-1]):
                Ediff = abs(energies[tp][ie] - energies[tp][ies_sorted[0]])
                if Ediff > max_Ediff:
                    break
                final_kpts_added += self.get_perturbed_ks(kpts[ies_sorted[idx]])

                # final_kpts_added += self.get_intermediate_kpoints_list(list(kpts[ies_sorted[idx]]),
                #                                    list(kpts[ies_sorted[idx+1]]), max(int(Ediff/target_Ediff) , 1))
        return kpts_to_first_BZ(final_kpts_added)



    def get_adaptive_kpoints(self, kpts, energies, adaptive_Erange, nsteps):
        kpoints_added = {"n": [], "p": []}
        for tp in ["n", "p"]:
            if tp not in self.all_types:
                continue
            # TODO: if this worked, change it so that if self.dopings does not involve either of the types, don't add k-points for it
            ies_sorted = list(np.argsort(energies[tp]))
            if tp == "p":
                ies_sorted.reverse()
            for ie in ies_sorted:
                Ediff = abs(energies[tp][ie] - energies[tp][ies_sorted[0]])
                if Ediff >= adaptive_Erange[0] and Ediff < adaptive_Erange[-1]:
                    kpoints_added[tp].append(kpts[ie])

        final_kpts_added = []
        for tp in ["n", "p"]:
            # final_kpts_added = []
            # TODO: in future only add the relevant k-poits for "kpoints" for each type separately
            # print kpoints_added[tp]
            for ik in range(len(kpoints_added[tp]) - 1):
                final_kpts_added += self.get_intermediate_kpoints_list(list(kpoints_added[tp][ik]),
                                                                       list(kpoints_added[tp][ik + 1]), nsteps)

        return kpts_to_first_BZ(final_kpts_added)


    def get_sym_eq_ks_in_first_BZ(self, k, cartesian=False):
        """

        :param k (numpy.array): kpoint fractional coordinates
        :param cartesian (bool): if True, the output would be in cartesian (but still reciprocal) coordinates
        :return:
        """
        fractional_ks = [np.dot(k, self.rotations[i]) for i in range(len(self.rotations))]
        fractional_ks = kpts_to_first_BZ(fractional_ks)
        if cartesian:
            return [self.get_cartesian_coords(k_frac) / A_to_nm for k_frac in fractional_ks]
        else:
            return fractional_ks


    def calc_poly_energy(self, xkpt, tp, ib):
        """

        Args:
            xkpt ([float]): fractional coordinates of a given k-point
            tp (str): 'n' or 'p' type
            ib (int): the band index

        Returns:
            (energy(eV), velocity (cm/s), effective mass) from a parabolic band
        """
        energy, velocity, effective_m = get_poly_energy(
            self.get_cartesian_coords(xkpt) / A_to_nm,
            poly_bands=self.poly_bands, type=tp, ib=ib, bandgap=self.dft_gap + self.scissor)
        return energy, velocity, effective_m


    def init_kgrid(self, kpts, important_points, analytical_band_tuple=None, once_called=False, delete_off_points=True):
        """

        Args:
            coeff_file (str): address to the cube file generated by BoltzTraP
            kgrid_tp (str): type of the mesh options are:
                'very coarse', 'coarse', 'fine', 'very fine'
        Returns:
        """
        # self.logger.debug('begin profiling init_kgrid: a "{}" grid'.format(kgrid_tp))
        # start_time = time.time()

        corrupt_tps = []
        if analytical_band_tuple is None:
            analytical_band_tuple = [None for _ in range(9)]

        if self.interpolation == "boltztrap1":
            analytical_bands, engre, nwave, nsym, nstv, vec, vec2, out_vec2, br_dir = analytical_band_tuple


        # TODO-JF (long-term): adaptive mesh is a good idea but current implementation is useless, see if you can come up with better method after talking to me
        if self.adaptive_mesh:
            raise NotImplementedError("adaptive mesh has not yet been "
                                      "implemented, please check back later!")

        # TODO: remove anything with "weight" later if ended up not using weights at all!
        kweights = {tp: [1.0 for i in kpts[tp]] for tp in ["n", "p"]}

        # actual initiation of the kgrid
        self.kgrid = {
            "n": {},
            "p": {}}
        # self.num_bands = {"n": {}, "p": {}}
        self.num_bands = {"n": 1, "p": 1}
        # self.logger.debug('here the n-type kgrid :\n{}'.format(kpts['n']))
        for tp in ["n", "p"]:
            self.num_bands[tp] = self.cbm_vbm[tp]["included"]
            self.kgrid[tp]["kpoints"] = [kpts[tp] for ib in range(self.num_bands[tp])]
            self.kgrid[tp]["kweights"] = [kweights[tp] for ib in range(self.num_bands[tp])]

        self.initialize_var("kgrid", ["energy", "a", "c", "norm(v)", "norm(k)"], "scalar", 0.0, is_nparray=False, c_T_idx=False)
        self.initialize_var("kgrid", ["velocity"], "vector", 0.0, is_nparray=False, c_T_idx=False)
        self.velocity_signed = {tp: np.array([[[0,0,0] for ik in range(len(kpts[tp]))] for ib in range(self.num_bands[tp])]) for tp in ['n', 'p']}
        self.initialize_var("kgrid", ["effective mass"], "tensor", 0.0, is_nparray=False, c_T_idx=False)

        start_time = time.time()

        rm_idx_list = {"n": [[] for i in range(self.cbm_vbm["n"]["included"])],
                       "p": [[] for i in range(self.cbm_vbm["p"]["included"])]}
        # @albalu why are these variables initialized separately from the ones above?
        self.initialize_var("kgrid", ["old cartesian kpoints", "cartesian kpoints"], "vector", 0.0, is_nparray=False, c_T_idx=False)
        self.initialize_var("kgrid", ["norm(k)", "norm(actual_k)"], "scalar", 0.0, is_nparray=False, c_T_idx=False)

        self.logger.debug("The DFT gap right before calculating final energy values: {}".format(self.dft_gap))

        for i, tp in enumerate(["p", "n"]):
            self.cbm_vbm[tp]["cartesian k"] = self.get_cartesian_coords(self.cbm_vbm[tp]["kpoint"])/A_to_nm
            self.cbm_vbm[tp]["all cartesian k"] = self.get_sym_eq_ks_in_first_BZ(self.cbm_vbm[tp]["kpoint"], cartesian=True)
            self.cbm_vbm[tp]["all cartesian k"] = remove_duplicate_kpoints(self.cbm_vbm[tp]["all cartesian k"])

            # self.important_pts[tp] = [self.get_cartesian_coords(k)/A_to_nm for k in self.important_pts[tp]]

            sgn = (-1) ** i
            for ib in range(self.cbm_vbm[tp]["included"]):
                self.kgrid[tp]["old cartesian kpoints"][ib] = self.get_cartesian_coords(
                    self.kgrid[tp]["kpoints"][ib]) / A_to_nm

                # WE MAKE A COPY HERE OTHERWISE THE TWO LISTS CHANGE TOGETHER
                self.kgrid[tp]["cartesian kpoints"][ib] = np.array(self.kgrid[tp]["old cartesian kpoints"][ib])

                s_orbital, p_orbital = self.get_dft_orbitals(bidx=self.cbm_vbm[tp]["bidx"] - 1 - sgn * ib)
                orbitals = {"s": s_orbital, "p": p_orbital}
                fit_orbs = {orb: griddata(points=np.array(self.DFT_cartesian_kpts), values=np.array(orbitals[orb]),
                    xi=np.array(self.kgrid[tp]["old cartesian kpoints"][ib]), method='nearest') for orb in orbitals.keys()}

                if self.interpolation == "boltztrap1":
                    if self.parallel and self.poly_bands is None:
                        results = Parallel(n_jobs=self.num_cores)(delayed(get_energy)(self.kgrid[tp]["kpoints"][ib][ik],
                                 engre[i * self.cbm_vbm["p"]["included"] + ib], nwave, nsym, nstv, vec, vec2, out_vec2,
                                 br_dir) for ik in range(len(self.kgrid[tp]["kpoints"][ib])))
                elif self.interpolation == "boltztrap2":
                    fitted = fite.getBands(np.array(kpts[tp]), *self.interp_params)
                else:
                    raise ValueError('Unsupported interpolation: "{}"'.format(self.interpolation))

                # TODO-JF: the general function for calculating the energy, velocity and effective mass can b
                for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                    # min_dist_ik = np.array([norm(ki - self.kgrid[tp]["old cartesian kpoints"][ib][ik]) for ki in self.cbm_vbm[tp]["all cartesian k"]]).argmin()
                    # self.kgrid[tp]["cartesian kpoints"][ib][ik] = self.kgrid[tp]["old cartesian kpoints"][ib][ik] - self.cbm_vbm[tp]["all cartesian k"][min_dist_ik]


                    # self.kgrid[tp]["cartesian kpoints"][ib][ik] = get_closest_k(self.kgrid[tp]["old cartesian kpoints"][ib][ik], self.important_pts[tp], return_diff=True)
                    self.kgrid[tp]["cartesian kpoints"][ib][ik] = \
                        self.get_cartesian_coords(get_closest_k(
                            self.kgrid[tp]["kpoints"][ib][ik], important_points[tp], return_diff=True)) / A_to_nm

                    # # The following 2 lines (i.e. when the closest kpoints to equivalent extrema are calculated in fractional coordinates) would change the anisotropic test! not sure why
                    # closest_frac_k = np.array(get_closest_k(self.kgrid[tp]["kpoints"][ib][ik], self.important_pts[tp]), return_diff=True)
                    # self.kgrid[tp]["cartesian kpoints"][ib][ik] = self.get_cartesian_coords(closest_frac_k) / A_to_nm

                    self.kgrid[tp]["norm(k)"][ib][ik] = norm(self.kgrid[tp]["cartesian kpoints"][ib][ik])
                    # if abs(self.kgrid[tp]["norm(k)"][ib][ik] - 9.8) < 1.7:
                    #     self.kgrid[tp]["norm(k)"][ib][ik] = abs(self.kgrid[tp]["norm(k)"][ib][ik] - 9.8)
                    self.kgrid[tp]["norm(actual_k)"][ib][ik] = norm(self.kgrid[tp]["old cartesian kpoints"][ib][ik])

                    if self.poly_bands is None:
                        if self.interpolation == "boltztrap1":
                            if not self.parallel:
                                energy, de, dde = get_energy(
                                    self.kgrid[tp]["kpoints"][ib][ik], engre[i * self.cbm_vbm["p"]["included"] + ib],
                                    nwave, nsym, nstv, vec, vec2, out_vec2, br_dir=br_dir)
                                energy = energy * Ry_to_eV - sgn * self.scissor / 2.0
                                velocity_signed = self.get_cartesian_coords(de) / hbar * A_to_m * m_to_cm * Ry_to_eV
                                # velocity =  abs(self.get_cartesian_coords(de, reciprocal=False)) / hbar * A_to_m * m_to_cm * Ry_to_eV  # to get v in cm/s
                                velocity = abs( self.get_cartesian_coords(de, reciprocal=False) ) / (hbar*2*pi) / 0.52917721067 * A_to_m * m_to_cm * Ry_to_eV

                                # effective_mass = hbar ** 2 / (dde * 4 * pi ** 2) / m_e / A_to_m ** 2 * e * Ry_to_eV  # m_tensor: the last part is unit conversion
                                effective_mass = 1/(dde/ 0.52917721067) * e / Ry_to_eV / A_to_m**2 * (hbar*2*np.pi)**2 / m_e
                            else:
                                energy = results[ik][0] * Ry_to_eV - sgn * self.scissor / 2.0
                                velocity_signed = self.get_cartesian_coords(results[ik][1]) / hbar * A_to_m * m_to_cm * Ry_to_eV
                                # velocity = abs(results[ik][1] / hbar * A_to_m * m_to_cm * Ry_to_eV)
                                # velocity =  abs(self.get_cartesian_coords(results[ik][1])) / hbar * A_to_m * m_to_cm * Ry_to_eV  # to get v in cm/s
                                # velocity = abs( np.dot(results[ik][1], self._rec_lattice.inv_matrix) )   / hbar * A_to_m * m_to_cm * Ry_to_eV  # to get v in cm/s
                                # velocity = abs( np.dot(np.linalg.inv(self._rec_lattice.matrix), results[ik][1]) ) / hbar * A_to_m * m_to_cm * Ry_to_eV  # to get v in cm/s # anisotropic InP
                                # velocity = abs( np.dot(results[ik][1], np.linalg.inv(self._rec_lattice.matrix)) ) / hbar * A_to_m * m_to_cm * Ry_to_eV  # to get v in cm/s # isotropic InP but GaAs v vs. E looks wrong
                                velocity = abs( self.get_cartesian_coords(results[ik][1], reciprocal=False) ) / (hbar*2*pi) / 0.52917721067 * A_to_m * m_to_cm * Ry_to_eV

                                # velocity = abs( np.dot(np.linalg.inv(self._rec_lattice.matrix), results[ik][1]) ) / hbar * A_to_m * m_to_cm * Ry_to_eV  # 20180320: test as the units work out only if A is in the numerator!
                                # velocity = abs( np.dot(results[ik][1], np.linalg.inv(self._rec_lattice.matrix)) ) / hbar * A_to_m * m_to_cm * Ry_to_eV  # 20180320: InP isotropic but both GaAs and InP very low mobility
                                # velocity = abs( 1.0 /( np.dot(self._rec_lattice.matrix, 1.0/results[ik][1]) ) ) / hbar * A_to_m * m_to_cm * Ry_to_eV  # 20180320: results in nan (1/zero!)
                                # velocity = abs( np.dot(self._rec_lattice.matrix, results[ik][1]) ) / hbar * A_to_m * m_to_cm * Ry_to_eV  # 20180320: THIS results in isotropic InP but mobility/velocity values too high!

                                # effective_mass = hbar ** 2 / (np.dot(self._rec_lattice.matrix**2, results[ik][2]) * 4 * pi ** 2) / m_e / A_to_m ** 2 * e * Ry_to_eV  # m_tensor: the last part is unit conversion
                                effective_mass = 1/(results[ik][2]/ 0.52917721067) * e / Ry_to_eV / A_to_m**2 * (hbar*2*np.pi)**2 / m_e

                            self.velocity_signed[tp][ib][ik] = velocity_signed
                        elif self.interpolation == "boltztrap2":
                            iband = self.cbm_vbm["p"]["bidx"]-1 + i*self.cbm_vbm["p"]["included"]
                            energy = fitted[0][iband, ik]*Ry_to_eV - sgn*self.scissor/2.
                            velocity = abs(fitted[1][:, ik, iband].T / hbar * A_to_m * m_to_cm * Ry_to_eV)
                            effective_mass =  hbar ** 2 / (
                                            fitted[2][:, :, ik, iband].T * 4 * pi ** 2) / m_e / A_to_m ** 2 * e * Ry_to_eV
                        else:
                            raise ValueError("")

                    else:
                        energy, velocity, effective_mass = get_poly_energy(self.kgrid[tp]["cartesian kpoints"][ib][ik],
                                                                           poly_bands=self.poly_bands,
                                                                           type=tp, ib=ib,
                                                                           bandgap=self.dft_gap + self.scissor)

                    self.kgrid[tp]["energy"][ib][ik] = energy
                    self.kgrid[tp]["velocity"][ib][ik] = velocity
                    # if tp == 'n':
                    #     self.logger.debug('here velocity:\n{}'.format(velocity))
                    # if tp == 'n':
                    #     print("k_frac = {}".format(self.kgrid['n']["kpoints"][ib][ik]))
                    #     print("k_cart = {}".format(self.kgrid['n']["cartesian kpoints"][ib][ik]))
                    #     print("k_old_cart = {}".format(self.kgrid['n']["old cartesian kpoints"][ib][ik]))
                    #     print("v = {}".format(velocity_signed))
                    # TODO: the following enforces isotropy but it's not necessary as bs_is_isotropic is just a different formulation and isotropy from bs should be taken into account
                    # if self.bs_is_isotropic:
                    #     self.kgrid[tp]["velocity"][ib][ik] = [norm(velocity)/sq3 for i in range(3)]
                    # else:
                    #     self.kgrid[tp]["velocity"][ib][ik] = velocity
                    self.kgrid[tp]["norm(v)"][ib][ik] = norm(velocity)

                    # if self.kgrid[tp]["velocity"][ib][ik][0] < self.v_min or  \
                    #                 self.kgrid[tp]["velocity"][ib][ik][1] < self.v_min \
                    #         or self.kgrid[tp]["velocity"][ib][ik][2] < self.v_min or \
                    # if ((self.kgrid[tp]["velocity"][ib][ik] < self.v_min).any() or \
                    #     abs(self.kgrid[tp]["energy"][ib][ik] - self.cbm_vbm[tp]["energy"]) > self.Ecut[tp]) \
                    #     and (len(rm_idx_list[tp][ib]) + 10 < len(self.kgrid[tp]['kpoints'][ib])):
                    #     # TODO: remove this if when treating valence valleys and conduction valleys separately
                    #     # print('here debug removing k-points')
                    #         # print(tp)
                    #         # print(self.ibrun)
                    #         # print(important_points[tp])
                    #         # print(self.count_mobility[self.ibrun])
                    #         # print(self.kgrid[tp]["kpoints"][ib][ik])
                    #         # print(self.kgrid[tp]["cartesian kpoints"][ib][ik])
                    #         # print(self.kgrid[tp]["energy"][ib][ik])
                    #         # print(self.kgrid[tp]["velocity"][ib][ik])
                    #         rm_idx_list[tp][ib].append(ik)

                    # self.logger.info('cbm_vbm right before checking for omission: {}'.format(self.cbm_vbm))

                    if (len(rm_idx_list[tp][ib]) + 20 < len(self.kgrid[tp]['kpoints'][ib])) and (
                            (self.kgrid[tp]["velocity"][ib][ik] < self.v_min).any() \
                        or \
                            (abs(self.kgrid[tp]["energy"][ib][ik] - self.cbm_vbm[tp]["energy"]) > self.Ecut[tp]) \
                        or \
                            ((self.max_normk[tp]) and (self.kgrid[tp]["norm(k)"][ib][ik] > self.max_normk[tp]) and (self.poly_bands0 is None))
                    ):
                        # if abs(self.kgrid[tp]["energy"][ib][ik] - self.cbm_vbm[tp]["energy"]) > self.Ecut[tp]:
                        #     print('here energy diff large')
                        #     print(self.cbm_vbm)
                        #     self.counter += 1
                        #     print(self.counter)
                        rm_idx_list[tp][ib].append(ik)

                    #
                    # # TODO: AF must test how large norm(k) affect ACD, IMP and POP and see if the following is necessary
                    # # if self.max_normk0:
                    # if (self.max_normk[tp]) and (self.kgrid[tp]["norm(k)"][ib][ik] > self.max_normk[tp]) \
                    #         # and (len(rm_idx_list[tp][ib]) + 0 < len(self.kgrid[tp]['kpoints'][ib])) \
                    #         and self.poly_bands0 is None: # this last part to avoid an error in test_poly_bands
                    #     if self.kgrid[tp]["norm(k)"][ib][ik] > self.max_normk[tp]:
                    #         print('here norm(k) too large')
                    #         print(self.kgrid[tp]["norm(k)"][ib][ik])
                    #         print(len(rm_idx_list[tp][ib]))
                    #     rm_idx_list[tp][ib].append(ik)
                    #
                    # # This caused some tests to break as it was changing mobility
                    # # values and making them more anisotropic since it was removing Gamma from GaAs
                    # # if self.kgrid[tp]["norm(k)"][ib][ik] < 0.0001:
                    # #     self.logger.debug('HERE removed k-point {} ; cartesian: {}'.format(
                    # #         self.kgrid[tp]["kpoints"][ib][ik], self.kgrid[tp]["cartesian kpoints"][ib][ik]))
                    # #     rm_idx_list[tp][ib].append(ik)

                    self.kgrid[tp]["effective mass"][ib][ik] = effective_mass

                    if self.poly_bands is None:
                        self.kgrid[tp]["a"][ib][ik] = fit_orbs["s"][ik]/ (fit_orbs["s"][ik]**2 + fit_orbs["p"][ik]**2)**0.5
                        self.kgrid[tp]["c"][ib][ik] = (1 - self.kgrid[tp]["a"][ib][ik]**2)**0.5
                    else:
                        self.kgrid[tp]["a"][ib][ik] = 1.0  # parabolic: s-only
                        self.kgrid[tp]["c"][ib][ik] = 0.0

            self.logger.debug("average of the {}-type group velocity in kgrid:\n {}".format(
                        tp, np.mean(self.kgrid[tp]["velocity"][0], 0)))

        rearranged_props = ["velocity",  "effective mass", "energy", "a", "c",
                            "kpoints", "cartesian kpoints",
                            "old cartesian kpoints", "kweights",
                            "norm(v)", "norm(k)", "norm(actual_k)"]

        self.logger.debug("time to calculate E, v, m_eff at all k-points: \n {}".format(time.time()-start_time))
        start_time = time.time()

        # TODO: the following is temporary, for some reason if # of kpts in different bands are NOT the same,
        # I get an error that _all_elastic is a list! so 1/self.kgrid[tp]["_all_elastic"][c][T][ib] cause error int/list!
        # that's why I am removing indexes from the first band at all bands! this is temperary
        # suggested solution: make the band index a key in the dictionary of kgrid rather than list index so we
        # can treat each band independently without their dimensions required to match!
        # TODO-AF or TODO-JF (mid-term): set the band index as a key in dictionary throughout AMSET to enable independent modification of bands information
        for tp in ["n", "p"]:
            rm_idx_list[tp] = [rm_idx_list[tp][0] for ib in range(self.cbm_vbm[tp]["included"])]

        self.rm_idx_list = deepcopy(rm_idx_list)   # format: [tp][ib][ik]
        # print('sanity check...')
        # print('ib={}'.format(ib))
        # if len(self.kgrid['n']["kpoints"]) > 1:
        #     print(self.kgrid['n'])
        #     raise ValueError('ib=0 must always')
        if delete_off_points:
            # print('BEFORE REMOVING')
            # print(len(self.kgrid['n']["kpoints"][ib]))
            # print(len(self.kgrid['p']["kpoints"][ib]))
            self.remove_indexes(rm_idx_list, rearranged_props=rearranged_props)
            # print('AFTER REMOVING')
            # print(len(self.kgrid['n']["kpoints"][ib]))
            # print(len(self.kgrid['p']["kpoints"][ib]))
        self.logger.debug("dos_emin = {} and dos_emax= {}".format(self.dos_emin, self.dos_emax))

        self.logger.debug('current cbm_vbm:\n{}'.format(self.cbm_vbm))
        for tp in ["n", "p"]:
            for ib in range(len(self.kgrid[tp]["energy"])):
                self.logger.info("Final # of {}-kpts in band #{}: {}".format(tp, ib, len(self.kgrid[tp]["kpoints"][ib])))

            if len(self.kgrid[tp]["kpoints"][0]) < 5:
                # raise ValueError("VERY BAD {}-type k-mesh; please change the k-mesh and try again!".format(tp))
                corrupt_tps.append(tp)
        self.logger.debug("time to calculate energy, velocity, m* for all: {} seconds".format(time.time() - start_time))

        # sort "energy", "kpoints", "kweights", etc based on energy in ascending order and keep track of old indexes
        e_sort_idx_2 = self.sort_vars_based_on_energy(args=rearranged_props, ascending=True)
        self.pos_idx_2 = deepcopy(e_sort_idx_2)
        for tp in ['n', 'p']:
            for ib in range(self.num_bands[tp]):
                self.pos_idx_2[tp][ib] = np.array(range(len(e_sort_idx_2[tp][ib])))[e_sort_idx_2[tp][ib]].argsort()

        # for ib in range(self.num_bands['n']):
        #     for ik in range(20):
        #         print("k_frac = {}".format(self.kgrid['n']["kpoints"][ib][ik]))
        #         print("k_cart = {}".format(self.kgrid['n']["cartesian kpoints"][ib][ik]))
        #         print("v = {}".format((self.velocity_signed["n"][0][e_sort_idx_2["n"][0]])[ik]))

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
                # TODO: change how W_POP is set, user set a number or a file that can be fitted and inserted to kgrid
                self.kgrid[tp]["W_POP"][ib] = [self.W_POP for i in range(len(self.kgrid[tp]["kpoints"][ib]))]
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
        """sort the list of variables specified by "args" (type: [str]) in self.kgrid based on the "energy" values
        in each band for both "n"- and "p"-type bands and in ascending order by default."""
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
        generates the indexes of k' points that have the same energy (for elastic scattering) as E(k) or
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
                    # TODO: this should be an exception but for now I turned to warning for testing.
                    warnings.warn("the k-grid is too coarse for an acceptable simulation of elastic scattering in {};"
                        .format(self.tp_title[tp]))

                avg_Ediff = sum(self.ediff_scat[tp]) / max(len(self.ediff_scat[tp]), 1)
                if avg_Ediff > avg_Ediff_tolerance:
                    #TODO: change it back to ValueError as it was originally, it was switched to warning for fast debug
                    warnings.warn("{}-type average energy difference of the enforced scattered k-points is more than"
                                  " {}, try running with a more dense k-point mesh".format(tp, avg_Ediff_tolerance))

        # inelastic scattering
        if "POP" in self.inelastic_scatterings:
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

        # fractional_ks = [np.dot(frac_k, self.rotations[i]) + self.translations[i] for i in range(len(self.rotations))]
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
        """Returns the sorted (based on angle, X) list of angle and band and k-point indexes of all the points
            that are within tolerance of E + E_change
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

                    #TODO: the following condition make the tests fail even for GaAs and Gamma only and max_normk of 4; see why!??!
                    # AF: Maybe because symmetrically equivalent pockets are far from each other in BZ but can/should have scattering with each other?
                    if norm(self.kgrid[tp]["old cartesian kpoints"][ib_prm][ik_prm] - self.kgrid[tp]["old cartesian kpoints"][ib][ik]) < 2*self.max_normk[tp]:
                        # if (X_ib_ik[1], X_ib_ik[2]) not in [(entry[1], entry[2]) for entry in result]: # 2nd condition to avoid inter-band scattering
                        # we don't need the above since we only look at one band at a time (ib=0 always)
                        result.append(X_ib_ik)
                    ik_prm += step

        if E_change != 0.0:
        # if True:
            # If fewer than forced_min_npoints number of points were found, just return a few surroundings of the same band
            ib_prm = ib
            ik_closest_E = np.abs(self.kgrid[tp]["energy"][ib_prm] - E_prm).argmin()

            for step, start in [(1, 0), (-1, -1)]:
                # step -1 is in case we reached the end (ik_prm == nk - 1); then we choose from the lower energy k-points
                ik_prm = ik_closest_E + start  # go up from ik_closest_E, down from ik_closest_E - 1

                # the following if statement, makes GaAs POP results anisotropic which they should not be
                # if norm(self.kgrid[tp]["old cartesian kpoints"][ib_prm][ik_prm] - self.kgrid[tp]["old cartesian kpoints"][ib][ik]) < 2*self.max_normk[tp]:
                #     if E_change == 0.0:
                #         if ik_prm != ik_closest_E:
                #             result.append((cos_angle(k, self.kgrid[tp]["cartesian kpoints"][ib_prm][ik_prm]), ib_prm, ik_prm))
                #             self.nforced_scat[tp] += 1
                #     else:
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
        return the scattering rate at wave vector k at a certain concentration and temperature
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


    def integrate_over_DOSxE_dE(self, func, tp, fermi, T, interpolation_nsteps=None, normalize_energy=False):
        if not interpolation_nsteps:
            interpolation_nsteps = max(200, int(500.0 / len(self.egrid[tp]["energy"])))
        integral = 0.0
        for ie in range(len(self.egrid[tp]["energy"]) - 1):
            E = self.egrid[tp]["energy"][ie]
            dE = (self.egrid[tp]["energy"][ie + 1] - E) / interpolation_nsteps
            if normalize_energy:
                E -= self.cbm_vbm[tp]["energy"]
                fermi -= self.cbm_vbm[tp]["energy"]
            dS = (self.egrid[tp]["DOS"][ie + 1] - self.egrid[tp]["DOS"][ie]) / interpolation_nsteps
            for i in range(interpolation_nsteps):
                # integral += dE * (self.egrid[tp]["DOS"][ie] + i * dS)*func(E + i * dE, fermi, T)*self.Efrequency[tp][ie]
                integral += dE * (self.egrid[tp]["DOS"][ie] + i * dS) * func(E + i * dE, fermi, T)
        return integral
        # return integral/sum(self.Efrequency[tp][:-1])


    # def grid_index_from_list_index(self, list_index, tp):
    #     N = self.kgrid_array[tp].shape
    #     count = list_index
    #     i, j, k = (0,0,0)
    #     while count >= N[2]*N[1]:
    #         count -= N[2]*N[1]
    #         i += 1
    #     while count >= N[2]:
    #         count -= N[2]
    #         j += 1
    #     k = count
    #     return (i,j,k)


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


    # takes a coordinate grid in the form of a numpy array (CANNOT have missing points) and a function to integrate and
    # finds the integral using finite differences; missing points should be input as 0 in the function
    def integrate_over_k(self, func_grid, tp):
        """
        Args:
            func_grid:

        Returns:

        in the interest of not prematurely optimizing, func_grid must be a perfect grid: the only deviation from
        the cartesian coordinate system can be uniform stretches, as in the distance between adjacent planes of points
        can be any value, but no points can be missing from the next plane

        in this case the format of fractional_grid is a 4d grid
        the last dimension is a vector of the k point fractional coordinates
        the dv grid is 3d and the indexes correspond to those of func_grid
        """
        if func_grid.ndim == 3:
            return np.sum(func_grid * self.dv_grid[tp])
        return [np.sum(func_grid[:,:,:,i] * self.dv_grid[tp]) for i in range(func_grid.shape[3])]



    def integrate_over_BZ(self, prop_list, tp, c, T, xDOS=False, xvel=False, weighted=True):

        weighted = False
        wpower = 1
        if xvel:
            wpower += 1
        integral = np.array([self.gs, self.gs, self.gs])
        for ie in range(len(self.egrid[tp]["energy"]) - 1):
            dE = abs(self.egrid[tp]["energy"][ie + 1] - self.egrid[tp]["energy"][ie])
            sum_over_k = np.array([self.gs, self.gs, self.gs])
            for ib, ik in self.kgrid_to_egrid_idx[tp][ie]:
                k_nrm = self.kgrid[tp]["norm(k)"][ib][ik]
                product = k_nrm ** 2 / self.kgrid[tp]["norm(v)"][ib][ik] * 4 * pi / hbar
                if xvel:
                    product *= self.kgrid[tp]["velocity"][ib][ik]
                for j, p in enumerate(prop_list):
                    if p[0] == "/":
                        product /= self.kgrid[tp][p.split("/")[-1]][c][T][ib][ik]
                    elif p[0] == "1":  # this assumes that the property is 1-f0 for example
                        product *= 1 - self.kgrid[tp][p.split("-")[-1].replace(" ", "")][c][T][ib][ik]
                    else:
                        product *= self.kgrid[tp][p][c][T][ib][ik]
                sum_over_k += product
            if xDOS:
                sum_over_k *= self.egrid[tp]["DOS"][ie]
            if weighted:
                sum_over_k *=self.Efrequency[tp][ie] / float(self.sym_freq[tp][ie])
            integral += sum_over_k * dE

        if weighted:
            return integral
            # return integral / sum([freq ** (wpower) for freq in self.Efrequency[tp][:-1]])
        else:
            return integral
            # return integral / sum([self.egrid[tp]["f0"][c][T][ie][0]*self.Efrequency[tp][ie] for ie in range(len(self.Efrequency[tp][:-1]))])



    def integrate_over_normk(self, prop_list, tp, c, T, xDOS, interpolation_nsteps=None):
        integral = self.gs
        normk_tp = "norm(k)"
        if not interpolation_nsteps:
            interpolation_nsteps = max(200, int(500.0 / len(self.kgrid[tp]["kpoints"][0])))
        for ib in [0]:
            normk_sorted_idx = np.argsort(self.kgrid[tp][normk_tp][ib])
            diff = [0.0 for prop in prop_list]

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
                # try:
                #     integral_vec *= np.mean(vec, axis=1)
                # except IndexError:
                #     integral_vec *= vec
            integral = np.sum(integral_vec)
            # for j, ik in enumerate(normk_sorted_idx[:-1]):
            #     ik_next = normk_sorted_idx[j+1]
            #     normk = self.kgrid[tp][normk_tp][ib][ik]
            #     dk = (self.kgrid[tp][normk_tp][ib][ik_next] - normk)/interpolation_nsteps
            #     if dk == 0.0:
            #         continue
            #     if xDOS:
            #         dS = ((self.kgrid[tp][normk_tp][ib][ik_next]/pi)**2 - \
            #              (self.kgrid[tp][normk_tp][ib][ik]/pi)**2)/interpolation_nsteps
            #     for j, p in enumerate(prop_list):
            #         if p[0] == "/":
            #             diff[j] = (self.kgrid[tp][p.split("/")[-1]][c][T][ib][ik_next] - \
            #                             self.kgrid[tp][p.split("/")[-1]][c][T][ib][ik]) / interpolation_nsteps
            #         elif p[0] == "1":
            #             diff[j] = ((1 - self.kgrid[tp][p.split("-")[-1].replace(" ", "")][c][T][ib][ik_next]) - \
            #                       (1 - self.kgrid[tp][p.split("-")[-1].replace(" ", "")][c][T][ib][ik])) / interpolation_nsteps
            #         else:
            #             diff[j] = (self.kgrid[tp][p][c][T][ib][ik_next] - self.kgrid[tp][p][c][T][ib][ik]) / interpolation_nsteps
            #
            #     for i in range(interpolation_nsteps):
            #         multi = dk
            #         for j, p in enumerate(prop_list):
            #             if p[0] == "/":
            #                 multi /= self.kgrid[tp][p.split("/")[-1]][c][T][ib][ik] + diff[j] * i
            #             elif "1" in p:
            #                 multi *= 1 - self.kgrid[tp][p.split("-")[-1].replace(" ", "")][c][T][ib][ik] + diff[j] * i
            #             else:
            #                 multi *= self.kgrid[tp][p][c][T][ib][ik] + diff[j] * i
            #         if xDOS:
            #             multi *= (self.kgrid[tp][normk_tp][ib][ik]/pi)**2 + dS * i
            #         integral += multi
        return integral



    def integrate_over_E(self, prop_list, tp, c, T, xDOS=False, xvel=False, weighted=False, interpolation_nsteps=None):
        # for now I keep weighted as False, to re-enable weighting, all GaAs tests should be re-evaluated.

        weighted = False

        wpower = 1
        if xvel:
            wpower += 1
        imax_occ = len(self.Efrequency[tp][:-1])

        if not interpolation_nsteps:
            interpolation_nsteps = max(200, int(500.0 / len(self.egrid[tp]["energy"])))
            # interpolation_nsteps = 1
        diff = [0.0 for prop in prop_list]
        integral = self.gs
        # for ie in range(len(self.egrid[tp]["energy"]) - 1):
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
            if weighted:
                dweight = (self.Efrequency[tp][ie+1] / float(self.sym_freq[tp][ie+1]) - \
                          self.Efrequency[tp][ie] / float(self.sym_freq[tp][ie]) ) /interpolation_nsteps
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
                if weighted:
                    # integral += multi * self.Efrequency[tp][ie] ** wpower
                    integral += multi * (self.Efrequency[tp][ie] / float(self.sym_freq[tp][ie]) + dweight * i)
                else:
                    integral += multi
        integral = np.array(integral)
        if weighted:
            return integral
            # return integral/(sum(self.Efrequency[tp][:-1]))
        else:
            return integral



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
        returns the evaluated (float) expression inside the elastic equations
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

        # 20180307: I removed /sq3 from all elastic and inelastics: iso-aniso consistency can still be established
        # return (1 - X) * self.kgrid[tp]["norm(k)"][ib_prm][ik_prm] ** 2 * self.s_el_eq(sname, tp, c, T, k, k_prm) \
        #        * self.G(tp, ib, ik, ib_prm, ik_prm, X) / (self.kgrid[tp]["norm(v)"][ib_prm][ik_prm])


    def inel_integrand_X(self, tp, c, T, ib, ik, ib_prm, ik_prm, X, sname=None, g_suffix=""):
        """
        returns the evaluated (float) expression of the S_o & S_i(g) integrals.
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
        # elif tp=='n':
        #     print('abs(energy_diff) = {}'.format(abs(self.kgrid[tp]['energy'][ib_prm][ik_prm] - self.kgrid[tp]['energy'][ib][ik])))
        k = self.kgrid[tp]["cartesian kpoints"][ib][ik]
        f_th = self.kgrid[tp]["f_th"][c][T][ib][ik]
        k_prm = self.kgrid[tp]["cartesian kpoints"][ib_prm][ik_prm]
        # v_prm = self.kgrid[tp]["velocity"][ib_prm][ik_prm]
        if tp == "n":
            f = self.kgrid[tp]["f"][c][T][ib][ik]
            f_prm = self.kgrid[tp]["f"][c][T][ib_prm][ik_prm]
        else:
            f = 1 - self.kgrid[tp]["f"][c][T][ib][ik]
            f_prm = 1 - self.kgrid[tp]["f"][c][T][ib_prm][ik_prm]

        if k[0] == k_prm[0] and k[1] == k_prm[1] and k[2] == k_prm[2]:
            # return np.array([0.0, 0.0, 0.0])  # self-scattering is not defined;regardless, the returned integrand must be a vector
            return 0.0
        fermi = self.fermi_level[c][T]

        N_POP = self.kgrid[tp]["N_POP"][c][T][ib][ik]
        # N_POP = 1 / (np.exp(hbar * self.kgrid[tp]["W_POP"][ib][ik] / (k_B * T)) - 1)

        norm_diff = norm(k - k_prm)
        # if tp=='n' and norm_diff < 0.1:
        #     print('energy: {}'.format(self.kgrid[tp]['energy'][ib][ik]))
        #     print('energy: {}'.format(self.kgrid[tp]['energy'][ib_prm][ik_prm]))
        #     print('norm_diff: {}'.format(norm_diff))
        #     print
        if norm_diff < 1e-4:
            return 0.0


        # the term norm(k_prm)**2 is wrong in practice as it can be too big and originally we integrate |k'| from 0
        #TODO: this norm(v) in the following may need a /sq3
        integ = self.kgrid[tp]["norm(k)"][ib_prm][ik_prm]**2*self.G(tp, ib, ik, ib_prm, ik_prm, X)/\
                (self.kgrid[tp]["norm(v)"][ib_prm][ik_prm]*norm_diff**2/sq3)

        # only changing ik_prm of norm(k) to ik made S_o look more like isotropic
        # integ = self.kgrid[tp]["norm(k)"][ib][ik]**2*self.G(tp, ib, ik, ib_prm, ik_prm, X)/\
        #         (self.kgrid[tp]["norm(v)"][ib][ik]*norm_diff**2)

        # the following worked ok at superfine, the final POP and g matches with isotropic but S_i and S_o match are not good!
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
                        # only when very large # of k-points are present, make sense to parallelize as this function
                        # has become fast after better energy window selection
                        if self.parallel and len(self.kgrid[tp]["size"]) * \
                                max(self.kgrid[tp]["size"]) > 1000000000000:
                            # if False: Parallel should never be used here as it gets stuck or it's slower than series, perhaps since too much data (kgrid) transfer and pickling happens
                            results = Parallel(n_jobs=self.num_cores)(
                                delayed(calculate_Sio)(tp, c, T, ib, ik,
                                once_called, self.kgrid, self.cbm_vbm,
                                self.epsilon_s, self.epsilon_inf) for ik in
                                    range(len(self.kgrid[tp]["kpoints"][ib])))
                        else:
                            results = [calculate_Sio(tp, c, T, ib, ik,
                                    once_called, self.kgrid, self.cbm_vbm,
                                    self.epsilon_s, self.epsilon_inf) for ik in
                                    range(len(self.kgrid[tp]["kpoints"][ib]))]

                        for ik, res in enumerate(results):
                            self.kgrid[tp]["S_i"][c][T][ib][ik] = res[0]
                            self.kgrid[tp]["S_i_th"][c][T][ib][ik] = res[1]
                            if not once_called:
                                self.kgrid[tp]["S_o"][c][T][ib][ik] = res[2]
                                self.kgrid[tp]["S_o_th"][c][T][ib][ik] = res[3]



    def s_inelastic(self, sname=None, g_suffix=""):
        for tp in ["n", "p"]:
            for c in self.dopings:
                for T in self.temperatures:
                    for ib in range(len(self.kgrid[tp]["energy"])):
                        for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                            summation = np.array([0.0, 0.0, 0.0])
                            for X_E_index_name in ["X_Eplus_ik", "X_Eminus_ik"]:
                                summation += self.integrate_over_X(tp, self.kgrid[tp][X_E_index_name],
                                                                   self.inel_integrand_X,
                                                                   ib=ib, ik=ik, c=c, T=T, sname=sname + X_E_index_name,
                                                                   g_suffix=g_suffix)
                            self.kgrid[tp][sname][c][T][ib][ik] = summation * e ** 2 * self.kgrid[tp]["W_POP"][ib][ik] / (4 * pi * hbar) * (1 / self.epsilon_inf - 1 / self.epsilon_s) / epsilon_0 * 100 / e


                            # if norm(self.kgrid[tp][sname][c][T][ib][ik]) < 1:
                            #     self.kgrid[tp][sname][c][T][ib][ik] = [1, 1, 1]
                            # if norm(self.kgrid[tp][sname][c][T][ib][ik]) > 1e5:
                            #     print tp, c, T, ik, ib, summation, self.kgrid[tp][sname][c][T][ib][ik]



    def s_el_eq_isotropic(self, sname, tp, c, T, ib, ik):
        """
        returns elastic scattering rate (a numpy vector) at given point
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
        # v = self.kgrid[tp]["norm(v)"][ib][ik] # 20180307: I don't think /sq3 is necessary, iso-aniso consistency can still be established by removing all /sq3 from v/sq3
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
        the scattering rate equation for each elastic scattering name (sname)
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
                                    print("WARNING!!! here scattering {} < 1".format(sname))
                                    # if self.kgrid[tp]["df0dk"][c][T][ib][ik][0] > 1e-32:
                                    #     print self.kgrid[tp]["df0dk"][c][T][ib][ik]
                                    print(self.kgrid[tp]["X_E_ik"][ib][ik])

                                    self.kgrid[tp][sname][c][T][ib][ik] = [1e10, 1e10, 1e10]

                                if norm(self.kgrid[tp][sname][c][T][ib][ik]) > 1e20:
                                    print(self.kgrid[tp]['kpoints'][ib][ik])
                                    print(self.kgrid[tp]['cartesian kpoints'][ib][ik])
                                    print(self.kgrid[tp]['velocity'][ib][ik])
                                    print("WARNING!!! TOO LARGE of scattering rate for {}:".format(sname))
                                    print(self.kgrid[tp][sname][c][T][ib][ik])
                                    print(self.kgrid[tp]["X_E_ik"][ib][ik])
                                    print()
                            self.kgrid[tp]["_all_elastic"][c][T][ib][ik] += self.kgrid[tp][sname][c][T][ib][ik]

                        # self.logger.debug("relaxation time at c={} and T= {}: \n {}".format(c, T, self.kgrid[tp]["relaxation time"][c][T][ib]))
                        # self.logger.debug("_all_elastic c={} and T= {}: \n {}".format(c, T, self.kgrid[tp]["_all_elastic"][c][T][ib]))
                        self.kgrid[tp]["relaxation time"][c][T][ib] = 1 / self.kgrid[tp]["_all_elastic"][c][T][ib]



    def map_to_egrid(self, prop_name, c_and_T_idx=True, prop_type="vector"):
        """
        maps a propery from kgrid to egrid conserving the nomenclature.
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

                if not self.gaussian_broadening:
                    for ie, en in enumerate(self.egrid[tp]["energy"]):
                        first_ib = self.kgrid_to_egrid_idx[tp][ie][0][0]
                        first_ik = self.kgrid_to_egrid_idx[tp][ie][0][1]
                        for ib, ik in self.kgrid_to_egrid_idx[tp][ie]:
                            # if norm(self.kgrid[tp][prop_name][ib][ik]) / norm(self.kgrid[tp][prop_name][first_ib][first_ik]) > 1.25 or norm(self.kgrid[tp][prop_name][ib][ik]) / norm(self.kgrid[tp][prop_name][first_ib][first_ik]) < 0.8:
                            #     self.logger.debug('ERROR! Some {} values are more than 25% different at k points with the same energy.'.format(prop_name))
                            #     print('first k: {}, current k: {}'.format(norm(self.kgrid[tp][prop_name][first_ib][first_ik]), norm(self.kgrid[tp][prop_name][ib][ik])))
                            #     print('current energy, first energy, ik, first_ik')
                            #     print(self.kgrid[tp]['energy'][ib][ik], self.kgrid[tp]['energy'][first_ib][first_ik], ik, first_ik)
                            # if self.bs_is_isotropic and prop_type == "vector":
                            if False:
                                self.egrid[tp][prop_name][ie] += norm(self.kgrid[tp][prop_name][ib][ik]) / sq3
                            else:
                                self.egrid[tp][prop_name][ie] += self.kgrid[tp][prop_name][ib][ik]
                        self.egrid[tp][prop_name][ie] /= len(self.kgrid_to_egrid_idx[tp][ie])
                else:
                    raise NotImplementedError(
                        "Guassian Broadening is NOT well tested and abandanded at the begining due to inaccurate results")
        else:
            self.initialize_var("egrid", prop_name, prop_type, initval=self.gs, is_nparray=True, c_T_idx=True)
            for tp in ["n", "p"]:
                if not self.gaussian_broadening:

                    for c in self.dopings:
                        for T in self.temperatures:
                            for ie, en in enumerate(self.egrid[tp]["energy"]):
                                first_ib = self.kgrid_to_egrid_idx[tp][ie][0][0]
                                first_ik = self.kgrid_to_egrid_idx[tp][ie][0][1]
                                for ib, ik in self.kgrid_to_egrid_idx[tp][ie]:
                                    # if self.bs_is_isotropic and prop_type == "vector":
                                    if False:
                                        self.egrid[tp][prop_name][c][T][ie] += norm(
                                            self.kgrid[tp][prop_name][c][T][ib][ik]) / sq3
                                    else:
                                        self.egrid[tp][prop_name][c][T][ie] += self.kgrid[tp][prop_name][c][T][ib][ik]
                                self.egrid[tp][prop_name][c][T][ie] /= len(self.kgrid_to_egrid_idx[tp][ie])

                            # df0dk must be negative but we used norm for df0dk when isotropic
                            # if prop_name in ["df0dk"] and self.bs_is_isotropic:
                            #     self.egrid[tp][prop_name][c][T] *= -1
                else:
                    raise NotImplementedError(
                        "Guassian Broadening is NOT well tested and abandanded at the begining due to inaccurate results")



    def find_fermi_k(self, tolerance=0.001, num_bands = None):
        num_bands = num_bands or self.num_bands
        closest_energy = {c: {T: None for T in self.temperatures} for c in self.dopings}
        self.f0_array = {c: {T: {tp: list(range(num_bands[tp])) for tp in ['n', 'p']} for T in self.temperatures} for c in self.dopings}
        #energy = self.array_from_kgrid('energy', 'n', fill=1000)
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



    def find_fermi(self, c, T, tolerance=0.01, tolerance_loose=0.03,
                   alpha=0.05, max_iter=5000):
        """
        finds the Fermi level at a given c and T at egrid (i.e. DOS)
        Args:
            c (float): The doping concentration;
                c < 0 indicate n-tp (i.e. electrons) and c > 0 for p-tp
            T (float): The temperature.
            tolerance (0<float<1): convergance threshold for relative error
            tolerance_loose (0<float<1): maximum relative error allowed
                between the calculated and input c
            alpha (float < 1): the fraction of the linear interpolation
                towards the actual fermi at each iteration
            max_iter (int): after this many iterations the function returns
                even if it is not converged
        Returns:
            The fitted/calculated Fermi level
        """
        funcs = [lambda E, fermi, T: 1 / (1 + np.exp((E - fermi) / (k_B * T))),
                 lambda E, fermi, T: 1 - 1 / (1 + np.exp((E - fermi) / (k_B * T)))]

        # initialize parameters
        relative_error = self.gl
        niter = 0.0
        temp_doping = {"n": -0.01, "p": +0.01}
        typ = get_tp(c)
        typj = ["n", "p"].index(typ)
        fermi0 = self.cbm_vbm[typ]["energy"] + 0.01 * (-1)**typj # addition is to ensure Fermi is not exactly 0.0
        # calc_doping = (-1) ** (typj + 1) / self.volume / (A_to_m * m_to_cm) ** 3 \
        #         * abs(self.integrate_over_DOSxE_dE(func=funcs[typj], tp=typ,fermi=fermi0, T=T))
        # initiate calc_doping w/o changing the sign of c
        calc_doping = c/5.0
        fermi=fermi0

        dos_emesh = np.array([d[0] for d in self.dos])
        dos_ediff = np.array([self.dos[i + 1][0] - self.dos[i][0] for i, _ in enumerate(self.dos[:-1])] + [0.0])
        dos_dosmesh = np.array([d[1] for d in self.dos])

        def linear_iteration(relative_error, fermi, calc_doping,
                             iterations, niter):
            tune_alpha = 1.0
            while (relative_error > tolerance) and (niter < iterations):
                niter += 1  # to avoid an infinite loop
                if niter / max_iter > 0.5:  # to avoid oscillation we re-adjust alpha at each iteration
                    tune_alpha = 1 - niter / max_iter
                fermi += alpha * tune_alpha * (calc_doping - c) / abs(
                    c + calc_doping) * abs(fermi)
                if abs(
                        fermi) < 1e-5:  # switch sign when getting really close to 0 as otherwise will never converge
                    fermi = fermi * -1

                for j, tp in enumerate(["n", "p"]):
                    idx_s = (1 - j) * self.cbm_dos_idx
                    idx_e = (1 - j) * len(self.dos) + j * self.vbm_dos_idx - 1
                    integral = np.sum(dos_dosmesh[idx_s:idx_e] * funcs[j](
                        dos_emesh[idx_s:idx_e], fermi, T) * dos_ediff[idx_s:idx_e])

                    temp_doping[tp] = (-1) ** (j + 1) * abs(
                        integral / (self.volume * (A_to_m * m_to_cm) ** 3))
                calc_doping = temp_doping["n"] + temp_doping["p"]
                if abs(calc_doping) < 1e-2:
                    calc_doping = np.sign(
                        calc_doping) * 0.01  # just so that calc_doping doesn't get stuck to zero!
                # calculate the relative error from the desired concentration, c
                relative_error = abs(calc_doping - c) / abs(c)
            return relative_error, fermi, calc_doping, niter

        ## start with a simple grid search with maximum 6(2nstep+1) iterations
        step = 0.1
        nstep = 20
        fermi0 = fermi

        print("calculating the fermi level at temperature: {} K".format(T))
        for i in range(15):
            if i > 0:
                nsetps = 10
            for coeff in range(-nstep, nstep+1):
                niter += 1
                fermi = fermi0 + coeff*step
                for j, tp in enumerate(["n", "p"]):
                    idx_s = (1 - j) * self.cbm_dos_idx
                    idx_e = (1 - j) * len(self.dos) + j * self.vbm_dos_idx - 1
                    integral = np.sum(dos_dosmesh[idx_s:idx_e] * funcs[j](
                        dos_emesh[idx_s:idx_e], fermi, T) * dos_ediff[idx_s:idx_e])

                    temp_doping[tp] = (-1) ** (j + 1) * abs(
                        integral / (self.volume * (A_to_m * m_to_cm) ** 3))
                calc_doping = temp_doping["n"] + temp_doping["p"]
                if abs(calc_doping - c) / abs(c) < relative_error:
                    relative_error = abs(calc_doping - c) / abs(c)
                    self.calc_doping[c][T]['n'] = temp_doping["n"]
                    self.calc_doping[c][T]['p'] = temp_doping["p"]
                    if relative_error < tolerance:
                        self.logger.info(
                            "fermi at {} 1/cm3 and {} K after {} iterations: {}".format(
                                c, T, int(niter), fermi))
                        return fermi
                    fermi0 = fermi
            step /= 10

        # adding this just in case the griddy grid search misses the  right fermi
        relative_error, fermi, calc_doping, niter = linear_iteration(
            relative_error, fermi, calc_doping, iterations=max_iter, niter=niter)

        self.logger.info("fermi at {} 1/cm3 and {} K after {} iterations: {}".format(c, T, int(niter), fermi))
        if relative_error > tolerance_loose:
            raise ValueError('The calculated concentration is not within {}%'
                             'of the given value ({}) at T={}'.format(
                                                tolerance_loose*100, c, T))
        return fermi



    def inverse_screening_length(self, c, T):
        """
        Args:
            c (float): the carrier concentration (to get the fermi level)
            T (float): the temperature
        Returns (float): the inverse screening length (beta) in 1/nm units
        """
        beta = {}
        for tp in ["n", "p"]:
            # TODO: the integration may need to be revised. Careful testing of IMP scattering against expt is necessary
            # integral = self.integrate_over_normk(prop_list=["f0","1-f0"], tp=tp, c=c, T=T, xDOS=True)
            # # integral = sum(integral)/3
            # # self.logger.debug('integral_over_norm_k')
            # # self.logger.debug(integral)
            # # from aMoBT ( or basically integrate_over_normk )
            # beta[tp] = (e**2 / (self.epsilon_s * epsilon_0*k_B*T) * integral * 6.241509324e27)**0.5


            #TODO: on 03/27/2018, I reverted this calculations to integrate_over_E from old commit. Did not double-checked the units
            integral = self.integrate_over_E(prop_list=["f0","1 - f0"], tp=tp, c=c, T=T, xDOS=True, weighted=False)
            # integral *= self.nelec
            # beta[tp] = (e**2 / (self.epsilon_s * epsilon_0*k_B*T) * integral * 6.241509324e27)**0.5
            beta[tp] = (e**2 / (self.epsilon_s * epsilon_0 * k_B * T) * integral / self.volume * 1e12 / e) ** 0.5

        return beta



    def to_file(self, path=None, dir_name='run_data', fname='amsetrun',
                force_write=True):
        if not path:
            path = os.path.join(os.getcwd(), dir_name)
            if not os.path.exists(path):
                os.makedirs(name=path)
        else:
            path = os.path.join(path, dir_name)
        if not force_write:
            n = 1
            fname0 = fname
            while os.path.exists(os.path.join(path, '{}.json.gz'.format(fname))):
                warnings.warn('The file, {} exists. AMSET outputs will be '
                        'written in {}'.format(fname, fname0+'_'+str(n)))
                fname = fname0 + '_' + str(n)
                n += 1

        # make the output dict
        out_d = {'kgrid0': self.kgrid0, 'egrid0': self.egrid0, 'cbm_vbm': self.cbm_vbm,
                 'mobility': self.mobility, 'epsilon_s': self.epsilon_s,
                 'elastic_scatterings': self.elastic_scatterings,
                 'inelastic_scatterings': self.inelastic_scatterings,
                 'Efrequency0': self.Efrequency0,
                 'dopings': self.dopings, 'temperatures': self.temperatures}

        # write the output dict to file
        with gzip.GzipFile(os.path.join(path, '{}.json.gz'.format(fname)),
                           mode='w') as fp:
            json_str = json.dumps(out_d, cls=MontyEncoder)
            json_bytes = json_str.encode('utf-8')
            fp.write(json_bytes)


    @staticmethod
    def from_file(path=None, dir_name="run_data", filename="amsetrun.json.gz"):
        #TODO: make this better, maybe organize these class attributes a bit?
        if not path:
            path = os.path.join(os.getcwd(), dir_name)

        with gzip.GzipFile(os.path.join(path, filename), mode='r') as fp:
            d = json.load(fp, cls=MontyDecoder)
        amset = AMSET(calc_dir=path, material_params={'epsilon_s': d['epsilon_s']})
        amset.kgrid0 = d['kgrid0']
        amset.egrid0 = d['egrid0']
        amset.mobility = d['mobility']
        amset.elastic_scatterings = d['elastic_scatterings']
        amset.inelastic_scatterings = d['inelastic_scatterings']
        amset.cbm_vbm = d['cbm_vbm']
        amset.Efrequency0 = d['Efrequency0']
        amset.dopings = [float(dope) for dope in d['dopings']]
        amset.temperatures = [float(T) for T in d['temperatures']]
        amset.all_types = list(set([get_tp(c) for c in amset.dopings]))
        return amset


    def to_json(self, kgrid=True, trimmed=False, max_ndata=None, nstart=0,
                valleys=True, path=None, dir_name="run_data"):
        """
        writes the kgrid and egird to json files
        Args:
            kgrid (bool): whether to also write kgrid to kgrid.json
            trimmed (bool): if trimmed some properties (dict keys) will be
                removed to save space
            max_ndata (int): the maximum index from the CBM/VBM written to file
            nstart (int): the initial list index of a property written to file
        Returns: egrid.json and (optional) kgrid.json file(s)
        """
        if not path:
            path = os.path.join(os.getcwd(), dir_name)
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
            start_time = time.time()
            kgrid = deepcopy(self.kgrid)
            print("time to copy kgrid = {} seconds".format(time.time() - start_time))
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
                json.dump(self.valleys, fp, sort_keys=True, indent=4, ensure_ascii=False, cls=MontyEncoder)


    def solve_BTE_iteratively(self):
        # calculating S_o scattering rate which is not a function of g
        if "POP" in self.inelastic_scatterings and not self.bs_is_isotropic:
            for g_suffix in ["", "_th"]:
                self.s_inelastic(sname="S_o" + g_suffix, g_suffix=g_suffix)

        # solve BTE to calculate S_i scattering rate and perturbation (g) in an iterative manner
        for iter in range(self.BTE_iters):
            print("Performing iteration # {}".format(iter))

            if "POP" in self.inelastic_scatterings:
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
                            # the following 5 lines are a hacky and dirty fix to the problem that the last (largest norm(k) of the opposite type has very large value and messes up mobility_POP
                            # means = np.mean(self.kgrid[tp]["g_POP"][c][T][ib], axis=1)
                            # g_POP_median = np.median(means)
                            # for igpop in range(len(means)):
                            #     if means[igpop] > 1e10 * g_POP_median:
                            #         self.kgrid[tp]["g_POP"][c][T][ib][igpop]= 0

                            self.kgrid[tp]["g"][c][T][ib] = (self.kgrid[tp]["S_i"][c][T][ib] +
                                                             self.kgrid[tp]["electric force"][c][
                                                                 T][ib]) / (self.kgrid[tp]["S_o"][c][T][ib] +
                                                                            self.kgrid[tp]["_all_elastic"][c][T][ib])

                            self.kgrid[tp]["g_th"][c][T][ib] = (self.kgrid[tp]["S_i_th"][c][T][ib] +
                                                                self.kgrid[tp]["thermal force"][c][
                                                                    T][ib]) / (self.kgrid[tp]["S_o_th"][c][T][ib] +
                                                                               self.kgrid[tp]["_all_elastic"][c][T][ib])

                            # TODO: correct these lines to reflect that f = f0 + x*g
                            self.kgrid[tp]["f"][c][T][ib] = self.kgrid[tp]["f0"][c][T][ib] + self.kgrid[tp]["g"][c][T][
                                ib]
                            self.kgrid[tp]["f_th"][c][T][ib] = self.kgrid[tp]["f0"][c][T][ib] + \
                                                               self.kgrid[tp]["g_th"][c][T][ib]

                            # for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                            #     if norm(self.kgrid[tp]["g_POP"][c][T][ib][ik]) > 1 and iter > 0:
                            #         # because only when there are no S_o/S_i scattering events, g_POP>>1 while it should be zero
                            #         self.kgrid[tp]["g_POP"][c][T][ib][ik] = [self.gs, self.gs, self.gs]

                        avg_g_diff = np.mean(
                            [abs(g_old[ik] - self.kgrid[tp]["g"][c][T][0][ik]) for ik in range(len(g_old))])
                        print("Average difference in {}-type g term at c={} and T={}: {}".format(tp, c, T, avg_g_diff))

        for prop in ["electric force", "thermal force", "g", "g_POP", "g_th", "S_i", "S_o", "S_i_th", "S_o_th"]:
            self.map_to_egrid(prop_name=prop, c_and_T_idx=True)

        # this code has been commented out because egrid is no longer in use, but it might still be necessary in kgrid
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

    # def calc_v_vec(self, tp):
    #     # TODO: Take into account the fact that this gradient is found in three directions specified by the lattice, not
    #     # the x, y, and z directions. It must be corrected to account for this.
    #     energy_grid = self.array_from_kgrid('energy', tp)
    #     # print('energy:')
    #     # np.set_printoptions(precision=3)
    #     # print(energy_grid[0,:,:,:,0])
    #     N = self.kgrid_array[tp].shape
    #     k_grid = self.kgrid_array[tp]
    #     v_vec_result = []
    #     for ib in range(self.num_bands[tp]):
    #         v_vec = np.gradient(energy_grid[ib][:,:,:,0], k_grid[:,0,0,0] * self._rec_lattice.a, k_grid[0,:,0,1] * self._rec_lattice.b, k_grid[0,0,:,2] * self._rec_lattice.c)
    #         v_vec_rearranged = np.zeros((N[0], N[1], N[2], 3))
    #         for i in range(N[0]):
    #             for j in range(N[1]):
    #                 for k in range(N[2]):
    #                     v_vec_rearranged[i,j,k,:] = np.array([v_vec[0][i,j,k], v_vec[1][i,j,k], v_vec[2][i,j,k]])
    #         v_vec_rearranged *= A_to_m * m_to_cm / hbar
    #         v_vec_result.append(v_vec_rearranged)
    #     return np.array(v_vec_result)


    # turns a kgrid property into a list of grid arrays of that property for k integration
    def array_from_kgrid(self, prop_name, tp, c=None, T=None, denom=False, none_missing=False, fill=None):
        if c:
            return np.array([self.grid_from_energy_list(self.kgrid[tp][prop_name][c][T][ib], tp, ib, denom=denom, none_missing=none_missing, fill=fill) for ib in range(self.num_bands[tp])])
        else:
            return np.array([self.grid_from_energy_list(self.kgrid[tp][prop_name][ib], tp, ib, denom=denom, none_missing=none_missing, fill=fill) for ib in range(self.num_bands[tp])])


    # takes a list that is sorted by energy and missing removed points
    def grid_from_energy_list(self, prop_list, tp, ib, denom=False, none_missing=False, fill=None):

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
            #for ib in range(self.num_bands[tp]):
            for ik in self.rm_idx_list[tp][ib]:
                adjusted_prop_list.insert(ik, fill) if not insert_list else adjusted_prop_list.insert(ik, [fill,fill,fill])

        # step 2 is reorder based on first sort
        adjusted_prop_list = np.array(adjusted_prop_list)[self.pos_idx[tp]]
        # then call grid_from_ordered_list
        return self.grid_from_ordered_list(adjusted_prop_list, tp, denom=denom, none_missing=True)


    # return a grid of the (x,y,z) k points in the proper grid
    def grid_from_ordered_list(self, prop_list, tp, denom=False, none_missing=False, scalar=False):
        # need:
        # self.kgrid_array[tp]
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


    # takes list or array of array grids
    def integrate_over_states(self, integrand_grid, tp='all'):

        integrand_grid = np.array(integrand_grid)

        if type(integrand_grid[0][0,0,0]) == list or type(integrand_grid[0][0,0,0]) == np.ndarray:
            result = np.zeros(3)
        else:
            result = 0
        num_bands = integrand_grid.shape[0]

        for ib in range(num_bands):
            result += self.integrate_over_k(integrand_grid[ib], tp)

        # if tp == 'n':
        #     for ib in range(self.num_bands['n']):
        #         result += self.integrate_over_k(integrand_grid[ib], tp)
        # if tp  == 'p':
        #     for ib in range(self.num_bands['p']):
        #         result += self.integrate_over_k(integrand_grid[ib], tp)
        # if tp == 'all':
        #     for ib in range(self.num_bands['n']):
        #         result += self.integrate_over_k(integrand_grid[ib], 'n')
        #     for ib in range(self.num_bands['p']):
        #         result += self.integrate_over_k(integrand_grid[ib + self.num_bands['n']], tp)
        return result


    # calculates transport properties for isotropic materials
    def calculate_transport_properties_with_k(self, test_anisotropic, important_points):
        # calculate mobility by averaging velocity per electric field strength
        mu_num = {tp: {el_mech: {c: {T: [0, 0, 0] for T in self.temperatures} for c in self.dopings} for el_mech in self.elastic_scatterings} for tp in ["n", "p"]}
        mu_denom = deepcopy(mu_num)
        valley_mobility = {tp: {el_mech: {c: {T: np.array([0., 0., 0.]) for T in self.temperatures} for c in
                  self.dopings} for el_mech in self.mo_labels+self.spb_labels} for tp in ["n", "p"]}

        #k_hat = np.array([self.k_hat_array[tp] for ib in range(self.num_bands)])

        for c in self.dopings:
            for T in self.temperatures:
                for j, tp in enumerate(["n", "p"]):
                    if not self.count_mobility[self.ibrun][tp]:
                        continue
                    N = self.kgrid_array[tp].shape
                    # if tp == 'n':
                    #     print(self.dv_grid['n'][(N[0] - 1) / 2, (N[1] - 1) / 2, :])

                    print('tp =  ' + tp + ':')
                    # get quantities that are independent of mechanism
                    num_k = [len(self.kgrid[tp]["energy"][ib]) for ib in range(self.num_bands[tp])]
                    df0dk = self.array_from_kgrid('df0dk', tp, c, T)
                    v = self.array_from_kgrid('velocity', tp)
                    # norm_v = np.array([self.grid_from_energy_list([norm(self.kgrid[tp]["velocity"][ib][ik]) / sq3 for ik in
                    #                                       range(num_k[ib])], tp, ib) for ib in range(self.num_bands[tp])])
                    # norm_v = np.array([self.grid_from_energy_list([self.kgrid[tp]["norm(v)"][ib][ik] for ik in
                    #                                       range(num_k[ib])], tp, ib) for ib in range(self.num_bands[tp])])
                    norm_v = v # 20180307: AF: I added this since we want to see the mobility in 3 main directions, in isotropic material it's the same as norm(v)/sq3
                    #norm_v = grid_norm(v)
                    f0_removed = self.array_from_kgrid('f0', tp, c, T)

                    # for ib in range(self.num_bands[tp]):
                    #     print('energy (type {}, band {}):'.format(tp, ib))
                    #     print(self.energy_array[tp][ib][(N[0] - 1) / 2, (N[1] - 1) / 2, :])
                    f0_all = 1 / (np.exp((self.energy_array[tp] - self.fermi_level[c][T]) / (k_B * T)) + 1)

                    np.set_printoptions(precision=3)
                    # print('v:')
                    # print(v[0,:3,:3,:3,:])
                    # print('df0dk:')
                    # print(df0dk[0,4,:,:,0])
                    # print(df0dk[0, 4, :, :, 1])
                    # print(df0dk[0, 4, :, :, 2])
                    # print('electric force:')
                    # print(self.kgrid[tp]["electric force"][c][T][0])

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
                            print('norm_v')
                            print(norm_v[0, (N[0] - 1) / 2, (N[1] - 1) / 2, :])
                            print('g*norm_v')
                            print((g*norm_v)[0, (N[0] - 1) / 2, (N[1] - 1) / 2, :])
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
                        # np.sum(np.exp(-(aa * 7.5) * k_norm) * k_hat_cartesian[0, :, :, :, 2] ** 2 * (aa * 7.5) ** 3 * self.dv_grid['n'][:, :, :], axis=(0, 1, 2))

                        # from equation 44 in Rode, elastic
                        #el_mech stands for elastic mechanism
                        # for el_mech in self.elastic_scatterings:
                        #     nu_el = self.array_from_kgrid(el_mech, tp, c, T, denom=True)
                        #     # includes e in numerator because hbar is in eV units, where e = 1
                        #     numerator = -self.integrate_over_states(v * self.k_hat_array[tp] * df0dk / nu_el, tp)
                        #     denominator = self.integrate_over_states(f0, tp) * hbar
                        #     # for ib in range(len(self.kgrid[tp]["energy"])):
                        #     #     #num_kpts = len(self.kgrid[tp]["energy"][ib])
                        #     #     # integrate numerator / norm(F) of equation 44 in Rode
                        #     #     for dim in range(3):
                        #     #         # TODO: add in f0 to the integral so that this works for anisotropic materials
                        #     #         mu_num[tp][el_mech][c][T][dim] += self.integrate_over_k(v_vec[ib] * k_hat[ib] * df0dk[ib] / nu_el[ib])[dim]
                        #     #         mu_denom[tp][el_mech][c][T][dim] += self.integrate_over_k(f0[ib])[dim]
                        #
                        #     # should be -e / hbar but hbar already in eV units, where e=1
                        #     self.mobility[tp][el_mech][c][T] = numerator / denominator
                        #
                        # # from equation 44 in Rode, inelastic
                        # for inel_mech in self.inelastic_scatterings:
                        #     nu_el = self.array_from_kgrid('_all_elastic', tp, c, T, denom=True)
                        #     S_i = 0
                        #     S_o = 1
                        #     self.mobility[tp][inel_mech][c][T] = self.integrate_over_states(
                        #         v * self.k_hat_array[tp] * (-1 / hbar) * df0dk / S_o, tp)

                    if self.bs_is_isotropic and not test_anisotropic:
                        if tp == get_tp(c):
                            self.logger.info('calculating mobility by integrating over'
                                         ' k-grid and isotropic BS assumption...')
                            self.logger.debug('current valley is at {}'.format(important_points))
                            self.logger.debug('the denominator is:\n{}'.format(self.denominator))
                        # from equation 45 in Rode, elastic mechanisms
                        # for ib in range(self.num_bands[tp]):
                        #     self.logger.info('f0 (type {}, band {}):'.format(tp, ib))
                        #     self.logger.info(f0_all[ib, (N[0]-1)/2, (N[1]-1)/2, :])
                        #     #self.logger.info(self.f0_array[c][T][tp][ib][(N[0]-1)/2, (N[1]-1)/2, :])

                        for el_mech in self.elastic_scatterings:
                            nu_el = self.array_from_kgrid(el_mech, tp, c, T, denom=True)
                            # this line should have -e / hbar except that hbar is in units of eV*s so in those units e=1
                            g = -1 / hbar * df0dk / nu_el
                            # print('g*norm(v) for {}:'.format(el_mech))
                            # print((g * norm_v)[0, (N[0]-1)/2, (N[1]-1)/2, :])
                            # valley_mobility[tp][el_mech][c][T] = self.integrate_over_states(g * norm_v, tp) / self.denominator[c][T][tp] * 1 #self.bs.get_kpoint_degeneracy(important_points[tp][0])
                            valley_mobility[tp][el_mech][c][T] = self.integrate_over_states(g * norm_v, tp)
             # from equation 45 in Rode, inelastic mechanisms
                        for inel_mech in self.inelastic_scatterings:
                            g = self.array_from_kgrid("g_"+inel_mech, tp, c, T)
                            # valley_mobility[tp][inel_mech][c][T] = self.integrate_over_states(g * norm_v, tp) / self.denominator[c][T][tp] * 1 #self.bs.get_kpoint_degeneracy(important_points[tp][0])
                            valley_mobility[tp][inel_mech][c][T] = self.integrate_over_states(g * norm_v, tp)

                        # from equation 45 in Rode, overall
                        g = self.array_from_kgrid("g", tp, c, T)
                        # for ib in range(self.num_bands[tp]):
                        #     self.logger.info('g for overall (type {}, band {}):'.format(tp, ib))
                        #     self.logger.info(g[ib, (N[0] - 1) / 2, (N[1] - 1) / 2, :])
                        #     self.logger.info('norm(v) for overall (type {}, band {}):'.format(tp, ib))
                        #     self.logger.info(norm_v[ib, (N[0] - 1) / 2, (N[1] - 1) / 2, :])
                        #     self.logger.info('g*norm(v) for overall (type {}, band {}):'.format(tp, ib))
                        #     self.logger.info((g * norm_v)[ib, (N[0]-1)/2, (N[1]-1)/2, :])
                        # valley_mobility[tp]['overall'][c][T] = self.integrate_over_states(g * norm_v, tp) / self.denominator[c][T][tp] * 1 # self.bs.get_kpoint_degeneracy(important_points[tp][0])
                        valley_mobility[tp]['overall'][c][T] = self.integrate_over_states(g * norm_v, tp)


                    # figure out average mobility
                    faulty_overall_mobility = False
                    mu_overrall_norm = norm(valley_mobility[tp]['overall'][c][T])
                    mu_average = np.array([0.0, 0.0, 0.0])
                    for transport in self.elastic_scatterings + self.inelastic_scatterings:
                        # averaging all mobility values via Matthiessen's rule
                        mu_average += 1 / (np.array(valley_mobility[tp][transport][c][T]) + 1e-32)
                        if mu_overrall_norm > norm(valley_mobility[tp][transport][c][T]):
                            faulty_overall_mobility = True  # because the overall mobility should be lower than all
                        valley_mobility[tp]["average"][c][T] = 1 / mu_average

                    # Decide if the overall mobility make sense or it should be equal to average (e.g. when POP is off)
                    if (mu_overrall_norm == 0.0 or faulty_overall_mobility) and not test_anisotropic:
                        self.logger.warning('There may be a problem with overall '
                                        'mobility; setting it to average...')

# 20180305 backup: didn't make sense
#                         valley_mobility[tp]['overall'][c][T] += 1 / mu_average
#                     # else:
#                     #     valley_mobility[tp]['overall'][c][T] += mu_overall

# 20180305 update:
                        valley_mobility[tp]['overall'][c][T] = valley_mobility[tp]["average"][c][T]
                    # else:
                    #     valley_mobility[tp]['overall'][c][T] += mu_overall



                    if self.independent_valleys:
                        for mu in self.mo_labels:
                            valley_mobility[tp][mu][c][T] /= self.denominator[c][T][tp]


                    # print('new {}-type overall mobility at T = {}: {}'.format(tp, T, self.mobility[tp]['overall'][c][T]))
                    # for el_mech in self.elastic_scatterings + self.inelastic_scatterings:
                    #     print('new {}-type {} mobility at T = {}: {}'.format(tp, el_mech, T, self.mobility[tp][el_mech][c][T])
        return valley_mobility


    def calculate_transport_properties_with_E(self, important_points):
        """
        Mobility and Seebeck coefficient are calculated by integrating the
            the perturbation to electron distribution as well as group velocity
            over the energy
        """
        integrate_over_kgrid = False
        valley_mobility = {tp: {el_mech: {c: {T: np.array([0., 0., 0.]) for T in self.temperatures} for c in
                  self.dopings} for el_mech in self.mo_labels+self.spb_labels} for tp in ["n", "p"]}

        for c in self.dopings:
            for T in self.temperatures:
                for j, tp in enumerate(["p", "n"]):
                    # if integrate_over_kgrid:
                    #     if tp == "n":
                    #         denom = self.integrate_over_BZ(["f0"], tp, c, T, xDOS=False, xvel=False,
                    #                                    weighted=False)*3*default_small_E #* 1e-7 * 1e-3 * self.volume
                    #         print('old denominator = ' + str(denom))
                    #     else:
                    #         denom = self.integrate_over_BZ(["1 - f0"], tp, c, T, xDOS=False, xvel=False,
                    #                                        weighted=False)*3*default_small_E
                    # else:
                    #     if tp == "n":
                    #         denom = self.integrate_over_E(prop_list=["f0"], tp=tp, c=c, T=T, xDOS=False, xvel=False,
                    #                                   weighted=False)*3*default_small_E
                    #     else:
                    #         denom = self.integrate_over_E(prop_list=["1 - f0"], tp=tp, c=c, T=T, xDOS=False, xvel=False,
                    #                                       weighted=False)*3*default_small_E
                    # # self.logger.debug("denominator {}-type valley {}: \n{}".format(
                    # #         tp, important_points[tp], denom))

                    denom = None

                    # mobility numerators
                    for mu_el in self.elastic_scatterings:
                        if integrate_over_kgrid:
                            # self.egrid[tp]["mobility"][mu_el][c][T] = \
                            #         (-1) * default_small_E / hbar * \
                            #         self.integrate_over_BZ(prop_list=[
                            #         "/" + mu_el, "df0dk"], tp=tp, c=c,T=T,
                            #             xDOS=False, xvel=True, weighted=True) #* 1e-7 * 1e-3 * self.volume
                            valley_mobility[tp][mu_el][c][T] = (-1) * default_small_E / hbar * \
                                    self.integrate_over_BZ(prop_list=[
                                    "/" + mu_el, "df0dk"], tp=tp, c=c,T=T,
                                        xDOS=False, xvel=True, weighted=True) #* 1e-7 * 1e-3 * self.volume

                        else:
                            # self.egrid[tp]["mobility"][mu_el][c][T] = \
                            #         (-1) * default_small_E / hbar * \
                            #         self.integrate_over_E(prop_list=[
                            #         "/" + mu_el, "df0dk"], tp=tp, c=c, T=T,
                            #             xDOS=False, xvel=True, weighted=True)
                            # if tp == "n":
                            #     print('old {} numerator = {}'.format(mu_el, self.egrid[tp]["mobility"][mu_el][c][T]))
                            valley_mobility[tp][mu_el][c][T] = (-1) * default_small_E / hbar * \
                                    self.integrate_over_E(prop_list=[
                                    "/" + mu_el, "df0dk"], tp=tp, c=c, T=T,
                                        xDOS=False, xvel=True, weighted=True)

                    # if integrate_over_kgrid:
                    #     for mu_inel in self.inelastic_scatterings:
                    #         self.egrid[tp]["mobility"][mu_inel][c][T] = \
                    #                 self.integrate_over_BZ(prop_list=[
                    #                 "g_" + mu_inel], tp=tp, c=c, T=T,
                    #                     xDOS=False, xvel=True, weighted=True)
                    #
                    #     self.egrid[tp]["mobility"]["overall"][c][T] = \
                    #             self.integrate_over_BZ(["g"], tp, c, T,
                    #             xDOS=False,xvel=True,weighted=True)
                    #
                    #     print("overll numerator")
                    #     print(self.egrid[tp]["mobility"]["overall"][c][T])
                    # else:
                    if True:
                        for mu_inel in self.inelastic_scatterings:
                            # calculate mobility["POP"] based on g_POP
                            # self.egrid[tp]["mobility"][mu_inel][c][T] = \
                            #         self.integrate_over_E(prop_list=[
                            #         "g_" + mu_inel], tp=tp, c=c, T=T,
                            #             xDOS=False, xvel=True, weighted=True)
                            valley_mobility[tp][mu_inel][c][T] = self.integrate_over_E(prop_list=[
                                    "g_" + mu_inel], tp=tp, c=c, T=T,
                                        xDOS=False, xvel=True, weighted=True)
                        mu_overall_valley = self.integrate_over_E(prop_list=["g"],
                               tp=tp, c=c, T=T, xDOS=False, xvel=True, weighted=True)

                    self.egrid[tp]["J_th"][c][T] = (self.integrate_over_E(prop_list=["g_th"], tp=tp, c=c, T=T,
                            xDOS=False, xvel=True, weighted=True)) * e * abs(c)  # in units of A/cm2

                    # for transport in self.elastic_scatterings + self.inelastic_scatterings + ["overall"]:
                    #     # self.egrid[tp]["mobility"][transport][c][T] /= 3 * default_small_E
                    #     self.mobility[tp][transport][c][T] /= (3*default_small_E)

                    # The following did NOT work as J_th only has one integral (see aMoBT but that one is over k)
                    # and with that one the units don't work out and if you use two integral, J_th will be of 1e6 order!
                    # self.egrid[tp]["J_th"][c][T] = self.integrate_over_E(prop_list=["g_th"], tp=tp, c=c, T=T,
                    #         xDOS=False, xvel=True, weighted=True) * e * 1e24  # to bring J to A/cm2 units
                    # self.egrid[tp]["J_th"][c][T] /= 3*self.volume*self.integrate_over_E(prop_list=["f0"], tp=tp, c=c,
                    #         T=T, xDOS=False, xvel=False, weighted=True)

                    # other semi-empirical mobility values:
                    #fermi = self.egrid["fermi"][c][T]
                    # fermi = self.fermi_level[c][T]
                    # # fermi_SPB = self.egrid["fermi_SPB"][c][T]
                    # energy = self.cbm_vbm[get_tp(c)]["energy"]
                    #
                    # # for mu in ["overall", "average"] + self.inelastic_scatterings + self.elastic_scatterings:
                    # #     self.egrid[tp]["mobility"][mu][c][T] /= 3.0
                    #
                    # # ACD mobility based on single parabolic band extracted from Thermoelectric Nanomaterials,
                    # # chapter 1, page 12: "Material Design Considerations Based on Thermoelectric Quality Factor"
                    # self.mobility[tp]["SPB_ACD"][c][T] = 2 ** 0.5 * pi * hbar ** 4 * e * self.C_el * 1e9 / (
                    # # C_el in GPa
                    #     3 * (self.cbm_vbm[tp]["eff_mass_xx"] * m_e) ** 2.5 * (k_B * T) ** 1.5 * self.E_D[tp] ** 2) \
                    #                                               * fermi_integral(0, fermi, T, energy, wordy=False) \
                    #                                               / fermi_integral(0.5, fermi, T, energy,
                    #                                                                wordy=False) * e ** 0.5 * 1e4  # to cm2/V.s

                    faulty_overall_mobility = False
                    # mu_overrall_norm = norm(self.egrid[tp]["mobility"]["overall"][c][T])
                    # mu_overall_norm = norm(self.mobility[tp]['overall'][c][T])
                    temp_avg = np.array([0.0, 0.0, 0.0])
                    for transport in self.elastic_scatterings + self.inelastic_scatterings:
                        # averaging all mobility values via Matthiessen's rule
                        # self.egrid[tp]["mobility"]["average"][c][T] += 1 / self.egrid[tp]["mobility"][transport][c][T]
                        temp_avg += 1/ valley_mobility[tp][transport][c][T]
                        # if mu_overrall_norm > norm(self.egrid[tp]["mobility"][transport][c][T]):
                        if norm(mu_overall_valley) > norm(valley_mobility[tp][transport][c][T]):
                            faulty_overall_mobility = True  # because the overall mobility should be lower than all
                    # self.egrid[tp]["mobility"]["average"][c][T] = 1 / self.egrid[tp]["mobility"]["average"][c][T]
                    valley_mobility[tp]['average'][c][T] = 1 / temp_avg
                    # Decide if the overall mobility make sense or it should be equal to average (e.g. when POP is off)

                    if norm(mu_overall_valley) == 0.0 or faulty_overall_mobility:
                        # self.egrid[tp]["mobility"]["overall"][c][T] = self.egrid[tp]["mobility"]["average"][c][T]
                        valley_mobility[tp]['overall'][c][T] = valley_mobility[tp]['average'][c][T]
                    else:
                        valley_mobility[tp]["overall"][c][T] = mu_overall_valley

                    if self.independent_valleys:
                        for mu in self.mo_labels:
                            valley_mobility[tp][mu][c][T] /= self.denominator[c][T][tp]

                    # for mu in self.mo_labels + self.spb_labels:
                    #     if self.count_mobility[self.ibrun][tp]:
                    #         self.mobility[tp][mu][c][T] += valley_mobility[tp][mu][c][T]

                    self.egrid[tp]["relaxation time constant"][c][T] = self.mobility[tp]["overall"][c][T] \
                            * 1e-4 * m_e * self.cbm_vbm[tp]["eff_mass_xx"] / e  # 1e-4 to convert cm2/V.s to m2/V.s

                    # calculating other overall transport properties:
                    # self.egrid[tp]["conductivity"][c][T] = self.egrid[tp]["mobility"]["overall"][c][T] * e * abs(c)
                    self.egrid[tp]["conductivity"][c][T] = self.mobility[tp]["overall"][c][T] * e * abs(c)
                    # self.egrid["seebeck"][c][T][tp] = -1e6 * k_B * (self.egrid["Seebeck_integral_numerator"][c][T][tp] \
                    #                                                 / self.egrid["Seebeck_integral_denominator"][c][T][
                    #                                                     tp] - (
                    #                                                 self.egrid["fermi"][c][T] - self.cbm_vbm[tp][
                    #                                                     "energy"]) / (k_B * T))

                    # TODO: to calculate Seebeck define a separate function after ALL important_points are exhausted and the overall sum of self.mobility is evaluated!
                    self.egrid[tp]["seebeck"][c][T] = -1e6 * k_B * (
                            self.egrid["Seebeck_integral_numerator"][c][T][tp]\
                            /self.egrid["Seebeck_integral_denominator"][c][T][
                            tp]-(self.fermi_level[c][T] - self.cbm_vbm[tp][
                                                    "energy"]) / (k_B * T))
                    self.egrid[tp]["TE_power_factor"][c][T] = \
                            self.egrid[tp]["seebeck"][c][T]** 2 * self.egrid[
                                tp]["conductivity"][c][T] / 1e6  # in uW/cm2K

                    # when POP is not available J_th is unreliable
                    if "POP" in self.inelastic_scatterings:
                        self.egrid[tp]["seebeck"][c][T] = np.array([self.egrid[
                                tp]["seebeck"][c][T] for i in range(3)])
                        self.egrid[tp]["seebeck"][c][T] += 0.0
                        # TODO: for now, we ignore the following until we figure out the units see why values are high!
                        # self.egrid["seebeck"][c][T][tp] += 1e6 \
                        #                 * self.egrid["J_th"][c][T][tp]/self.egrid["conductivity"][c][T][tp]/dTdz

                    # print("3 {}-seebeck terms at c={} and T={}:".format(tp, c, T))
                    # print(self.egrid["Seebeck_integral_numerator"][c][T][tp] \
                    #       / self.egrid["Seebeck_integral_denominator"][c][T][tp] * -1e6 * k_B)
                    # print((self.fermi_level[c][T] - self.cbm_vbm[tp]["energy"]) * 1e6 * k_B / (k_B * T))
                    # print(self.egrid[tp]["J_th"][c][T] / self.egrid[tp]["conductivity"][c][T]/ dTdz * 1e6)


                    #TODO: not sure about the following part yet specially as sometimes due to position of fermi I get very off other type mobility values! (sometimes very large)
                    other_type = ["p", "n"][1 - j]
                    self.egrid[tp]["seebeck"][c][T] = (
                            self.egrid[tp]["conductivity"][c][T] * \
                            self.egrid[tp]["seebeck"][c][T] -
                            self.egrid[other_type]["conductivity"][c][T] * \
                            self.egrid[other_type]["seebeck"][c][T]) / (
                            self.egrid[tp]["conductivity"][c][T] +
                            self.egrid[other_type]["conductivity"][c][T])
                    ## since sigma = c_e x e x mobility_e + c_h x e x mobility_h:
                    ## self.egrid["conductivity"][c][T][tp] += self.egrid["conductivity"][c][T][other_type]
        return valley_mobility


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



    def create_plots(self, x_label, y_label, show_interactive, save_format, c, tp, file_suffix,
                     textsize, ticksize, path, margins, fontfamily, plot_data, names=None, labels=None,
                     x_label_short='', y_label_short=None, mode='markers', y_axis_type='linear', plot_title=None):
        from matminer.figrecipes.plot import PlotlyFig
        if not plot_title:
            plot_title = '{} for {}, c={}'.format(y_label, self.tp_title[tp], c)
        if not y_label_short:
            y_label_short = y_label
        if show_interactive:
            if not x_label_short:
                filename = os.path.join(path, "{}_{}.{}".format(y_label_short, file_suffix, 'html'))
            else:
                filename = os.path.join(path, "{}_{}_{}.{}".format(y_label_short, x_label_short, file_suffix, 'html'))
            pf = PlotlyFig(x_title=x_label, y_title=y_label, y_scale=y_axis_type,
                            title=plot_title, fontsize=textsize,
                           mode='offline', filename=filename, ticksize=ticksize,
                            margins=margins, fontfamily=fontfamily)
            # if all_plots:plt.xy_plot(x_col=[], y_col=[], add_xy_plot=all_plots, y_axis_type=y_axis_type, color='black', showlegend=True)
            #
            # else:
            #     plt.xy((x_data, y_data), colors='black')

            pf.xy(plot_data, names=names, labels=labels, modes=mode)
        if save_format is not None:
            if not x_label_short:
                filename = os.path.join(path, "{}_{}.{}".format(y_label_short, file_suffix, save_format))
            else:
                filename = os.path.join(path, "{}_{}_{}.{}".format(y_label_short, x_label_short, file_suffix, save_format))
            pf = PlotlyFig(x_title=x_label, y_title=y_label,
                            title=plot_title, fontsize=textsize,
                            mode='static', filename=filename, ticksize=ticksize,
                            margins=margins, fontfamily=fontfamily)
            pf.xy(plot_data, names=names, labels=labels, modes=mode)
            # if all_plots:
            #     plt.xy(x_col=[], y_col=[], add_xy_plot=all_plots, y_axis_type=y_axis_type, color='black', showlegend=True)
            # else:
            #     plt.xy_plot(x_col=x_data, y_col=y_data, y_axis_type=y_axis_type, color='black')



    def plot(self, k_plots=[], E_plots=[], mobility=True, concentrations='all', carrier_types=['n', 'p'],
             direction=['avg'], show_interactive=True, save_format=None, textsize=30, ticksize=25, path=None,
             margins=100, fontfamily="serif"):
        """
        plots the given k_plots and E_plots properties.
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
            textsize: (int) size of title and axis label text
            ticksize: (int) size of axis tick label text
            path: (string) location to save plots
            margins: (int) figrecipes plotly margins
            fontfamily: (string) plotly font
        """
        supported_k_plots = ['energy', 'df0dk', 'velocity'] + self.elastic_scatterings
        supported_E_plots = ['frequency', 'relaxation time', 'df0dk', 'velocity'] + self.elastic_scatterings
        if "POP" in self.inelastic_scatterings:
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

        mu_list = ["overall", "average"] + self.elastic_scatterings + self.inelastic_scatterings
        mu_markers = {mu: i for i, mu in enumerate(mu_list)}
        temp_markers = {T: i for i,T in enumerate(self.temperatures)}

        if not path:
            path = os.path.join(os.getcwd(), "plots")
            if not os.path.exists(path):
                os.makedirs(name=path)

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
                            plot_title = None
                            if y_value == 'frequency':
                                plot_title = 'Energy Histogram for {}, c={}'.format(self.tp_title[tp], c)
                            self.create_plots(x_axis_label[x_value], y_value, show_interactive, save_format, c, tp, tp_c,
                                              textsize, ticksize, path, margins, fontfamily, plot_data=[(x_data[x_value], y_data_temp_independent[x_value][y_value])],
                                              x_label_short=x_value, plot_title=plot_title)


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
                                self.create_plots(x_axis_label[x_value], y_value, show_interactive,
                                                  save_format, c, tp, tp_c_dir,
                                                  textsize, ticksize, path, margins, fontfamily, plot_data=(x_data[x_value], y_data_temp_independent[x_value][y_value]), x_label_short=x_value)

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
                            self.create_plots(x_axis_label[x_value], y_value, show_interactive,
                                              save_format, c, tp, tp_c_dir,
                                              textsize, ticksize, path, margins, fontfamily, plot_data=plot_data,
                                              x_label_short=x_value, names=names)

                    # mobility plots as a function of temperature (the only plot that does not have k or E on the x axis)
                    if mobility:
                        # all_plots = []
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

                        self.create_plots("Temperature (K)",
                                "Mobility (cm2/V.s)", show_interactive,
                                save_format, c, tp, tp_c_dir, textsize-5,
                                ticksize-5, path, margins,
                                fontfamily, plot_data=plot_data, names=names, mode='lines+markers',
                                y_label_short="mobility", y_axis_type='log')



    def to_csv(self, path=None, dir_name="run_data", csv_filename='amset_results.csv'):
        """
        writes the calculated transport properties to a csv file.
        Args:
            csv_filename (str):
        Returns (.csv file)
        """
        import csv
        if not path:
            path = os.path.join(os.getcwd(), dir_name)
            if not os.path.exists(path):
                os.makedirs(name=path)

        with open(os.path.join(path, csv_filename), 'w') as csvfile:
            fieldnames = ['type', 'c(cm-3)', 'T(K)', 'overall', 'average'] + \
                         self.elastic_scatterings + self.inelastic_scatterings + ['seebeck']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for c in self.dopings:
                tp = get_tp(c)
                for T in self.temperatures:
                    row = {'type': tp, 'c(cm-3)': abs(c), 'T(K)': T}
                    for p in ['overall', 'average'] + self.elastic_scatterings + self.inelastic_scatterings:
                        row[p] = sum(self.mobility[tp][p][c][T])/3
                    try:
                        row["seebeck"] = sum(self.egrid[tp]["seebeck"][c][T])/3
                    except TypeError:
                        row["seebeck"] = self.egrid[tp]["seebeck"][c][T]
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
        #print(self.kgrid_array)


if __name__ == "__main__":
    # setting up inputs:
    mass = 0.25
    use_poly_bands = False
    add_extrema = None
    # add_extrema = {'n': [[0.5, 0.5, 0.5]], 'p':[]}
    PRE_DETERMINED_FERMI = None

    model_params = {'bs_is_isotropic': True,
                    'elastic_scatterings': ['ACD', 'IMP', 'PIE'],
                    'inelastic_scatterings': ['POP']
        , 'independent_valleys': False
                    }
    if use_poly_bands:
        model_params["poly_bands"] = [[
            [[0.0, 0.0, 0.0], [0.0, mass]],
        ]]

    # TODO: see why job fails with any k-mesh but max_normk==1 ?? -AF update 20180207: didn't return error with very coarse
    performance_params = {"dE_min": 0.0001, "nE_min": 2, "parallel": False,
            "BTE_iters": 5, "max_nbands": 1, "max_normk": 1.6, "max_ncpu": 4
                          , "fermi_kgrid_tp": "uniform", "max_nvalleys": 1
                          , "pre_determined_fermi": PRE_DETERMINED_FERMI
                          , "interpolation": "boltztrap1"
                          }

    ### for PbTe
    # material_params = {"epsilon_s": 44.4, "epsilon_inf": 25.6, "W_POP": 10.0, "C_el": 128.8,
    #                "E_D": {"n": 4.0, "p": 4.0}}
    # cube_path = "../test_files/PbTe/nscf_line"
    # coeff_file = os.path.join(cube_path, "..", "fort.123")
    # #coeff_file = os.path.join(cube_path, "fort.123")

    material_params = {"epsilon_s": 12.9, "epsilon_inf": 10.9, "W_POP": 8.73,
            "C_el": 139.7, "E_D": {"n": 8.6, "p": 8.6}, "P_PIE": 0.052, 'add_extrema': add_extrema
            # , "scissor": 0.5818, "user_bandgap": 1.54,
            # , 'important_points': {'n': [[0.0, 0.0, 0.0]], 'p':[[0, 0, 0]]}
            # , 'important_points': {'n': [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], 'p': [[0, 0, 0]]}
                       }
    cube_path = "../test_files/GaAs/"
    #####coeff_file = os.path.join(cube_path, "fort.123_GaAs_k23")
    # coeff_file = os.path.join(cube_path, "fort.123_GaAs_1099kp") # good results!
    coeff_file = os.path.join(cube_path, "nscf-uniform/boltztrap/fort.123")

    ## coeff_file = os.path.join(cube_path, "fort.123_GaAs_sym_23x23x23") # bad results! (because the fitting not good)
    ## coeff_file = os.path.join(cube_path, "fort.123_GaAs_11x11x11_ISYM0") # good results

    ### For Si
    # material_params = {"epsilon_s": 11.7, "epsilon_inf": 11.6, "W_POP": 15.23, "C_el": 190.2,
    #                    "E_D": {"n": 6.5, "p": 6.5}, "P_PIE": 0.01, "scissor": 0.5154}
    # cube_path = "../test_files/Si/"
    # coeff_file = os.path.join(cube_path, "Si_fort.123")

    # ## For AlCuS2
    # cube_path = '../test_files/AlCuS2'
    # coeff_file = None
    # material_params = {"epsilon_s": 7.6, "epsilon_inf": 4.85, "W_POP": 12.6,
    #                    "C_el": 110, "E_D": {"n": 9.67, "p": 3.175}, "P_PIE": 0.052, "scissor":  1.42}
    # # in terms of anisotropy at 5e19 300K BoltzTraP return sigma/tau of [8.55e17, 8.86e17, 1.08e18] for xx, yy, zz respectively

    amset = AMSET(calc_dir=cube_path, material_params=material_params,
                  model_params=model_params, performance_params=performance_params,
                  dopings = [-3e13],
                  # dopings = [-1e20],
                  # dopings = [5.10E+18, 7.10E+18, 1.30E+19, 2.80E+19, 6.30E+19],
                  # dopings = [3.32e14],
                  temperatures = [300],
                  # temperatures = [300, 400, 500, 600, 700, 800, 900, 1000],
                  # temperatures = [201.36, 238.991, 287.807, 394.157, 502.575, 596.572],

                  # temperatures = range(100, 1100, 100),
                  k_integration=False, e_integration=True, fermi_type='e',
                  # loglevel=logging.DEBUG
                  )
    amset.run_profiled(coeff_file, kgrid_tp='very coarse', write_outputs=True)


    # stats.print_callers(10)

    amset.write_input_files()
    amset.to_csv()
    # amset.plot(k_plots=['energy', 'S_o', 'S_i']\
    #                    # +model_params['elastic_scatterings']
    #            , E_plots=['velocity', 'df0dk'], show_interactive=True
    #            , carrier_types=amset.all_types
    #            , save_format=None)

    amset.to_json(kgrid=True, trimmed=True, max_ndata=100, nstart=0)
