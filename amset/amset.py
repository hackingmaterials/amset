# coding: utf-8
import warnings
import time
import logging
import json
from pstats import Stats
from random import random
from scipy.interpolate import griddata
from scipy.constants.codata import value as _cd
from pprint import pprint
import os
from sys import stdout as STDOUT

import numpy as np
from math import log, pi

from pymatgen.io.vasp import Vasprun, Spin, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from monty.json import MontyEncoder
import cProfile
from copy import deepcopy
import multiprocessing
from joblib import Parallel, delayed
from analytical_band_from_BZT import Analytical_bands, outer, get_dos_from_poly_bands, get_energy, get_poly_energy

from tools import norm, grid_norm, f0, df0dE, cos_angle, fermi_integral, GB, \
        calculate_Sio, calculate_Sio_list, remove_from_grid, \
        remove_duplicate_kpoints


__author__ = "Alireza Faghaninia, Jason Frost, Anubhav Jain"
__copyright__ = "Copyright 2017, HackingMaterials"
__version__ = "0.1"
__maintainer__ = "Alireza Faghaninia"
__email__ = "alireza.faghaninia@gmail.com"
__status__ = "Development"
__date__ = "July 2017"


# some global constants
hbar = _cd('Planck constant in eV s') / (2 * pi)
m_e = _cd('electron mass')  # in kg
Ry_to_eV = 13.605698066
A_to_m = 1e-10
m_to_cm = 100.00
A_to_nm = 0.1
e = _cd('elementary charge')
k_B = _cd("Boltzmann constant in eV/K")
epsilon_0 = 8.854187817e-12  # dielectric constant in vacuum [C**2/m**2N]
default_small_E = 1  # eV/cm the value of this parameter does not matter
dTdz = 10.0  # K/cm
sq3 = 3 ** 0.5


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
                 dopings=None, temperatures=None, k_integration=True, e_integration=False, fermi_type='k'):
        """
        Args:
            calc_dir (str): path to the vasprun.xml (a required argument)
            material_params (dict): parameters related to the material (a required argument)
            model_params (dict): parameters related to the model used and the level of theory
            performance_params (dict): parameters related to convergence, speed, etc.
            dopings ([float]): list of input carrier concentrations; c<0 for electrons and c>0 for holes
            temperatures ([float]): list of input temperatures
        """

        self.calc_dir = calc_dir
        self.dopings = dopings or [-1e16, -1e17, -1e18, -1e19, -1e20, -1e21, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21]
        self.all_types = list(set([self.get_tp(c) for c in self.dopings]))
        self.tp_title = {"n": "conduction band(s)", "p": "valence band(s)"}
        self.temperatures = temperatures or map(float, [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        self.debug_tp = self.get_tp(self.dopings[0])
        logging.debug("""debug_tp: "{}" """.format(self.debug_tp))
        self.set_material_params(material_params)
        self.set_model_params(model_params)
        self.set_performance_params(performance_params)
        self.k_integration = k_integration
        self.e_integration = e_integration
        self.fermi_calc_type = fermi_type

        self.read_vrun(calc_dir=self.calc_dir, filename="vasprun.xml")
        if self.poly_bands:
            self.cbm_vbm["n"]["energy"] = self.dft_gap
            self.cbm_vbm["p"]["energy"] = 0.0
            self.cbm_vbm["n"]["kpoint"] = self.cbm_vbm["p"]["kpoint"] = self.poly_bands[0][0][0]

        self.num_cores = max(int(multiprocessing.cpu_count()/4), 8)
        if self.parallel:
            logging.info("number of cpu used in parallel mode: {}".format(self.num_cores))



    def run(self, coeff_file, kgrid_tp="coarse", loglevel=logging.WARNING):
        """
        Function to run AMSET and generate the main outputs.

        Args:
        coeff_file: the fort.123* file which contains the coefficients of the interpolated band structure
                it is generated by a modified version of BoltzTraP
        kgrid_tp (str): define the density of k-point mesh.
            options: 'very coarse', 'coarse', 'fine'
        loglevel (int): e.g. logging.DEBUG
        """
        logging.basicConfig(level=loglevel)
        self.init_kgrid(coeff_file=coeff_file, kgrid_tp=kgrid_tp)
        logging.debug("self.cbm_vbm: {}".format(self.cbm_vbm))

        self.f0_array = {c: {T: {'n': None, 'p': None} for T in self.temperatures} for c in self.dopings}
        if self.fermi_calc_type == 'k':
            self.fermi_level = self.find_fermi_k()
            self.calc_doping = {c: {T: {'n': None, 'p': None} for T in self.temperatures} for c in self.dopings}
            for c in self.dopings:
                for T in self.temperatures:
                    for tp in ['n', 'p']:
                        self.f0_array[c][T][tp] = 1 / (np.exp((self.energy_array[tp] - self.fermi_level[c][T]) / (k_B * T)) + 1)
                    self.calc_doping[c][T]['n'] = -self.integrate_over_states(self.f0_array[c][T]['n'])
                    self.calc_doping[c][T]['p'] = self.integrate_over_states(1-self.f0_array[c][T]['p'])

        self.init_egrid(dos_tp="standard")
        logging.info('fermi level = {}'.format(self.fermi_level))
        self.bandgap = min(self.egrid["n"]["all_en_flat"]) - max(self.egrid["p"]["all_en_flat"])
        if abs(self.bandgap - (self.cbm_vbm["n"]["energy"] - self.cbm_vbm["p"]["energy"] + self.scissor)) > k_B * 300:
            warnings.warn("The band gaps do NOT match! The selected k-mesh is probably too coarse.")
            # raise ValueError("The band gaps do NOT match! The selected k-mesh is probably too coarse.")

        # initialize g in the egrid
        self.map_to_egrid("g", c_and_T_idx=True, prop_type="vector")
        self.map_to_egrid(prop_name="velocity", c_and_T_idx=False, prop_type="vector")

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
                dop_tp = self.get_tp(c)
                f0_removed = self.array_from_kgrid('f0', dop_tp, c, T)
                f0_all = 1 / (np.exp((self.energy_array[dop_tp] - self.fermi_level[c][T]) / (k_B * T)) + 1)
                if c < 0:
                    result = self.integrate_over_states(f0_removed)
                    result2 = self.integrate_over_states(f0_all)
                    print('integral (points removed) of f0 over k at c={}, T={}: {}'.format(c, T, result))
                    print('integral (all points) of f0 over k at c={}, T={}: {}'.format(c, T, result2))
                if c > 0:
                    p_result = self.integrate_over_states(1-f0_removed)
                    p_result2 = self.integrate_over_states(1-f0_all)
                    print('integral (points removed) of 1-f0 over k at c={}, T={}: {}'.format(c, T, p_result))
                    print('integral (all points) of 1-f0 over k at c={}, T={}: {}'.format(c, T, p_result2))


        self.map_to_egrid(prop_name="f0", c_and_T_idx=True, prop_type="vector")
        self.map_to_egrid(prop_name="df0dk", c_and_T_idx=True, prop_type="vector")

        # solve BTE in presence of electric and thermal driving force to get perturbation to Fermi-Dirac: g
        self.solve_BTE_iteratively()

        if self.k_integration:
            self.calculate_transport_properties_with_k()
        if self.e_integration:
            self.calculate_transport_properties_with_E()

        kgrid_rm_list = ["effective mass", "kweights",
                         "f_th", "S_i_th", "S_o_th"]
        self.kgrid = remove_from_grid(self.kgrid, kgrid_rm_list)

        if self.k_integration:
            pprint(self.mobility)
        if self.e_integration:
            pprint(self.egrid["mobility"])



    def write_input_files(self):
        """writes all 3 types of inputs in json files for example to
        conveniently track what inputs had been used later or read
        inputs from files (see from_files method)"""
        material_params = {
            "epsilon_s": self.epsilon_s,
            "epsilon_inf": self.epsilon_inf,
            "C_el": self.C_el,
            "W_POP": self.W_POP / (1e12 * 2 * pi),
            "P_PIE": self.P_PIE,
            "E_D": self.E_D,
            "N_dis": self.N_dis,
            "scissor": self.scissor,
            "donor_charge": self.charge["n"],
            "acceptor_charge": self.charge["p"],
            "dislocations_charge": self.charge["dislocations"]
        }

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
            "BTE_iters": self.BTE_iters
        }

        with open("material_params.json", "w") as fp:
            json.dump(material_params, fp, sort_keys=True, indent=4, ensure_ascii=False, cls=MontyEncoder)
        with open("model_params.json", "w") as fp:
            json.dump(model_params, fp, sort_keys=True, indent=4, ensure_ascii=False, cls=MontyEncoder)
        with open("performance_params.json", "w") as fp:
            json.dump(performance_params, fp, sort_keys=True, indent=4, ensure_ascii=False, cls=MontyEncoder)



    def set_material_params(self, params):

        self.epsilon_s = params["epsilon_s"]
        self.epsilon_inf = params["epsilon_inf"]
        self.C_el = params["C_el"]
        self.W_POP = params["W_POP"] * 1e12 * 2 * pi

        self.P_PIE = params.get("P_PIE", 0.15)  # unitless
        self.E_D = params.get("E_D", {"n": 4.0, "p": 4.0})

        self.N_dis = params.get("N_dis", 0.1)  # in 1/cm**2
        self.scissor = params.get("scissor", 0.0)

        donor_charge = params.get("donor_charge", 1.0)
        acceptor_charge = params.get("acceptor_charge", 1.0)
        dislocations_charge = params.get("dislocations_charge", 1.0)
        self.charge = {"n": donor_charge, "p": acceptor_charge, "dislocations": dislocations_charge}



    def set_model_params(self, params):
        """function to set instant variables related to the model and the level of the theory;
        these are set based on params (dict) set by the user or their default values"""

        self.bs_is_isotropic = params.get("bs_is_isotropic", False)
        self.elastic_scatterings = params.get("elastic_scatterings", ["ACD", "IMP", "PIE"])
        self.inelastic_scatterings = params.get("inelastic_scatterings", ["POP"])

        self.poly_bands = params.get("poly_bands", None)

        # TODO: self.gaussian_broadening is designed only for development version and must be False, remove it later.
        # because if self.gaussian_broadening the mapping to egrid will be done with the help of Gaussian broadening
        # and that changes the actual values
        self.gaussian_broadening = False
        self.soc = params.get("soc", False)
        logging.info("bs_is_isotropic: {}".format(self.bs_is_isotropic))



    def set_performance_params(self, params):
        self.nkibz = params.get("nkibz", 40)
        self.dE_min = params.get("dE_min", 0.0001)
        self.nE_min = params.get("nE_min", 2)
        c_factor = max(1, 2*abs(max([log(abs(ci)/float(1e19)) for ci in self.dopings]))**0.15)
        Ecut = params.get("Ecut", c_factor * 15 * k_B * max(self.temperatures + [300]))
        self.Ecut = {tp: Ecut if tp in self.all_types else Ecut/2.0 for tp in ["n", "p"]}
        for tp in ["n", "p"]:
            logging.debug("{}-Ecut: {} eV \n".format(tp, self.Ecut[tp]))
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
        logging.info("parallel: {}".format(self.parallel))



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
        self.volume = self._vrun.final_structure.volume
        logging.info("unitcell volume = {} A**3".format(self.volume))
        self.density = self._vrun.final_structure.density
        self._rec_lattice = self._vrun.final_structure.lattice.reciprocal_lattice
        bs = self._vrun.get_band_structure()
        self.nbands = bs.nb_bands
        self.lorbit = 11 if len(sum(self._vrun.projected_eigenvalues[Spin.up][0][10])) > 5 else 10

        self.DFT_cartesian_kpts = np.array(
                [self._rec_lattice.get_cartesian_coords(k) for k in self._vrun.actual_kpoints])/ A_to_nm


        # Remember that python band index starts from 0 so bidx==9 refers to the 10th band in VASP
        cbm_vbm = {"n": {"kpoint": [], "energy": 0.0, "bidx": 0, "included": 0, "eff_mass_xx": [0.0, 0.0, 0.0]},
                   "p": {"kpoint": [], "energy": 0.0, "bidx": 0, "included": 0, "eff_mass_xx": [0.0, 0.0, 0.0]}}
        cbm = bs.get_cbm()
        vbm = bs.get_vbm()

        logging.info("total number of bands: {}".format(self._vrun.get_band_structure().nb_bands))

        cbm_vbm["n"]["energy"] = cbm["energy"]
        try:
            cbm_vbm["n"]["bidx"] = cbm["band_index"][Spin.up][0]
        except IndexError:
            cbm_vbm["n"]["bidx"] = cbm["band_index"][Spin.down][0] # in case spin down has a lower CBM
        cbm_vbm["n"]["kpoint"] = bs.kpoints[cbm["kpoint_index"][0]].frac_coords

        cbm_vbm["p"]["energy"] = vbm["energy"]
        try:
            cbm_vbm["p"]["bidx"] = vbm["band_index"][Spin.up][-1]
        except IndexError:
            cbm_vbm["p"]["bidx"] = vbm["band_index"][Spin.down][-1]
        cbm_vbm["p"]["kpoint"] = bs.kpoints[vbm["kpoint_index"][0]].frac_coords

        self.dft_gap = cbm["energy"] - vbm["energy"]
        logging.debug("DFT gap from vasprun.xml : {} eV".format(self.dft_gap))

        if self.soc:
            self.nelec = cbm_vbm["p"]["bidx"] + 1
        else:
            self.nelec = (cbm_vbm["p"]["bidx"] + 1) * 2

        logging.debug("total number of electrons nelec: {}".format(self.nelec))

        bs = bs.as_dict()
        if bs["is_spin_polarized"]:
            self.dos_emin = min(bs["bands"]["1"][0], bs["bands"]["-1"][0])
            self.dos_emax = max(bs["bands"]["1"][-1], bs["bands"]["-1"][-1])
        else:
            self.dos_emin = min(bs["bands"]["1"][0])
            self.dos_emax = max(bs["bands"]["1"][-1])

        if not self.poly_bands:
            for i, tp in enumerate(["n", "p"]):
                Ecut = self.Ecut[tp]
                sgn = (-1) ** i
                while abs(min(sgn * bs["bands"]["1"][cbm_vbm[tp]["bidx"] + sgn * cbm_vbm[tp]["included"]]) -
                                          sgn * cbm_vbm[tp]["energy"]) < Ecut:
                    cbm_vbm[tp]["included"] += 1

                # TODO: for now, I only include 1 band for quicker testing
                # cbm_vbm[tp]["included"] = 1
        else:
            cbm_vbm["n"]["included"] = cbm_vbm["p"]["included"] = len(self.poly_bands)

        cbm_vbm["p"]["bidx"] += 1
        cbm_vbm["n"]["bidx"] = cbm_vbm["p"]["bidx"] + 1

        self.cbm_vbm = cbm_vbm
        logging.info("original cbm_vbm:\n {}".format(self.cbm_vbm))



    def get_tp(self, c):
        """returns "n" for n-tp or negative carrier concentration or "p" (p-tp)."""
        if c < 0:
            return "n"
        elif c > 0:
            return "p"
        else:
            raise ValueError("The carrier concentration cannot be zero! AMSET stops now!")

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
        N_II = abs(self.egrid["calc_doping"][c][T]["n"]) * self.charge["n"] ** 2 + \
               abs(self.egrid["calc_doping"][c][T]["p"]) * self.charge["p"] ** 2 + \
               self.N_dis / self.volume ** (1 / 3) * 1e8 * self.charge["dislocations"] ** 2
        return N_II



    def init_egrid(self, dos_tp="simple"):
        """
        :param
            dos_tp (string): options are "simple", ...

        :return: an updated grid that contains the field DOS
        """

        self.egrid = {
            "n": {"energy": [], "DOS": [], "all_en_flat": [], "all_ks_flat": []},
            "p": {"energy": [], "DOS": [], "all_en_flat": [], "all_ks_flat": []},
            "mobility": {}
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

        logging.debug("here total number of ks from self.Efrequency for {}-type".format(self.debug_tp))
        logging.debug(sum(self.Efrequency[self.debug_tp]))

        min_nE = 2

        if len(self.Efrequency["n"]) < min_nE or len(self.Efrequency["p"]) < min_nE:
            raise ValueError("The final egrid have fewer than {} energy values, AMSET stops now".format(min_nE))

        # initialize some fileds/properties
        self.egrid["calc_doping"] = {c: {T: {"n": 0.0, "p": 0.0} for T in self.temperatures} for c in self.dopings}
        for sn in self.elastic_scatterings + self.inelastic_scatterings + ["overall", "average", "SPB_ACD"]:
            self.egrid["mobility"][sn] = {c: {T: {"n": 0.0, "p": 0.0} for T in self.temperatures} for c in self.dopings}
        for transport in ["conductivity", "J_th", "seebeck", "TE_power_factor", "relaxation time constant"]:
            self.egrid[transport] = {c: {T: {"n": 0.0, "p": 0.0} for T in self.temperatures} for c in self.dopings}

        # populate the egrid at all c and T with properties; they can be called via self.egrid[prop_name][c][T] later
        if self.fermi_calc_type == 'k':
            self.egrid["calc_doping"] = self.calc_doping
        if self.fermi_calc_type == 'e':
            self.calculate_property(prop_name="fermi", prop_func=self.find_fermi)
            self.fermi_level = self.egrid["fermi"]

        #TODO: comment out these 3 lines and test, these were commented out in master 9/27/2017
        self.calculate_property(prop_name="f0", prop_func=f0, for_all_E=True)
        self.calculate_property(prop_name="f", prop_func=f0, for_all_E=True)
        self.calculate_property(prop_name="f_th", prop_func=f0, for_all_E=True)

        for prop in ["f", "f_th"]:
            self.map_to_egrid(prop_name=prop, c_and_T_idx=True)

        self.calculate_property(prop_name="f0x1-f0", prop_func=lambda E, fermi, T: f0(E, fermi, T)
                                                                                   * (1 - f0(E, fermi, T)),
                                for_all_E=True)

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
                logging.debug("# of {}-type kpoints indexes with low velocity or off-energy: {}".format(tp,len(rm_idx_list_ib)))
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
        return self.kpts_to_first_BZ(final_kpts_added)



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

        print "here initial k-points for {}-type with low energy distance".format(self.debug_tp)
        print len(kpoints_added[self.debug_tp])
        # print kpoints_added[self.debug_tp]
        final_kpts_added = []
        for tp in ["n", "p"]:
            # final_kpts_added = []
            # TODO: in future only add the relevant k-poits for "kpoints" for each type separately
            # print kpoints_added[tp]
            for ik in range(len(kpoints_added[tp]) - 1):
                final_kpts_added += self.get_intermediate_kpoints_list(list(kpoints_added[tp][ik]),
                                                                       list(kpoints_added[tp][ik + 1]), nsteps)

        return self.kpts_to_first_BZ(final_kpts_added)



    def kpts_to_first_BZ(self, kpts):
        for i, k in enumerate(kpts):
            for alpha in range(3):
                if k[alpha] > 0.5:
                    k[alpha] -= 1
                if k[alpha] < -0.5:
                    k[alpha] += 1
            kpts[i] = k
        return kpts



    def get_sym_eq_ks_in_first_BZ(self, k, cartesian=False):
        """

        :param k (numpy.array): kpoint fractional coordinates
        :param cartesian (bool): if True, the output would be in cartesian (but still reciprocal) coordinates
        :return:
        """
        fractional_ks = [np.dot(k, self.rotations[i]) + self.translations[i] for i in range(len(self.rotations))]
        #TODO: not sure if I should include also the translations or not (see Si example to see if it makes a difference)
        # fractional_ks = [np.dot(k, self.rotations[i]) for i in range(len(self.rotations))]

        fractional_ks = self.kpts_to_first_BZ(fractional_ks)
        if cartesian:
            return [self._rec_lattice.get_cartesian_coords(k_frac) / A_to_nm for k_frac in fractional_ks]
        else:
            return fractional_ks



    # @albalu I created this function but do not understand what most of the arguments are. It may make sense to contain
    # them all in a single labeled tuple so the code is more readable?
    # engre through sgn: use for analytical bands energy; tp and ib: use for poly bands energy
    def calc_analytical_energy(self, xkpt, engre, nwave, nsym, nstv, vec, vec2, out_vec2, br_dir, sgn):
        """
            :param xkpt (?): ?
            :param engre (?): ?
            :param nwave (?): ?
            :param nsym (?): ?
            :param nstv (?): ?
            :param vec (?): ?
            :param vec2 (?): ?
            :param out_vec2 (?): ?
            :param br_dir (?): ?
            :param sgn (int): -1 or 1
        """
        energy, de, dde = get_energy(xkpt, engre, nwave, nsym, nstv, vec, vec2, out_vec2, br_dir=br_dir)
        energy = energy * Ry_to_eV - sgn * self.scissor / 2.0
        velocity = abs(de / hbar * A_to_m * m_to_cm * Ry_to_eV)
        effective_m = hbar ** 2 / (
            dde * 4 * pi ** 2) / m_e / A_to_m ** 2 * e * Ry_to_eV
        return energy, velocity, effective_m



    def calc_poly_energy(self, xkpt, tp, ib):
        '''
        :param tp: "p" or "n"
        :param ib: band index...?
        :return:
        '''
        energy, velocity, effective_m = get_poly_energy(
            self._rec_lattice.get_cartesian_coords(xkpt) / A_to_nm,
            poly_bands=self.poly_bands, type=tp, ib=ib, bandgap=self.dft_gap + self.scissor)
        return energy, velocity, effective_m


    def init_kgrid(self, coeff_file, kgrid_tp="coarse"):
        logging.debug("begin profiling init_kgrid: a {} grid".format(kgrid_tp))
        start_time = time.time()
        sg = SpacegroupAnalyzer(self._vrun.final_structure)
        self.rotations, self.translations = sg._get_symmetry()

        logging.info("self.nkibz = {}".format(self.nkibz))
        self.kgrid_array = {}
        points_1d = {dir: [] for dir in ['x', 'y', 'z']}
        # TODO: figure out which other points need a fine grid around them
        # TODO-JF: can we have a separate mesh for n- and p-type from here?
        important_pts = [self.cbm_vbm["n"]["kpoint"]]
        if (np.array(self.cbm_vbm["p"]["kpoint"]) != np.array(self.cbm_vbm["n"]["kpoint"])).any():
            important_pts.append(self.cbm_vbm["p"]["kpoint"])

        for center in important_pts:
            for dim, dir in enumerate(['x', 'y', 'z']):
                points_1d[dir].append(center[dim])
                one_list = True
                if not one_list:
                    for step, nsteps in [[0.002, 2], [0.005, 4], [0.01, 4], [0.05, 2],[0.1, 5]]:
                        for i in range(nsteps - 1):
                            points_1d[dir].append(center[dim]-(i+1)*step)
                            points_1d[dir].append(center[dim]+(i+1)*step)

                else:
                    if kgrid_tp == 'fine':
                        mesh = [0.004, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.25, 0.35, 0.5]
                    elif kgrid_tp == 'coarse':
                        mesh = [0.001, 0.005, 0.01, 0.05, 0.15, 0.5]
                    elif kgrid_tp == 'very coarse':
                        mesh = [0.001, 0.01, 0.1, 0.5]
                    else:
                        raise ValueError('Unsupported value for kgrid_tp: {}'.format(kgrid_tp))
                    for step in mesh:
                        points_1d[dir].append(center[dim] + step)
                        points_1d[dir].append(center[dim] - step)
        logging.info('included points in the mesh: {}'.format(points_1d))

        # ensure all points are in "first BZ" (parallelepiped)
        for dir in ['x', 'y', 'z']:
            for ik1d in range(len(points_1d[dir])):
                if points_1d[dir][ik1d] > 0.5:
                    points_1d[dir][ik1d] -= 1
                if points_1d[dir][ik1d] < -0.5:
                    points_1d[dir][ik1d] += 1

        # remove duplicates
        for dir in ['x', 'y', 'z']:
            points_1d[dir] = list(set(np.array(points_1d[dir]).round(decimals=14)))
        self.kgrid_array['k_points'] = self.create_grid(points_1d)
        kpts = self.array_to_kgrid(self.kgrid_array['k_points'])

        N = self.kgrid_array['k_points'].shape
        self.k_hat_grid = np.zeros(N)
        for i in range(N[0]):
            for j in range(N[1]):
                for k in range(N[2]):
                    k_vec = self.kgrid_array['k_points'][i,j,k]
                    if norm(k_vec) == 0:
                        self.k_hat_grid[i,j,k] = [0, 0, 0]
                    else:
                        self.k_hat_grid[i,j,k] = k_vec / norm(k_vec)
        self.dv_grid = self.find_dv(self.kgrid_array['k_points'])

        logging.info("number of original ibz k-points: {}".format(len(kpts)))
        logging.debug("time to get the ibz k-mesh: \n {}".format(time.time()-start_time))
        start_time = time.time()
        # TODO-JF: this if setup energy calculation for SPB and actual BS it would be nice to do this in two separate functions
        # if using analytical bands: create the object, determine list of band indices, and get energy info
        if not self.poly_bands:
            logging.debug("start interpolating bands from {}".format(coeff_file))
            analytical_bands = Analytical_bands(coeff_file=coeff_file)
            all_ibands = []
            for i, tp in enumerate(["p", "n"]):
                sgn = (-1) ** (i + 1)
                for ib in range(self.cbm_vbm[tp]["included"]):
                    all_ibands.append(self.cbm_vbm[tp]["bidx"] + sgn * ib)

            logging.debug("all_ibands: {}".format(all_ibands))

            # @albalu what are all of these variables (in the next 5 lines)? I don't know but maybe we can lump them together
            engre, latt_points, nwave, nsym, nsymop, symop, br_dir = analytical_bands.get_engre(iband=all_ibands)
            nstv, vec, vec2 = analytical_bands.get_star_functions(latt_points, nsym, symop, nwave, br_dir=br_dir)
            out_vec2 = np.zeros((nwave, max(nstv), 3, 3))
            for nw in xrange(nwave):
                for i in xrange(nstv[nw]):
                    out_vec2[nw, i] = outer(vec2[nw, i], vec2[nw, i])


        # if using poly bands, remove duplicate k points (@albalu I'm not really sure what this is doing)
        else:
            # first modify the self.poly_bands to include all symmetrically equivalent k-points (k_i)
            # these points will be used later to generate energy based on the minimum norm(k-k_i)
            for ib in range(len(self.poly_bands)):
                for j in range(len(self.poly_bands[ib])):
                    self.poly_bands[ib][j][0] = remove_duplicate_kpoints(
                        self.get_sym_eq_ks_in_first_BZ(self.poly_bands[ib][j][0], cartesian=True))

        logging.debug("time to get engre and calculate the outvec2: {} seconds".format(time.time() - start_time))

        # calculate only the CBM and VBM energy values - @albalu why is this separate from the other energy value calculations?
        # here we assume that the cbm and vbm k-point coordinates read from vasprun.xml are correct:

        for i, tp in enumerate(["p", "n"]):
            sgn = (-1) ** i

            if not self.poly_bands:
                energy, velocity, effective_m = self.calc_analytical_energy(
                        self.cbm_vbm[tp]["kpoint"],engre[i * self.cbm_vbm["p"][
                        "included"]],nwave, nsym, nstv, vec, vec2, out_vec2,
                        br_dir, sgn)
            else:
                energy, velocity, effective_m = self.calc_poly_energy(
                        self.cbm_vbm[tp]["kpoint"], tp, 0)

            # @albalu why is there already an energy value calculated from vasp that this code overrides? # we are renormalizing the E vlues as the new energy has a different reference energy
            self.offset_from_vrun = energy - self.cbm_vbm[tp]["energy"]
            logging.debug("offset from vasprun energy values for {}-type = {} eV".format(tp, self.offset_from_vrun))
            self.cbm_vbm[tp]["energy"] = energy
            self.cbm_vbm[tp]["eff_mass_xx"] = effective_m.diagonal()

        if not self.poly_bands:
            self.dos_emax += self.offset_from_vrun
            self.dos_emin += self.offset_from_vrun

        logging.debug("cbm_vbm after recalculating their energy values:\n {}".format(self.cbm_vbm))
        self._avg_eff_mass = {tp: abs(np.mean(self.cbm_vbm[tp]["eff_mass_xx"])) for tp in ["n", "p"]}

        # calculate the energy at initial ibz k-points and look at the first band to decide on additional/adaptive ks
        start_time = time.time()
        energies = {"n": [0.0 for ik in kpts], "p": [0.0 for ik in kpts]}
        velocities = {"n": [[0.0, 0.0, 0.0] for ik in kpts], "p": [[0.0, 0.0, 0.0] for ik in kpts]}

        # These two lines should be commented out when kpts is already for each carrier type
        kpts_copy = np.array(kpts)
        kpts = {"n": np.array(kpts_copy), "p": np.array(kpts_copy)}

        self.pos_idx = {'n': [], 'p': []}
        self.num_bands = {tp: self.cbm_vbm[tp]["included"] for tp in ['n', 'p']}
        self.energy_array = {'n': [], 'p': []}

        # calculate energies
        for i, tp in enumerate(["p", "n"]):
            sgn = (-1) ** i
            for ib in range(self.cbm_vbm[tp]["included"]):
                if not self.parallel or self.poly_bands:  # The PB generator is fast enough no need for parallelization
                    for ik in range(len(kpts[tp])):
                        if not self.poly_bands:
                            energy, velocities[tp][ik], effective_m = self.calc_analytical_energy(kpts[tp][ik],engre[i * self.cbm_vbm[
                                "p"]["included"] + ib],nwave, nsym, nstv, vec, vec2,out_vec2, br_dir, sgn)
                        else:
                            energy, velocities[tp][ik], effective_m = self.calc_poly_energy(kpts[tp][ik], tp, ib)
                        energies[tp][ik] = energy

                        # @albalu why do we exclude values of k that have a small component of velocity?
                        # @Jason: because scattering equations have v in the denominator: get too large for such points
                        # if velocity[0] < self.v_min or velocity[1] < self.v_min or velocity[2] < self.v_min or \
                        #                 abs(energy - self.cbm_vbm[tp]["energy"]) > Ecut:
                        #     rm_list[tp].append(ik)
                else:
                    results = Parallel(n_jobs=self.num_cores)(delayed(get_energy)(kpts[tp][ik],engre[i * self.cbm_vbm["p"][
                        "included"] + ib], nwave, nsym, nstv, vec, vec2, out_vec2, br_dir) for ik in range(len(kpts[tp])))
                    for ik, res in enumerate(results):
                        energies[tp][ik] = res[0] * Ry_to_eV - sgn * self.scissor / 2.0
                        velocities[tp][ik] = abs(res[1] / hbar * A_to_m * m_to_cm * Ry_to_eV)
                        # if velocity[0] < self.v_min or velocity[1] < self.v_min or velocity[2] < self.v_min or \
                        #                 abs(energies[tp][ik] - self.cbm_vbm[tp]["energy"]) > Ecut:
                        #     # if tp=="p":
                        #     #     print "reason for removing the k-point:"
                        #     #     print "energy: {}".format(energies[tp][ik])
                        #     #     print "velocity: {}".format(velocity)
                        #     rm_list[tp].append(ik)

                self.energy_array[tp].append(self.grid_from_ordered_list(energies[tp], none_missing=True))

                if ib == 0:      # we only include the first band to decide on order of ibz k-points
                    e_sort_idx = np.array(energies[tp]).argsort() if tp == "n" else np.array(energies[tp]).argsort()[::-1]
                    energies[tp] = [energies[tp][ie] for ie in e_sort_idx]
                    velocities[tp] = [velocities[tp][ie] for ie in e_sort_idx]
                    self.pos_idx[tp] = np.array(range(len(e_sort_idx)))[e_sort_idx].argsort()
                    kpts[tp] = [kpts[tp][ie] for ie in e_sort_idx]


            # self.energy_array[tp] = [self.grid_from_ordered_list(energies[tp], none_missing=True) for ib in range(self.num_bands[tp])]

            # e_sort_idx = np.array(energies[tp]).argsort() if tp =="n" else np.array(energies[tp]).argsort()[::-1]

            # energies[tp] = [energies[tp][ie] for ie in e_sort_idx]

            # self.dos_end = max(energies["n"])
            # self.dos_start = min(energies["p"])

            # velocities[tp] = [velocities[tp][ie] for ie in e_sort_idx]
            # self.pos_idx[tp] = np.array(range(len(e_sort_idx)))[e_sort_idx].argsort()

            # kpts[tp] = [kpts[tp][ie] for ie in e_sort_idx]

        N = self.kgrid_array['k_points'].shape

        for ib in range(self.num_bands['n']):
            print('energy (type n, band {}):'.format(ib))
            print(self.energy_array['n'][ib][(N[0] - 1) / 2, (N[1] - 1) / 2, :])

        logging.debug("time to calculate ibz energy, velocity info and store them to variables: \n {}".format(time.time()-start_time))
        start_time = time.time()
        #TODO: the following for-loop is crucial but undone! it decides which k-points remove for speed and accuracy
        '''for tp in ["p", "n"]:
            Ecut = self.Ecut[tp]
            Ediff_old = 0.0
            # print "{}-type all Ediffs".format(tp)
            for ib in [0]:
                ik = -1
                # for ik in range(len(kpts[tp])):
                while ik < len(kpts[tp])-1:
                    ik += 1
                    Ediff = abs(energies[tp][ik] - self.cbm_vbm[tp]["energy"])
                    if Ediff > Ecut:
                        rm_list[tp] += range(ik, len(kpts[tp]))
                        break  # because the energies are sorted so after this point all energy points will be off
                    if velocities[tp][ik][0] < self.v_min or velocities[tp][ik][1] < self.v_min or\
                                    velocities[tp][ik][2] < self.v_min:
                        rm_list[tp].append(ik)

                    # the following if implements an adaptive dE_min as higher energy points are less important
                    #TODO: note that if k-mesh integration on a regular grid (not tetrahedron) is implemented, the
                    #TODO:following will make the results wrong as in that case we would assume the removed points are 0
                    while ik < len(kpts[tp])-1 and \
                            (Ediff > Ecut/5.0 and Ediff - Ediff_old < min(self.dE_min*10.0, 0.001) or
                            (Ediff > Ecut / 2.0 and Ediff - Ediff_old < min(self.dE_min * 100.0,0.01))):
                        rm_list[tp].append(ik)
                        ik += 1
                        Ediff = abs(energies[tp][ik] - self.cbm_vbm[tp]["energy"])

                    # if Ediff>Ecut/5.0 and Ediff - Ediff_old < min(self.dE_min*10.0, 0.001):
                            # or \
                            # Ediff>Ecut/2.0 and Ediff - Ediff_old < min(self.dE_min*100.0, 0.01):
                        # rm_list[tp].append(ik)
                    Ediff_old = Ediff

            rm_list[tp] = list(set(rm_list[tp]))'''

        logging.debug("time to filter energies from ibz k-mesh: \n {}".format(time.time()-start_time))
        start_time = time.time()
        # this step is crucial in DOS normalization when poly_bands to cover the whole energy range in BZ
        if self.poly_bands:
            all_bands_energies = {"n": [], "p": []}
            for tp in ["p", "n"]:
                all_bands_energies[tp] = energies[tp]
                for ib in range(1, len(self.poly_bands)):
                    for ik in range(len(kpts[tp])):
                        energy, velocity, effective_m = get_poly_energy(
                            self._rec_lattice.get_cartesian_coords(kpts[ik]) / A_to_nm,
                            poly_bands=self.poly_bands, type=tp, ib=ib, bandgap=self.dft_gap + self.scissor)
                        all_bands_energies[tp].append(energy)
            self.dos_emin = min(all_bands_energies["p"])
            self.dos_emax = max(all_bands_energies["n"])

        # logging.debug("energies before removing k-points with off-energy:\n {}".format(energies))
        # remove energies that are out of range



        # print "n-rm_list"
        # print rm_list["n"]
        # print "p-rm_list"
        # print rm_list["p"]
        '''
        for tp in ["n", "p"]:
            # if tp in self.all_types:
            if True:
                kpts[tp] = list(np.delete(kpts[tp], rm_list[tp], axis=0))
                # energies[tp] = np.delete(energies[tp], rm_list[tp], axis=0)
            else: # in this case it doesn't matter if the k-mesh is loose
                kpts[tp] = list(np.delete(kpts[tp], rm_list["n"]+rm_list["p"], axis=0))
                # energies[tp] = np.delete(energies[tp], rm_list["n"]+rm_list["p"], axis=0)
            if len(kpts[tp]) > 10000:
                warnings.warn("Too desne of a {}-type k-mesh (nk={}!); AMSET will be slow!".format(tp, len(kpts[tp])))

            logging.info("number of {}-type ibz k-points AFTER ENERGY-FILTERING: {}".format(tp, len(kpts[tp])))'''

        # 2 lines debug printing
        # energies["n"].sort()
        # print "{}-type energies for ibz after filtering: \n {}".format("n", energies["n"])
        del energies, velocities, e_sort_idx

        # TODO-JF (long-term): adaptive mesh is a good idea but current implementation is useless, see if you can come up with better method after talking to me
        if self.adaptive_mesh:
            raise NotImplementedError("adaptive mesh has not yet been "
                                      "implemented, please check back later!")

        # TODO: remove anything with "weight" later if ended up not using weights at all!
        kweights = {tp: [1.0 for i in kpts[tp]] for tp in ["n", "p"]}




        # logging.debug("time to add the symmetrically equivalent k-points: \n {}".format(time.time() - start_time))
        # start_time = time.time()

        # actual initiation of the kgrid
        self.kgrid = {
            "n": {},
            "p": {}}

        self.num_bands = {"n": {}, "p": {}}

        for tp in ["n", "p"]:
            self.num_bands[tp] = self.cbm_vbm[tp]["included"]
            self.kgrid[tp]["kpoints"] = [kpts[tp] for ib in range(self.num_bands[tp])]
            self.kgrid[tp]["kweights"] = [kweights[tp] for ib in range(self.num_bands[tp])]
            # self.kgrid[tp]["kpoints"] = [[k for k in kpts] for ib in range(self.cbm_vbm[tp]["included"])]
            # self.kgrid[tp]["kweights"] = [[kw for kw in kweights] for ib in range(self.cbm_vbm[tp]["included"])]

        self.initialize_var("kgrid", ["energy", "a", "c", "norm(v)", "norm(k)"], "scalar", 0.0, is_nparray=False, c_T_idx=False)
        self.initialize_var("kgrid", ["velocity"], "vector", 0.0, is_nparray=False, c_T_idx=False)
        self.initialize_var("kgrid", ["effective mass"], "tensor", 0.0, is_nparray=False, c_T_idx=False)

        start_time = time.time()

        rm_idx_list = {"n": [[] for i in range(self.cbm_vbm["n"]["included"])],
                       "p": [[] for i in range(self.cbm_vbm["p"]["included"])]}
        # @albalu why are these variables initialized separately from the ones above?
        self.initialize_var("kgrid", ["old cartesian kpoints", "cartesian kpoints"], "vector", 0.0, is_nparray=False, c_T_idx=False)
        self.initialize_var("kgrid", ["norm(k)", "norm(actual_k)"], "scalar", 0.0, is_nparray=False, c_T_idx=False)

        logging.debug("The DFT gap right before calculating final energy values: {}".format(self.dft_gap))

        for i, tp in enumerate(["p", "n"]):
            self.cbm_vbm[tp]["cartesian k"] = self._rec_lattice.get_cartesian_coords(self.cbm_vbm[tp]["kpoint"])/A_to_nm
            self.cbm_vbm[tp]["all cartesian k"] = self.get_sym_eq_ks_in_first_BZ(self.cbm_vbm[tp]["kpoint"], cartesian=True)
            self.cbm_vbm[tp]["all cartesian k"] = remove_duplicate_kpoints(self.cbm_vbm[tp]["all cartesian k"])

            sgn = (-1) ** i
            for ib in range(self.cbm_vbm[tp]["included"]):
                self.kgrid[tp]["old cartesian kpoints"][ib] = self._rec_lattice.get_cartesian_coords(
                    self.kgrid[tp]["kpoints"][ib]) / A_to_nm

                # REMEMBER TO MAKE A COPY HERE OTHERWISE THEY CHANGE TOGETHER
                self.kgrid[tp]["cartesian kpoints"][ib] = np.array(self.kgrid[tp]["old cartesian kpoints"][ib])
                # [1/nm], these are PHYSICS convention k vectors (with a factor of 2 pi included)

                if self.parallel and not self.poly_bands:
                    results = Parallel(n_jobs=self.num_cores)(delayed(get_energy)(self.kgrid[tp]["kpoints"][ib][ik],
                             engre[i * self.cbm_vbm["p"]["included"] + ib], nwave, nsym, nstv, vec, vec2, out_vec2,
                             br_dir) for ik in range(len(self.kgrid[tp]["kpoints"][ib])))

                s_orbital, p_orbital = self.get_dft_orbitals(bidx=self.cbm_vbm[tp]["bidx"] - 1 - sgn * ib)
                orbitals = {"s": s_orbital, "p": p_orbital}
                fit_orbs = {orb: griddata(points=np.array(self.DFT_cartesian_kpts), values=np.array(orbitals[orb]),
                    xi=np.array(self.kgrid[tp]["old cartesian kpoints"][ib]), method='nearest') for orb in orbitals.keys()}

                # TODO-JF: the general function for calculating the energy, velocity and effective mass can b
                for ik in range(len(self.kgrid[tp]["kpoints"][ib])):

                    min_dist_ik = np.array([norm(ki - self.kgrid[tp]["old cartesian kpoints"][ib][ik]) for ki in\
                                           self.cbm_vbm[tp]["all cartesian k"]]).argmin()
                    self.kgrid[tp]["cartesian kpoints"][ib][ik] = self.kgrid[tp]["old cartesian kpoints"][ib][ik] - \
                                                                  self.cbm_vbm[tp]["all cartesian k"][min_dist_ik]


                    self.kgrid[tp]["norm(k)"][ib][ik] = norm(self.kgrid[tp]["cartesian kpoints"][ib][ik])
                    self.kgrid[tp]["norm(actual_k)"][ib][ik] = norm(self.kgrid[tp]["old cartesian kpoints"][ib][ik])

                    if not self.poly_bands:
                        if not self.parallel:
                            energy, de, dde = get_energy(
                                self.kgrid[tp]["kpoints"][ib][ik], engre[i * self.cbm_vbm["p"]["included"] + ib],
                                nwave, nsym, nstv, vec, vec2, out_vec2, br_dir=br_dir)
                            energy = energy * Ry_to_eV - sgn * self.scissor / 2.0
                            velocity = abs(de / hbar * A_to_m * m_to_cm * Ry_to_eV)  # to get v in cm/s
                            effective_mass = hbar ** 2 / (
                                dde * 4 * pi ** 2) / m_e / A_to_m ** 2 * e * Ry_to_eV  # m_tensor: the last part is unit conversion
                        else:
                            energy = results[ik][0] * Ry_to_eV - sgn * self.scissor / 2.0
                            velocity = abs(results[ik][1] / hbar * A_to_m * m_to_cm * Ry_to_eV)
                            effective_mass = hbar ** 2 / (
                                results[ik][
                                    2] * 4 * pi ** 2) / m_e / A_to_m ** 2 * e * Ry_to_eV  # m_tensor: the last part is unit conversion

                    else:
                        energy, velocity, effective_mass = get_poly_energy(self.kgrid[tp]["old cartesian kpoints"][ib][ik],
                                                                           poly_bands=self.poly_bands,
                                                                           type=tp, ib=ib,
                                                                           bandgap=self.dft_gap + self.scissor)

                    self.kgrid[tp]["energy"][ib][ik] = energy
                    self.kgrid[tp]["velocity"][ib][ik] = velocity
                    self.kgrid[tp]["norm(v)"][ib][ik] = norm(velocity)

                    # self.kgrid[tp]["velocity"][ib][ik] = de/hbar * A_to_m * m_to_cm * Ry_to_eV # to get v in units of cm/s
                    # TODO: what's the implication of negative group velocities? check later after scattering rates are calculated
                    # TODO: actually using abs() for group velocities mostly increase nu_II values at each energy
                    # TODO: should I have de*2*pi for the group velocity and dde*(2*pi)**2 for effective mass?
                    if self.kgrid[tp]["velocity"][ib][ik][0] < self.v_min or  \
                                    self.kgrid[tp]["velocity"][ib][ik][1] < self.v_min \
                            or self.kgrid[tp]["velocity"][ib][ik][2] < self.v_min or \
                                    abs(self.kgrid[tp]["energy"][ib][ik] - self.cbm_vbm[tp]["energy"]) > self.Ecut[tp]:
                        rm_idx_list[tp][ib].append(ik)
                    # else:
                        # print "this point remains in {}-type: extrema, current energy. ib, ik: {}, {}".format(tp,ib,ik)
                        # , self.cbm_vbm[tp]["energy"], self.kgrid[tp]["energy"][ib][ik]

                    # TODO: AF must test how large norm(k) affect ACD, IMP and POP and see if the following is necessary
                    # if self.kgrid[tp]["norm(k)"][ib][ik] > 5:
                    #     rm_idx_list[tp][ib].append(ik)

                    self.kgrid[tp]["effective mass"][ib][ik] = effective_mass

                    if self.poly_bands:
                        self.kgrid[tp]["a"][ib][ik] = 1.0 # parabolic band s-orbital only
                        self.kgrid[tp]["c"][ib][ik] = 0.0
                    else:
                        self.kgrid[tp]["a"][ib][ik] = fit_orbs["s"][ik]/ (fit_orbs["s"][ik]**2 + fit_orbs["p"][ik]**2)**0.5
                        self.kgrid[tp]["c"][ib][ik] = (1 - self.kgrid[tp]["a"][ib][ik]**2)**0.5

            logging.debug("average of the {}-type group velocity in kgrid:\n {}".format(
                        tp, np.mean(self.kgrid[self.debug_tp]["velocity"][0], 0)))

        rearranged_props = ["velocity", "effective mass", "energy", "a", "c", "kpoints", "cartesian kpoints",
                            "old cartesian kpoints", "kweights",
                            "norm(v)", "norm(k)", "norm(actual_k)"]

        logging.debug("time to calculate E, v, m_eff at all k-points: \n {}".format(time.time()-start_time))
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

        # remove the k-points with off-energy values (>Ecut away from CBM/VBM) that are not removed already
        self.remove_indexes(rm_idx_list, rearranged_props=rearranged_props)

        logging.debug("dos_emin = {} and dos_emax= {}".format(self.dos_emin, self.dos_emax))

        for tp in ["n", "p"]:
            for ib in range(len(self.kgrid[tp]["energy"])):
                logging.info("Final # of {}-kpts in band #{}: {}".format(tp, ib, len(self.kgrid[tp]["kpoints"][ib])))

            if len(self.kgrid[tp]["kpoints"][0]) < 5:
                raise ValueError("VERY BAD {}-type k-mesh; please change the k-mesh and try again!".format(tp))

        logging.debug("time to calculate energy, velocity, m* for all: {} seconds".format(time.time() - start_time))

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
                # TODO: change how W_POP is set, user set a number or a file that can be fitted and inserted to kgrid
                self.kgrid[tp]["W_POP"][ib] = [self.W_POP for i in range(len(self.kgrid[tp]["kpoints"][ib]))]
                for c in self.dopings:
                    for T in self.temperatures:
                        self.kgrid[tp]["N_POP"][c][T][ib] = np.array(
                            [1 / (np.exp(hbar * W_POP / (k_B * T)) - 1) for W_POP in self.kgrid[tp]["W_POP"][ib]])

        self.initialize_var(grid="kgrid", names=["_all_elastic", "S_i", "S_i_th", "S_o", "S_o_th", "g", "g_th", "g_POP",
                                                 "f", "f_th", "relaxation time", "df0dk", "electric force",
                                                 "thermal force"],
                            val_type="vector", initval=self.gs, is_nparray=True, c_T_idx=True)

        self.initialize_var("kgrid", ["f0", "f_plus", "f_minus", "g_plus", "g_minus"], "vector", self.gs,
                            is_nparray=True, c_T_idx=True)
        # self.initialize_var("kgrid", ["lambda_i_plus", "lambda_i_minus"]
        #                     , "vector", self.gs, is_nparray=True, c_T_idx=False)


        # calculation of the density of states (DOS)
        if not self.poly_bands:
            emesh, dos, dos_nbands, bmin = analytical_bands.get_dos_from_scratch(self._vrun.final_structure,
                                                                           [self.nkdos, self.nkdos, self.nkdos],
                                                                           self.dos_emin, self.dos_emax,
                                                                           int(round(
                                                                               (self.dos_emax - self.dos_emin) / max(
                                                                                   self.dE_min, 0.0001))),
                                                                           width=self.dos_bwidth, scissor=self.scissor,
                                                                           vbmidx=self.cbm_vbm["p"]["bidx"])
            logging.debug("dos_nbands: {} \n".format(dos_nbands))
            self.dos_normalization_factor = dos_nbands if self.soc else dos_nbands * 2
            # self.dos_normalization_factor = self.nbands*2 if not self.soc else self.nbands

            self.dos_start = min(self._vrun.get_band_structure().as_dict()["bands"]["1"][bmin]) \
                             + self.offset_from_vrun - self.scissor/2.0
            self.dos_end = max(self._vrun.get_band_structure().as_dict()["bands"]["1"][bmin+dos_nbands]) \
                           + self.offset_from_vrun + self.scissor / 2.0
        else:
            logging.debug("here self.poly_bands: \n {}".format(self.poly_bands))
            emesh, dos = get_dos_from_poly_bands(self._vrun.final_structure, self._rec_lattice,
                                                 [self.nkdos, self.nkdos, self.nkdos], self.dos_emin, self.dos_emax,
                                                 int(round(
                                                     (self.dos_emax - self.dos_emin) / max(self.dE_min, 0.0001))),
                                                 poly_bands=self.poly_bands,
                                                 bandgap=self.cbm_vbm["n"]["energy"] - self.cbm_vbm["p"][
                                                     "energy"],  # we include here the actual or after-scissor gap here
                                                 width=self.dos_bwidth, SPB_DOS=False)
            self.dos_normalization_factor = len(
                self.poly_bands) * 2 * 2  # it is *2 elec/band & *2 because DOS is repeated in valence/conduction
            self.dos_start = self.dos_emin
            self.dos_end = self.dos_emax


        print("DOS normalization factor: {}".format(self.dos_normalization_factor))
        # print("The actual emsh used for dos normalization: {}".format(emesh))
        # print("The actual dos: {}".format(dos))

        integ = 0.0
        # for idos in range(len(dos) - 2):

        # here is the dos normalization story: to normalize DOS we first calculate the integral of the following two
        # energy ranges (basically the min and max of the original energy range) and normalize it based on the DOS
        # that is generated for a limited number of bands.

        self.dos_start = abs(emesh - self.dos_start).argmin()
        self.dos_end = abs(emesh - self.dos_end).argmin()

        # self.dos_start = 0
        # self.dos_end = len(dos) - 1
        for idos in range(self.dos_start, self.dos_end):
            # if emesh[idos] > self.cbm_vbm["n"]["energy"]: # we assume anything below CBM as 0 occupation
            #     break
            integ += (dos[idos + 1] + dos[idos]) / 2 * (emesh[idos + 1] - emesh[idos])

        print "dos integral from {} index to {}: {}".format(self.dos_start,  self.dos_end, integ)

        # normalize DOS
        # logging.debug("dos before normalization: \n {}".format(zip(emesh, dos)))
        dos = [g / integ * self.dos_normalization_factor for g in dos]
        # logging.debug("integral of dos: {} stoped at index {} and energy {}".format(integ, idos, emesh[idos]))

        self.dos = zip(emesh, dos)
        self.dos_emesh = np.array(emesh)
        self.vbm_dos_idx = self.get_Eidx_in_dos(self.cbm_vbm["p"]["energy"])
        self.cbm_dos_idx = self.get_Eidx_in_dos(self.cbm_vbm["n"]["energy"])

        print("vbm and cbm DOS index")
        print self.vbm_dos_idx
        print self.cbm_dos_idx
        # logging.debug("full dos after normalization: \n {}".format(self.dos))
        # logging.debug("dos after normalization from vbm idx to cbm idx: \n {}".format(self.dos[self.vbm_dos_idx-10:self.cbm_dos_idx+10]))

        self.dos = [list(a) for a in self.dos]

        logging.debug("time to finish the remaining part of init_kgrid: \n {}".format(time.time() - start_time))


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
                            E_change=0.0, forced_min_npoints=2, tolerance=self.dE_min)
                enforced_ratio = self.nforced_scat[tp] / sum([len(points) for points in self.kgrid[tp]["X_E_ik"][ib]])
                logging.info("enforced scattering ratio for {}-type elastic scattering at band {}:\n {}".format(
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
                                E_change= + hbar * self.kgrid[tp]["W_POP"][ib][ik],forced_min_npoints=2,
                                   tolerance=self.dE_min)
                        self.kgrid[tp]["X_Eminus_ik"][ib][ik] = self.get_X_ib_ik_near_new_E(tp, ib, ik,
                                E_change= - hbar * self.kgrid[tp]["W_POP"][ib][ik],forced_min_npoints=2,
                                        tolerance=self.dE_min)
                    enforced_ratio = self.nforced_scat[tp] / (
                        sum([len(points) for points in self.kgrid[tp]["X_Eplus_ik"][ib]]) + \
                        sum([len(points) for points in self.kgrid[tp]["X_Eminus_ik"][ib]]))
                    logging.info(
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

        fractional_ks = [np.dot(frac_k, self.rotations[i]) + self.translations[i] for i in range(len(self.rotations))]

        k = self.kgrid[tp]["kpoints"][ib][ik]
        seks = [self._rec_lattice.get_cartesian_coords(frac_k) / A_to_nm for frac_k in fractional_ks]

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


    def get_X_ib_ik_near_new_E(self, tp, ib, ik, E_change, forced_min_npoints=0, tolerance=0.01):
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
        E = self.kgrid[tp]["energy"][ib][ik]
        E_prm = E + E_change  # E_prm is E prime, the new energy
        k = self.kgrid[tp]["cartesian kpoints"][ib][ik]
        # we count the point itself; it does not result in self-scattering (due to 1-X term); however, it is necessary
        # to avoid zero scattering as in the integration each term is (X[i+1]-X[i])*(integrand[i]+integrand[i+1)/2
        result = [(1, ib, ik)]

        nk = len(self.kgrid[tp]["kpoints"][ib])

        for ib_prm in range(self.cbm_vbm[tp]["included"]):
            # this code is commented out because it is unnecessary unless it saves a lot of time
            # if ib==ib_prm and E_change==0.0:
            #    ik_closest_E = ik
            # else:
            ik_closest_E = np.abs(self.kgrid[tp]["energy"][ib_prm] - E_prm).argmin()

            for step, start in [(1, 0), (-1, -1)]:
                ik_prm = ik_closest_E + start  # go up from ik_closest_E, down from ik_closest_E - 1
                while ik_prm >= 0 and ik_prm < nk and abs(self.kgrid[tp]["energy"][ib_prm][ik_prm] - E_prm) < tolerance:
                    X_ib_ik = (cos_angle(k, self.kgrid[tp]["cartesian kpoints"][ib_prm][ik_prm]), ib_prm, ik_prm)
                    if (X_ib_ik[1], X_ib_ik[2]) not in [(entry[1], entry[2]) for entry in result]:
                        result.append(X_ib_ik)
                    ik_prm += step

        # If fewer than forced_min_npoints number of points were found, just return a few surroundings of the same band
        ib_prm = ib
        # if E_change == 0.0:
        #    ik_closest_E = ik
        # else:
        ik_closest_E = np.abs(self.kgrid[tp]["energy"][ib_prm] - E_prm).argmin()

        for step, start in [(1, 0), (-1, -1)]:
            # step -1 is in case we reached the end (ik_prm == nk - 1); then we choose from the lower energy k-points
            ik_prm = ik_closest_E + start  # go up from ik_closest_E, down from ik_closest_E - 1
            while ik_prm >= 0 and ik_prm < nk and len(result) - 1 < forced_min_npoints:
                # add all the k-points that have the same energy as E_prime E(k_pm); these values are stored in X_E_ik
                # @albalu isn't this the function that is used to generate self.kgrid[tp]["X_E_ik"]? How will there already be something in self.kgrid[tp]["X_E_ik"] at this point?
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



    # points_1d now a dictionary with 'x', 'y', and 'z' lists of points
    # points_1d lists do not need to be sorted
    def create_grid(self, points_1d):
        for dir in ['x', 'y', 'z']:
            points_1d[dir].sort()
        grid = np.zeros((len(points_1d['x']), len(points_1d['y']), len(points_1d['z']), 3))
        for i, x in enumerate(points_1d['x']):
            for j, y in enumerate(points_1d['y']):
                for k, z in enumerate(points_1d['z']):
                    grid[i, j, k, :] = np.array([x, y, z])
        return grid


    # grid is a 4d numpy array, where last dimension is vectors in a 3d grid specifying fractional position in BZ
    def array_to_kgrid(self, grid):
        kgrid = []
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                for k in range(grid.shape[2]):
                    kgrid.append(grid[i,j,k])
        return kgrid


    def grid_index_from_list_index(self, list_index):
        N = self.kgrid_array['k_points'].shape
        count = list_index
        i, j, k = (0,0,0)
        while count >= N[2]*N[1]:
            count -= N[2]*N[1]
            i += 1
        while count >= N[2]:
            count -= N[2]
            j += 1
        k = count
        return (i,j,k)


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
    def integrate_over_k(self, func_grid):#, xDOS=False, xvel=False, weighted=True):
        '''
        :return: result of the integral
        '''

        # in the interest of not prematurely optimizing, func_grid must be a perfect grid: the only deviation from
        # the cartesian coordinate system can be uniform stretches, as in the distance between adjacent planes of points
        # can be any value, but no points can be missing from the next plane

        # in this case the format of fractional_grid is a 4d grid
        # the last dimension is a vector of the k point fractional coordinates
        # the dv grid is 3d and the indexes correspond to those of func_grid

        if func_grid.ndim == 3:
            return np.sum(func_grid * self.dv_grid)
        return [np.sum(func_grid[:,:,:,i] * self.dv_grid) for i in range(func_grid.shape[3])]



    def integrate_over_BZ(self, prop_list, tp, c, T, xDOS=False, xvel=False, weighted=True):

        weighted = False

        """

        :param tp:
        :param c:
        :param T:
        :param distribution (str): can be switched between f, f0, g, g_POP, etc
        :param xvel:
        :return:
        """
        wpower = 1
        if xvel:
            wpower += 1
        integral = np.array([self.gs, self.gs, self.gs])
        for ie in range(len(self.egrid[tp]["energy"]) - 1):
            dE = abs(self.egrid[tp]["energy"][ie + 1] - self.egrid[tp]["energy"][ie])
            sum_over_k = np.array([self.gs, self.gs, self.gs])
            for ib, ik in self.kgrid_to_egrid_idx[tp][ie]:
                k_nrm = self.kgrid[tp]["norm(k)"][ib][ik]
                # k_nrm = norm(self.kgrid[tp]["old cartesian kpoints"][ib][ik])

                # 4*pi, hbar and norm(v) are coming from the conversion of dk to dE
                product = k_nrm ** 2 / self.kgrid[tp]["norm(v)"][ib][ik] * 4 * pi / hbar
                # product = 1.0
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
            # if not weighted:
            #     sum_over_k /= len(self.kgrid_to_egrid_idx[tp][ie])
            if xDOS:
                sum_over_k *= self.egrid[tp]["DOS"][ie]
            if weighted:
            #     sum_over_k *= self.Efrequency[tp][ie] ** (wpower)
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
            # normk_sorted_idx = np.argsort([norm(k) for k in self.kgrid[tp]["old cartesian kpoints"][ib]])
            normk_sorted_idx = np.argsort(self.kgrid[tp][normk_tp][ib])
            diff = [0.0 for prop in prop_list]


            for j, ik in enumerate(normk_sorted_idx[:-1]):
                ik_next = normk_sorted_idx[j+1]
                normk = self.kgrid[tp][normk_tp][ib][ik]
                dk = (self.kgrid[tp][normk_tp][ib][ik_next] - normk)/interpolation_nsteps
                if dk == 0.0:
                    continue
                # print normk
                # print dk
                if xDOS:
                    dS = ((self.kgrid[tp][normk_tp][ib][ik_next]/pi)**2 - \
                         (self.kgrid[tp][normk_tp][ib][ik]/pi)**2)/interpolation_nsteps
                for j, p in enumerate(prop_list):
                    if p[0] == "/":
                        diff[j] = (self.kgrid[tp][p.split("/")[-1]][c][T][ib][ik_next] - \
                                        self.kgrid[tp][p.split("/")[-1]][c][T][ib][ik]) / interpolation_nsteps
                    elif p[0] == "1":
                        diff[j] = ((1 - self.kgrid[tp][p.split("-")[-1].replace(" ", "")][c][T][ib][ik_next]) - \
                                  (1 - self.kgrid[tp][p.split("-")[-1].replace(" ", "")][c][T][ib][ik])) / interpolation_nsteps
                    else:
                        diff[j] = (self.kgrid[tp][p][c][T][ib][ik_next] - self.kgrid[tp][p][c][T][ib][ik]) / interpolation_nsteps
                    # product *= (self.kgrid[tp][p][c][T][ib][ik+1] + self.kgrid[tp][p][c][T][ib][ik])/2


                for i in range(interpolation_nsteps):
                    multi = dk
                    for j, p in enumerate(prop_list):
                        if p[0] == "/":
                            multi /= self.kgrid[tp][p.split("/")[-1]][c][T][ib][ik] + diff[j] * i
                        elif "1" in p:
                            multi *= 1 - self.kgrid[tp][p.split("-")[-1].replace(" ", "")][c][T][ib][ik] + diff[j] * i
                        else:
                            multi *= self.kgrid[tp][p][c][T][ib][ik] + diff[j] * i
                    if xDOS:
                        multi *= (self.kgrid[tp][normk_tp][ib][ik]/pi)**2 + dS * i
                    integral += multi

        # print "sorted cartesian kpoints for {}-type: {}".format(tp,[self.kgrid[tp]["old cartesian kpoints"][ib][ik] for ik in normk_sorted_idx])
        # print "sorted cartesian kpoints for {}-type: {}".format(tp,[self.kgrid[tp]["norm(actual_k)"][ib][ik] for ik in normk_sorted_idx])
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
                    # integral += multi * self.Efrequency[tp][ie]**wpower * (-(dfdE + ddfdE))
                    # integral += multi * self.Efrequency[tp][ie]**wpower *dfdE
                    # integral += multi * self.Efrequency[tp][ie]**wpower * self.egrid[tp]["f0"][c][T][ie]
                    # integral += multi * self.Efrequency[tp][ie] ** wpower
                    integral += multi * (self.Efrequency[tp][ie] / float(self.sym_freq[tp][ie]) + dweight * i)
                else:
                    integral += multi
        if weighted:
            return integral
            # return integral/(sum(self.Efrequency[tp][:-1]))

        else:
            return integral



    def integrate_over_X(self, tp, X_E_index, integrand, ib, ik, c, T, sname=None, g_suffix=""):
        """integrate numerically with a simple trapezoidal algorithm."""
        summation = np.array([0.0, 0.0, 0.0])
        if len(X_E_index[ib][ik]) == 0:
            raise ValueError("enforcing scattering points did NOT work, {}[{}][{}] is empty".format(X_E_index, ib, ik))
            # return summation
        X, ib_prm, ik_prm = X_E_index[ib][ik][0]
        current_integrand = integrand(tp, c, T, ib, ik, ib_prm, ik_prm, X, sname=sname, g_suffix=g_suffix)
        for i in range(len(X_E_index[ib][ik]) - 1):
            DeltaX = X_E_index[ib][ik][i + 1][0] - X_E_index[ib][ik][i][0]
            if DeltaX == 0.0:
                continue

            X, ib_prm, ik_prm = X_E_index[ib][ik][i + 1]

            dum = current_integrand / 2.0

            current_integrand = integrand(tp, c, T, ib, ik, ib_prm, ik_prm, X, sname=sname, g_suffix=g_suffix)

            # This condition is to exclude self-scattering from the integration
            if np.sum(current_integrand) == 0.0:
                dum *= 2
            elif np.sum(dum) == 0.0:
                dum = current_integrand
            else:
                dum += current_integrand / 2.0

            summation += dum * DeltaX  # In case of two points with the same X, DeltaX==0 so no duplicates
        return summation



    def el_integrand_X(self, tp, c, T, ib, ik, ib_prm, ik_prm, X, sname=None, g_suffix=""):

        # The following (if passed on to s_el_eq) result in many cases k and k_prm being equal which we don't want.
        # k = m_e * self._avg_eff_mass[tp] * self.kgrid[tp]["norm(v)"][ib][ik] / (hbar * e * 1e11)
        # k_prm = m_e * self._avg_eff_mass[tp] * self.kgrid[tp]["normv"][ib_prm][ik_prm] / (hbar * e * 1e11)

        k = self.kgrid[tp]["cartesian kpoints"][ib][ik]
        k_prm = self.kgrid[tp]["cartesian kpoints"][ib_prm][ik_prm]

        if k[0] == k_prm[0] and k[1] == k_prm[1] and k[2] == k_prm[2]:
            return np.array(
                [0.0, 0.0, 0.0])  # self-scattering is not defined;regardless, the returned integrand must be a vector


        return (1 - X) * norm(k_prm) ** 2 * self.s_el_eq(sname, tp, c, T, k, k_prm) \
               * self.G(tp, ib, ik, ib_prm, ik_prm, X) / (self.kgrid[tp]["norm(v)"][ib_prm][ik_prm] / sq3)



    def inel_integrand_X(self, tp, c, T, ib, ik, ib_prm, ik_prm, X, sname=None, g_suffix=""):
        """
        returns the evaluated number (float) of the expression inside the S_o and S_i(g) integrals.
        :param tp (str): "n" or "p" type
        :param c (float): carrier concentration/doping in cm**-3
        :param T:
        :param ib:
        :param ik:
        :param ib_prm:
        :param ik_prm:
        :param X:
        :param alpha:
        :param sname:
        :return:
        """
        k = self.kgrid[tp]["cartesian kpoints"][ib][ik]
        f = self.kgrid[tp]["f"][c][T][ib][ik]
        f_th = self.kgrid[tp]["f_th"][c][T][ib][ik]
        k_prm = self.kgrid[tp]["cartesian kpoints"][ib_prm][ik_prm]

        v_prm = self.kgrid[tp]["velocity"][ib_prm][ik_prm]
        if tp == "n":
            f_prm = self.kgrid[tp]["f"][c][T][ib_prm][ik_prm]
        else:
            f_prm = 1 - self.kgrid[tp]["f"][c][T][ib_prm][ik_prm]

        if k[0] == k_prm[0] and k[1] == k_prm[1] and k[2] == k_prm[2]:
            return np.array(
            [0.0, 0.0, 0.0])  # self-scattering is not defined;regardless, the returned integrand must be a vector
        #fermi = self.egrid["fermi"][c][T]
        fermi = self.fermi_level[c][T]

        # test
        # f = self.f(self.kgrid[tp]["energy"][ib][ik], fermi, T, tp, c, alpha)
        # f_prm = self.f(self.kgrid[tp]["energy"][ib_prm][ik_prm], fermi, T, tp, c, alpha)

        N_POP = 1 / (np.exp(hbar * self.kgrid[tp]["W_POP"][ib][ik] / (k_B * T)) - 1)
        # norm_diff = max(norm(k-k_prm), 1e-10)
        norm_diff = norm(k - k_prm)
        # print norm(k_prm)**2
        # the term norm(k_prm)**2 is wrong in practice as it can be too big and originally we integrate |k'| from 0
        integ = self.kgrid[tp]["norm(k)"][ib_prm][ik_prm]**2*self.G(tp, ib, ik, ib_prm, ik_prm, X)/\
                (self.kgrid[tp]["norm(v)"][ib_prm][ik_prm]*norm_diff**2)



        if "S_i" in sname:
            integ *= abs(X * self.kgrid[tp]["g" + g_suffix][c][T][ib][ik])
            # integ *= X*self.kgrid[tp]["g" + g_suffix][c][T][ib][ik][alpha]
            if "minus" in sname:
                if tp == "p" or (tp == "n" and \
                    self.kgrid[tp]["energy"][ib][ik]-hbar*self.kgrid[tp]["W_POP"][ib][ik]>=self.cbm_vbm[tp]["energy"]):
                    integ *= (1 - f) * N_POP + f * (1 + N_POP)
            elif "plus" in sname:
                if tp == "n" or (tp == "p" and \
                    self.kgrid[tp]["energy"][ib][ik]+hbar*self.kgrid[tp]["W_POP"][ib][ik]<=self.cbm_vbm[tp]["energy"]):
                    integ *= (1 - f) * (1 + N_POP) + f * N_POP
            else:
                raise ValueError('"plus" or "minus" must be in sname for phonon absorption and emission respectively')
        elif "S_o" in sname:
            if "minus" in sname:
                if tp == "p" or (tp=="n" and \
                    self.kgrid[tp]["energy"][ib][ik]-hbar*self.kgrid[tp]["W_POP"][ib][ik]>=self.cbm_vbm[tp]["energy"]):
                    integ *= (1 - f_prm) * (1 + N_POP) + f_prm * N_POP
            elif "plus" in sname:
                if tp == "n" or (tp == "p" and \
                    self.kgrid[tp]["energy"][ib][ik]+hbar*self.kgrid[tp]["W_POP"][ib][ik]<=self.cbm_vbm[tp]["energy"]):
                    integ *= (1 - f_prm) * N_POP + f_prm * (1 + N_POP)
            else:
                raise ValueError('"plus" or "minus" must be in sname for phonon absorption and emission respectively')
        else:
            raise ValueError("The inelastic scattering name: {} is NOT supported".format(sname))
        return integ



    def s_inel_eq_isotropic(self, once_called=False):
        for tp in ["n", "p"]:
            for c in self.dopings:
                for T in self.temperatures:
                    for ib in range(len(self.kgrid[tp]["energy"])):
                        # only when very large # of k-points are present, make sense to parallelize as this function
                        # has become fast after better energy window selection
                        if self.parallel and len(self.kgrid[tp]["size"]) * max(self.kgrid[tp]["size"]) > 100000:
                            # if False:
                            results = Parallel(n_jobs=self.num_cores)(delayed(calculate_Sio) \
                                                                          (tp, c, T, ib, ik, once_called, self.kgrid,
                                                                           self.cbm_vbm, self.epsilon_s,
                                                                           self.epsilon_inf
                                                                           ) for ik in
                                                                      range(len(self.kgrid[tp]["kpoints"][ib])))
                        else:
                            results = [calculate_Sio(tp, c, T, ib, ik, once_called, self.kgrid, self.cbm_vbm,
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
                            # self.kgrid[tp][sname][c][T][ib][ik] = abs(summation) * e**2*self.kgrid[tp]["W_POP"][ib][ik]/(4*pi*hbar) \
                            self.kgrid[tp][sname][c][T][ib][ik] = summation * e ** 2 * self.kgrid[tp]["W_POP"][ib][ik] \
                                                                  / (4 * pi * hbar) * (
                                                                  1 / self.epsilon_inf - 1 / self.epsilon_s) / epsilon_0 * 100 / e
                            # if norm(self.kgrid[tp][sname][c][T][ib][ik]) < 1:
                            #     self.kgrid[tp][sname][c][T][ib][ik] = [1, 1, 1]
                            # if norm(self.kgrid[tp][sname][c][T][ib][ik]) > 1e5:
                            #     print tp, c, T, ik, ib, summation, self.kgrid[tp][sname][c][T][ib][ik]



    def s_el_eq_isotropic(self, sname, tp, c, T, ib, ik):
        """returns elastic scattering rate (a numpy vector) at certain point (e.g. k-point, T, etc)
        with the assumption that the band structure is isotropic (i.e. self.bs_is_isotropic==True).
        This assumption significantly simplifies the model and the integrated rates at each
        k/energy directly extracted from the literature can be used here."""

        v = self.kgrid[tp]["norm(v)"][ib][ik] / sq3  # because of isotropic assumption, we treat the BS as 1D
        # v = self.kgrid[tp]["velocity"][ib][ik] # because it's isotropic, it doesn't matter which one we choose
        # perhaps more correct way of defining knrm is as follows since at momentum is supposed to be proportional to
        # velocity as it is in free-electron formulation so we replaced hbar*knrm with m_e*v/(1e11*e) (momentum)


        # if self.poly_bands: # the first one should be v and NOT v * sq3 so that the two match in SPB
        # if False:  # I'm 90% sure that there is not need for the first type of knrm and that's why I added if False for now
        #     knrm = m_e * self._avg_eff_mass[tp] * (v) / (
        #     hbar * e * 1e11)  # in nm given that v is in cm/s and hbar in eV.s; this resulted in very high ACD and IMP scattering rates, actually only PIE would match with aMoBT results as it doesn't have k_nrm in its formula
        ##TODO: make sure that ACD scattering as well as others match in SPB between bs_is_isotropic and when knrm is the following and not above (i.e. not m*v/hbar*e)
        # else:
        knrm = self.kgrid[tp]["norm(k)"][ib][ik]
        par_c = self.kgrid[tp]["c"][ib][ik]

        if sname.upper() == "ACD":
            # The following two lines are from Rode's chapter (page 38)
            return (k_B * T * self.E_D[tp] ** 2 * knrm ** 2) / (3 * pi * hbar ** 2 * self.C_el * 1e9 * v) \
                   * (3 - 8 * par_c ** 2 + 6 * par_c ** 4) * e * 1e20

            # return (k_B * T * self.E_D[tp] ** 2 * knrm ** 2) *norm(1.0/v)/ (3 * pi * hbar ** 2 * self.C_el * 1e9) \
            #     * (3 - 8 * self.kgrid[tp]["c"][ib][ik] ** 2 + 6 * self.kgrid[tp]["c"][ib][ik] ** 4) * e * 1e20

            # it is equivalent to the following also from Rode but always isotropic
            # return m_e * knrm * self.E_D[tp] ** 2 * k_B * T / ( 3* pi * hbar ** 3 * self.C_el) \
            #            * (3 - 8 * par_c ** 2 + 6 * par_c ** 4) * 1  # units work out! that's why conversion is 1


            # The following is from Deformation potentials and... Ref. [Q] (DOI: 10.1103/PhysRev.80.72 ) page 82?
            # if knrm < 1/(0.1*self._vrun.lattice.c*A_to_nm):

            # replaced hbar*knrm with m_e*norm(v)/(1e11*e) which is momentum
            # return m_e * m_e*v * self.E_D[tp] ** 2 * k_B * T / (3 * pi * hbar ** 4 * self.C_el) \
            #        * (3 - 8 * par_c ** 2 + 6 * par_c ** 4) / (1e11*e) # 1/1e11*e is to convert kg.cm/s to hbar.k units (i.e. ev.s/nm)

        elif sname.upper() == "IMP":  # double-checked the units and equation on 5/12/2017
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
        the scattering rate equation for each elastic scattering name is entered in s_func and returned the integrated
        scattering rate.
        :param sname (st): the name of the tp of elastic scattering, options are 'IMP', 'ADE', 'PIE', 'POP', 'DIS'
        :param s_func:
        :return:
        """
        sname = sname.upper()

        for tp in ["n", "p"]:
            self.egrid[tp][sname] = {c: {T: np.array([[0.0, 0.0, 0.0] for i in
                                                      range(len(self.egrid[tp]["energy"]))]) for T in
                                         self.temperatures} for c in self.dopings}
            self.kgrid[tp][sname] = {
            c: {T: np.array([[[0.0, 0.0, 0.0] for i in range(len(self.kgrid[tp]["kpoints"][j]))]
                             for j in range(self.cbm_vbm[tp]["included"])]) for T in self.temperatures} for c in
            self.dopings}
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
                                    print "WARNING!!! here scattering {} < 1".format(sname)
                                    # if self.kgrid[tp]["df0dk"][c][T][ib][ik][0] > 1e-32:
                                    #     print self.kgrid[tp]["df0dk"][c][T][ib][ik]
                                    print self.kgrid[tp]["X_E_ik"][ib][ik]

                                    self.kgrid[tp][sname][c][T][ib][ik] = [1e10, 1e10, 1e10]

                                if norm(self.kgrid[tp][sname][c][T][ib][ik]) > 1e20:
                                    print "WARNING!!! TOO LARGE of scattering rate for {}:".format(sname)
                                    print summation
                                    print self.kgrid[tp]["X_E_ik"][ib][ik]
                                    print
                            self.kgrid[tp]["_all_elastic"][c][T][ib][ik] += self.kgrid[tp][sname][c][T][ib][ik]

                        # logging.debug("relaxation time at c={} and T= {}: \n {}".format(c, T, self.kgrid[tp]["relaxation time"][c][T][ib]))
                        # logging.debug("_all_elastic c={} and T= {}: \n {}".format(c, T, self.kgrid[tp]["_all_elastic"][c][T][ib]))
                        self.kgrid[tp]["relaxation time"][c][T][ib] = 1 / self.kgrid[tp]["_all_elastic"][c][T][ib]



    def map_to_egrid(self, prop_name, c_and_T_idx=True, prop_type="vector"):
        """
        maps a propery from kgrid to egrid conserving the nomenclature. The mapped property should have the
            kgrid[tp][prop_name][c][T][ib][ik] data structure and will have egrid[tp][prop_name][c][T][ie] structure
        :param prop_name (string): the name of the property to be mapped. It must be available in the kgrid.
        :param c_and_T_idx (bool): if True, the propetry will be calculated and maped at each concentration, c, and T
        :param prop_type (str): options are "scalar", "vector", "tensor"
        :return:
        """
        # scalar_properties = ["g"]
        if not c_and_T_idx:
            self.initialize_var("egrid", prop_name, prop_type, initval=self.gs, is_nparray=True, c_T_idx=False)
            for tp in ["n", "p"]:

                if not self.gaussian_broadening:
                    for ie, en in enumerate(self.egrid[tp]["energy"]):
                        first_ib = self.kgrid_to_egrid_idx[tp][ie][0][0]
                        first_ik = self.kgrid_to_egrid_idx[tp][ie][0][1]
                        for ib, ik in self.kgrid_to_egrid_idx[tp][ie]:
                            # if norm(self.kgrid[tp][prop_name][ib][ik]) / norm(self.kgrid[tp][prop_name][first_ib][first_ik]) > 1.25 or norm(self.kgrid[tp][prop_name][ib][ik]) / norm(self.kgrid[tp][prop_name][first_ib][first_ik]) < 0.8:
                            #     logging.debug('ERROR! Some {} values are more than 25% different at k points with the same energy.'.format(prop_name))
                            #     print('first k: {}, current k: {}'.format(norm(self.kgrid[tp][prop_name][first_ib][first_ik]), norm(self.kgrid[tp][prop_name][ib][ik])))
                            #     print('current energy, first energy, ik, first_ik')
                            #     print(self.kgrid[tp]['energy'][ib][ik], self.kgrid[tp]['energy'][first_ib][first_ik], ik, first_ik)
                            if self.bs_is_isotropic and prop_type == "vector":
                                self.egrid[tp][prop_name][ie] += norm(self.kgrid[tp][prop_name][ib][ik]) / sq3
                            else:
                                self.egrid[tp][prop_name][ie] += self.kgrid[tp][prop_name][ib][ik]
                        self.egrid[tp][prop_name][ie] /= len(self.kgrid_to_egrid_idx[tp][ie])

                        # if self.bs_is_isotropic and prop_type=="vector":
                        #     self.egrid[tp][prop_name][ie]=np.array([norm(self.egrid[tp][prop_name][ie])/sq3 for i in range(3)])


                else:
                    raise ValueError(
                        "Guassian Broadening is NOT well tested and abandanded at the begining due to inaccurate results")
                    # for ie, en in enumerate(self.egrid[tp]["energy"]):
                    #     N = 0.0  # total number of instances with the same energy
                    #     for ib in range(self.cbm_vbm[tp]["included"]):
                    #         for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                    #             self.egrid[tp][prop_name][ie] += self.kgrid[tp][prop_name][ib][ik] * \
                    #                 GB(self.kgrid[tp]["energy"][ib][ik]-self.egrid[tp]["energy"][ie], 0.005)
                    #
                    #     self.egrid[tp][prop_name][ie] /= self.cbm_vbm[tp]["included"] * len(self.kgrid[tp]["kpoints"][0])
                    #
                    #     if self.bs_is_isotropic and prop_type=="vector":
                    #         self.egrid[tp][prop_name][ie]=np.array([norm(self.egrid[tp][prop_name][ie])/sq3 for i in range(3)])


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
                                    # if norm(self.kgrid[tp][prop_name][c][T][ib][ik]) / norm(
                                    #         self.kgrid[tp][prop_name][c][T][first_ib][first_ik]) > 1.25 or norm(
                                    #         self.kgrid[tp][prop_name][c][T][ib][ik]) / norm(
                                    #         self.kgrid[tp][prop_name][c][T][first_ib][first_ik]) < 0.8:
                                    #     logging.debug('ERROR! Some {} values are more than 25% different at k points with the same energy.'.format(prop_name))
                                    #     print('first k: {}, current k: {}'.format(
                                    #         norm(self.kgrid[tp][prop_name][c][T][first_ib][first_ik]),
                                    #         norm(self.kgrid[tp][prop_name][c][T][ib][ik])))

                                    if self.bs_is_isotropic and prop_type == "vector":
                                        self.egrid[tp][prop_name][c][T][ie] += norm(
                                            self.kgrid[tp][prop_name][c][T][ib][ik]) / sq3
                                    else:
                                        self.egrid[tp][prop_name][c][T][ie] += self.kgrid[tp][prop_name][c][T][ib][ik]
                                self.egrid[tp][prop_name][c][T][ie] /= len(self.kgrid_to_egrid_idx[tp][ie])

                                # if self.bs_is_isotropic and prop_type == "vector":
                                #     self.egrid[tp][prop_name][c][T][ie] = np.array(
                                #         [norm(self.egrid[tp][prop_name][c][T][ie])/sq3 for i in range(3)])

                            # df0dk must be negative but we used norm for df0dk when isotropic
                            if prop_name in ["df0dk"] and self.bs_is_isotropic:
                                self.egrid[tp][prop_name][c][T] *= -1
                else:
                    raise ValueError(
                        "Guassian Broadening is NOT well tested and abandanded at the begining due to inaccurate results")
                    # for c in self.dopings:
                    #     for T in self.temperatures:
                    #         for ie, en in enumerate(self.egrid[tp]["energy"]):
                    #             N = 0.0 # total number of instances with the same energy
                    #             for ib in range(self.cbm_vbm[tp]["included"]):
                    #                 for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                    #                     self.egrid[tp][prop_name][c][T][ie] += self.kgrid[tp][prop_name][c][T][ib][ik] * \
                    #                            GB(self.kgrid[tp]["energy"][ib][ik] -
                    #                                                         self.egrid[tp]["energy"][ie], 0.005)
                    #             self.egrid[tp][prop_name][c][T][ie] /= self.cbm_vbm[tp]["included"] * len(self.kgrid[tp]["kpoints"][0])
                    #
                    #
                    #             if self.bs_is_isotropic and prop_type == "vector":
                    #                 self.egrid[tp][prop_name][c][T][ie] = np.array(
                    #                     [norm(self.egrid[tp][prop_name][c][T][ie])/sq3 for i in range(3)])
                    #
                    #         if prop_name in ["df0dk"]: # df0dk is always negative
                    #             self.egrid[tp][c][T][prop_name] *= -1



    def find_fermi_SPB(self, c, T, tolerance=0.001, tolerance_loose=0.03, alpha=0.02, max_iter=1000):

        tp = self.get_tp(c)
        sgn = np.sign(c)
        m_eff = np.prod(self.cbm_vbm[tp]["eff_mass_xx"]) ** (1.0 / 3.0)
        c *= sgn
        initial_energy = self.cbm_vbm[tp]["energy"]
        fermi = initial_energy + 0.02
        iter = 0
        for iter in range(max_iter):
            calc_doping = 4 * pi * (2 * m_eff * m_e * k_B * T / hbar ** 2) ** 1.5 * fermi_integral(0.5, fermi, T,
                                                                                                   initial_energy) * 1e-6 / e ** 1.5
            fermi += alpha * sgn * (calc_doping - c) / abs(c + calc_doping) * fermi
            relative_error = abs(calc_doping - c) / abs(c)
            if relative_error <= tolerance:
                # This here assumes that the SPB generator set the VBM to 0.0 and CBM=  gap + scissor
                if sgn < 0:
                    return fermi
                else:
                    return -(fermi - initial_energy)
        if relative_error > tolerance:
            raise ValueError("could NOT find a corresponding SPB fermi level after {} itenrations".format(max_iter))




    def find_fermi_k(self, tolerance=0.001):

        closest_energy = {c: {T: None for T in self.temperatures} for c in self.dopings}
        #energy = self.array_from_kgrid('energy', 'n', fill=1000)
        for c in self.dopings:
            tp = self.get_tp(c)
            tol = tolerance * abs(c)
            for T in self.temperatures:
                step = 0.1
                range_of_energies = np.arange(self.cbm_vbm[tp]['energy'] - 2, self.cbm_vbm[tp]['energy'] + 2.1, step)
                diff = 1000 * abs(c)
                while(diff > tol):
                    # try a number for fermi level
                    diffs = {}
                    for e_f in range_of_energies:
                        # calculate distribution
                        f = 1 / (np.exp((self.energy_array[tp] - e_f) / (k_B * T)) + 1)
                        # see if it is close to concentration
                        if tp == 'n':
                            diffs[e_f] = abs(self.integrate_over_states(f)[0] - abs(c))
                        if tp == 'p':
                            diffs[e_f] = abs(self.integrate_over_states(1 - f)[0] - abs(c))
                    # compare all the numbers and zoom in on the closest
                    closest_energy[c][T] = min(diffs, key=diffs.get)
                    range_of_energies = np.arange(closest_energy[c][T] - step, closest_energy[c][T] + step, step / 10)
                    step /= 10
                    diff = diffs[closest_energy[c][T]]

        return closest_energy



    def find_fermi(self, c, T, tolerance=0.001, tolerance_loose=0.03, alpha=0.05, max_iter=5000):
        """
        To find the Fermi level at a carrier concentration and temperature at kgrid (i.e. band structure, DOS, etc)
        :param c (float): The doping concentration; c < 0 indicate n-tp (i.e. electrons) and c > 0 for p-tp
        :param T (float): The temperature.
        :param tolerance (0<float<1): convergance threshold for relative error
        :param tolerance_loose (0<float<1): maximum relative error allowed between the calculated and input c
        :param alpha (float < 1): the fraction of the linear interpolation towards the actual fermi at each iteration
        :param max_iter (int): after this many iterations the function returns even if it is not converged
        :return:
            The fitted/calculated Fermi level
        """

        # initialize parameters
        relative_error = self.gl
        iter = 0.0
        tune_alpha = 1.0
        temp_doping = {"n": -0.01, "p": +0.01}
        typ = self.get_tp(c)
        typj = ["n", "p"].index(typ)
        fermi = self.cbm_vbm[typ]["energy"] + 0.01 * (-1)**typj # addition is to ensure Fermi is not exactly 0.0
        # fermi = self.egrid[typ]["energy"][0]

        print("calculating the fermi level at temperature: {} K".format(T))
        funcs = [lambda E, fermi0, T: f0(E, fermi0, T), lambda E, fermi0, T: 1 - f0(E, fermi0, T)]
        calc_doping = (-1) ** (typj + 1) / self.volume / (A_to_m * m_to_cm) ** 3 \
                      * abs(self.integrate_over_DOSxE_dE(func=funcs[typj], tp=typ, fermi=fermi, T=T))

        while (relative_error > tolerance) and (iter < max_iter):
            # print iter
            # print calc_doping
            # print fermi
            # print (-1) ** (typj)
            # print
            iter += 1  # to avoid an infinite loop
            if iter / max_iter > 0.5:  # to avoid oscillation we re-adjust alpha at each iteration
                tune_alpha = 1 - iter / max_iter
            # fermi += (-1) ** (typj) * alpha * tune_alpha * (calc_doping - c) / abs(c + calc_doping) * fermi
            fermi += alpha * tune_alpha * (calc_doping - c) / abs(c + calc_doping) * abs(fermi)
            if abs(fermi) < 1e-5: # switch sign when getting really close to 0 as otherwise will never converge
                fermi = fermi * -1

            for j, tp in enumerate(["n", "p"]):
                integral = 0.0

                # for ie in range((1 - j) * self.cbm_dos_idx + j * 0,
                #                 (1 - j) * len(self.dos) - 1 + j * self.vbm_dos_idx - 1):
                for ie in range((1 - j) * self.cbm_dos_idx,
                                    (1 - j) * len(self.dos) + j * self.vbm_dos_idx - 1):
                    integral += (self.dos[ie + 1][1] + self.dos[ie][1]) / 2 * funcs[j](self.dos[ie][0], fermi, T) * \
                                (self.dos[ie + 1][0] - self.dos[ie][0])
                temp_doping[tp] = (-1) ** (j + 1) * abs(integral / (self.volume * (A_to_m * m_to_cm) ** 3))

            calc_doping = temp_doping["n"] + temp_doping["p"]
            if abs(calc_doping) < 1e-2:
                calc_doping = np.sign(calc_doping) * 0.01  # just so that calc_doping doesn't get stuck to zero!

            # calculate the relative error from the desired concentration, c
            relative_error = abs(calc_doping - c) / abs(c)

        self.egrid["calc_doping"][c][T]["n"] = temp_doping["n"]
        self.egrid["calc_doping"][c][T]["p"] = temp_doping["p"]

        # check to see if the calculated concentration is close enough to the desired value
        if relative_error > tolerance and relative_error <= tolerance_loose:
            warnings.warn("The calculated concentration {} is not accurate compared to {}; results may be unreliable"
                          .format(calc_doping, c))
        elif relative_error > tolerance_loose:
            raise ValueError("The calculated concentration {} is more than {}% away from {}; "
                             "possible cause may low band gap, high temperature, small nsteps, etc; AMSET stops now!"
                             .format(calc_doping, tolerance_loose * 100, c))

        logging.info("fermi at {} 1/cm3 and {} K after {} iterations: {}".format(c, T, int(iter), fermi))
        return fermi



    def inverse_screening_length(self, c, T):
        """
        calculates the inverse screening length (beta) in 1/nm units
        :param tp:
        :param fermi:
        :param T:
        :param interpolation_nsteps:
        :return:
        """
        beta = {}
        for tp in ["n", "p"]:
            # TODO: the integration may need to be revised. Careful testing of IMP scattering against expt is necessary
            # integral = self.integrate_over_E(func=func, tp=tp, fermi=self.egrid["fermi"][c][T], T=T)

            # because this integral has no denominator to cancel the effect of weights, we do non-weighted integral
            # integrate in egrid with /volume and proper unit conversion
            # we assume here that DOS is normalized already
            # integral = self.integrate_over_E(prop_list=["f0x1-f0"], tp=tp, c=c, T=T, xDOS=True, weighted=False)
            integral = self.integrate_over_normk(prop_list=["f0","1-f0"], tp=tp, c=c, T=T, xDOS=True)
            integral = sum(integral)/3
            print('integral_over_norm_k') # for egrid it is 8.68538649689e-06
            print(integral)
            # integral = sum(self.integrate_over_BZ(["f0", "1-f0"], tp, c, T, xDOS=False, xvel=False, weighted=False))/3

            # from aMoBT ( or basically integrate_over_normk )
            beta[tp] = (e**2 / (self.epsilon_s * epsilon_0*k_B*T) * integral * 6.241509324e27)**0.5

            # for integrate_over_E
            # beta[tp] = (e ** 2 / (self.epsilon_s * epsilon_0 * k_B * T) * integral / self.volume * 1e12 / e) ** 0.5

            # for integrate_over_BZ: incorrect (tested on 7/18/2017)
            # beta[tp] = (e**2 / (self.epsilon_s * epsilon_0*k_B*T) * integral * 100/e)**0.5

        return beta



    def to_json(self, kgrid=True, trimmed=False, max_ndata=None, nstart=0):

        if not max_ndata:
            max_ndata = int(self.gl)

        egrid = deepcopy(self.egrid)
        if trimmed:
            nmax = min([max_ndata + 1, min([len(egrid["n"]["energy"]), len(egrid["p"]["energy"])])])

            for tp in ["n", "p"]:
                for key in egrid[tp]:
                    if key in ["size"]:
                        continue
                    try:
                        for c in self.dopings:
                            for T in self.temperatures:
                                if tp == "n":
                                    egrid[tp][key][c][T] = self.egrid[tp][key][c][T][nstart:nstart + nmax]
                                else:
                                    egrid[tp][key][c][T] = self.egrid[tp][key][c][T][::-1][nstart:nstart + nmax]
                                    # egrid[tp][key][c][T] = self.egrid[tp][key][c][T][-(nstart+nmax):-max(nstart,1)][::-1]
                    except:
                        try:
                            if tp == "n":
                                egrid[tp][key] = self.egrid[tp][key][nstart:nstart + nmax]
                            else:
                                egrid[tp][key] = self.egrid[tp][key][::-1][nstart:nstart + nmax]
                                # egrid[tp][key] = self.egrid[tp][key][-(nstart+nmax):-max(nstart,1)][::-1]
                        except:
                            print "cutting data for {} numbers in egrid was NOT successful!".format(key)
                            pass

        with open("egrid.json", 'w') as fp:
            json.dump(egrid, fp, sort_keys=True, indent=4, ensure_ascii=False, cls=MontyEncoder)

        # self.kgrid trimming
        if kgrid:
            start_time = time.time()
            kgrid = deepcopy(self.kgrid)
            print "time to copy kgrid = {} seconds".format(time.time() - start_time)
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
                                        # kgrid[tp][key][c][T] = [self.kgrid[tp][key][c][T][b][-(nstart+nmax):-max(nstart,1)][::-1]
                                        #                         for b in range(self.cbm_vbm[tp]["included"])]
                        except:
                            try:
                                if tp == "n":
                                    kgrid[tp][key] = [self.kgrid[tp][key][b][nstart:nstart + nmax]
                                                  for b in range(self.cbm_vbm[tp]["included"])]
                                else:
                                    kgrid[tp][key] = [self.kgrid[tp][key][b][::-1][nstart:nstart + nmax]
                                                      for b in range(self.cbm_vbm[tp]["included"])]
                                    # kgrid[tp][key] = [self.kgrid[tp][key][b][-(nstart+nmax):-max(nstart,1)][::-1]
                                    #                   for b in range(self.cbm_vbm[tp]["included"])]
                            except:
                                print "cutting data for {} numbers in kgrid was NOT successful!".format(key)
                                pass

            with open("kgrid.json", 'w') as fp:
                json.dump(kgrid, fp, sort_keys=True, indent=4, ensure_ascii=False, cls=MontyEncoder)



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
                                                                    self.kgrid[tp]["S_o"][c][T][ib] + self.gs)

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

                            for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                                if norm(self.kgrid[tp]["g_POP"][c][T][ib][ik]) > 1 and iter > 0:
                                    # because only when there are no S_o/S_i scattering events, g_POP>>1 while it should be zero
                                    self.kgrid[tp]["g_POP"][c][T][ib][ik] = [self.gs, self.gs, self.gs]

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
        # TODO: Take into account the fact that this gradient is found in three directions specified by the lattice, not
        # the x, y, and z directions. It must be corrected to account for this.
        energy_grid = self.array_from_kgrid('energy', tp)
        # print('energy:')
        # np.set_printoptions(precision=3)
        # print(energy_grid[0,:,:,:,0])
        N = self.kgrid_array['k_points'].shape
        k_grid = self.kgrid_array['k_points']
        v_vec_result = []
        for ib in range(self.num_bands[tp]):
            v_vec = np.gradient(energy_grid[ib][:,:,:,0], k_grid[:,0,0,0] * self._rec_lattice.a, k_grid[0,:,0,1] * self._rec_lattice.b, k_grid[0,0,:,2] * self._rec_lattice.c)
            v_vec_rearranged = np.zeros((N[0], N[1], N[2], 3))
            for i in range(N[0]):
                for j in range(N[1]):
                    for k in range(N[2]):
                        v_vec_rearranged[i,j,k,:] = np.array([v_vec[0][i,j,k], v_vec[1][i,j,k], v_vec[2][i,j,k]])
            v_vec_rearranged *= A_to_m * m_to_cm / hbar
            v_vec_result.append(v_vec_rearranged)
        return np.array(v_vec_result)


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
            for ib in range(self.num_bands[tp]):
                for ik in self.rm_idx_list[tp][ib]:
                    adjusted_prop_list.insert(ik, fill) if not insert_list else adjusted_prop_list.insert(ik, [fill,fill,fill])

        # step 2 is reorder based on first sort
        adjusted_prop_list = np.array(adjusted_prop_list)[self.pos_idx[tp]]
        # then call grid_from_ordered_list
        return self.grid_from_ordered_list(adjusted_prop_list, tp, denom=denom, none_missing=True)


    # return a grid of the (x,y,z) k points in the proper grid
    def grid_from_ordered_list(self, prop_list, tp=None, denom=False, none_missing=False):
        # need:
        # self.kgrid_array
        N = self.kgrid_array['k_points'].shape
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
    def integrate_over_states(self, integrand_grid):

        integrand_grid = np.array(integrand_grid)

        if type(integrand_grid[0][0,0,0]) == list or type(integrand_grid[0][0,0,0]) == np.ndarray:
            result = np.zeros(3)
        else:
            result = 0
        num_bands = integrand_grid.shape[0]

        for ib in range(num_bands):
            result += self.integrate_over_k(integrand_grid[ib])
        return result


    # calculates transport properties for isotropic materials
    def calculate_transport_properties_with_k(self):
        # calculate mobility by averaging velocity per electric field strength
        mu_num = {tp: {el_mech: {c: {T: [0, 0, 0] for T in self.temperatures} for c in self.dopings} for el_mech in self.elastic_scatterings} for tp in ["n", "p"]}
        mu_denom = deepcopy(mu_num)
        mo_labels = self.elastic_scatterings + self.inelastic_scatterings + ['overall', 'average']
        self.mobility = {tp: {el_mech: {c: {T: [0, 0, 0] for T in self.temperatures} for c in self.dopings} for el_mech in mo_labels} for tp in ["n", "p"]}

        #k_hat = np.array([self.k_hat_grid for ib in range(self.num_bands)])
        N = self.kgrid_array['k_points'].shape

        for c in self.dopings:
            for T in self.temperatures:
                for j, tp in enumerate(["p", "n"]):

                    print('tp =  ' + tp + ':')
                    # get quantities that are independent of mechanism
                    num_k = [len(self.kgrid[tp]["energy"][ib]) for ib in range(self.num_bands[tp])]
                    df0dk = self.array_from_kgrid('df0dk', tp, c, T)
                    v = self.array_from_kgrid('velocity', tp)
                    #v = self.calc_v_vec(tp)
                    norm_v = np.array([self.grid_from_energy_list([norm(self.kgrid[tp]["velocity"][ib][ik]) / sq3 for ik in
                                                          range(num_k[ib])], tp, ib) for ib in range(self.num_bands[tp])])
                    #norm_v = grid_norm(v)
                    f0_removed = self.array_from_kgrid('f0', tp, c, T)
                    #energy = self.array_from_kgrid('energy', tp, fill=1000000)
                    #f0 = 1 / (np.exp((energy - self.fermi_level[c][T]) / (k_B * T)) + 1)
                    for ib in range(self.num_bands[tp]):
                        print('energy (type {}, band {}):'.format(tp, ib))
                        print(self.energy_array[tp][ib][(N[0] - 1) / 2, (N[1] - 1) / 2, :])
                    f0_all = 1 / (np.exp((self.energy_array[tp] - self.fermi_level[c][T]) / (k_B * T)) + 1)
                    #f0_all = 1 / (np.exp((self.energy_array[self.get_tp(c)] - self.fermi_level[c][T]) / (k_B * T)) + 1)

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
                    if not self.bs_is_isotropic:   # this is NOT working

                        # from equation 44 in Rode, overall
                        nu_el = self.array_from_kgrid('_all_elastic', tp, c, T, denom=True)
                        S_i = 0
                        S_o = 1
                        numerator = -self.integrate_over_states(v * self.k_hat_grid * (-1 / hbar) * df0dk / nu_el)
                        denominator = self.integrate_over_states(f0) * hbar * default_small_E
                        self.mobility[tp]['overall'][c][T] = numerator / denominator

                        # from equation 44 in Rode, elastic
                        #el_mech stands for elastic mechanism
                        for el_mech in self.elastic_scatterings:
                            nu_el = self.array_from_kgrid(el_mech, tp, c, T, denom=True)
                            # includes e in numerator because hbar is in eV units, where e = 1
                            numerator = -self.integrate_over_states(v * self.k_hat_grid * df0dk / nu_el)
                            denominator = self.integrate_over_states(f0) * hbar
                            # for ib in range(len(self.kgrid[tp]["energy"])):
                            #     #num_kpts = len(self.kgrid[tp]["energy"][ib])
                            #     # integrate numerator / norm(F) of equation 44 in Rode
                            #     for dim in range(3):
                            #         # TODO: add in f0 to the integral so that this works for anisotropic materials
                            #         mu_num[tp][el_mech][c][T][dim] += self.integrate_over_k(v_vec[ib] * k_hat[ib] * df0dk[ib] / nu_el[ib])[dim]
                            #         mu_denom[tp][el_mech][c][T][dim] += self.integrate_over_k(f0[ib])[dim]

                            # should be -e / hbar but hbar already in eV units, where e=1
                            self.mobility[tp][el_mech][c][T] = numerator / denominator

                        # from equation 44 in Rode, inelastic
                        for inel_mech in self.inelastic_scatterings:
                            nu_el = self.array_from_kgrid('_all_elastic', tp, c, T, denom=True)
                            S_i = 0
                            S_o = 1
                            self.mobility[tp][inel_mech][c][T] = self.integrate_over_states(
                                v * self.k_hat_grid * (-1 / hbar) * df0dk / S_o)

                        if tp == "n":
                            for mech in self.elastic_scatterings + ['overall']:
                                print('new {} mobility at T={}: {}'.format(mech, T, self.mobility[tp][mech][c][T]))

                    if self.bs_is_isotropic:
                        # from equation 45 in Rode, elastic mechanisms
                        for ib in range(self.num_bands[tp]):
                            print('f0 (type {}, band {}):'.format(tp, ib))
                            print(f0_all[ib, (N[0]-1)/2, (N[1]-1)/2, :])
                        if tp == 'n':
                            denominator = 3 * default_small_E * self.integrate_over_states(f0_all)
                        if tp == 'p':
                            denominator = 3 * default_small_E * self.integrate_over_states(1-f0_all)
                        # print('denominator:')
                        # print(denominator)
                        for el_mech in self.elastic_scatterings:
                            nu_el = self.array_from_kgrid(el_mech, tp, c, T, denom=True)
                            # this line should have -e / hbar except that hbar is in units of eV*s so in those units e=1
                            g = -1 / hbar * df0dk / nu_el
                            # print('g*norm(v) for {}:'.format(el_mech))
                            # print((g * norm_v)[0, (N[0]-1)/2, (N[1]-1)/2, :])
                            self.mobility[tp][el_mech][c][T] = self.integrate_over_states(g * norm_v) / denominator

                        # from equation 45 in Rode, inelastic mechanisms
                        for inel_mech in self.inelastic_scatterings:
                            g = self.array_from_kgrid("g_"+inel_mech, tp, c, T)
                            # print('g*norm(v) for {}:'.format(inel_mech))
                            # print((g * norm_v)[0, (N[0]-1)/2, (N[1]-1)/2, :])
                            self.mobility[tp][inel_mech][c][T] = self.integrate_over_states(g * norm_v) / denominator

                        # from equation 45 in Rode, overall
                        g = self.array_from_kgrid("g", tp, c, T)
                        #print('g: {}'.format(g))
                        #print('norm_v: {}'.format(norm_v))
                        for ib in range(self.num_bands[tp]):
                            print('g for overall (type {}, band {}):'.format(tp, ib))
                            print(g[ib, (N[0] - 1) / 2, (N[1] - 1) / 2, :])
                            print('norm(v) for overall (type {}, band {}):'.format(tp, ib))
                            print(norm_v[ib, (N[0] - 1) / 2, (N[1] - 1) / 2, :])
                            print('g*norm(v) for overall (type {}, band {}):'.format(tp, ib))
                            print((g * norm_v)[ib, (N[0]-1)/2, (N[1]-1)/2, :])
                        self.mobility[tp]['overall'][c][T] = self.integrate_over_states(g * norm_v) / denominator

                    print('new {}-type overall mobility at T = {}: {}'.format(tp, T, self.mobility[tp]['overall'][c][T]))
                    for el_mech in self.elastic_scatterings + self.inelastic_scatterings:
                        print('new {}-type {} mobility at T = {}: {}'.format(tp, el_mech, T, self.mobility[tp][el_mech][c][T]))

                    # figure out average mobility
                    faulty_overall_mobility = False
                    mu_overrall_norm = norm(self.mobility[tp]["overall"][c][T])
                    for transport in self.elastic_scatterings + self.inelastic_scatterings:
                        # averaging all mobility values via Matthiessen's rule
                        self.mobility[tp]["average"][c][T] += 1 / (np.array(self.mobility[tp][transport][c][T]) + 1e-50)
                        if mu_overrall_norm > norm(self.mobility[tp][transport][c][T]):
                            faulty_overall_mobility = True  # because the overall mobility should be lower than all
                    self.mobility[tp]["average"][c][T] = 1 / np.array(self.mobility[tp]["average"][c][T])

                    # Decide if the overall mobility make sense or it should be equal to average (e.g. when POP is off)
                    if mu_overrall_norm == 0.0 or faulty_overall_mobility:
                        self.mobility[tp]["overall"][c][T] = self.mobility[tp]["average"][c][T]



    def calculate_transport_properties_with_E(self):
        integrate_over_kgrid = False
        for c in self.dopings:
            for T in self.temperatures:
                for j, tp in enumerate(["p", "n"]):

                    # mobility numerators
                    for mu_el in self.elastic_scatterings:
                        if integrate_over_kgrid:
                            self.egrid["mobility"][mu_el][c][T][tp] = (-1) * default_small_E / hbar * \
                                                                      self.integrate_over_BZ(
                                                                          prop_list=["/" + mu_el, "df0dk"], tp=tp, c=c,
                                                                          T=T, xDOS=False, xvel=True,
                                                                          weighted=True) #* 1e-7 * 1e-3 * self.volume

                        else:
                            self.egrid["mobility"][mu_el][c][T][tp] = (-1) * default_small_E / hbar * \
                                                                      self.integrate_over_E(
                                                                          prop_list=["/" + mu_el, "df0dk"], tp=tp, c=c,
                                                                          T=T, xDOS=False, xvel=True, weighted=True)
                            if tp == "n":
                                print('old {} numerator = {}'.format(mu_el, self.egrid["mobility"][mu_el][c][T][tp]))

                    if integrate_over_kgrid:
                        if tp == "n":
                            denom = self.integrate_over_BZ(["f0"], tp, c, T, xDOS=False, xvel=False,
                                                       weighted=False) #* 1e-7 * 1e-3 * self.volume
                            print('old denominator = ' + str(denom))
                        else:
                            denom = self.integrate_over_BZ(["1 - f0"], tp, c, T, xDOS=False, xvel=False,
                                                           weighted=False)
                    else:
                        if tp == "n":
                            denom = self.integrate_over_E(prop_list=["f0"], tp=tp, c=c, T=T, xDOS=False, xvel=False,
                                                      weighted=False)
                        else:
                            denom = self.integrate_over_E(prop_list=["1 - f0"], tp=tp, c=c, T=T, xDOS=False, xvel=False,
                                                          weighted=False)

                    print "denom for {}-type with integrate_over_kgrid: {}: \n {}".format(tp, integrate_over_kgrid, denom)

                    if integrate_over_kgrid:
                        for mu_inel in self.inelastic_scatterings:
                            self.egrid["mobility"][mu_inel][c][T][tp] = self.integrate_over_BZ(
                                prop_list=["g_" + mu_inel], tp=tp, c=c, T=T, xDOS=False, xvel=True, weighted=True)
                        self.egrid["mobility"]["overall"][c][T][tp] = self.integrate_over_BZ(["g"], tp, c, T,
                                                                                             xDOS=False, xvel=True,
                                                                                             weighted=True)
                        print "overll numerator"
                        print self.egrid["mobility"]["overall"][c][T][tp]
                    else:
                        for mu_inel in self.inelastic_scatterings:
                            # calculate mobility["POP"] based on g_POP
                            self.egrid["mobility"][mu_inel][c][T][tp] = self.integrate_over_E(
                                prop_list=["g_" + mu_inel], tp=tp, c=c, T=T, xDOS=False,xvel=True, weighted=True)

                        self.egrid["mobility"]["overall"][c][T][tp] = self.integrate_over_E(prop_list=["g"],
                                                                                            tp=tp, c=c, T=T, xDOS=False,
                                                                                            xvel=True, weighted=True)

                    self.egrid["J_th"][c][T][tp] = (self.integrate_over_E(prop_list=["g_th"], tp=tp, c=c, T=T,
                                                                          xDOS=False, xvel=True,
                                                                          weighted=True) / denom) * e * abs(
                        c)  # in units of A/cm2

                    for transport in self.elastic_scatterings + self.inelastic_scatterings + ["overall"]:
                        self.egrid["mobility"][transport][c][T][tp] /= 3 * default_small_E * denom

                    # The following did NOT work as J_th only has one integral (see aMoBT but that one is over k)
                    # and with that one the units don't work out and if you use two integral, J_th will be of 1e6 order!
                    # self.egrid["J_th"][c][T][tp] = self.integrate_over_E(prop_list=["g_th"], tp=tp, c=c, T=T,
                    #         xDOS=False, xvel=True, weighted=True) * e * 1e24  # to bring J to A/cm2 units
                    # self.egrid["J_th"][c][T][tp] /= 3*self.volume*self.integrate_over_E(prop_list=["f0"], tp=tp, c=c,
                    #         T=T, xDOS=False, xvel=False, weighted=True)

                    # other semi-empirical mobility values:
                    #fermi = self.egrid["fermi"][c][T]
                    fermi = self.fermi_level[c][T]
                    # fermi_SPB = self.egrid["fermi_SPB"][c][T]
                    energy = self.cbm_vbm[self.get_tp(c)]["energy"]

                    # for mu in ["overall", "average"] + self.inelastic_scatterings + self.elastic_scatterings:
                    #     self.egrid["mobility"][mu][c][T][tp] /= 3.0

                    # ACD mobility based on single parabolic band extracted from Thermoelectric Nanomaterials,
                    # chapter 1, page 12: "Material Design Considerations Based on Thermoelectric Quality Factor"
                    self.egrid["mobility"]["SPB_ACD"][c][T][tp] = 2 ** 0.5 * pi * hbar ** 4 * e * self.C_el * 1e9 / (
                    # C_el in GPa
                        3 * (self.cbm_vbm[tp]["eff_mass_xx"] * m_e) ** 2.5 * (k_B * T) ** 1.5 * self.E_D[tp] ** 2) \
                                                                  * fermi_integral(0, fermi, T, energy, wordy=True) \
                                                                  / fermi_integral(0.5, fermi, T, energy,
                                                                                   wordy=True) * e ** 0.5 * 1e4  # to cm2/V.s

                    faulty_overall_mobility = False
                    mu_overrall_norm = norm(self.egrid["mobility"]["overall"][c][T][tp])
                    for transport in self.elastic_scatterings + self.inelastic_scatterings:
                        # averaging all mobility values via Matthiessen's rule
                        self.egrid["mobility"]["average"][c][T][tp] += 1 / self.egrid["mobility"][transport][c][T][tp]
                        if mu_overrall_norm > norm(self.egrid["mobility"][transport][c][T][tp]):
                            faulty_overall_mobility = True  # because the overall mobility should be lower than all
                    self.egrid["mobility"]["average"][c][T][tp] = 1 / self.egrid["mobility"]["average"][c][T][tp]

                    # Decide if the overall mobility make sense or it should be equal to average (e.g. when POP is off)
                    if mu_overrall_norm == 0.0 or faulty_overall_mobility:
                        self.egrid["mobility"]["overall"][c][T][tp] = self.egrid["mobility"]["average"][c][T][tp]

                    self.egrid["relaxation time constant"][c][T][tp] = self.egrid["mobility"]["overall"][c][T][tp] \
                                                                       * 1e-4 * m_e * self.cbm_vbm[tp][
                                                                           "eff_mass_xx"] / e  # 1e-4 to convert cm2/V.s to m2/V.s

                    print('old {}-type overall mobility at T = {}: {}'.format(tp, T, self.egrid["mobility"]["overall"][c][T][tp]))
                    for mech in self.elastic_scatterings + self.inelastic_scatterings:
                        print('old {}-type {} mobility at T = {}: {}'.format(tp, mech, T, self.egrid["mobility"][mech][c][T][tp]))

                    # calculating other overall transport properties:
                    self.egrid["conductivity"][c][T][tp] = self.egrid["mobility"]["overall"][c][T][tp] * e * abs(c)
                    # self.egrid["seebeck"][c][T][tp] = -1e6 * k_B * (self.egrid["Seebeck_integral_numerator"][c][T][tp] \
                    #                                                 / self.egrid["Seebeck_integral_denominator"][c][T][
                    #                                                     tp] - (
                    #                                                 self.egrid["fermi"][c][T] - self.cbm_vbm[tp][
                    #                                                     "energy"]) / (k_B * T))
                    self.egrid["seebeck"][c][T][tp] = -1e6 * k_B * (self.egrid["Seebeck_integral_numerator"][c][T][tp] \
                                                                    / self.egrid["Seebeck_integral_denominator"][c][T][
                                                                        tp] - (
                                                                        self.fermi_level[c][T] - self.cbm_vbm[tp][
                                                                            "energy"]) / (k_B * T))
                    self.egrid["TE_power_factor"][c][T][tp] = self.egrid["seebeck"][c][T][tp] ** 2 \
                                                              * self.egrid["conductivity"][c][T][tp] / 1e6  # in uW/cm2K
                    if "POP" in self.inelastic_scatterings:  # when POP is not available J_th is unreliable
                        self.egrid["seebeck"][c][T][tp] = np.array([self.egrid["seebeck"][c][T][tp] for i in range(3)])
                        self.egrid["seebeck"][c][T][tp] += 0.0
                        # TODO: for now, we ignore the following until we figure out the units see why values are high!
                        # self.egrid["seebeck"][c][T][tp] += 1e6 \
                        #                 * self.egrid["J_th"][c][T][tp]/self.egrid["conductivity"][c][T][tp]/dTdz

                    print "3 {}-seebeck terms at c={} and T={}:".format(tp, c, T)
                    print self.egrid["Seebeck_integral_numerator"][c][T][tp] \
                          / self.egrid["Seebeck_integral_denominator"][c][T][tp] * -1e6 * k_B
                    #print + (self.egrid["fermi"][c][T] - self.cbm_vbm[tp]["energy"]) * 1e6 * k_B / (k_B * T)
                    print + (self.fermi_level[c][T] - self.cbm_vbm[tp]["energy"]) * 1e6 * k_B / (k_B * T)
                    print + self.egrid["J_th"][c][T][tp] / self.egrid["conductivity"][c][T][tp] / dTdz * 1e6


                    #TODO: not sure about the following part yet specially as sometimes due to position of fermi I get very off other type mobility values! (sometimes very large)
                    other_type = ["p", "n"][1 - j]
                    self.egrid["seebeck"][c][T][tp] = (self.egrid["conductivity"][c][T][tp] * \
                                                       self.egrid["seebeck"][c][T][tp] -
                                                       self.egrid["conductivity"][c][T][other_type] * \
                                                       self.egrid["seebeck"][c][T][other_type]) / (
                                                      self.egrid["conductivity"][c][T][tp] +
                                                      self.egrid["conductivity"][c][T][other_type])
                    ## since sigma = c_e x e x mobility_e + c_h x e x mobility_h:
                    ## self.egrid["conductivity"][c][T][tp] += self.egrid["conductivity"][c][T][other_type]



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
                     textsize, ticksize, path, margin_left, margin_bottom, fontfamily, x_data=None, y_data=None,
                     all_plots=None, x_label_short='', y_label_short=None, y_axis_type='linear', plot_title=None):
        from matminer.figrecipes.plotly.make_plots import PlotlyFig
        if not plot_title:
            plot_title = '{} for {}, c={}'.format(y_label, self.tp_title[tp], c)
        if not y_label_short:
            y_label_short = y_label
        if show_interactive:
            if not x_label_short:
                filename = os.path.join(path, "{}_{}.{}".format(y_label_short, file_suffix, 'html'))
            else:
                filename = os.path.join(path, "{}_{}_{}.{}".format(y_label_short, x_label_short, file_suffix, 'html'))
            plt = PlotlyFig(x_title=x_label, y_title=y_label,
                            plot_title=plot_title, textsize=textsize,
                            plot_mode='offline', filename=filename, ticksize=ticksize,
                            margin_left=margin_left, margin_bottom=margin_bottom, fontfamily=fontfamily)
            if all_plots:plt.xy_plot(x_col=[], y_col=[], add_xy_plot=all_plots, y_axis_type=y_axis_type, color='black', showlegend=True)

            else:
                plt.xy_plot(x_col=x_data, y_col=y_data, y_axis_type=y_axis_type, color='black')
        if save_format is not None:
            if not x_label_short:
                filename = os.path.join(path, "{}_{}.{}".format(y_label_short, file_suffix, save_format))
            else:
                filename = os.path.join(path, "{}_{}_{}.{}".format(y_label_short, x_label_short, file_suffix, save_format))
            plt = PlotlyFig(x_title=x_label, y_title=y_label,
                            plot_title=plot_title, textsize=textsize,
                            plot_mode='static', filename=filename, ticksize=ticksize,
                            margin_left=margin_left, margin_bottom=margin_bottom, fontfamily=fontfamily)
            if all_plots:
                plt.xy_plot(x_col=[], y_col=[], add_xy_plot=all_plots, y_axis_type=y_axis_type, color='black', showlegend=True)
            else:
                plt.xy_plot(x_col=x_data, y_col=y_data, y_axis_type=y_axis_type, color='black')



    def plot(self, k_plots=[], E_plots=[], mobility=True, concentrations='all', carrier_types=['n', 'p'],
             direction=['avg'], show_interactive=True, save_format='png', textsize=40, ticksize=30, path=None,
             margin_left=160, margin_bottom=120, fontfamily="serif"):
        """
        plots the calculated values
        :param k_plots: (list of strings) the names of the quantities to be plotted against norm(k)
            options: 'energy', 'df0dk', 'velocity', or just string 'all' (not in a list) to plot everything
        :param E_plots: (list of strings) the names of the quantities to be plotted against E
            options: 'frequency', 'relaxation time', '_all_elastic', 'df0dk', 'velocity', 'ACD', 'IMP', 'PIE', 'g',
            'g_POP', 'g_th', 'S_i', 'S_o', or just string 'all' (not in a list) to plot everything
        :param mobility: (boolean) if True, create a mobility against temperature plot
        :param concentrations: (list of strings) a list of carrier concentrations, or the string 'all' to plot the
            results of calculations done with all input concentrations
        :param carrier_types: (list of strings) select carrier types to plot data for - ['n'], ['p'], or ['n', 'p']
        :param direction: (list of strings) options to include in list are 'x', 'y', 'z', 'avg'; determines which
            components of vector quantities are plotted
        :param show_interactive: (boolean) if True creates and shows interactive html plots
        :param save_format: (str) format for saving plots; options are 'png', 'jpeg', 'svg', 'pdf', None (None does not
            save the plots). NOTE: plotly credentials are needed, see figrecipes documentation
        :param textsize: (int) size of title and axis label text
        :param ticksize: (int) size of axis tick label text
        :param path: (string) location to save plots
        :param margin_left: (int) plotly left margin
        :param margin_bottom: (int) plotly bottom margin
        :param fontfamily: (string) plotly font
        """

        if k_plots == 'all':
            k_plots = ['energy', 'df0dk', 'velocity']
        if E_plots == 'all':
            E_plots = ['frequency', 'relaxation time', 'df0dk', 'velocity'] + self.elastic_scatterings
            if "POP" in self.inelastic_scatterings:
                E_plots += ['g', 'g_POP', 'S_i', 'S_o']

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
            if prop in all_temp_independent_k_props:
                temp_independent_k_props.append(prop)
            else:
                temp_dependent_k_props.append(prop)
        temp_dependent_E_props = []
        for prop in E_plots:
            if prop in all_temp_independent_E_props:
                temp_independent_E_props.append(prop)
            else:
                temp_dependent_E_props.append(prop)

        vec = {'energy': False,
               'velocity': True,
               'frequency': False}

        for tp in carrier_types:
            x_data = {'k': self.kgrid[tp]["norm(k)"][0],
                      'E': [E - self.cbm_vbm[tp]["energy"] for E in self.egrid[tp]["energy"]]}
            x_axis_label = {'k': 'norm(k)', 'E': 'energy (eV)'}

            for c in concentrations:

                # plots of scalar properties first
                tp_c = tp + '_' + str(c)
                for x_value, y_values in [('k', temp_independent_k_props), ('E', temp_independent_E_props)]:
                    y_data_temp_independent = {'k': {'energy': self.kgrid[tp]['energy'][0],
                                                     'velocity': self.kgrid[tp]["norm(v)"][0]},
                                               'E': {'frequency': self.Efrequency[tp]}}
                    for y_value in y_values:
                        if not vec[y_value]:
                            plot_title = None
                            if y_value == 'frequency':
                                plot_title = 'Energy Histogram for {}, c={}'.format(self.tp_title[tp], c)
                            self.create_plots(x_axis_label[x_value], y_value, show_interactive, save_format, c, tp, tp_c,
                                              textsize, ticksize, path, margin_left,
                                              margin_bottom, fontfamily, x_data=x_data[x_value], y_data=y_data_temp_independent[x_value][y_value], x_label_short=x_value, plot_title=plot_title)

                for dir in direction:
                    y_data_temp_independent = {'k': {'energy': self.kgrid[tp]['energy'][0],
                                                     'velocity': self.kgrid[tp]["norm(v)"][0]},
                                               'E': {'frequency': self.Efrequency[tp],
                                                     'velocity': [self.get_scalar_output(p, dir) for p in self.egrid[tp]['velocity']]}}

                    tp_c_dir = tp_c + '_' + dir

                    # temperature independent k and E plots: energy(k), velocity(k), histogram(E), velocity(E)
                    for x_value, y_values in [('k', temp_independent_k_props), ('E', temp_independent_E_props)]:
                        for y_value in y_values:
                            if vec[y_value]:
                                self.create_plots(x_axis_label[x_value], y_value, show_interactive,
                                                  save_format, c, tp, tp_c_dir,
                                                  textsize, ticksize, path, margin_left,
                                                  margin_bottom, fontfamily, x_data=x_data[x_value],
                                                  y_data=y_data_temp_independent[x_value][y_value], x_label_short=x_value)

                    # want variable of the form: y_data_temp_dependent[k or E][prop][temp] (the following lines reorganize
                    # kgrid and egrid data)
                    y_data_temp_dependent = {'k': {prop: {T: [self.get_scalar_output(p, dir) for p in self.kgrid[tp][prop][c][T][0]]
                                                          for T in self.temperatures} for prop in temp_dependent_k_props},
                                             'E': {prop: {T: [self.get_scalar_output(p, dir) for p in self.egrid[tp][prop][c][T]]
                                                          for T in self.temperatures} for prop in temp_dependent_E_props}}

                    # temperature dependent k and E plots
                    for x_value, y_values in [('k', temp_dependent_k_props), ('E', temp_dependent_E_props)]:
                        for y_value in y_values:
                            all_plots = []
                            for T in self.temperatures:
                                all_plots.append({"x_col": x_data[x_value],
                                                  "y_col": y_data_temp_dependent[x_value][y_value][T],
                                                  "text": T, 'legend': str(T) + ' K', 'size': 6, "mode": "markers",
                                                  "color": "", "marker": temp_markers[T]})
                            self.create_plots(x_axis_label[x_value], y_value, show_interactive,
                                              save_format, c, tp, tp_c_dir,
                                              textsize, ticksize, path, margin_left,
                                              margin_bottom, fontfamily, all_plots=all_plots, x_label_short=x_value)

                    # mobility plots as a function of temperature (the only plot that does not have k or E on the x axis)
                    if mobility:
                        all_plots = []
                        for mo in mu_list:
                            all_plots.append({"x_col": self.temperatures,
                                              "y_col": [
                                                  abs(self.get_scalar_output(self.mobility[tp][mo][c][T], dir))
                                                  # I temporarily (for debugging purposes) added abs() for cases when mistakenly I get negative mobility values!
                                                  for T in self.temperatures],
                                              "text": mo, 'legend': mo, 'size': 6, "mode": "lines+markers",
                                              "color": "", "marker": mu_markers[mo]})
                        self.create_plots("Temperature (K)", "Mobility (cm2/V.s)", show_interactive,
                                          save_format, c, tp, tp_c_dir,
                                          textsize-5, ticksize-5, path, margin_left,
                                          margin_bottom, fontfamily, all_plots=all_plots, y_label_short="mobility", y_axis_type='log')



    def to_csv(self, path=None, csv_filename='AMSET_results.csv'):
        """
        this function writes the calculated transport properties to a csv file for convenience.
        :param csv_filename (str):
        :return:
        """
        import csv
        if not path:
            path = os.getcwd()

        with open(os.path.join(path, csv_filename), 'w') as csvfile:
            fieldnames = ['type', 'c(cm-3)', 'T(K)', 'overall', 'average'] + \
                         self.elastic_scatterings + self.inelastic_scatterings + ['seebeck']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for c in self.dopings:
                tp = self.get_tp(c)
                for T in self.temperatures:
                    row = {'type': tp, 'c(cm-3)': abs(c), 'T(K)': T}
                    for p in ['overall', 'average'] + self.elastic_scatterings + self.inelastic_scatterings:
                        row[p] = self.egrid["mobility"][p][c][T][tp]
                        #row[p] = sum(self.egrid["mobility"][p][c][T][tp]) / 3
                    row["seebeck"] = self.egrid["seebeck"][c][T][tp]
                    # row["seebeck"] = sum(self.egrid["seebeck"][c][T][tp]) / 3
                    writer.writerow(row)


    def test_run(self):
        self.kgrid_array = {}

        # points_1d = {dir: [-0.4 + i*0.1 for i in range(9)] for dir in ['x', 'y', 'z']}
        # points_1d = {dir: [-0.475 + i * 0.05 for i in range(20)] for dir in ['x', 'y', 'z']}
        points_1d = {dir: [] for dir in ['x', 'y', 'z']}
        # TODO: figure out which other points need a fine grid around them
        important_pts = [self.cbm_vbm["n"]["kpoint"]]
        if (np.array(self.cbm_vbm["p"]["kpoint"]) != np.array(self.cbm_vbm["n"]["kpoint"])).any():
            important_pts.append(self.cbm_vbm["p"]["kpoint"])

        for center in important_pts:
            for dim, dir in enumerate(['x', 'y', 'z']):
                points_1d[dir].append(center[dim])
                one_list = True
                if not one_list:
                    # for step, nsteps in [[0.0015, 3], [0.005, 4], [0.01, 4], [0.05, 2]]:
                    for step, nsteps in [[0.002, 2], [0.005, 4], [0.01, 4], [0.05, 2]]:
                        # for step, nsteps in [[0.01, 2]]:
                        # print "mesh: 10"
                        # loop goes from 0 to nsteps-2, so added values go from step to step*(nsteps-1)
                        for i in range(nsteps - 1):
                            points_1d[dir].append(center[dim] - (i + 1) * step)
                            points_1d[dir].append(center[dim] + (i + 1) * step)

                else:
                    # number of points options are: 175,616, 74,088, 15,625, 4,913
                    # for step in [0.001, 0.002, 0.0035, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.018, 0.021, 0.025, 0.03, 0.035, 0.0425, 0.05, 0.06, 0.075, 0.1, 0.125, 0.15, 0.18, 0.21, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
                    for step in [0.001, 0.002, 0.0035, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1,
                                 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
                        # for step in [0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.15, 0.25, 0.35, 0.45]:
                        # for step in [0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.35, 0.45]:
                        points_1d[dir].append(center[dim] + step)
                        points_1d[dir].append(center[dim] - step)

        # ensure all points are in first BZ
        for dir in ['x', 'y', 'z']:
            for ik1d in range(len(points_1d[dir])):
                if points_1d[dir][ik1d] > 0.5:
                    points_1d[dir][ik1d] -= 1
                if points_1d[dir][ik1d] <= -0.5:
                    points_1d[dir][ik1d] += 1
        # remove duplicates
        for dir in ['x', 'y', 'z']:
            points_1d[dir] = list(set(np.array(points_1d[dir]).round(decimals=14)))
        self.kgrid_array['k_points'] = self.create_grid(points_1d)
        kpts = self.array_to_kgrid(self.kgrid_array['k_points'])

        N = self.kgrid_array['k_points'].shape
        self.k_hat_grid = np.zeros(N)
        for i in range(N[0]):
            for j in range(N[1]):
                for k in range(N[2]):
                    k_vec = self.kgrid_array['k_points'][i, j, k]
                    if norm(k_vec) == 0:
                        self.k_hat_grid[i, j, k] = [0, 0, 0]
                    else:
                        self.k_hat_grid[i, j, k] = k_vec / norm(k_vec)

        self.dv_grid = self.find_dv(self.kgrid_array['k_points'])

        k_x = self.kgrid_array['k_points'][:, :, :, 0]
        k_y = self.kgrid_array['k_points'][:, :, :, 1]
        k_z = self.kgrid_array['k_points'][:, :, :, 2]
        result = self.integrate_over_k(np.cos(k_x))
        print(result)
        #print(self.kgrid_array['k_points'])


if __name__ == "__main__":
    # setting up inputs:
    mass = 0.25
    use_poly_bands = False

    model_params = {"bs_is_isotropic": True, "elastic_scatterings": ["ACD", "IMP", "PIE"],
                    "inelastic_scatterings": ["POP"] }
    if use_poly_bands:
        model_params["poly_bands"] = [[[[0.0, 0.0, 0.0], [0.0, mass]]]]

    performance_params = {"dE_min": 0.0001, "nE_min": 2, "parallel": True, "BTE_iters": 5}

    ### for PbTe
    # material_params = {"epsilon_s": 44.4, "epsilon_inf": 25.6, "W_POP": 10.0, "C_el": 128.8,
    #                "E_D": {"n": 4.0, "p": 4.0}}
    # cube_path = "../test_files/PbTe/nscf_line"
    # coeff_file = os.path.join(cube_path, "..", "fort.123")
    # #coeff_file = os.path.join(cube_path, "fort.123")

    ## For GaAs
    material_params = {"epsilon_s": 12.9, "epsilon_inf": 10.9, "W_POP": 8.73, "C_el": 139.7,
                       "E_D": {"n": 8.6, "p": 8.6}, "P_PIE": 0.052, "scissor":  0.5818}
    cube_path = "../test_files/GaAs/"
    #####coeff_file = os.path.join(cube_path, "fort.123_GaAs_k23")
    coeff_file = os.path.join(cube_path, "fort.123_GaAs_1099kp") # good results!
    # coeff_file = os.path.join(cube_path, "fort.123_GaAs_sym_23x23x23") # bad results! (because the fitting not good)
    # coeff_file = os.path.join(cube_path, "fort.123_GaAs_11x11x11_ISYM0") # good results

    ### For Si
    # material_params = {"epsilon_s": 11.7, "epsilon_inf": 11.6, "W_POP": 15.23, "C_el": 190.2,
    #                    "E_D": {"n": 6.5, "p": 6.5}, "P_PIE": 0.01, "scissor": 0.5154}
    # cube_path = "../test_files/Si/"
    # coeff_file = os.path.join(cube_path, "Si_fort.123")

    amset = AMSET(calc_dir=cube_path, material_params=material_params,
                  model_params=model_params, performance_params=performance_params,
                  dopings = [-2e15], temperatures = [300], k_integration=True, e_integration=True, fermi_type='e'
                  )   # -3.3e13
    # cProfile.run('amset.run(coeff_file=coeff_file, kgrid_tp="very coarse")')
    profiler = cProfile.Profile()
    profiler.runcall(lambda: amset.run(coeff_file=coeff_file,
            kgrid_tp="very coarse", loglevel=logging.WARNING))
    stats = Stats(profiler, stream=STDOUT)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # only print the top 10 (10 slowest functions)
    print()
    # stats.print_callers(10)

    amset.write_input_files()
    amset.to_csv()
    # amset.plot(k_plots=['energy'], E_plots='all', show_interactive=True,
    #            carrier_types=amset.all_types, save_format=None)

    amset.to_json(kgrid=True, trimmed=True, max_ndata=100, nstart=0)
