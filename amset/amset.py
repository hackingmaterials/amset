# coding: utf-8

import warnings

import time


import logging
from analytical_band_from_BZT import Analytical_bands, outer, get_dos_from_poly_bands, get_poly_energy
from pprint import pprint

import numpy as np
from math import log
# import sys
from pymatgen.io.vasp import Vasprun, Spin, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.constants.codata import value as _cd
from math import pi
import os
import json
from monty.json import MontyEncoder, MontyDecoder
from random import random
import cProfile
from copy import deepcopy

# global constants
hbar = _cd('Planck constant in eV s')/(2*pi)
m_e = _cd('electron mass') # in kg
Ry_to_eV = 13.605698066
A_to_m = 1e-10
m_to_cm = 100.00
A_to_nm = 0.1
e = _cd('elementary charge')
k_B = _cd("Boltzmann constant in eV/K")
epsilon_0 = 8.854187817e-12     # Absolute value of dielectric constant in vacuum [C**2/m**2N]
default_small_E = 1 # eV/cm the value of this parameter does not matter
dTdz = 10.0 # K/cm
sq3 = 3**0.5

# The following are example constants taken from aMoBT calculation on PbTe that was done before
# None for now

__author__ = "Alireza Faghaninia, Francesco Ricci, Anubhav Jain"
__copyright__ = "Copyright 2017, HackingMaterials"
__version__ = "0.1"
__maintainer__ = "Alireza Faghaninia"
__email__ = "alireza.faghaninia@gmail.com"
__status__ = "Development"
__date__ = "January 2017"



def norm(v):
    """method to quickly calculate the norm of a vector (v: 1x3 or 3x1) as numpy.linalg.norm is slower for this case"""
    return (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5



def f0(E, fermi, T):
    """returns the value of Fermi-Dirac at equilibrium for E (energy), fermi [level] and T (temperature)"""
    if E-fermi > 5:
        return 0.0
    elif E-fermi < -5:
        return 1.0
    else:
        return 1 / (1 + np.exp((E - fermi) / (k_B * T)))



def df0dE(E, fermi, T):
    """returns the energy derivative of the Fermi-Dirac equilibrium distribution"""
    if E-fermi > 5 or E-fermi < -5: # This is necessary so at too low numbers python doesn't return NaN
        return 0.0
    else:
        return -1 / (k_B * T) * np.exp((E - fermi) / (k_B * T)) / (1 + np.exp((E - fermi) / (k_B * T))) ** 2



def fermi_integral(order, fermi, T, initial_energy=0, wordy=False):
    fermi = fermi - initial_energy
    integral = 0.
    #TODO: 1000000 works better (converges!) but for faster testing purposes we use larger steps
    # emesh = np.linspace(0.0, 30*k_B*T, 1000000.0) # We choose 20kBT instead of infinity as the fermi distribution will be 0
    emesh = np.linspace(0.0, 30*k_B*T, 100000.0) # We choose 20kBT instead of infinity as the fermi distribution will be 0
    dE = (emesh[-1]-emesh[0])/(1000000.0-1.0)
    for E in emesh:
        integral += dE * (E/(k_B*T))**order / (1. + np.exp((E-fermi)/(k_B*T)))

    if wordy:
        print "order {} fermi integral at fermi={} and {} K".format(order,fermi, T)
        print integral
    return integral


def GB(x, eta):
    """Gaussian broadening. At very small eta values (e.g. 0.005 eV) this function goes to the dirac-delta of x."""

    return 1/np.pi*1/eta*np.exp(-(x/eta)**2)

    ## although both expressions conserve the final transport properties, the one below doesn't conserve the scat. rates
    # return np.exp(-(x/eta)**2)



class AMSET(object):
    """ This class is used to run AMSET on a pymatgen Vasprun object. AMSET is an ab initio model for calculating
    the mobility and Seebeck coefficient using Boltzmann transport equation. The band structure is extracted from
    vasprun.xml to calculate the group velocity and transport properties in presence of various scattering mechanisms.

     Currently the following scattering mechanisms with their corresponding three-letter abbreviations implemented are:
     ionized impurity scattering (IMP), acoustic phonon deformation potential (ACD), piezoelectric (PIE), and charged
     dislocation scattering (DIS). Also, longitudinal polar optical phonon (POP) in implemented as an inelastic
     scattering mechanism that can alter the electronic distribution (the reason BTE has to be solved explicitly).

     AMSET is designed in a modular way so that users can add more scattering mechanisms as followed:
     ??? (instruction to add a scattering mechanism) ???

     you can control the level of theory via various inputs. For example, constant relaxation time approximation (cRTA),
     constant mean free path (cMFP) can be used by setting these variables to True


     References:
         [R]: D. L. Rode, Low-Field Electron Transport, Elsevier, 1975, vol. 10., DOI: 10.1016/S0080-8784(08)60331-2
         [A]: A. Faghaninia, C. S. Lo and J. W. Ager, Phys. Rev. B, "Ab initio electronic transport model with explicit
          solution to the linearized Boltzmann transport equation" 2015, 91(23), 5100., DOI: 10.1103/PhysRevB.91.235123
         [Q]: B. K. Ridley, Quantum Processes in Semiconductors, oxford university press, Oxford, 5th edn., 2013.
          DOI: 10.1093/acprof:oso/9780199677214.001.0001

     """



    def __init__(self, calc_dir, material_params, model_params = {}, performance_params = {},
                 dopings=None, temperatures=None):
        """
        required parameters:
            calc_dir (str): path to the vasprun.xml
            material_params (dict): parameters related to the material

        """

        self.calc_dir = calc_dir

        # self.dopings = dopings or [-1e20] # 1/cm**3 list of carrier concentrations
        # self.temperatures = temperatures or map(float, [300, 600]) # in K, list of temperatures

        self.dopings = dopings
        self.temperatures = map(float, temperatures)

        self.set_material_params(material_params)
        self.set_model_params(model_params)
        self.set_performance_params(performance_params)

        self.read_vrun(calc_dir=self.calc_dir, filename="vasprun.xml")
        if self.poly_bands:
            self.cbm_vbm["n"]["energy"] = self.dft_gap
            self.cbm_vbm["p"]["energy"] = 0.0
            self.cbm_vbm["n"]["kpoint"] = self.cbm_vbm["p"]["kpoint"] = self.poly_bands[0][0][0]

        self.all_types = [self.get_tp(c) for c in self.dopings]



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
            "wordy": self.wordy,
            "maxiters": self.maxiters
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

        # what scattering mechanisms to be included
        self.elastic_scatterings = params.get("elastic_scatterings", ["ACD", "IMP", "PIE"])
        self.inelastic_scatterings = params.get("inelastic_scatterings", ["POP"])

        self.poly_bands = params.get("poly_bands", None)


        # TODO: self.gaussian_broadening is designed only for development version and must be False, remove it later.
        # because if self.gaussian_broadening the mapping to egrid will be done with the help of Gaussian broadening
        # and that changes the actual values
        self.gaussian_broadening = False
        self.soc = params.get("soc", False)



    def set_performance_params(self, params):
        self.nkibz = params.get("nkibz", 40)
        self.dE_min = params.get("dE_min", 0.01)
        self.nE_min = params.get("nE_min", 2)
        # max eV range after which occupation is zero, we set this at least to 10*kB*300
        self.Ecut = params.get("Ecut", 10 * k_B * max(self.temperatures + [300]))
        self.adaptive_mesh = params.get("adaptive_mesh", False)

        self.dos_bwidth = params.get("dos_bwidth",
                                     0.1)  # in eV the bandwidth used for calculation of the total DOS (over all bands & IBZ k-points)
        self.nkdos = params.get("nkdos", 35)

        self.gs = 1e-32  # a global small value (generally used for an initial non-zero value)
        self.gl = 1e32  # a global large value

        # TODO: some of the current global constants should be omitted, taken as functions inputs or changed!
        self.wordy = params.get("wordy", False)
        self.maxiters = params.get("maxiters", 5)



    def run(self, coeff_file, kgrid_tp="coarse"):
        """
        Function to run AMSET and generate the main outputs kgrid and egrid

        :param center_kpt:
        :param coeff_file:
        :param cbm_bidx:
        :param grid_tp:
        :return:
        """
        self.init_kgrid(coeff_file=coeff_file, kgrid_tp=kgrid_tp)
        logging.debug("self.cbm_vbm: {}".format(self.cbm_vbm))


        if True:
            self.init_egrid(dos_tp="standard")
        else:
            pass

        self.bandgap = min(self.egrid["n"]["all_en_flat"]) - max(self.egrid["p"]["all_en_flat"])
        if abs(self.bandgap - (
                self.cbm_vbm["n"]["energy"] - self.cbm_vbm["p"]["energy"] + self.scissor)) > k_B * 300:
            warnings.warn("The band gaps do NOT match! The selected k-mesh is probably too coarse.")
            # raise ValueError("The band gaps do NOT match! The selected k-mesh is probably too coarse.")

        # initialize g in the egrid
        self.map_to_egrid("g", c_and_T_idx=True, prop_type="vector")
        self.map_to_egrid(prop_name="velocity", c_and_T_idx=False, prop_type="vector")

        print "average of the group velocity in e-grid!"
        print np.mean(self.egrid["n"]["velocity"], 0)

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
                fermi = self.egrid["fermi"][c][T]
                for tp in ["n", "p"]:
                    for ib in range(len(self.kgrid[tp]["energy"])):
                        for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                            E = self.kgrid[tp]["energy"][ib][ik]
                            v = self.kgrid[tp]["velocity"][ib][ik]

                            self.kgrid[tp]["f0"][c][T][ib][ik] = f0(E, fermi, T) * 1.0
                            self.kgrid[tp]["df0dk"][c][T][ib][ik] = hbar * df0dE(E, fermi, T) * v  # in cm
                            self.kgrid[tp]["electric force"][c][T][ib][ik] = -1 * \
                                                                             self.kgrid[tp]["df0dk"][c][T][ib][
                                                                                 ik] * default_small_E / hbar  # in 1/s
                            # self.kgrid[tp]["electric force"][c][T][ib][ik] = 1
                            self.kgrid[tp]["thermal force"][c][T][ib][ik] = - v * f0(E, fermi, T) * (1 - f0(E, fermi, T)) * ( \
                                E / (k_B * T) - self.egrid["Seebeck_integral_numerator"][c][T][tp] /
                                self.egrid["Seebeck_integral_denominator"][c][T][tp]) * dTdz / T

        self.map_to_egrid(prop_name="f0", c_and_T_idx=True, prop_type="vector")
        self.map_to_egrid(prop_name="df0dk", c_and_T_idx=True, prop_type="vector")

        # solve BTE in presence of electric and thermal driving force to get perturbation to Fermi-Dirac: g
        # if "POP" in self.inelastic_scatterings:
        self.solve_BTE_iteratively()

        self.calculate_transport_properties()

        # logging.debug('self.kgrid_to_egrid_idx["n"]: \n {}'.format(self.kgrid_to_egrid_idx["n"]))
        # logging.debug('self.kgrid["velocity"]["n"][0]: \n {}'.format(self.kgrid["n"]["velocity"][0]))
        # logging.debug('self.egrid["velocity"]["n"]: \n {}'.format(self.egrid["n"]["velocity"]))

        # kremove_list = ["W_POP", "effective mass", "kweights", "a", "c""",
        #                 "f_th", "g_th", "S_i_th", "S_o_th"]

        kremove_list = ["effective mass", "kweights", "a", "c""",
                        "f_th", "g_th", "S_i_th", "S_o_th"]

        for tp in ["n", "p"]:
            for rm in kremove_list:
                try:
                    del (self.kgrid[tp][rm])
                except:
                    pass
            for erm in ["all_en_flat", "f_th", "g_th", "S_i_th", "S_o_th"]:
                try:
                    del (self.egrid[tp][erm])
                except:
                    pass

        pprint(self.egrid["mobility"])
        if self.wordy:
            pprint(self.egrid)
            pprint(self.kgrid)


    def __getitem__(self, key):
        if key=="kgrid":
            return self.kgrid
        elif key=="egrid":
            return self.egrid
        else:
            raise KeyError



    def read_vrun(self, calc_dir=".", filename="vasprun.xml"):
        self._vrun = Vasprun(os.path.join(calc_dir, filename))
        self.volume = self._vrun.final_structure.volume
        logging.info("unitcell volume = {} A**3".format(self.volume))
        self.density = self._vrun.final_structure.density
        self._lattice_matrix = self._vrun.lattice_rec.matrix / (2 * pi)
        print self._lattice_matrix

        # kpoints and actual_kpoints are the same fractional k-points in pymatgen, just in different formats.
        # print self._vrun.kpoints
        # print self._vrun.actual_kpoints
        # print self._vrun.lattice_rec.get_cartesian_coords(self._vrun.actual_kpoints) #This gives the same result as np.dot(k, self._lattice_matrix*2*pi) I checked with [0.5, 0.5, 0.5]


        bs = self._vrun.get_band_structure()

        # Remember that python band index starts from 0 so bidx==9 refers to the 10th band (e.g. in VASP)
        cbm_vbm = {"n": {"kpoint": [], "energy": 0.0, "bidx": 0, "included": 0, "eff_mass_xx": [0.0, 0.0, 0.0]},
                   "p": {"kpoint": [], "energy": 0.0, "bidx": 0, "included": 0, "eff_mass_xx": [0.0, 0.0, 0.0]}}
        cbm = bs.get_cbm()
        vbm = bs.get_vbm()

        print "total number of bands"
        print self._vrun.get_band_structure().nb_bands
        # print bs.nb_bands

        cbm_vbm["n"]["energy"] = cbm["energy"]
        cbm_vbm["n"]["bidx"] = cbm["band_index"][Spin.up][0]
        cbm_vbm["n"]["kpoint"] = bs.kpoints[cbm["kpoint_index"][0]].frac_coords

        cbm_vbm["p"]["energy"] = vbm["energy"]
        cbm_vbm["p"]["bidx"] = vbm["band_index"][Spin.up][-1]
        cbm_vbm["p"]["kpoint"] = bs.kpoints[vbm["kpoint_index"][0]].frac_coords

        self.dft_gap = cbm["energy"] - vbm["energy"]
        print "DFT gap from vasprun.xml : {} eV".format(self.dft_gap)

        if self.soc:
            self.nelec = cbm_vbm["p"]["bidx"] + 1
            self.dos_normalization_factor = self._vrun.get_band_structure().nb_bands
        else:
            self.nelec = (cbm_vbm["p"]["bidx"]+1)*2
            self.dos_normalization_factor = self._vrun.get_band_structure().nb_bands*2

        print("total number of electrons nelec: {}".format(self.nelec))

        bs = bs.as_dict()
        if bs["is_spin_polarized"]:
            self.dos_emin = min(bs["bands"]["1"][0] , bs["bands"]["-1"][0])
            self.dos_emax = max(bs["bands"]["1"][-1] , bs["bands"]["-1"][-1])
        else:
            self.dos_emin = min(bs["bands"]["1"][0])
            self.dos_emax = max(bs["bands"]["1"][-1])

        if not self.poly_bands:
            for i, tp in enumerate(["n", "p"]):
                sgn = (-1)**i
                while abs(min(sgn*bs["bands"]["1"][cbm_vbm[tp]["bidx"]+sgn*cbm_vbm[tp]["included"]])-
                                          sgn*cbm_vbm[tp]["energy"])<self.Ecut:
                    cbm_vbm[tp]["included"] += 1

            # TODO: for now, I only include 1 band for each as I get some errors if I inlude more bands
                cbm_vbm[tp]["included"] = 1
        else:
            cbm_vbm["n"]["included"] = cbm_vbm["p"]["included"] = len(self.poly_bands)

# TODO: change this later if the band indecies are corrected in Analytical_band class
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



    #TODO: very inefficient code, see if you can change the way f is implemented
    # def get_E_idx(self, E, tp):
    #     """tp (str): "n" or "p" type"""
    #     min_Ediff = 1e30
    #     for ie, en in enumerate(self.egrid[tp]["energy"]):
    #         if abs(E-en)< min_Ediff:
    #             min_Ediff = abs(E-en)
    #             ie_select = ie
    #     return ie_select
    #
    #
    # def f(self, E, fermi, T, tp, c, alpha):
    #     """returns the perturbed Fermi-Dirac in presence of a small driving force"""
    #     return 1 / (1 + np.exp((E - fermi) / (k_B * T))) + self.egrid[tp]["g"][c][T][self.get_E_idx(E, tp)][alpha]



    def seeb_int_num(self, c, T):
        """wrapper function to do an integration taking only the concentration, c, and the temperature, T, as inputs"""
        fn = lambda E, fermi, T: f0(E, fermi, T) * (1 - f0(E, fermi, T)) * E / (k_B * T)
        return {t:self.integrate_over_DOSxE_dE(func=fn,tp=t,fermi=self.egrid["fermi"][c][T],T=T) for t in ["n", "p"]}



    def seeb_int_denom(self, c, T):
        """wrapper function to do an integration taking only the concentration, c, and the temperature, T, as inputs"""
        return {t:self.gs + self.integrate_over_E(prop_list=["f0x1-f0"],tp=t,c=c,T=T,xDOS=True) for t in ["n", "p"]}



    def calculate_property(self, prop_name, prop_func, for_all_E=False):
        """
        calculate the propery at all concentrations and Ts using the given function and insert it into self.egrid
        :param prop_name:
        :param prop_func (obj): the given function MUST takes c and T as required inputs in this order.
        :return:
        """
        if for_all_E:
            for tp in ["n", "p"]:
                self.egrid[tp][prop_name]={c:{T: [self.gs for E in self.egrid[tp]["energy"]] for T in self.temperatures}
                                             for c in self.dopings}
        else:
            self.egrid[prop_name] = {c: {T: self.gs for T in self.temperatures} for c in self.dopings}
        for c in self.dopings:
            for T in self.temperatures:
                if for_all_E:
                    fermi = self.egrid["fermi"][c][T]
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
               self.N_dis/self.volume**(1/3)*1e8*self.charge["dislocations"]** 2
        return N_II



    def init_egrid(self, dos_tp="simple"):
        """
        :param
            dos_tp (string): options are "simple", ...

        :return: an updated grid that contains the field DOS
        """

        self.egrid = {
            # "energy": {"n": [], "p": []},
            # "DOS": {"n": [], "p": []},
            # "all_en_flat": {"n": [], "p": []},
            "n": {"energy": [], "DOS": [], "all_en_flat": []},
            "p": {"energy": [], "DOS": [], "all_en_flat": []},
            "mobility": {}
             }
        self.kgrid_to_egrid_idx = {"n":[], "p":[]} # list of band and k index that are mapped to each memeber of egrid
        self.Efrequency = {"n": [], "p": []}

        # reshape energies of all bands to one vector:
        E_idx = {"n": [], "p": []}
        for tp in ["n", "p"]:
            for ib, en_vec in enumerate(self.kgrid[tp]["energy"]):
                self.egrid[tp]["all_en_flat"] += list(en_vec)
                # also store the flatten energy (i.e. no band index) as a tuple of band and k-indexes
                E_idx[tp] += [(ib, iek) for iek in range(len(en_vec))]

            # get the indexes of sorted flattened energy
            ieidxs = np.argsort(self.egrid[tp]["all_en_flat"])
            self.egrid[tp]["all_en_flat"] = [self.egrid[tp]["all_en_flat"][ie] for ie in ieidxs]

            # sort the tuples of band and energy based on their energy
            E_idx[tp] = [E_idx[tp][ie] for ie in ieidxs]


        # setting up energy grid and DOS:
        for tp in ["n", "p"]:
            energy_counter = []
            i = 0
            last_is_counted = False
            while i<len(self.egrid[tp]["all_en_flat"])-1:
                sum_e = self.egrid[tp]["all_en_flat"][i]
                counter = 1.0
                current_ib_ie_idx = [E_idx[tp][i]]
                j = i
                while j<len(self.egrid[tp]["all_en_flat"])-1 and (counter <= self.nE_min or \
                        abs(self.egrid[tp]["all_en_flat"][i]-self.egrid[tp]["all_en_flat"][j+1]) < self.dE_min):
                # while i < len(self.egrid[tp]["all_en_flat"]) - 1 and \
                #          self.egrid[tp]["all_en_flat"][i] == self.egrid[tp]["all_en_flat"][i + 1] :
                    counter += 1
                    current_ib_ie_idx.append(E_idx[tp][j+1])
                    sum_e += self.egrid[tp]["all_en_flat"][j+1]
                    if j+1 == len(self.egrid[tp]["all_en_flat"])-1:
                        last_is_counted = True
                    j+=1
                self.egrid[tp]["energy"].append(sum_e/counter)
                self.kgrid_to_egrid_idx[tp].append(current_ib_ie_idx)
                energy_counter.append(counter)

                if dos_tp.lower()=="simple":
                    self.egrid[tp]["DOS"].append(counter/len(self.egrid[tp]["all_en_flat"]))
                elif dos_tp.lower() == "standard":
                    self.egrid[tp]["DOS"].append(self.dos[self.get_Eidx_in_dos(sum_e/counter)][1])
                i = j + 1

            if not last_is_counted:
                self.egrid[tp]["energy"].append(self.egrid[tp]["all_en_flat"][-1])
                self.kgrid_to_egrid_idx[tp].append([E_idx[tp][-1]])
                if dos_tp.lower() == "simple":
                    self.egrid[tp]["DOS"].append(self.nelec / len(self.egrid[tp]["all_en_flat"]))
                elif dos_tp.lower() == "standard":
                    self.egrid[tp]["DOS"].append(self.dos[self.get_Eidx_in_dos(self.egrid[tp]["energy"][-1])][1])

            self.egrid[tp]["size"] = len(self.egrid[tp]["energy"])
            # if dos_tp.lower()=="standard":
            #     energy_counter = [ne/len(self.egrid[tp]["all_en_flat"]) for ne in energy_counter]
                #TODO: what is the best value to pick for width here?I guess the lower is more precisely at each energy?
                # dum, self.egrid[tp]["DOS"] = get_dos(self.egrid[tp]["energy"], energy_counter,width = 0.05)


        # logging.debug("here self.kgrid_to_egrid_idx: {}".format(self.kgrid_to_egrid_idx["n"]))
        # logging.debug(self.kgrid["n"]["energy"])


        for tp in ["n", "p"]:
            self.Efrequency[tp] = [len(Es) for Es in self.kgrid_to_egrid_idx[tp]]

        print "here total number of ks from self.Efrequency for n-type"
        print sum(self.Efrequency["n"])

        min_nE = 2
        if len(self.Efrequency["n"]) < min_nE or len(self.Efrequency["p"]) < min_nE:
            raise ValueError("The final egrid have fewer than {} energy values, AMSET stops now".format(min_nE))

        # initialize some fileds/properties
        self.egrid["calc_doping"] = {c: {T: {"n": 0.0, "p": 0.0} for T in self.temperatures} for c in self.dopings}
        for sn in self.elastic_scatterings + self.inelastic_scatterings +["overall", "average", "SPB_ACD"]:
            # self.egrid["mobility"+"_"+sn]={c:{T:{"n": 0.0, "p": 0.0} for T in self.temperatures} for c in self.dopings}
            self.egrid["mobility"][sn] = {c: {T: {"n": 0.0, "p": 0.0} for T in self.temperatures} for c in self.dopings}
        for transport in ["conductivity", "J_th", "seebeck", "TE_power_factor", "relaxation time constant"]:
            self.egrid[transport] = {c: {T: {"n": 0.0, "p": 0.0} for T in self.temperatures} for c in self.dopings}

        # populate the egrid at all c and T with properties; they can be called via self.egrid[prop_name][c][T] later
        self.calculate_property(prop_name="fermi", prop_func=self.find_fermi)
        for c in self.dopings:
            for T in self.temperatures:
                print "Fermi level at {} 1/cm3 and {} K: {}".format(c, T, self.egrid["fermi"][c][T])

        # Since the SPB generated band structure may have several valleys, it's better to use the Fermi calculated from the actual band structure
        # self.calculate_property(prop_name="fermi_SPB", prop_func=self.find_fermi_SPB)

        #
        ##  in case specific fermi levels are to be tested:



        # self.egrid["fermi"]= {
        #     -1e+20: {
        #         300.0: 1.2166,
        #         600.0: 1.1791
        #     } }



        # self.calculate_property(prop_name="f0", prop_func=f0, for_all_E=True)
        # self.calculate_property(prop_name="f", prop_func=f0, for_all_E=True)
        # self.calculate_property(prop_name="f_th", prop_func=f0, for_all_E=True)

        for prop in ["f", "f_th"]:
            self.map_to_egrid(prop_name=prop, c_and_T_idx=True)

        self.calculate_property(prop_name="f0x1-f0", prop_func=lambda E, fermi, T: f0(E, fermi, T)
                                                                        * (1 - f0(E, fermi, T)), for_all_E=True)
        self.calculate_property(prop_name="beta", prop_func=self.inverse_screening_length)

        # self.egrid["beta"] =  {
        #             -1e+20: {
        #                 300: {
        #                     "n": 0.36166217210814655,
        #                     "p": 7.366123307507122e-16
        #                 },
        #                 600: {
        #                     "n": 0.2643208622509363,
        #                     "p": 6.141492700626864e-05
        #                 }
        #             }
        #         }

        self.calculate_property(prop_name="N_II", prop_func=self.calculate_N_II)
        self.calculate_property(prop_name="Seebeck_integral_numerator", prop_func=self.seeb_int_num)
        self.calculate_property(prop_name="Seebeck_integral_denominator", prop_func=self.seeb_int_denom)



    def get_Eidx_in_dos(self, E, Estep=None):
        if not Estep:
            Estep = max(self.dE_min, 0.0001)
        return int(round((E - self.dos_emin) / Estep))

        # return min(int(round((E - self.dos_emin) / Estep)) , len(self.dos)-1)



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
        return (a * self.kgrid[tp]["a"][ib_prm][ik_prm]+ X * c * self.kgrid[tp]["c"][ib_prm][ik_prm])**2



    def cos_angle(self, v1, v2):
        """
        Args:
            v1, v2 (np.array): vectors
        return:
            the cosine of the angle between twp numpy vectors: v1 and v2"""
        norm_v1, norm_v2 = norm(v1), norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 1.0  # In case of the two points are the origin, we assume 0 degree; i.e. no scattering: 1-X==0
        else:
            return np.dot(v1, v2) / (norm_v1 * norm_v2)



    def remove_indexes(self, rm_idx_list, rearranged_props):
        """
        The k-points with velocity < 1 cm/s (either in valence or conduction band) are taken out as those are
            troublesome later with extreme values (e.g. too high elastic scattering rates)
        :param rm_idx_list ([int]): the kpoint indexes that need to be removed for each property
        :param rearranged_props ([str]): list of properties for which some indexes need to be removed
        :return:
        """




        # omit the points with v~0 and try to find the parabolic band equivalent effective mass at the CBM and the VBM
        # temp_min = {"n": self.gl, "p": self.gl}
        for i, tp in enumerate(["n", "p"]):
            for ib in range(self.cbm_vbm[tp]["included"]):
                rm_idx_list_ib = list(set(rm_idx_list[tp][ib]))
                rm_idx_list_ib.sort(reverse=True)
                rm_idx_list[tp][ib] = rm_idx_list_ib
                logging.debug("# of kpoints indexes with low velocity: {}".format(len(rm_idx_list_ib)))
            for prop in rearranged_props:
                self.kgrid[tp][prop] = np.array([np.delete(self.kgrid[tp][prop][ib], rm_idx_list[tp][ib], axis=0)\
                                    for ib in range(self.cbm_vbm[tp]["included"])])
            # for ib in range(self.cbm_vbm[tp]["included"]):

                # for ik in rm_idx_list:
                #     if (-1)**i * self.kgrid[tp]["energy"][ib][ik] < temp_min[tp]:
                #         temp_min[tp] = (-1)**i * self.kgrid[tp]["energy"][ib][ik]
                #         self.cbm_vbm[tp]["eff_mass_xx"]=(-1)**i*self.kgrid[tp]["effective mass"][ib][ik].diagonal()
                #         self.cbm_vbm[tp]["energy"] = self.kgrid[tp]["energy"][ib][ik]


                # for prop in rearranged_props:
                #     self.kgrid[tp][prop][ib] = np.delete(self.kgrid[tp][prop][ib], rm_idx_list_ib, axis=1)



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
            initial_val = [ [initval, initval, initval], [initval, initval, initval], [initval, initval, initval] ]

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
                        raise ValueError("For now keep using is_nparray=True to see why for not is_nparray everything becomes equal at all temepratures (lists are not copied but they are all the same)")
                    else:
                        if not c_T_idx:
                            self[grid][tp][name] = init_content
                        else:
                            self[grid][tp][name] = {c: {T: init_content for T in self.temperatures} for c in self.dopings}


    @staticmethod
    def remove_duplicate_kpoints(kpts, dk = 0.0001):
        """kpts (list of list): list of coordinates of electrons
         ALWAYS return either a list or ndarray: BE CONSISTENT with the input!!!

         Attention: it is better to call this method only once as calculating the norms takes time.
         """
        start_time = time.time()

        rm_list = []

        kdist = [norm(k) for k in kpts]
        ktuple = zip(kdist, kpts)
        ktuple.sort(key=lambda x: x[0])
        kpts = [tup[1] for tup in ktuple]

        # kpts.sort(key=lambda x: norm(x))

        i = 0
        while i < len(kpts)-1:
            j = i
            while j < len(kpts)-1 and ktuple[j+1][0] - ktuple[i][0] < dk :

            # for i in range(len(kpts)-2):
                # if kpts[i][0] == kpts[i+1][0] and kpts[i][1] == kpts[i+1][1] and kpts[i][2] == kpts[i+1][2]:

                if (abs(kpts[i][0]-kpts[j+1][0])<dk or abs(kpts[i][0])==abs(kpts[j+1][0])==0.5) and \
                    (abs(kpts[i][1]-kpts[j+1][1]) < dk or abs(kpts[i][1]) == abs(kpts[j+1][1]) == 0.5) and \
                    (abs(kpts[i][2]-kpts[j+1][2]) < dk or abs(kpts[i][2]) == abs(kpts[j+1][2]) == 0.5):
                #
                # if abs(kpts[i][0] - kpts[j + 1][0]) < dk and \
                #     abs(kpts[i][1] - kpts[j + 1][1]) < dk and \
                #     abs(kpts[i][2] - kpts[j + 1][2]) < dk:
                    rm_list.append(j+1)
                j += 1
            i+=1


        # The reason the following does NOT work is this example: [[0,3,4], [4,3,0], [0.001, 3, 4]]: In this example,
        # the k-points are correctly sorted based on their norm but 0&1 or 1&2 are NOT equal but 0&3 are but not captured
        # for i in range(len(kpts)-2):
        #     if (abs(kpts[i][0]-kpts[i+1][0])<dk or abs(kpts[i][0])==abs(kpts[i+1][0])==0.5) and \
        #             (abs(kpts[i][1]-kpts[i+1][1]) < dk or abs(kpts[i][1]) == abs(kpts[i+1][1]) == 0.5) and \
        #             (abs(kpts[i][2]-kpts[i+1][2]) < dk or abs(kpts[i][2]) == abs(kpts[i+1][2]) == 0.5):
        #             rm_list.append(i+1)




        kpts = np.delete(kpts, rm_list, axis=0)
        kpts = list(kpts)


        # even if this works (i.e. the shape of kpts is figured out, etc), it's not good as does not consider 0.0001 and 0.0002 equal
        # kpts = np.vstack({tuple(row) for row in kpts})



        # TODO: failed attempt to speed up duplication removal process by sorting the k-points in one direction first
        # kpts.sort(key=lambda x: x[0])
        # old_idx = 0
        # for i in range(len(kpts)-2):
        #     # look for duplicate k-points in a sublist of kpts where kx values are almost equal
        #     # if abs(abs(kpts[i][0]) - 0.5) < 0.0001 and abs(abs(kpts[i][1]) - 0.5) < 0.0001 and abs(abs(kpts[i][2]) - 0.5) < 0.0001:
        #     #     rm_list.append(i)
        #     #     continue
        #     if abs(kpts[i][0]-kpts[i + 1][0]) > 0.0001:
        #         for j in range(old_idx, i):
        #             for jj in range(j+1, i+1):
        #                 if abs(kpts[j][1]-kpts[jj][1])<0.0001 and abs(kpts[j][2]-kpts[jj][2])<0.0001:
        #                     rm_list.append(jj)
        #         old_idx = i+1


        # CORRECT BUT TIME CONSUMING WAY OF REMOVING DUPLICATES
        # for i in range(len(kpts)-2):
        #     # if abs(abs(kpts[i][0]) - 0.5) < 0.0001 and abs(abs(kpts[i][1]) - 0.5) < 0.0001 and abs(abs(kpts[i][2]) - 0.5) < 0.0001:
        #     #     rm_list.append(i)
        #     #     continue
        #     for j in range(i+1, len(kpts)-1):
        #         if (abs(kpts[i][0] - kpts[j][0]) < 0.0001 or abs(kpts[i][0])==abs(kpts[j][0])==0.5) and \
        #                 (abs(kpts[i][1] - kpts[j][1]) < 0.0001 or abs(kpts[i][1]) == abs(kpts[j][1]) == 0.5) and\
        #             (abs(kpts[i][2] - kpts[j][2]) < 0.0001 or abs(kpts[i][2]) == abs(kpts[j][2]) == 0.5):
        #
        #             rm_list.append(j)
        #
        # kpts = np.delete(kpts, rm_list, axis=0)
        # kpts = list(kpts)


        print "total time to remove duplicate k-points = {} seconds".format(time.time() - start_time)
        print "number of duplicates removed:"
        print len(rm_list)


        # print kpts

        return kpts



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
        return [[k1[i] + n*dk[i] for i in  range(len(k1))] for n in range(1, nsteps+1)]



    @staticmethod
    def get_perturbed_ks(k):
        all_perturbed_ks = []
        # for p in [0.01, 0.03, 0.05]:
        for p in [0.05, 0.1]:
                all_perturbed_ks.append([ k_i+p*np.sign(random()-0.5) for k_i in k] )
        return all_perturbed_ks



    def get_ks_with_intermediate_energy(self, kpts, energies, max_Ediff = None, target_Ediff = None):
        final_kpts_added = []
        max_Ediff = max_Ediff or min(self.Ecut, 10*k_B*max(self.temperatures))
        target_Ediff = target_Ediff or self.dE_min
        for tp in ["n", "p"]:
            if tp not in self.all_types:
                continue
            ies_sorted = list(np.argsort(energies[tp]))
            if tp=="p":
                ies_sorted.reverse()
            for idx, ie in enumerate(ies_sorted[:-1]):
                Ediff = abs(energies[tp][ie] - energies[tp][ies_sorted[0]])
                if Ediff > max_Ediff:
                    break
                final_kpts_added += self.get_perturbed_ks(kpts[ies_sorted[idx]])

                # final_kpts_added += self.get_intermediate_kpoints_list(list(kpts[ies_sorted[idx]]),
                #                                    list(kpts[ies_sorted[idx+1]]), max(int(Ediff/target_Ediff) , 1))
        return  self.kpts_to_first_BZ(final_kpts_added)


    def get_adaptive_kpoints(self, kpts, energies, adaptive_Erange, nsteps):
        #TODO: make this function which is meant to be called several times more efficient to sort energies ONLY ONCE outside of the function
        kpoints_added = {"n": [], "p": []}
        for tp in ["n", "p"]:
            if tp not in self.all_types:
                continue
            # TODO: if this worked, change it so that if self.dopings does not involve either of the types, don't add k-points for it
            ies_sorted = list(np.argsort(energies[tp]))
            if tp=="p":
                ies_sorted.reverse()
            # print ies_sorted
            # print len(ies_sorted)
            # print len(self.kgrid[tp]["energy"][0])
            for ie in ies_sorted:
                Ediff = abs(energies[tp][ie] - energies[tp][ies_sorted[0]])
                if Ediff >= adaptive_Erange[0] and Ediff < adaptive_Erange[-1]:
                    kpoints_added[tp].append(kpts[ie])


        print "here initial k-points with low energy distance"
        print len(kpoints_added["n"])
        # print kpoints_added["n"]
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
        fractional_ks = [np.dot(k, self.rotations[i]) + self.translations[i] for i in range(len(self.rotations))]
        if cartesian:
            return [np.dot(k, self._lattice_matrix/A_to_nm*2*pi) for k in self.kpts_to_first_BZ(fractional_ks)]
        else:
            return self.kpts_to_first_BZ(fractional_ks)



    def init_kgrid(self,coeff_file, kgrid_tp="coarse"):
        Tmx = max(self.temperatures)
        if kgrid_tp=="coarse":
            nkstep = self.nkibz

        sg = SpacegroupAnalyzer(self._vrun.final_structure)
        self.rotations, self.translations = sg._get_symmetry() # this returns unique symmetry operations


        # test_k = [0.5, 0.5, 0.5]
        # print "equivalent ks"
        # # a = self.remove_duplicate_kpoints([np.dot(test_k, self.rotations[i]) + self.translations[i] for i in range(len(self.rotations))])
        # a = self.get_sym_eq_ks_in_first_BZ(test_k)
        # # a = self.remove_duplicate_kpoints(a)
        # # a = [np.dot(test_k, self.rotations[i]) + self.translations[i] for i in range(len(self.rotations))]
        # a = [i.tolist() for i in self.remove_duplicate_kpoints(a)]
        # print a # would print [[-0.5, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, -0.5], [0.5, 0.5, 0.5]]

        # TODO: is_shift with 0.03 for y and 0.06 for z might give an error due to _all_elastic having twice length in kgrid compared to S_o, etc. I haven't figured out why
        # kpts_and_weights = sg.get_ir_reciprocal_mesh(mesh=(nkstep, nkstep, nkstep), is_shift=(0.00, 0.00, 0.00))


        logging.info("self.nkibz = {}".format(self.nkibz))

        #TODO: the following is NOT a permanent solution to speed up generation/loading of k-mesh, speed up get_ir_reciprocal_mesh later
        # ibzkpt_filename = "all_ibzkpt.json"
        # try:
        #     with open(ibzkpt_filename, 'r') as fp:
        #         all_kpts = json.load(fp, cls=MontyDecoder)
        # except:
        #     logging.info('reading {} failed!'.format(ibzkpt_filename))
        #     all_kpts = {}
        # try:
        #     kpts = all_kpts["{}x{}x{}".format(nkstep, nkstep, nkstep)]
        #     logging.info('reading {}x{}x{} k-mesh from "{}"'.format(nkstep, nkstep, nkstep, ibzkpt_filename))
        # except:
        #     logging.info("generating {}x{}x{} IBZ k-mesh".format(nkstep, nkstep, nkstep))
        #     kpts_and_weights = sg.get_ir_reciprocal_mesh(mesh=(nkstep, nkstep, nkstep), is_shift=[0, 0, 0])
        #     kpts = [i[0] for i in kpts_and_weights]
        #     kpts = self.kpts_to_first_BZ(kpts)
        #
        # all_kpts["{}x{}x{}".format(nkstep, nkstep, nkstep)] = kpts
        # os.system("cp {} {}_backup.json".format(ibzkpt_filename, ibzkpt_filename.split(".")[0]))
        # with open(ibzkpt_filename, 'w') as fp:
        #     json.dump(all_kpts, fp, cls=MontyEncoder)


        all_kpts = {}
        try:
            ibzkpt_filename = os.path.join(os.environ["AMSET_ROOT"], "{}_ibzkpt.json".format(nkstep))
        except:
            ibzkpt_filename = "{}_ibzkpt.json".format(nkstep)
        try:
            with open(ibzkpt_filename, 'r') as fp:
                all_kpts = json.load(fp, cls=MontyDecoder)
            kpts = all_kpts["{}x{}x{}".format(nkstep, nkstep, nkstep)]
            logging.info('reading {}x{}x{} k-mesh from "{}"'.format(nkstep, nkstep, nkstep, ibzkpt_filename))
        except:
            logging.info('reading {} failed!'.format(ibzkpt_filename))
            logging.info("generating {}x{}x{} IBZ k-mesh".format(nkstep, nkstep, nkstep))
            kpts_and_weights = sg.get_ir_reciprocal_mesh(mesh=(nkstep, nkstep, nkstep), is_shift=[0, 0, 0])
            kpts = [i[0] for i in kpts_and_weights]
            kpts = self.kpts_to_first_BZ(kpts)
            all_kpts["{}x{}x{}".format(nkstep, nkstep, nkstep)] = kpts
            with open(ibzkpt_filename, 'w') as fp:
                json.dump(all_kpts, fp, cls=MontyEncoder)



        # the following end up with 2 kpoints withing 10 kB * 300 ONLY!!!
        # kpts = []
        # nsteps = 17
        # step = 0.5/nsteps
        # for ikx in range(nsteps+1):
        #     for iky in range(nsteps+1):
        #         for ikz in range(nsteps+1):
        #             kpts.append([0.0 + ikx*step, 0.0 + iky*step, 0.0 + ikz*step])

        # explicitly add the CBM/VBM k-points to calculate the parabolic band effective mass hence the relaxation time
        kpts.append(self.cbm_vbm["n"]["kpoint"])
        kpts.append(self.cbm_vbm["p"]["kpoint"])

        print "number of original ibz k-points"
        print len(kpts)


        if not self.poly_bands:
            logging.debug("start interpolating bands from {}".format(coeff_file))
            analytical_bands = Analytical_bands(coeff_file=coeff_file)
            all_ibands = []
            for i, tp in enumerate(["p", "n"]):
                for ib in range(self.cbm_vbm[tp]["included"]):
                    sgn = (-1) ** (i+1)
                    all_ibands.append(self.cbm_vbm[tp]["bidx"] + sgn * ib)

            start_time = time.time()
            logging.debug("all_ibands: {}".format(all_ibands))

            engre, latt_points, nwave, nsym, nsymop, symop, br_dir = analytical_bands.get_engre(iband=all_ibands)
            nstv, vec, vec2 = analytical_bands.get_star_functions(latt_points, nsym, symop, nwave, br_dir=br_dir)
            out_vec2 = np.zeros((nwave, max(nstv), 3, 3))
            for nw in xrange(nwave):
                for i in xrange(nstv[nw]):
                    out_vec2[nw, i] = outer(vec2[nw, i], vec2[nw, i])

            print("time to get engre and calculate the outvec2: {} seconds".format(time.time() - start_time))

        else:
            # first modify the self.poly_bands to include all symmetrically equivalent k-points (k_i)
            # these points will be used later to generate energy based on the minimum norm(k-k_i)
            for ib in range(len(self.poly_bands)):
                for j in range(len(self.poly_bands[ib])):
                    self.poly_bands[ib][j][0] = self.remove_duplicate_kpoints(
                        self.get_sym_eq_ks_in_first_BZ(self.poly_bands[ib][j][0],cartesian=True))


        # calculate only the CBM and VBM energy values (ib == 0)
        # here we assume that the cbm and vbm k-points read from vasprun.xml are correct:
        for i, tp in enumerate(["p", "n"]):
            sgn = (-1) ** i
            if not self.poly_bands:
                energy, de, dde = analytical_bands.get_energy(
                    self.cbm_vbm[tp]["kpoint"], engre[i * self.cbm_vbm["p"]["included"] + 0],
                    nwave, nsym, nstv, vec, vec2, out_vec2, br_dir=br_dir)
                energy = energy * Ry_to_eV - sgn * self.scissor / 2.0
                effective_m = hbar ** 2 / (
                        dde * 4 * pi ** 2) / m_e / A_to_m ** 2 * e * Ry_to_eV

            else:
                energy, velocity, effective_m = get_poly_energy(
                    np.dot(self.cbm_vbm[tp]["kpoint"], self._lattice_matrix / A_to_nm * 2 * pi),
                    poly_bands=self.poly_bands, type=tp, ib=0, bandgap=self.dft_gap + self.scissor)

            self.offset_from_vrun = energy - self.cbm_vbm[tp]["energy"]
            logging.debug("offset from vasprun energy values for {}-type = {} eV".format(tp, self.offset_from_vrun))
            self.cbm_vbm[tp]["energy"] = energy
            self.cbm_vbm[tp]["eff_mass_xx"] = effective_m.diagonal()

        # if not self.poly_bands:
        #     self.dos_emax += self.offset_from_vrun
        #     self.dos_emin += self.offset_from_vrun

        logging.debug("cbm_vbm after recalculating their energy values:\n {}".format(self.cbm_vbm))
        self._avg_eff_mass = {tp: abs(np.mean(self.cbm_vbm["n"]["eff_mass_xx"])) for tp in ["n", "p"]}

        # calculate the  in initial ibz k-points and look at the first band to decide on additional/adaptive ks
        # temp_min = {"n": self.gl, "p": self.gl}
        energies = {"n": [0.0 for ik in kpts], "p": [0.0 for ik in kpts]}
        rm_list = {"n": [], "p": []}
        for i, tp in enumerate(["p", "n"]):
            sgn = (-1) ** i
            # for ib in range(self.cbm_vbm[tp]["included"]):
            for ib in [0]: # we only include the first band now (same for energies) to decide on ibz k-points
                for ik in range(len(kpts)):
                    if not self.poly_bands:
                        energy, de, dde = analytical_bands.get_energy(
                            kpts[ik], engre[i*self.cbm_vbm["p"]["included"]+ib],
                                nwave, nsym, nstv, vec, vec2, out_vec2, br_dir=br_dir)
                        energy = energy * Ry_to_eV - sgn * self.scissor / 2.0
                        velocity = abs(de / hbar * A_to_m * m_to_cm * Ry_to_eV)
                        # effective_m = hbar ** 2 / (
                        # dde * 4 * pi ** 2) / m_e / A_to_m ** 2 * e * Ry_to_eV
                        energies[tp][ik] = energy
                    else:
                        energy,velocity,effective_m=get_poly_energy(np.dot(kpts[ik],self._lattice_matrix/A_to_nm*2*pi),
                                poly_bands=self.poly_bands, type=tp, ib=ib, bandgap=self.dft_gap + self.scissor)
                        energies[tp][ik] = energy


                    # TODO: this may be a better place to get rid of k-points that have off-energy values to avoid calculating their energy early on
                    # if (-1) ** (i + 1) * energy < temp_min[tp]:
                    #     temp_min[tp] = (-1) ** (i + 1) * energy
                    #     self.cbm_vbm[tp]["eff_mass_xx"] = effective_m
                    #     self.cbm_vbm[tp]["energy"] = energy

                    if velocity[0] < 1 or velocity[1] < 1 or velocity[2] < 1 or \
                            abs(energy - self.cbm_vbm[tp]["energy"]) > self.Ecut:
                        rm_list[tp].append(ik)

        # logging.debug("energies before removing k-points with off-energy:\n {}".format(energies))
        kpts = np.delete(kpts, list(set(rm_list["n"]+rm_list["p"])), axis=0)
        kpts = list(kpts)

        print "number of ibz k-points AFTER ENERGY-FILTERING"
        print len(kpts)

        logging.debug("initial # of kpts after off-energy points are removed: {}".format(len(kpts)))
        for tp in ["n", "p"]:
            energies[tp] = np.delete(energies[tp], list(set(rm_list["n"]+rm_list["p"])), axis=0)

        if self.adaptive_mesh:

            all_added_kpoints = []
            all_added_kpoints += self.get_adaptive_kpoints(kpts, energies,adaptive_Erange=[0*k_B*Tmx, 1*k_B*Tmx], nsteps=30)

            # it seems it works mostly at higher energy values!
            # all_added_kpoints += self.get_ks_with_intermediate_energy(kpts,energies)

            print "here the number of added k-points"
            print len(all_added_kpoints)
            print all_added_kpoints
            print type(kpts)

            kpts += all_added_kpoints


        symmetrically_equivalent_ks = []
        for k in kpts:
            symmetrically_equivalent_ks += self.get_sym_eq_ks_in_first_BZ(k)
        kpts += symmetrically_equivalent_ks
        kpts = self.remove_duplicate_kpoints(kpts)

        if len(kpts) < 3:
            raise ValueError("The k-point mesh is too loose (number of kpoints = {}) "
                             "after filtering the initial k-mesh".format(len(kpts)))

        logging.debug("number of kpoints after symmetrically equivalent kpoints are added: {}".format(len(kpts)))

        kweights = [1.0 for i in kpts]

        self.kgrid = {
                "n": {},
                "p": {} }


        for tp in ["n", "p"]:
            self.kgrid[tp]["kpoints"] = [[k for k in kpts] for ib in range(self.cbm_vbm[tp]["included"])]
            self.kgrid[tp]["kweights"] = [[kw for kw in kweights] for ib in range(self.cbm_vbm[tp]["included"])]

        self.initialize_var("kgrid", ["energy", "a", "c", "norm(v)"], "scalar", 0.0, is_nparray=False, c_T_idx=False)
        self.initialize_var("kgrid", ["velocity"], "vector", 0.0, is_nparray=False, c_T_idx=False)
        self.initialize_var("kgrid", ["effective mass"], "tensor", 0.0, is_nparray=False, c_T_idx=False)

        start_time = time.time()

        rm_idx_list={"n":[[] for i in range(self.cbm_vbm["n"]["included"])],
                     "p": [[]for i in range(self.cbm_vbm["p"]["included"])]}
        self.initialize_var("kgrid", ["cartesian kpoints"], "vector", 0.0, is_nparray=False, c_T_idx=False)
        self.initialize_var("kgrid", ["norm(k)"], "scalar", 0.0, is_nparray=False, c_T_idx=False)

        # logging.debug("here the initial n-kpoints:\n {}".format(self.kgrid["n"]["kpoints"]))
        # logging.debug("here the initial p-kpoints:\n {}".format(self.kgrid["p"]["kpoints"]))
        # initialize energy, velocity, etc in self.kgrid

        for i, tp in enumerate(["p", "n"]):
            sgn = (-1) ** i
            for ib in range(self.cbm_vbm[tp]["included"]):
                self.kgrid[tp]["cartesian kpoints"][ib]=np.dot(np.array(self.kgrid[tp]["kpoints"][ib]),self._lattice_matrix)/A_to_nm*2*pi #[1/nm]
                self.kgrid[tp]["norm(k)"][ib] = [norm(k) for k in self.kgrid[tp]["cartesian kpoints"][ib]]


                for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                    if not self.poly_bands:
                        energy, de, dde = analytical_bands.get_energy(
                            self.kgrid[tp]["kpoints"][ib][ik], engre[i*self.cbm_vbm["p"]["included"]+ib],
                                nwave, nsym, nstv, vec, vec2, out_vec2, br_dir=br_dir)
                        energy = energy * Ry_to_eV - sgn * self.scissor/2.0
                        velocity = abs(de / hbar * A_to_m * m_to_cm * Ry_to_eV)# to get v in cm/s
                        effective_mass = hbar ** 2 / (
                        dde * 4 * pi ** 2) / m_e / A_to_m ** 2 * e * Ry_to_eV  # m_tensor: the last part is unit conversion
                    else:
                        energy, velocity, effective_mass=get_poly_energy(self.kgrid[tp]["cartesian kpoints"][ib][ik],
                                                                              poly_bands=self.poly_bands,
                            type=tp, ib=ib,bandgap=self.dft_gap+self.scissor)

                    # velocity /= 2

                    self.kgrid[tp]["energy"][ib][ik] = energy
                    self.kgrid[tp]["velocity"][ib][ik] = velocity
                    self.kgrid[tp]["norm(v)"][ib][ik] = norm(velocity)

                    # self.kgrid[tp]["velocity"][ib][ik] = de/hbar * A_to_m * m_to_cm * Ry_to_eV # to get v in units of cm/s
                    # TODO: what's the implication of negative group velocities? check later after scattering rates are calculated
                    # TODO: actually using abs() for group velocities mostly increase nu_II values at each energy
                    # TODO: should I have de*2*pi for the group velocity and dde*(2*pi)**2 for effective mass?
                    if self.kgrid[tp]["velocity"][ib][ik][0] < 100 or self.kgrid[tp]["velocity"][ib][ik][1] < 100 \
                            or self.kgrid[tp]["velocity"][ib][ik][2] < 100 or \
                            abs(self.kgrid[tp]["energy"][ib][ik] - self.cbm_vbm[tp]["energy"]) > self.Ecut:
                        rm_idx_list[tp][ib].append(ik)
                    self.kgrid[tp]["effective mass"][ib][ik] = effective_mass
                    self.kgrid[tp]["a"][ib][ik] = 1.0


        logging.debug("k-indexes to be removed: \n {}".format(rm_idx_list))

        # for i, tp in enumerate(["p", "n"]):
        #     for ib in range(self.cbm_vbm[tp]["included"]):
        #         for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
        #             if (-1)**(i+1) * (self.kgrid[tp]["energy"][ib][ik] - self.cbm_vbm[tp]["energy"]) > self.Ecut:
        #                 rm_idx_list[tp][ib].append(ik)

        print "average of the group velocity (to detect inherent or artificially created anisotropy"
        print np.mean(self.kgrid["n"]["velocity"][0], 0)


        rearranged_props = ["velocity","effective mass","energy", "a", "c", "kpoints","cartesian kpoints","kweights",
                             "norm(v)", "norm(k)"]

        if self.poly_bands:
            self.dos_emin = min(self.kgrid["p"]["energy"][-1])
            self.dos_emax = max(self.kgrid["n"]["energy"][-1])

        #TODO: the following is temporary, for some reason if # of kpts in different bands are NOT the same,
        # I get an error that _all_elastic is a list! so 1/self.kgrid[tp]["_all_elastic"][c][T][ib] cause error int/list!
        for tp in ["n", "p"]:
            rm_idx_list[tp] = [rm_idx_list[tp][0] for ib in range(self.cbm_vbm[tp]["included"])]

        # remove the k-points with off-energy values (>Ecut away from CBM/VBM) that are not removed already
        self.remove_indexes(rm_idx_list, rearranged_props=rearranged_props)


        # # emin & emax were initialized based on the real input band structure but that should change if self.poly_bands
        # self.dos_emin = min(self.kgrid["p"]["energy"][-1])
        # # logging.debug('here test self.kgrid["n"]["energy"]\n {}'.format(self.kgrid["n"]["energy"]))
        # self.dos_emax = max(self.kgrid["n"]["energy"][-1])

        print "emin and emax"
        print self.dos_emin
        print self.dos_emax


        print "here the final number of k-points"
        print len(self.kgrid["n"]["kpoints"][0])

        if len(self.kgrid["n"]["kpoints"][0]) < 5:
            raise ValueError("VERY BAD k-mesh; please change the setting for k-mesh and try again!")

        print("time to calculate energy, velocity, m* for all: {} seconds".format(time.time() - start_time))
        # print self.kgrid["n"]["energy"]

        # sort "energy", "kpoints", "kweights", etc based on energy in ascending order
        self.sort_vars_based_on_energy(args=rearranged_props, ascending=True)

        for tp in ["n", "p"]:
            for ib in range(len(self.kgrid[tp]["energy"])):
                print "length of final {}-kpts in band #{}: {}".format(tp, ib, len(self.kgrid[tp]["kpoints"][ib]))


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
                        self.kgrid[tp]["N_POP"][c][T][ib] = np.array([ 1/(np.exp(hbar * W_POP/(k_B * T))-1) for W_POP in self.kgrid[tp]["W_POP"][ib]])

        self.initialize_var(grid="kgrid", names=["_all_elastic", "S_i", "S_i_th", "S_o", "S_o_th", "g", "g_th", "g_POP",
                "f", "f_th", "relaxation time", "df0dk", "electric force", "thermal force"],
                        val_type="vector", initval=self.gs, is_nparray=True, c_T_idx=True)


        self.initialize_var("kgrid", ["f0", "f_plus", "f_minus","g_plus", "g_minus"], "vector", self.gs, is_nparray=True, c_T_idx=True)
        # self.initialize_var("kgrid", ["lambda_i_plus", "lambda_i_minus"]
        #                     , "vector", self.gs, is_nparray=True, c_T_idx=False)


        if not self.poly_bands:
            # caluclate and normalize the global density of states (DOS) so the integrated DOS == total number of electrons
            emesh, dos=analytical_bands.get_dos_from_scratch(self._vrun.final_structure,[self.nkdos,self.nkdos,self.nkdos],
                        self.dos_emin, self.dos_emax, int(round((self.dos_emax-self.dos_emin)/max(self.dE_min, 0.0001)))+1, width=self.dos_bwidth)
        else:
            logging.debug("here self.poly_bands: \n {}".format(self.poly_bands))

            # now construct the DOS
            emesh, dos = get_dos_from_poly_bands(self._vrun.final_structure, self._lattice_matrix,
                                                 [self.nkdos, self.nkdos, self.nkdos],
                                                 self.dos_emin, self.dos_emax,
                                                 int(round((self.dos_emax - self.dos_emin) / max(self.dE_min, 0.0001)))+1,
                                                 poly_bands=self.poly_bands,
                                                 # bandgap=self.dft_gap + self.scissor,
                                                 bandgap=self.cbm_vbm["n"]["energy"] - self.cbm_vbm["p"][
                                                     "energy"] + self.scissor,
                                                 width=self.dos_bwidth, SPB_DOS=True)
            # total_nelec = len(self.poly_bands) * 2 # basically 2x number of included occupied bands (valence bands)
            # total_nelec = self.nelec

            self.dos_normalization_factor = len(
                self.poly_bands)*2*2  # it is *2 elec/band & *2 because DOS is repeated in valence/conduction

        integ = 0.0
        for idos in range(len(dos) - 2):
            # if emesh[idos] > self.cbm_vbm["n"]["energy"]: # we assume anything below CBM as 0 occupation
            #     break
            integ += (dos[idos + 1] + dos[idos]) / 2 * (emesh[idos + 1] - emesh[idos])
        # normalize DOS
        # logging.debug("dos before normalization: \n {}".format(zip(emesh, dos)))
        # dos = [g / integ * self.nelec for g in dos]
        dos = [g / integ * self.dos_normalization_factor for g in dos]

        # logging.debug("integral of dos: {} stoped at index {} and energy {}".format(integ, idos, emesh[idos]))

        self.dos = zip(emesh, dos)
        # logging.debug("dos after normalization: \n {}".format(self.dos))

        self.dos = [list(a) for a in self.dos]



    def sort_vars_based_on_energy(self, args, ascending=True):
        """sort the list of variables specified by "args" (type: [str]) in self.kgrid based on the "energy" values
        in each band for both "n"- and "p"-type bands and in ascending order by default."""

        for tp in ["n", "p"]:
            for ib in range(self.cbm_vbm[tp]["included"]):
                ikidxs = np.argsort(self.kgrid[tp]["energy"][ib])
                if not ascending:
                    ikidxs.reverse()
                for arg in args:
                    # if arg in ["k-points", "kweights"]:
                    #     self.kgrid[arg] = np.array([self.kgrid[arg][ik] for ik in ikidxs])
                    # else:
                    self.kgrid[tp][arg][ib] = np.array([self.kgrid[tp][arg][ib][ik] for ik in ikidxs])


    def generate_angles_and_indexes_for_integration(self, avg_Ediff_tolerance=0.02):


        self.initialize_var("kgrid",["X_E_ik", "X_Eplus_ik", "X_Eminus_ik"],"scalar",[],is_nparray=False, c_T_idx=False)

        # elastic scattering
        for tp in ["n", "p"]:
            for ib in range(len(self.kgrid[tp]["energy"])):
                self.nforced_scat = {"n": 0.0, "p": 0.0}
                self.ediff_scat = {"n": [], "p": []}
                for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                    self.kgrid[tp]["X_E_ik"][ib][ik] = self.get_X_ib_ik_within_E_radius(tp,ib,ik,
                                                    E_radius=0.0, forced_min_npoints=2, tolerance=self.dE_min)
                print "here nforced k-points ratio for {}-type elastic scattering".format(tp)
                print self.nforced_scat[tp] / (2 * len(self.kgrid[tp]["kpoints"][ib]))
                if self.nforced_scat[tp] / (2 * len(self.kgrid[tp]["kpoints"][ib])) > 0.1:
                    print "here this ib x k length:"
                    print (len(self.kgrid[tp]["energy"]) * len(self.kgrid[tp]["kpoints"][ib]))
                    # TODO: this should be an exception but for now I turned to warning for testing.
                    warnings.warn("the k-grid is too coarse for an acceptable simulation of elastic scattering in {} bands;"
                                  .format(["conduction", "valence"][["n", "p"].index(tp)]))

                avg_Ediff = sum(self.ediff_scat[tp])/max(len(self.ediff_scat[tp]), 1)
                if avg_Ediff > avg_Ediff_tolerance:
                    #TODO: change it back to ValueError as it was originally, it was switched to warning for fast debug
                    warnings.warn("{}-type average energy difference of the enforced scattered k-points is more than"
                                     " {}, try running with a more dense k-point mesh".format(tp, avg_Ediff_tolerance))


        # inelastic scattering
        if "POP" in self.inelastic_scatterings:
            for tp in ["n", "p"]:
                for ib in range(len(self.kgrid[tp]["energy"])):
                    self.nforced_scat = {"n": 0.0, "p": 0.0}
                    self.ediff_scat = {"n": [], "p": []}
                    for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                        self.kgrid[tp]["X_Eplus_ik"][ib][ik] = self.get_X_ib_ik_within_E_radius(tp,ib,ik,
                            E_radius= + hbar * self.kgrid[tp]["W_POP"][ib][ik], forced_min_npoints=2, tolerance=self.dE_min)
                        self.kgrid[tp]["X_Eminus_ik"][ib][ik] = self.get_X_ib_ik_within_E_radius(tp, ib, ik,
                            E_radius= - hbar * self.kgrid[tp]["W_POP"][ib][ik],forced_min_npoints=2, tolerance=self.dE_min)


                    print "here nforced k-points ratio for {}-type POP scattering".format(tp)
                    # one of the 2s is for plus and minus and the other one is the selected forced_min_npoints
                    print self.nforced_scat[tp] / (2 * 2 * len(self.kgrid[tp]["kpoints"][ib]))
                    print self.nforced_scat[tp]
                    print (2 * 2 * len(self.kgrid[tp]["kpoints"][ib]))

                    if self.nforced_scat[tp] / (2*2*len(self.kgrid[tp]["kpoints"][ib])) > 0.1:
                        print "here this ib x k length:"
                        print (len(self.kgrid[tp]["energy"])*len(self.kgrid[tp]["kpoints"][ib]))
                        # TODO: this should be an exception but for now I turned to warning for testing.
                        warnings.warn("the k-grid is too coarse for an acceptable simulation of POP scattering in {} bands;"
                          " you can try this k-point grid but without POP as an inelastic scattering.".format(
                        ["conduction", "valence"][["n", "p"].index(tp)]))

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
        seks = [np.dot(frac_k, self._lattice_matrix)*1/A_to_nm*2*pi for frac_k in fractional_ks]

        all_Xs = []
        new_X_ib_ik = []
        for sek in seks:
            X = self.cos_angle(k, sek)
            if X in all_Xs:
                continue
            else:
                new_X_ib_ik.append((X, ib, ik, sek))
                all_Xs.append(X)
        all_Xs.sort()
        return new_X_ib_ik


    def get_X_ib_ik_within_E_radius(self, tp, ib, ik, E_radius, forced_min_npoints=0, tolerance=0.01):
        """Returns the sorted (based on angle, X) list of angle and band and k-point indexes of all the points
            that are withing the E_radius of E
            Attention! this function assumes self.kgrid is sorted based on the energy in ascending order."""
        E = self.kgrid[tp]["energy"][ib][ik]
        k = self.kgrid[tp]["cartesian kpoints"][ib][ik]
        result = []
        counter = 0
        nk = len(self.kgrid[tp]["kpoints"][ib])


        for ib_prm in range(self.cbm_vbm[tp]["included"]):
            if ib==ib_prm and E_radius==0.0:
                ik_prm = ik
            else:
                ik_prm = np.abs(self.kgrid[tp]["energy"][ib_prm] - (E + E_radius)).argmin() - 1
            while (ik_prm<nk-1) and abs(self.kgrid[tp]["energy"][ib_prm][ik_prm+1]-(E+E_radius)) < tolerance:
                ik_prm += 1
                result.append((self.cos_angle(k, self.kgrid[tp]["cartesian kpoints"][ib_prm][ik_prm]),ib_prm,ik_prm))
                counter += 1

            if ib==ib_prm and E_radius==0.0:
                ik_prm = ik
            else:
                ik_prm = np.abs(self.kgrid[tp]["energy"][ib_prm] - (E + E_radius)).argmin() + 1
            while (ik_prm>0) and abs(E+E_radius - self.kgrid[tp]["energy"][ib_prm][ik_prm-1]) < tolerance:
                ik_prm -= 1
                result.append((self.cos_angle(k, self.kgrid[tp]["cartesian kpoints"][ib_prm][ik_prm]), ib_prm, ik_prm))
                counter += 1


        # If fewer than forced_min_npoints number of points were found, just return a few surroundings of the same band
        ik_prm = ik
        while counter < forced_min_npoints and ik_prm < nk - 1:
            ik_prm += 1
            k_prm = self.kgrid[tp]["cartesian kpoints"][ib][ik_prm]

            result.append((self.cos_angle(k, k_prm), ib, ik_prm))
            result += self.kgrid[tp]["X_E_ik"][ib][ik_prm]
            counter += 1

            self.nforced_scat[tp] += 1
            self.ediff_scat[tp].append(self.kgrid[tp]["energy"][ib][ik_prm]-self.kgrid[tp]["energy"][ib][ik])


        ik_prm = ik
        while counter < forced_min_npoints and ik_prm > 0:
            ik_prm -= 1
            result.append((self.cos_angle(k, self.kgrid[tp]["cartesian kpoints"][ib][ik_prm]), ib, ik_prm))

            # also add all values with the same energy at ik_prm
            result += self.kgrid[tp]["X_E_ik"][ib][ik_prm]
            counter += 1
            self.nforced_scat[tp] += 1
            self.ediff_scat[tp].append(self.kgrid[tp]["energy"][ib][ik]-self.kgrid[tp]["energy"][ib][ik_prm])

        result.sort(key=lambda x: x[0])
        return result



    def s_el_eq(self, sname, tp, c, T, k, k_prm):
        """
        return the scattering rate at wave vector k at a certain concentration and temperature
        for a specific elastic scattering mechanisms determined by sname
        :param sname (string): abbreviation of the name of the elastic scatteirng mechanisms; options: IMP, ADE, PIE, DIS
        :param c:
        :param T:
        :param k:
        :param k_prm:
        :return:
        """
        norm_diff_k = norm(k - k_prm) # the slope for PIE and IMP don't match with bs_is_isotropic
        # norm_diff_k = norm(k) # slope kind of matches with bs_is_isotropic at least for PIE but it's 5X larger
        # norm_diff_k = norm(k_prm) # slope kind of matches with bs_is_isotropic at least for PIE but it's 5X larger
        # norm_diff_k = (norm(k_prm)**2 + norm(k)**2)**0.5 # doesn't work, the ratios are not a fixed number

        if norm_diff_k == 0:
            print "WARNING!!! same k and k' vectors as input of the elastic scattering equation"
            # warnings.warn("same k and k' vectors as input of the elastic scattering equation")
            # raise ValueError("same k and k' vectors as input of the elastic scattering equation."
            #                  "Check get_X_ib_ik_within_E_radius for possible error")
            return 0.0

        if sname.upper() in ["IMP"]: # ionized impurity scattering
            unit_conversion = 0.001 / e**2
            return unit_conversion * e ** 4 * self.egrid["N_II"][c][T] /\
                        (4.0 * pi**2 * self.epsilon_s ** 2 * epsilon_0 ** 2 * hbar)\
                                    / ((norm_diff_k ** 2 + self.egrid["beta"][c][T][tp] ** 2) ** 2)

        elif sname.upper() in ["ACD"]: # acoustic deformation potential scattering
            unit_conversion = 1e18 * e
            return unit_conversion * k_B * T * self.E_D[tp]**2 / (4.0 * pi**2 * hbar * self.C_el)

        elif sname.upper() in ["PIE"]: # piezoelectric scattering
            unit_conversion = 1e9 / e
            return unit_conversion * e**2 * k_B * T * self.P_PIE**2 \
                   /(norm_diff_k ** 2 * 4.0 * pi**2 * hbar * epsilon_0 * self.epsilon_s)

        elif sname.upper() in ["DIS"]:
            return self.gs

        else:
            raise ValueError("The elastic scattering name {} is not supported!".format(sname))



    def integrate_over_DOSxE_dE(self, func, tp, fermi, T, interpolation_nsteps=None):
        if not interpolation_nsteps:
            interpolation_nsteps = max(5, int(500.0/len(self.egrid[tp]["energy"])) )
        integral = 0.0
        for ie in range(len(self.egrid[tp]["energy"]) - 1):
            E = self.egrid[tp]["energy"][ie]
            dE = abs(self.egrid[tp]["energy"][ie + 1] - E) / interpolation_nsteps
            dS = (self.egrid[tp]["DOS"][ie + 1] - self.egrid[tp]["DOS"][ie]) / interpolation_nsteps
            for i in range(interpolation_nsteps):
                integral += dE * (self.egrid[tp]["DOS"][ie] + i * dS)*func(E + i * dE, fermi, T)*self.Efrequency[tp][ie]
        return integral
        # return integral/sum(self.Efrequency[tp][:-1])



    def integrate_over_BZ(self,prop_list, tp, c, T, xDOS=False, xvel=False, weighted=True):

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
                # k_nrm = m_e* self.kgrid[tp]["norm(v)"][ib][ik] / (hbar * e * 1e11)

                # 4*pi, hbar and norm(v) are coming from the conversion of dk to dE
                product = k_nrm**2/self.kgrid[tp]["norm(v)"][ib][ik] *4*pi/hbar
                if xvel:
                    product *= self.kgrid[tp]["velocity"][ib][ik]
                for j, p in enumerate(prop_list):
                    if p[0] == "/":
                        product /= self.kgrid[tp][p.split("/")[-1]][c][T][ib][ik]
                    elif p[0] == "1": # this assumes that the property is 1-f0 for example
                        product *= 1 - self.kgrid[tp][p.split("-")[-1]][c][T][ib][ik]
                    else:
                        product *= self.kgrid[tp][p][c][T][ib][ik]
                sum_over_k += product
            if xDOS:
                sum_over_k *= self.egrid[tp]["DOS"][ie]
            if weighted:
                sum_over_k *= self.Efrequency[tp][ie]**(wpower)
            integral += sum_over_k*dE


        if weighted:
            return integral / sum([freq**(wpower) for freq in self.Efrequency[tp][:-1]])
        else:
            return integral
        # return integral / sum([self.egrid[tp]["f0"][c][T][ie][0]*self.Efrequency[tp][ie] for ie in range(len(self.Efrequency[tp][:-1]))])


    def integrate_over_E(self, prop_list, tp, c, T, xDOS=False, xvel=False, weighted=False, interpolation_nsteps=None):

        weighted = False

        wpower = 1
        if xvel:
            wpower += 1
        imax_occ = len(self.Efrequency[tp][:-1])
        # imax_occ = 50

        if not interpolation_nsteps:
            interpolation_nsteps = max(5, int(500.0/len(self.egrid[tp]["energy"])) )
        diff = [0.0 for prop in prop_list]
        integral = self.gs
        # for ie in range(len(self.egrid[tp]["energy"]) - 1):
        for ie in range(imax_occ):
            # if weighted:
            #     f0 = self.egrid[tp]["f0"][c][T][ie]
            #     dfdE = self.egrid[tp]["df0dE"][c][T][ie]
            #     df0 = (self.egrid[tp]["f0"][c][T][ie + 1] - f0) / interpolation_nsteps
            #     ddfdE = self.egrid[tp]["df0dE"][c][T][ie+1] - dfdE
            E = self.egrid[tp]["energy"][ie]
            dE = abs(self.egrid[tp]["energy"][ie + 1] - E) / interpolation_nsteps
            if xDOS:
                dS = (self.egrid[tp]["DOS"][ie + 1] - self.egrid[tp]["DOS"][ie]) / interpolation_nsteps
            if xvel:
                dv = (self.egrid[tp]["velocity"][ie + 1] - self.egrid[tp]["velocity"][ie]) / interpolation_nsteps
            for j, p in enumerate(prop_list):
                try:
                    diff[j] = (self.egrid[tp][p][c][T][ie + 1] - self.egrid[tp][p][c][T][ie]) / interpolation_nsteps
                except:
                    diff[j] = (self.egrid[tp][p.split("/")[-1]][c][T][ie + 1] -
                               self.egrid[tp][p.split("/")[-1]][c][T][ie]) / interpolation_nsteps
            for i in range(interpolation_nsteps):
                multi = dE
                for j, p in enumerate(prop_list):
                    if p[0] == "/":
                        multi /= self.egrid[tp][p.split("/")[-1]][c][T][ie] + diff[j] * i
                    else:
                        multi *= self.egrid[tp][p][c][T][ie] + diff[j]*i
                if xDOS:
                    multi *= self.egrid[tp]["DOS"][ie] + dS*i
                if xvel:
                    multi *= self.egrid[tp]["velocity"][ie] + dv*i
                if weighted:
                    # integral += multi * self.Efrequency[tp][ie]**wpower * (-(dfdE + ddfdE))
                    # integral += multi * self.Efrequency[tp][ie]**wpower *dfdE
                    # integral += multi * self.Efrequency[tp][ie]**wpower * self.egrid[tp]["f0"][c][T][ie]
                    # integral += multi * self.Efrequency[tp][ie]**wpower
                    # integral += multi * self.Efrequency[tp][ie]**wpower
                    integral += multi * self.Efrequency[tp][ie]**wpower
                else:
                    integral += multi
        if weighted:
            # return integral
            # return integral/(sum(self.Efrequency[tp][:-1]))
            # return integral / sum([freq ** wpower for freq in self.Efrequency[tp][:-1]]) / sum(self.egrid[tp]["df0dE"][c][T][:-1])
            # return integral / sum([freq**wpower for freq in self.Efrequency[tp][0:imax_occ]])
            # return integral / (sum([freq**wpower for ie, freq in enumerate(self.Efrequency[tp][0:imax_occ])]))/(-sum(self.egrid[tp]["df0dE"][c][T]))

            return integral / (sum([freq**wpower for ie, freq in enumerate(self.Efrequency[tp][0:imax_occ])]))

            # return integral / (sum([(-self.egrid[tp]["df0dE"][c][T][ie]) * self.Efrequency[tp][ie]**wpower for ie in
            #                    range(len(self.Efrequency[tp][:-1]))]))
        else:
            return integral



    def integrate_over_X(self, tp, X_E_index, integrand, ib, ik, c, T, sname=None, g_suffix=""):
        """integrate numerically with a simple trapezoidal algorithm."""
        sum = np.array([0.0, 0.0, 0.0])
        if len(X_E_index[ib][ik]) == 0:
            raise ValueError("enforcing scattering points did NOT work, {}[{}][{}] is empty".format(X_E_index,ib,ik))
            # return sum
        X, ib_prm, ik_prm = X_E_index[ib][ik][0]
        current_integrand = integrand(tp, c, T, ib, ik, ib_prm, ik_prm, X, sname=sname, g_suffix=g_suffix)
        for i in range(len(X_E_index[ib][ik]) - 1):
            DeltaX = X_E_index[ib][ik][i + 1][0] - \
                     X_E_index[ib][ik][i][0]
            if DeltaX == 0.0:
                continue



            # dum = np.array([0.0, 0.0, 0.0])
            # for j in range(2):
            #     # extract the indecies
            #     X, ib_prm, ik_prm = X_E_index[ib][ik][i + j]
            #     dum += integrand(tp, c, T, ib, ik, ib_prm, ik_prm, X, sname=sname, g_suffix=g_suffix)

            # dum /= 2.0  # the average of points i and i+1 to integrate via the trapezoidal rule

            dum = current_integrand/2

            X, ib_prm, ik_prm = X_E_index[ib][ik][i+1]
            current_integrand = integrand(tp, c, T, ib, ik, ib_prm, ik_prm, X, sname=sname, g_suffix=g_suffix)

            dum += current_integrand/2
            sum += dum * DeltaX  # In case of two points with the same X, DeltaX==0 so no duplicates
        # return sum/len(X_E_index[ib][ik])
        return sum



    def el_integrand_X(self, tp, c, T, ib, ik, ib_prm, ik_prm, X, sname=None, g_suffix=""):

        # The following (if passed on to s_el_eq) result in many cases k and k_prm being equal which we don't want.
        # k = m_e * self._avg_eff_mass[tp] * self.kgrid[tp]["norm(v)"][ib][ik] / (hbar * e * 1e11)
        # k_prm = m_e * self._avg_eff_mass[tp] * self.kgrid[tp]["normv"][ib_prm][ik_prm] / (hbar * e * 1e11)

        k = self.kgrid[tp]["cartesian kpoints"][ib][ik]
        k_prm = self.kgrid[tp]["cartesian kpoints"][ib_prm][ik_prm]


        # print "compare"
        # print self.kgrid[tp]["norm(1/v)"][ib_prm][ik_prm]
        # print norm(1/self.kgrid[tp]["velocity"][ib_prm][ik_prm])
        # print

        # TODO: if only use norm(k_prm), I get ACD mobility that is almost exactly inversely proportional to temperature
        # return (1 - X) * norm(k_prm)** 2 * self.s_el_eq(sname, tp, c, T, k, k_prm) \
        #        * self.G(tp, ib, ik, ib_prm, ik_prm, X) \
        #        * self.kgrid[tp]["norm(1/v)"][ib_prm][ik_prm]

        # / self.kgrid[tp]["velocity"][ib_prm][ik_prm] # this is wrong: when converting
                    # from dk (k being norm(k)) to dE, dk = 1/hbar*norm(1/v)*dE rather than simply 1/(hbar*v)*dE

        # This results in VERY LOW scattering rates (in order of 1-10 1/s!) in some k-points
        # return (1 - X) * (m_e * norm(self.kgrid[tp]["velocity"][ib_prm][ik_prm]) / (hbar * e * 1e11))**2 * self.s_el_eq(sname, tp, c, T, k, k_prm) \
        #        * self.G(tp, ib, ik, ib_prm, ik_prm, X)  \
        #        * self.kgrid[tp]["norm(1/v)"][ib_prm][ik_prm]/3 # not sure where this 3 comes from yet but for iso and aniso to match in SPB, its presence is necessary

        # return (1 - X) * (m_e * self.kgrid[tp]["velocity"][ib_prm][ik_prm] / (
        #     hbar * e * 1e11)) ** 2 * self.s_el_eq(sname, tp, c, T, k, k_prm) \
        #        * self.G(tp, ib, ik, ib_prm, ik_prm, X) \
        #        * 1.0/self.kgrid[tp]["velocity"][ib_prm][ik_prm]

        # previous match (commented on 5/26/2017)
        # return (1 - X) * (m_e * self.kgrid[tp]["norm(v)"][ib_prm][ik_prm] / (
        #     hbar * e * 1e11)) ** 2 * self.s_el_eq(sname, tp, c, T, k, k_prm) \
        #        * self.G(tp, ib, ik, ib_prm, ik_prm, X) \
        #        * 1.0 / self.kgrid[tp]["norm(v)"][ib_prm][ik_prm]

        # in the numerator we use velocity as k_nrm is defined in each direction as they are treated independently (see s_el_eq_isotropic for more info)
        return (1 - X) * (m_e * self._avg_eff_mass[tp] * self.kgrid[tp]["velocity"][ib_prm][ik_prm] / (
            hbar * e * 1e11)) ** 2 * self.s_el_eq(sname, tp, c, T, k, k_prm) \
               * self.G(tp, ib, ik, ib_prm, ik_prm, X) \
               * 1.0 / self.kgrid[tp]["norm(v)"][ib_prm][ik_prm]

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
        f = self.kgrid[tp]["f0"][c][T][ib][ik]
        k_prm = self.kgrid[tp]["cartesian kpoints"][ib_prm][ik_prm]
        # v_prm = self.kgrid[tp]["velocity"][ib_prm][ik_prm]
        f_prm = self.kgrid[tp]["f0"][c][T][ib_prm][ik_prm]


        fermi = self.egrid["fermi"][c][T]

        # test
        # f = self.f(self.kgrid[tp]["energy"][ib][ik], fermi, T, tp, c, alpha)
        # f_prm = self.f(self.kgrid[tp]["energy"][ib_prm][ik_prm], fermi, T, tp, c, alpha)

        N_POP = 1 / ( np.exp(hbar*self.kgrid[tp]["W_POP"][ib][ik]/(k_B*T)) - 1 )
        # norm_diff = max(norm(k-k_prm), 1e-10)
        norm_diff = norm(k-k_prm)
        # print norm(k_prm)**2
        # the term norm(k_prm)**2 is wrong in practice as it can be too big and originally we integrate |k'| from 0
        # integ = norm(k_prm)**2*self.G(tp, ib, ik, ib_prm, ik_prm, X)/(v[alpha]*norm_diff**2)
        # integ = self.G(tp, ib, ik, ib_prm, ik_prm, X)/v_prm # simply /v_prm is wrong and creates artificial anisotropy
        # integ = self.G(tp, ib, ik, ib_prm, ik_prm, X)*norm(1.0/v_prm)
        integ = self.G(tp, ib, ik, ib_prm, ik_prm, X)/self.kgrid[tp]["norm(v)"][ib_prm][ik_prm]
        if "S_i" in sname:
            integ *= abs(X*self.kgrid[tp]["g" + g_suffix][c][T][ib][ik])
            # integ *= X*self.kgrid[tp]["g" + g_suffix][c][T][ib][ik][alpha]
            if "minus" in sname:
                integ *= (1-f)*N_POP + f*(1+N_POP)
            elif "plus" in sname:
                integ *= (1-f)*(1+N_POP) + f*N_POP
            else:
                raise ValueError('"plus" or "minus" must be in sname for phonon absorption and emission respectively')
        elif "S_o" in sname:
            if "minus" in sname:
                integ *= (1-f_prm)*(1+N_POP) + f_prm*N_POP
            elif "plus" in sname:
                integ *= (1-f_prm)*N_POP + f_prm*(1+N_POP)
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
                        for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                            # S_i = np.array([self.gs, self.gs, self.gs])
                            S_i = self.gs
                            S_i_th = self.gs
                            # S_o = np.array([self.gs, self.gs, self.gs])

                            # v = sum(self.kgrid[tp]["velocity"][ib][ik]) / 3
                            v = self.kgrid[tp]["norm(v)"][ib][ik] / sq3 # 3**0.5 is to treat each direction as 1D BS

                            # k = m_e * self._avg_eff_mass[tp] * v / (hbar * e * 1e11)
                            k = self.kgrid[tp]["norm(k)"][ib][ik]

                            a = self.kgrid[tp]["a"][ib][ik]
                            c_ = self.kgrid[tp]["c"][ib][ik]
                            # f = self.kgrid[tp]["f0"][c][T][ib][ik]
                            f = self.kgrid[tp]["f"][c][T][ib][ik]
                            f_th = self.kgrid[tp]["f_th"][c][T][ib][ik]
                            # N_POP = 1 / (np.exp(hbar * self.kgrid[tp]["W_POP"][ib][ik] / (k_B * T)) - 1)
                            N_POP = self.kgrid[tp]["N_POP"][c][T][ib][ik]
                            for j, X_Epm in enumerate(["X_Eplus_ik", "X_Eminus_ik"]):
                                # bypass k-points that cannot have k- associated with them (even though indexes may be available due to enforced scattering)
                                if X_Epm == "X_Eminus_ik" and self.kgrid[tp]["energy"][ib][ik] - hbar *\
                                        self.kgrid[tp]["W_POP"] < self.cbm_vbm[tp]["energy"]:
                                    continue

                                #TODO: see how does dividing by len_eqE affect results, set to 1 to test
                                len_eqE = len(self.kgrid[tp][X_Epm][ib][ik])
                                # if len_eqE == 0:
                                #     print "WARNING!!!! element {} of {} is empty!!".format(ik, X_Epm)
                                for X_ib_ik in self.kgrid[tp][X_Epm][ib][ik]:
                                    X, ib_pm, ik_pm = X_ib_ik
                                    g_pm = self.kgrid[tp]["g"][c][T][ib_pm][ik_pm]
                                    g_pm_th = self.kgrid[tp]["g_th"][c][T][ib_pm][ik_pm]
                                    v_pm= self.kgrid[tp]["norm(v)"][ib_pm][ik_pm]/ sq3 # 3**0.5 is to treat each direction as 1D BS
                                    # k_pm  = m_e*self._avg_eff_mass[tp]*v_pm/(hbar*e*1e11)
                                    k_pm = self.kgrid[tp]["norm(k)"][ib_pm][ik_pm]
                                    abs_kdiff = abs(k_pm - k)
                                    if abs_kdiff < 1e-4:
                                        continue

                                    a_pm = self.kgrid[tp]["a"][ib_pm][ik_pm]
                                    c_pm = self.kgrid[tp]["c"][ib_pm][ik_pm]
                                    # g_pm = sum(self.kgrid[tp]["g"+g_suffix][c][T][ib_pm][ik_pm])/3
                                    # f_pm = self.kgrid[tp]["f0"][c][T][ib_pm][ik_pm]
                                    f_pm = self.kgrid[tp]["f"][c][T][ib_pm][ik_pm]
                                    f_pm_th = self.kgrid[tp]["f_th"][c][T][ib_pm][ik_pm]
                                    A_pm = a*a_pm + c_*c_pm*(k_pm**2+k**2)/(2*k_pm*k)

                                    beta_pm = (e**2*self.kgrid[tp]["W_POP"][ib_pm][ik_pm]*k_pm)/(4*pi*hbar*k*v_pm)*\
                                        (1/(self.epsilon_inf*epsilon_0)-1/(self.epsilon_s*epsilon_0))*6.2415093e20

                                    if not once_called:
                                        lamb_opm=beta_pm*(A_pm**2*log((k_pm+k)/abs_kdiff+1e-4)-A_pm*c_*c_pm-a*a_pm*c_*c_pm)
                                        # because in the scalar form k+ or k- is suppused to be unique, here we take average

                                        self.kgrid[tp]["S_o"][c][T][ib][ik] +=((N_POP + j+(-1)**j*f_pm)*lamb_opm)/len_eqE
                                        self.kgrid[tp]["S_o_th"][c][T][ib][ik] +=((N_POP + j+(-1)**j*f_pm_th)*lamb_opm)/len_eqE


                                    lamb_ipm = beta_pm * (
                                            A_pm ** 2 * log((k_pm + k) / abs_kdiff + 1e-4) * (k_pm ** 2 + k ** 2) / (
                                            2 * k * k_pm) - A_pm ** 2 - c_ ** 2 * c_pm ** 2 / 3)
                                    S_i += ((N_POP + (1 - j) + (-1) ** (1 - j) * f) * lamb_ipm * g_pm) / len_eqE
                                    S_i_th += ((N_POP + (1 - j) + (-1) ** (1 - j) * f_th) * lamb_ipm * g_pm_th) / len_eqE


                                        # The rest is failed attempt to save time in calculating S_i. It does NOT work because beta_pm also changes at each k_pm so we can't have fixed lambda_i_plus for example at each ib_om and ik_pm

                                        # lamb_ipm = beta_pm * (
                                        #     A_pm ** 2 * log((k_pm + k) / abs_kdiff + 1e-4) * (k_pm ** 2 + k ** 2) / (
                                        #     2 * k * k_pm) - A_pm ** 2 - c_ ** 2 * c_pm ** 2 / 3)
                                        # S_i += ((N_POP + (1 - j) + (-1) ** (1 - j) * f) * lamb_ipm * g_pm) / len_eqE

                                    #     if X_Epm == "X_Eplus_ik":
                                    #         self.kgrid[tp]["lambda_i_plus"][ib_pm][ik_pm] = lamb_ipm
                                    #     elif X_Epm == "X_Eminus_ik":
                                    #         self.kgrid[tp]["lambda_i_minus"][ib_pm][ik_pm] = lamb_ipm
                                    #
                                    #
                                    #
                                    # else:
                                    #     if X_Epm == "X_Eplus_ik":
                                    #         S_i += (N_POP + 1 - f)*self.kgrid[tp]["lambda_i_plus"][ib_pm][ik_pm] * g_pm
                                    #     elif X_Epm == "X_Eminus_ik":
                                    #         S_i += (N_POP + f)*self.kgrid[tp]["lambda_i_minus"][ib_pm][ik_pm] * g_pm



                            # self.kgrid[tp]["S_o" + g_suffix][c][T][ib][ik] = S_o
                            self.kgrid[tp]["S_i"][c][T][ib][ik] = S_i
                            self.kgrid[tp]["S_i_th"][c][T][ib][ik] = S_i_th



    def s_inelastic(self, sname = None, g_suffix=""):
        for tp in ["n", "p"]:
            for c in self.dopings:
                for T in self.temperatures:
                    for ib in range(len(self.kgrid[tp]["energy"])):
                        for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                            sum = np.array([0.0, 0.0, 0.0])
                            for X_E_index_name in ["X_Eplus_ik", "X_Eminus_ik"]:
                                sum += self.integrate_over_X(tp, self.kgrid[tp][X_E_index_name], self.inel_integrand_X,
                                        ib=ib, ik=ik, c=c, T=T, sname=sname+X_E_index_name, g_suffix=g_suffix)
                            # self.kgrid[tp][sname][c][T][ib][ik] = abs(sum) * e**2*self.kgrid[tp]["W_POP"][ib][ik]/(4*pi*hbar) \
                            self.kgrid[tp][sname][c][T][ib][ik] = sum*e**2*self.kgrid[tp]["W_POP"][ib][ik] \
                                /(4 * pi * hbar) * (1/self.epsilon_inf-1/self.epsilon_s)/epsilon_0 * 100/e
                            # if norm(self.kgrid[tp][sname][c][T][ib][ik]) < 1:
                            #     self.kgrid[tp][sname][c][T][ib][ik] = [1, 1, 1]
                            # if norm(self.kgrid[tp][sname][c][T][ib][ik]) > 1e5:
                            #     print tp, c, T, ik, ib, sum, self.kgrid[tp][sname][c][T][ib][ik]



    def s_el_eq_isotropic(self, sname, tp, c, T, ib, ik):
        """returns elastic scattering rate (a numpy vector) at certain point (e.g. k-point, T, etc)
        with the assumption that the band structure is isotropic.
        This assumption significantly simplifies the model and the integrated rates at each
        k/energy directly extracted from the literature can be used here."""

        #TODO: decide on knrm and whether it needs a reference (i.e. CBM/VBM). No ref. result in large rates in PbTe.
        # I justify subtracting the CBM/VBM actual k-points as follows:
        # knrm = norm(self.kgrid[tp]["cartesian kpoints"][ib][ik]-np.dot(self.cbm_vbm[tp]["kpoint"], self._lattice_matrix)*2*pi*1/A_to_nm)
        # v = sum(self.kgrid[tp]["velocity"][ib][ik])/3
        # v = norm(self.kgrid[tp]["velocity"][ib][ik])
        v = self.kgrid[tp]["norm(v)"][ib][ik] / 3**0.5 # because of isotropic assumption, we treat the BS as 1D
        # v = self.kgrid[tp]["velocity"][ib][ik] # because it's isotropic, it doesn't matter which one we choose
        # perhaps more correct way of defining knrm is as follows since at momentum is supposed to be proportional to
        # velocity as it is in free-electron formulation so we replaced hbar*knrm with m_e*v/(1e11*e) (momentum)

        # knrm = norm(self.kgrid[tp]["kpoints"][ib][ik]-np.dot(self.cbm_vbm[tp]["kpoint"], self._lattice_matrix)*2*pi*1/A_to_nm)

        knrm = m_e * self._avg_eff_mass[tp] * v/(hbar*e*1e11) # in nm given that v is in cm/s and hbar in eV.s; this resulted in very high ACD and IMP scattering rates, actually only PIE would match with aMoBT results as it doesn't have k_nrm in its formula
        #TODO: make sure that ACD scattering as well as others match in SPB between bs_is_isotropic and when knrm is the following and not above (i.e. not m*v/hbar*e)
        # knrm = norm(self.kgrid[tp]["cartesian kpoints"][ib][ik])
        par_c = self.kgrid[tp]["c"][ib][ik]

        if sname.upper() == "ACD":
            # The following two lines are from Rode's chapter (page 38)
            return (k_B*T*self.E_D[tp]**2*knrm**2)/(3*pi*hbar**2*self.C_el*1e9*v)\
            *(3-8*par_c**2+6*par_c**4)*e*1e20



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

        elif sname.upper() == "IMP": # double-checked the units and equation on 5/12/2017
            # knrm = self.kgrid[tp]["norm(k)"][ib][ik] don't use this! it's wrong anyway and shouldn't change knrm just for IMP
            beta = self.egrid["beta"][c][T][tp]
            B_II = (4*knrm**2/beta**2)/(1+4*knrm**2/beta**2)+8*(beta**2+2*knrm**2)/(beta**2+4*knrm**2)*par_c**2+\
                   (3*beta**4+6*beta**2*knrm**2-8*knrm**4)/((beta**2+4*knrm**2)*knrm**2)*par_c**4
            D_II = 1+2*beta**2*par_c**2/knrm**2+3*beta**4*par_c**4/(4*knrm**4)

            return abs( (e**4*abs(self.egrid["N_II"][c][T]))/(8*pi*v*self.epsilon_s**2*epsilon_0**2*hbar**2*
                        knrm**2)*(D_II*log(1+4*knrm**2/beta**2)-B_II)*3.89564386e27 )

        elif sname.upper() == "PIE":
            return (e ** 2 * k_B * T * self.P_PIE ** 2) / (
                6 * pi * hbar ** 2 * self.epsilon_s * epsilon_0 * v) * (
                           3 - 6 * par_c ** 2 + 4 * par_c ** 4) * 100 / e

        elif sname.upper() == "DIS":
            return (self.N_dis*e**4*knrm)/(hbar**2*epsilon_0**2*self.epsilon_s**2*(self._vrun.lattice.c*A_to_nm)**2*v)\
                   /(self.egrid["beta"][c][T][tp]**4*(1+(4*knrm**2)/(self.egrid["beta"][c][T][tp]**2))**1.5)\
                   *2.43146974985767e42*1.60217657/1e8;

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
            self.kgrid[tp][sname] = {c: {T: np.array([[[0.0, 0.0, 0.0] for i in range(len(self.kgrid[tp]["kpoints"][j]))]
                    for j in range(self.cbm_vbm[tp]["included"])]) for T in self.temperatures} for c in self.dopings}
            for c in self.dopings:
                for T in self.temperatures:
                    for ib in range(len(self.kgrid[tp]["energy"])):
                        for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                            if self.bs_is_isotropic:
                                self.kgrid[tp][sname][c][T][ib][ik] = self.s_el_eq_isotropic(sname, tp, c, T, ib, ik)
                            else:
                                sum = self.integrate_over_X(tp, X_E_index=self.kgrid[tp]["X_E_ik"],
                                                            integrand=self.el_integrand_X,
                                                          ib=ib, ik=ik, c=c, T=T, sname = sname, g_suffix="")
                                self.kgrid[tp][sname][c][T][ib][ik] = abs(sum) * 2e-7 * pi/hbar
                                if norm(self.kgrid[tp][sname][c][T][ib][ik]) < 100 and sname not in ["DIS"]:
                                    print "WARNING!!! here scattering {} < 1".format(sname)
                                    # if self.kgrid[tp]["df0dk"][c][T][ib][ik][0] > 1e-32:
                                    #     print self.kgrid[tp]["df0dk"][c][T][ib][ik]
                                    print self.kgrid[tp]["X_E_ik"][ib][ik]

                                    self.kgrid[tp][sname][c][T][ib][ik] = [1e10, 1e10, 1e10]

                                if norm(self.kgrid[tp][sname][c][T][ib][ik]) > 1e20:
                                    print "WARNING!!! TOO LARGE of scattering rate for {}:".format(sname)
                                    print sum
                                    print self.kgrid[tp]["X_E_ik"][ib][ik]
                                    print
                            self.kgrid[tp]["_all_elastic"][c][T][ib][ik] += self.kgrid[tp][sname][c][T][ib][ik]

                        # logging.debug("relaxation time at c={} and T= {}: \n {}".format(c, T, self.kgrid[tp]["relaxation time"][c][T][ib]))
                        # logging.debug("_all_elastic c={} and T= {}: \n {}".format(c, T, self.kgrid[tp]["_all_elastic"][c][T][ib]))
                        self.kgrid[tp]["relaxation time"][c][T][ib] = 1/self.kgrid[tp]["_all_elastic"][c][T][ib]



    def map_to_egrid(self, prop_name, c_and_T_idx=True, prop_type = "vector"):
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
                        for ib, ik in self.kgrid_to_egrid_idx[tp][ie]:
                            self.egrid[tp][prop_name][ie] += self.kgrid[tp][prop_name][ib][ik]
                        self.egrid[tp][prop_name][ie] /= len(self.kgrid_to_egrid_idx[tp][ie])

                        if self.bs_is_isotropic and prop_type=="vector":
                            self.egrid[tp][prop_name][ie]=np.array([norm(self.egrid[tp][prop_name][ie])/sq3 for i in range(3)])


                else:
                    for ie, en in enumerate(self.egrid[tp]["energy"]):
                        N = 0.0  # total number of instances with the same energy
                        for ib in range(self.cbm_vbm[tp]["included"]):
                            for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                                self.egrid[tp][prop_name][ie] += self.kgrid[tp][prop_name][ib][ik] * \
                                    GB(self.kgrid[tp]["energy"][ib][ik]-self.egrid[tp]["energy"][ie], 0.005)

                        self.egrid[tp][prop_name][ie] /= self.cbm_vbm[tp]["included"] * len(self.kgrid[tp]["kpoints"][0])

                        if self.bs_is_isotropic and prop_type=="vector":
                            self.egrid[tp][prop_name][ie]=np.array([norm(self.egrid[tp][prop_name][ie])/sq3 for i in range(3)])
        else:
            self.initialize_var("egrid", prop_name, prop_type, initval=self.gs, is_nparray=True, c_T_idx=True)

            for tp in ["n", "p"]:

                if not self.gaussian_broadening:

                    for c in self.dopings:
                        for T in self.temperatures:
                            for ie, en in enumerate(self.egrid[tp]["energy"]):
                                # print self.kgrid_to_egrid_idx[tp][ie]
                                for ib, ik in self.kgrid_to_egrid_idx[tp][ie]:
                                    self.egrid[tp][prop_name][c][T][ie] += self.kgrid[tp][prop_name][c][T][ib][ik]
                                self.egrid[tp][prop_name][c][T][ie] /= len(self.kgrid_to_egrid_idx[tp][ie])

                                if self.bs_is_isotropic and prop_type == "vector":
                                    self.egrid[tp][prop_name][c][T][ie] = np.array(
                                        [norm(self.egrid[tp][prop_name][c][T][ie])/sq3 for i in range(3)])

                else:
                    for c in self.dopings:
                        for T in self.temperatures:
                            for ie, en in enumerate(self.egrid[tp]["energy"]):
                                N = 0.0 # total number of instances with the same energy
                                for ib in range(self.cbm_vbm[tp]["included"]):
                                    for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                                        self.egrid[tp][prop_name][c][T][ie] += self.kgrid[tp][prop_name][c][T][ib][ik] * \
                                               GB(self.kgrid[tp]["energy"][ib][ik] -
                                                                            self.egrid[tp]["energy"][ie], 0.005)
                                self.egrid[tp][prop_name][c][T][ie] /= self.cbm_vbm[tp]["included"] * len(self.kgrid[tp]["kpoints"][0])


                                if self.bs_is_isotropic and prop_type == "vector":
                                    self.egrid[tp][prop_name][c][T][ie] = np.array(
                                        [norm(self.egrid[tp][prop_name][c][T][ie])/sq3 for i in range(3)])



    def find_fermi_SPB(self, c, T , tolerance=0.001, tolerance_loose=0.03, alpha = 0.4, max_iter = 1000):

        sgn = np.sign(c)
        m_eff = np.prod(self.cbm_vbm["n"]["eff_mass_xx"])**(1.0/3.0)
        c *= sgn
        initial_energy = self.cbm_vbm["n"]["energy"]
        fermi = initial_energy + 0.02
        iter = 0
        for iter in range(max_iter):
            calc_doping = 4*pi*(2*m_eff*m_e*k_B*T/hbar**2)**1.5 *fermi_integral(0.5,fermi,T,initial_energy)*1e-6/e**1.5
            fermi += alpha * sgn*(calc_doping - c) / abs(c + calc_doping) * fermi
            relative_error = abs(calc_doping - c) / abs(c)
            if relative_error <= tolerance:
                # This here assumes that the SPB generator set the VBM to 0.0 and CBM=  gap + scissor
                if sgn < 0:
                    return fermi
                else:
                    return -(fermi - initial_energy)
        if relative_error > tolerance:
            raise ValueError("could NOT find a corresponding SPB fermi level after {} itenrations".format(max_iter))



    def find_fermi(self, c, T, tolerance=0.001, tolerance_loose=0.03, alpha = 0.02, max_iter = 5000):
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
        fermi = self.cbm_vbm[typ]["energy"]
        # fermi = self.egrid[typ]["energy"][0]

        print("temperature: {} K".format(T))
        j = ["n", "p"].index(typ)
        funcs = [lambda E, fermi0, T: f0(E,fermi0,T), lambda E, fermi0, T: 1-f0(E,fermi0,T)]
        calc_doping = (-1)**(j+1) /self.volume / (A_to_m*m_to_cm)**3 \
                *abs(self.integrate_over_DOSxE_dE(func=funcs[j], tp=typ, fermi=fermi, T=T))

        while (relative_error > tolerance) and (iter<max_iter):
            iter += 1 # to avoid an infinite loop
            # tune_alpha = 1 - iter/max_iter
            if iter / max_iter > 0.5: # to avoid oscillation we re-adjust alpha at each iteration
                tune_alpha = 1 - iter / max_iter
            fermi += alpha * tune_alpha * (calc_doping - c)/abs(c + calc_doping) * fermi
            # print(fermi)
            # print(calc_doping)
            # print(temp_doping)
            # print

            ## DOS re-normalization: NOT NECESSARY; this changes DOS at each T and makes AMSET slow;
            ## initial normalization based on zero-T should suffice
            # integ = 0.0
            # for idos in range(len(self.dos)-1):
            #     integ+= (self.dos[idos+1][0] - self.dos[idos][0])*self.dos[idos][1]*f0(self.dos[idos][0], fermi, T)
            # for idos in range(len(self.dos)):
            #     self.dos[idos] *= self.nelec/integ



            for j, tp in enumerate(["n", "p"]):
                integral = 0.0


                for ie in range((1-j)*self.get_Eidx_in_dos(self.cbm_vbm["n"]["energy"])+j*0,
                                (1-j)*len(self.dos)-1 + j*self.get_Eidx_in_dos(self.cbm_vbm["p"]["energy"])-1):
                    integral += (self.dos[ie+1][1] + self.dos[ie][1])/2*funcs[j](self.dos[ie][0],fermi,T)*\
                                (self.dos[ie+1][0] - self.dos[ie][0])
                temp_doping[tp] = (-1) ** (j + 1) * abs(integral/(self.volume * (A_to_m*m_to_cm)**3) )

            # calculate the overall concentration at the current fermi
            # for j, tp in enumerate(["n", "p"]):
            #     integral = self.integrate_over_DOSxE_dE(func=funcs[j], tp=tp, fermi=fermi, T=T)
            #     temp_doping[tp] = (-1)**(j+1) * abs(integral/self.volume / (A_to_m*m_to_cm)**3)
            calc_doping = temp_doping["n"] + temp_doping["p"]
            if abs(calc_doping) < 1e-2:
                calc_doping = np.sign(calc_doping)*0.01 # just so that calc_doping doesn't get stuck to zero!

            # calculate the relative error from the desired concentration, c
            relative_error = abs(calc_doping - c)/abs(c)

        self.egrid["calc_doping"][c][T]["n"] = temp_doping["n"]
        self.egrid["calc_doping"][c][T]["p"] = temp_doping["p"]

        # check to see if the calculated concentration is close enough to the desired value
        if relative_error > tolerance and relative_error <= tolerance_loose:
            warnings.warn("The calculated concentration {} is not accurate compared to {}; results may be unreliable"
                          .format(calc_doping, c))
        elif relative_error > tolerance_loose:
            raise ValueError("The calculated concentration {} is more than {}% away from {}; "
                             "possible cause may low band gap, high temperature, small nsteps, etc; AMSET stops now!"
                             .format(calc_doping, tolerance_loose*100, c))
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
            # TODO: The DOS needs to be revised, if a more accurate DOS is implemented
            # integral = self.integrate_over_E(func=func, tp=tp, fermi=self.egrid["fermi"][c][T], T=T)

            # because this integral has no denominator to cancel the effect of weights, we do non-weighted integral
            # integrate in egrid with /volume and proper unit conversion
            # we assume here that DOS is normalized already
            integral = self.integrate_over_E(prop_list=["f0x1-f0"], tp=tp, c=c, T=T, xDOS=True, weighted=True)
            # integral = sum(self.integrate_over_BZ(["f0", "1-f0"], tp, c, T, xDOS=True, xvel=False, weighted=False))/3

            # beta[tp] = (e**2 / (self.epsilon_s * epsilon_0*k_B*T) * integral * 6.241509324e27)**0.5

            beta[tp] = (e**2 / (self.epsilon_s * epsilon_0*k_B*T) * integral/self.volume * 1e12/e)**0.5
            # beta[tp] = (e**2 / (self.epsilon_s * epsilon_0*k_B*T) * integral * 100/e)**0.5

        return beta



    def to_json(self, kgrid=True, trimmed=False, max_ndata = None, nstart=0):

        if not max_ndata:
            max_ndata = int(self.gl)

        egrid = deepcopy(self.egrid)
        # self.egrid trimming
        if trimmed:
            nmax = min([max_ndata+1, min([len(egrid["n"]["energy"]), len(egrid["p"]["energy"])]) ])
            # print nmax
            # remove_list = []
            for tp in ["n", "p"]:
                # for rm in remove_list:
                #     try:
                #         del (egrid[tp][rm])
                #     except:
                #         pass

                for key in egrid[tp]:
                    if key in ["size"]:
                        continue
                    try:
                        for c in self.dopings:
                            for T in self.temperatures:
                                egrid[tp][key][c][T] = self.egrid[tp][key][c][T][nstart:nstart+nmax]
                    except:
                        try:
                            egrid[tp][key] = self.egrid[tp][key][nstart:nstart+nmax]
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
                nmax = min([max_ndata+1, min([len(kgrid["n"]["kpoints"][0]), len(kgrid["p"]["kpoints"][0])])])
                # remove_list = ["W_POP", "effective mass", "cartesian kpoints", "X_E_ik", "X_Eplus_ik", "X_Eminus_ik"]
                # remove_list = ["W_POP", "effective mass"]

                # remove_list = ["effective mass"]
                for tp in ["n", "p"]:
                #     for rm in remove_list:
                #         try:
                #             del (kgrid[tp][rm])
                #         except:
                #             pass

                    for key in kgrid[tp]:
                        if key in ["size"]:
                            continue
                        try:
                            for c in self.dopings:
                                for T in self.temperatures:
                                    kgrid[tp][key][c][T] = [self.kgrid[tp][key][c][T][b][nstart:nstart+nmax]
                                                            for b in range(self.cbm_vbm[tp]["included"])]
                        except:
                            try:
                                kgrid[tp][key] = [self.kgrid[tp][key][b][nstart:nstart+nmax]
                                                  for b in range(self.cbm_vbm[tp]["included"])]
                            except:
                                print "cutting data for {} numbers in kgrid was NOT successful!".format(key)
                                pass



            with open("kgrid.json", 'w') as fp:
                json.dump(kgrid, fp,sort_keys = True, indent = 4, ensure_ascii=False, cls=MontyEncoder)



    def solve_BTE_iteratively(self):

        # calculating S_o scattering rate which is not a function of g
        if "POP" in self.inelastic_scatterings and not self.bs_is_isotropic:
            for g_suffix in ["", "_th"]:
                self.s_inelastic(sname="S_o"+ g_suffix, g_suffix=g_suffix)

        # solve BTE to calculate S_i scattering rate and perturbation (g) in an iterative manner
        for iter in range(self.maxiters):
            print("Performing iteration # {}".format(iter))
            if "POP" in self.inelastic_scatterings:
                if self.bs_is_isotropic:
                    if iter==0:
                        self.s_inel_eq_isotropic(once_called=False)
                    else:
                        self.s_inel_eq_isotropic(once_called=True)

                else:
                    for g_suffix in ["", "_th"]:
                        self.s_inelastic(sname="S_i" + g_suffix, g_suffix=g_suffix)
            for c in self.dopings:
                for T in self.temperatures:
                    for tp in ["n", "p"]:
                        for ib in range(self.cbm_vbm[tp]["included"]):
                            # with convergence test:
                            # temp=(self.kgrid[tp]["S_i"][c][T][ib]+self.kgrid[tp]["electric force"][c][T][ib])/(
                            #     self.kgrid[tp]["S_o"][c][T][ib] + self.kgrid[tp]["_all_elastic"][c][T][ib])
                            # if sum([norm(self.kgrid[tp]["g"][c][T][ib][i] - temp[i]) for i in range(len(temp))]) \
                            #     / sum([norm(gi) for gi in self.kgrid[tp]["g"][c][T][ib]]) < 0.01:
                            #     print "CONVERGED!"
                            # self.kgrid[tp]["g"][c][T][ib] = temp

                            self.kgrid[tp]["g_POP"][c][T][ib] = (self.kgrid[tp]["S_i"][c][T][ib] +
                                                                 self.kgrid[tp]["electric force"][c][T][ib]) / (
                                                                self.kgrid[tp]["S_o"][c][T][ib] + self.gs)

                            self.kgrid[tp]["g"][c][T] = (self.kgrid[tp]["S_i"][c][T] + self.kgrid[tp]["electric force"][c][
                                T]) / (self.kgrid[tp]["S_o"][c][T] + self.kgrid[tp]["_all_elastic"][c][T])

                            self.kgrid[tp]["g_th"][c][T][ib]=(self.kgrid[tp]["S_i_th"][c][T][ib]+self.kgrid[tp]["thermal force"][c][
                                T][ib]) / (self.kgrid[tp]["S_o_th"][c][T][ib] + self.kgrid[tp]["_all_elastic"][c][T][ib])

                            self.kgrid[tp]["f"][c][T] = self.kgrid[tp]["f0"][c][T] + self.kgrid[tp]["g"][c][T]
                            self.kgrid[tp]["f_th"][c][T] = self.kgrid[tp]["f0"][c][T] + self.kgrid[tp]["g_th"][c][T]

                            for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                                if  norm(self.kgrid[tp]["g_POP"][c][T][ib][ik]) > 1 and iter > 0:
                            # because only when there are no S_o/S_i scattering events, g_POP>>1 while it should be zero
                                    self.kgrid[tp]["g_POP"][c][T][ib][ik] = [self.gs, self.gs, self.gs]

                                    # print("g_POP > 1 !!!!!")
                                    # print self.kgrid[tp]["g_POP"][c][T][ib][ik]
                                    # print ib
                                    # print ik
                                    # print self.kgrid[tp]["S_i"][c][T][ib][ik]
                                    # print self.kgrid[tp]["electric force"][c][T][ib][ik]
                                    # print self.kgrid[tp]["S_o"][c][T][ib][ik]
                                    # print self.kgrid[tp]["velocity"][ib][ik]

                                    # print self.kgrid[tp]["X_Eplus_ik"][ib][ik]
                                    # print self.kgrid[tp]["X_Eminus_ik"][ib][ik]

        for prop in ["electric force", "thermal force", "g", "g_POP", "g_th", "S_i", "S_o", "S_i_th", "S_o_th"]:
            self.map_to_egrid(prop_name=prop, c_and_T_idx=True)

        for tp in ["n", "p"]:
            for c in self.dopings:
                for T in self.temperatures:
                    for ie in range(len(self.egrid[tp]["g_POP"][c][T])):
                        if norm(self.egrid[tp]["g_POP"][c][T][ie]) > 1:
                            self.egrid[tp]["g_POP"][c][T][ie] = [1e-5, 1e-5, 1e-5]


    def calculate_transport_properties(self):
        integrate_over_kgrid = False
        for c in self.dopings:
            for T in self.temperatures:
                for tp in ["n", "p"]:
                    # norm is only for one vector but g has the ibxikx3 dimensions
                    # self.kgrid[tp]["f"][c][T] = self.kgrid[tp]["f0"][c][T] + self.kgrid[tp]["g"][c][T]

                    # this ONLY makes a difference if f and f_th are used in the denominator; but f0 is currently used!
                    # self.egrid[tp]["f"][c][T] = self.egrid[tp]["f0"][c][T] + norm(self.egrid[tp]["g"][c][T])
                    # self.egrid[tp]["f_th"][c][T]=self.egrid[tp]["f0"][c][T]+norm(self.egrid[tp]["g_th"][c][T])

                    # mobility numerators
                    for mu_el in self.elastic_scatterings:
                        if integrate_over_kgrid:
                            self.egrid["mobility"][mu_el][c][T][tp] = (-1) * default_small_E / hbar  * \
                                 self.integrate_over_BZ(prop_list=["/" + mu_el, "df0dk"], tp=tp, c=c,
                                        T=T, xDOS=False, xvel=True, weighted=True) * 1e-7 * 1e-3 * self.volume

                        else:
                            self.egrid["mobility"][mu_el][c][T][tp] = (-1) * default_small_E / hbar * \
                                self.integrate_over_E(prop_list=["/" + mu_el, "df0dk"], tp=tp, c=c,T=T, xDOS=False, xvel=True, weighted=True)


                    if integrate_over_kgrid:
                        denom = self.integrate_over_BZ(["f0"], tp,c,T, xDOS=False, xvel=False, weighted=True) * 1e-7*1e-3 *self.volume
                        if tp=="n":
                            print "{}-type common denominator at {} K".format(tp, T)
                            print denom
                    else:
                        denom = self.integrate_over_E(prop_list=["f0"], tp=tp, c=c, T=T, xDOS=False, xvel=False, weighted=True)

                    for mu_inel in self.inelastic_scatterings:
                            # calculate mobility["POP"] based on g_POP
                            self.egrid["mobility"][mu_inel][c][T][tp] = self.integrate_over_E(prop_list=["g_"+mu_inel],
                                                                tp=tp,c=c,T=T,xDOS=False,xvel=True, weighted=True)

                    if integrate_over_kgrid:
                        self.egrid["mobility"]["overall"][c][T][tp] = self.integrate_over_BZ(["g"], tp, c, T, xDOS=False, xvel=True, weighted=True)
                        print "overll numerator"
                        print self.egrid["mobility"]["overall"][c][T][tp]
                    else:
                        self.egrid["mobility"]["overall"][c][T][tp] = self.integrate_over_E(prop_list=["g"],
                            tp=tp,c=c,T=T,xDOS=False,xvel=True, weighted=True)

                    self.egrid["J_th"][c][T][tp] = self.integrate_over_E(prop_list=["g_th"],
                            tp=tp, c=c, T=T, xDOS=False, xvel=True, weighted=True) * e * 1e24 # to bring J to A/cm2 units

                    for transport in self.elastic_scatterings + self.inelastic_scatterings + ["overall"]:
                        self.egrid["mobility"][transport][c][T][tp]/=default_small_E * denom

                    self.egrid["J_th"][c][T][tp] /= self.volume*self.integrate_over_E(prop_list=["f0"], tp=tp, c=c,
                                                                    T=T, xDOS=True, xvel=False, weighted=True)

                    # other semi-empirical mobility values:
                    fermi = self.egrid["fermi"][c][T]


                    # fermi_SPB = self.egrid["fermi_SPB"][c][T]
                    energy = self.cbm_vbm["n"]["energy"]

                    # ACD mobility based on single parabolic band extracted from Thermoelectric Nanomaterials,
                    # chapter 1, page 12: "Material Design Considerations Based on Thermoelectric Quality Factor"
                    self.egrid["mobility"]["SPB_ACD"][c][T][tp] = 2**0.5*pi*hbar**4*e*self.C_el*1e9/( # C_el in GPa
                        3*(self.cbm_vbm[tp]["eff_mass_xx"]*m_e)**2.5*(k_B*T)**1.5*self.E_D[tp]**2)\
                        *fermi_integral(0,fermi,T,energy,wordy=True)\
                            /fermi_integral(0.5,fermi,T,energy, wordy=True) * e**0.5*1e4 #  to cm2/V.s


                    faulty_overall_mobility = False
                    mu_overrall_norm = norm(self.egrid["mobility"]["overall"][c][T][tp])
                    for transport in self.elastic_scatterings + self.inelastic_scatterings:
                        # averaging all mobility values via Matthiessen's rule
                        self.egrid["mobility"]["average"][c][T][tp] += 1 / self.egrid["mobility"][transport][c][T][tp]
                        if mu_overrall_norm > norm(self.egrid["mobility"][transport][c][T][tp]):
                            faulty_overall_mobility = True # because the overall mobility should be lower than all
                    self.egrid["mobility"]["average"][c][T][tp] = 1 / self.egrid["mobility"]["average"][c][T][tp]

                    # Decide if the overall mobility make sense or it should be equal to average (e.g. when POP is off)
                    #TODO: uncomment the following two I just commented them for a test.
                    # if mu_overrall_norm == 0.0 or faulty_overall_mobility:
                    #     self.egrid["mobility"]["overall"][c][T][tp] = self.egrid["mobility"]["average"][c][T][tp]

                    self.egrid["relaxation time constant"][c][T][tp] =  self.egrid["mobility"]["overall"][c][T][tp] \
                        * 1e-4 * m_e * self.cbm_vbm[tp]["eff_mass_xx"] / e  # 1e-4 to convert cm2/V.s to m2/V.s

                    # calculating other overall transport properties:
                    self.egrid["conductivity"][c][T][tp] = self.egrid["mobility"]["overall"][c][T][tp]* e * abs(c)
                    self.egrid["seebeck"][c][T][tp] = -1e6*k_B*( self.egrid["Seebeck_integral_numerator"][c][T][tp] \
                        / self.egrid["Seebeck_integral_denominator"][c][T][tp] - self.egrid["fermi"][c][T]/(k_B*T) )
                    self.egrid["TE_power_factor"][c][T][tp] = self.egrid["seebeck"][c][T][tp]**2 \
                        * self.egrid["conductivity"][c][T][tp] / 1e6 # in uW/cm2K
                    if "POP" in self.inelastic_scatterings:     # when POP is not available J_th is unreliable
                        self.egrid["seebeck"][c][T][tp] += 0.0
                        # TODO: for now, we ignore the following until we figure out the units see why values are high!
                        # self.egrid["seebeck"][c][T][tp] += 1e6 \
                        #                 * self.egrid["J_th"][c][T][tp]/self.egrid["conductivity"][c][T][tp]/dTdz

                    # print "3 seebeck terms at c={} and T={}:".format(c, T)
                    # print self.egrid["Seebeck_integral_numerator"][c][T][tp] \
                    #     / self.egrid["Seebeck_integral_denominator"][c][T][tp] * -1e6 * k_B
                    # print + self.egrid["fermi"][c][T]/(k_B*T) * 1e6 * k_B
                    # print + self.egrid["J_th"][c][T][tp]/self.egrid["conductivity"][c][T][tp]/dTdz*1e6

                actual_type = self.get_tp(c)
                other_type = self.get_tp(-c)
                self.egrid["seebeck"][c][T][actual_type] = (
                    self.egrid["conductivity"][c][T][actual_type] * self.egrid["seebeck"][c][T][actual_type] -
                    self.egrid["conductivity"][c][T][other_type] * self.egrid["seebeck"][c][T][other_type]) \
                    / (self.egrid["conductivity"][c][T][actual_type] + self.egrid["conductivity"][c][T][other_type])
                # since sigma = c_e x e x mobility_e + c_h x e x mobility_h:
                self.egrid["conductivity"][c][T][actual_type] += self.egrid["conductivity"][c][T][other_type]



    def plot(self, plotT=300, path=None, textsize=40, ticksize=35, margin_left = 160, margin_bottom=120):
        """plots some of the outputs for more detailed analysis, debugging, etc"""
        from matminer.figrecipes.plotly.make_plots import PlotlyFig
        plotT = float(plotT)

        if not path:
            path = os.path.join( os.getcwd(), "plots" )
        fformat = "html"

        for tp in ["n"]:
            print('plotting: first set of plots: "relaxation time", "_all_elastic", "ACD", "df0dk"')
            plt = PlotlyFig(plot_mode='offline', y_title="# of repeated energy in kgrid", x_title="Energy (eV)",
                   plot_title=None, filename=os.path.join(path, "{}_{}.{}".format("E_histogram", tp, fformat)),
                            textsize=textsize, ticksize=ticksize, scale=1, margin_left=margin_left, margin_bottom=margin_bottom)

            plt.xy_plot(x_col=self.egrid[tp]["energy"], y_col=self.Efrequency[tp])


            for prop in ["energy", "df0dk"]:
                plt = PlotlyFig(plot_mode='offline', y_title=prop, x_title="norm(k)",
                            plot_title="{} in kgrid".format(prop), filename=os.path.join(path, "{}_{}.{}".format("{}_kgrid".format(prop), tp, fformat)),
                            textsize=textsize, ticksize=ticksize, scale=1, margin_left=margin_left,
                            margin_bottom=margin_bottom)
                if prop in ["energy"]:
                    plt.xy_plot(x_col=self.kgrid[tp]["norm(k)"][0], y_col=self.kgrid[tp][prop][0])
                if prop in ["df0dk"]:
                    for c in self.dopings:
                        for T in [plotT]:
                            plt.xy_plot(x_col=self.kgrid[tp]["norm(k)"][0],
                                        y_col=[sum(p/3) for p in self.kgrid[tp][prop][c][T][0]])



            plt = PlotlyFig(plot_mode='offline', y_title="norm(velocity) (cm/s)", x_title="norm(k)",
                            plot_title="velocity in kgrid",
                            filename=os.path.join(path, "{}_{}.{}".format("v_kgrid", tp, fformat)),
                            textsize=textsize, ticksize=ticksize, scale=1, margin_left=margin_left,
                            margin_bottom=margin_bottom)

            plt.xy_plot(x_col=self.kgrid[tp]["norm(k)"][0], y_col=self.kgrid[tp]["norm(v)"][0])

            prop_list = ["relaxation time", "_all_elastic", "df0dk"] + self.elastic_scatterings
            if "POP" in self.inelastic_scatterings:
                prop_list += ["g", "g_POP", "S_i", "S_o"]
            for c in self.dopings:
                # for T in self.temperatures:
                for T in [plotT]:
                    for prop_name in prop_list:
                        plt = PlotlyFig(plot_title="c={} 1/cm3, T={} K".format(c, T), x_title="Energy (eV)",
                                y_title=prop_name, hovermode='closest',
                            filename=os.path.join(path, "{}_{}_{}_{}.{}".format(prop_name, tp, c, T, fformat)),
                            plot_mode='offline', username=None, api_key=None, textsize=textsize, ticksize=ticksize, fontfamily=None,
                            height=800, width=1000, scale=None, margin_top=100, margin_bottom=margin_bottom, margin_left=margin_left,
                            margin_right=80,
                            pad=0)
                        prop = [sum(p)/3 for p in self.egrid[tp][prop_name][c][T]] # scat. rates are not vectors all 3 numbers represent single isotropic scattering rate
                        plt.xy_plot(x_col=self.egrid[tp]["energy"], y_col=prop)

            print('plotting: second set of plots: "velocity", "Ediff"')

            # plot versus energy in self.egrid
            prop_list = ["velocity", "Ediff"]
            for prop_name in prop_list:
                plt = PlotlyFig(plot_title=None, x_title="Energy (eV)", y_title=prop_name, hovermode='closest',
                            filename=os.path.join(path, "{}_{}.{}".format(prop_name, tp, fformat)),
                 plot_mode='offline', username=None, api_key=None, textsize=textsize, ticksize=ticksize, fontfamily=None,
                 height=800, width=1000, scale=None, margin_top=100, margin_bottom=margin_bottom, margin_left=margin_left, margin_right=80,
                 pad=0)
                if "Ediff" in prop_name:
                    y_col = [self.egrid[tp]["energy"][i+1]-\
                                        self.egrid[tp]["energy"][i] for i in range(len(self.egrid[tp]["energy"])-1)]
                else:
                    y_col = [sum(p)/3 for p in self.egrid[tp][prop_name]] # velocity is actually a vector so we take norm
                    plt.xy_plot(x_col=self.egrid[tp]["energy"][:len(y_col)], y_col=y_col, error_type="data",
                                error_array=[np.std(p) for p in self.egrid[tp][prop_name]], error_direction="y")
                    # xrange=[self.egrid[tp]["energy"][0], self.egrid[tp]["energy"][0]+0.6])

            # plot versus norm(k) in self.kgrid
            prop_list = ["energy"]
            # eff_m = 0.1
            for prop_name in prop_list:
                # x_col = [norm(k-np.dot(np.array([0.5, 0.5, 0.5]), self._lattice_matrix)/A_to_nm*2*pi) for k in self.kgrid[tp]["cartesian kpoints"][0]]
                if not self.poly_bands:
                    # x_col = [norm(v)*m_e*sum(self.cbm_vbm[tp]["eff_mass_xx"])/3/ (hbar*1e11*e) for v in
                    #          self.kgrid[tp]["velocity"][0]]
                    x_col = m_e*sum(self.cbm_vbm[tp]["eff_mass_xx"])/3/ (hbar*1e11*e) * self.kgrid[tp]["norm(v)"][0]
                else:
                    # x_col = [norm(k)/(2*pi) for k in self.kgrid[tp]["cartesian kpoints"][0]]
                    # x_col = [norm(k) for k in self.kgrid[tp]["cartesian kpoints"][0]]
                    x_col = self.kgrid[tp]["norm(k)"][0]

                plt = PlotlyFig(plot_title=None, x_title="k [1/nm]",
                                y_title="{} at the 1st band".format(prop_name), hovermode='closest',
                            filename=os.path.join(path, "{}_{}.{}".format(prop_name, tp, fformat)),
                        plot_mode='offline', username=None, api_key=None, textsize=textsize, ticksize=ticksize, fontfamily=None,
                    height=800, width=1000, scale=None, margin_left=margin_left, margin_right=80, margin_bottom=margin_bottom)
                try:
                    y_col = [norm(p) for p in self.kgrid[tp][prop_name][0] ]
                except:
                    y_col = self.kgrid[tp][prop_name][0]
                plt.xy_plot(x_col=x_col, y_col=y_col)
                # xrange=[self.egrid[tp]["energy"][0], self.egrid[tp]["energy"][0]+0.6])






if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # defaults:
    mass = 0.25
    model_params = {"bs_is_isotropic": True, "elastic_scatterings": ["ACD", "IMP", "PIE"],
                    "inelastic_scatterings": []}
                    # TODO: for testing, remove this part later:
                    # "poly_bands":[[[[0.0, 0.0, 0.0], [0.0, mass]]]]}
                  # "poly_bands" : [[[[0.0, 0.0, 0.0], [0.0, mass]],
                  #       [[0.25, 0.25, 0.25], [0.0, mass]],
                  #       [[0.15, 0.15, 0.15], [0.0, mass]]]]}
    # TODO: see why poly_bands = [[[[0.0, 0.0, 0.0], [0.0, 0.32]], [[0.5, 0.5, 0.5], [0.0, 0.32]]]] will tbe reduced to [[[[0.0, 0.0, 0.0], [0.0, 0.32]]


    performance_params = {"nkibz": 200, "dE_min": 0.0001, "adaptive_mesh": False}

    # test
    # material_params = {"epsilon_s": 44.4, "epsilon_inf": 25.6, "W_POP": 10.0, "C_el": 128.8,
    #                "E_D": {"n": 4.0, "p": 4.0}}
    # cube_path = "../test_files/PbTe/nscf_line"
    # coeff_file = os.path.join(cube_path, "..", "fort.123")
    #
    material_params = {"epsilon_s": 12.9, "epsilon_inf": 10.9, "W_POP": 8.73, "C_el": 139.7,
                   "E_D": {"n": 8.6, "p": 8.6}, "P_PIE": 0.052, "scissor": 0.0} #0.5818
    cube_path = "../test_files/GaAs/"
    # coeff_file = os.path.join(cube_path, "fort.123_GaAs_k23")
    coeff_file = os.path.join(cube_path, "fort.123_GaAs_1099kp")


    AMSET = AMSET(calc_dir=cube_path, material_params=material_params,
        model_params = model_params, performance_params= performance_params,
                  # dopings= [-2.7e13], temperatures=[100, 200, 300, 400, 500, 600])
                  # dopings= [-2.7e13], temperatures=[100, 300])
                  # dopings=[-2e15], temperatures=[300, 400, 500, 600, 700, 800])
                  dopings=[-1e20], temperatures=[300, 600])
                  #   dopings = [-1e20], temperatures = [100])
    # AMSET.run(coeff_file=coeff_file, kgrid_tp="coarse")
    cProfile.run('AMSET.run(coeff_file=coeff_file, kgrid_tp="coarse")')

    AMSET.write_input_files()
    AMSET.plot(plotT=300)

    AMSET.to_json(kgrid=True, trimmed=True, max_ndata=None, nstart=0)
    # AMSET.to_json(kgrid=True, trimmed=True)
