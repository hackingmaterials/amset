# coding: utf-8

import warnings

import time

# from pymatgen.electronic_structure.plotter import plot_brillouin_zone, plot_brillouin_zone_from_kpath
# from pymatgen.symmetry.bandstructure import HighSymmKpath

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
from monty.json import MontyEncoder
from random import random
from matminer.figrecipes.plotly.make_plots import PlotlyFig
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
    return 1 / (1 + np.exp((E - fermi) / (k_B * T)))



def df0dE(E, fermi, T):
    """returns the energy derivative of the Fermi-Dirac equilibrium distribution"""
    return -1 / (k_B * T) * np.exp((E - fermi) / (k_B * T)) / (1 + np.exp((E - fermi) / (k_B * T))) ** 2



def fermi_integral(order, fermi, T, initial_energy=0):
    # fermi = fermi - initial_energy
    integral = 0.
    emesh = np.linspace(0.0, 30*k_B*T, 1000000.0) # We choose 20kBT instead of infinity as the fermi distribution will be 0
    dE = (emesh[-1]-emesh[0])/(1000000.0-1.0)
    for E in emesh:
        integral += dE * (E/(k_B*T))**order / (1. + np.exp((E-fermi)/(k_B*T)))

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


    def __init__(self, path_dir=None,

                 N_dis=None, scissor=None, elastic_scatterings=None, include_POP=False, bs_is_isotropic=False,
                 donor_charge=None, acceptor_charge=None, dislocations_charge=None, adaptive_mesh=False,
                 # poly_bands = None):
                 poly_bands=[ [ [[0.0, 0.0, 0.0], [0.0, 0.1] ] ] ]):

        self.nkibz = 12

        #TODO: self.gaussian_broadening is designed only for development version and must be False, remove it later.
        # because if self.gaussian_broadening the mapping to egrid will be done with the help of Gaussian broadening
        # and that changes the actual values
        self.gaussian_broadening = False

        self.dE_global = 0.001 # 0.01/(self.nkibz*50)**0.5 # in eV: the dE below which two energy values are assumed equal
        self.dopings = [-1e19] # 1/cm**3 list of carrier concentrations
        self.temperatures = map(float, [300, 600]) # in K, list of temperatures
        self.epsilon_s = 44.360563 # example for PbTe
        self.epsilon_inf = 25.57 # example for PbTe
        self._vrun = {}
        self.max_e_range = 10*k_B*max(self.temperatures) # we set the max energy range after which occupation is zero
        self.path_dir = path_dir or "../test_files/PbTe_nscf_uniform/nscf_line"
        self.charge = {"n": donor_charge or 1, "p": acceptor_charge or 1, "dislocations": dislocations_charge or 1}
        self.N_dis = N_dis or 0.1 # in 1/cm**2
        # self.elastic_scatterings = elastic_scatterings or ["IMP", "ACD", "PIE", "DIS"]
        self.elastic_scatterings = elastic_scatterings or ["ACD"]
        self.inelastic_scatterings = []
        if include_POP:
            self.inelastic_scatterings += ["POP"]
        self.scissor = scissor or 0.0 # total value added to the band gap by adding to the CBM and subtracting from VBM
        self.bs_is_isotropic = bs_is_isotropic
        self.gs = 1e-32 # a global small value (generally used for an initial non-zero value)
        self.gl = 1e32 # a global large value
        self.dos_bwidth = 0.1 # in eV the bandwidth used for calculation of the total DOS (over all bands & IBZ k-points)
        self.nkdos = 31
        self.poly_bands = poly_bands
        self.adaptive_mesh = adaptive_mesh

#TODO: some of the current global constants should be omitted, taken as functions inputs or changed!

        self.wordy = False
        self.maxiters = 5
        self.soc = False
        self.read_vrun(path_dir=self.path_dir, filename="vasprun.xml")
        if self.poly_bands:
            self.cbm_vbm["n"]["energy"] = self.dft_gap
            self.cbm_vbm["p"]["energy"] = 0.0
            self.cbm_vbm["n"]["kpoint"] = self.cbm_vbm["p"]["kpoint"] = self.poly_bands[0][0][0]

        self.W_POP = 10e12 * 2*pi # POP frequency in Hz
        self.P_PIE = 0.15
        # self.E_D = {"n": 4.0, "p": 3.93}
        self.E_D = {"n": 4.0, "p": 4.0}
        self.C_el = 128.84 #77.3 # [Gpa]:spherically averaged elastic constant for longitudinal modes

        self.all_types = [self.get_tp(c) for c in self.dopings]

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
        print self.cbm_vbm

        # TODO: later add a more sophisticated DOS function, if developed
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

        kremove_list = ["W_POP", "effective mass", "kweights", "a", "c", "f",
                        "f_th", "g_th", "S_i_th", "S_o_th"]

        for tp in ["n", "p"]:
            for rm in kremove_list:
                try:
                    del (self.kgrid[tp][rm])
                except:
                    pass
            for erm in ["all_en_flat", "f0x1-f0", "f_th", "g_th", "S_i_th", "S_o_th"]:
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



    def read_vrun(self, path_dir=".", filename="vasprun.xml"):
        self._vrun = Vasprun(os.path.join(path_dir, filename))
        self.volume = self._vrun.final_structure.volume
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
            self.nelec = cbm_vbm["p"]["bidx"]
            self.dos_normalization_factor = self._vrun.get_band_structure().nb_bands
        else:
            self.nelec = cbm_vbm["p"]["bidx"]*2
            self.dos_normalization_factor = self._vrun.get_band_structure().nb_bands*2

        print("total number of electrons nelec: {}".format(self.nelec))

        bs = bs.as_dict()
        if bs["is_spin_polarized"]:
            self.emin = min(bs["bands"]["1"][0] + bs["bands"]["-1"][0])
            self.emax = max(bs["bands"]["1"][-1] + bs["bands"]["-1"][-1])
        else:
            self.emin = min(bs["bands"]["1"][0])
            self.emax = max(bs["bands"]["1"][-1])

        for i, tp in enumerate(["n", "p"]):
            sgn = (-1)**i
            while abs(min(sgn*bs["bands"]["1"][cbm_vbm[tp]["bidx"]+sgn*cbm_vbm[tp]["included"]])-
                                      sgn*cbm_vbm[tp]["energy"])<self.max_e_range:
                cbm_vbm[tp]["included"] += 1

# TODO: change this later if the band indecies are fixed in Analytical_band class
        cbm_vbm["p"]["bidx"] += 1
        cbm_vbm["n"]["bidx"] = cbm_vbm["p"]["bidx"] + 1

        self.cbm_vbm = cbm_vbm


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
        # fn = lambda E, fermi, T: f0(E, fermi, T) * (1 - f0(E, fermi, T))
        #
        # # To avoid returning a denominator that is zero:
        # return {t:max(self.integrate_over_DOSxE_dE(func=fn,tp=t,fermi=self.egrid["fermi"][c][T],T=T), 1e-30)
        #         for t in ["n", "p"]}
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
        E_idx = []
        for tp in ["n", "p"]:
            for ib, en_vec in enumerate(self.kgrid[tp]["energy"]):
                self.egrid[tp]["all_en_flat"] += list(en_vec)
                # also store the flatten energy (i.e. no band index) as a tuple of band and k-indexes
                E_idx += [(ib, iek) for iek in range(len(en_vec))]

            # get the indexes of sorted flattened energy
            ieidxs = np.argsort(self.egrid[tp]["all_en_flat"])
            self.egrid[tp]["all_en_flat"] = [self.egrid[tp]["all_en_flat"][ie] for ie in ieidxs]

            # sort the tuples of band and energy based on their energy
            E_idx = [E_idx[ie] for ie in ieidxs]


        # setting up energy grid and DOS:
        for tp in ["n", "p"]:
            energy_counter = []
            i = 0
            last_is_counted = False
            while i<len(self.egrid[tp]["all_en_flat"])-1:
                sum_e = self.egrid[tp]["all_en_flat"][i]
                counter = 1.0
                current_ib_ie_idx = [E_idx[i]]
                j = i
                while j<len(self.egrid[tp]["all_en_flat"])-1 and \
                        abs(self.egrid[tp]["all_en_flat"][i]-self.egrid[tp]["all_en_flat"][j+1]) < self.dE_global:
                # while i < len(self.egrid[tp]["all_en_flat"]) - 1 and \
                #          self.egrid[tp]["all_en_flat"][i] == self.egrid[tp]["all_en_flat"][i + 1] :
                    counter += 1
                    current_ib_ie_idx.append(E_idx[j+1])
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
                self.kgrid_to_egrid_idx[tp].append([E_idx[-1]])
                if dos_tp.lower() == "simple":
                    self.egrid[tp]["DOS"].append(self.nelec / len(self.egrid[tp]["all_en_flat"]))
                elif dos_tp.lower() == "standard":
                    self.egrid[tp]["DOS"].append(self.dos[self.get_Eidx_in_dos(self.egrid[tp]["energy"][-1])][1])

            self.egrid[tp]["size"] = len(self.egrid[tp]["energy"])
            # if dos_tp.lower()=="standard":
            #     energy_counter = [ne/len(self.egrid[tp]["all_en_flat"]) for ne in energy_counter]
                #TODO: what is the best value to pick for width here?I guess the lower is more precisely at each energy?
                # dum, self.egrid[tp]["DOS"] = get_dos(self.egrid[tp]["energy"], energy_counter,width = 0.05)

        print "here self.kgrid_to_egrid_idx"
        # print self.kgrid_to_egrid_idx["n"]
        print len(self.kgrid_to_egrid_idx["n"])



        for tp in ["n", "p"]:
            self.Efrequency[tp] = [len(Es) for Es in self.kgrid_to_egrid_idx[tp]]

        print "here total number of ks from self.Efrequency for n-type"
        print sum(self.Efrequency["n"])

        # initialize some fileds/properties
        self.egrid["calc_doping"] = {c: {T: {"n": 0.0, "p": 0.0} for T in self.temperatures} for c in self.dopings}
        for sn in self.elastic_scatterings + self.inelastic_scatterings +["overall", "average", "SPB_ACD"]:
            # self.egrid["mobility"+"_"+sn]={c:{T:{"n": 0.0, "p": 0.0} for T in self.temperatures} for c in self.dopings}
            self.egrid["mobility"][sn] = {c: {T: {"n": 0.0, "p": 0.0} for T in self.temperatures} for c in self.dopings}
        for transport in ["conductivity", "J_th", "seebeck", "TE_power_factor", "relaxation time constant"]:
            self.egrid[transport] = {c: {T: {"n": 0.0, "p": 0.0} for T in self.temperatures} for c in self.dopings}

        # populate the egrid at all c and T with properties; they can be called via self.egrid[prop_name][c][T] later
        self.calculate_property(prop_name="fermi", prop_func=self.find_fermi)
        #
        ##  in case specific fermi levels are to be tested:
        # self.egrid["fermi"] ={
        # -1e+19: {
        #     # original
        #     300.0: 0.28698518995648691,
        #     600.0: 0.174478632328484025,
        #
        #     # new
        #     # 300.0: hbar**2/(2*m_e)*(3*pi**2*2/self.volume)**(2./3.0)*1e20*e,
        #
        #     # 300.0: 0.83321207744528014,
        #     # 600.0: 0.83321207744528014
        # } }


        # self.calculate_property(prop_name="f0", prop_func=f0, for_all_E=True)
        # self.calculate_property(prop_name="f", prop_func=f0, for_all_E=True)
        # self.calculate_property(prop_name="f_th", prop_func=f0, for_all_E=True)

        for prop in ["f", "f_th"]:
            self.map_to_egrid(prop_name=prop, c_and_T_idx=True)
        self.calculate_property(prop_name="f0x1-f0", prop_func=lambda E, fermi, T: f0(E, fermi, T)
                                                                        * (1 - f0(E, fermi, T)), for_all_E=True)
        self.calculate_property(prop_name="beta", prop_func=self.inverse_screening_length)
        self.calculate_property(prop_name="N_II", prop_func=self.calculate_N_II)
        self.calculate_property(prop_name="Seebeck_integral_numerator", prop_func=self.seeb_int_num)
        self.calculate_property(prop_name="Seebeck_integral_denominator", prop_func=self.seeb_int_denom)



    def get_Eidx_in_dos(self, E, Estep=None):
        if not Estep:
            Estep = max(self.dE_global, 0.0001)
        return int(round((E - self.emin) / Estep))



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



    def omit_kpoints(self, low_v_ik, rearranged_props):
        """
        The k-points with velocity < 1 cm/s (either in valence or conduction band) are taken out as those are
            troublesome later with extreme values (e.g. too high elastic scattering rates)
        :param low_v_ik:
        :return:
        """

        print "kpoints indexes with low velocity"
        print low_v_ik

        ik_list = list(set(low_v_ik))
        ik_list.sort(reverse=True)

        # omit the points with v~0 and try to find the parabolic band equivalent effective mass at the CBM and the VBM
        temp_min = {"n": self.gl, "p": self.gl}
        for i, tp in enumerate(["n", "p"]):
            for ib in range(self.cbm_vbm[tp]["included"]):
                # for ik in ik_list:
                for ik in ik_list:
                    if (-1)**i * self.kgrid[tp]["energy"][ib][ik] < temp_min[tp]:
                        self.cbm_vbm[tp]["eff_mass_xx"]=(-1)**i*self.kgrid[tp]["effective mass"][ib][ik].diagonal()
                        self.cbm_vbm[tp]["energy"] = self.kgrid[tp]["energy"][ib][ik]
                    # for prop in ["energy", "a", "c", "kpoints", "kweights"]:
                    #     print prop
                    #     self.kgrid[tp][prop][ib].pop(ik)

                for prop in rearranged_props:
                    self.kgrid[tp][prop] = np.delete(self.kgrid[tp][prop], ik_list, axis=1)



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
        for p in [0.03, 0.05, 0.08]:
                all_perturbed_ks.append([ k_i+p*np.sign(random()-0.5) for k_i in k] )
        return all_perturbed_ks



    def get_ks_with_intermediate_energy(self, kpts, energies, max_Ediff = None, target_Ediff = 0.01):
        final_kpts_added = []
        Tmx = max(self.temperatures)
        if not max_Ediff:
            max_Ediff = 10*k_B*Tmx
        for tp in ["n", "p"]:
            if tp not in self.all_types:
                continue
            ies_sorted = list(np.argsort(energies[tp]))
            if tp=="p":
                ies_sorted.reverse()
            for idx, ie in enumerate(ies_sorted[:-1]):
                Ediff = abs(energies[tp][ie] - energies[tp][ies_sorted[0]])
                final_kpts_added += self.get_perturbed_ks(kpts[ies_sorted[idx]])
                if Ediff > max_Ediff:
                    break
                final_kpts_added += self.get_intermediate_kpoints_list(list(kpts[ies_sorted[idx]]),
                                                   list(kpts[ies_sorted[idx+1]]), max(int(Ediff/target_Ediff) , 1))
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
        if kgrid_tp=="coarse":
            nkstep = self.nkibz

        # # k = list(np.linspace(0.25, 0.75-0.5/nstep, nstep))
        # kx = list(np.linspace(-0.5, 0.5, nkstep))
        # ky = kz = kx
        # # ky = list(np.linspace(0.27, 0.67, nkstep))
        # # kz = list(np.linspace(0.21, 0.71, nkstep))
        # kpts = np.array([[x, y, z] for x in kx for y in ky for z in kz])

        sg = SpacegroupAnalyzer(self._vrun.final_structure)
        self.rotations, self.translations = sg._get_symmetry() # this returns unique symmetry operations


        test_k = [0.5, 0.5, 0.5]
        print "equivalent ks"
        # a = self.remove_duplicate_kpoints([np.dot(test_k, self.rotations[i]) + self.translations[i] for i in range(len(self.rotations))])
        a = self.get_sym_eq_ks_in_first_BZ(test_k)
        # a = self.remove_duplicate_kpoints(a)
        # a = [np.dot(test_k, self.rotations[i]) + self.translations[i] for i in range(len(self.rotations))]
        a = [i.tolist() for i in self.remove_duplicate_kpoints(a)]
        print a

        kpts_and_weights = sg.get_ir_reciprocal_mesh(mesh=(nkstep, nkstep, nkstep), is_shift=(0.01, 0.01, 0.01))
        kpts = [i[0] for i in kpts_and_weights]
        kpts = self.kpts_to_first_BZ(kpts)


        # explicitly add the CBM/VBM k-points to calculate the parabolic band effective mass hence the relaxation time
        kpts.append(self.cbm_vbm["n"]["kpoint"])
        kpts.append(self.cbm_vbm["p"]["kpoint"])

        print "number of original ibz k-points"
        print len(kpts)


        if not self.poly_bands:
            analytical_bands = Analytical_bands(coeff_file=coeff_file)
            all_ibands = []
            for i, tp in enumerate(["p", "n"]):
                for ib in range(self.cbm_vbm[tp]["included"]):
                    sgn = (-1) ** i
                    all_ibands.append(self.cbm_vbm[tp]["bidx"] + sgn * ib)

            start_time = time.time()

            engre, latt_points, nwave, nsym, nsymop, symop, br_dir = analytical_bands.get_engre(iband=all_ibands)
            nstv, vec, vec2 = analytical_bands.get_star_functions(latt_points, nsym, symop, nwave, br_dir=br_dir)
            out_vec2 = np.zeros((nwave, max(nstv), 3, 3))
            for nw in xrange(nwave):
                for i in xrange(nstv[nw]):
                    out_vec2[nw, i] = outer(vec2[nw, i], vec2[nw, i])

            print("time to get engre and calculate the outvec2: {} seconds".format(time.time() - start_time))


            # caluclate and normalize the global density of states (DOS) so the integrated DOS == total number of electrons
            emesh, dos=analytical_bands.get_dos_from_scratch(self._vrun.final_structure,[self.nkdos,self.nkdos,self.nkdos],
                        self.emin, self.emax, int(self.emax-self.emin)/max(self.dE_global, 0.0001), width=self.dos_bwidth)
        else:
            # first modify the self.poly_bands to include all symmetrically equivalent k-points
            # poly_band_short = [i for i in self.poly_bands]
            for ib in range(len(self.poly_bands)):
                for j in range(len(self.poly_bands[ib])):
                    # poly_band_short[ib][j][0] = [self.poly_bands[ib][j][0]]
                    self.poly_bands[ib][j][0] = self.remove_duplicate_kpoints(
                        self.get_sym_eq_ks_in_first_BZ(self.poly_bands[ib][j][0],cartesian=True))

            print "here self.poly_bands"
            print self.poly_bands

            # now construct the DOS
            emesh, dos = get_dos_from_poly_bands(self._vrun.final_structure, self._lattice_matrix, [self.nkdos,self.nkdos,self.nkdos],
                self.emin, self.emax, int(self.emax-self.emin)/max(self.dE_global, 0.0001), poly_bands=self.poly_bands,
                    bandgap=self.cbm_vbm["n"]["energy"]-self.cbm_vbm["p"]["energy"]+self.scissor, width=self.dos_bwidth, SPB_DOS=True)
            # total_nelec = len(self.poly_bands) * 2 # basically 2x number of included occupied bands (valence bands)
            # total_nelec = self.nelec
            self.dos_normalization_factor = len(self.poly_bands) # it is *2 elec/band but /2 because DOS is repeated in valence/conduction

        integ = 0.0
        for idos in range(len(dos) - 2):
            if emesh[idos] > self.cbm_vbm["n"]["energy"]:
                break
            integ += (dos[idos + 1] + dos[idos]) / 2 * (emesh[idos + 1] - emesh[idos])
        # normalize DOS
        dos = [g / integ * self.nelec for g in dos]
        self.dos = zip(emesh, dos)
        self.dos = [list(a) for a in self.dos]

        # calculate the energies in initial ibz k-points and look at the first band to decide on additional/adaptive ks
        energies = {"n": [0.0 for ik in kpts], "p": [0.0 for ik in kpts]}
        for i, tp in enumerate(["p", "n"]):
            sgn = (-1) ** i
            # for ib in range(self.cbm_vbm[tp]["included"]):
            # for now we only look at
            for ib in range(self.cbm_vbm[tp]["included"]):
                # engre, latt_points, nwave, nsym, nsymop, symop, br_dir = \
                #     analytical_bands.get_engre(iband=[self.cbm_vbm[tp]["bidx"] + sgn * ib])
                # bands_data[tp][ib] = (engre, latt_points, nwave, nsym, nsymop, symop, br_dir)

                # if not once_called:
                #     start_time = time.time()
                #     nstv, vec, vec2 = analytical_bands.get_star_functions(latt_points, nsym, symop, nwave,br_dir=br_dir)
                #     out_vec2 = np.zeros((nwave, max(nstv), 3, 3))
                #     for nw in xrange(nwave):
                #         for i in xrange(nstv[nw]):
                #             out_vec2[nw, i] = outer(vec2[nw, i], vec2[nw, i])
                #     print("time to calculate the outvec2: {} seconds".format(time.time() - start_time))
                #
                #     once_called = True

                for ik in range(len(kpts)):
                    if not self.poly_bands:
                        energy, de, dde = analytical_bands.get_energy(
                            kpts[ik], engre[i*self.cbm_vbm["p"]["included"]+ib],
                                nwave, nsym, nstv, vec, vec2, out_vec2, br_dir=br_dir)

                        energies[tp][ik] = energy * Ry_to_eV + sgn * self.scissor/2
                    else:
                        energy,velocity,effective_m=get_poly_energy(np.dot(kpts[ik],self._lattice_matrix/A_to_nm*2*pi),
                                poly_bands=self.poly_bands, type=tp, ib=ib, bandgap=self.dft_gap + self.scissor)
                        # energy,velocity,effective_m=get_poly_energy(kpts[ik], poly_bands=self.poly_bands, type=tp,ib=ib,
                        #     bandgap=self.dft_gap + self.scissor)
                        energies[tp][ik] = energy

        # print "first energies:"
        # print energies


        if self.adaptive_mesh:

            all_added_kpoints = []
            Tmx = max(self.temperatures)
            # print "enegies of valence and conduction bands"
            # print energies
            # all_added_kpoints += self.get_adaptive_kpoints(kpts, energies,adaptive_Erange=[0*k_B*Tmx, 1*k_B*Tmx], nsteps=30)
            # all_added_kpoints += self.get_adaptive_kpoints(kpts, energies,adaptive_Erange=[1*k_B*Tmx, 2*k_B*Tmx], nsteps=15)


            # all_added_kpoints += self.get_ks_with_intermediate_energy(kpts,energies,max_Ediff=1*k_B*Tmx,target_Ediff=0.0001)

            all_added_kpoints += self.get_ks_with_intermediate_energy(kpts,energies,max_Ediff=2*k_B*Tmx,target_Ediff=0.01)

                # temp = kpoints_added[tp]
                # kpoints_added[tp] = np.concatenate( (kpoints_added[tp], temp + np.array([0.0 , 0.0, offset])), axis=0 )
                # kpoints_added[tp] = np.concatenate( (kpoints_added[tp], temp + np.array([0.0 , offset, 0.0])), axis=0 )
                # kpoints_added[tp] = np.concatenate( (kpoints_added[tp], temp + np.array([offset , 0.0, 0.0])), axis=0 )
            # print "here 1"
            # print len(kpoints_added["n"])
            # print kpoints_added["n"]


            print "here the number of added k-points"
            print len(all_added_kpoints)
            print all_added_kpoints
            # print final_kpts_added

            print type(kpts)
            # kpts.tolist()

            kpts += all_added_kpoints
            # if len(final_kpts_added) > 0:
            #     kpts = np.concatenate((kpts, final_kpts_added), axis=0)

        # kpts = self.remove_duplicate_kpoints(kpts)
        # print type(kpts)
        # final_kpts_added = self.remove_duplicate_kpoints(final_kpts_added

        print "debug0"
        # print kpts

        symmetrically_equivalent_ks = []
        for k in kpts:
            symmetrically_equivalent_ks += self.get_sym_eq_ks_in_first_BZ(k)
            # fractional_ks = [np.dot(k, self.rotations[i]) + self.translations[i] for i in range(len(self.rotations))]
            # symmetrically_equivalent_ks += fractional_ks
            # symmetrically_equivalent_ks = np.concatenate((symmetrically_equivalent_ks, fractional_ks), axis=0)
        kpts += symmetrically_equivalent_ks
        # kpts += self.kpts_to_first_BZ(symmetrically_equivalent_ks)
        # kpts = np.concatenate((kpts, symmetrically_equivalent_ks))
        kpts = self.remove_duplicate_kpoints(kpts)

        print "length of final kpts"
        print len(kpts)
        # print kpts

        #test, remove this later:
        # symmetrically_equivalent_ks = []
        # for k in [[0.5, 0.5, 0.5]]:
        #     fractional_ks = [np.dot(k, self.rotations[i]) + self.translations[i] for i in range(len(self.rotations))]
        #     symmetrically_equivalent_ks += fractional_ks
        # print "symmetrically equivalent k-points to X"
        # print symmetrically_equivalent_ks
        # print "now back to first BZ:"
        # print self.kpts_to_first_BZ(symmetrically_equivalent_ks)

        kweights = [1.0 for i in kpts]
        # kweights = np.array(kweights)

        # kweights /= sum(kweights)

        # kpath = HighSymmKpath(self._vrun.final_structure)
        # plt = plot_brillouin_zone_from_kpath(kpath=kpath)

        # ATTENTION!!!!: there are two ways to generate symmetrically equivalent points (actually I checked the second
        # and is NOT equivalent to the first one; I'd say use the first one for now because I am sure of this:
        #  k-point in cartesian = np.dot( self._lattice_matrix.transpose(), fractional-k-point)
        #   1) use the symmetry operations given by SpacegroupAnalyzer._get_symmetry() on the fractional coordinates and
        #  then convert to cartesian via lattice_matrix
        #   2) use the cartesian symmetry operations given by SpacegroupAnalyzer.get_symmetry_operations(cartesian=True)
        # directly on "cartesian kpoints" to get new cartesian (2pi*lattice_matrix*fractional-k) equivalent k-point


        #


        # self.symmetry_operations = sg.get_symmetry_operations(cartesian=True)
        # print self.symmetry_operations
        # new_rot = np.array([[ 0.16666667,  0.47140451, -0.86602538],
        #             [-0.94280907,  0.33333333,  0.        ],
        #              [ 0.28867514,  0.81649659,  0.5       ]])



        # for rot in self.rotations:
        #     print np.dot(self._lattice_matrix.transpose(), np.dot(rot, k))
        #     print np.dot(new_rot, (np.dot(k, self._lattice_matrix)) )
        ##    print np.dot(self._lattice_matrix.transpose(), rot)
            # print


        self.kgrid = {
                # "kpoints": kpts,
                # "kweights": kweights,
                "n": {},
                "p": {} }


        for tp in ["n", "p"]:
            self.kgrid[tp]["kpoints"] = [[k for k in kpts] for ib in range(self.cbm_vbm[tp]["included"])]
            self.kgrid[tp]["kweights"] = [[kw for kw in kweights] for ib in range(self.cbm_vbm[tp]["included"])]

        self.initialize_var("kgrid", ["energy", "a", "c", "norm(1/v)"], "scalar", 0.0, is_nparray=False, c_T_idx=False)
        self.initialize_var("kgrid", ["velocity"], "vector", 0.0, is_nparray=False, c_T_idx=False)
        self.initialize_var("kgrid", ["effective mass"], "tensor", 0.0, is_nparray=False, c_T_idx=False)
        # for tp in ["n", "p"]:
        #     for prop in ["velocity"]:
        #         self.kgrid[tp][prop] = \
        #         np.array([ [[0.0, 0.0, 0.0] for i in range(len(kpts))] for j in range(self.cbm_vbm[tp]["included"])])
        #     self.kgrid[tp]["effective mass"] = \
        #         [ np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] for i in range(len(kpts))]) for j in
        #                                                                         range(self.cbm_vbm[tp]["included"])]
            # self.kgrid[tp]["effective mass"] = \
            #     np.array([[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] for i in range(len(kpts))] for j in
            #           range(self.cbm_vbm[tp]["included"])])


        # low_v_ik = []
        # low_v_ik.sort()
        # analytical_bands = Analytical_bands(coeff_file=coeff_file)
        # # once_called = False
        # # bands_data = {tp: [() for ib in range(self.cbm_vbm[tp]["included"])] for tp in ["n", "p"]}
        # all_ibands = []
        # for i, tp in enumerate(["p", "n"]):
        #     for ib in range(self.cbm_vbm[tp]["included"]):
        #         sgn = (-1) ** i
        #         all_ibands.append(self.cbm_vbm[tp]["bidx"] + sgn * ib)
        #
        # start_time = time.time()
        #
        # engre, latt_points, nwave, nsym, nsymop, symop, br_dir = analytical_bands.get_engre(iband=all_ibands)
        # nstv, vec, vec2 = analytical_bands.get_star_functions(latt_points, nsym, symop, nwave, br_dir=br_dir)
        # out_vec2 = np.zeros((nwave, max(nstv), 3, 3))
        # for nw in xrange(nwave):
        #     for i in xrange(nstv[nw]):
        #         out_vec2[nw, i] = outer(vec2[nw, i], vec2[nw, i])
        #
        # print("time to get engre and calculate the outvec2: {} seconds".format(time.time() - start_time))
        #
        #
        # # caluclate and normalize the global density of states (DOS) so the integrated DOS == total number of electrons
        # emesh, dos=analytical_bands.get_dos_from_scratch(self._vrun.final_structure,[self.nkdos,self.nkdos,self.nkdos],
        #             self.emin, self.emax, int(self.emax-self.emin)/max(self.dE_global, 0.0001), width=self.dos_bwidth)
        # integ = 0.0
        # for idos in range(len(dos)-2):
        #     integ += (dos[idos+1]+ dos[idos])/2 * (emesh[idos+1] - emesh[idos])
        # # normalize DOS
        # dos = [g/integ * self.nelec for g in dos]
        # self.dos = zip(emesh, dos)

        start_time = time.time()

        low_v_ik = []
        self.initialize_var("kgrid", ["cartesian kpoints"], "vector", 0.0, is_nparray=False, c_T_idx=False)

        # initialize energy, velocity, etc in self.kgrid
        for i, tp in enumerate(["p", "n"]):
            sgn = (-1) ** i
            for ib in range(self.cbm_vbm[tp]["included"]):
                self.kgrid[tp]["cartesian kpoints"][ib]=np.dot(np.array(self.kgrid[tp]["kpoints"][ib]),self._lattice_matrix)/A_to_nm*2*pi #[1/nm]

                # engre, latt_points, nwave, nsym, nsymop, symop, br_dir = \
                #     analytical_bands.get_engre(iband=[self.cbm_vbm[tp]["bidx"] + sgn * ib])
                # bands_data[tp][ib] = (engre, latt_points, nwave, nsym, nsymop, symop, br_dir)

                # if not once_called:
                #     start_time = time.time()
                #     nstv, vec, vec2 = analytical_bands.get_star_functions(latt_points, nsym, symop, nwave,br_dir=br_dir)
                #     out_vec2 = np.zeros((nwave, max(nstv), 3, 3))
                #     for nw in xrange(nwave):
                #         for i in xrange(nstv[nw]):
                #             out_vec2[nw, i] = outer(vec2[nw, i], vec2[nw, i])
                #     print("time to calculate the outvec2: {} seconds".format(time.time() - start_time))
                #
                #     once_called = True

                for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                    if not self.poly_bands:
                        energy, de, dde = analytical_bands.get_energy(
                            self.kgrid[tp]["kpoints"][ib][ik], engre[i*self.cbm_vbm["p"]["included"]+ib],
                                nwave, nsym, nstv, vec, vec2, out_vec2, br_dir=br_dir)
                        energy = energy * Ry_to_eV + sgn * self.scissor/2
                        velocity = abs(de / hbar * A_to_m * m_to_cm * Ry_to_eV)# to get v in cm/s
                        # velocity = de / hbar * A_to_m * m_to_cm * Ry_to_eV# to get v in cm/s
                        effective_mass = hbar ** 2 / (
                        dde * 4 * pi ** 2) / m_e / A_to_m ** 2 * e * Ry_to_eV  # m_tensor: the last part is unit conversion
                    else:
                        # energy, de, dde = get_poly_energy(self.kgrid[tp]["kpoints"][ib][ik], self._lattice_matrix,
                        energy, velocity, effective_mass=get_poly_energy(self.kgrid[tp]["cartesian kpoints"][ib][ik],
                                                                              poly_bands=self.poly_bands,
                            type=tp, ib=ib,bandgap=self.dft_gap+self.scissor)

                        # velocity = abs(de / hbar * A_to_m * m_to_cm)# to get v in cm/s
                        # effective_mass = hbar ** 2 / (dde * 4 * pi ** 2) / m_e / A_to_m ** 2 * e  # m_tensor

                    self.kgrid[tp]["energy"][ib][ik] = energy
                    self.kgrid[tp]["velocity"][ib][ik] = velocity
                    self.kgrid[tp]["norm(1/v)"][ib][ik] = norm(1.0/velocity)

                    # self.kgrid[tp]["velocity"][ib][ik] = de/hbar * A_to_m * m_to_cm * Ry_to_eV # to get v in units of cm/s
                    # TODO: what's the implication of negative group velocities? check later after scattering rates are calculated
                    # TODO: actually using abs() for group velocities mostly increase nu_II values at each energy
                    # TODO: should I have de*2*pi for the group velocity and dde*(2*pi)**2 for effective mass?
                    if self.kgrid[tp]["velocity"][ib][ik][0] < 1 or self.kgrid[tp]["velocity"][ib][ik][1] < 1 \
                            or self.kgrid[tp]["velocity"][ib][ik][2] < 1:
                        low_v_ik.append(ik)
                    self.kgrid[tp]["effective mass"][ib][ik] = effective_mass
                    self.kgrid[tp]["a"][ib][ik] = 1.0


        print "average of the group velocity (to detect inherent or artificially created anisotropy"
        print np.mean(self.kgrid["n"]["velocity"][0], 0)

        #TODO: add kpoints and make smarter kgrid where energy values below CBM+dE and above VBM-dE (if necessary) are added in a way that Ediff is smaller so POP scattering is done more accurately and convergance obtained more easily and with much less k-points
        # kpoints_added = {"n": [], "p": []}
        # adaptive_E = 0.5
        # for tp in ["n", "p"]:
        #     #TODO: if this worked, change it so that if self.dopings does not involve either of the types, don't add k-points for it
        #     ies_sorted = np.argsort(self.kgrid[tp]["energy"][0])
        #     # print ies_sorted
        #     # print len(ies_sorted)
        #     # print len(self.kgrid[tp]["energy"][0])
        #     for ie in ies_sorted:
        #         if abs(self.kgrid[tp]["energy"][0][ie]-self.kgrid[tp]["energy"][0][ies_sorted[0]]) < adaptive_E:
        #             kpoints_added[tp].append(ie)
        #         else:
        #             break
        #     kpoints_added[tp] = np.array([self.kgrid[tp]["kpoints"][0][ik] for ik in kpoints_added[tp]])
        #
        # offset = 0.05
        # nsteps = 10
        # print "here initial k-points with low energy distance"
        # print len(kpoints_added["n"])
        # # print kpoints_added["n"]
        # for tp in ["n", "p"]:
        #     final_kpts_added = []
        #     for ik in range(len(kpoints_added[tp])-2):
        #         final_kpts_added += self.get_intermediat_kpoints(kpoints_added[tp][ik], kpoints_added[tp][ik+1], nsteps)
        #
        #     # temp = kpoints_added[tp]
        #     # kpoints_added[tp] = np.concatenate( (kpoints_added[tp], temp + np.array([0.0 , 0.0, offset])), axis=0 )
        #     # kpoints_added[tp] = np.concatenate( (kpoints_added[tp], temp + np.array([0.0 , offset, 0.0])), axis=0 )
        #     # kpoints_added[tp] = np.concatenate( (kpoints_added[tp], temp + np.array([offset , 0.0, 0.0])), axis=0 )
        # # print "here 1"
        # # print len(kpoints_added["n"])
        # # print kpoints_added["n"]
        #
        #
        # print "here length of added k-points"
        # print len(final_kpts_added)
        # final_kpts_added = self.remove_duplicate_kpoints(final_kpts_added)
        # print len(final_kpts_added)


        # for i, tp in enumerate(["p", "n"]):
        #     sgn = (-1) ** i
        #     added_energy = []
        #     added_velocity = [[] for j in range(self.cbm_vbm[tp]["included"])]
        #     added_mass = [[] for j in range(self.cbm_vbm[tp]["included"]) ]
        #     added_a = []
        #     for ib in range(self.cbm_vbm[tp]["included"]):
        #         for ik in range(len(final_kpts_added)):
        #             # self.kgrid[tp]["kpoints"][ib].append(final_kpts_added[ik])
        #             energy, de, dde = analytical_bands.get_energy(
        #                 final_kpts_added[ik], engre[i*self.cbm_vbm["p"]["included"]+ib],
        #                     nwave, nsym, nstv, vec, vec2, out_vec2, br_dir=br_dir)
        #
        #             added_energy.append(energy * Ry_to_eV + sgn * self.scissor/2)
        #             added_velocity[ib].append(abs(de / hbar * A_to_m * m_to_cm * Ry_to_eV))  # to get v in units of cm/s
        #             if added_velocity[ib][-1][0] < 1 or added_velocity[ib][-1][1] < 1 or added_velocity[ib][-1][2] < 1:
        #                 low_v_ik.append(len(self.kgrid[tp]["kpoints"][ib]) + ik)
        #             added_mass[ib].append(hbar ** 2 / (dde * 4 * pi ** 2) / m_e / A_to_m ** 2 * e * Ry_to_eV)  # m_tensor: the last part is unit conversion
        #             added_a.append(1.0)
        #
        #         self.kgrid[tp]["kpoints"][ib] = np.concatenate((self.kgrid[tp]["kpoints"][ib], final_kpts_added))
        #         self.kgrid[tp]["energy"][ib] = np.concatenate((self.kgrid[tp]["energy"][ib], added_energy))
        #         # print self.kgrid[tp]["velocity"][ib]
        #         # print added_velocity
        #         self.kgrid[tp]["a"][ib] = np.concatenate((self.kgrid[tp]["a"][ib], added_a))
        #         self.kgrid[tp]["c"][ib] += np.array([0.0 for i in range(len(final_kpts_added))])
        #         self.kgrid[tp]["kweights"][ib] += [0.0 for i in range(len(final_kpts_added))]
        #
        #     temp = self.kgrid[tp]["velocity"]
        #     self.kgrid[tp]["velocity"] = [ [v for v in np.concatenate((temp[ib], added_velocity[ib]),axis=0)] for ib in range(self.cbm_vbm[tp]["included"])]
        #     temp = self.kgrid[tp]["effective mass"]
        #     self.kgrid[tp]["effective mass"] = [ m for m in np.concatenate((temp[ib], added_mass[ib])) for ib in range(self.cbm_vbm[tp]["included"]) ]



        print "here are the iks with very low velocity"
        print low_v_ik

        rearranged_props = ["velocity","effective mass","energy", "a", "c", "kpoints","cartesian kpoints","kweights",
                             "norm(1/v)"]
        if len(low_v_ik) > 0:
            self.omit_kpoints(low_v_ik, rearranged_props=rearranged_props)

        print "here the final number of k-points"
        print len(self.kgrid["n"]["kpoints"][0])

        if len(self.kgrid["n"]["kpoints"][0]) < 5:
            raise ValueError("VERY BAD k-mesh; please change the setting for k-mesh and try again!")

        print("time to calculate energy, velocity, m* for all: {} seconds".format(time.time() - start_time))
        # print self.kgrid["n"]["energy"]

        # sort "energy", "kpoints", "kweights", etc based on energy in ascending order
        self.sort_vars_based_on_energy(args=rearranged_props, ascending=True)

        # to save memory avoiding storage of variables that we don't need down the line
        for tp in ["n", "p"]:
            self.kgrid[tp].pop("effective mass", None)
            self.kgrid[tp].pop("kweights", None)
            self.kgrid[tp]["size"] = len(self.kgrid[tp]["kpoints"][0])

            print "energy of {} band:".format(["conduction", "valence"][["n", "p"].index(tp)])
            # self.kgrid["n"]["energy"][0].sort()
            maxdata = 20
            print self.kgrid[tp]["energy"][0][0:min(maxdata,len(self.kgrid[tp]["energy"][0]))]
            print "..."
            print self.kgrid[tp]["energy"][0][-min(maxdata,len(self.kgrid[tp]["energy"][0])):-1]

        # print "velocity of conduction band:"
        # a = [norm (v) for v in self.kgrid["n"]["velocity"][0]]
        # a.sort()
        # print a[0:min(maxdata,len(a))]
        # print "..."
        # print a[-min(maxdata,len(a)):-1]


        self.initialize_var("kgrid", ["W_POP"], "scalar", 0.0, is_nparray=False, c_T_idx=False)
        self.initialize_var("kgrid", ["N_POP"], "scalar", 0.0, is_nparray=False, c_T_idx=True)

        for tp in ["n", "p"]:
            for ib in range(self.cbm_vbm[tp]["included"]):
        # TODO: change how W_POP is set, user set a number or a file that can be fitted and inserted to kgrid
                self.kgrid[tp]["W_POP"][ib] = [self.W_POP for i in range(len(self.kgrid[tp]["kpoints"][ib]))]
                for c in self.dopings:
                    for T in self.temperatures:
                        self.kgrid[tp]["N_POP"][c][T][ib] = np.array([ 1/(np.exp(hbar * W_POP/(k_B * T))-1) for W_POP in self.kgrid[tp]["W_POP"][ib]])
        # Match the CBM/VBM energy values to those obtained from the coefficients file rather than vasprun.xml
        # self.cbm_vbm["n"]["energy"] = self.kgrid["n"]["energy"][0][0]
        # self.cbm_vbm["p"]["energy"] = self.kgrid["p"]["energy"][0][-1]

        self.initialize_var(grid="kgrid", names=["_all_elastic", "S_i", "S_i_th", "S_o", "S_o_th", "g", "g_th", "g_POP",
                "f", "f_th", "relaxation time", "df0dk", "electric force", "thermal force"],
                        val_type="vector", initval=self.gs, is_nparray=True, c_T_idx=True)
        self.initialize_var("kgrid", ["f0", "f_plus", "f_minus","g_plus", "g_minus"], "vector", self.gs, is_nparray=True, c_T_idx=True)
        # self.initialize_var("kgrid", ["lambda_i_plus", "lambda_i_minus", "lambda_o_plus", "lambda_o_minus"]
        #                     , "vector", self.gs, is_nparray=True, c_T_idx=False)


    def sort_vars_based_on_energy(self, args, ascending=True):
        """sort the list of variables specified by "args" (type: [str]) in self.kgrid based on the "energy" values
        in each band for both "n"- and "p"-type bands and in ascending order by default."""


        # ikidxs = {"n": [], "p": []}
        # for tp in ["n", "p"]:
        #     ikidxs[tp] = [np.argsort(self.kgrid[tp]["energy"][ib]) for ib in range(self.cbm_vbm[tp]["included"])]
        #
        # for tp in ["n", "p"]:
        #     for arg in args:
        #         # if arg in ["k-points", "kweights"]:
        #         #     self.kgrid[arg] = np.array([self.kgrid[arg][ik] for ik in ikidxs])
        #         # else:
        #         self.kgrid[tp][arg] = np.array([[self.kgrid[tp][arg][ib][ik] for ik in ikidxs[tp][ib]] for ib in range(self.cbm_vbm[tp]["included"])])


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


    def generate_angles_and_indexes_for_integration(self, avg_Ediff_tolerance=0.01):


        # def is_sparse(list_of_lists, threshold=0.1):
        #     """check to see if a list of lists has more than certain fraction (threshold) of empty lists inside."""
        #     counter = 0
        #     for i in list_of_lists:
        #         if len(i)==0:
        #             counter += 1
        #     if counter/len(list_of_lists) > threshold:
        #         return True
        #     return False

        # for each energy point, we want to store the ib and ik of those points with the same E, E士hbar*W_POP
        # for tp in ["n", "p"]:
        #     for angle_index_for_integration in ["X_E_ik", "X_Eplus_ik", "X_Eminus_ik"]:
        #         self.kgrid[tp][angle_index_for_integration] = [ [ [] for i in range(len(self.kgrid[tp]["kpoints"][ib])) ]
        #                                                           for j in range(self.cbm_vbm[tp]["included"]) ]
        self.initialize_var("kgrid",["X_E_ik", "X_Eplus_ik", "X_Eminus_ik"],"scalar",[],is_nparray=False, c_T_idx=False)
        # for tp in ["n", "p"]:
        #     for ib in range(len(self.kgrid[tp]["energy"])):
        #         for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
        #             for ib_prm in range(len(self.kgrid[tp]["energy"])):
        #                 for ik_prm in range(len(self.kgrid[tp]["kpoints"][ib])):
        #                     k = self.kgrid[tp]["cartesian kpoints"][ib][ik]
                            # E = self.kgrid[tp]["energy"][ib][ik]
                            # X = self.cos_angle(self.kgrid[tp]["cartesian kpoints"][ib][ik], self.kgrid[tp]["cartesian kpoints"][ib][ik_prm])
                            #
                            # if abs(E - self.kgrid[tp]["energy"][ib_prm][ik_prm]) < self.dE_global:
                            #      self.kgrid[tp]["X_E_ik"][ib][ik].append((X, ib_prm, ik_prm))
                            # if abs( (E +  hbar * self.kgrid[tp]["W_POP"][ib][ik] ) \
                            #                      - self.kgrid[tp]["energy"][ib_prm][ik_prm]) < self.dE_global:
                            #     self.kgrid[tp]["X_Eplus_ik"][ib][ik].append((X, ib_prm, ik_prm))
                            # if abs( (E -  hbar * self.kgrid[tp]["W_POP"][ib][ik] ) \
                            #                      - self.kgrid[tp]["energy"][ib_prm][ik_prm]) < self.dE_global:
                            #     self.kgrid[tp]["X_Eminus_ik"][ib][ik].append((X, ib_prm, ik_prm))
                    #
                    # self.kgrid[tp]["X_E_ik"][ib][ik].sort()
                    # self.kgrid[tp]["X_Eplus_ik"][ib][ik].sort()
                    # self.kgrid[tp]["X_Eminus_ik"][ib][ik].sort()


        for tp in ["n", "p"]:
            for ib in range(len(self.kgrid[tp]["energy"])):
                self.nforced_scat = {"n": 0.0, "p": 0.0}
                self.ediff_scat = {"n": [], "p": []}
                for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                    self.kgrid[tp]["X_E_ik"][ib][ik] = self.get_X_ib_ik_within_E_radius(tp,ib,ik,
                                                    E_radius=0.0, forced_min_npoints=2, tolerance=self.dE_global)
                print "here nforced k-points ratio for elastic scattering"
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
                    # raise ValueError("{}-type average energy difference of the enforced scattered k-points is more than"
                    #                  " {}, try running with a more dense k-point mesh".format(tp, avg_Ediff_tolerance))




        if "POP" in self.inelastic_scatterings:
            for tp in ["n", "p"]:
                for ib in range(len(self.kgrid[tp]["energy"])):
                    self.nforced_scat = {"n": 0.0, "p": 0.0}
                    self.ediff_scat = {"n": [], "p": []}
                    for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                        self.kgrid[tp]["X_Eplus_ik"][ib][ik] = self.get_X_ib_ik_within_E_radius(tp,ib,ik,
                            E_radius= + hbar * self.kgrid[tp]["W_POP"][ib][ik], forced_min_npoints=2, tolerance=self.dE_global)
                        self.kgrid[tp]["X_Eminus_ik"][ib][ik] = self.get_X_ib_ik_within_E_radius(tp, ib, ik,
                            E_radius= - hbar * self.kgrid[tp]["W_POP"][ib][ik],forced_min_npoints=2, tolerance=self.dE_global)


                # if is_sparse(self.kgrid[tp]["X_E_ik"][ib]):
                #     raise ValueError("the k-grid is too coarse for an acceptable simulation of elastic scattering")
                # if is_sparse(self.kgrid[tp]["X_Eplus_ik"][ib]) or is_sparse(self.kgrid[tp]["X_Eminus_ik"][ib]):
                #     if "POP" in self.inelastic_scatterings:
                #         raise ValueError("the k-grid is too coarse for an acceptable simulation of POP scattering, "
                        # warnings.warn("the k-grid is too coarse for an acceptable simulation of POP scattering, "
                        #                  "you can try this k-point grid but without POP as an inelastic scattering")

                    # if self.nforced_scat[tp]/(len(self.kgrid[tp]["energy"])*len(self.kgrid[tp]["kpoints"][ib])) > 0.1:

                    print "here nforced k-points ratio for POP scattering"
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
                        raise ValueError(
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
        forced = False
        E = self.kgrid[tp]["energy"][ib][ik]
        k = self.kgrid[tp]["cartesian kpoints"][ib][ik]
        result = []
        counter = 0
        # if E_radius == 0.0: # because then all the symmetrically equivalent k-points to current k have the same energy
        #     # print "BASE", ib, ik, E_radius
        #     symmetrically_equivalent_ks = self.unique_X_ib_ik_symmetrically_equivalent(tp, ib, ik)
        #     result += symmetrically_equivalent_ks
        #     counter += len(symmetrically_equivalent_ks)
        nk = len(self.kgrid[tp]["kpoints"][ib])

        min_vdiff = 3*hbar*e*1e11*1e-4/m_e

        for ib_prm in range(self.cbm_vbm[tp]["included"]):
            if ib==ib_prm and E_radius==0.0:
                ik_prm = ik
            else:
                ik_prm = np.abs(self.kgrid[tp]["energy"][ib_prm] - (E + E_radius)).argmin() - 1
                # ik_prm = max(0, ik-100) # different bands, we start comparing energies ahead as energies aren't equal
            while (ik_prm<nk-1) and abs(self.kgrid[tp]["energy"][ib_prm][ik_prm+1]-(E+E_radius)) < tolerance:
            # while (ik_prm < nk - 1) and self.kgrid[tp]["energy"][ib_prm][ik_prm + 1] == (E + E_radius):
                ik_prm += 1
                # if E_radius > 0.0:
                #     print "AFTER", ib, ik, E_radius
                # k_prm = self.kgrid[tp]["cartesian kpoints"][ib_prm][ik_prm+1]
                # X = self.cos_angle(k, k_prm)
                # result.append((X, ib_prm, ik_prm + 1))
                # result.append((X, ib_prm, ik_prm, k_prm))
                result.append((self.cos_angle(k, self.kgrid[tp]["cartesian kpoints"][ib_prm][ik_prm]),ib_prm,ik_prm))
                # symmetrically_equivalent_ks = self.unique_X_ib_ik_symmetrically_equivalent(tp, ib_prm, ik_prm)
                # result += symmetrically_equivalent_ks


                # counter += len(symmetrically_equivalent_ks) + 1
                counter += 1

                # if abs(sum(self.kgrid["n"]["velocity"][ib][ik]) - sum(self.kgrid["n"]["velocity"][ib][ik_prm])) < min_vdiff:
                #     counter -= 1

            if ib==ib_prm and E_radius==0.0:
                ik_prm = ik
            else:
                ik_prm = np.abs(self.kgrid[tp]["energy"][ib_prm] - (E + E_radius)).argmin() + 1
                # ik_prm = min(nk-1, ik+100)
            while (ik_prm>0) and abs(E+E_radius - self.kgrid[tp]["energy"][ib_prm][ik_prm-1]) < tolerance:
            # while (ik_prm > 0) and E + E_radius == self.kgrid[tp]["energy"][ib_prm][ik_prm - 1]:
                ik_prm -= 1
                # if E_radius > 0.0:
                #     print "BEFORE", ib, ik, E_radius
                # X = self.cos_angle(k, self.kgrid[tp]["cartesian kpoints"][ib_prm][ik_prm - 1])
                # result.append((X, ib_prm, ik_prm - 1))
                result.append((self.cos_angle(k, self.kgrid[tp]["cartesian kpoints"][ib_prm][ik_prm]), ib_prm, ik_prm))
                # symmetrically_equivalent_ks = self.unique_X_ib_ik_symmetrically_equivalent(tp, ib_prm, ik_prm)
                # result += symmetrically_equivalent_ks

                # counter += len(symmetrically_equivalent_ks) + 1
                counter += 1

                # if abs(sum(self.kgrid["n"]["velocity"][ib][ik]) - sum(self.kgrid["n"]["velocity"][ib][ik_prm])) < min_vdiff:
                #     counter -= 1

        # If fewer than forced_min_npoints number of points were found, just return a few surroundings of the same band

        ik_prm = ik
        while counter < forced_min_npoints and ik_prm < nk - 1:
            # if E_radius >  0.0:
            #     print "EXTRA 1", ib, ik, E_radius
            ik_prm += 1
            k_prm = self.kgrid[tp]["cartesian kpoints"][ib][ik_prm]


            result.append((self.cos_angle(k, k_prm), ib, ik_prm))

            # also add all values with the same energy at ik_prm
            result += self.kgrid[tp]["X_E_ik"][ib][ik_prm]

            # for X_ik_ib in self.kgrid[tp]["X_E_ik"][ib][ik_prm]:
            #     print "added"
            #     X, ib_new, ik_new = X_ik_ib
            #     k_new = self.kgrid[tp]["cartesian kpoints"][ib_new][ik_new]
            #     result.append(self.cos_angle(k_prm, k_new), ib_new, ik_new)
            #     counter += 1

            # symmetrically_equivalent_ks = self.unique_X_ib_ik_symmetrically_equivalent(tp, ib_prm, ik_prm)
            # result += symmetrically_equivalent_ks
            # result.append((self.cos_angle(k, self.kgrid[tp]["cartesian kpoints"][ib][ik_prm + 1]), ib, ik_prm + 1))

            # counter += 1 + len(symmetrically_equivalent_ks)
            counter += 1

            if abs(sum(self.kgrid["n"]["velocity"][ib][ik]) - sum(self.kgrid["n"]["velocity"][ib][ik_prm])) < min_vdiff:
                counter -= 1
                self.nforced_scat[tp] -= 1

            self.nforced_scat[tp] += 1
            self.ediff_scat[tp].append(self.kgrid[tp]["energy"][ib][ik_prm]-self.kgrid[tp]["energy"][ib][ik])


        ik_prm = ik
        while counter < forced_min_npoints and ik_prm > 0:
            # if E_radius > 0.0:
            #     print "EXTRA 2", ib, ik, E_radius
            # k_prm = self.kgrid[tp]["cartesian kpoints"][ib][ik_prm - 1]
            ik_prm -= 1
            result.append((self.cos_angle(k, self.kgrid[tp]["cartesian kpoints"][ib][ik_prm]), ib, ik_prm))

            # also add all values with the same energy at ik_prm
            result += self.kgrid[tp]["X_E_ik"][ib][ik_prm]


            # symmetrically_equivalent_ks = self.unique_X_ib_ik_symmetrically_equivalent(tp, ib_prm, ik_prm)
            # result += symmetrically_equivalent_ks
            # result.append((self.cos_angle(k, self.kgrid[tp]["cartesian kpoints"][ib][ik_prm - 1]), ib, ik_prm - 1))

            # counter += 1 + len(symmetrically_equivalent_ks)
            counter += 1

            if abs(sum(self.kgrid["n"]["velocity"][ib][ik]) - sum(self.kgrid["n"]["velocity"][ib][ik_prm])) < min_vdiff:
                counter -= 1
                self.nforced_scat[tp] -= 1
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
        norm_diff_k = norm(k - k_prm)
        if norm_diff_k == 0:
            print "WARNING!!! same k and k' vectors as input of the elastic scattering equation"
            # warnings.warn("same k and k' vectors as input of the elastic scattering equation")
            # raise ValueError("same k and k' vectors as input of the elastic scattering equation."
            #                  "Check get_X_ib_ik_within_E_radius for possible error")
            return 0

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

        # isotropic PIE
        #     return (e ** 2 * k_B * T * self.P_PIE ** 2) / (
        #         6 * pi * hbar ** 2 * self.epsilon_s * epsilon_0 * v) * (
        #                    3 - 6 * par_c ** 2 + 4 * par_c ** 4) * 100 / e


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
                # TODO:The DOS used is too simplistic and wrong (e.g., calc_doping might hit a limit), try 2*[2pim_hk_BT/hbar**2]**1.5
                # integral += dE * (self.egrid[tp]["DOS"][ie] + i * dS) * func(E + i * dE, fermi, T)
                integral += dE * (self.egrid[tp]["DOS"][ie] + i * dS)*func(E + i * dE, fermi, T)*self.Efrequency[tp][ie]
        return integral
        # return integral/sum(self.Efrequency[tp][:-1])



    def integrate_over_E(self, prop_list, tp, c, T, xDOS=True, xvel=False, weighted=False, interpolation_nsteps=None):
        if not interpolation_nsteps:
            interpolation_nsteps = max(5, int(500.0/len(self.egrid[tp]["energy"])) )
        diff = [0.0 for prop in prop_list]
        integral = self.gs
        for ie in range(len(self.egrid[tp]["energy"]) - 1):
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
                    integral += multi * self.Efrequency[tp][ie]
                else:
                    integral += multi
        if weighted:
            return integral/sum(self.Efrequency[tp][:-1])
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
        # k = m_e * self.kgrid[tp]["velocity"][ib][ik] / (hbar * e * 1e11)
        # k_prm = m_e * self.kgrid[tp]["velocity"][ib_prm][ik_prm] / (hbar * e * 1e11)

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

        return (1 - X) * (m_e * self.kgrid[tp]["velocity"][ib_prm][ik_prm] / (
        hbar * e * 1e11)) ** 2 * self.s_el_eq(sname, tp, c, T, k, k_prm) \
               * self.G(tp, ib, ik, ib_prm, ik_prm, X) \
               * 1.0/self.kgrid[tp]["velocity"][ib_prm][ik_prm]


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
        integ = self.G(tp, ib, ik, ib_prm, ik_prm, X)*self.kgrid[tp]["norm(1/v)"][ib_prm][ik_prm]
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


    def s_inel_eq_isotropic(self, g_suffix, once_called=False):
        for tp in ["n", "p"]:
            for c in self.dopings:
                for T in self.temperatures:
                    for ib in range(len(self.kgrid[tp]["energy"])):
                        for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                            # S_i = np.array([self.gs, self.gs, self.gs])
                            S_i = self.gs
                            # S_o = np.array([self.gs, self.gs, self.gs])

                            v = sum(self.kgrid[tp]["velocity"][ib][ik]) / 3
                            # v = self.kgrid[tp]["velocity"][ib][ik]

                            k = m_e * v / (hbar * e * 1e11)
                            a = self.kgrid[tp]["a"][ib][ik]
                            c_ = self.kgrid[tp]["c"][ib][ik]
                            f = self.kgrid[tp]["f0"][c][T][ib][ik]
                            N_POP = 1 / (np.exp(hbar * self.kgrid[tp]["W_POP"][ib][ik] / (k_B * T)) - 1)
                            # N_POP = self.kgrid[tp]["N_POP"][c][T][ib][ik]
                            for j, X_Epm in enumerate(["X_Eplus_ik", "X_Eminus_ik"]):
                                len_eqE = len(self.kgrid[tp][X_Epm][ib][ik])
                                # if len_eqE == 0:
                                #     print "WARNING!!!! element {} of {} is empty!!".format(ik, X_Epm)
                                for X_ib_ik in self.kgrid[tp][X_Epm][ib][ik]:
                                    X, ib_pm, ik_pm = X_ib_ik

                                    v_pm = sum(self.kgrid[tp]["velocity"][ib_pm][ik_pm])/3
                                    # v_pm = self.kgrid[tp]["velocity"][ib_pm][ik_pm]

                                    k_pm  = m_e*v_pm/(hbar*e*1e11)

                                    abs_kdiff = abs(k_pm - k)
                                    if abs_kdiff < 1e-4:
                                        continue

                                    a_pm = self.kgrid[tp]["a"][ib_pm][ik_pm]
                                    c_pm = self.kgrid[tp]["c"][ib_pm][ik_pm]
                                    # g_pm = sum(self.kgrid[tp]["g"+g_suffix][c][T][ib_pm][ik_pm])/3
                                    g_pm = self.kgrid[tp]["g"+g_suffix][c][T][ib_pm][ik_pm]

                                    f_pm = self.kgrid[tp]["f0"][c][T][ib_pm][ik_pm]

                                    A_pm = a*a_pm + c_*c_pm*(k_pm**2+k**2)/(2*k_pm*k)

                                    beta_pm = (e**2*self.kgrid[tp]["W_POP"][ib_pm][ik_pm]*k_pm)/(4*pi*hbar*k*v_pm)*\
                                        (1/(self.epsilon_inf*epsilon_0)-1/(self.epsilon_s*epsilon_0))*6.2415093e20

                                    if not once_called:
                                        lamb_opm=beta_pm*(A_pm**2*log((k_pm+k)/abs_kdiff+1e-4)-A_pm*c_*c_pm-a*a_pm*c_*c_pm)
                                        # because in the scalar form k+ or k- is suppused to be unique, here we take average
                                        self.kgrid[tp]["S_o" + g_suffix][c][T][ib][ik] +=((N_POP + j+(-1)**j*f_pm)*lamb_opm)/len_eqE
                                        # S_o +=((N_POP + j+(-1)**j*f_pm)*lamb_opm)/len_eqE
                                        # if S_o[0] < 1:
                                        #     print "WARNINGGGGG!!!!"
                                        #     warnings.warn("abnormal S_o")
                                        #     print ik
                                        #     print ik_pm
                                        #     print abs_kdiff

                                    lamb_ipm=beta_pm*(A_pm**2*log((k_pm+k)/abs_kdiff+1e-4)*(k_pm**2+k**2)/(2*k*k_pm)-
                                                      A_pm**2-c_**2*c_pm**2/3)

                                    S_i += ((N_POP + (1-j) + (-1)**(1-j)*f) * lamb_ipm * g_pm)/len_eqE

                            # self.kgrid[tp]["S_o" + g_suffix][c][T][ib][ik] = S_o
                            self.kgrid[tp]["S_i" + g_suffix][c][T][ib][ik] = S_i



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
        v = self.kgrid[tp]["velocity"][ib][ik] # because it's isotropic, it doesn't matter which one we choose
        # perhaps more correct way of defining knrm is as follows since at momentum is supposed to be proportional to
        # velocity as it is in free-electron formulation so we replaced hbar*knrm with m_e*v/(1e11*e) (momentum)

        # knrm = norm(self.kgrid[tp]["kpoints"][ib][ik]-np.dot(self.cbm_vbm[tp]["kpoint"], self._lattice_matrix)*2*pi*1/A_to_nm)
        knrm = m_e*v/(hbar*e*1e11)

        par_c = self.kgrid[tp]["c"][ib][ik]

        if sname.upper() == "ACD":
            # The following two lines are from Rode's chapter (page 38) which seems incorrect!
            return (k_B*T*self.E_D[tp]**2*knrm**2)/(3*pi*hbar**2*self.C_el*1e9* v)\
            *(3-8*self.kgrid[tp]["c"][ib][ik]**2+6*self.kgrid[tp]["c"][ib][ik]**4)*e*1e20

            # return (k_B * T * self.E_D[tp] ** 2 * knrm ** 2) *norm(1.0/v)/ (3 * pi * hbar ** 2 * self.C_el * 1e9) \
            #     * (3 - 8 * self.kgrid[tp]["c"][ib][ik] ** 2 + 6 * self.kgrid[tp]["c"][ib][ik] ** 4) * e * 1e20

            # it is equivalent to the following also from Rode but always isotropic
            # return m_e * knrm * self.E_D[tp] ** 2 * k_B * T / ( 3* pi * hbar ** 3 * self.C_el) \
            #            * (3 - 8 * par_c ** 2 + 6 * par_c ** 4) * 1  # units work out! that's why conversion is 1


            # The following is from Deformation potentials and... Ref. [Q] (DOI: 10.1103/PhysRev.80.72 ) page 82?
            # if knrm < 1/(0.1*self._vrun.lattice.c*A_to_nm):

            # replaced hbar*knrm with m_e*norm(v)/(1e11*e) which is momentum
            # return m_e * m_e*norm(v) * self.E_D[tp] ** 2 * k_B * T / (2 * pi * hbar ** 4 * self.C_el) \
            #        * (3 - 8 * par_c ** 2 + 6 * par_c ** 4) / (1e11*e) # 1/1e11*e is to convert kg.cm/s to hbar.k units (i.e. ev.s/nm)

        elif sname.upper() == "IMP":
            beta = self.egrid["beta"][c][T][tp]
            B_II = (4*knrm**2/beta**2)/(1+4*knrm**2/beta**2)+8*(beta**2+2*knrm**2)/(beta**2+4*knrm**2)*par_c**2+\
                   (3*beta**4+6*beta**2*knrm**2-8*knrm**4)/((beta**2+4*knrm**2)*knrm**2)*par_c**4
            D_II = 1+(2*beta**2*par_c**2/knrm**2)+(3*beta**4*par_c**4/(4*knrm**4))
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
                            self.egrid[tp][prop_name][ie]=np.array([sum(self.egrid[tp][prop_name][ie])/3.0 for i in range(3)])


                else:
                    for ie, en in enumerate(self.egrid[tp]["energy"]):
                        N = 0.0  # total number of instances with the same energy
                        for ib in range(self.cbm_vbm[tp]["included"]):
                            for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                                self.egrid[tp][prop_name][ie] += self.kgrid[tp][prop_name][ib][ik] * \
                                    GB(self.kgrid[tp]["energy"][ib][ik]-self.egrid[tp]["energy"][ie], 0.005)

                        self.egrid[tp][prop_name][ie] /= self.cbm_vbm[tp]["included"] * len(self.kgrid[tp]["kpoints"][0])

                        if self.bs_is_isotropic and prop_type=="vector":
                            self.egrid[tp][prop_name][ie]=np.array([sum(self.egrid[tp][prop_name][ie])/3.0 for i in range(3)])
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
                                        [sum(self.egrid[tp][prop_name][c][T][ie])/3.0 for i in range(3)])

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
                                        [sum(self.egrid[tp][prop_name][c][T][ie])/3.0 for i in range(3)])

    def find_fermi(self, c, T, tolerance=0.001, tolerance_loose=0.03, alpha = 0.02, max_iter = 1000):
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
        iter = 0
        temp_doping = {"n": 0.0, "p": 0.0}
        typ = self.get_tp(c)
        fermi = self.cbm_vbm[typ]["energy"]
        # fermi = self.egrid[typ]["energy"][0]
        j = ["n", "p"].index(typ)
        funcs = [lambda E, fermi0, T: f0(E,fermi0,T), lambda E, fermi0, T: 1-f0(E,fermi0,T)]
        calc_doping = (-1)**(j+1) /self.volume / (A_to_m*m_to_cm)**3 \
                *abs(self.integrate_over_DOSxE_dE(func=funcs[j], tp=typ, fermi=fermi, T=T))

        print "fermi"
        print fermi
        while (relative_error > tolerance) and (iter<max_iter):
            iter += 1 # to avoid an infinite loop
            fermi += alpha * (calc_doping - c)/abs(c + calc_doping) * fermi

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
            integral = self.integrate_over_E(prop_list=["f0x1-f0"], tp=tp, c=c, T=T, xDOS=True)
            integral *= self.nelec
            beta[tp] = (e**2 / (self.epsilon_s * epsilon_0*k_B*T) * integral * 6.241509324e27)**0.5
        return beta



    def to_json(self, kgrid=True, trimmed=False, max_ndata = None, nstart=0):

        if not max_ndata:
            max_ndata = int(self.gl)

        egrid = deepcopy(self.egrid)
        # self.egrid trimming
        if trimmed:
            nmax = min([max_ndata+1, min([len(egrid["n"]["energy"]), len(egrid["p"]["energy"])]) ])
            print nmax
            remove_list = []
            for tp in ["n", "p"]:
                for rm in remove_list:
                    try:
                        del (egrid[tp][rm])
                    except:
                        pass

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
                remove_list = ["W_POP", "effective mass"]
                for tp in ["n", "p"]:
                    for rm in remove_list:
                        try:
                            del (kgrid[tp][rm])
                        except:
                            pass

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
                for g_suffix in ["", "_th"]:
                    if self.bs_is_isotropic:
                        if iter==0:
                            self.s_inel_eq_isotropic(g_suffix=g_suffix)
                        else:
                            self.s_inel_eq_isotropic(g_suffix=g_suffix, once_called=True)

                        # print "here S_o"
                        # print self.kgrid["n"]["S_o"][-1e19][300.0]
                    else:
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

                            self.kgrid[tp]["g"][c][T] = (self.kgrid[tp]["S_i"][c][T] + self.kgrid[tp]["electric force"][c][
                                T]) / (self.kgrid[tp]["S_o"][c][T] + self.kgrid[tp]["_all_elastic"][c][T])
                            self.kgrid[tp]["g_POP"][c][T][ib] = (self.kgrid[tp]["S_i"][c][T][ib] +
                                self.kgrid[tp]["electric force"][c][T][ib]) / (self.kgrid[tp]["S_o"][c][T][ib]+ self.gs)
                            self.kgrid[tp]["g_th"][c][T][ib]=(self.kgrid[tp]["S_i_th"][c][T][ib]+self.kgrid[tp]["thermal force"][c][
                                T][ib]) / (self.kgrid[tp]["S_o_th"][c][T][ib] + self.kgrid[tp]["_all_elastic"][c][T][ib])

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
        for c in self.dopings:
            for T in self.temperatures:
                for tp in ["n", "p"]:
                    # norm is only for one vector but g has the ibxikx3 dimensions
                    # self.egrid[tp]["f"][c][T] = self.egrid[tp]["f0"][c][T] + norm(self.egrid[tp]["g"][c][T])
                    # self.egrid[tp]["f_th"][c][T]=self.egrid[tp]["f0"][c][T]+norm(self.egrid[tp]["g_th"][c][T])

                    # this ONLY makes a difference if f and f_th are used in the denominator; but f0 is currently used!
                    # self.egrid[tp]["f"][c][T] = self.egrid[tp]["f0"][c][T] + norm(self.egrid[tp]["g"][c][T])
                    # self.egrid[tp]["f_th"][c][T]=self.egrid[tp]["f0"][c][T]+norm(self.egrid[tp]["g_th"][c][T])

                    # mobility numerators
                    for mu_el in self.elastic_scatterings:
                        self.egrid["mobility"][mu_el][c][T][tp] = (-1)*default_small_E/hbar* \
                            self.integrate_over_E(prop_list=["/"+mu_el, "df0dk"], tp=tp, c=c, T=T, xDOS=True, xvel=True, weighted=False)


                    for mu_inel in self.inelastic_scatterings:
                            # calculate mobility["POP"] based on g_POP
                            self.egrid["mobility"][mu_inel][c][T][tp] = self.integrate_over_E(prop_list=["g_"+mu_inel],
                                                                                tp=tp,c=c,T=T,xDOS=True,xvel=True, weighted=False)
                    self.egrid["mobility"]["overall"][c][T][tp]=self.integrate_over_E(prop_list=["g"],
                                                                                tp=tp,c=c,T=T,xDOS=True,xvel=True, weighted=False)

                    self.egrid["J_th"][c][T][tp] = self.integrate_over_E(prop_list=["g_th"],
                            tp=tp, c=c, T=T, xDOS=True, xvel=True, weighted=True) * e * 1e24 # to bring J to A/cm2 units

                    # mobility denominators
                    for transport in self.elastic_scatterings + self.inelastic_scatterings + ["overall"]:
                        self.egrid["mobility"][transport][c][T][tp]/=default_small_E*\
                                        self.integrate_over_E(prop_list=["f0"],tp=tp, c=c, T=T, xDOS=True, xvel=False)

                    self.egrid["J_th"][c][T][tp] /= self.volume*self.integrate_over_E(prop_list=["f0"], tp=tp, c=c,
                                                                                      T=T, xDOS=True, xvel=False)

                    # other semi-empirical mobility values:
                    fermi = self.egrid["fermi"][c][T]
                    energy = self.cbm_vbm[tp]["energy"]

                    # ACD mobility based on single parabolic band extracted from Thermoelectric Nanomaterials,
                    # chapter 1, page 12: "Material Design Considerations Based on Thermoelectric Quality Factor"
                    self.egrid["mobility"]["SPB_ACD"][c][T][tp] = 2**0.5*pi*hbar**4*e*self.C_el*1e9/( # C_el in GPa
                        3*(self.cbm_vbm[tp]["eff_mass_xx"]*m_e)**2.5*(k_B*T)**1.5*self.E_D[tp]**2)\
                        *fermi_integral(0,fermi,T,energy)/fermi_integral(0.5,fermi,T,energy) * e**0.5*1e4 #  to cm2/V.s


                    faulty_overall_mobility = False
                    mu_overrall_norm = norm(self.egrid["mobility"]["overall"][c][T][tp])
                    for transport in self.elastic_scatterings + self.inelastic_scatterings:
                        # averaging all mobility values via Matthiessen's rule
                        self.egrid["mobility"]["average"][c][T][tp] += 1 / self.egrid["mobility"][transport][c][T][tp]
                        if mu_overrall_norm > norm(self.egrid["mobility"][transport][c][T][tp]):
                            faulty_overall_mobility = True # because the overall mobility should be lower than all
                    self.egrid["mobility"]["average"][c][T][tp] = 1 / self.egrid["mobility"]["average"][c][T][tp]

                    # Decide if the overall mobility make sense or it should be equal to average (e.g. when POP is off)
                    if mu_overrall_norm == 0.0 or faulty_overall_mobility:
                        self.egrid["mobility"]["overall"][c][T][tp] = self.egrid["mobility"]["average"][c][T][tp]

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


    def plot(self, path=None):
        if not path:
            path = os.path.join( os.getcwd(), "plots" )
        fformat = "html"

        for tp in ["n"]:
            print('plotting: first set of plots: "relaxation time", "_all_elastic", "ACD", "df0dk"')
            plt = PlotlyFig(plot_mode='offline', y_title="# of repeated energy in kgrid", x_title="Energy (eV)",
                   plot_title=None, filename=os.path.join(path, "{}_{}.{}".format("E_histogram", tp, fformat)),
                            textsize=30, ticksize=45, scale=1, margin_left=120)

            # plt.histogram(x=self.egrid[tp]["energy"], bin_size=binsize, x_start=min(data) - binsize / 2)
            plt.xy_plot(x_col=self.egrid[tp]["energy"], y_col=self.Efrequency[tp])

            # xrange=[self.egrid[tp]["energy"][0], self.egrid[tp]["energy"][0]+0.6])

            # prop_list = ["relaxation time", "_all_elastic", "ACD", "IMP", "PIE", "df0dk"]
            prop_list = ["relaxation time", "_all_elastic", "ACD", "df0dk"]
            if "POP" in self.inelastic_scatterings:
                prop_list += ["g", "g_POP", "S_i", "S_o"]
            for c in self.dopings:
                # for T in self.temperatures:
                for T in [300.0]:
                    for prop_name in prop_list:
                        plt = PlotlyFig(plot_title="c={} 1/cm3, T={} K".format(c, T), x_title="Energy (eV)",
                                y_title=prop_name, hovermode='closest',
                            filename=os.path.join(path, "{}_{}_{}_{}.{}".format(prop_name, tp, c, T, fformat)),
                            plot_mode='offline', username=None, api_key=None, textsize=30, ticksize=25, fontfamily=None,
                            height=800, width=1000, scale=None, margin_top=100, margin_bottom=80, margin_left=120,
                            margin_right=80,
                            pad=0)
                        prop = [norm(p) for p in self.egrid[tp][prop_name][c][T]]
                        plt.xy_plot(x_col=self.egrid[tp]["energy"], y_col=prop)

            print('plotting: second set of plots: "velocity", "Ediff"')

            # plot versus energy in self.egrid
            prop_list = ["velocity", "Ediff"]
            for prop_name in prop_list:
                plt = PlotlyFig(plot_title=None, x_title="Energy (eV)", y_title=prop_name, hovermode='closest',
                            filename=os.path.join(path, "{}_{}.{}".format(prop_name, tp, fformat)),
                 plot_mode='offline', username=None, api_key=None, textsize=30, ticksize=25, fontfamily=None,
                 height=800, width=1000, scale=None, margin_top=100, margin_bottom=80, margin_left=120, margin_right=80,
                 pad=0)
                if "Ediff" in prop_name:
                    y_col = [self.egrid[tp]["energy"][i+1]-\
                                        self.egrid[tp]["energy"][i] for i in range(len(self.egrid[tp]["energy"])-1)]
                else:
                    y_col = [norm(p) for p in self.egrid[tp][prop_name]]
                    plt.xy_plot(x_col=self.egrid[tp]["energy"][:len(y_col)], y_col=y_col)
                    # xrange=[self.egrid[tp]["energy"][0], self.egrid[tp]["energy"][0]+0.6])

            # plot versus norm(k) in self.kgrid
            prop_list = ["energy"]
            eff_m = 0.1
            for prop_name in prop_list:
                # x_col = [norm(k-np.dot(np.array([0.5, 0.5, 0.5]), self._lattice_matrix)/A_to_nm*2*pi) for k in self.kgrid[tp]["cartesian kpoints"][0]]
                x_col = [norm(v)*m_e*eff_m/ (hbar*1e11*e) for v in self.kgrid[tp]["velocity"][0]]
                plt = PlotlyFig(plot_title=None, x_title="k [1/nm] (extracted from momentum, mv)",
                                y_title="{} at the 1st band".format(prop_name), hovermode='closest',
                            filename=os.path.join(path, "{}_{}.{}".format(prop_name, tp, fformat)),
                        plot_mode='offline', username=None, api_key=None, textsize=30, ticksize=25, fontfamily=None,
                    height=800, width=1000, scale=None, margin_left=120, margin_right=80)
                try:
                    y_col = [norm(p) for p in self.kgrid[tp][prop_name][0] ]
                except:
                    y_col = self.kgrid[tp][prop_name][0]
                plt.xy_plot(x_col=x_col, y_col=y_col)
                # xrange=[self.egrid[tp]["energy"][0], self.egrid[tp]["energy"][0]+0.6])

if __name__ == "__main__":
    coeff_file = 'fort.123'

    # test
    AMSET = AMSET()
    # AMSET.run(coeff_file=coeff_file, kgrid_tp="coarse")
    cProfile.run('AMSET.run(coeff_file=coeff_file, kgrid_tp="coarse")')

    AMSET.to_json(kgrid=True, trimmed=True, max_ndata=200, nstart=0)
    # AMSET.to_json(kgrid=True, trimmed=True)
    AMSET.plot()