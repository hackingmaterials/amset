# coding: utf-8

import warnings
from analytical_band_from_BZT import Analytical_bands
from pprint import pprint

import numpy as np
import sys
from pymatgen.io.vasp import Vasprun, Spin
from scipy.constants.codata import value as _cd
from math import pi
import os
import json
from monty.json import MontyEncoder


import cProfile
import re


# global constants
hbar = _cd('Planck constant in eV s')/(2*pi)
m_e = _cd('electron mass') # in kg
Ry_to_eV = 13.605698066
A_to_m = 1e-10
m_to_cm = 100.00
A_to_nm = 0.1
e = _cd('elementary charge')
k_B = _cd("Boltzmann constant in eV/K")
epsilon_0 = 8.854187817e-12     # Absolute value of dielectric constant in vacuum [C^2/m^2N]
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

class AMSET(object):
    """ This class is used to run AMSET on a pymatgen Vasprun object. AMSET is an ab initio model for calculating
    the mobility and Seebeck coefficient using Boltzmann transport equation. The band structure is extracted from
    vasprun.xml to calculate the group velocity and transport properties in presence of various scattering mechanisms.

     Currently the following scattering mechanisms with their corresponding three-letter abbreviations implemented are:
     ionized impurity scattering (iim), acoustic phonon deformation potential (acd), piezoelectric (pie), and charged
     dislocation scattering (dis). Also, longitudinal polar optical phonon (pop) in implemented as an inelastic
     scattering mechanism that can alter the electronic distribution (the reason BTE has to be solved explicitly).

     AMSET is designed in a modular way so that users can add more scattering mechanisms as followed:
     ??? (instruction to add a scattering mechanism) ???
     """


    def __init__(self,

                 N_dis=None, scissor=None,
                 donor_charge=None, acceptor_charge=None, dislocations_charge=None):
        self.dE_global = 0.01 # in eV, the energy difference threshold below which two energy values are assumed equal
        self.dopings = [-1e21] # 1/cm**3 list of carrier concentrations
        self.temperatures = map(float, [300, 600]) # in K, list of temperatures
        self.epsilon_s = 44.360563 # example for PbTe
        self.epsilon_inf = 25.57 # example for PbTe
        self._vrun = {}
        self.max_e_range = 10*k_B*max(self.temperatures) # we set the max energy range after which occupation is zero
        self.path_dir = "../test_files/PbTe_nscf_uniform/nscf_line"
        self.charge = {"n": donor_charge or 1, "p": acceptor_charge or 1, "dislocations": dislocations_charge or 1}
        self.N_dis = N_dis or 0.1 # in 1/cm**2
        self.elastic_scattering_mechanisms = ["IMP", "ACD", "PIE"]
        self.scissor = scissor or 0.0 # total value added to the band gap by adding to the CBM and subtracting from VBM

#TODO: some of the current global constants should be omitted, taken as functions inputs or changed!

        self.wordy = False
        self.maxiters = 20
        self.soc = False
        self.read_vrun(path_dir=self.path_dir, filename="vasprun.xml")
        self.W_POP = 10e12 * 2*pi # POP frequency in Hz
        self.P_PIE = 0.15
        self.E_D = {"n": 4.0, "p": 3.93}
        self.C_el = 139.7 # [Gpa]: spherically averaged elastic constant

    def read_vrun(self, path_dir=".", filename="vasprun.xml"):
        vrun = Vasprun(os.path.join(path_dir, filename))
        self.volume = vrun.final_structure.volume
        self.density = vrun.final_structure.density
        self._lattice_matrix = vrun.lattice_rec.matrix / (2 * pi)
        print self._lattice_matrix
        bs = vrun.get_band_structure()

        # Remember that python band index starts from 0 so bidx==9 refers to the 10th band (e.g. in VASP)
        cbm_vbm = {"n": {"energy": 0.0, "bidx": 0, "included": 0}, "p": {"energy": 0.0, "bidx": 0, "included": 0}}
        cbm = bs.get_cbm()
        vbm = bs.get_vbm()

        cbm_vbm["n"]["energy"] = cbm["energy"]
        cbm_vbm["n"]["bidx"] = cbm["band_index"][Spin.up][0]

        cbm_vbm["p"]["energy"] = vbm["energy"]
        cbm_vbm["p"]["bidx"] = vbm["band_index"][Spin.up][-1]

        if self.soc:
            self.nelec = cbm_vbm["p"]["bidx"]
        else:
            self.nelec = cbm_vbm["p"]["bidx"]*2

        bs = bs.as_dict()

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



    @staticmethod
    def f0(E, fermi, T):
        """returns the value of Fermi-Dirac at equilibrium for E (energy), fermi [level] and T (temperature)"""
        return 1 / (1 + np.exp((E - fermi) / (k_B * T)))


    #TODO: very inefficient code, see if you can change the way f is implemented
    def get_E_idx(self, E, tp):
        """tp (str): "n" or "p" type"""
        min_Ediff = 1e30
        for ie, en in enumerate(self.egrid[tp]["energy"]):
            if abs(E-en)< min_Ediff:
                min_Ediff = abs(E-en)
                ie_select = ie
        return ie_select


    def f(self, E, fermi, T, tp, c, alpha):
        """returns the perturbed Fermi-Dirac in presence of a small driving force"""
        return 1 / (1 + np.exp((E - fermi) / (k_B * T))) + self.egrid[tp]["g"][c][T][self.get_E_idx(E, tp)][alpha]



    @staticmethod
    def df0dE(E, fermi, T):
        return -1/(k_B*T) * np.exp((E - fermi) / (k_B * T)) / (1 + np.exp((E - fermi) / (k_B * T))) ** 2



    def seeb_int_num(self, c, T):
        """wrapper function to do an integration taking only the concentration, c, and the temperature, T, as inputs"""
        fn = lambda E, fermi, T: self.f0(E, fermi, T) * (1 - self.f0(E, fermi, T)) * E / (k_B * T)
        return {t:self.integrate_over_DOSxE_dE(func=fn,tp=t,fermi=self.egrid["fermi"][c][T],T=T) for t in ["n", "p"]}



    def seeb_int_denom(self, c, T):
        """wrapper function to do an integration taking only the concentration, c, and the temperature, T, as inputs"""
        fn = lambda E, fermi, T: self.f0(E, fermi, T) * (1 - self.f0(E, fermi, T))
        # To avoid returning a denominator that is zero:
        return {t:max(self.integrate_over_DOSxE_dE(func=fn,tp=t,fermi=self.egrid["fermi"][c][T],T=T), 1e-30)
                for t in ["n", "p"]}



    def calculate_property(self, prop_name, prop_func, for_all_E=False):
        """
        calculate the propery at all concentrations and Ts using the given function and insert it into self.egrid
        :param prop_name:
        :param prop_func (obj): the given function MUST takes c and T as required inputs in this order.
        :return:
        """
        if for_all_E:
            for tp in ["n", "p"]:
                self.egrid[tp][prop_name] = {c: {T: [0.0 for E in self.egrid[tp]["energy"]] for T in self.temperatures}
                                             for c in self.dopings}
        else:
            self.egrid[prop_name] = {c: {T: 0.0 for T in self.temperatures} for c in self.dopings}
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

        # reshape energies of all bands to one vector:
        for tp in ["n", "p"]:
            for en_vec in self.kgrid[tp]["energy"]:
                self.egrid[tp]["all_en_flat"] += en_vec
            self.egrid[tp]["all_en_flat"].sort()

        # setting up energy grid and DOS:
        for tp in ["n", "p"]:
            i = 0
            last_is_counted = False
            while i<len(self.egrid[tp]["all_en_flat"])-1:
                sum = self.egrid[tp]["all_en_flat"][i]
                counter = 1.0
                while i<len(self.egrid[tp]["all_en_flat"])-1 and \
                        abs(self.egrid[tp]["all_en_flat"][i]-self.egrid[tp]["all_en_flat"][i+1]) < self.dE_global:
                    counter += 1
                    sum += self.egrid[tp]["all_en_flat"][i+1]
                    if i+1 == len(self.egrid[tp]["all_en_flat"])-1:
                        last_is_counted = True
                    i+=1
                self.egrid[tp]["energy"].append(sum/counter)
                if dos_tp=="simple":
                    self.egrid[tp]["DOS"].append(counter/len(self.egrid[tp]["all_en_flat"]))
                i+=1
            if not last_is_counted:
                self.egrid[tp]["energy"].append(self.egrid[tp]["all_en_flat"][-1])
                if dos_tp == "simple":
                    self.egrid[tp]["DOS"].append(1.0 / len(self.egrid[tp]["all_en_flat"]))

        # initialize some fileds/properties
        self.egrid["calc_doping"] = {c: {T: {"n": 0.0, "p": 0.0} for T in self.temperatures} for c in self.dopings}
        for sn in self.elastic_scattering_mechanisms + ["POP", "overall", "average"]:
            # self.egrid["mobility"+"_"+sn]={c:{T:{"n": 0.0, "p": 0.0} for T in self.temperatures} for c in self.dopings}
            self.egrid["mobility"][sn] = {c: {T: {"n": 0.0, "p": 0.0} for T in self.temperatures} for c in self.dopings}
        for transport in ["conductivity", "J_th", "seebeck", "TE_power_factor"]:
            self.egrid[transport] = {c: {T: {"n": 0.0, "p": 0.0} for T in self.temperatures} for c in self.dopings}

        # populate the egrid at all c and T with properties; they can be called via self.egrid[prop_name][c][T] later
        self.calculate_property(prop_name="fermi", prop_func=self.find_fermi)
        self.calculate_property(prop_name="f0", prop_func=self.f0, for_all_E=True)
        self.calculate_property(prop_name="f", prop_func=self.f0, for_all_E=True)
        self.calculate_property(prop_name="f0x1-f0", prop_func=lambda E, fermi, T: self.f0(E, fermi, T)
                                                                        * (1 - self.f0(E, fermi, T)), for_all_E=True)
        self.calculate_property(prop_name="beta", prop_func=self.inverse_screening_length)
        self.calculate_property(prop_name="N_II", prop_func=self.calculate_N_II)
        self.calculate_property(prop_name="Seebeck_integral_numerator", prop_func=self.seeb_int_num)
        self.calculate_property(prop_name="Seebeck_integral_denominator", prop_func=self.seeb_int_denom)



    def G(self, tp, ib, ik, ib_prime, ik_prime, X):
        """
        The overlap integral betweek vectors k and k'
        :param ik (int): index of vector k in kgrid
        :param ik_prime (int): index of vector k' in kgrid
        :param X (float): cosine of the angle between vectors k and k'
        :return: overlap integral
        """
        return self.kgrid[tp]["a"][ib][ik] * self.kgrid[tp]["a"][ib_prime][ik_prime]+ \
               self.kgrid[tp]["c"][ib][ik] * self.kgrid[tp]["c"][ib_prime][ik_prime]



    def cos_angle(self, v1, v2):
        """
        Args:
            v1, v2 (np.array): vectors
        return:
            the cosine of the angle between twp numpy vectors: v1 and v2"""
        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 1.0  # In case of the two points are the origin, we assume 0 degree; i.e. no scattering: 1-X==0
        else:
            return np.dot(v1, v2) / (norm_v1 * norm_v2)



    def omit_kpoints(self, low_v_ik):
        """
        The k-points with velocity < 1 cm/s (either in valence or conduction band) are taken out as those are
            troublesome later with extreme values (e.g. too high elastic scattering rates)
        :param low_v_ik:
        :return:
        """

        print low_v_ik

        ik_list = list(set(low_v_ik))
        ik_list.sort(reverse=True)


        self.kgrid["kpoints"] = np.delete(self.kgrid["kpoints"], ik_list, axis=0)
        # self.kgrid["kpoints"].pop(ik)
        for i, tp in enumerate(["n", "p"]):
            for ib in range(self.cbm_vbm[tp]["included"]):
                # for ik in ik_list:
                for prop in ["energy", "a", "c"]:
                    for ik in ik_list:
                        self.kgrid[tp][prop][ib].pop(ik)
                for prop in ["velocity", "effective mass"]:
                    self.kgrid[tp][prop] = np.delete(self.kgrid[tp][prop], ik_list, axis=1)



    def init_kgrid(self,coeff_file, kgrid_tp="coarse"):
        if kgrid_tp=="coarse":
            nkstep = 4
        # k = list(np.linspace(0.25, 0.75-0.5/nstep, nstep))
        kx = list(np.linspace(-0.5, 0.5, nkstep))
        ky = kz = kx
        # ky = list(np.linspace(0.27, 0.67, nkstep))
        # kz = list(np.linspace(0.21, 0.71, nkstep))
        kpts = np.array([[x, y, z] for x in kx for y in ky for z in kz])
        print len(kpts)

        # TODO this deletion is just a test, change it later once confirmed that the order of mobility is good!
        # kpts = np.delete(kpts, (0, 21, 42, -1), axis=0)
        # kpts = np.delete(kpts, (-1), axis=0)

        # # Total range around the center k-point
        # rang = 0.14
        # if kgrid_tp == "coarse":
        #     nstep = 2
        #
        # step = rang/nstep
        #
        # kpts = [[0, 0, 0] for i in range((nstep+1)**3)]
        # counter = 0
        # for i in range(nstep+1):
        #     for j in range(nstep+1):
        #         for k in range(nstep+1):
        #             kpts[counter] = [center_kpt[0] - rang / 2.0 + 0.02*(-1)**j  + i * step,
        #                              center_kpt[1] - rang / 2.0 + 0.02*(-1)**i  +j * step,
        #                              center_kpt[2] - rang / 2.0 + k * step]
        #             counter += 1

        # kpts = np.array(kpts)
        # initialize the kgrid
        self.kgrid = {
                "kpoints": kpts,
                "n": {},
                "p": {} }
        for tp in ["n", "p"]:
            for property in ["energy", "a", "c"]:
                self.kgrid[tp][property] = [ [0.0 for i in range(len(kpts))] for j in
                                                                                range(self.cbm_vbm[tp]["included"])]
            for property in ["velocity"]:
                self.kgrid[tp][property] = \
                np.array([ [[0.0, 0.0, 0.0] for i in range(len(kpts))] for j in range(self.cbm_vbm[tp]["included"])])
            self.kgrid[tp]["effective mass"] = \
                [ np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] for i in range(len(kpts))]) for j in
                                                                                range(self.cbm_vbm[tp]["included"])]
            # self.kgrid[tp]["effective mass"] = \
            #     np.array([[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] for i in range(len(kpts))] for j in
            #           range(self.cbm_vbm[tp]["included"])])

        low_v_ik = []
        analytical_bands = Analytical_bands(coeff_file=coeff_file)
        for i, tp in enumerate(["n", "p"]):
            sgn = (-1) ** i
            for ib in range(self.cbm_vbm[tp]["included"]):
                engre, latt_points, nwave, nsym, nsymop, symop, br_dir = \
                    analytical_bands.get_engre(iband=self.cbm_vbm[tp]["bidx"] + sgn * ib)
                for ik in range(len(self.kgrid["kpoints"])):
                    energy, de, dde = analytical_bands.get_energy(
                        self.kgrid["kpoints"][ik], engre, latt_points, nwave, nsym, nsymop, symop, br_dir)

                    self.kgrid[tp]["energy"][ib][ik] = energy * Ry_to_eV + sgn * self.scissor/2
                    self.kgrid[tp]["velocity"][ib][ik] = abs(
                        de / hbar * A_to_m * m_to_cm * Ry_to_eV)  # to get v in units of cm/s
                    # self.kgrid[tp]["velocity"][ib][ik] = de/hbar * A_to_m * m_to_cm * Ry_to_eV # to get v in units of cm/s
                    # TODO: what's the implication of negative group velocities? check later after scattering rates are calculated
                    # TODO: actually using abs() for group velocities mostly increase nu_II values at each energy
                    # TODO: should I have de*2*pi for the group velocity and dde*(2*pi)**2 for effective mass?
                    if self.kgrid[tp]["velocity"][ib][ik][0] < 1 or self.kgrid[tp]["velocity"][ib][ik][1] < 1 \
                            or self.kgrid[tp]["velocity"][ib][ik][2] < 1:
                        low_v_ik.append(ik)
                    self.kgrid[tp]["effective mass"][ib][ik] = hbar ** 2 / (
                        dde * 4 * pi ** 2) / m_e / A_to_m ** 2 * e * Ry_to_eV  # m_tensor: the last part is unit conversion
                    self.kgrid[tp]["a"][ib][ik] = 1.0
            # Match the CBM/VBM energy values to those obtained from the coefficients file rather than vasprun.xml
            self.cbm_vbm[tp]["energy"] = sgn * min(sgn * np.array(self.kgrid[tp]["energy"][0]))


        if len(low_v_ik) > 0:
            self.omit_kpoints(low_v_ik)

        if len(self.kgrid["kpoints"]) < 5:
            raise ValueError("VERY BAD k-mesh; please change the setting for k-mesh and try again!")


        for tp in ["n", "p"]:
            for prop in ["_all_elastic", "S_i", "S_i_th", "S_o", "S_o_th", "g", "g_th", "g_POP",
                         "df0dk", "df0dE", "electric force", "thermal force"]:
                self.kgrid[tp][prop] = {c: {T: np.array([[[1e-32, 1e-32, 1e-32] for i in range(len(self.kgrid["kpoints"]))]
                    for j in range(self.cbm_vbm[tp]["included"])]) for T in self.temperatures} for c in self.dopings}

        self.kgrid["actual kpoints"]=np.dot(np.array(self.kgrid["kpoints"]),self._lattice_matrix)*2*pi*1/A_to_nm #[1/nm]
        # TODO: change how W_POP is set, user set a number or a file that can be fitted and inserted to kgrid
        self.kgrid["W_POP"] = [self.W_POP for i in range(len(self.kgrid["kpoints"]))]



    def generate_angles_and_indexes_for_integration(self):
        # for each energy point, we want to store the ib and ik of those points with the same E, E士hbar*W_POP
        for tp in ["n", "p"]:
            for angle_index_for_integration in ["X_E_ik", "X_Eplus_ik", "X_Eminus_ik"]:
                self.kgrid[tp][angle_index_for_integration] = [ [ [] for i in range(len(self.kgrid["kpoints"])) ]
                                                                  for j in range(self.cbm_vbm[tp]["included"]) ]
        for tp in ["n", "p"]:
            for ik in range(len(self.kgrid["kpoints"])):
                for ib in range(len(self.kgrid[tp]["energy"])):
                    for ib_prime in range(len(self.kgrid[tp]["energy"])):
                        for ik_prime in range(len(self.kgrid["kpoints"])):
                            k = self.kgrid["actual kpoints"][ik]
                            X = self.cos_angle(k, self.kgrid["actual kpoints"][ik_prime])

                            if abs(self.kgrid[tp]["energy"][ib][ik] -
                                           self.kgrid[tp]["energy"][ib_prime][ik_prime]) < self.dE_global:
                                self.kgrid[tp]["X_E_ik"][ib][ik].append((X, ib_prime, ik_prime))
                            if abs( (self.kgrid[tp]["energy"][ib][ik] +  hbar * self.kgrid["W_POP"][ik] ) \
                                                 - self.kgrid[tp]["energy"][ib_prime][ik_prime]) < self.dE_global:
                                self.kgrid[tp]["X_Eplus_ik"][ib][ik].append((X, ib_prime, ik_prime))
                            if abs( (self.kgrid[tp]["energy"][ib][ik] -  hbar * self.kgrid["W_POP"][ik] ) \
                                                 - self.kgrid[tp]["energy"][ib_prime][ik_prime]) < self.dE_global:
                                self.kgrid[tp]["X_Eminus_ik"][ib][ik].append((X, ib_prime, ik_prime))

                    self.kgrid[tp]["X_E_ik"][ib][ik].sort()
                    self.kgrid[tp]["X_Eplus_ik"][ib][ik].sort()
                    self.kgrid[tp]["X_Eminus_ik"][ib][ik].sort()



    def s_el_eq(self, sname, tp, c, T, k, k_prime):
        """
        return the scattering rate at wave vector k at a certain concentration and temperature
        for a specific elastic scattering mechanisms determined by sname
        :param sname (string): abbreviation of the name of the elastic scatteirng mechanisms; options: IMP, ADE, PIE, DIS
        :param c:
        :param T:
        :param k:
        :param k_prime:
        :return:
        """
        norm_diff_k = np.linalg.norm(k - k_prime)
        if norm_diff_k == 0:
            warnings.warn("same k and k' vectors as input of the elastic scattering equation")
            return 0

        if sname in ["IMP"]: # ionized impurity scattering
            unit_conversion = 0.001 / e**2
            return unit_conversion * e ** 4 * self.egrid["N_II"][c][T] /\
                        (4.0 * pi**2 * self.epsilon_s ** 2 * epsilon_0 ** 2 * hbar)* norm_diff_k ** 2 \
                                    / ((norm_diff_k ** 2 + self.egrid["beta"][c][T][tp] ** 2) ** 2)
        elif sname in ["ACD"]: # acoustic deformation potential scattering
            unit_conversion = 1e18 * e
            return unit_conversion * k_B * T * self.E_D[tp]**2 / (4.0 * pi**2 * hbar * self.C_el)
        elif sname in ["PIE"]: # piezoelectric scattering
            unit_conversion = 1e9/e
            return unit_conversion * e**2 * k_B * T * self.P_PIE**2 \
                   /(norm_diff_k ** 2 * 4.0 * pi**2 * hbar * epsilon_0 * self.epsilon_s)
        else:
            raise ValueError("The elastic scattering name {} is not supported!".format(sname))



    def integrate_over_DOSxE_dE(self, func, tp, fermi, T, interpolation_nsteps=100):
        integral = 0.0
        for ie in range(len(self.egrid[tp]["energy"]) - 1):
            E = self.egrid[tp]["energy"][ie]
            dE = abs(self.egrid[tp]["energy"][ie + 1] - E) / interpolation_nsteps
            dS = (self.egrid[tp]["DOS"][ie + 1] - self.egrid[tp]["DOS"][ie]) / interpolation_nsteps
            for i in range(interpolation_nsteps):
                # TODO:The DOS used is too simplistic and wrong (e.g., calc_doping might hit a limit), try 2*[2pim_hk_BT/hbar**2]**1.5
                integral += dE * (self.egrid[tp]["DOS"][ie] + i * dS) * func(E + i * dE, fermi, T)
        return integral



    def integrate_over_E(self, prop_list, tp, c, T, xDOS=True, xvel=False, interpolation_nsteps=100):
        diff = [0.0 for prop in prop_list]
        integral = 0.0
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
                integral += multi
        return integral



    def integrate_over_X(self, tp, X_E_index, integrand, ib, ik, c, T, sname=None, g_suffix=""):
        sum = np.array([0.0, 0.0, 0.0])
        if len(X_E_index[ib][ik]) == 0:
            return sum
        for i in range(len(X_E_index[ib][ik]) - 1):
            DeltaX = X_E_index[ib][ik][i + 1][0] - \
                     X_E_index[ib][ik][i][0]
            if DeltaX == 0.0:
                continue
            for alpha in range(3):
                dum = 0
                for j in range(2):
                    # extract the indecies
                    X, ib_prime, ik_prime = X_E_index[ib][ik][i + j]
                    dum += integrand(tp, c, T, ib, ik, ib_prime, ik_prime, X, alpha, sname=sname, g_suffix=g_suffix)

                dum /= 2.0  # the average of points i and i+1 to integrate via the trapezoidal rule
                sum[alpha] += dum * DeltaX  # In case of two points with the same X, DeltaX==0 so no duplicates
        return sum



    def el_integrand_X(self, tp, c, T, ib, ik, ib_prime, ik_prime, X, alpha, sname=None, g_suffix=""):
        k = self.kgrid["actual kpoints"][ik]
        k_prime = self.kgrid["actual kpoints"][ik_prime]
        return (1 - X) * self.s_el_eq(sname, tp, c, T, k, k_prime) \
               * self.G(tp, ib, ik, ib_prime, ik_prime, X) ** 2 \
               / self.kgrid[tp]["velocity"][ib_prime][ik_prime][alpha]
                # / abs(self.kgrid[tp]["velocity"][ib_prime][ik_prime][alpha])
        # We take |v| as scattering depends on the velocity itself and not the direction



    def inel_integrand_X(self, tp, c, T, ib, ik, ib_prime, ik_prime, X, alpha, sname=None, g_suffix=""):
        """
        returns the evaluated number (float) of the expression inside the S_o and S_i(g) integrals.
        :param tp (str): "n" or "p" type
        :param c (float): carrier concentration/doping in cm**-3
        :param T:
        :param ib:
        :param ik:
        :param ib_prime:
        :param ik_prime:
        :param X:
        :param alpha:
        :param sname:
        :return:
        """
        k = self.kgrid["actual kpoints"][ik]
        k_prime = self.kgrid["actual kpoints"][ik_prime]
        v = self.kgrid[tp]["velocity"][ib][ik]
        fermi = self.egrid["fermi"][c][T]

        f = self.f0(self.kgrid[tp]["energy"][ib][ik], fermi, T)
        f_prime = self.f0(self.kgrid[tp]["energy"][ib_prime][ik_prime], fermi, T)
        # test
        # f = self.f(self.kgrid[tp]["energy"][ib][ik], fermi, T, tp, c, alpha)
        # f_prime = self.f(self.kgrid[tp]["energy"][ib_prime][ik_prime], fermi, T, tp, c, alpha)

        N_POP = 1 / ( np.exp(hbar*self.kgrid["W_POP"][ik]/(k_B*T)) - 1 )
        # norm_diff = max(np.linalg.norm(k-k_prime), 1e-10)
        norm_diff = np.linalg.norm(k-k_prime)
        # print np.linalg.norm(k_prime)**2
        # the term np.linalg.norm(k_prime)**2 is wrong in practice as it can be too big and originally we integrate |k'| from 0
        # integ = np.linalg.norm(k_prime)**2*self.G(tp, ib, ik, ib_prime, ik_prime, X)/(v[alpha]*norm_diff**2)
        integ = self.G(tp, ib, ik, ib_prime, ik_prime, X)/(v[alpha])
        if "S_i" in sname:
            integ *= abs(X*self.kgrid[tp]["g" + g_suffix][c][T][ib][ik][alpha])
            # integ *= X*self.kgrid[tp]["g" + g_suffix][c][T][ib][ik][alpha]
            if "minus" in sname:
                integ *= (1-f)*N_POP + f*(1+N_POP)
            elif "plus" in sname:
                integ *= (1-f)*(1+N_POP) + f*N_POP
            else:
                raise ValueError('"plus" or "minus" must be in sname for phonon absorption and emission respectively')
        elif "S_o" in sname:
            if "minus" in sname:
                integ *= (1-f_prime)*(1+N_POP) + f_prime*N_POP
            elif "plus" in sname:
                integ *= (1-f_prime)*N_POP + f_prime*(1+N_POP)
            else:
                raise ValueError('"plus" or "minus" must be in sname for phonon absorption and emission respectively')
        else:
            raise ValueError("The inelastic scattering name: {} is NOT supported".format(sname))
        assert(type(integ), float)
        # assert(integ>=0)
        return integ



    def s_inelastic(self, sname = None, g_suffix=""):
        for tp in ["n", "p"]:
            for c in self.dopings:
                for T in self.temperatures:
                    for ik in range(len(self.kgrid["kpoints"])):
                        for ib in range(len(self.kgrid[tp]["energy"])):
                            sum = 0
                            for X_E_index_name in ["X_Eplus_ik", "X_Eminus_ik"]:
                                sum += self.integrate_over_X(tp, self.kgrid[tp][X_E_index_name], self.inel_integrand_X,
                                                ib=ib, ik=ik, c=c, T=T, sname=sname+X_E_index_name, g_suffix=g_suffix)
                            # self.kgrid[tp][sname][c][T][ib][ik] = abs(sum) * e**2*self.kgrid["W_POP"][ik]/(4*pi*hbar) \
                            self.kgrid[tp][sname][c][T][ib][ik] = sum*e**2*self.kgrid["W_POP"][ik] / (4 * pi * hbar) \
                                                            * (1/self.epsilon_inf-1/self.epsilon_s)/epsilon_0 * 100/e
                            # if np.linalg.norm(self.kgrid[tp][sname][c][T][ib][ik]) > 1e5:
                            #     print tp, c, T, ik, ib, sum, self.kgrid[tp][sname][c][T][ib][ik]



    def s_elastic(self, sname="IMP"):
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
            self.kgrid[tp][sname] = {c: {T: np.array([[[0.0, 0.0, 0.0] for i in range(len(self.kgrid["kpoints"]))]
                    for j in range(self.cbm_vbm[tp]["included"])]) for T in self.temperatures} for c in self.dopings}
            for c in self.dopings:
                for T in self.temperatures:
                    for ik in range(len(self.kgrid["kpoints"])):
                        for ib in range(len(self.kgrid[tp]["energy"])):
                            sum = self.integrate_over_X(tp, X_E_index=self.kgrid[tp]["X_E_ik"],
                                                        integrand=self.el_integrand_X,
                                                      ib=ib, ik=ik, c=c, T=T, sname = sname, g_suffix="")
                            self.kgrid[tp][sname][c][T][ib][ik] = abs(sum) *2e-7*pi/hbar
                            for alpha in range(3):
                                if self.kgrid[tp][sname][c][T][ib][ik][alpha] < 1:
                                    self.kgrid[tp][sname][c][T][ib][ik][alpha] = 1e9
                            self.kgrid[tp]["_all_elastic"][c][T][ib][ik] += self.kgrid[tp][sname][c][T][ib][ik]



    def map_to_egrid(self, prop_name, c_and_T_idx=True):
        """
        maps a propery from kgrid to egrid conserving the nomenclature. The mapped property should have the
            kgrid[tp][prop_name][c][T][ib][ik] data structure and will have egrid[tp][prop_name][c][T][ie] structure
        :param prop_name (string): the name of the property to be mapped. It must be available in the kgrid.
        :return:
        """
        scalar_properties = ["g"]
        if not c_and_T_idx:
            for tp in ["n", "p"]:
                try:
                    self.egrid[tp][prop_name]
                except:
                    # if prop_name in scalar_properties:
                    #     self.egrid[tp][prop_name] = np.array([1e-20 for i in range(len(self.egrid[tp]["energy"]))])
                    # else:
                        self.egrid[tp][prop_name] = np.array([[1e-20, 1e-20, 1e-20] \
                            for i in range(len(self.egrid[tp]["energy"]))])
                for ie, en in enumerate(self.egrid[tp]["energy"]):
                    N = 0  # total number of instances with the same energy
                    for ik in range(len(self.kgrid["kpoints"])):
                        for ib in range(len(self.kgrid[tp]["energy"])):
                            if abs(self.kgrid[tp]["energy"][ib][ik] - en) < self.dE_global:
                                if prop_name in scalar_properties:
                                    self.egrid[tp][prop_name][ie] += np.linalg.norm(self.kgrid[tp][prop_name][ib][ik])
                                else:
                                    self.egrid[tp][prop_name][ie] += self.kgrid[tp][prop_name][ib][ik]
                            N += 1
                    self.egrid[tp][prop_name][ie] /= N
        else:
            for tp in ["n", "p"]:
                try:
                    self.egrid[tp][prop_name]
                except:
                    self.egrid[tp][prop_name] = {c: {T: np.array([[1e-20, 1e-20, 1e-20]
                        for i in range(len(self.egrid[tp]["energy"]))]) for T in self.temperatures}
                                                                                                for c in self.dopings}
                for c in self.dopings:
                    for T in self.temperatures:
                        for ie, en in enumerate(self.egrid[tp]["energy"]):
                            N = 0 # total number of instances with the same energy
                            for ik in range(len(self.kgrid["kpoints"])):
                                for ib in range(len(self.kgrid[tp]["energy"])):
                                    if abs(self.kgrid[tp]["energy"][ib][ik] - en) < self.dE_global:
                                        self.egrid[tp][prop_name][c][T][ie] += self.kgrid[tp][prop_name][c][T][ib][ik]
                                        N += 1
                            self.egrid[tp][prop_name][c][T][ie] /= N



    def find_fermi(self, c, T, tolerance=0.001, tolerance_loose=0.03,
                   interpolation_nsteps = 100 , step0 = 0.01, nsteps = 300):
        """
        To find the Fermi level at a carrier concentration and temperature at kgrid (i.e. band structure, DOS, etc)
        :param c (float): The doping concentration; c < 0 indicate n-tp (i.e. electrons) and c > 0 for p-tp
        :param T (float): The temperature.
        :param interpolation_nsteps (int): the number of steps with which the energy points are
                    linearly interpolated for smoother numerical integration
        :param maxitr (int): Number of trials to fit the Fermi level, higher maxitr result in more accurate Fermi
        :param step0 (float): The initial step of changing Fermi level (in eV)
        :param nsteps (int): The number of steps are looked lower and higher than initial guess for Fermi level
        :param tolerance (0<float<1): convergance threshold for relative error
        :param tolerance_loose (0<float<1): maximum relative error allowed between the calculated and input c
        :return:
            The fitted/calculated Fermi level
        """
        # initialize parameters
        calc_doping = 1e52 # initial concentration, just has to be very far from any expected concentration
        relative_error = calc_doping
        nfloat = 4 # essentially the number of floating points in accuracy
        iter = 0
        actual_tp = self.get_tp(c)
        temp_doping = {"n": 0.0, "p": 0.0}
        fermi0 = self.cbm_vbm[actual_tp]["energy"]
        fermi_selected = fermi0

        # iterate around the CBM/VBM with finer and finer steps to find the Fermi level with a matching doping
        # for iter in range(maxitr):
        funcs = [lambda E, fermi, T: self.f0(E,fermi,T), lambda E, fermi, T: 1-self.f0(E,fermi,T)]
        while (relative_error > tolerance) and (iter<nfloat):
            step = step0 / 10**iter
            for fermi in np.linspace(fermi0-nsteps*step,fermi0+nsteps*step, nsteps*2):
                for j, tp in enumerate(["n", "p"]):
                    # func = lambda E, fermi, T: j-(-1)**(j+1)*self.f0(E,fermi,T)
                    integral = self.integrate_over_DOSxE_dE(func=funcs[j], tp=tp, fermi=fermi, T=T)
                    temp_doping[tp] = (-1)**(j+1) * abs(integral*self.nelec/self.volume / (A_to_m*m_to_cm)**3)
                if abs(temp_doping["n"] + temp_doping["p"] - c) < abs(calc_doping - c):
                    calc_doping = temp_doping["n"] + temp_doping["p"]
                    fermi_selected = fermi
                    self.egrid["calc_doping"][c][T]["n"] = temp_doping["n"]
                    self.egrid["calc_doping"][c][T]["p"] = temp_doping["p"]
            fermi0 = fermi_selected
            iter += 1
        # evaluate the calculated carrier concentration (Fermi level)
        relative_error = abs(calc_doping - c)/abs(c)
        if relative_error > tolerance and relative_error <= tolerance_loose:
            warnings.warn("The calculated concentration {} is not accurate compared to {}; results may be unreliable"
                          .format(calc_doping, c))
        elif relative_error > tolerance_loose:
            raise ValueError("The calculated concentration {} is more than {}% away from {}; "
                             "possible cause may low band gap, high temperature, small nsteps, etc; AMSET stops now!"
                             .format(calc_doping, tolerance_loose*100, c))
        return fermi_selected



    def inverse_screening_length(self, c, T, interpolation_nsteps = 100):
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



    def to_json(self, kgrid=True, trimmed=False):
        if trimmed:
            remove_list = []
            for tp in ["n", "p"]:
                for rm in remove_list:
                    try:
                        del (self.egrid[tp][rm])
                    except:
                        pass
        with open("egrid.json", 'w') as fp:
            json.dump(self.egrid, fp, sort_keys=True, indent=4, ensure_ascii=False, cls=MontyEncoder)

        if kgrid:
            if trimmed:
                remove_list = ["W_POP"]
                for rm in remove_list:
                    del (self.kgrid[rm])
                remove_list = ["effective mass", "actual kpoints", "X_E_ik", "X_Eplus_ik", "X_Eminus_ik"]
                for tp in ["n", "p"]:
                    for rm in remove_list:
                        try:
                            del (self.kgrid[tp][rm])
                        except:
                            pass
            with open("kgrid.json", 'w') as fp:
                json.dump(self.kgrid, fp,sort_keys = True, indent = 4, ensure_ascii=False, cls=MontyEncoder)



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
            self.init_egrid(dos_tp = "simple")
        else:
            pass

        # initialize g in the egrid
        self.map_to_egrid("g")

        # find the indexes of equal energy or those with ±hbar*W_POP for scattering via phonon emission and absorption
        self.generate_angles_and_indexes_for_integration()

        # calculate all elastic scattering rates in kgrid and then map it to egrid:
        for sname in self.elastic_scattering_mechanisms:
            self.s_elastic(sname=sname)
            self.map_to_egrid(prop_name=sname)


        self.map_to_egrid(prop_name="_all_elastic")

        for tp in ["n", "p"]:
            for c in self.dopings:
                for T in self.temperatures:
                    fermi = self.egrid["fermi"][c][T]
                    for ik in range(len(self.kgrid["kpoints"])):
                        for ib in range(len(self.kgrid[tp]["energy"])):
                            E = self.kgrid[tp]["energy"][ib][ik]
                            v = self.kgrid[tp]["velocity"][ib][ik]
                            self.kgrid[tp]["df0dk"][c][T][ib][ik] = hbar * self.df0dE(E,fermi, T) * v # in cm
                            self.kgrid[tp]["electric force"][c][T][ib][ik] = -1 * \
                                self.kgrid[tp]["df0dk"][c][T][ib][ik] * default_small_E / hbar # in 1/s
                            # self.kgrid[tp]["electric force"][c][T][ib][ik] = 1
                            self.kgrid[tp]["thermal force"][c][T][ib][ik] = - v * \
                                self.f0(E, fermi, T) * (1 - self.f0(E, fermi, T)) * (
                                    E/(k_B*T)-self.egrid["Seebeck_integral_numerator"][c][T][tp]/
                                    self.egrid["Seebeck_integral_denominator"][c][T][tp] ) * dTdz/T



                            # self.kgrid[tp]["thermal force"][c][T][ib][ik] = v * df0dz * unit_conversion
                            # df0dz_temp = self.f0(E, fermi, T) * (1 - self.f0(E, fermi, T)) * (
                                # E / (k_B * T) - df0dz_integral) * 1 / T * dTdz
        self.map_to_egrid(prop_name="df0dk") # This mapping is not correct as df0dk(E) is meaningless

        # calculating S_o scattering rate which is not a function of g
        self.s_inelastic(sname="S_o")
        self.s_inelastic(sname="S_o_th")

        # solve BTE to calculate S_i scattering rate and perturbation (g) in an iterative manner
        for iter in range(self.maxiters):
            for g_suffix in ["", "_th"]:
                # self.s_inelastic(sname="S_o" + g_suffix, g_suffix=g_suffix)
                self.s_inelastic(sname="S_i" + g_suffix, g_suffix=g_suffix)
                # for tp in ["n", "p"]:
                #     self.kgrid[tp]["S_i" + g_suffix] = {c: {T: np.array([[[1., 1., 1.] for i in range(len(self.kgrid["kpoints"]))]
                #         for j in range(self.cbm_vbm[tp]["included"])]) for T in self.temperatures} for c in self.dopings}
                #     self.kgrid[tp]["S_o" + g_suffix] = {
                #     c: {T: np.array([[[1e8, 1e8, 1e8] for i in range(len(self.kgrid["kpoints"]))]
                #                      for j in range(self.cbm_vbm[tp]["included"])]) for T in self.temperatures} for c in
                #     self.dopings}
            for c in self.dopings:
                for T in self.temperatures:
                    for tp in ["n", "p"]:
                        self.kgrid[tp]["g"][c][T]=(self.kgrid[tp]["S_i"][c][T]+self.kgrid[tp]["electric force"][c][T])/(
                            self.kgrid[tp]["S_o"][c][T] + self.kgrid[tp]["_all_elastic"][c][T])
                        self.kgrid[tp]["g_POP"][c][T] = (self.kgrid[tp]["S_i"][c][T] +
                            self.kgrid[tp]["electric force"][c][T]) / (self.kgrid[tp]["S_o"][c][T]+ 1e-32 )
                        self.kgrid[tp]["g_th"][c][T]=(self.kgrid[tp]["S_i_th"][c][T]+self.kgrid[tp]["thermal force"][c][
                            T]) / (self.kgrid[tp]["S_o_th"][c][T] + self.kgrid[tp]["_all_elastic"][c][T])

            for prop in ["electric force", "thermal force", "g", "g_POP", "g_th", "S_i", "S_o", "S_i_th", "S_o_th"]:
                self.map_to_egrid(prop_name=prop, c_and_T_idx=True)

        self.map_to_egrid(prop_name="velocity", c_and_T_idx=False)

        for c in self.dopings:
            for T in self.temperatures:
                fermi = self.egrid["fermi"][c][T]
                for tp in ["n", "p"]:
                    self.egrid[tp]["f"][c][T] += np.linalg.norm(self.egrid[tp]["g"][c][T])
                    # self.egrid["mobility"][c][T][tp]=self.integrate_over_DOSxE_dE(self.mobility_integrand,tp,fermi,T)
                    # mobility numerators
                    for nu in self.elastic_scattering_mechanisms :
                        self.egrid["mobility"][nu][c][T][tp] = (-1)*default_small_E/hbar* \
                            self.integrate_over_E(prop_list=["/"+nu, "df0dk"], tp=tp, c=c, T=T, xDOS=True, xvel=True)
                    self.egrid["mobility"]["POP"][c][T][tp] = self.integrate_over_E(prop_list=["g_POP"],
                                                                                    tp=tp,c=c,T=T,xDOS=True,xvel=True)
                    self.egrid["mobility"]["overall"][c][T][tp]=self.integrate_over_E(prop_list=["g"],
                                                                                      tp=tp,c=c,T=T,xDOS=True,xvel=True)
                    self.egrid["J_th"][c][T][tp] = default_small_E * self.integrate_over_E(prop_list=["g_th"],
                                                                                tp=tp, c=c, T=T, xDOS=True,xvel=True)

                    # mobility denominators
                    for transport in self.elastic_scattering_mechanisms + ["POP", "overall"]:
                        self.egrid["mobility"][transport][c][T][tp]/=default_small_E*\
                                        self.integrate_over_E(prop_list=["f0"],tp=tp, c=c, T=T, xDOS=True, xvel=False)
                    self.egrid["J_th"][c][T][tp] /= default_small_E * \
                                                                   self.integrate_over_E(prop_list=["f0"], tp=tp, c=c,
                                                                                         T=T, xDOS=True, xvel=False)
                    # for transport in self.elastic_scattering_mechanisms + ["POP", "overall"]:
                    #     self.egrid["mobility"][transport][c][T][tp]/=default_small_E*\
                    #                     self.integrate_over_E(prop_list=["f"],tp=tp, c=c, T=T, xDOS=True, xvel=False)
                    # self.egrid["J_th"][c][T][tp] /= default_small_E * \
                    #                                                self.integrate_over_E(prop_list=["f"], tp=tp, c=c,
                    #                                                                      T=T, xDOS=True, xvel=False)

                    for transport in self.elastic_scattering_mechanisms + ["POP"]:
                        self.egrid["mobility"]["average"][c][T][tp] += 1 / self.egrid["mobility"][transport][c][T][tp]
                    self.egrid["mobility"]["average"][c][T][tp] = 1/ self.egrid["mobility"]["average"][c][T][tp]

                    # calculating other overall transport properties:
                    self.egrid["conductivity"][c][T][tp] = self.egrid["mobility"]["overall"][c][T][tp]* e * abs(c)
                    self.egrid["seebeck"][c][T][tp] = 1e6 * (k_B/e * ( self.egrid["Seebeck_integral_numerator"][c][T][tp] \
                        / self.egrid["Seebeck_integral_numerator"][c][T][tp] - self.egrid["fermi"][c][T]/(k_B*T) ) \
                        - self.egrid["J_th"][c][T][tp]/self.egrid["conductivity"][c][T][tp]/dTdz )
                    print "3 seebeck terms at c={} and T={}:".format(c, T)
                    print self.egrid["Seebeck_integral_numerator"][c][T][tp] \
                        / self.egrid["Seebeck_integral_numerator"][c][T][tp] * 1e6
                    print - self.egrid["fermi"][c][T]/(k_B*T) * 1e6
                    print - self.egrid["J_th"][c][T][tp]/self.egrid["conductivity"][c][T][tp]/dTdz*1e6
                    # thermopower_n = -k_B*(df0dz_integral_n-efef_n/(k_B*T))*1e6+(J(k_grid,T,m,g_th,Ds_n,energy_n,volume,v_n,free_e)/sigma)/dTdz*1e6;

        # remove_list = ["actual kpoints"]
        # for rm in remove_list:
        #     del (self.kgrid[rm])
        # remove_list = ["effective mass"]
        # for tp in ["n", "p"]:
        #     for rm in remove_list:
        #         del (self.kgrid[tp][rm])

        if self.wordy:
            pprint(self.egrid)
            pprint(self.kgrid)

        with open("kgrid.txt", "w") as fout:
            pprint(self.kgrid, stream=fout)
        with open("egrid.txt", "w") as fout:
            pprint(self.egrid, stream=fout)



if __name__ == "__main__":
    coeff_file = 'fort.123'

    # test
    AMSET = AMSET()
    # AMSET.run(coeff_file=coeff_file, kgrid_tp="coarse")
    cProfile.run('AMSET.run(coeff_file=coeff_file, kgrid_tp="coarse")')

    AMSET.to_json(trimmed=True)
