# coding: utf-8

import warnings

from pymatgen.electronic_structure.plotter import plot_brillouin_zone, plot_brillouin_zone_from_kpath
from pymatgen.symmetry.bandstructure import HighSymmKpath

from analytical_band_from_BZT import Analytical_bands
from pprint import pprint

import numpy as np
from math import log
import sys
from pymatgen.io.vasp import Vasprun, Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, generate_full_symmops
from scipy.constants.codata import value as _cd
from math import pi
import os
import json
from monty.json import MontyEncoder
from random import random

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

                 N_dis=None, scissor=None, elastic_scatterings=None, include_POP=True, bs_is_isotropic=True,
                 donor_charge=None, acceptor_charge=None, dislocations_charge=None):
        self.dE_global = k_B*300 # in eV, the energy difference threshold below which two energy values are assumed equal
        self.dopings = [-1e19] # 1/cm**3 list of carrier concentrations
        self.temperatures = map(float, [300, 600]) # in K, list of temperatures
        self.epsilon_s = 44.360563 # example for PbTe
        self.epsilon_inf = 25.57 # example for PbTe
        self._vrun = {}
        self.max_e_range = 10*k_B*max(self.temperatures) # we set the max energy range after which occupation is zero
        self.path_dir = path_dir or "../test_files/PbTe_nscf_uniform/nscf_line"
        self.charge = {"n": donor_charge or 1, "p": acceptor_charge or 1, "dislocations": dislocations_charge or 1}
        self.N_dis = N_dis or 0.1 # in 1/cm**2
        self.elastic_scatterings = elastic_scatterings or ["IMP", "ACD", "PIE", "DIS"]
        self.inelastic_scatterings = []
        if include_POP:
            self.inelastic_scatterings += ["POP"]
        self.scissor = scissor or 0.0 # total value added to the band gap by adding to the CBM and subtracting from VBM
        self.bs_is_isotropic = bs_is_isotropic
        self.ds = 1e-32 # a global small/initial value
        self.dl = 1e32 # a global large/initial value

#TODO: some of the current global constants should be omitted, taken as functions inputs or changed!

        self.wordy = False
        self.maxiters = 6
        self.soc = False
        self.read_vrun(path_dir=self.path_dir, filename="vasprun.xml")
        self.W_POP = 10e12 * 2*pi # POP frequency in Hz
        self.P_PIE = 0.15
        self.E_D = {"n": 4.0, "p": 3.93}
        self.C_el = 128.84 #77.3 # [Gpa]:spherically averaged elastic constant for longitudinal modes
        self.nforced_POP = 0



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
        bs = self._vrun.get_band_structure()

        # Remember that python band index starts from 0 so bidx==9 refers to the 10th band (e.g. in VASP)
        cbm_vbm = {"n": {"kpoint": [], "energy": 0.0, "bidx": 0, "included": 0, "eff_mass_xx": [0.0, 0.0, 0.0]},
                   "p": {"kpoint": [], "energy": 0.0, "bidx": 0, "included": 0, "eff_mass_xx": [0.0, 0.0, 0.0]}}
        cbm = bs.get_cbm()
        vbm = bs.get_vbm()

        cbm_vbm["n"]["energy"] = cbm["energy"]
        cbm_vbm["n"]["bidx"] = cbm["band_index"][Spin.up][0]
        cbm_vbm["n"]["kpoint"] = bs.kpoints[cbm["kpoint_index"][0]].frac_coords

        cbm_vbm["p"]["energy"] = vbm["energy"]
        cbm_vbm["p"]["bidx"] = vbm["band_index"][Spin.up][-1]
        cbm_vbm["p"]["kpoint"] = bs.kpoints[vbm["kpoint_index"][0]].frac_coords

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
        return {t:self.ds + self.integrate_over_E(prop_list=["f0x1-f0"],tp=t,c=c,T=T,xDOS=True) for t in ["n", "p"]}



    def calculate_property(self, prop_name, prop_func, for_all_E=False):
        """
        calculate the propery at all concentrations and Ts using the given function and insert it into self.egrid
        :param prop_name:
        :param prop_func (obj): the given function MUST takes c and T as required inputs in this order.
        :return:
        """
        if for_all_E:
            for tp in ["n", "p"]:
                self.egrid[tp][prop_name]={c:{T: [self.ds for E in self.egrid[tp]["energy"]] for T in self.temperatures}
                                             for c in self.dopings}
        else:
            self.egrid[prop_name] = {c: {T: self.ds for T in self.temperatures} for c in self.dopings}
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
                self.egrid[tp]["all_en_flat"] += list(en_vec)
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
        for sn in self.elastic_scatterings + self.inelastic_scatterings +["overall", "average"]:
            # self.egrid["mobility"+"_"+sn]={c:{T:{"n": 0.0, "p": 0.0} for T in self.temperatures} for c in self.dopings}
            self.egrid["mobility"][sn] = {c: {T: {"n": 0.0, "p": 0.0} for T in self.temperatures} for c in self.dopings}
        for transport in ["conductivity", "J_th", "seebeck", "TE_power_factor", "relaxation time constant"]:
            self.egrid[transport] = {c: {T: {"n": 0.0, "p": 0.0} for T in self.temperatures} for c in self.dopings}

        # populate the egrid at all c and T with properties; they can be called via self.egrid[prop_name][c][T] later
        self.calculate_property(prop_name="fermi", prop_func=self.find_fermi)
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

        # omit the points with v~0 and try to find the parabolic band equivalent effective mass at the CBM and the VBM
        temp_min = {"n": 1e32, "p": 1e32}
        # self.kgrid[tp]["kpoints"][ib] = np.delete(self.kgrid[tp]["kpoints"][ib], ik_list, axis=0)
        # self.kgrid[tp]["kpoints"][ib].pop(ik)
        for i, tp in enumerate(["n", "p"]):
            for ib in range(self.cbm_vbm[tp]["included"]):
                # for ik in ik_list:
                for ik in ik_list:
                    if (-1)**i * self.kgrid[tp]["energy"][ib][ik] < temp_min[tp]:
                        self.cbm_vbm[tp]["eff_mass_xx"]=(-1)**i*self.kgrid[tp]["effective mass"][ib][ik].diagonal()
                    for prop in ["energy", "a", "c", "kpoints", "kweights"]:
                        self.kgrid[tp][prop][ib].pop(ik)

                for prop in ["velocity", "effective mass"]:
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
                    if not c_T_idx:
                        self[grid][tp][name] = init_content
                    else:
                        self[grid][tp][name] = {c: {T: init_content for T in self.temperatures} for c in self.dopings}



    def init_kgrid(self,coeff_file, kgrid_tp="coarse"):
        if kgrid_tp=="coarse":
            nkstep = 4 #99 #32


        # # k = list(np.linspace(0.25, 0.75-0.5/nstep, nstep))
        # kx = list(np.linspace(-0.5, 0.5, nkstep))
        # ky = kz = kx
        # # ky = list(np.linspace(0.27, 0.67, nkstep))
        # # kz = list(np.linspace(0.21, 0.71, nkstep))
        # kpts = np.array([[x, y, z] for x in kx for y in ky for z in kz])

        sg = SpacegroupAnalyzer(self._vrun.final_structure)
        self.rotations, self.translations = sg._get_symmetry() # this returns unique symmetry operations

        kpts_and_weights = sg.get_ir_reciprocal_mesh(mesh=(nkstep, nkstep, nkstep), is_shift=(0.01, 0.01, 0.01))
        print len(kpts_and_weights)
        kpts = []
        for i in kpts_and_weights:
            k = i[0]
            kpts.append(k)
            fractional_ks = [np.dot(self.rotations[i], k) + self.translations[i] for i in range(len(self.rotations))]
            for k_seq in fractional_ks:
                if abs(k_seq[0]-k[0])<0.01 and (k_seq[1]-k[1])<0.01 and (k_seq[2]-k[2])<0.01:
                    continue
                else:
                    kpts.append(k_seq)
                    print k_seq

        # explicitly add the CBM/VBM k-points to calculate the parabolic band effective mass hence the relaxation time
        kpts.append(self.cbm_vbm["n"]["kpoint"])
        kpts.append(self.cbm_vbm["p"]["kpoint"])

        # remove further duplications due to copying a k-point twice because it is equivalent to two different k-points
        rm_list = []
        for i in range(len(kpts)-2):
            for j in range(i+1, len(kpts)-1):
                if np.array_equal(kpts[i], kpts[j]):
                    rm_list.append(j)
        kpts = np.delete(kpts, rm_list, axis=0)

        print len(kpts)
        # kweights = [float(i[1]) for i in kpts_and_weights]
        # kweights.append(0.0)
        # kweights.append(0.0)

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
        # directly on "actual kpoints" to get new cartesian (2pi*lattice_matrix*fractional-k) equivalent k-point


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

        self.initialize_var("kgrid", ["energy", "a", "c", "W_POP"], "scalar", 0.0, is_nparray=False, c_T_idx=False)
        self.initialize_var("kgrid", ["velocity", "actual kpoints"], "vector", 0.0, is_nparray=False, c_T_idx=False)
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


        low_v_ik = []
        low_v_ik.sort()
        analytical_bands = Analytical_bands(coeff_file=coeff_file)
        once_called = False
        bands_data = {tp: [() for ib in range(self.cbm_vbm[tp]["included"])] for tp in ["n", "p"]}
        for i, tp in enumerate(["n", "p"]):
            sgn = (-1) ** i
            for ib in range(self.cbm_vbm[tp]["included"]):
                engre, latt_points, nwave, nsym, nsymop, symop, br_dir = \
                    analytical_bands.get_engre(iband=self.cbm_vbm[tp]["bidx"] + sgn * ib)
                bands_data[tp][ib] = (engre, latt_points, nwave, nsym, nsymop, symop, br_dir)
                if not once_called:
                    nstv, vec, vec2 = analytical_bands.get_star_functions(latt_points, nsym, symop, nwave,br_dir=br_dir)
                    once_called = True
                for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                    energy, de, dde = analytical_bands.get_energy(
                        self.kgrid[tp]["kpoints"][ib][ik], engre, nwave, nsym, nstv, vec, vec2, br_dir=br_dir, cbm=True)

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

        if len(low_v_ik) > 0:
            self.omit_kpoints(low_v_ik)

        if len(self.kgrid["n"]["kpoints"][0]) < 5:
            raise ValueError("VERY BAD k-mesh; please change the setting for k-mesh and try again!")

        # sort "energy", "kpoints", "kweights", etc based on energy in ascending order
        self.sort_vars_based_on_energy(args=["kpoints", "kweights", "velocity", "effective mass", "a"], ascending=True)


        for tp in ["n", "p"]:
            for ib in range(self.cbm_vbm[tp]["included"]):
                self.kgrid[tp]["actual kpoints"][ib]=np.dot(np.array(self.kgrid[tp]["kpoints"][ib]),self._lattice_matrix)*1/A_to_nm*2*pi #[1/nm]
        # TODO: change how W_POP is set, user set a number or a file that can be fitted and inserted to kgrid
                self.kgrid[tp]["W_POP"][ib] = [self.W_POP for i in range(len(self.kgrid[tp]["kpoints"][ib]))]

        # Match the CBM/VBM energy values to those obtained from the coefficients file rather than vasprun.xml
        self.cbm_vbm["n"]["energy"] = self.kgrid["n"]["energy"][0][0]
        self.cbm_vbm["p"]["energy"] = self.kgrid["n"]["energy"][0][-1]

        self.initialize_var(grid="kgrid", names=["_all_elastic", "S_i", "S_i_th", "S_o", "S_o_th", "g", "g_th", "g_POP",
                "f", "f_th", "relaxation time", "df0dk", "electric force", "thermal force"],
                        val_type="vector", initval=self.ds, is_nparray=True, c_T_idx=True)
        self.initialize_var("kgrid", ["f0"], "scalar", self.ds, is_nparray=False, c_T_idx=True)



    def sort_vars_based_on_energy(self, args, ascending=True):
        """sort the list of variables specified by "args" (type: [str]) in self.kgrid based on the "energy" values
        in each band for both "n"- and "p"-type bands and in ascending order by default."""
        for tp in ["n", "p"]:
            for ib in range(self.cbm_vbm[tp]["included"]):
                ikidxs = np.argsort(self.kgrid[tp]["energy"][ib])
                if not ascending:
                    ikidxs.reverse()
                for arg in args + ["energy"]:
                    # if arg in ["k-points", "kweights"]:
                    #     self.kgrid[arg] = np.array([self.kgrid[arg][ik] for ik in ikidxs])
                    # else:
                    self.kgrid[tp][arg][ib] = np.array([self.kgrid[tp][arg][ib][ik] for ik in ikidxs])



    def generate_angles_and_indexes_for_integration(self):


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
        #                     k = self.kgrid[tp]["actual kpoints"][ib][ik]
                            # E = self.kgrid[tp]["energy"][ib][ik]
                            # X = self.cos_angle(self.kgrid[tp]["actual kpoints"][ib][ik], self.kgrid[tp]["actual kpoints"][ib][ik_prm])
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
            self.nforced_POP = 0
            for ib in range(len(self.kgrid[tp]["energy"])):
                for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                    self.kgrid[tp]["X_E_ik"][ib][ik] = self.get_X_ib_ik_within_E_radius(tp,ib,ik,
                                                    E_radius=0.0, forced_min_npoints=2, tolerance=self.dE_global)
                    self.kgrid[tp]["X_Eplus_ik"][ib][ik] = self.get_X_ib_ik_within_E_radius(tp,ib,ik,
                        E_radius= +hbar * self.kgrid[tp]["W_POP"][ib][ik], forced_min_npoints=2, tolerance=self.dE_global)
                    self.kgrid[tp]["X_Eminus_ik"][ib][ik] = self.get_X_ib_ik_within_E_radius(tp, ib, ik,
                        E_radius= -hbar * self.kgrid[tp]["W_POP"][ib][ik],forced_min_npoints=2, tolerance=self.dE_global)


                # if is_sparse(self.kgrid[tp]["X_E_ik"][ib]):
                #     raise ValueError("the k-grid is too coarse for an acceptable simulation of elastic scattering")
                # if is_sparse(self.kgrid[tp]["X_Eplus_ik"][ib]) or is_sparse(self.kgrid[tp]["X_Eminus_ik"][ib]):
                #     if "POP" in self.inelastic_scatterings:
                #         raise ValueError("the k-grid is too coarse for an acceptable simulation of POP scattering, "
                        # warnings.warn("the k-grid is too coarse for an acceptable simulation of POP scattering, "
                        #                  "you can try this k-point grid but without POP as an inelastic scattering")

            if self.nforced_POP/(len(self.kgrid[tp]["energy"])*len(self.kgrid[tp]["kpoints"][ib])) > 0.1:
                # TODO: this should be an exception but for now I turned to warning for testing.
                    warnings.warn("the k-grid is too coarse for an acceptable simulation of POP scattering in {} bands;"
                          " you can try this k-point grid but without POP as an inelastic scattering.".format(
                        ["conduction", "valence"][["n", "p"].index(tp)]))



    def unique_X_ib_ik_symmetrically_equivalent(self, tp, ib, ik):
        frac_k = self.kgrid[tp]["kpoints"][ib][ik]
        fractional_ks = [np.dot(self.rotations[i], frac_k) + self.translations[i] for i in range(len(self.rotations))]

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
        k = self.kgrid[tp]["actual kpoints"][ib][ik]
        result = []
        counter = 0
        # if E_radius == 0.0: # because then all the symmetrically equivalent k-points to current k have the same energy
        #     # print "BASE", ib, ik, E_radius
        #     symmetrically_equivalent_ks = self.unique_X_ib_ik_symmetrically_equivalent(tp, ib, ik)
        #     result += symmetrically_equivalent_ks
        #     counter += len(symmetrically_equivalent_ks)
        nk = len(self.kgrid[tp]["kpoints"][ib])

        for ib_prm in range(self.cbm_vbm[tp]["included"]):
            if ib==ib_prm:
                ik_prm = ik
            else:
                ik_prm = max(0, ik-100) # different bands, we start comparing energies ahead as energies aren't equal
            while (ik_prm<nk-1) and abs(self.kgrid[tp]["energy"][ib_prm][ik_prm+1]-(E+E_radius)) < tolerance:
                # if E_radius > 0.0:
                #     print "AFTER", ib, ik, E_radius
                # k_prm = self.kgrid[tp]["actual kpoints"][ib_prm][ik_prm+1]
                # X = self.cos_angle(k, k_prm)
                # result.append((X, ib_prm, ik_prm + 1))
                # result.append((X, ib_prm, ik_prm, k_prm))
                result.append((self.cos_angle(k,self.kgrid[tp]["actual kpoints"][ib_prm][ik_prm+1]),ib_prm,ik_prm))
                # symmetrically_equivalent_ks = self.unique_X_ib_ik_symmetrically_equivalent(tp, ib_prm, ik_prm)
                # result += symmetrically_equivalent_ks
                ik_prm += 1
                # counter += len(symmetrically_equivalent_ks) + 1
                counter += 1
            if ib==ib_prm:
                ik_prm = ik
            else:
                ik_prm = min(nk-1, ik+100)
            while (ik_prm>0) and abs(E+E_radius - self.kgrid[tp]["energy"][ib_prm][ik_prm-1]) < tolerance:
                # if E_radius > 0.0:
                #     print "BEFORE", ib, ik, E_radius
                # X = self.cos_angle(k, self.kgrid[tp]["actual kpoints"][ib_prm][ik_prm - 1])
                # result.append((X, ib_prm, ik_prm - 1))
                result.append((self.cos_angle(k, self.kgrid[tp]["actual kpoints"][ib_prm][ik_prm - 1]), ib_prm, ik_prm))
                # symmetrically_equivalent_ks = self.unique_X_ib_ik_symmetrically_equivalent(tp, ib_prm, ik_prm)
                # result += symmetrically_equivalent_ks
                ik_prm -= 1
                # counter += len(symmetrically_equivalent_ks) + 1
                counter += 1

        # If fewer than forced_min_npoints number of points were found, just return a few surroundings of the same band
        ik_prm = ik
        while counter < forced_min_npoints and ik_prm < nk - 1:
            # if E_radius >  0.0:
            #     print "EXTRA 1", ib, ik, E_radius
            # k_prm = self.kgrid[tp]["actual kpoints"][ib][ik_prm + 1]
            result.append((self.cos_angle(k, self.kgrid[tp]["actual kpoints"][ib][ik_prm + 1]), ib, ik_prm))
            # symmetrically_equivalent_ks = self.unique_X_ib_ik_symmetrically_equivalent(tp, ib_prm, ik_prm)
            # result += symmetrically_equivalent_ks
            # result.append((self.cos_angle(k, self.kgrid[tp]["actual kpoints"][ib][ik_prm + 1]), ib, ik_prm + 1))
            ik_prm += 1
            # counter += 1 + len(symmetrically_equivalent_ks)
            counter += 1
            self.nforced_POP += 1
        ik_prm = ik
        while counter < forced_min_npoints and ik_prm > 0:
            # if E_radius > 0.0:
            #     print "EXTRA 2", ib, ik, E_radius
            # k_prm = self.kgrid[tp]["actual kpoints"][ib][ik_prm - 1]
            result.append((self.cos_angle(k, self.kgrid[tp]["actual kpoints"][ib][ik_prm - 1]), ib, ik_prm))
            # symmetrically_equivalent_ks = self.unique_X_ib_ik_symmetrically_equivalent(tp, ib_prm, ik_prm)
            # result += symmetrically_equivalent_ks
            # result.append((self.cos_angle(k, self.kgrid[tp]["actual kpoints"][ib][ik_prm - 1]), ib, ik_prm - 1))
            ik_prm -= 1
            # counter += 1 + len(symmetrically_equivalent_ks)
            counter += 1
            self.nforced_POP += 1

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
            warnings.warn("same k and k' vectors as input of the elastic scattering equation")
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

        elif sname.upper() in ["DIS"]:
            return self.ds

        else:
            raise ValueError("The elastic scattering name {} is not supported!".format(sname))



    def integrate_over_DOSxE_dE(self, func, tp, fermi, T, interpolation_nsteps=None):
        if not interpolation_nsteps:
            interpolation_nsteps = max(5, int(500/len(self.egrid[tp]["energy"])) )
        integral = 0.0
        for ie in range(len(self.egrid[tp]["energy"]) - 1):
            E = self.egrid[tp]["energy"][ie]
            dE = abs(self.egrid[tp]["energy"][ie + 1] - E) / interpolation_nsteps
            dS = (self.egrid[tp]["DOS"][ie + 1] - self.egrid[tp]["DOS"][ie]) / interpolation_nsteps
            for i in range(interpolation_nsteps):
                # TODO:The DOS used is too simplistic and wrong (e.g., calc_doping might hit a limit), try 2*[2pim_hk_BT/hbar**2]**1.5
                integral += dE * (self.egrid[tp]["DOS"][ie] + i * dS) * func(E + i * dE, fermi, T)
        return integral



    def integrate_over_E(self, prop_list, tp, c, T, xDOS=True, xvel=False, interpolation_nsteps=None):
        if not interpolation_nsteps:
            interpolation_nsteps = max(5, int(500/len(self.egrid[tp]["energy"])) )
        diff = [0.0 for prop in prop_list]
        integral = self.ds
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
            # for alpha in range(3):
            dum = np.array([0.0, 0.0, 0.0])
            for j in range(2):
                # extract the indecies
                X, ib_prm, ik_prm = X_E_index[ib][ik][i + j]
                dum += integrand(tp, c, T, ib, ik, ib_prm, ik_prm, X, sname=sname, g_suffix=g_suffix)

            dum /= 2.0  # the average of points i and i+1 to integrate via the trapezoidal rule
            sum += dum * DeltaX  # In case of two points with the same X, DeltaX==0 so no duplicates
        return sum



    def el_integrand_X(self, tp, c, T, ib, ik, ib_prm, ik_prm, X, sname=None, g_suffix=""):
        k = self.kgrid[tp]["actual kpoints"][ib][ik]
        k_prm = self.kgrid[tp]["actual kpoints"][ib_prm][ik_prm]
        return (1 - X) * norm(k_prm)** 2 * self.s_el_eq(sname, tp, c, T, k, k_prm) \
               * self.G(tp, ib, ik, ib_prm, ik_prm, X)  \
               / self.kgrid[tp]["velocity"][ib_prm][ik_prm]
                # / abs(self.kgrid[tp]["velocity"][ib_prm][ik_prm][alpha])
        # We take |v| as scattering depends on the velocity itself and not the direction



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
        k = self.kgrid[tp]["actual kpoints"][ib][ik]
        f = self.kgrid[tp]["f0"][c][T][ib][ik]
        k_prm = self.kgrid[tp]["actual kpoints"][ib_prm][ik_prm]
        v_prm = self.kgrid[tp]["velocity"][ib_prm][ik_prm]
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
        integ = self.G(tp, ib, ik, ib_prm, ik_prm, X)/v_prm
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



    def s_inel_eq_isotropic(self, g_suffix):
        for tp in ["n", "p"]:
            for c in self.dopings:
                for T in self.temperatures:
                    for ib in range(len(self.kgrid[tp]["energy"])):
                        for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                            S_i = self.ds
                            S_o = self.ds
                            for j, X_Epm in enumerate(["X_Eplus_ik", "X_Eminus_ik"]):
                                for X_ib_ik in self.kgrid[tp][X_Epm][ib][ik]:
                                    X, ib_pm, ik_pm = X_ib_ik
                                    k = norm(self.kgrid[tp]["actual kpoints"][ib][ik])
                                    k_pm = norm(self.kgrid[tp]["actual kpoints"][ib_pm][ik_pm])
                                    if k == k_pm: # to prevent division by zero in eq-117&123 of ref. [R]
                                    #     print "k = k_prm"
                                    #     print k
                                    #     print ib, ik
                                    #     print k_pm
                                    #     print ib_pm, ik_pm
                                    #     print
                                        continue
                                    a = self.kgrid[tp]["a"][ib][ik]
                                    c_ = self.kgrid[tp]["c"][ib][ik]
                                    v_pm = self.kgrid[tp]["velocity"][ib_pm][ik_pm]
                                    a_pm = self.kgrid[tp]["a"][ib_pm][ik_pm]
                                    c_pm = self.kgrid[tp]["c"][ib_pm][ik_pm]
                                    g_pm = self.kgrid[tp]["g"+g_suffix][c][T][ib_pm][ik_pm]
                                    f = self.kgrid[tp]["f0"][c][T][ib][ik]
                                    f_pm = self.kgrid[tp]["f0"][c][T][ib_pm][ik_pm]
                                    N_POP = 1 / (np.exp(hbar * self.kgrid[tp]["W_POP"][ib][ik] / (k_B * T)) - 1)

                                    A_pm = a*a_pm + c_*c_pm*(k_pm**2+k**2)/(2*k_pm*k)
                                    beta_pm = (e**2*self.kgrid[tp]["W_POP"][ib_pm][ik_pm]*k_pm)/(4*pi*hbar*k*v_pm)*\
                                        (1/(self.epsilon_inf*epsilon_0)-1/(self.epsilon_s*epsilon_0))*6.2415093e20
                                    lamb_opm=beta_pm*(A_pm**2*log((k_pm+k)/abs(k_pm-k))-A_pm*c_*c_pm-a*a_pm*c_*c_pm)
                                    lamb_ipm=beta_pm*(A_pm**2*log((k_pm+k)/abs(k_pm-k))*(k_pm**2+k**2)/(2*k*k_pm)-
                                                      A_pm**2-c_**2*c_pm**2/3)

                                    S_o +=((N_POP + j+(-1)**j*f_pm)*lamb_opm)/len(self.kgrid[tp][X_Epm][ib][ik])
                                    S_i += ((N_POP + (1-j) + (-1)**(1-j)*f) * lamb_ipm * g_pm) \
                                           / len(self.kgrid[tp][X_Epm][ib][ik])

                            self.kgrid[tp]["S_o" + g_suffix][c][T][ib][ik] = S_o
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

        knrm = norm(self.kgrid[tp]["actual kpoints"][ib][ik])
        v = self.kgrid[tp]["velocity"][ib][ik]
        par_c = self.kgrid[tp]["c"][ib][ik]

        if sname.upper() == "ACD":
            # The following two lines are from Rode's chapter (page 38) which seems incorrect!
            # el_srate = (k_B*T*self.E_D[tp]**2*norm(k)**2)/(3*pi*hbar**2*self.C_el*1e9*v)\
            # *(3-8*self.kgrid[tp]["c"][ib][ik]**2+6*self.kgrid[tp]["c"][ib][ik]**4)*16.0217657

            # The following is from Deformation potentials and... Ref. [Q] (DOI: 10.1103/PhysRev.80.72 )
            return m_e * knrm * self.E_D[tp] ** 2 * k_B * T / (2 * pi * hbar ** 3 * self.C_el) \
                       * (3 - 8 * par_c ** 2 + 6 * par_c ** 4) * 1  # units work out! that's why conversion is 1

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
                                if norm(self.kgrid[tp][sname][c][T][ib][ik]) < 1:
                                    self.kgrid[tp][sname][c][T][ib][ik] = [1e9, 1e9, 1e9]
                                # for alpha in range(3):
                                #     if self.kgrid[tp][sname][c][T][ib][ik][alpha] < 1:
                                #         self.kgrid[tp][sname][c][T][ib][ik][alpha] = 1e9
                            self.kgrid[tp]["_all_elastic"][c][T][ib][ik] += self.kgrid[tp][sname][c][T][ib][ik]
                        self.kgrid[tp]["relaxation time"][c][T][ib] = 1/self.kgrid[tp]["_all_elastic"][c][T][ib]



    def map_to_egrid(self, prop_name, c_and_T_idx=True):
        """
        maps a propery from kgrid to egrid conserving the nomenclature. The mapped property should have the
            kgrid[tp][prop_name][c][T][ib][ik] data structure and will have egrid[tp][prop_name][c][T][ie] structure
        :param prop_name (string): the name of the property to be mapped. It must be available in the kgrid.
        :return:
        """
        # scalar_properties = ["g"]
        if not c_and_T_idx:
            self.initialize_var("egrid", prop_name, "vector", initval=self.ds, is_nparray=True, c_T_idx=False)
            for tp in ["n", "p"]:
                # try:
                #     self.egrid[tp][prop_name]
                #     print prop_name
                # except:
                #     # if prop_name in scalar_properties:
                #     #     self.egrid[tp][prop_name] = np.array([1e-20 for i in range(len(self.egrid[tp]["energy"]))])
                #     # else:
                #     self.egrid[tp][prop_name] = np.array([[1e-20, 1e-20, 1e-20] \
                #             for i in range(len(self.egrid[tp]["energy"]))])


                for ie, en in enumerate(self.egrid[tp]["energy"]):
                    N = 0.0  # total number of instances with the same energy
                    for ib in range(self.cbm_vbm[tp]["included"]):
                        for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                            if abs(self.kgrid[tp]["energy"][ib][ik] - en) < self.dE_global:

                                # weight = self.kgrid[tp]["kweights"][ib][ik]
                                # if prop_name in scalar_properties:
                                #     self.egrid[tp][prop_name][ie] += norm(self.kgrid[tp][prop_name][ib][ik]) * weight
                                # else:
                                self.egrid[tp][prop_name][ie] += self.kgrid[tp][prop_name][ib][ik] * 1
                                # N += 1
                                N += 1
                    self.egrid[tp][prop_name][ie] /= N
        else:
            self.initialize_var("egrid", prop_name, "vector", initval=self.ds, is_nparray=True, c_T_idx=True)

            for tp in ["n", "p"]:
                # try:
                #     self.egrid[tp][prop_name]
                # except:
                #     self.egrid[tp][prop_name] = {c: {T: np.array([[1e-20, 1e-20, 1e-20]
                #         for i in range(len(self.egrid[tp]["energy"]))]) for T in self.temperatures}
                #                                                                                 for c in self.dopings}

                for c in self.dopings:
                    for T in self.temperatures:
                        for ie, en in enumerate(self.egrid[tp]["energy"]):
                            N = 0.0 # total number of instances with the same energy
                            for ib in range(self.cbm_vbm[tp]["included"]):
                                for ik in range(len(self.kgrid[tp]["kpoints"][ib])):                            
                                    if abs(self.kgrid[tp]["energy"][ib][ik] - en) < self.dE_global:
                                        # wgt = self.kgrid[tp]["kweights"][ib][ik]
                                        self.egrid[tp][prop_name][c][T][ie]+=self.kgrid[tp][prop_name][c][T][ib][ik]*1
                                        N += 1
                            self.egrid[tp][prop_name][c][T][ie] /= N


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
        relative_error = 1000
        iter = 0
        temp_doping = {"n": 0.0, "p": 0.0}
        typ = self.get_tp(c)
        fermi = self.cbm_vbm[typ]["energy"]
        j = ["n", "p"].index(typ)
        funcs = [lambda E, fermi, T: f0(E,fermi,T), lambda E, fermi, T: 1-f0(E,fermi,T)]
        calc_doping = (-1)**(j+1) *self.nelec/self.volume / (A_to_m*m_to_cm)**3 \
                *abs(self.integrate_over_DOSxE_dE(func=funcs[j], tp=typ, fermi=fermi, T=T))


        # iterate around the CBM/VBM with finer and finer steps to find the Fermi level with a matching doping
        while (relative_error > tolerance) and (iter<max_iter):
            iter += 1 # to avoid an infinite loop
            fermi += alpha * (calc_doping - c)/abs(c + calc_doping) * fermi

            # calculate the overall concentration at the current fermi
            for j, tp in enumerate(["n", "p"]):
                integral = self.integrate_over_DOSxE_dE(func=funcs[j], tp=tp, fermi=fermi, T=T)
                temp_doping[tp] = (-1)**(j+1) * abs(integral*self.nelec/self.volume / (A_to_m*m_to_cm)**3)
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
                remove_list = ["W_POP", "effective mass", "actual kpoints", "X_E_ik", "X_Eplus_ik", "X_Eminus_ik"]
                for tp in ["n", "p"]:
                    for rm in remove_list:
                        try:
                            del (self.kgrid[tp][rm])
                        except:
                            pass
            with open("kgrid.json", 'w') as fp:
                json.dump(self.kgrid, fp,sort_keys = True, indent = 4, ensure_ascii=False, cls=MontyEncoder)



    def solve_BTE_iteratively(self):

        # calculating S_o scattering rate which is not a function of g
        if "POP" in self.inelastic_scatterings and not self.bs_is_isotropic:
            for g_suffix in ["", "_th"]:
                self.s_inelastic(sname="S_o"+ g_suffix, g_suffix=g_suffix)

        # solve BTE to calculate S_i scattering rate and perturbation (g) in an iterative manner
        for iter in range(self.maxiters):
            if "POP" in self.inelastic_scatterings:
                for g_suffix in ["", "_th"]:
                    if self.bs_is_isotropic:
                        self.s_inel_eq_isotropic(g_suffix=g_suffix)
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
                                self.kgrid[tp]["electric force"][c][T]) / (self.kgrid[tp]["S_o"][c][T]+ self.ds)
                            self.kgrid[tp]["g_th"][c][T][ib]=(self.kgrid[tp]["S_i_th"][c][T][ib]+self.kgrid[tp]["thermal force"][c][
                                T][ib]) / (self.kgrid[tp]["S_o_th"][c][T][ib] + self.kgrid[tp]["_all_elastic"][c][T][ib])

            for prop in ["electric force", "thermal force", "g", "g_POP", "g_th", "S_i", "S_o", "S_i_th", "S_o_th"]:
                self.map_to_egrid(prop_name=prop, c_and_T_idx=True)



    def calculate_transport_properties(self):
        for c in self.dopings:
            for T in self.temperatures:
                for tp in ["n", "p"]:
                    # norm is only for one vector but g has the ibxikx3 dimensions
                    # self.egrid[tp]["f"][c][T] = self.egrid[tp]["f0"][c][T] + norm(self.egrid[tp]["g"][c][T])
                    # self.egrid[tp]["f_th"][c][T]=self.egrid[tp]["f0"][c][T]+norm(self.egrid[tp]["g_th"][c][T])

                    self.egrid[tp]["f"][c][T] = self.egrid[tp]["f0"][c][T] + norm(self.egrid[tp]["g"][c][T])
                    self.egrid[tp]["f_th"][c][T]=self.egrid[tp]["f0"][c][T]+norm(self.egrid[tp]["g_th"][c][T])

                    # mobility numerators
                    for mu_el in self.elastic_scatterings:
                        self.egrid["mobility"][mu_el][c][T][tp] = (-1)*default_small_E/hbar* \
                            self.integrate_over_E(prop_list=["/"+mu_el, "df0dk"], tp=tp, c=c, T=T, xDOS=True, xvel=True)
                    for mu_inel in self.inelastic_scatterings:
                            self.egrid["mobility"][mu_inel][c][T][tp] = self.integrate_over_E(prop_list=["g_"+mu_inel],
                                                                                tp=tp,c=c,T=T,xDOS=True,xvel=True)
                    self.egrid["mobility"]["overall"][c][T][tp]=self.integrate_over_E(prop_list=["g"],
                                                                                tp=tp,c=c,T=T,xDOS=True,xvel=True)
                    self.egrid["J_th"][c][T][tp] = self.integrate_over_E(prop_list=["g_th"],
                            tp=tp, c=c, T=T, xDOS=True, xvel=True) * e * 1e24 # to bring J to A/cm2 units

                    # mobility denominators
                    for transport in self.elastic_scatterings + self.inelastic_scatterings + ["overall"]:
                        self.egrid["mobility"][transport][c][T][tp]/=default_small_E*\
                                        self.integrate_over_E(prop_list=["f0"],tp=tp, c=c, T=T, xDOS=True, xvel=False)
                    self.egrid["J_th"][c][T][tp] /= self.volume*self.integrate_over_E(prop_list=["f0"], tp=tp, c=c,
                                                                                         T=T, xDOS=True, xvel=False)

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

        self.bandgap = min(self.egrid["n"]["all_en_flat"]) - max(self.egrid["p"]["all_en_flat"])
        if abs(self.bandgap - (self.cbm_vbm["n"]["energy"] - self.cbm_vbm["p"]["energy"]+self.scissor)) > k_B*300:
            warnings.warn("The band gaps do NOT match! The selected k-mesh is probably too coarse.")
            # raise ValueError("The band gaps do NOT match! The selected k-mesh is probably too coarse.")

        # initialize g in the egrid
        self.map_to_egrid("g")
        self.map_to_egrid(prop_name="velocity", c_and_T_idx=False)

        # find the indexes of equal energy or those with ±hbar*W_POP for scattering via phonon emission and absorption
        self.generate_angles_and_indexes_for_integration()

        # calculate all elastic scattering rates in kgrid and then map it to egrid:
        for sname in self.elastic_scatterings:
            self.s_elastic(sname=sname)
            self.map_to_egrid(prop_name=sname)


        self.map_to_egrid(prop_name="_all_elastic")
        self.map_to_egrid(prop_name="relaxation time")

        for tp in ["n", "p"]:
            for c in self.dopings:
                for T in self.temperatures:
                    fermi = self.egrid["fermi"][c][T]
                    for ib in range(len(self.kgrid[tp]["energy"])):
                        for ik in range(len(self.kgrid[tp]["kpoints"][ib])):                    
                            E = self.kgrid[tp]["energy"][ib][ik]
                            v = self.kgrid[tp]["velocity"][ib][ik]

                            self.kgrid[tp]["f0"][c][T][ib][ik] = f0_value = f0(E, fermi, T)
                            self.kgrid[tp]["df0dk"][c][T][ib][ik] = hbar * df0dE(E,fermi, T) * v # in cm
                            self.kgrid[tp]["electric force"][c][T][ib][ik] = -1 * \
                                self.kgrid[tp]["df0dk"][c][T][ib][ik] * default_small_E / hbar # in 1/s
                            # self.kgrid[tp]["electric force"][c][T][ib][ik] = 1
                            self.kgrid[tp]["thermal force"][c][T][ib][ik] = - v * f0_value *(1-f0_value) *(\
                                    E/(k_B*T)-self.egrid["Seebeck_integral_numerator"][c][T][tp]/
                                    self.egrid["Seebeck_integral_denominator"][c][T][tp] ) * dTdz/T



                            # self.kgrid[tp]["thermal force"][c][T][ib][ik] = v * df0dz * unit_conversion
                            # df0dz_temp = f0(E, fermi, T) * (1 - f0(E, fermi, T)) * (
                                # E / (k_B * T) - df0dz_integral) * 1 / T * dTdz
        self.map_to_egrid(prop_name="f0")
        self.map_to_egrid(prop_name="df0dk") # This mapping is not correct as df0dk(E) is meaningless

        # solve BTE in presence of electric and thermal driving force to get perturbation to Fermi-Dirac: g
        # if "POP" in self.inelastic_scatterings:
        self.solve_BTE_iteratively()

        self.calculate_transport_properties()

        remove_list = ["W_POP", "effective mass", "actual kpoints", "kweights", "a", "c"]
        for tp in ["n", "p"]:
            for rm in remove_list:
                try:
                    del (self.kgrid[tp][rm])
                except:
                    pass


        if self.wordy:
            pprint(self.egrid)
            pprint(self.kgrid)

        with open("kgrid.txt", "w") as fout:
            # pprint(self.kgrid["n"], stream=fout)
            pprint(self.kgrid, stream=fout)
        with open("egrid.txt", "w") as fout:
            pprint(self.egrid, stream=fout)
            # pprint(self.egrid["n"], stream=fout)



if __name__ == "__main__":
    coeff_file = 'fort.123'

    # test
    AMSET = AMSET()
    # AMSET.run(coeff_file=coeff_file, kgrid_tp="coarse")
    cProfile.run('AMSET.run(coeff_file=coeff_file, kgrid_tp="coarse")')

    AMSET.to_json(trimmed=True)
