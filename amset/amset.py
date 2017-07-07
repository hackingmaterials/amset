# coding: utf-8
import warnings

import time

import logging

from scipy.interpolate import griddata

from analytical_band_from_BZT import Analytical_bands, outer, get_dos_from_poly_bands, get_poly_energy
from pprint import pprint

import numpy as np
from math import log

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
import multiprocessing
from joblib import Parallel, delayed
from analytical_band_from_BZT import get_energy

# global constants
hbar = _cd('Planck constant in eV s') / (2 * pi)
m_e = _cd('electron mass')  # in kg
Ry_to_eV = 13.605698066
A_to_m = 1e-10
m_to_cm = 100.00
A_to_nm = 0.1
e = _cd('elementary charge')
k_B = _cd("Boltzmann constant in eV/K")
epsilon_0 = 8.854187817e-12  # Absolute value of dielectric constant in vacuum [C**2/m**2N]
default_small_E = 1  # eV/cm the value of this parameter does not matter
dTdz = 10.0  # K/cm
sq3 = 3 ** 0.5

__author__ = "Alireza Faghaninia, Jason Frost, Anubhav Jain"
__copyright__ = "Copyright 2017, HackingMaterials"
__version__ = "0.1"
__maintainer__ = "Alireza Faghaninia"
__email__ = "alireza.faghaninia@gmail.com"
__status__ = "Development"
__date__ = "June 2017"


def norm(v):
    """method to quickly calculate the norm of a vector (v: 1x3 or 3x1) as numpy.linalg.norm is slower for this case"""
    return (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5


def f0(E, fermi, T):
    """returns the value of Fermi-Dirac at equilibrium for E (energy), fermi [level] and T (temperature)"""
    if E - fermi > 5:
        return 0.0
    elif E - fermi < -5:
        return 1.0
    else:
        return 1 / (1 + np.exp((E - fermi) / (k_B * T)))


def df0dE(E, fermi, T):
    """returns the energy derivative of the Fermi-Dirac equilibrium distribution"""
    if E - fermi > 5 or E - fermi < -5:  # This is necessary so at too low numbers python doesn't return NaN
        return 0.0
    else:
        return -1 / (k_B * T) * np.exp((E - fermi) / (k_B * T)) / (1 + np.exp((E - fermi) / (k_B * T))) ** 2


def cos_angle(v1, v2):
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


def fermi_integral(order, fermi, T, initial_energy=0, wordy=False):
    fermi = fermi - initial_energy
    integral = 0.
    nsteps = 100000.0
    # TODO: 1000000 works better (converges!) but for faster testing purposes we use larger steps
    # emesh = np.linspace(0.0, 30*k_B*T, nsteps) # We choose 20kBT instead of infinity as the fermi distribution will be 0
    emesh = np.linspace(0.0, 30 * k_B * T,
                        nsteps)  # We choose 20kBT instead of infinity as the fermi distribution will be 0
    dE = (emesh[-1] - emesh[0]) / (nsteps - 1.0)
    for E in emesh:
        integral += dE * (E / (k_B * T)) ** order / (1. + np.exp((E - fermi) / (k_B * T)))

    if wordy:
        print "order {} fermi integral at fermi={} and {} K".format(order, fermi, T)
        print integral
    return integral


def GB(x, eta):
    """Gaussian broadening. At very small eta values (e.g. 0.005 eV) this function goes to the dirac-delta of x."""

    return 1 / np.pi * 1 / eta * np.exp(-(x / eta) ** 2)

    ## although both expressions conserve the final transport properties, the one below doesn't conserve the scat. rates
    # return np.exp(-(x/eta)**2)


def calculate_Sio_list(tp, c, T, ib, once_called, kgrid, cbm_vbm, epsilon_s, epsilon_inf):
    S_i_list = [0.0 for ik in kgrid[tp]["kpoints"][ib]]
    S_i_th_list = [0.0 for ik in kgrid[tp]["kpoints"][ib]]
    S_o_list = [0.0 for ik in kgrid[tp]["kpoints"][ib]]
    S_o_th_list = [0.0 for ik in kgrid[tp]["kpoints"][ib]]

    for ik in range(len(kgrid[tp]["kpoints"][ib])):
        S_i_list[ik], S_i_th_list[ik], S_o_list[ik], S_o_th_list[ik] = \
            calculate_Sio(tp, c, T, ib, ik, once_called, kgrid, cbm_vbm, epsilon_s, epsilon_inf)

    return [S_i_list, S_i_th_list, S_o_list, S_o_th_list]


def calculate_Sio(tp, c, T, ib, ik, once_called, kgrid, cbm_vbm, epsilon_s, epsilon_inf):
    # print "calculating S_i and S_o for ib: {} and ik: {}".format(ib, ik)

    # S_i = np.array([self.gs, self.gs, self.gs])
    S_i = [np.array([1e-32, 1e-32, 1e-32]), np.array([1e-32, 1e-32, 1e-32])]
    S_i_th = [np.array([1e-32, 1e-32, 1e-32]), np.array([1e-32, 1e-32, 1e-32])]
    S_o = [np.array([1e-32, 1e-32, 1e-32]), np.array([1e-32, 1e-32, 1e-32])]
    S_o_th = [np.array([1e-32, 1e-32, 1e-32]), np.array([1e-32, 1e-32, 1e-32])]
    # S_o = np.array([self.gs, self.gs, self.gs])

    # v = sum(self.kgrid[tp]["velocity"][ib][ik]) / 3
    v = kgrid[tp]["norm(v)"][ib][ik] / sq3  # 3**0.5 is to treat each direction as 1D BS

    # k = m_e * self._avg_eff_mass[tp] * v / (hbar * e * 1e11)
    # k = m_e * abs(sum(cbm_vbm[tp]["eff_mass_xx"]) / 3) * v / (hbar * e * 1e11)
    k = kgrid[tp]["norm(k)"][ib][ik]

    a = kgrid[tp]["a"][ib][ik]
    c_ = kgrid[tp]["c"][ib][ik]
    # f = self.kgrid[tp]["f0"][c][T][ib][ik]
    f = kgrid[tp]["f"][c][T][ib][ik]
    f_th = kgrid[tp]["f_th"][c][T][ib][ik]




    N_POP = kgrid[tp]["N_POP"][c][T][ib][ik]
    for j, X_Epm in enumerate(["X_Eplus_ik", "X_Eminus_ik"]):
        # bypass k-points that cannot have k- associated with them (even though indexes may be available due to enforced scattering)
        if tp == "n" and X_Epm == "X_Eminus_ik" and kgrid[tp]["energy"][ib][ik] - hbar * \
                kgrid[tp]["W_POP"][ib][ik] < cbm_vbm[tp]["energy"]:
            continue

        if tp == "p" and X_Epm == "X_Eplus_ik" and kgrid[tp]["energy"][ib][ik] + hbar * \
                kgrid[tp]["W_POP"][ib][ik] > cbm_vbm[tp]["energy"]:
            continue

        # TODO: see how does dividing by counted affects results, set to 1 to test: #20170614: in GaAs, they are all equal anyway (at least among the ones checked)
        counted = len(kgrid[tp][X_Epm][ib][ik])
        # if len_eqE == 0:
        #     print "WARNING!!!! element {} of {} is empty!!".format(ik, X_Epm)
        for X_ib_ik in kgrid[tp][X_Epm][ib][ik]:
            X, ib_pm, ik_pm = X_ib_ik
            g_pm = kgrid[tp]["g"][c][T][ib_pm][ik_pm]
            g_pm_th = kgrid[tp]["g_th"][c][T][ib_pm][ik_pm]
            v_pm = kgrid[tp]["norm(v)"][ib_pm][ik_pm] / sq3  # 3**0.5 is to treat each direction as 1D BS
            # k_pm  = m_e*abs(sum(cbm_vbm[tp]["eff_mass_xx"])/3)*v_pm/(hbar*e*1e11)
            k_pm = kgrid[tp]["norm(k)"][ib_pm][ik_pm]
            abs_kdiff = abs(k_pm - k)
            if abs_kdiff < 1e-4:
                counted -= 1
                continue

            a_pm = kgrid[tp]["a"][ib_pm][ik_pm]
            c_pm = kgrid[tp]["c"][ib_pm][ik_pm]

            if tp == "n":
                f_pm = kgrid[tp]["f"][c][T][ib_pm][ik_pm]
                f_pm_th = kgrid[tp]["f_th"][c][T][ib_pm][ik_pm]
            else:
                f_pm = 1 - kgrid[tp]["f"][c][T][ib_pm][ik_pm]
                f_pm_th = 1 - kgrid[tp]["f_th"][c][T][ib_pm][ik_pm]


            A_pm = a * a_pm + c_ * c_pm * (k_pm ** 2 + k ** 2) / (2 * k_pm * k)

            # beta_pm = (e ** 2 * kgrid[tp]["W_POP"][ib_pm][ik_pm] * k_pm) / (4 * pi * hbar * k * v_pm) * \
            #           (1 / (epsilon_inf * epsilon_0) - 1 / (epsilon_s * epsilon_0)) * 6.2415093e20

            beta_pm = (e ** 2 * kgrid[tp]["W_POP"][ib_pm][ik_pm]) / (4 * pi * hbar * v_pm) * \
                      (1 / (epsilon_inf * epsilon_0) - 1 / (epsilon_s * epsilon_0)) * 6.2415093e20

            if not once_called:
                lamb_opm = beta_pm * (
                    A_pm ** 2 * log((k_pm + k) / (abs_kdiff + 1e-4)) - A_pm * c_ * c_pm - a * a_pm * c_ * c_pm)
                # because in the scalar form k+ or k- is suppused to be unique, here we take average

                S_o[j] += (N_POP + j + (-1) ** j * f_pm) * lamb_opm
                # print "ib_pm: {} ik_pm: {}, S_o-{}: {}".format(ib_pm, ik_pm, X_Epm, ((N_POP + j + (-1) ** j * f_pm) * lamb_opm))
                S_o_th[j] += (N_POP + j + (-1) ** j * f_pm_th) * lamb_opm

            lamb_ipm = beta_pm * (
                (k_pm ** 2 + k ** 2) / (2 * k * k_pm) * \
                A_pm ** 2 * log((k_pm + k) / (abs_kdiff + 1e-4)) - A_pm ** 2 - c_ ** 2 * c_pm ** 2 / 3)
            S_i[j] += (N_POP + (1 - j) + (-1) ** (1 - j) * f) * lamb_ipm * g_pm

            # print "ib_pm: {} ik_pm: {}, S_i-{}: {}".format(ib_pm, ik_pm, X_Epm, ((N_POP + (1 - j) + (-1) ** (1 - j) * f) * lamb_ipm * g_pm))

            S_i_th[j] += (N_POP + (1 - j) + (-1) ** (1 - j) * f_th) * lamb_ipm * g_pm_th

        if counted > 0:
            S_i[j] /= counted
            S_i_th[j] /= counted
            S_o[j] /= counted
            S_o_th[j] /= counted

    return [sum(S_i), sum(S_i_th), sum(S_o), sum(S_o_th)]


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

     AMSET is designed in a modular way so that users can add more scattering mechanisms as followed:
     ??? (instruction to add a scattering mechanism) ???

     you can control the level of theory via various inputs. For example, by assuming that the band structure is
     isotropic at the surrounding point of each k-point (i.e. bs_is_isotropic == True), once can significantly reduce
     the computational effort needed for accurate numerical integration of the scatterings. Furthermore,  ...
     (this part is not implemented yet: constant relaxation time approximation (cRTA),
     constant mean free path (cMFP) can be used by setting these variables to True )


    * a small comment on the structure of this code: the calculations are done and stroed in two main dictionary type
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

    # TODO-JF: if you ended up using Ashcroft for any part of AMSET, please add it above



    def __init__(self, calc_dir, material_params, model_params={}, performance_params={},
                 dopings=None, temperatures=None):
        """
        required parameters:
            calc_dir (str): path to the vasprun.xml
            material_params (dict): parameters related to the material

        """

        self.calc_dir = calc_dir
        self.dopings = dopings or [-1e19, -1e20]  # TODO: change the default to [-1e16,...,-1e21,1e21, ...,1e16] later
        self.all_types = [self.get_tp(c) for c in self.dopings]
        self.temperatures = temperatures or map(float,
                                                [300, 600])  # TODO: change the default to [50,100,...,1300] later
        self.debug_tp = self.get_tp(self.dopings[0])
        logging.debug("""debug_tp: "{}" """.format(self.debug_tp))
        self.set_material_params(material_params)
        self.set_model_params(model_params)
        self.set_performance_params(performance_params)

        self.read_vrun(calc_dir=self.calc_dir, filename="vasprun.xml")
        if self.poly_bands:
            self.cbm_vbm["n"]["energy"] = self.dft_gap
            self.cbm_vbm["p"]["energy"] = 0.0
            # @albalu why are the conduction and valence band k points being set to the same value?
                # because the way poly band generates a band structure is by mirroring conduction and valence bands
            # @albalu what is the format of self.poly_bands? it's a nested list, this can be improved actually
            self.cbm_vbm["n"]["kpoint"] = self.cbm_vbm["p"]["kpoint"] = self.poly_bands[0][0][0]

        self.num_cores = multiprocessing.cpu_count()
        if self.parallel:
            logging.info("number of cpu used in parallel mode: {}".format(self.num_cores))



    def remove_from_grids(self, kgrid_rm_list, egrid_rm_list):
        """deletes dictionaries storing properties about k points and E points that are no longer
        needed from fgrid and egrid"""
        for tp in ["n", "p"]:
            for rm in kgrid_rm_list:
                try:
                    del (self.kgrid[tp][rm])
                except:
                    pass
            # for erm in ["all_en_flat", "f_th", "g_th", "S_i_th", "S_o_th"]:
            for erm in egrid_rm_list:
                try:
                    del (self.egrid[tp][erm])
                except:
                    pass



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

        self.init_egrid(dos_tp="standard")

        self.bandgap = min(self.egrid["n"]["all_en_flat"]) - max(self.egrid["p"]["all_en_flat"])
        if abs(self.bandgap - (self.cbm_vbm["n"]["energy"] - self.cbm_vbm["p"]["energy"] + self.scissor)) > k_B * 300:
            warnings.warn("The band gaps do NOT match! The selected k-mesh is probably too coarse.")
            # raise ValueError("The band gaps do NOT match! The selected k-mesh is probably too coarse.")

        # initialize g in the egrid
        self.map_to_egrid("g", c_and_T_idx=True, prop_type="vector")
        self.map_to_egrid(prop_name="velocity", c_and_T_idx=False, prop_type="vector")

        print "average of the group velocity in e-grid!"
        print np.mean(self.egrid[self.debug_tp]["velocity"], 0)

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
                    fermi_norm = fermi - self.cbm_vbm[tp]["energy"]
                    for ib in range(len(self.kgrid[tp]["energy"])):
                        for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                            E = self.kgrid[tp]["energy"][ib][ik]
                            v = self.kgrid[tp]["velocity"][ib][ik]

                            self.kgrid[tp]["f0"][c][T][ib][ik] = f0(E, fermi, T) * 1.0
                            self.kgrid[tp]["df0dk"][c][T][ib][ik] = hbar * df0dE(E, fermi, T) * v  # in cm
                            self.kgrid[tp]["electric force"][c][T][ib][ik] = -1 * \
                                                                             self.kgrid[tp]["df0dk"][c][T][ib][
                                                                                 ik] * default_small_E / hbar  # in 1/s

                            E_norm = E - self.cbm_vbm[tp]["energy"]
                            # self.kgrid[tp]["electric force"][c][T][ib][ik] = 1
                            self.kgrid[tp]["thermal force"][c][T][ib][ik] = - v * f0(E_norm, fermi_norm, T) * (
                            1 - f0(E_norm, fermi_norm, T)) * ( \
                                                                                E_norm / (k_B * T) - self.egrid[
                                                                                    "Seebeck_integral_numerator"][c][T][
                                                                                    tp] /
                                                                                self.egrid[
                                                                                    "Seebeck_integral_denominator"][c][
                                                                                    T][tp]) * dTdz / T

        self.map_to_egrid(prop_name="f0", c_and_T_idx=True, prop_type="vector")
        self.map_to_egrid(prop_name="df0dk", c_and_T_idx=True, prop_type="vector")

        # solve BTE in presence of electric and thermal driving force to get perturbation to Fermi-Dirac: g
        self.solve_BTE_iteratively()

        if "POP" in self.inelastic_scatterings:
            for key in ["plus", "minus"]:
                with open("X_E{}_ik".format(key), "w") as fp:
                    json.dump(self.kgrid[self.debug_tp]["X_E{}_ik".format(key)][0], fp, cls=MontyEncoder)

        self.calculate_transport_properties()

        # logging.debug('self.kgrid_to_egrid_idx[self.debug_tp]: \n {}'.format(self.kgrid_to_egrid_idx[self.debug_tp]))
        # logging.debug('self.kgrid["velocity"][self.debug_tp][0]: \n {}'.format(self.kgrid[self.debug_tp]["velocity"][0]))
        # logging.debug('self.egrid["velocity"][self.debug_tp]: \n {}'.format(self.egrid[self.debug_tp]["velocity"]))

        # kremove_list = ["W_POP", "effective mass", "kweights", "a", "c""",
        #                 "f_th", "g_th", "S_i_th", "S_o_th"]

        kgrid_rm_list = ["effective mass", "kweights",
                         "f_th", "S_i_th", "S_o_th"]
        egrid_rm_list = ["f_th", "S_i_th", "S_o_th"]
        self.remove_from_grids(kgrid_rm_list, egrid_rm_list)

        pprint(self.egrid["mobility"])
        pprint(self.egrid["seebeck"])



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
        # TODO: remove this if later when anisotropic band structure is supported
        # if not self.bs_is_isotropic:
        #     raise IOError("Anisotropic option or bs_is_isotropic==False is NOT supported yet, please check back later")
        # what scattering mechanisms to be included
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
        self.dE_min = params.get("dE_min", 0.01)
        self.nE_min = params.get("nE_min", 2)
        # max eV range after which occupation is zero, we set this at least to 10*kB*300
        self.Ecut = params.get("Ecut", 10 * k_B * max(self.temperatures + [300]))
        self.adaptive_mesh = params.get("adaptive_mesh", False)

        self.dos_bwidth = params.get("dos_bwidth",
                                     0.1)  # in eV the bandwidth used for calculation of the total DOS (over all bands & IBZ k-points)
        self.nkdos = params.get("nkdos", 31)

        self.gs = 1e-32  # a global small value (generally used for an initial non-zero value)
        self.gl = 1e32  # a global large value

        # TODO: some of the current global constants should be omitted, taken as functions inputs or changed!
        self.wordy = params.get("wordy", False)
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
        projected = self._vrun.projected_eigenvalues
        # print len(projected[Spin.up][0][10])  # indexes : Spin, kidx, bidx, atomidx, s,py,pz,px,dxy,dyz,dz2,dxz,dx2

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
        # @albalu why is this not called the reciprocal lattice? your _rec_lattice is better
        # self._lattice_matrix = self._vrun.lattice_rec.matrix / (2 * pi)
        # @albalu is there a convention to name variables that are other objects with a "_" in the front?
            # yes, these are the ones that users are absolutely not supposed to change
        self._rec_lattice = self._vrun.final_structure.lattice.reciprocal_lattice

        bs = self._vrun.get_band_structure()
        self.lorbit = 11 if len(sum(self._vrun.projected_eigenvalues[Spin.up][0][10])) > 5 else 10

        self.DFT_cartesian_kpts = np.array(
                [self._rec_lattice.get_cartesian_coords(k) for k in self._vrun.actual_kpoints])/ A_to_nm


        # Remember that python band index starts from 0 so bidx==9 refers to the 10th band in VASP
        cbm_vbm = {"n": {"kpoint": [], "energy": 0.0, "bidx": 0, "included": 0, "eff_mass_xx": [0.0, 0.0, 0.0]},
                   "p": {"kpoint": [], "energy": 0.0, "bidx": 0, "included": 0, "eff_mass_xx": [0.0, 0.0, 0.0]}}
        cbm = bs.get_cbm()
        vbm = bs.get_vbm()

        logging.info("total number of bands: {}".format(self._vrun.get_band_structure().nb_bands))
        # print bs.nb_bands

        cbm_vbm["n"]["energy"] = cbm["energy"]
        cbm_vbm["n"]["bidx"] = cbm["band_index"][Spin.up][0]
        cbm_vbm["n"]["kpoint"] = bs.kpoints[cbm["kpoint_index"][0]].frac_coords

        cbm_vbm["p"]["energy"] = vbm["energy"]
        cbm_vbm["p"]["bidx"] = vbm["band_index"][Spin.up][-1]
        cbm_vbm["p"]["kpoint"] = bs.kpoints[vbm["kpoint_index"][0]].frac_coords

        self.dft_gap = cbm["energy"] - vbm["energy"]
        logging.debug("DFT gap from vasprun.xml : {} eV".format(self.dft_gap))

        if self.soc:
            self.nelec = cbm_vbm["p"]["bidx"] + 1
            # self.dos_normalization_factor = self._vrun.get_band_structure().nb_bands
        else:
            self.nelec = (cbm_vbm["p"]["bidx"] + 1) * 2
            # self.dos_normalization_factor = self._vrun.get_band_structure().nb_bands*2

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
                Ecut = self.Ecut if tp in self.all_types else min(self.Ecut/2.0, 0.25)
                sgn = (-1) ** i
                # @albalu what is this next line doing (even though it doesn't appear to be in use)?
                    # this part determines how many bands are have energy values close enough to CBM/VBM to be included
                while abs(min(sgn * bs["bands"]["1"][cbm_vbm[tp]["bidx"] + sgn * cbm_vbm[tp]["included"]]) -
                                          sgn * cbm_vbm[tp]["energy"]) < Ecut:
                    cbm_vbm[tp]["included"] += 1

                    # TODO: for now, I only include 1 band for quicker testing
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

    def seeb_int_num(self, c, T):
        """wrapper function to do an integration taking only the concentration, c, and the temperature, T, as inputs"""
        fn = lambda E, fermi, T: f0(E, fermi, T) * (1 - f0(E, fermi, T)) * E / (k_B * T)
        return {
        t: self.integrate_over_DOSxE_dE(func=fn, tp=t, fermi=self.egrid["fermi"][c][T], T=T, normalize_energy=True) for
        t in ["n", "p"]}

    def seeb_int_denom(self, c, T):
        """wrapper function to do an integration taking only the concentration, c, and the temperature, T, as inputs"""
        # fn = lambda E, fermi, T: f0(E, fermi, T) * (1 - f0(E, fermi, T))
        # return {t:self.integrate_over_DOSxE_dE(func=fn,tp=t,fermi=self.egrid["fermi"][c][T],T=T, normalize_energy=True) for t in ["n", "p"]}

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
               self.N_dis / self.volume ** (1 / 3) * 1e8 * self.charge["dislocations"] ** 2
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
            "n": {"energy": [], "DOS": [], "all_en_flat": [], "all_ks_flat": []},
            "p": {"energy": [], "DOS": [], "all_en_flat": [], "all_ks_flat": []},
            "mobility": {}
        }
        self.kgrid_to_egrid_idx = {"n": [],
                                   "p": []}  # list of band and k index that are mapped to each memeber of egrid
        self.Efrequency = {"n": [], "p": []}
        self.sym_freq = {"n": [], "p":[]}
        # reshape energies of all bands to one vector:
        E_idx = {"n": [], "p": []}
        for tp in ["n", "p"]:
            for ib, en_vec in enumerate(self.kgrid[tp]["energy"]):
                self.egrid[tp]["all_en_flat"] += list(en_vec)
                self.egrid[tp]["all_ks_flat"] += list(self.kgrid[tp]["kpoints"][ib])
                # also store the flatten energy (i.e. no band index) as a tuple of band and k-indexes
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
                sum_nksym = len(self.remove_duplicate_kpoints(self.get_sym_eq_ks_in_first_BZ(self.egrid[tp]["all_ks_flat"][i])))
                counter = 1.0  # because the ith member is already included in sum_E
                current_ib_ie_idx = [E_idx[tp][i]]
                j = i
                # while j<len(self.egrid[tp]["all_en_flat"])-1 and (counter <= self.nE_min or \
                #         abs(self.egrid[tp]["all_en_flat"][i]-self.egrid[tp]["all_en_flat"][j+1]) < self.dE_min):
                while j < len(self.egrid[tp]["all_en_flat"]) - 1 and \
                        abs(self.egrid[tp]["all_en_flat"][i] - self.egrid[tp]["all_en_flat"][j + 1]) < self.dE_min:
                    # while i < len(self.egrid[tp]["all_en_flat"]) - 1 and \
                    #          self.egrid[tp]["all_en_flat"][i] == self.egrid[tp]["all_en_flat"][i + 1] :
                    counter += 1
                    current_ib_ie_idx.append(E_idx[tp][j + 1])
                    sum_E += self.egrid[tp]["all_en_flat"][j + 1]
                    sum_nksym += len(self.remove_duplicate_kpoints(self.get_sym_eq_ks_in_first_BZ(self.egrid[tp]["all_ks_flat"][i+1])))

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
            # if dos_tp.lower()=="standard":
            #     energy_counter = [ne/len(self.egrid[tp]["all_en_flat"]) for ne in energy_counter]
            # TODO: what is the best value to pick for width here?I guess the lower is more precisely at each energy?
            # dum, self.egrid[tp]["DOS"] = get_dos(self.egrid[tp]["energy"], energy_counter,width = 0.05)

        # logging.debug("here self.kgrid_to_egrid_idx: {}".format(self.kgrid_to_egrid_idx[self.debug_tp]))
        # logging.debug(self.kgrid[self.debug_tp]["energy"])


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
            # self.egrid["mobility"+"_"+sn]={c:{T:{"n": 0.0, "p": 0.0} for T in self.temperatures} for c in self.dopings}
            self.egrid["mobility"][sn] = {c: {T: {"n": 0.0, "p": 0.0} for T in self.temperatures} for c in self.dopings}
        for transport in ["conductivity", "J_th", "seebeck", "TE_power_factor", "relaxation time constant"]:
            self.egrid[transport] = {c: {T: {"n": 0.0, "p": 0.0} for T in self.temperatures} for c in self.dopings}

        # populate the egrid at all c and T with properties; they can be called via self.egrid[prop_name][c][T] later
        self.calculate_property(prop_name="fermi", prop_func=self.find_fermi)

        # self.egrid["fermi"]= {
        #              2000000000000000.0: {
        #                  300: -0.575512702461
        #              }
        #          }


        # Since the SPB generated band structure may have several valleys, it's better to use the Fermi calculated from the actual band structure
        # self.calculate_property(prop_name="fermi_SPB", prop_func=self.find_fermi_SPB)

        ##  in case specific fermi levels are to be tested:


        # self.calculate_property(prop_name="f0", prop_func=f0, for_all_E=True)
        # self.calculate_property(prop_name="f", prop_func=f0, for_all_E=True)
        # self.calculate_property(prop_name="f_th", prop_func=f0, for_all_E=True)

        for prop in ["f", "f_th"]:
            self.map_to_egrid(prop_name=prop, c_and_T_idx=True)

        self.calculate_property(prop_name="f0x1-f0", prop_func=lambda E, fermi, T: f0(E, fermi, T)
                                                                                   * (1 - f0(E, fermi, T)),
                                for_all_E=True)
        self.calculate_property(prop_name="beta", prop_func=self.inverse_screening_length)
        self.calculate_property(prop_name="N_II", prop_func=self.calculate_N_II)
        self.calculate_property(prop_name="Seebeck_integral_numerator", prop_func=self.seeb_int_num)
        self.calculate_property(prop_name="Seebeck_integral_denominator", prop_func=self.seeb_int_denom)



    def get_Eidx_in_dos(self, E, Estep=None):
        if not Estep:
            Estep = max(self.dE_min, 0.0001)
        # there might not be anything wrong with the following but for now I thought using argmin() is safer
        # return int(round((E - self.dos_emin) / Estep))
        return abs(self.dos_emesh - E).argmin()

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
        return (a * self.kgrid[tp]["a"][ib_prm][ik_prm] + X * c * self.kgrid[tp]["c"][ib_prm][ik_prm]) ** 2



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
                logging.debug("# of kpoints indexes with low velocity: {}".format(len(rm_idx_list_ib)))
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


    @staticmethod
    def remove_duplicate_kpoints(kpts, dk=0.0001):
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

        i = 0
        while i < len(kpts) - 1:
            j = i
            while j < len(kpts) - 1 and ktuple[j + 1][0] - ktuple[i][0] < dk:

                # for i in range(len(kpts)-2):
                # if kpts[i][0] == kpts[i+1][0] and kpts[i][1] == kpts[i+1][1] and kpts[i][2] == kpts[i+1][2]:

                if (abs(kpts[i][0] - kpts[j + 1][0]) < dk or abs(kpts[i][0]) == abs(kpts[j + 1][0]) == 0.5) and \
                        (abs(kpts[i][1] - kpts[j + 1][1]) < dk or abs(kpts[i][1]) == abs(kpts[j + 1][1]) == 0.5) and \
                        (abs(kpts[i][2] - kpts[j + 1][2]) < dk or abs(kpts[i][2]) == abs(kpts[j + 1][2]) == 0.5):
                    rm_list.append(j + 1)
                j += 1
            i += 1

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


        # print "total time to remove duplicate k-points = {} seconds".format(time.time() - start_time)
        # print "number of duplicates removed:"
        # print len(rm_list)

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
        max_Ediff = max_Ediff or min(self.Ecut, 10 * k_B * max(self.temperatures))
        target_Ediff = target_Ediff or self.dE_min
        for tp in ["n", "p"]:
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



    # ultimately it might be most clean for this function to largely be two different functions (one for poly bands and one for analytical),
    # and then the parts they share can be separate functions called by both
    def init_kgrid(self, coeff_file, kgrid_tp="coarse"):
        Tmx = max(self.temperatures)
        if kgrid_tp == "coarse":
            nkstep = self.nkibz

        sg = SpacegroupAnalyzer(self._vrun.final_structure)
        self.rotations, self.translations = sg._get_symmetry()  # this returns unique symmetry operations
        logging.info("self.nkibz = {}".format(self.nkibz))

        # TODO: the following is NOT a permanent solution to speed up generation/loading of k-mesh, speed up get_ir_reciprocal_mesh later
        # TODO-JF (mid-term): you can take on this project to speed up get_ir_reciprocal_mesh or a similar function, right now it scales very poorly with larger mesh

        # create a mesh of k-points
        all_kpts = {}
        try:
            ibzkpt_filename = os.path.join(os.environ["AMSET_ROOT"], "{}_ibzkpt_{}.json".format(nkstep,
                                                        self._vrun.final_structure.formula.replace(" ", "")))
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
            # @albalu why is there an option to shift the k points and what are the weights?
            kpts_and_weights = sg.get_ir_reciprocal_mesh(mesh=(nkstep, nkstep, nkstep), is_shift=[0, 0, 0])
            # TODO: is_shift with 0.03 for y and 0.06 for z might give an error due to _all_elastic having twice length in kgrid compared to S_o, etc. I haven't figured out why
            # kpts_and_weights = sg.get_ir_reciprocal_mesh(mesh=(nkstep, nkstep, nkstep), is_shift=(0.00, 0.03, 0.06))
            kpts = [i[0] for i in kpts_and_weights]
            kpts = self.kpts_to_first_BZ(kpts)
            all_kpts["{}x{}x{}".format(nkstep, nkstep, nkstep)] = kpts
            with open(ibzkpt_filename, 'w') as fp:
                json.dump(all_kpts, fp, cls=MontyEncoder)

        # explicitly add the CBM/VBM k-points to calculate the parabolic band effective mass hence the relaxation time
        kpts.append(self.cbm_vbm["n"]["kpoint"])
        kpts.append(self.cbm_vbm["p"]["kpoint"])

        # @albalu why are there far fewer than nkstep^3 k points (printed out 8114 < 70^3)?
        logging.info("number of original ibz k-points: {}".format(len(kpts)))

        # TODO-JF: this if setup energy calculation for SPB and actual BS it would be nice to do this in two separate functions
        # if using analytical bands: create the object, determine list of band indices, and get energy info
        if not self.poly_bands:
            logging.debug("start interpolating bands from {}".format(coeff_file))
            analytical_bands = Analytical_bands(coeff_file=coeff_file)
            # @albalu Is this a list of the band indexes used in the calculation? Why is there an "i" in the name?
            all_ibands = []
            for i, tp in enumerate(["p", "n"]):
                sgn = (-1) ** (i + 1)
                for ib in range(self.cbm_vbm[tp]["included"]):
                    # @albalu what is self.cbm_vbm[tp]["bidx"]? I looked at self._vrun where this is set but I'm still confused
                    all_ibands.append(self.cbm_vbm[tp]["bidx"] + sgn * ib)

            start_time = time.time()
            logging.debug("all_ibands: {}".format(all_ibands))

            # @albalu what are all of these variables (in the next 5 lines)?
            engre, latt_points, nwave, nsym, nsymop, symop, br_dir = analytical_bands.get_engre(iband=all_ibands)
            nstv, vec, vec2 = analytical_bands.get_star_functions(latt_points, nsym, symop, nwave, br_dir=br_dir)
            out_vec2 = np.zeros((nwave, max(nstv), 3, 3))
            for nw in xrange(nwave):
                for i in xrange(nstv[nw]):
                    out_vec2[nw, i] = outer(vec2[nw, i], vec2[nw, i])

            print("time to get engre and calculate the outvec2: {} seconds".format(time.time() - start_time))

        # if using poly bands, remove duplicate k points (@albalu I'm not really sure what this is doing)
        else:
            # first modify the self.poly_bands to include all symmetrically equivalent k-points (k_i)
            # these points will be used later to generate energy based on the minimum norm(k-k_i)
            for ib in range(len(self.poly_bands)):
                for j in range(len(self.poly_bands[ib])):
                    self.poly_bands[ib][j][0] = self.remove_duplicate_kpoints(
                        self.get_sym_eq_ks_in_first_BZ(self.poly_bands[ib][j][0], cartesian=True))

        # calculate only the CBM and VBM energy values - @albalu why is this separate from the other energy value calculations?
        # here we assume that the cbm and vbm k-point coordinates read from vasprun.xml are correct:

        for i, tp in enumerate(["p", "n"]):
            sgn = (-1) ** i

            if not self.poly_bands:
                energy, velocity, effective_m = self.calc_analytical_energy(self.cbm_vbm[tp]["kpoint"],
                                                                            engre[i * self.cbm_vbm["p"]["included"]],
                                                                            nwave, nsym, nstv, vec, vec2, out_vec2,
                                                                            br_dir, sgn)
            else:
                energy, velocity, effective_m = self.calc_poly_energy(self.cbm_vbm[tp]["kpoint"], tp, 0)

            # @albalu why is there already an energy value calculated from vasp that this code overrides?
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
        energies = {"n": [0.0 for ik in kpts], "p": [0.0 for ik in kpts]}
        rm_list = {"n": [], "p": []}

        # calculate energies and choose which ones to remove
        for i, tp in enumerate(["p", "n"]):
            Ecut = self.Ecut if tp in self.all_types else min(self.Ecut / 2.0, 0.25)
            sgn = (-1) ** i
            # for ib in range(self.cbm_vbm[tp]["included"]):
            for ib in [0]:  # we only include the first band now (same for energies) to decide on ibz k-points
                if not self.parallel or self.poly_bands:  # The PB generator is fast enough no need for parallelization
                    for ik in range(len(kpts)):
                        if not self.poly_bands:
                            energy, velocity, effective_m = self.calc_analytical_energy(kpts[ik],engre[i * self.cbm_vbm[
                                "p"]["included"] + ib],nwave, nsym, nstv, vec, vec2,out_vec2, br_dir, sgn)
                        else:
                            energy, velocity, effective_m = self.calc_poly_energy(kpts[ik], tp, ib)
                        energies[tp][ik] = energy

                        # @albalu why do we exclude values of k that have a small component of velocity?
                        # @Jason: because scattering equations have v in the denominator: get too large for such points
                        if velocity[0] < 100 or velocity[1] < 100 or velocity[2] < 100 or \
                                        abs(energy - self.cbm_vbm[tp]["energy"]) > Ecut:
                            rm_list[tp].append(ik)
                else:
                    results = Parallel(n_jobs=self.num_cores)(delayed(get_energy)(kpts[ik],engre[i * self.cbm_vbm["p"][
                        "included"] + ib], nwave, nsym, nstv, vec, vec2, out_vec2, br_dir) for ik in range(len(kpts)))
                    for ik, res in enumerate(results):
                        energies[tp][ik] = res[0] * Ry_to_eV - sgn * self.scissor / 2.0
                        velocity = abs(res[1] / hbar * A_to_m * m_to_cm * Ry_to_eV)
                        if velocity[0] < 100 or velocity[1] < 100 or velocity[2] < 100 or \
                                        abs(energies[tp][ik] - self.cbm_vbm[tp]["energy"]) > Ecut:
                            # if tp=="p":
                            #     print "reason for removing the k-point:"
                            #     print "energy: {}".format(energies[tp][ik])
                            #     print "velocity: {}".format(velocity)
                            rm_list[tp].append(ik)
            rm_list[tp] = list(set(rm_list[tp]))

        # this step is crucial in DOS normalization when poly_bands to cover the whole energy range in BZ
        if self.poly_bands:
            all_bands_energies = {"n": [], "p": []}
            for tp in ["p", "n"]:
                all_bands_energies[tp] = energies[tp]
                for ib in range(1, len(self.poly_bands)):
                    for ik in range(len(kpts)):
                        energy, velocity, effective_m = get_poly_energy(
                            self._rec_lattice.get_cartesian_coords(kpts[ik]) / A_to_nm,
                            poly_bands=self.poly_bands, type=tp, ib=ib, bandgap=self.dft_gap + self.scissor)
                        all_bands_energies[tp].append(energy)
            self.dos_emin = min(all_bands_energies["p"])
            self.dos_emax = max(all_bands_energies["n"])

        # logging.debug("energies before removing k-points with off-energy:\n {}".format(energies))
        # remove energies that are out of range

        # rm_list_all = list(set(rm_list["n"] + rm_list["p"]))
        # kpts = np.delete(kpts, rm_list_all, axis=0)
        # kpts = list(kpts)
        # for tp in ["n", "p"]:
        #     energies[tp] = np.delete(energies[tp], rm_list_all, axis=0)

        kpts_copy = np.array(kpts)
        kpts = {"n": np.array(kpts_copy), "p": np.array(kpts_copy)}
        # print "n-rm_list"
        # print rm_list["n"]
        # print "p-rm_list"
        # print rm_list["p"]

        print "all types"
        print self.all_types
        for tp in ["n", "p"]:
            # if tp in self.all_types:
            if True:
                kpts[tp] = list(np.delete(kpts[tp], rm_list[tp], axis=0))
                energies[tp] = np.delete(energies[tp], rm_list[tp], axis=0)
            else: # in this case it doesn't matter if the k-mesh is loose
                kpts[tp] = list(np.delete(kpts[tp], rm_list["n"]+rm_list["p"], axis=0))
                energies[tp] = np.delete(energies[tp], rm_list["n"]+rm_list["p"], axis=0)
            if len(kpts[tp]) > 10000:
                warnings.warn("Too desne of a {}-type k-mesh (nk={}!); AMSET will be slow!".format(tp, len(kpts[tp])))

            logging.info("number of {}-type ibz k-points AFTER ENERGY-FILTERING: {}".format(tp, len(kpts[tp])))

        # TODO-JF (long-term): adaptive mesh is a good idea but current implementation is useless, see if you can come up with better method after talking to me
        if self.adaptive_mesh:
            raise IOError("adaptive mesh has not yet been implemented, please check back later!")
        #
        # if self.adaptive_mesh:
        #     all_added_kpoints = []
        #     all_added_kpoints += self.get_adaptive_kpoints(kpts, energies,
        #                                                    adaptive_Erange=[0 * k_B * Tmx, 1 * k_B * Tmx], nsteps=30)
        #
        #     # it seems it works mostly at higher energy values!
        #     # all_added_kpoints += self.get_ks_with_intermediate_energy(kpts,energies)
        #
        #     print "here the number of added k-points"
        #     print len(all_added_kpoints)
        #     print all_added_kpoints
        #     print type(kpts)
        #     kpts += all_added_kpoints

        # add in symmetrically equivalent k points
        for tp in ["n", "p"]:
            symmetrically_equivalent_ks = []
            for k in kpts[tp]:
                symmetrically_equivalent_ks += self.get_sym_eq_ks_in_first_BZ(k)
            kpts[tp] += symmetrically_equivalent_ks
            kpts[tp] = self.remove_duplicate_kpoints(kpts[tp])

            if len(kpts[tp]) < 3:
                raise ValueError("The k-point mesh for {}-type is too loose (number of kpoints = {}) "
                                "after filtering the initial k-mesh".format(tp, len(kpts)))

            logging.info("number of {}-type k-points after symmetrically equivalent kpoints are added: {}".format(
                        tp, len(kpts[tp])))


        # TODO: remove anything with "weight" later if ended up not using weights at all!
        kweights = {tp: [1.0 for i in kpts[tp]] for tp in ["n", "p"]}






        # actual initiation of the kgrid
        self.kgrid = {
            "n": {},
            "p": {}}

        for tp in ["n", "p"]:
            # @albalu [k for k in kpts] is always the same as kpts, right? yes but it is always a "list" regardless
            # of the type kpts (e.g. even if kpts was "numpy.array", the new list is "list"
            num_bands = self.cbm_vbm[tp]["included"]
            self.kgrid[tp]["kpoints"] = [kpts[tp] for ib in range(num_bands)]
            self.kgrid[tp]["kweights"] = [kweights[tp] for ib in range(num_bands)]
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
        self.initialize_var("kgrid", ["norm(k)"], "scalar", 0.0, is_nparray=False, c_T_idx=False)

        logging.debug("The DFT gap right before calculating final energy values: {}".format(self.dft_gap))



        for i, tp in enumerate(["p", "n"]):
            self.cbm_vbm[tp]["cartesian k"] = self._rec_lattice.get_cartesian_coords(self.cbm_vbm[tp]["kpoint"])/A_to_nm
            self.cbm_vbm[tp]["all cartesian k"] = self.get_sym_eq_ks_in_first_BZ(self.cbm_vbm[tp]["kpoint"], cartesian=True)
            self.cbm_vbm[tp]["all cartesian k"] = self.remove_duplicate_kpoints(self.cbm_vbm[tp]["all cartesian k"])

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
                    if self.kgrid[tp]["velocity"][ib][ik][0] < 100 or self.kgrid[tp]["velocity"][ib][ik][1] < 100 \
                            or self.kgrid[tp]["velocity"][ib][ik][2] < 100 or \
                                    abs(self.kgrid[tp]["energy"][ib][ik] - self.cbm_vbm[tp]["energy"]) > self.Ecut:
                        rm_idx_list[tp][ib].append(ik)
                    self.kgrid[tp]["effective mass"][ib][ik] = effective_mass

                    if self.poly_bands:
                        self.kgrid[tp]["a"][ib][ik] = 1.0 # parabolic band s-orbital only
                        self.kgrid[tp]["c"][ib][ik] = 0.0
                    else:
                        self.kgrid[tp]["a"][ib][ik] = fit_orbs["s"][ik]/ (fit_orbs["s"][ik]**2 + fit_orbs["p"][ik]**2)**0.5
                        self.kgrid[tp]["c"][ib][ik] = (1 - self.kgrid[tp]["a"][ib][ik]**2)**0.5

            logging.info("average of the {}-type group velocity in kgrid:\n {}".format(
                        tp, np.mean(self.kgrid[self.debug_tp]["velocity"][0], 0)))

        rearranged_props = ["velocity", "effective mass", "energy", "a", "c", "kpoints", "cartesian kpoints",
                            "old cartesian kpoints", "kweights",
                            "norm(v)", "norm(k)"]


        # TODO: the following is temporary, for some reason if # of kpts in different bands are NOT the same,
        # I get an error that _all_elastic is a list! so 1/self.kgrid[tp]["_all_elastic"][c][T][ib] cause error int/list!
        # that's why I am removing indexes from the first band at all bands! this is temperary
        # suggested solution: make the band index a key in the dictionary of kgrid rather than list index so we
        # can treat each band independently without their dimensions required to match!
        # TODO-AF or TODO-JF (mid-term): set the band index as a key in dictionary throughout AMSET to enable independent modification of bands information
        for tp in ["n", "p"]:
            rm_idx_list[tp] = [rm_idx_list[tp][0] for ib in range(self.cbm_vbm[tp]["included"])]

        # remove the k-points with off-energy values (>Ecut away from CBM/VBM) that are not removed already
        self.remove_indexes(rm_idx_list, rearranged_props=rearranged_props)

        logging.debug("dos_emin = {} and dos_emax= {}".format(self.dos_emin, self.dos_emax))

        for tp in ["n", "p"]:
            for ib in range(len(self.kgrid[tp]["energy"])):
                logging.info("Final # of {}-kpts in band #{}: {}".format(tp, ib, len(self.kgrid[tp]["kpoints"][ib])))

            if len(self.kgrid[tp]["kpoints"][0]) < 5:
                raise ValueError("VERY BAD {}-type k-mesh; please change the k-mesh and try again!".format(tp))

        logging.debug("time to calculate energy, velocity, m* for all: {} seconds".format(time.time() - start_time))

        # sort "energy", "kpoints", "kweights", etc based on energy in ascending order
        self.sort_vars_based_on_energy(args=rearranged_props, ascending=True)

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
            emesh, dos, dos_nbands = analytical_bands.get_dos_from_scratch(self._vrun.final_structure,
                                                                           [self.nkdos, self.nkdos, self.nkdos],
                                                                           self.dos_emin, self.dos_emax,
                                                                           int(round(
                                                                               (self.dos_emax - self.dos_emin) / max(
                                                                                   self.dE_min, 0.0001))),
                                                                           width=self.dos_bwidth, scissor=self.scissor,
                                                                           vbmidx=self.cbm_vbm["p"]["bidx"])
            self.dos_normalization_factor = dos_nbands if self.soc else dos_nbands * 2
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

        print("DOS normalization factor: {}".format(self.dos_normalization_factor))

        integ = 0.0
        for idos in range(len(dos) - 2):
            # if emesh[idos] > self.cbm_vbm["n"]["energy"]: # we assume anything below CBM as 0 occupation
            #     break
            integ += (dos[idos + 1] + dos[idos]) / 2 * (emesh[idos + 1] - emesh[idos])

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



    def sort_vars_based_on_energy(self, args, ascending=True):
        """sort the list of variables specified by "args" (type: [str]) in self.kgrid based on the "energy" values
        in each band for both "n"- and "p"-type bands and in ascending order by default."""
        for tp in ["n", "p"]:
            for ib in range(self.cbm_vbm[tp]["included"]):
                ikidxs = np.argsort(self.kgrid[tp]["energy"][ib])
                if not ascending:
                    ikidxs.reverse()
                for arg in args:
                    self.kgrid[tp][arg][ib] = np.array([self.kgrid[tp][arg][ib][ik] for ik in ikidxs])



    def generate_angles_and_indexes_for_integration(self, avg_Ediff_tolerance=0.02):
        self.initialize_var("kgrid", ["X_E_ik", "X_Eplus_ik", "X_Eminus_ik"], "scalar", [], is_nparray=False,
                            c_T_idx=False)

        # elastic scattering
        for tp in ["n", "p"]:
            for ib in range(len(self.kgrid[tp]["energy"])):
                self.nforced_scat = {"n": 0.0, "p": 0.0}
                self.ediff_scat = {"n": [], "p": []}
                for ik in range(len(self.kgrid[tp]["kpoints"][ib])):
                    self.kgrid[tp]["X_E_ik"][ib][ik] = self.get_X_ib_ik_near_new_E(tp, ib, ik,
                                                                                   E_change=0.0, forced_min_npoints=2,
                                                                                   tolerance=self.dE_min)
                enforced_ratio = self.nforced_scat[tp] / sum([len(points) for points in self.kgrid[tp]["X_E_ik"][ib]])
                logging.info("enforced scattering ratio for {}-type elastic scattering at band {}:\n {}".format(tp, ib,
                                                                                                                enforced_ratio))
                # print self.nforced_scat[tp] / (2 * len(self.kgrid[tp]["kpoints"][ib]))
                # if self.nforced_scat[tp] / (2 * len(self.kgrid[tp]["kpoints"][ib])) > 0.1:
                if enforced_ratio > 0.1:
                    # TODO: this should be an exception but for now I turned to warning for testing.
                    warnings.warn(
                        "the k-grid is too coarse for an acceptable simulation of elastic scattering in {} bands;"
                        .format(["conduction", "valence"][["n", "p"].index(tp)]))

                avg_Ediff = sum(self.ediff_scat[tp]) / max(len(self.ediff_scat[tp]), 1)
                if avg_Ediff > avg_Ediff_tolerance:
                    # TODO: change it back to ValueError as it was originally, it was switched to warning for fast debug
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
                                # E_change=sgn * hbar * self.kgrid[tp]["W_POP"][ib][ik],forced_min_npoints=2,
                                E_change= + hbar * self.kgrid[tp]["W_POP"][ib][ik],forced_min_npoints=2,
                                   tolerance=self.dE_min)
                        self.kgrid[tp]["X_Eminus_ik"][ib][ik] = self.get_X_ib_ik_near_new_E(tp, ib, ik,
                                # E_change= (sgn*-1)* hbar * self.kgrid[tp]["W_POP"][ib][ik],forced_min_npoints=2,
                                E_change= - hbar * self.kgrid[tp]["W_POP"][ib][ik],forced_min_npoints=2,
                                        tolerance=self.dE_min)

                    enforced_ratio = self.nforced_scat[tp] / (
                        sum([len(points) for points in self.kgrid[tp]["X_Eplus_ik"][ib]]) + \
                        sum([len(points) for points in self.kgrid[tp]["X_Eminus_ik"][ib]]))
                    logging.info(
                        "enforced scattering ratio: {}-type inelastic at band {}:\n{}".format(tp, ib, enforced_ratio))

                    if enforced_ratio > 0.1:
                        # TODO: this should be an exception but for now I turned to warning for testing.
                        warnings.warn(
                            "the k-grid is too coarse for an acceptable simulation of POP scattering in {} bands;"
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
            Attention! this function assumes self.kgrid is sorted based on the energy in ascending order."""
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
                while (ik_prm >= 0) and (ik_prm < nk) and (
                    abs(self.kgrid[tp]["energy"][ib_prm][ik_prm] - E_prm) < tolerance):
                    X_ib_ik = (cos_angle(k, self.kgrid[tp]["cartesian kpoints"][ib_prm][ik_prm]), ib_prm, ik_prm)
                    if (X_ib_ik[1], X_ib_ik[2]) not in [(entry[1], entry[2]) for entry in result]:
                        result.append(X_ib_ik)
                    ik_prm += step

        # If fewer than forced_min_npoints number of points were found, just return a few surroundings of the same band
        ib_prm = ib
        # if E_change == 0.0:
        #    ik_prm = ik
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
        :param sname (string): abbreviation of the name of the elastic scatteirng mechanisms; options: IMP, ADE, PIE, DIS
        :param c:
        :param T:
        :param k:
        :param k_prm:
        :return:
        """
        norm_diff_k = norm(k - k_prm)  # the slope for PIE and IMP don't match with bs_is_isotropic
        # norm_diff_k = norm(k) # slope kind of matches with bs_is_isotropic at least for PIE but it's 5X larger
        # norm_diff_k = norm(k_prm) # slope kind of matches with bs_is_isotropic at least for PIE but it's 5X larger
        # norm_diff_k = (norm(k_prm)**2 + norm(k)**2)**0.5 # doesn't work, the ratios are not a fixed number

        if norm_diff_k == 0.0:
            print "WARNING!!! same k and k' vectors as input of the elastic scattering equation"

            # warnings.warn("same k and k' vectors as input of the elastic scattering equation")
            # raise ValueError("same k and k' vectors as input of the elastic scattering equation."
            #                  "Check get_X_ib_ik_within_E_radius for possible error")
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



    def integrate_over_BZ(self, prop_list, tp, c, T, xDOS=False, xvel=False, weighted=True):

        # weighted = False

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



    def integrate_over_E(self, prop_list, tp, c, T, xDOS=False, xvel=False, weighted=False, interpolation_nsteps=None):

        # for now I keep weighted as False, to re-enable weighting, all GaAs tests should be re-evaluated.

        # weighted = False

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
            # return integral / sum([freq ** wpower for freq in self.Efrequency[tp][:-1]]) / sum(self.egrid[tp]["df0dE"][c][T][:-1])
            # return integral / sum([freq**wpower for freq in self.Efrequency[tp][0:imax_occ]])
            # return integral / (sum([freq**wpower for ie, freq in enumerate(self.Efrequency[tp][0:imax_occ])]))/(-sum(self.egrid[tp]["df0dE"][c][T]))

            # return integral / (sum([freq ** wpower for ie, freq in enumerate(self.Efrequency[tp][0:imax_occ])]))

            # return integral / (sum([(-self.egrid[tp]["df0dE"][c][T][ie]) * self.Efrequency[tp][ie]**wpower for ie in
            #                    range(len(self.Efrequency[tp][:-1]))]))
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
        # previous match (commented on 6/26/2017)
        # return (1 - X) * (m_e * self._avg_eff_mass[tp] * self.kgrid[tp]["velocity"][ib_prm][ik_prm] / (
        #     hbar * e * 1e11)) ** 2 * self.s_el_eq(sname, tp, c, T, k, k_prm) \
        #        * self.G(tp, ib, ik, ib_prm, ik_prm, X) \
        #        * 1.0 / (self.kgrid[tp]["norm(v)"][ib_prm][ik_prm]/sq3)

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
        fermi = self.egrid["fermi"][c][T]

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
        # integ = self.G(tp, ib, ik, ib_prm, ik_prm, X)/v_prm # simply /v_prm is wrong and creates artificial anisotropy
        # integ = self.G(tp, ib, ik, ib_prm, ik_prm, X)*norm(1.0/v_prm)
        # integ = self.G(tp, ib, ik, ib_prm, ik_prm, X) / self.kgrid[tp]["norm(v)"][ib_prm][ik_prm]


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
                        if self.parallel and len(self.kgrid[tp]["size"]) * max(self.kgrid[tp]["size"]) > 10000:
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

        # TODO: decide on knrm and whether it needs a reference k-point (i.e. CBM/VBM) where the reference point is not Gamma. No ref. result in large rates in PbTe.
        # I justify subtracting the CBM/VBM actual k-points as follows:
        # v = sum(self.kgrid[tp]["velocity"][ib][ik])/3
        # v = norm(self.kgrid[tp]["velocity"][ib][ik])
        v = self.kgrid[tp]["norm(v)"][ib][ik] / sq3  # because of isotropic assumption, we treat the BS as 1D
        # v = self.kgrid[tp]["velocity"][ib][ik] # because it's isotropic, it doesn't matter which one we choose
        # perhaps more correct way of defining knrm is as follows since at momentum is supposed to be proportional to
        # velocity as it is in free-electron formulation so we replaced hbar*knrm with m_e*v/(1e11*e) (momentum)


        # if self.poly_bands: # the first one should be v and NOT v * sq3 so that the two match in SPB
        if False:  # I'm 90% sure that there is not need for the first type of knrm and that's why I added if False for now
            knrm = m_e * self._avg_eff_mass[tp] * (v) / (
            hbar * e * 1e11)  # in nm given that v is in cm/s and hbar in eV.s; this resulted in very high ACD and IMP scattering rates, actually only PIE would match with aMoBT results as it doesn't have k_nrm in its formula
        # TODO: make sure that ACD scattering as well as others match in SPB between bs_is_isotropic and when knrm is the following and not above (i.e. not m*v/hbar*e)
        else:
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
            # knrm = self.kgrid[tp]["norm(k)"][ib][ik] don't use this! it's wrong anyway and shouldn't change knrm just for IMP
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
                        for ib, ik in self.kgrid_to_egrid_idx[tp][ie]:
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
                                # print self.kgrid_to_egrid_idx[tp][ie]
                                for ib, ik in self.kgrid_to_egrid_idx[tp][ie]:
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
            integral = self.integrate_over_E(prop_list=["f0x1-f0"], tp=tp, c=c, T=T, xDOS=True, weighted=True)
            # integral = sum(self.integrate_over_BZ(["f0", "1-f0"], tp, c, T, xDOS=True, xvel=False, weighted=False))/3

            # beta[tp] = (e**2 / (self.epsilon_s * epsilon_0*k_B*T) * integral * 6.241509324e27)**0.5

            beta[tp] = (e ** 2 / (self.epsilon_s * epsilon_0 * k_B * T) * integral / self.volume * 1e12 / e) ** 0.5
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

        for tp in ["n", "p"]:
            for c in self.dopings:
                for T in self.temperatures:
                    for ie in range(len(self.egrid[tp]["g_POP"][c][T])):
                        if norm(self.egrid[tp]["g_POP"][c][T][ie]) > 1:
                            self.egrid[tp]["g_POP"][c][T][ie] = [1e-5, 1e-5, 1e-5]



    def calculate_transport_properties(self):
        integrate_over_kgrid = True
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

                    if integrate_over_kgrid:
                        if tp == "n":
                            denom = self.integrate_over_BZ(["f0"], tp, c, T, xDOS=False, xvel=False,
                                                       weighted=False) #* 1e-7 * 1e-3 * self.volume
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
                    fermi = self.egrid["fermi"][c][T]
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

                    # calculating other overall transport properties:
                    self.egrid["conductivity"][c][T][tp] = self.egrid["mobility"]["overall"][c][T][tp] * e * abs(c)
                    self.egrid["seebeck"][c][T][tp] = -1e6 * k_B * (self.egrid["Seebeck_integral_numerator"][c][T][tp] \
                                                                    / self.egrid["Seebeck_integral_denominator"][c][T][
                                                                        tp] - (
                                                                    self.egrid["fermi"][c][T] - self.cbm_vbm[tp][
                                                                        "energy"]) / (k_B * T))
                    self.egrid["TE_power_factor"][c][T][tp] = self.egrid["seebeck"][c][T][tp] ** 2 \
                                                              * self.egrid["conductivity"][c][T][tp] / 1e6  # in uW/cm2K
                    if "POP" in self.inelastic_scatterings:  # when POP is not available J_th is unreliable
                        self.egrid["seebeck"][c][T][tp] += 0.0
                        # TODO: for now, we ignore the following until we figure out the units see why values are high!
                        # self.egrid["seebeck"][c][T][tp] += 1e6 \
                        #                 * self.egrid["J_th"][c][T][tp]/self.egrid["conductivity"][c][T][tp]/dTdz

                    print "3 {}-seebeck terms at c={} and T={}:".format(tp, c, T)
                    print self.egrid["Seebeck_integral_numerator"][c][T][tp] \
                          / self.egrid["Seebeck_integral_denominator"][c][T][tp] * -1e6 * k_B
                    print + (self.egrid["fermi"][c][T] - self.cbm_vbm[tp]["energy"]) * 1e6 * k_B / (k_B * T)
                    print + self.egrid["J_th"][c][T][tp] / self.egrid["conductivity"][c][T][tp] / dTdz * 1e6

                    other_type = ["p", "n"][1 - j]
                    self.egrid["seebeck"][c][T][tp] = (self.egrid["conductivity"][c][T][tp] * \
                                                       self.egrid["seebeck"][c][T][tp] -
                                                       self.egrid["conductivity"][c][T][other_type] * \
                                                       self.egrid["seebeck"][c][T][other_type]) / (
                                                      self.egrid["conductivity"][c][T][tp] +
                                                      self.egrid["conductivity"][c][T][other_type])
                    # since sigma = c_e x e x mobility_e + c_h x e x mobility_h:
                    # self.egrid["conductivity"][c][T][tp] += self.egrid["conductivity"][c][T][other_type]



    # TODO-JF: this function needs a MAJOR revision and it does not interfere with the main code so it might be a good
    # place to start; I would first figure out how to directly save a Plotly plot to a file (it doesn't matter what the
    # format is (i.e. png, jpeg, etc) and then based on that decide where to do plot. I encourage keeping everything in
    # plotly as the plots look nice and the interactive mode on browser is a very nice tool for display and debugging
    def plot(self, plotc=None, plotT=None, path=None, textsize=40, ticksize=35, margin_left=160, margin_bottom=120,
             fontfamily="serif"):
        """plots some of the outputs for more detailed analysis, debugging, etc"""
        from matminer.figrecipes.plotly.make_plots import PlotlyFig
        if not plotc:
            plotc = self.dopings[0]
        if not plotT:
            plotT = 300.0
        if not path:
            path = os.path.join(os.getcwd(), "plots")
            if not os.path.exists(path):
                os.makedirs(name=path)
        fformat = "html"


        for tp in [self.debug_tp]:
            print(
            'plotting: first set of plots: "log10 of mobility", "relaxation time", "_all_elastic", "ACD", "df0dk"')
            plt = PlotlyFig(x_title="Temperature (K)", y_title="log10 of mobility (^10 cm2/V.s)", textsize=textsize,
                            filename=os.path.join(path, "{}_{}.{}".format("log10_of_mobility", tp, fformat)),
                            ticksize=ticksize, margin_left=margin_left, margin_bottom=margin_bottom,
                            fontfamily=fontfamily)
            all_plots = []
            for mo in ["overall", "average"] + self.elastic_scatterings + self.inelastic_scatterings:
                all_plots.append({"x_col": self.temperatures,
                                  # I temporarity (for debugging purposes) added abs() for cases when mistakenly I get negative mobility values!
                                  "y_col": [log(abs(sum(self.egrid["mobility"][mo][plotc][T][tp]) / 3), 10) for T in
                                            self.temperatures],
                                  "text": mo, "size": textsize / 2, "mode": "lines+markers", "legend": "", "color": ""
                                  })
            plt.xy_plot(x_col=[],
                        y_col=[],
                        add_xy_plot=all_plots)

            # plt = PlotlyFig(plot_mode='offline', y_title="# of repeated energy in kgrid", x_title="Energy (eV)",
            #                 plot_title=None, filename=os.path.join(path, "{}_{}.{}".format("E_histogram", tp, fformat)),
            #                 textsize=textsize, ticksize=ticksize, scale=1, margin_left=margin_left,
            #                 margin_bottom=margin_bottom)
            #
            # plt.xy_plot(x_col=[E - self.cbm_vbm[tp]["energy"] for E in self.egrid[tp]["energy"]],
            #             y_col=self.Efrequency[tp])
            #
            # plt = PlotlyFig(plot_mode='offline', y_title="# of symmetrically equivalent kpoints", x_title="Energy (eV)",
            #                 plot_title=None, filename=os.path.join(path, "{}_{}.{}".format("E_histogram_symk", tp, fformat)),
            #                 textsize=textsize, ticksize=ticksize, scale=1, margin_left=margin_left,
            #                 margin_bottom=margin_bottom)
            #
            # plt.xy_plot(x_col=[E - self.cbm_vbm[tp]["energy"] for E in self.egrid[tp]["energy"]],
            #             y_col=self.sym_freq[tp])


            for prop in ["energy", "df0dk"]:
                plt = PlotlyFig(plot_mode='offline', y_title=prop, x_title="norm(k)",
                                plot_title="{} in kgrid".format(prop),
                                filename=os.path.join(path, "{}_{}.{}".format("{}_kgrid".format(prop), tp, fformat)),
                                textsize=textsize, ticksize=ticksize, scale=1, margin_left=margin_left,
                                margin_bottom=margin_bottom)
                if prop in ["energy"]:
                    plt.xy_plot(x_col=self.kgrid[tp]["norm(k)"][0], y_col=self.kgrid[tp][prop][0])
                if prop in ["df0dk"]:
                    for c in self.dopings:
                        for T in [plotT]:
                            plt.xy_plot(x_col=self.kgrid[tp]["norm(k)"][0],
                                        y_col=[sum(p / 3) for p in self.kgrid[tp][prop][c][T][0]])

            plt = PlotlyFig(plot_mode='offline', y_title="norm(velocity) (cm/s)", x_title="norm(k)",
                            plot_title="velocity in kgrid",
                            filename=os.path.join(path, "{}_{}.{}".format("v_kgrid", tp, fformat)),
                            textsize=textsize, ticksize=ticksize, scale=1, margin_left=margin_left,
                            margin_bottom=margin_bottom)
            plt.xy_plot(x_col=self.kgrid[tp]["norm(k)"][0], y_col=self.kgrid[tp]["norm(v)"][0])

            prop_list = ["relaxation time", "_all_elastic", "df0dk"] + self.elastic_scatterings
            if "POP" in self.inelastic_scatterings:
                prop_list += ["g", "g_POP", "g_th", "S_i", "S_o"]
            for c in self.dopings:
                # for T in self.temperatures:
                for T in [plotT]:
                    plt = PlotlyFig(plot_mode='offline', y_title="ACD scattering rate (1/s)", x_title="energy (eV)",
                                    plot_title="ACD in kgrid",
                                    filename=os.path.join(path, "{}_{}.{}".format("ACD_kgrid", tp, fformat)),
                                    textsize=textsize, ticksize=ticksize, scale=1, margin_left=margin_left,
                                    margin_bottom=margin_bottom)
                    plt.xy_plot(x_col=[E - self.cbm_vbm[tp]["energy"] for E in self.kgrid[tp]["energy"][0]]
                                , y_col=[sum(p) / 3 for p in self.kgrid[tp]["ACD"][c][T][0]])

                    for prop_name in prop_list:
                        plt = PlotlyFig(plot_title="c={} 1/cm3, T={} K".format(c, T), x_title="Energy (eV)",
                                        y_title=prop_name, hovermode='closest',
                                        filename=os.path.join(path,
                                                              "{}_{}_{}_{}.{}".format(prop_name, tp, c, T, fformat)),
                                        plot_mode='offline', username=None, api_key=None, textsize=textsize,
                                        ticksize=ticksize, fontfamily=None,
                                        height=800, width=1000, scale=None, margin_top=100, margin_bottom=margin_bottom,
                                        margin_left=margin_left,
                                        margin_right=80,
                                        pad=0)
                        prop = [sum(p) / 3 for p in self.egrid[tp][prop_name][c][
                            T]]  # scat. rates are not vectors all 3 numbers represent single isotropic scattering rate
                        plt.xy_plot(x_col=[E - self.cbm_vbm[tp]["energy"] for E in self.egrid[tp]["energy"]],
                                    y_col=prop)

            print('plotting: second set of plots: "velocity", "Ediff"')

            # plot versus energy in self.egrid
            # prop_list = ["velocity", "Ediff"]
            prop_list = ["velocity"]
            for prop_name in prop_list:
                plt = PlotlyFig(plot_title=None, x_title="Energy (eV)", y_title=prop_name, hovermode='closest',
                                filename=os.path.join(path, "{}_{}.{}".format(prop_name, tp, fformat)),
                                plot_mode='offline', username=None, api_key=None, textsize=textsize, ticksize=ticksize,
                                fontfamily=None,
                                height=800, width=1000, scale=None, margin_top=100, margin_bottom=margin_bottom,
                                margin_left=margin_left, margin_right=80,
                                pad=0)
                if prop_name == "Ediff":
                    y_col = [self.egrid[tp]["energy"][i + 1] - \
                             self.egrid[tp]["energy"][i] for i in range(len(self.egrid[tp]["energy"]) - 1)]
                else:
                    y_col = [sum(p) / 3 for p in
                             self.egrid[tp][prop_name]]  # velocity is actually a vector so we take norm
                    plt.xy_plot(x_col=[E - self.cbm_vbm[tp]["energy"] for E in self.egrid[tp]["energy"][:len(y_col)]],
                                y_col=y_col, error_type="data",
                                error_array=[np.std(p) for p in self.egrid[tp][prop_name]], error_direction="y")
                    # xrange=[self.egrid[tp]["energy"][0], self.egrid[tp]["energy"][0]+0.6])

            # plot versus norm(k) in self.kgrid
            prop_list = ["energy"]
            for prop_name in prop_list:
                x_col = self.kgrid[tp]["norm(k)"][0]

                plt = PlotlyFig(plot_title=None, x_title="k [1/nm]",
                                y_title="{} at the 1st band".format(prop_name), hovermode='closest',
                                filename=os.path.join(path, "{}_{}.{}".format(prop_name, tp, fformat)),
                                plot_mode='offline', username=None, api_key=None, textsize=textsize, ticksize=ticksize,
                                fontfamily=None,
                                height=800, width=1000, scale=None, margin_left=margin_left, margin_right=80,
                                margin_bottom=margin_bottom)
                y_col = self.kgrid[tp][prop_name][0]
                plt.xy_plot(x_col=x_col, y_col=y_col)



    def to_csv(self, csv_filename='AMSET_results.csv'):
        """
        this function writes the calculated transport properties to a csv file for convenience.
        :param csv_filename (str):
        :return:
        """
        import csv

        with open(csv_filename, 'w') as csvfile:
            fieldnames = ['type', 'c(cm-3)', 'T(K)', 'overall', 'average'] + \
                         self.elastic_scatterings + self.inelastic_scatterings + ['seebeck']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for tp in ["n", "p"]:
                for c in self.dopings:
                    for T in self.temperatures:
                        row = {'type': tp, 'c(cm-3)': c, 'T(K)': T}
                        for p in ['overall', 'average'] + self.elastic_scatterings + self.inelastic_scatterings:
                            row[p] = sum(self.egrid["mobility"][p][c][T][tp]) / 3
                        row["seebeck"] = sum(self.egrid["seebeck"][c][T][tp]) / 3
                        writer.writerow(row)
                writer.writerow({})  # to more clear separation of n-type and p-type resutls


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # defaults:
    mass = 0.25
    # Model params available:
    #   bs_is_isotropic:
    #   elastic_scatterings:
    #   inelastic_scatterings:
    #   poly_bands: if specified, uses polynomial interpolation for band structure; otherwise default is None and the
    #               model uses Analytical_Bands with the specified coefficient file

    model_params = {"bs_is_isotropic": True, "elastic_scatterings": ["ACD", "IMP", "PIE"],
                    "inelastic_scatterings": ["POP"]
                    # TODO: for testing, remove this part later:
                    # , "poly_bands": [[[[0.0, 0.0, 0.0], [0.0, mass]]]]
    # , "poly_bands" : [[[[0.0, 0.0, 0.0], [0.0, mass]],
    #       [[0.25, 0.25, 0.25], [0.0, mass]],
    #       [[0.15, 0.15, 0.15], [0.0, mass]]]]
                    }
    # TODO: see why poly_bands = [[[[0.0, 0.0, 0.0], [0.0, 0.32]], [[0.5, 0.5, 0.5], [0.0, 0.32]]]] will tbe reduced to [[[[0.0, 0.0, 0.0], [0.0, 0.32]]



    performance_params = {"nkibz": 100, "dE_min": 0.0001, "nE_min": 2,
                          "parallel": True, "BTE_iters": 5}


    ### for PbTe
    # material_params = {"epsilon_s": 44.4, "epsilon_inf": 25.6, "W_POP": 10.0, "C_el": 128.8,
    #                "E_D": {"n": 4.0, "p": 4.0}}
    # cube_path = "../test_files/PbTe/nscf_line"
    # coeff_file = os.path.join(cube_path, "..", "fort.123")

    ### For GaAs
    material_params = {"epsilon_s": 12.9, "epsilon_inf": 10.9, "W_POP": 8.73, "C_el": 139.7,
                       "E_D": {"n": 8.6, "p": 8.6}, "P_PIE": 0.052, "scissor": 0.5818}
    cube_path = "../test_files/GaAs/"
    ### coeff_file = os.path.join(cube_path, "fort.123_GaAs_k23")
    coeff_file = os.path.join(cube_path, "fort.123_GaAs_1099kp")

    ### For Si
    # material_params = {"epsilon_s": 11.7, "epsilon_inf": 11.6, "W_POP": 15.23, "C_el": 190.2,
    #                    "E_D": {"n": 6.5, "p": 6.5}, "P_PIE": 0.15, "scissor": 0.0} #0.5154}
    # cube_path = "../test_files/Si/"
    # coeff_file = os.path.join(cube_path, "Si_fort.123")

    AMSET = AMSET(calc_dir=cube_path, material_params=material_params,
                  model_params=model_params, performance_params=performance_params,
                  # dopings= [-2.7e13], temperatures=[100, 200, 300, 400, 500, 600])
                  # dopings= [-2.7e13], temperatures=[100, 300])
                  # dopings=[-2e15], temperatures=[100, 200, 300, 400, 500, 600, 700, 800])
                  # dopings=[-2e15], temperatures=[300, 400, 500, 600])
                  dopings=[-2e15], temperatures=[300])
                    # dopings=[-2e15], temperatures=[50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    # dopings=[-1e20], temperatures=[300, 600])
    #   dopings = [-1e20], temperatures = [300])
    # AMSET.run(coeff_file=coeff_file, kgrid_tp="coarse")
    # @albalu what exactly does coeff_file store?
    cProfile.run('AMSET.run(coeff_file=coeff_file, kgrid_tp="coarse")')

    AMSET.write_input_files()
    AMSET.to_csv()
    # AMSET.plot()

    AMSET.to_json(kgrid=True, trimmed=True, max_ndata=20, nstart=0)
    # AMSET.to_json(kgrid=True, trimmed=True)
