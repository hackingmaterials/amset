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
print hbar
print e
# some temporary global constants as inputs
coeff_file = 'fort.123'

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


    def __init__(self):
        self.dE_global = 0.0001 # in eV, the energy difference threshold before which two energy values are assumed equal
        self.dopings = [1e20] # 1/cm**3 list of carrier concentrations
        self.temperatures = [300, 600] # in K, list of temperatures
        self.epsilon_s = 44.360563 # example for PbTe
        self._vrun = {}
        self.max_e_range = 10*k_B*300 # assuming a maximum temperature, we set the max energy range after which occupation is zero
        self.path_dir = "../test_files/PbTe_nscf_uniform/nscf_line"
        self.wordy = True

        #TODO: some of the current global constants should be omitted, taken as functions inputs or changed!
        self.example_beta = 0.3288  # 1/nm for n=1e121 and T=300K
        self.N_II = 1e20  # 1/cm**3
        self.soc = False
        self.read_vrun(path_dir=self.path_dir, filename="vasprun.xml")



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
        bs = bs.as_dict()

        # print vrun.actual_kpoints[vbm["kpoint_index"][0]]
        # print vrun.actual_kpoints

        for i, type in enumerate(["n", "p"]):
            sgn = (-1)**i
            while abs(min(sgn*bs["bands"]["1"][cbm_vbm[type]["bidx"]+sgn*cbm_vbm[type]["included"]])-sgn*cbm_vbm[type]["energy"])<self.max_e_range:
                cbm_vbm[type]["included"] += 1

        # TODO: change this later if the band indecies are fixed in Analytical_band class
        cbm_vbm["p"]["bidx"] += 1
        cbm_vbm["n"]["bidx"] = cbm_vbm["p"]["bidx"] + 1

        self.cbm_vbm = cbm_vbm



    @staticmethod
    def f0(E, fermi, T):
        return 1 / (1 + np.exp((E - fermi) / (k_B * T)))



    def init_egrid(self, dos_type="simple"):
        """
        :param
            dos_type (string): options are "simple", ...

        :return: an updated grid that contains the field DOS
        """

        self.egrid = {
            "n": {"energy": [], "DOS": [], "all_en_flat": []},
            "p": {"energy": [], "DOS": [], "all_en_flat": []}
             }

        # reshape energies of all bands to one vector:
        for type in ["n", "p"]:
            for en_vec in self.kgrid[type]["energy"]:
                self.egrid[type]["all_en_flat"] += en_vec
            self.egrid[type]["all_en_flat"].sort()

        # setting up energy grid and DOS:
        for type in ["n", "p"]:
            i = 0
            last_is_counted = False
            while i<len(self.egrid[type]["all_en_flat"])-1:
                sum = self.egrid[type]["all_en_flat"][i]
                counter = 1.0
                while i<len(self.egrid[type]["all_en_flat"])-1 and abs(self.egrid[type]["all_en_flat"][i]-self.egrid[type]["all_en_flat"][i+1]) < self.dE_global:
                    counter += 1
                    sum += self.egrid[type]["all_en_flat"][i+1]
                    if i+1 == len(self.egrid[type]["all_en_flat"])-1:
                        last_is_counted = True
                    i+=1
                self.egrid[type]["energy"].append(sum/counter)
                if dos_type=="simple":
                    self.egrid[type]["DOS"].append(counter/len(self.egrid[type]["all_en_flat"]))
                i+=1
            if not last_is_counted:
                self.egrid[type]["energy"].append(self.egrid[type]["all_en_flat"][-1])
                if dos_type == "simple":
                    self.egrid[type]["DOS"].append(1.0 / len(self.egrid[type]["all_en_flat"]))

        # calculate Fermi levels:
        self.egrid["fermi"] = {c: {T: 0.0 for T in self.temperatures} for c in self.dopings}
        for c in self.dopings:
            for T in self.temperatures:
                self.egrid["fermi"][c][T] = self.find_fermi(c, T)



    def G(self, type, ib, ik, ib_prime, ik_prime, X):
        """
        The overlap integral betweek vectors k and k'
        :param ik (int): index of vector k in kgrid
        :param ik_prime (int): index of vector k' in kgrid
        :param X (float): cosine of the angle between vectors k and k'
        :return: overlap integral
        """
        return self.kgrid[type]["a"][ib][ik] * self.kgrid[type]["a"][ib_prime][ik_prime]+ \
               self.kgrid[type]["c"][ib][ik] * self.kgrid[type]["c"][ib_prime][ik_prime]



    def cos_angle(self, v1, v2):
        """
        Args:
            v1, v2 (np.array): vectors
        return:
            the cosine of the angle between twp numpy vectors: v1 and v2"""
        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 1.0  # In case of the two points is the origin, we assume 0 degree; i.e. no scattering because of 1-X term
        else:
            return np.dot(v1, v2) / (norm_v1 * norm_v2)



    def generate_kgrid(self, coeff_file, kgrid_type="coarse"):
        """
        Function to generate a grid of k-points and generate en, v, and effective mass on that

        :param center_kpt:
        :param coeff_file:
        :param cbm_bidx:
        :param grid_type:
        :return:
        """

        if kgrid_type=="coarse":
            nstep = 3
        # k = list(np.linspace(0.25, 0.75-0.5/nstep, nstep))
        kx = list(np.linspace(0.25, 0.75, nstep))
        ky = kz = kx
        # ky = list(np.linspace(0.26, 0.76, nstep))
        # kz = list(np.linspace(0.24, 0.74, nstep))
        kpts = [[x, y, z] for x in kx for y in ky for z in kz]


        # # Total range around the center k-point
        # rang = 0.14
        # if kgrid_type == "coarse":
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



        #TODO remove this later because the center point doesn't have to be there:
        # if center_kpt not in kpts:
        #     kpts.append(center_kpt)


        # kpts = np.array(kpts)
        print(len(kpts))
        # initialize the kgrid
        self.kgrid = {
                "kpoints": kpts,
                # "actual kpoints": np.dot(kpts-center_kpt, self._lattice_matrix)*2*pi*1/A_to_nm, # actual k-points in 1/nm
                "actual kpoints": np.dot(np.array(kpts), self._lattice_matrix)*2*pi*1/A_to_nm, # actual k-points in 1/nm
                "n": {},
                "p": {}
                }
        # actual_kcenter = np.dot(center_kpt, self._lattice_matrix)*2*pi*1/A_to_nm
        # self.kgrid["distance"] = [np.linalg.norm(actual_kpoint-actual_kcenter) for actual_kpoint in self.kgrid["actual kpoints"]]
        for type in ["n", "p"]:
            for property in ["energy", "a", "c"]:
                self.kgrid[type][property] = [ [0.0 for i in range(len(kpts))] for j in range(self.cbm_vbm[type]["included"])]
            for property in ["velocity"]:
                self.kgrid[type][property] = \
                np.array([ [[0.0, 0.0, 0.0] for i in range(len(kpts))] for j in range(self.cbm_vbm[type]["included"])])
            self.kgrid[type]["effective mass"] = \
                [ np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] for i in range(len(kpts))]) for j in range(self.cbm_vbm[type]["included"])]

        analytical_bands = Analytical_bands(coeff_file=coeff_file)
        for i, type in enumerate(["n", "p"]):
            sgn = (-1)**i
            for ib in range(self.cbm_vbm[type]["included"]):
                engre, latt_points, nwave, nsym, nsymop, symop, br_dir = analytical_bands.get_engre(iband=self.cbm_vbm[type]["bidx"]+sgn*ib)
                for idx in range(len(self.kgrid["kpoints"])):
                    energy, de, dde = analytical_bands.get_energy(self.kgrid["kpoints"][idx], engre, latt_points, nwave, nsym,
                                                              nsymop, symop, br_dir)

                    self.kgrid[type]["energy"][ib][idx] = energy * Ry_to_eV
                    # self.kgrid[type]["velocity"][ib][idx] = abs( de/hbar * A_to_m * m_to_cm * Ry_to_eV ) # to get v in units of cm/s
                    self.kgrid[type]["velocity"][ib][idx] = de/hbar * A_to_m * m_to_cm * Ry_to_eV # to get v in units of cm/s
                    # TODO: what's the implication of negative group velocities? check later after scattering rates are calculated
                    # TODO: actually using abs() for group velocities mostly increase nu_II values at each energy
                    #TODO: should I have de*2*pi for the group velocity and dde*(2*pi)**2 for effective mass?
                    self.kgrid[type]["effective mass"][ib][idx] = hbar ** 2 / (
                    dde * 4 * pi ** 2) / m_e / A_to_m ** 2 * e * Ry_to_eV  # m_tensor: the last part is unit conversion
                    self.kgrid[type]["a"][ib][idx] = 1.0
            # Match the CBM/VBM energy values to those obtained from the coefficients file rather than vasprun.xml
            self.cbm_vbm[type]["energy"] = sgn*min(sgn*np.array(self.kgrid[type]["energy"][0]))

        print self.cbm_vbm
        if True: # TODO: later add a more sophisticated DOS function, if developed
            self.init_egrid(dos_type = "simple")
        else:
            pass

        # Ionized Impurity
        for type in ["n", "p"]:
            self.egrid[type]["nu_II"] = {c:{T: np.array([[0.0, 0.0, 0.0] for i in range(len(self.egrid[type]["energy"]))]) for T in self.temperatures} for c in self.dopings}
            # self.egrid[type]["nu_II"] = np.array([[0.0, 0.0, 0.0] for i in range(len(self.egrid[type]["energy"]))])
            self.kgrid[type]["nu_II"] = \
                np.array([[[0.0, 0.0, 0.0] for i in range(len(kpts))] for j in range(self.cbm_vbm[type]["included"])])
            for idx in range(len(self.kgrid["kpoints"])):
                for ib in range(len(self.kgrid[type]["energy"])):

                    # preparation for integration of II scattering
                    X_and_idx = []
                    for ib_prime in range(len(self.kgrid[type]["energy"])):
                        for ik_prime in range(len(self.kgrid["kpoints"])):
                            # We might not need this because of the (1-X) terms in scattering going to zero when X==1
                            # if ik_prime == idx and ib == ib_prime:
                            #     continue
                            if abs(self.kgrid[type]["energy"][ib][idx]-self.kgrid[type]["energy"][ib_prime][ik_prime]) < self.dE_global:
                                k = self.kgrid["actual kpoints"][idx]
                                k_prime = self.kgrid["actual kpoints"][ik_prime]
                                X = self.cos_angle(k,k_prime)
                                X_and_idx.append((X, ib_prime, ik_prime))
                    X_and_idx.sort()
                    # print "X_and_idx"
                    # print X_and_idx

                    # integrate over X (the angle between the k vectors)
                    sum = np.array([0.0, 0.0, 0.0])
                    for i in range(len(X_and_idx)-1):
                        DeltaX = X_and_idx[i+1][0]-X_and_idx[i][0]
                        if DeltaX == 0.0:
                            continue
                        for alpha in range(3):
                        # for alpha in range(0):
                            dum = 0
                            for j in range(2):
                            # if True:
                                # extract the indecies
                                X, ib_prime, ik_prime = X_and_idx[i+j]
                                # print "cosine of angle:"
                                # print X
                                # if self.kgrid[type]["velocity"][ib_prime][ik_prime][alpha] < 1:
                                #     continue
                                # if X > 0.5:
                                dum += (1 - X) * self.G(type, ib, idx, ib_prime, ik_prime, X) ** 2 * np.linalg.norm(k-k_prime) ** 2 / \
                                        ((np.linalg.norm(k - k_prime) ** 2 + self.example_beta ** 2) ** 2 * abs(self.kgrid[type]["velocity"][ib_prime][ik_prime][alpha])) # We take the absolute value of the velocity as the scattering depends on the velocity itself and not the direction

                            dum /= 2 # the average of points i and i+1 to integrate via the trapezoidal rule
                            sum[alpha] += dum*DeltaX # If there are two points with the same X, DeltaX==0 so no duplicates
                            # print "here"
                            # print sum
                            # print (1-X)
                            # print np.linalg.norm(k_prime) ** 2
                            # print 1/(np.linalg.norm(k - k_prime) ** 2 + self.example_beta ** 2) ** 2
                            # print 1/self.kgrid[type]["velocity"][ib_prime][ik_prime][alpha]
                            # print DeltaX

                    # print sum
                    # fix this! there are scattering rates close to 1e24!!!! check with bands included=1 (override!) and see what happens becaues before I was getting 1e12!
                    # TODO: the units seem to be correct but I still get 1e24 order, perhaps divide this expression by that of aMoBT to see what differs.
                    self.kgrid[type]["nu_II"][ib][idx] = abs(sum) * e ** 4 * self.N_II /\
                        (2 * pi * self.epsilon_s ** 2 * epsilon_0 ** 2 * hbar ** 2) * 3.89564386e27 # coverted to 1/s

        c = self.dopings[0]
        T = self.temperatures[0]
        # Map from k-space to energy-space
        for type in ["n", "p"]:
            for ie, en in enumerate(self.egrid[type]["energy"]):
                N = 0 # total number of instances with the same energy
                for idx in range(len(self.kgrid["kpoints"])):
                    for ib in range(len(self.kgrid[type]["energy"])):
                        if abs(self.kgrid[type]["energy"][ib][idx] - en) < self.dE_global:
                            self.egrid[type]["nu_II"][c][T][ie] += self.kgrid[type]["nu_II"][ib][idx]
                            N += 1
                self.egrid[type]["nu_II"][c][T][ie] /= N

        if self.wordy:
            pprint(self.egrid)
            pprint(self.kgrid)

        with open("kgrid.txt", "w") as fout:
            pprint(self.kgrid, stream=fout)
        with open("egrid.txt", "w") as fout:
            pprint(self.egrid, stream=fout)

        self.grid = {"kgrid": self.kgrid,
                     "egrid": self.egrid}

        # for idx, dist
        #TODO: make the kgrid fill it out



    def find_fermi(self, c, T, tolerance=0.001, tolerance_loose=0.03,
                   interpolation_nsteps = 100 , step0 = 0.01, nsteps = 300):
        """
        To find the Fermi level at a carrier concentration and temperature at kgrid (i.e. band structure, DOS, etc)
        :param c (float): The doping concentration; c < 0 indicate n-type (i.e. electrons) and c > 0 for p-type
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
        maxitr = 10 # essentially the number of floating points in accuracy
        iter = 0
        if c < 0:
            actual_type = "n"
        else:
            actual_type = "p"
        temp_doping = {"n": 0.0, "p": 0.0}
        fermi0 = self.cbm_vbm[actual_type]["energy"]
        print fermi0
        fermi_selected = fermi0

        # iterate around the CBM/VBM with finer and finer steps to find the Fermi level with a matching doping
        # for iter in range(maxitr):
        while (relative_error > tolerance) and (iter<maxitr):
            step = step0 / 10**iter
            for fermi in np.linspace(fermi0-nsteps*step,fermi0+nsteps*step, nsteps*2):
                for j, type in enumerate(["n", "p"]):
                    sgn = (-1)**(j+1)
                    integral = 0.0
                    for ie in range(len(self.egrid[type])-1):
                        E = self.egrid[type]["energy"][ie]
                        dE = abs(self.egrid[type]["energy"][ie+1] - E)/interpolation_nsteps
                        dS = (self.egrid[type]["DOS"][ie+1] - self.egrid[type]["DOS"][ie])/interpolation_nsteps
                        for i in range(interpolation_nsteps):
                            integral += dE*(self.egrid[type]["DOS"][ie]+i*dS)*(j-sgn*self.f0(E+i*dE,fermi,T))
                    temp_doping[type] = sgn * abs(integral/self.volume / (A_to_m*m_to_cm)**3)
                if abs(temp_doping["n"] + temp_doping["p"] - c) < abs(calc_doping - c):
                    calc_doping = temp_doping["n"] + temp_doping["p"]
                    fermi_selected = fermi
            fermi0 = fermi_selected
            iter += 1
        print fermi_selected
        # evaluate the calculated carrier concentration (Fermi level)
        relative_error = abs(calc_doping - c)/abs(c)
        if relative_error > tolerance and relative_error <= tolerance_loose:
            warnings.warn("The calculated concentration {} is not accurate compared to {}; results may be unreliable"
                          .format(calc_doping, c))
        elif relative_error > tolerance_loose:
            raise ValueError("The calculated concentration {} is more than {}% away from {}; possible cause may low band gap, high temperature, small nsteps, etc; AMSET stops now!"
                             .format(calc_doping, tolerance_loose*100, c))
        return fermi_selected



    def inverse_screening_length(self, type, fermi, T, interpolation_nsteps = 100):
        integral = 0.0
        for ie in range(len(self.egrid[type]) - 1):
            E = self.egrid[type]["energy"][ie]
            dE = abs(self.egrid[type]["energy"][ie + 1] - E) / interpolation_nsteps
            dS = (self.egrid[type]["DOS"][ie + 1] - self.egrid[type]["DOS"][ie]) / interpolation_nsteps
            for i in range(interpolation_nsteps):
                integral += dE*(self.egrid[type]["DOS"][ie] + i*dS)*self.f0(E+i*dE, fermi, T)*(1-self.f0(E+i*dE, fermi, T))

def to_json(self, filename="grid.json"):
        del(self.grid["kgrid"]["actual kpoints"])
        remove_list = ["effective mass"]
        for type in ["n", "p"]:
            for rm in remove_list:
                del(self.grid["kgrid"][type][rm])
        d = {
            "kpoints": self.grid["kgrid"]["kpoints"],
            "k-nu_II": {"n": self.grid["kgrid"]["n"]["nu_II"], "p": self.grid["kgrid"]["p"]["nu_II"]},
            "e-nu_II": {"n": self.grid["egrid"]["n"]["nu_II"], "p": self.grid["egrid"]["p"]["nu_II"]}
             }
        with open(filename, 'w') as fp:
            json.dump(self.grid, fp,sort_keys = True, indent = 4, ensure_ascii=False)

if __name__ == "__main__":

    # test
    AMSET = AMSET()
    AMSET.generate_kgrid(coeff_file=coeff_file, kgrid_type="coarse")
    # AMSET.to_json()
