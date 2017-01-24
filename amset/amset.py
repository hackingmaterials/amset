# coding: utf-8
from analytical_band_from_BZT import Analytical_bands
from pprint import pprint

import numpy as np
import sys
from pymatgen.io.vasp import Vasprun, Spin
from scipy.constants.codata import value as _cd
from math import pi
import os

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

class AMSETRunner(object):
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
        self.dE_global = 0.01 # in eV, the energy difference threshold before which two energy values are assumed equal
        self.c_list = [1e20] # 1/cm**3 list of carrier concentrations
        self.T_list = [300, 600] # in K, list of temperatures
        self.epsilon_s = 44.360563 # example for PbTe
        self._vrun = {}
        self.max_e_range = 10*k_B*200 # assuming a maximum temperature, we set the maximum energy range after which occupation is zero
        self.path_dir = "../test_files/PbTe_nscf_uniform"
        self.wordy = True

        #TODO: some of the current global constants should be omitted, taken as functions inputs or changed!
        self.example_beta = 0.3288  # 1/nm for n=1e121 and T=300K
        self.N_II = 1e21  # 1/cm**3
        self.read_vrun(path_dir=self.path_dir, filename="vasprun.xml")
        print self.cbm_vbm

    def read_vrun(self, path_dir=".", filename="vasprun.xml"):
        vrun = Vasprun(os.path.join(path_dir, filename))
        self.volume = vrun.final_structure.volume
        self.density = vrun.final_structure.density
        self.lattice_matrix = vrun.lattice_rec.matrix / (2 * pi)
        print self.lattice_matrix
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
        nband_included = {"n": 0, "p": 0}

        for i, type in enumerate(["n", "p"]):
            sgn = (-1)**i
            while abs(min(sgn*bs["bands"]["1"][cbm_vbm[type]["bidx"]+sgn*cbm_vbm[type]["included"]])-sgn*cbm_vbm[type]["energy"])<self.max_e_range:
                cbm_vbm[type]["included"] += 1

        self.cbm_vbm = cbm_vbm

    def init_egrid(self, kgrid, dos_type="simple"):
        """
        take a kgrid (dictionary) that contains the key "energy" and make a simple DOS just by counting how many of that energy value is available
        :param kgrid (turn):
            dos_type (string): options are "simple", ...

        :return: an updated grid that contains the field DOS
        """

        egrid = {
            "n": {"energy": [], "DOS": [], "all_en_flat": []},
            "p": {"energy": [], "DOS": [], "all_en_flat": []}
             }

        for type in ["n", "p"]:
            for en_vec in kgrid[type]["energy"]:
                egrid[type]["all_en_flat"] += en_vec
            egrid[type]["all_en_flat"].sort()

        for type in ["n", "p"]:
            i = 0
            last_is_counted = False
            while i<len(egrid[type]["all_en_flat"])-1:
                sum = egrid[type]["all_en_flat"][i]
                counter = 1.0
                while i<len(egrid[type]["all_en_flat"])-1 and abs(egrid[type]["all_en_flat"][i]-egrid[type]["all_en_flat"][i+1]) < self.dE_global:
                    counter += 1
                    sum += egrid[type]["all_en_flat"][i+1]
                    i+=1
                    if i+1 == len(egrid[type]["all_en_flat"])-1:
                        last_is_counted = True
                egrid[type]["energy"].append(sum/counter)
                if dos_type=="simple":
                    egrid[type]["DOS"].append(counter/len(egrid[type]["all_en_flat"]))
                i+=1
            if not last_is_counted:
                egrid[type]["energy"].append(egrid[type]["all_en_flat"][-1])
                if dos_type == "simple":
                    egrid[type]["DOS"].append(1.0 / len(egrid[type]["all_en_flat"]))

            egrid[type]["nu_II"] = np.array([ [0.0, 0.0, 0.0] for i in range(len(egrid[type]["energy"]))])
        # for i in range(len(kgrid["kpoints"])):
        #     for j in range(len(kgrid["kpoints"])):
        #         if abs(kgrid["energy"][i]-kgrid["energy"][j]) < dE_global:
        #             kgrid["DOS"][i] += 1
        #     kgrid["DOS"][i] /= len(kgrid["kpoints"])
        return egrid

    def G(self, kgrid, type, ib, ik, ib_prime, ik_prime, X):
        """
        The overlap integral betweek vectors k and k'
        :param kgrid (dict):
        :param ik (int): index of vector k in kgrid
        :param ik_prime (int): index of vector k' in kgrid
        :param X (float): cosine of the angle between vectors k and k'
        :return: overlap integral
        """
        return kgrid[type]["a"][ib][ik] * kgrid[type]["a"][ib_prime][ik_prime]+ \
               kgrid[type]["c"][ib][ik] * kgrid[type]["c"][ib_prime][ik_prime]


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

    def generate_kgrid(self, center_kpt, coeff_file, kgrid_type="coarse"):
        """
        Function to generate a grid of k-points around the center_kpt and generate en, v, and effective mass arround that

        :param center_kpt:
        :param coeff_file:
        :param cbm_bidx:
        :param grid_type:
        :return:
        """

        # Total range around the center k-point
        rang = 0.13
        if kgrid_type == "coarse":
            nstep = 2

        step = rang/nstep

        kpts = [[0, 0, 0] for i in range((nstep+1)**3)]
        counter = 0
        for i in range(nstep+1):
            for j in range(nstep+1):
                for k in range(nstep+1):
                    kpts[counter] = [center_kpt[0] - rang / 2.0 + 0.02*(-1)**j  + i * step,
                                     center_kpt[1] - rang / 2.0 + 0.02*(-1)**i  +j * step,
                                     center_kpt[2] - rang / 2.0 + k * step]
                    counter += 1
        if center_kpt not in kpts:
            kpts.append(center_kpt)

        #TODO remove this later because the center point doesn't have to be there:
        kpts.append(center_kpt)


        kpts = np.array(kpts)
        print(len(kpts))
        # initialize the kgrid
        kgrid = {
                "kpoints": kpts,
                # "actual kpoints": np.dot(kpts-center_kpt, self.lattice_matrix)*2*pi*1/A_to_nm, # actual k-points in 1/nm
                "actual kpoints": np.dot(kpts, self.lattice_matrix)*2*pi*1/A_to_nm, # actual k-points in 1/nm
                "n": {},
                "p": {}
                }
        actual_kcenter = np.dot(center_kpt, self.lattice_matrix)*2*pi*1/A_to_nm
        kgrid["distance"] = [np.linalg.norm(actual_kpoint-actual_kcenter) for actual_kpoint in kgrid["actual kpoints"]]
        # kgrid["distance"] = [np.linalg.norm(actual_kpoint) for actual_kpoint in kgrid["actual kpoints"]]
        for type in ["n", "p"]:
            for property in ["energy", "a", "c"]:
                kgrid[type][property] = [ [0.0 for i in range(len(kpts))] for j in range(self.cbm_vbm[type]["included"])]
            for property in ["velocity", "nu_II"]:
                kgrid[type][property] = \
                [ np.array([[0.0, 0.0, 0.0] for i in range(len(kpts))]) for j in range(self.cbm_vbm[type]["included"])]
            kgrid[type]["effective mass"] = \
                [ np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] for i in range(len(kpts))]) for j in range(self.cbm_vbm[type]["included"])]

        analytical_bands = Analytical_bands(coeff_file=coeff_file)
        for i, type in enumerate(["n", "p"]):
            sgn = (-1)**i
            for ib in range(self.cbm_vbm[type]["included"]):
                engre, latt_points, nwave, nsym, nsymop, symop, br_dir = analytical_bands.get_engre(iband=self.cbm_vbm[type]["bidx"]+sgn*ib)
                for idx in range(len(kgrid["kpoints"])):
                    energy, de, dde = analytical_bands.get_energy(kgrid["kpoints"][idx], engre, latt_points, nwave, nsym,
                                                              nsymop, symop, br_dir)

                    kgrid[type]["energy"][ib][idx] = energy * Ry_to_eV
                    # kgrid[type]["velocity"][ib][idx] = abs( de/hbar * A_to_m * m_to_cm * Ry_to_eV ) # to get v in units of cm/s
                    kgrid[type]["velocity"][ib][idx] = de/hbar * A_to_m * m_to_cm * Ry_to_eV # to get v in units of cm/s
                    # TODO: what's the implication of negative group velocities? check later after scattering rates are calculated
                    #TODO: should I have de*2*pi for the group velocity and dde*(2*pi)**2 for effective mass?
                    kgrid[type]["effective mass"][ib][idx] = hbar ** 2 / (
                    dde * 4 * pi ** 2) / m_e / A_to_m ** 2 * e * Ry_to_eV  # m_tensor: the last part is unit conversion
                    kgrid[type]["a"][ib][idx] = 1.0

        if True: # TODO: later add a more sophisticated DOS function, if developed
            egrid = self.init_egrid(kgrid, dos_type = "simple")
        else:
            pass

        for type in ["n", "p"]:
            for idx in range(len(kgrid["kpoints"])):
                for ib in range(len(kgrid[type]["energy"])):

                    # preparation for integration of II scattering
                    X_and_idx = []
                    for ib_prime in range(len(kgrid[type]["energy"])):
                        for ik_prime in range(len(kgrid["kpoints"])):
                            if ik_prime == idx:
                                continue
                            if (kgrid[type]["energy"][ib][idx]-kgrid[type]["energy"][ib_prime][ik_prime]) < self.dE_global:
                                k = kgrid["actual kpoints"][idx]
                                k_prime = kgrid["actual kpoints"][ik_prime]
                                X = self.cos_angle(k,k_prime)
                                X_and_idx.append((X, ib_prime, ik_prime))
                    X_and_idx.sort()
                    print X_and_idx

                    # integrate over X (the angle between the k vectors)
                    sum = np.array([0.0, 0.0, 0.0])
                    for i in range(len(X_and_idx)-1):
                        DeltaX = X_and_idx[i+1][0]-X_and_idx[i][0]
                        for alpha in range(3):
                        # for alpha in range(0):
                            dum = 0
                            for j in range(2):
                            # if True:
                                # extract the indecies
                                X, ib_prime, ik_prime = X_and_idx[i+j]
                                print "cosine of angle:"
                                print X
                                if kgrid[type]["velocity"][ib_prime][ik_prime][alpha] < 1:
                                    continue
                                dum += (1 - X) * self.G(kgrid, type, ib, idx, ib_prime, ik_prime, X) ** 2 * np.linalg.norm(k_prime) ** 2 / \
                                ((np.linalg.norm(k - k_prime) ** 2 + self.example_beta ** 2) ** 2 * kgrid[type]["velocity"][ib_prime][ik_prime][alpha])

                            dum /= 2 # the average of points i and i+1 to integrate via the trapezoidal rule
                            sum[alpha] += dum*DeltaX # If there are two points with the same X, DeltaX==0 so no duplicates
                            print "here"
                            print sum
                            print (1-X)
                            print np.linalg.norm(k_prime) ** 2
                            print 1/(np.linalg.norm(k - k_prime) ** 2 + self.example_beta ** 2)** 2
                            print 1/kgrid[type]["velocity"][ib_prime][ik_prime][alpha]
                            print DeltaX

                    print sum
                    # fix this! there are scattering rates close to 1e24!!!! check with bands included=1 (override!) and see what happens becaues before I was getting 1e12!
                    # TODO: the units seem to be correct but I still get 1e24 order, perhaps divide this expression by that of aMoBT to see what differs.
                    kgrid[type]["nu_II"][ib][idx] = sum * e ** 4 * self.N_II /\
                        (2 * pi * self.epsilon_s ** 2 * epsilon_0 ** 2 * hbar ** 2) * 3.89564386e27 # coverted to 1/s

        for type in ["n", "p"]:
            for ie, en in enumerate(egrid[type]["energy"]):
                N = 0 # total number of instances with the same energy
                for idx in range(len(kgrid["kpoints"])):
                    for ib in range(len(kgrid[type]["energy"])):
                        if abs(kgrid[type]["energy"][ib][idx] - en) < self.dE_global:
                            egrid[type]["nu_II"][ie] += kgrid[type]["nu_II"][ib][idx]
                            N += 1
                egrid[type]["nu_II"][ie] /= N

        if self.wordy:
            pprint(egrid)
            pprint(kgrid)

        # for idx, dist
        #TODO: make the kgrid fill it out
        return kgrid

if __name__ == "__main__":

    #user inpits
    kpts = np.array([[0.5, 0.5, 0.5]])


    # test
    AMSET = AMSETRunner()
    kgrid = AMSET.generate_kgrid(center_kpt=[0.5, 0.5, 0.5], coeff_file=coeff_file, kgrid_type="coarse")

    # pprint(kgrid)