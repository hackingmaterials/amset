# coding: utf-8
from amset.analytical_band_from_BZT import Analytical_bands

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
        self.c_list = [1e21] # 1/cm**3 list of carrier concentrations
        self.T_list = [300, 600] # in K, list of temperatures
        self.epsilon_s = 446.360563 # example for PbTe
        self._vrun = {}
        self.max_e_range = 10*k_B*1300 # assuming a maximum temperature, we set the maximum energy range after which occupation is zero

        example_beta = 0.3288  # 1/nm for n=1e121 and T=300K
        N_II = 1e21  # 1/cm**3

    def read_vrun(self, path_dir=".", filename="vasprun.xml"):
        # vrun = Vasprun('../test_files/PbTe_nscf_uniform/vasprun.xml')
        vrun = Vasprun(os.path.join(path_dir, filename))
        self.volume = vrun.final_structure.volume
        self.density = vrun.final_structure.density
        self.lattice_matrix = vrun.lattice_rec.matrix / (2 * pi)
        bs = vrun.get_band_structure()
        self.vbm = bs.get_vbm()
        self.cbm = bs.get_cbm()
        self.vbm_bindex = self.vbm["band_index"][Spin.up][-1]
        # vbm_kindex = vbm["kpoint_index"][0]
        self.cbm_bindex = self.cbm["band_index"][Spin.up][0]
        # cbm_kindex = cbm["kpoint_index"][0]
        # vbm_k = vrun.actual_kpoints[vbm_kindex]
        # cbm_k = vrun.actual_kpoints[cbm_kindex]
        print('index of last valence band = ' + str(self.vbm_bindex))
        print('index of first conduction band = ' + str(self.cbm_bindex))
        print "vbm"
        print self.vbm

    def init_energy_grid(self, grid, dos_type="simple"):
        """
        take a grid (dictionary) that contains the key "energy" and make a simple DOS just by counting how many of that energy value is available
        :param grid (turn):
            dos_type (string): options are "simple", ...

        :return: an updated grid that contains the field DOS
        """
        energy = grid["energy"]
        energy.sort()
        energy_grid = {"energy": [], "DOS": []}

        i = 0
        last_is_counted = False
        while i<len(energy)-1:
            sum = energy[i]
            counter = 1.0
            while i<len(energy)-1 and abs(energy[i]-energy[i+1]) < self.dE_global:
                counter += 1
                sum += energy[i+1]
                i+=1
                if i+1 == len(energy)-1:
                    last_is_counted = True
            energy_grid["energy"].append(sum/counter)
            if dos_type=="simple":
                energy_grid["DOS"].append(counter/len(grid["kpoints"]))
            i+=1
        if not last_is_counted:
            energy_grid["energy"].append(energy[-1])
            if dos_type == "simple":
                energy_grid["DOS"].append(1.0 / len(grid["kpoints"]))
        # for i in range(len(grid["kpoints"])):
        #     for j in range(len(grid["kpoints"])):
        #         if abs(grid["energy"][i]-grid["energy"][j]) < dE_global:
        #             grid["DOS"][i] += 1
        #     grid["DOS"][i] /= len(grid["kpoints"])
        return energy_grid

    def G(self, grid, ik, ik_prime, X):
        """
        The overlap integral betweek vectors k and k'
        :param grid (dict):
        :param ik (int): index of vector k in grid
        :param ik_prime (int): index of vector k' in grid
        :param X (float): cosine of the angle between vectors k and k'
        :return: overlap integral
        """
        return grid["a"][ik] * grid["a"][ik_prime] + grid["c"][ik] * grid["c"][ik_prime] * X


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

    def generate_grid(self, center_kpt, coeff_file, iband, grid_type="coarse"):
        """
        Function to generate a grid of k-points around the center_kpt and generate en, v, and effective mass arround that

        :param center_kpt:
        :param coeff_file:
        :param cbm_bidx:
        :param grid_type:
        :return:
        """

        self.read_vrun(path_dir="../test_files/PbTe_nscf_uniform", filename="vasprun.xml")

        # Total range around the center k-point
        rang = 0.15
        if grid_type == "coarse":
            nstep = 3

        step = rang/nstep

        kpts = [[0, 0, 0] for i in range((nstep+1)**3)]
        counter = 0
        for i in range(nstep+1):
            for j in range(nstep+1):
                for k in range(nstep+1):
                    kpts[counter] = [center_kpt[1] - rang / 2.0 + j * step,
                                     center_kpt[1] - rang / 2.0 + j * step,
                                     center_kpt[2] - rang / 2.0 + k * step]
                    counter += 1
        if center_kpt not in kpts:
            kpts.append(center_kpt)

        #TODO remove this later:
        kpts.append(center_kpt)


        kpts = np.array(kpts)
        print len(kpts)

        # initialize the grid
        grid = {
                "kpoints": kpts,
                "actual kpoints": np.dot(kpts, self.lattice_matrix)*2*pi*1/A_to_nm, # actual k-points in 1/nm
                "distance": [0.0 for i in range(len(kpts))], # in 1/nm

                "energy": [0.0 for i in range(len(kpts))],
                "nu_II": np.array([[0.0, 0.0, 0.0] for i in range(len(kpts))] ),
                "a": [0.0 for i in range(len(kpts))], # wavefunction admixture of s-like orbitals, a**2+c**2=1
                "c": [0.0 for i in range(len(kpts))],
                "velocity": np.array([[0.0, 0.0, 0.0] for i in range(len(kpts))] ),
                "effective mass": np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] for i in range(len(kpts))])
                }

        analytical_bands = Analytical_bands(coeff_file=coeff_file)
        engre, latt_points, nwave, nsym, nsymop, symop, br_dir = analytical_bands.get_engre(iband=iband)

        for idx in range(len(grid["kpoints"])):
            grid["distance"][idx] = np.linalg.norm((grid["kpoints"][idx]-center_kpt)*self.lattice_matrix*2*pi)*1/A_to_nm
            energy, de, dde = analytical_bands.get_energy(grid["kpoints"][idx], engre, latt_points, nwave, nsym, nsymop, symop, br_dir)
            grid["energy"][idx] = energy * Ry_to_eV
            #TODO: what's the implication of negative group velocities? check later after scattering rates are calculated
            #TODO: should I have de*2*pi for the group velocity and dde*(2*pi)**2 for effective mass?
            grid["velocity"][idx] = de /hbar*A_to_m*m_to_cm * Ry_to_eV # to get v in units of cm/s
            # grid["effective mass"][idx] = np.dot(hbar**2/dde/m_e / A_to_m**2*e*Ry_to_eV, np.linalg.inv(2*pi*self.lattice_matrix)) # m_tensor: the last part is unit conversion
            grid["effective mass"][idx] = hbar ** 2 /(dde*4*pi**2) / m_e / A_to_m ** 2 * e * Ry_to_eV  # m_tensor: the last part is unit conversion
            grid["a"][idx] = 1

        if True: # TODO: later add a more sophisticated DOS function, if developed
            energy_grid = self.init_energy_grid(grid, dos_type = "simple")
        else:
            pass
        print energy_grid

        for idx in range(len(grid["kpoints"])):
            sum = np.array([0.0, 0.0, 0.0])

            # preparation for integration of II scattering
            X_and_idx = []
            for ik_prime in range(len(grid["kpoints"])):
                if ik_prime == idx:
                    continue
                if (grid["energy"][idx]-grid["energy"][ik_prime]) < self.dE_global:
                    k = grid["actual kpoints"][idx]
                    k_prime = grid["actual kpoints"][ik_prime]
                    X = self.cos_angle(k,k_prime)
                    X_and_idx.append((X, ik_prime))

            # integrate over X (the angle between the k vectors)
            X_and_idx.sort()
            for i in range(len(X_and_idx)-1):
                DeltaX = X_and_idx[i+1][0]-X
                for alpha in range(3):
                    dum = 0
                    for j in range(2):
                        X = X_and_idx[i+j][0]
                        ik_prime = X_and_idx[i+j][1]
                        dum+= (1 - X) * self.G(grid, idx, ik_prime, X) ** 2 * np.linalg.norm(k_prime) ** 2 / \
                        ((np.linalg.norm(k - k_prime) ** 2 + self.example_beta ** 2) ** 2 * grid["velocity"][ik_prime][alpha])
                    dum /= 2
                    sum[alpha] += dum/2*DeltaX # If there are two points with the same X, DeltaX==0 so no duplicates



            grid["nu_II"][idx] = sum * e**4*self.N_II/(2*pi*self.epsilon_s**2*epsilon_0**2*hbar**2) * 3.89564386e27 # coverted to 1/s
        #TODO: make the grid fill it out
        return grid

if __name__ == "__main__":

    #user inpits
    kpts = np.array([[0.5, 0.5, 0.5]])
    vbm_bidx = 10
    cbm_bidx = 11


    # test
    AMSET = AMSETRunner()
    grid = AMSET.generate_grid(center_kpt=[0.5, 0.5, 0.5], coeff_file=coeff_file, iband=cbm_bidx, grid_type="coarse")
    pprint(grid)