import numpy as np

from scipy.constants import physical_constants

hbar = physical_constants["Planck constant over 2 pi in eV s"][0]
m_e = physical_constants["electron mass"][0]  # in kg
Ry_to_eV = physical_constants["Rydberg constant times hc in eV"][0]
hartree_to_ev = physical_constants["Hartree energy in eV"][0]
A_to_m = 1e-10
m_to_cm = 100.0
A_to_nm = 0.1
e = physical_constants["elementary charge"][0]
k_B = physical_constants["Boltzmann constant in eV/K"][0]
default_small_e = 1.0  # eV/cm the value of this parameter does not matter
over_sqrt_pi = 1 / np.sqrt(np.pi)
small_val = 1e-32  # e.g. used for an initial non-zero val

output_width = 69
