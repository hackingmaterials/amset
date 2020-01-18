import numpy as np
from monty.serialization import loadfn
from pkg_resources import resource_filename

from scipy.constants import physical_constants

from pymatgen import Spin

bohr_si = physical_constants["Bohr radius"][0]
hartree_si = physical_constants["Hartree energy"][0]

hbar = physical_constants["Planck constant over 2 pi in eV s"][0]
m_e = physical_constants["electron mass"][0]  # in kg
hartree_to_ev = physical_constants["Hartree energy in eV"][0]
ev_to_hartree = 1 / hartree_to_ev
bohr_to_angstrom = bohr_si * 1e10
bohr_to_cm = bohr_si * 100
cm_to_bohr = 1 / (bohr_si * 100)
angstrom_to_bohr = 1 / (bohr_si * 1e10)

gpa_to_au = bohr_si ** 3 / (1e-9 * hartree_si)

A_to_m = 1e-10
m_to_cm = 100.0
e = physical_constants["elementary charge"][0]
k_B = physical_constants["Boltzmann constant in eV/K"][0]
default_small_e = 1.0  # eV/cm the value of this parameter does not matter
over_sqrt_pi = 1 / np.sqrt(np.pi)
sqrt2 = np.sqrt(2)
small_val = 1e-32  # e.g. used for an initial non-zero val
output_width = 69
spin_name = {Spin.up: "spin-up", Spin.down: "spin-down"}
numeric_types = (float, int, np.integer, np.floating)
float_types = (float, np.floating)
int_types = (int, np.integer)

amset_defaults = loadfn(resource_filename("amset", "defaults.yaml"))
