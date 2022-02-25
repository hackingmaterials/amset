"""Module defining constants and default parameters."""
import numpy as np
from monty.serialization import loadfn
from pkg_resources import resource_filename
from pymatgen.electronic_structure.core import Spin
from scipy.constants import physical_constants

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

# distance
bohr_to_m = physical_constants["Bohr radius"][0]
bohr_to_cm = bohr_to_m * 100
bohr_to_nm = bohr_to_m * 1e9
bohr_to_angstrom = bohr_to_m * 1e10
m_to_bohr = 1 / bohr_to_m
cm_to_bohr = 1 / (bohr_to_m * 100)
nm_to_bohr = 1 / (bohr_to_m * 1e9)
angstrom_to_bohr = 1 / bohr_to_angstrom

# energy
hartree_to_joule = physical_constants["Hartree energy"][0]
hartree_to_ev = physical_constants["Hartree energy in eV"][0]
joule_to_hartree = 1 / hartree_to_joule
ev_to_hartree = 1 / hartree_to_ev

# time
au_to_s = physical_constants["atomic unit of time"][0]
s_to_au = 1 / au_to_s

# current
e_si = physical_constants["elementary charge"][0]
ampere_to_au = 1 / physical_constants["atomic unit of current"][0]
coulomb_to_au = ampere_to_au * s_to_au

# misc
boltzmann_si = physical_constants["Boltzmann constant"][0]
boltzmann_ev = physical_constants["Boltzmann constant in eV/K"][0]
boltzmann_au = boltzmann_si * joule_to_hartree
hbar = physical_constants["Planck constant over 2 pi in eV s"][0]
gpa_to_au = joule_to_hartree * 1e9 / m_to_bohr**3

ktol = 1e-5
small_val = 1e-32  # e.g. used for an initial non-zero val
output_width = 69
numeric_types = (float, int, np.integer, np.floating)

spin_name = {Spin.up: "spin-up", Spin.down: "spin-down"}
str_to_spin = {"up": Spin.up, "down": Spin.down}
spin_to_int = {Spin.up: 0, Spin.down: 1}
int_to_spin = {0: Spin.up, 1: Spin.down}

defaults = loadfn(resource_filename("amset", "defaults.yaml"))
