import numpy as np
from monty.serialization import loadfn
from pkg_resources import resource_filename
from scipy.constants import physical_constants

from pymatgen import Spin

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

bohr_to_m = physical_constants["Bohr radius"][0]
hartree_si = physical_constants["Hartree energy"][0]
hbar = physical_constants["Planck constant over 2 pi in eV s"][0]
hartree_to_ev = physical_constants["Hartree energy in eV"][0]
ev_to_hartree = 1 / hartree_to_ev
bohr_to_cm = bohr_to_m * 100
cm_to_bohr = 1 / (bohr_to_m * 100)
bohr_to_angstrom = bohr_to_m * 1e10
angstrom_to_bohr = 1 / bohr_to_angstrom
bohr_to_nm = bohr_to_m * 1e9
nm_to_bohr = 1 / (bohr_to_m * 1e9)
gpa_to_au = bohr_to_m ** 3 / (1e-9 * hartree_si)
e = physical_constants["elementary charge"][0]
k_B = physical_constants["Boltzmann constant in eV/K"][0]
small_val = 1e-32  # e.g. used for an initial non-zero val
output_width = 69
spin_name = {Spin.up: "spin-up", Spin.down: "spin-down"}
numeric_types = (float, int, np.integer, np.floating)

defaults = loadfn(resource_filename("amset", "defaults.yaml"))
