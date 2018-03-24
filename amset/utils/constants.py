from scipy.constants.codata import value as _cd
from math import pi

# some global constants
hbar = _cd('Planck constant in eV s') / (2 * pi)
m_e = _cd('electron mass')  # in kg
Ry_to_eV = 13.605698066
Hartree_to_eV = 2.0 * Ry_to_eV
A_to_m = 1e-10
m_to_cm = 100.0
A_to_nm = 0.1
e = _cd('elementary charge')
k_B = _cd("Boltzmann constant in eV/K")
epsilon_0 = 8.854187817e-12  # dielectric constant in vacuum [C**2/m**2N]
default_small_E = 1.0  # eV/cm the value of this parameter does not matter
dTdz = 10.0  # K/cm
sq3 = 3.0 ** 0.5