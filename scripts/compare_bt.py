import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from pymatgen.io.vasp.outputs import Vasprun
from amset.interpolate.boltztrap1 import BoltzTraP1Interpolater
from amset.interpolate.boltztrap2 import BoltzTraP2Interpolater

"""
This script is to compare the energy, velocity and effective mass calculated 
from the band structures interpolated via BoltzTraP1 vs. BoltzTraP2 to check
their consistency. 
"""


vr = Vasprun("vasprun.xml")
bs = vr.get_band_structure()
kpts = np.array(vr.actual_kpoints)
print("band gap: {:.3f}".format(bs.get_band_gap()['energy']))

print("loading BoltzTraP1")
bt1 = BoltzTraP1Interpolater(bs, vr.parameters['NELECT'],
                             coeff_file='fort.123')
bt1.initialize()

allowed_bands = list(bt1.parameters.allowed_ibands)
print("Interpolating using BoltzTraP1")
bt1_energies, bt1_velocities, bt1_masses = bt1.get_energies(
    kpts, allowed_bands, return_velocity=True, return_effective_mass=True)

print("loading BoltzTraP2")
bt2 = BoltzTraP2Interpolater(bs, vr.parameters['NELECT'])
bt2.initialize()

print("Interpolating using BoltzTraP2")
bt2_energies, bt2_velocities, bt2_masses = bt2.get_energies(
    kpts, allowed_bands, return_velocity=True, return_effective_mass=True)

print("avg diff energy: {}".format((bt1_energies - bt2_energies).mean()))
print("avg diff velocity: {}".format((bt1_velocities - bt2_velocities).mean()))
print("avg diff mass: {}".format((bt1_masses - bt2_masses).mean()))

f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))

kpt_index_mesh = [list(range(len(kpts))) for _ in range(len(bt1_energies))]
ax1.scatter(kpt_index_mesh, bt1_energies, label='bt1')
ax1.scatter(kpt_index_mesh, bt2_energies, label='bt2')
ax1.set(xlabel='k-point index', ylabel='energy / eV')

ax2.scatter(bt1_energies, np.linalg.norm(bt1_velocities, axis=2), label='bt1')
ax2.scatter(bt2_energies, np.linalg.norm(bt2_velocities, axis=2), label='bt2')
ax2.set(xlabel='energy / eV', ylabel='velocity')

ax3.scatter(bt1_energies, np.mean(np.linalg.norm(bt1_masses, axis=2), axis=2),
            label='bt1')
ax3.scatter(bt2_energies, np.mean(np.linalg.norm(bt2_masses, axis=2), axis=2),
            label='bt2')
ax3.set(xlabel='energy / eV', ylabel='effective mass')

plt.legend()
plt.show()

bt1_dos = bt1.get_dos([10, 10, 10])
bt2_dos = bt2.get_dos([10, 10, 10])

plt.plot(bt1_dos[:, 0], bt1_dos[:, 1], label='bt1')
plt.plot(bt2_dos[:, 0], bt2_dos[:, 1], label='bt2')
plt.legend()
plt.show()
