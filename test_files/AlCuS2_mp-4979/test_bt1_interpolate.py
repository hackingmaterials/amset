# coding: utf-8
from amset.interpolate.boltztrap1 import BoltzTraP1Interpolater
from pymatgen.io.vasp.outputs import Vasprun

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

vr = Vasprun("vasprun.xml")
bs = vr.get_band_structure()

bi = BoltzTraP1Interpolater(bs, vr.parameters['NELECT'], coeff_file='fort.123', n_jobs=1)
bi.initialize()
kpts = np.array(vr.actual_kpoints)
energies = bi.get_energies(kpts, 26)

new_mesh, new_dos, new_nbands = bi.get_dos([10, 10, 10], emin=-10, emax=10, width=0.075)


from monty.serialization import loadfn
mesh, dos, nbands = loadfn('dos.json')
plt.plot(mesh, dos, label='old')
plt.plot(new_mesh, new_dos, label='new')
plt.legend()
plt.show()
