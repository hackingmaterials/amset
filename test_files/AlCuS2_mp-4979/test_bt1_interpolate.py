# coding: utf-8
from amset.interpolate.boltztrap1 import BoltzTraP1Interpolater
from pymatgen.io.vasp.outputs import Vasprun
import numpy as np

vr = Vasprun("vasprun.xml")
bs = vr.get_band_structure()

bi = BoltzTraP1Interpolater(bs, vr.parameters['NELECT'], coeff_file='fort.123', n_jobs=1)
bi.initialize()
kpts = np.array(vr.actual_kpoints)
energies, velocities, effective_masses = bi.get_energies(kpts, 26)
