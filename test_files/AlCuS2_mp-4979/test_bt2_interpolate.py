# coding: utf-8
from amset.interpolate.boltztrap2 import BoltzTraP2Interpolater
from pymatgen.io.vasp.outputs import Vasprun

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

vr = Vasprun("vasprun.xml")
bs = vr.get_band_structure()

bi = BoltzTraP2Interpolater(bs, vr.parameters['NELECT'])
bi.initialize()
kpts = np.array(vr.actual_kpoints)
energies = bi.get_energies(kpts, 26)
