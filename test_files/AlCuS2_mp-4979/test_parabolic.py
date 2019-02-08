get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# coding: utf-8
from amset.interpolate.parabolic import ParabolicInterpolater
from amset.utils.band_structure import remove_duplicate_kpoints
from amset.utils.band_parabolic import get_parabolic_energy
from pymatgen.io.vasp.outputs import Vasprun

import copy
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

vr = Vasprun("vasprun.xml")
bs = vr.get_band_structure()
parabolic_bands = [[[[[0.5, 0.5, 0.5]], [0, 0.1]], [[[0, 0, 0]], [0.5, 0.2]]]]
pi = ParabolicInterpolater(bs, vr.parameters['NELECT'], parabolic_bands)

kpts = np.array(vr.actual_kpoints)
energies = pi.get_energies(kpts, 31)

cart_kpoints = [np.dot(bs.structure.lattice.reciprocal_lattice.matrix,
                       np.array(kpoint))
                for kpoint in kpts]


parabolic_bands_new = copy.deepcopy(parabolic_bands)

for ib in range(len(parabolic_bands)):
    for valley in range(len(parabolic_bands[ib])):
        equivalent_points = bs.get_sym_eq_kpoints(
            parabolic_bands[ib][valley][0], cartesian=True)[0]
        parabolic_bands_new[ib][valley][0] = remove_duplicate_kpoints(
            equivalent_points)

energies_old = [get_parabolic_energy(k, parabolic_bands_new, type="p",
                                     bandgap=bs.get_band_gap()['energy'])[0]
                for k in cart_kpoints]
