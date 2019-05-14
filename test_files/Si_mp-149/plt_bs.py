# coding: utf-8
import matplotlib
matplotlib.use("TkAgg")
from amset.interpolate.boltztrap2 import BoltzTraP2Interpolater
from pymatgen.io.vasp.outputs import Vasprun
from sumo.plotting.bs_plotter import SBSPlotter

vr = Vasprun('vasprun.xml')
bs = vr.get_band_structure()
nelect = vr.parameters['NELECT']

bt = BoltzTraP2Interpolater(bs, nelect)
lm_bs = bt.get_line_mode_band_structure()

plotter = SBSPlotter(lm_bs)
plt = plotter.get_plot(vbm_cbm_marker=True)
plt.savefig("lm_bs.pdf")
plt.show()
