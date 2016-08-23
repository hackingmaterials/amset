#!/usr/bin/env python

import os
from numpy import linspace, vstack
from pymatgen.electronic_structure.boltztrap import BoltztrapRunner
from pymatgen.io.vasp.outputs import Vasprun, Spin, Procar
from collections import defaultdict
import pymatgen
from math import pi
from tools_for_AMSET import read_vrun, fit_dos, fit_procar, eval_poly_multi, write_to_file

"""
This module provides classes to run and analyze AMSET on pymatgen Vasprun objects. AMSET which stands for "ab initio Model for calculating
Mobility and Seebeck coefficient using Boltzmann Transport equation, calculates
such properties in BTE framework taking into various scattering mechanisms such
as longitudinal polar optical (PO) scattering, transverse optical (TO) phonon,
acoustic phonon deformation potential (PO) ionized impurity (ii), piezoelectric (pe),
charged dislocation (dis) as well as user defined (usr) scattering mechanisms.

AMSET (a.k.a aMoBT) has been developed by Alireza Faghaninia.

https://github.com/albalu/amobt

References are::

    Faghaninia, A. and Ager, J. W. and Lo, C. S. (2015)
    Ab initio electronic transport model with explicit solution to the linearized Boltzmann transport equation
    Phys. Rev. B 91, 235123
"""


__author__ = "Alireza Faghaninia, Anubhav Jain"
__copyright__ = "Copyright 2016"
__version__ = "0.0.1"
__maintainer__ = "Alireza Faghaninia"
__email__ = "alireza.faghaninia@gmail.com"
__status__ = "Development"
__date__ = "June 22, 2016"

class AMSETRunner(object):
    """ This class is used to run AMSET on a pymatgen Vasprun object. AMSET is an ab initio model for calculating
    the mobility and Seebeck coefficient using Boltzmann transport equation. The band structure is extracted from
    vasprun.xml to calculate the group velocity and transport properties in presence of various scattering mechanisms.

     Currently the following scattering mechanisms with their corresponding three-letter abbreviations implemented are:
     ionized impurity scattering (iim), acoustic phonon deformation potential (acd), piezoelectric (pie), and charged
     dislocation scattering (dis). Also, longitudinal polar optical phonon (pop) in implemented as an inelastic
     scattering mechanism that can alter the electronic distribution (the reason BTE has to be solved explicitly).

     AMSET is designed in a modular way so that users can add more scattering mechanisms as followed:
     ??? (instruction to add a scattering mechanism) ???
     """

    def __init__(self, vrun, gmaxiter=10, plotdir='plots', k_F=0.2, k_max=3.5,
                 charge=1, N_dis=0.1, C_long=0, C_trans=0, P_piezo=0.15, Bgap=1,
                 elec={"m": 0, "m2":0, "nbands":0, "Edef":0}, hole={"m": 0, "m2":0, "nbands":0, "Edef":0},
                 VBMoverride=0, free_e=True, fitprocar=True, ii_pwe=False,
                 isntype=True, omega_LO=0, omega_TO=0, eps_s=0, eps_inf=0,
                 Tarray=range(50,1301,50), narray=[1e17, 1e18, 1e19, 1e20, 1e21]):
        self._vrun = vrun
        self.gmaxiter = gmaxiter
        self.plotdir = plotdir
        self.k_F = k_F
        self.k_max = k_max
        self.charge = charge
        self.N_dis = N_dis
        self.C_long = C_long
        self.C_trans = C_trans
        self.P_piezo = P_piezo
        self.Bgap = Bgap
        self.elec = elec
        self.hole = hole
        self.VBMoverride = VBMoverride
        self.free_e = free_e
        self.readprocar = fitprocar
        self.ii_pwe = ii_pwe
        self.isntype = isntype
        self.omega_LO = omega_LO
        self.omega_TO = omega_TO
        self.eps_s = eps_s
        self.eps_inf = eps_inf
        self.Tarray = Tarray
        self.narray = narray

        if not self.plotdir.endswith('/'):
            self.plotdir += '/'
        if not os.path.exists(plotdir):
            os.system('mkdir ' + plotdir)
        def make_kgrid(k_max, fine=False):
            if fine:
                return [i*0.0001 for i in range(1,10)] + [i*0.001 for i in range(1,51)] + \
                       [0.05 + i*0.01 for i in range(1, int((k_max-0.05)/0.01)+1)]
            else:
                return [i*0.0001 for i in range(1,10)] + [i*0.001 for i in range(1,10)] + \
                       [i*0.01 for i in range(1, int(k_max/0.01)+1)]

        k_grid = make_kgrid(k_max = self.k_max, fine=True)
        print len(k_grid)
    # Read the vrun object from pymatgen and setup proper inputs for AMSET
        volume, density, lattice, self.elec, self.hole, \
            fitted_cond_band \
                = read_vrun(self._vrun, self.elec, self.hole, self.plotdir)

        print "m*_e1 = {}".format(str(self.elec["m"]))
        print "m*_e2 = {}".format(str(self.elec["m2"]))
        print "m*_h1 = {}".format(str(self.hole["m"]))
        print "m*_h2 = {}".format(str(self.hole["m2"]))

        energy = eval_poly_multi(fitted_cond_band, k_grid)
        energy = [i - energy[0] for i in energy] # Renormalization to flatten small deviations from 0 due to fitting
        print len(energy)
        with open(self.plotdir + 'fitted_bands.txt', 'w') as f:
            # f.write("%8s,%8s\n" % ("k-1/nm", "cond-eV"))
            # for i in range(len(energy)):
            #     f.write("%8.4f, %8.4f\n" % (k_grid[i], energy[i]))
            write_to_file(f, data=[k_grid, energy], legend=["k-1/nm", "cond-eV"])
        print lattice
        print self.elec["m"]
        print self.elec["nbands"]
        print self.elec["m"]
        Ds_n, Ds_p = fit_dos(self.free_e)
        a, c = fit_procar(self.isntype, self.readprocar, len(k_grid))

if __name__ == "__main__":
    vrun = Vasprun('vasprun.xml')


    bt = AMSETRunner(vrun)
    print(bt.free_e)

