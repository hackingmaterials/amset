# coding: utf-8

from __future__ import unicode_literals, absolute_import
import os
import unittest

from amset.core import AMSET
from pymatgen.io.vasp import Vasprun

test_dir = os.path.dirname(__file__)

class AmsetTest(unittest.TestCase):
    def setUp(self):
        self.GaAs_path = os.path.join(test_dir, '..', '..', 'test_files', 'GaAs')
        self.GaAs_cube = os.path.join(self.GaAs_path, "nscf-uniform/boltztrap/fort.123")
        self.GaAs_vrun = Vasprun(os.path.join(self.GaAs_path, "nscf-uniform", "vasprun.xml"))
    def tearDown(self):
        pass

    def test_get_bs_extrema(self):
        amset = AMSET(calc_dir='.', material_params={'epsilon_s': 12.9})
        amset.read_vrun(calc_dir=os.path.join(self.GaAs_path, 'nscf-uniform'))
        extrema = amset.get_bs_extrema(bs=self.GaAs_vrun.get_band_structure(),
                    coeff_file=self.GaAs_cube, nbelow_vbm=0, nabove_cbm=0)
        self.assertTrue(any(([.0, .0, .0] == x).all() for x in extrema['n']))
        self.assertTrue(any(([.5, .5, .5] == x).all() for x in extrema['n']))
        self.assertTrue(any(([.0, .0, .0] == x).all() for x in extrema['p']))


if __name__ == '__main__':
    unittest.main()