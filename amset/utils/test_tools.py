# coding: utf-8

from __future__ import unicode_literals, absolute_import
import numpy as np
import os
import unittest

from amset.core import AMSET
from amset.utils.tools import kpts_to_first_BZ, get_closest_k, \
    remove_duplicate_kpoints
from pymatgen.io.vasp import Vasprun

test_dir = os.path.dirname(__file__)

class AmsetToolsTest(unittest.TestCase):
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


    def test_kpts_to_first_BZ(self):
        kpts_orig = [[0.51, 1.00, -0.50], [1.40, -1.20, 0.49]]
        kpts_trns = [[-0.49, 0.00, -0.50], [0.40, -0.20, 0.49]]
        # self.assertListEqual() #doesn't work as they differ at 7th decimal
        for ik, k in enumerate(kpts_to_first_BZ(kpts_orig)):
            np.testing.assert_array_almost_equal(kpts_trns[ik], k, 7)
        self.assertTrue(isinstance(kpts_to_first_BZ(kpts_orig), list))


    def test_get_closest_k(self):
        kpts = np.array([[0.51, -0.5, 0.5], [0.4, 0.5, 0.51]])
        np.testing.assert_array_equal([0.4 , 0.5, 0.51],
            get_closest_k(np.array([0.5, 0.5, 0.5]), kpts, return_diff=False))
        np.testing.assert_array_almost_equal([0.1 , 0.0, -0.01],
            get_closest_k(np.array([0.5, 0.5, 0.5]), kpts, return_diff=True))


    def test_remove_duplicate_kpoints(self):
        kpts_orig = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.00999],
                     [0.25, 0.25, 0.25], [0.25, 0.25, 0.25],
                     [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5]]
        kpts_out = [[0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25],
                    [0.5, 0.5, 0.5]]
        self.assertListEqual(kpts_out, remove_duplicate_kpoints(kpts_orig))


if __name__ == '__main__':
    unittest.main()