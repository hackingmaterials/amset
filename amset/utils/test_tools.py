# coding: utf-8

from __future__ import unicode_literals, absolute_import
import numpy as np
import os
import unittest

from amset.core import AMSET
from amset.utils.tools import kpts_to_first_BZ, get_closest_k, \
    remove_duplicate_kpoints, get_energy_args, get_bindex_bspin, interpolate_bs
from pymatgen.io.vasp import Vasprun

test_dir = os.path.dirname(__file__)

class AmsetToolsTest(unittest.TestCase):
    def setUp(self):
        self.GaAs_path = os.path.join(test_dir, '..', '..', 'test_files', 'GaAs')
        self.GaAs_cube = os.path.join(self.GaAs_path, "nscf-uniform/boltztrap/fort.123")
        self.GaAs_vrun = Vasprun(os.path.join(self.GaAs_path, "nscf-uniform", "vasprun.xml"))


    def tearDown(self):
        pass


    # def test_get_bs_extrema(self):
    #     amset = AMSET(calc_dir='.', material_params={'epsilon_s': 12.9})
    #     amset.read_vrun(calc_dir=os.path.join(self.GaAs_path, 'nscf-uniform'))
    #     extrema = amset.get_bs_extrema(bs=self.GaAs_vrun.get_band_structure(),
    #                 coeff_file=self.GaAs_cube, nbelow_vbm=0, nabove_cbm=0)
    #     self.assertTrue(any(([.0, .0, .0] == x).all() for x in extrema['n']))
    #     self.assertTrue(any(([.5, .5, .5] == x).all() for x in extrema['n']))
    #     self.assertTrue(any(([.0, .0, .0] == x).all() for x in extrema['p']))
    #
    #
    # def test_kpts_to_first_BZ(self):
    #     kpts_orig = [[0.51, 1.00, -0.50], [1.40, -1.20, 0.49]]
    #     kpts_trns = [[-0.49, 0.00, -0.50], [0.40, -0.20, 0.49]]
    #     # self.assertListEqual() #doesn't work as they differ at 7th decimal
    #     for ik, k in enumerate(kpts_to_first_BZ(kpts_orig)):
    #         np.testing.assert_array_almost_equal(kpts_trns[ik], k, 7)
    #     self.assertTrue(isinstance(kpts_to_first_BZ(kpts_orig), list))
    #
    #
    # def test_get_closest_k(self):
    #     kpts = np.array([[0.51, -0.5, 0.5], [0.4, 0.5, 0.51]])
    #     np.testing.assert_array_equal([0.4 , 0.5, 0.51],
    #         get_closest_k(np.array([0.5, 0.5, 0.5]), kpts, return_diff=False))
    #     np.testing.assert_array_almost_equal([0.1 , 0.0, -0.01],
    #         get_closest_k(np.array([0.5, 0.5, 0.5]), kpts, return_diff=True))
    #
    #
    # def test_remove_duplicate_kpoints(self):
    #     kpts_orig = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.00999],
    #                  [0.25, 0.25, 0.25], [0.25, 0.25, 0.25],
    #                  [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5]]
    #     kpts_out = [[0.0, 0.0, 0.0],
    #                 [0.25, 0.25, 0.25],
    #                 [0.5, 0.5, 0.5]]
    #     self.assertListEqual(kpts_out, remove_duplicate_kpoints(kpts_orig))

    def test_interpolate_bs(self, check_bzt2=False):
        bs = self.GaAs_vrun.get_band_structure()
        vbm_idx, vbm_bidx = get_bindex_bspin(bs.get_vbm(), is_cbm=False)
        cbm_idx, cbm_bidx = get_bindex_bspin(bs.get_cbm(), is_cbm=True)
        dft_vbm = bs.get_vbm()['energy']
        dft_cbm = bs.get_cbm()['energy']
        dft_vb = np.array(bs.bands[vbm_bidx][vbm_idx]) - dft_vbm
        dft_cb = np.array(bs.bands[cbm_bidx][cbm_idx]) - dft_cbm
        vbm_idx += 1 # in boltztrap1 interpolation the first band is 1st not 0th
        cbm_idx += 1

        kpts = [k.frac_coords for k in bs.kpoints]
        matrix = self.GaAs_vrun.lattice.matrix
        print(matrix)

        # get the interpolation parameters
        interp_params1 = get_energy_args(self.GaAs_cube, [vbm_idx, cbm_idx])

        # calculate and check the last valence and the first conduction bands:
        vb_en1, vb_vel1, vb_masses1 = interpolate_bs(kpts, interp_params1,
                iband=0, method="boltztrap1", scissor=0.0, matrix=matrix)
        cb_en1, cb_vel1, cb_masses1 = interpolate_bs(kpts, interp_params1,
                iband=1, method="boltztrap1", scissor=0.0, matrix=matrix)

        vbm = np.max(vb_en1)
        vb_en1 -= vbm
        cb_en1 -= vbm
        interp_gap = min(cb_en1) - max(vb_en1)
        self.assertAlmostEqual(bs.get_band_gap()['energy'], interp_gap, 4)
        self.assertAlmostEqual(interp_gap, 0.9582, 4)

        print(vb_en1 - dft_vb)
        print(cb_en1 - dft_cb)

        print(np.mean(vb_en1))
        print(np.mean(cb_en1))

        if check_bzt2:
            vb_en2, vb_ve2, vb_masses2 = interpolate_bs(kpts, interp_params2,
                iband=vbm_idx, method="boltztrap2", scissor=0.0, matrix=matrix)
            cb_en2, cb_ve2, cb_masses2 = interpolate_bs(kpts, interp_params2,
                iband=cbm_idx, method="boltztrap2", scissor=0.0, matrix=matrix)


if __name__ == '__main__':
    unittest.main()