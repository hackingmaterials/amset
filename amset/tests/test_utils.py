import warnings

import numpy as np
import os
import unittest

from amset.utils.constants import comp_to_dirname
from amset.utils.band_structure import kpts_to_first_bz, get_closest_k, \
    remove_duplicate_kpoints
from amset.utils.band_interpolation import get_bs_extrema
from pymatgen.io.vasp.outputs import Vasprun

tdir = os.path.join(os.path.dirname(__file__), '..', '..', 'test_files')
vruns = {c: Vasprun(os.path.join(tdir, comp_to_dirname[c], 'vasprun.xml'))
         for c in comp_to_dirname}
coeff_files = {c: os.path.join(tdir, comp_to_dirname[c], 'fort.123')
               for c in comp_to_dirname}

warnings.simplefilter("ignore")


class AmsetToolsTest(unittest.TestCase):

    def listalmostequal(self, list1, list2, places=3):
        for l1, l2 in zip(list1, list2):
            self.assertAlmostEqual(l1, l2, places=places)

    def test_get_bs_extrema(self):
        extrema = get_bs_extrema(bs=vruns['GaAs'].get_band_structure(),
                                 coeff_file=coeff_files['GaAs'], Ecut=1.0)
        # first conduction band extrema:
        self.listalmostequal(extrema['n'][0], [.0, .0, .0], 10)
        self.listalmostequal(extrema['n'][1], [.0, -0.4701101, .0], 4)
        self.assertEqual(len(extrema['n']), 2)
        # last valence band extrema
        self.listalmostequal(extrema['p'][0], [.0, .0, .0], 10)
        self.listalmostequal(extrema['p'][1], [-0.31693, -0.04371, 0.], 4)
        self.assertEqual(len(extrema['p']), 2)

        si_extrema = get_bs_extrema(bs=vruns['Si'].get_band_structure(),
                                    coeff_file=coeff_files['Si'], Ecut=1.0)
        self.listalmostequal(si_extrema['n'][0], [0.419204, 0.419204, 0.], 4)
        self.listalmostequal(si_extrema['n'][1], [-0.4638, -0.4638, -0.4638], 4)
        self.listalmostequal(si_extrema['p'][0], [0.0, 0.0, 0.0], 10)
        self.listalmostequal(si_extrema['p'][1], [-0.226681, -0.049923, 0.], 3)

        # test can take a very long time. Commenting this out for now to
        # avoid issues
        # pbte_extrema = get_bs_extrema(bs=vruns['PbTe'].get_band_structure(),
        #                               coeff_file=coeff_files['PbTe'], Ecut=1.0)
        # self.listalmostequal(pbte_extrema['n'][0], [0., 0.5, 0.], 10)
        # self.listalmostequal(pbte_extrema['n'][1], [.1522, -.0431, .1522], 4)
        # self.listalmostequal(pbte_extrema['p'][0], [0., 0.5, 0.], 10)
        # self.listalmostequal(pbte_extrema['p'][1], [.4784, -.2709, .2278], 3)
        # self.listalmostequal(pbte_extrema['p'][2], [.162054, .162054, 0.], 3)
        #
        # inp_extrema = get_bs_extrema(bs=vruns['InP'].get_band_structure(),
        #                              coeff_file=coeff_files['InP'], Ecut=1.0)
        # self.listalmostequal(inp_extrema['n'][0], [0., 0.0, 0.], 10)
        # self.listalmostequal(inp_extrema['n'][1], [0., 0.5, 0.], 10)
        # self.listalmostequal(inp_extrema['p'][0], [0., 0.0, 0.], 10)
        # self.listalmostequal(inp_extrema['p'][1], [-0.3843, -0.0325, 0.], 4)
        #
        # alcus2_extrema = get_bs_extrema(bs=vruns['AlCuS2'].get_band_structure(),
        #                                 coeff_file=coeff_files['AlCuS2'],
        #                                 Ecut=1.0)
        # self.listalmostequal(alcus2_extrema['n'][0], [0., 0.0, 0.0], 10)
        # self.listalmostequal(alcus2_extrema['n'][1], [0., 0.0, 0.5], 10)
        # self.listalmostequal(alcus2_extrema['n'][2], [-0.49973, -0.49973, 0.],
        #                      4)
        # self.listalmostequal(alcus2_extrema['n'][3],
        #                      [0.49047, 0.49047, 0.49818], 4)
        # self.listalmostequal(alcus2_extrema['p'][0], [0., 0.0, 0.0], 10)
        # self.listalmostequal(alcus2_extrema['p'][1], [0.28291, 0., -0.40218], 4)
        # self.listalmostequal(alcus2_extrema['p'][2], [-0.25765, 0.25148, 0.], 4)
        # self.listalmostequal(alcus2_extrema['p'][3], [-0.49973, -0.49973, 0.],
        #                      4)
        #
        # in2o3_extrema = get_bs_extrema(bs=vruns['In2O3'].get_band_structure(),
        #                                coeff_file=coeff_files['In2O3'],
        #                                Ecut=1.0)
        # self.listalmostequal(in2o3_extrema['n'][0], [0., 0.0, 0.0], 10)
        # self.listalmostequal(in2o3_extrema['p'][0], [0., 0.09631, 0.0], 4)
        # self.listalmostequal(in2o3_extrema['p'][1], [0.30498, 0.30498, 0.18299],
        #                      4)

    def test_kpts_to_first_BZ(self):
        kpts_orig = [[0.51, 1.00, -0.50], [1.40, -1.20, 0.49]]
        kpts_trns = [[-0.49, 0.00, -0.50], [0.40, -0.20, 0.49]]
        # self.assertListEqual() #doesn't work as they differ at 7th decimal
        for ik, k in enumerate(kpts_to_first_bz(kpts_orig)):
            np.testing.assert_array_almost_equal(kpts_trns[ik], k, 7)
        self.assertTrue(isinstance(kpts_to_first_bz(kpts_orig), list))

    def test_get_closest_k(self):
        kpts = np.array([[0.51, -0.5, 0.5], [0.4, 0.5, 0.51]])
        np.testing.assert_array_equal([0.4, 0.5, 0.51],
                                      get_closest_k(np.array([0.5, 0.5, 0.5]),
                                                    kpts, return_diff=False))
        np.testing.assert_array_almost_equal([0.1, 0.0, -0.01],
                                             get_closest_k(
                                                 np.array([0.5, 0.5, 0.5]),
                                                 kpts, return_diff=True))

    def test_remove_duplicate_kpoints(self):
        kpts_orig = [[0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.00999],
                     [0.25, 0.25, 0.25],
                     [0.25, 0.25, 0.25],
                     [0.5, 0.5, 0.5],
                     [0.5, 0.5, 0.5],
                     [0.5, 0.5, -0.5]]
        kpts_out = [[0.0, 0.0, 0.00999],
                    [0.25, 0.25, 0.25],
                    [0.5, 0.5, -0.5]]
        self.assertListEqual(kpts_out, remove_duplicate_kpoints(kpts_orig))
