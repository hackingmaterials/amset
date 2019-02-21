import warnings

import numpy as np
import os
import unittest

from amset.utils.constants import comp_to_dirname
from amset.utils.band_structure import kpoints_to_first_bz, get_closest_k, \
    remove_duplicate_kpoints
from pymatgen.io.vasp.outputs import Vasprun

tdir = os.path.join(os.path.dirname(__file__), '..', '..', 'test_files')
vruns = {c: Vasprun(os.path.join(tdir, comp_to_dirname[c], 'vasprun.xml'))
         for c in comp_to_dirname}
coeff_files = {c: os.path.join(tdir, comp_to_dirname[c], 'fort.123')
               for c in comp_to_dirname}

warnings.simplefilter("ignore")


class AmsetToolsTest(unittest.TestCase):

    def test_kpts_to_first_BZ(self):
        kpts_orig = [[0.51, 1.00, -0.50], [1.40, -1.20, 0.49]]
        kpts_trns = [[-0.49, 0.00, -0.50], [0.40, -0.20, 0.49]]

        np.testing.assert_array_almost_equal(kpoints_to_first_bz(kpts_orig),
                                             kpts_trns, 7)

    def test_get_closest_k(self):
        kpts = np.array([[0.51, -0.5, 0.5], [0.4, 0.5, 0.51]])
        np.testing.assert_array_equal(
            [0.4, 0.5, 0.51], get_closest_k(np.array([0.5, 0.5, 0.5]),
                                            kpts, return_diff=False))
        np.testing.assert_array_almost_equal(
            [0.1, 0.0, -0.01], get_closest_k(np.array([0.5, 0.5, 0.5]),
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
        np.testing.assert_array_almost_equal(
            remove_duplicate_kpoints(kpts_orig), kpts_out)
