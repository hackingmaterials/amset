import os
import unittest

from amset.utils.band_structure import get_dft_orbitals
from pymatgen.io.vasp.outputs import Vasprun

test_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'test_files')


class BandStructureTest(unittest.TestCase):

    def setUp(self):
        vr = Vasprun(os.path.join(test_dir, "GaAs_mp-2534", 'vasprun.xml'),
                     parse_projected_eigen=True)
        self.bs = vr.get_band_structure()

    def test_get_dft_orbitals(self):
        s_orbital, p_orbital = get_dft_orbitals(self.bs, 0)
        self.assertEqual(s_orbital[6], 0.0166)
        self.assertEqual(s_orbital[8], 0.0193)
        self.assertEqual(p_orbital[0], 0.0031)
        self.assertEqual(p_orbital[5], 0.0014)

        s_orbital, p_orbital = get_dft_orbitals(self.bs, 10)
        self.assertEqual(s_orbital[10], 0.0312)
        self.assertEqual(s_orbital[12], 0.0329)
        self.assertEqual(p_orbital[2], 0.4226)
        self.assertEqual(p_orbital[5], 0.3713)
