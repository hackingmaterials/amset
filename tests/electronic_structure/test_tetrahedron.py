import unittest

import numpy as np
from pymatgen import Spin


class TetrahedralBandStructureTest(unittest.TestCase):
    def setUp(self):
        self.kpoints = np.array(
            [
                [0.0, 0.0, 0.0],
                [-0.5, 0.0, 0.0],
                [0.0, -0.5, 0.0],
                [-0.5, -0.5, 0.0],
                [0.0, 0.0, -0.5],
                [-0.5, 0.0, -0.5],
                [0.0, -0.5, -0.5],
                [-0.5, -0.5, -0.5],
            ]
        )

        self.energies = {Spin.up: np.array([[1, 2, 2, 2, 2, 2, 2, 3]])}

        self.tetrahedra = np.array(
            [
                [0, 1, 3, 7],
                [0, 1, 5, 7],
                [0, 2, 3, 7],
                [0, 2, 6, 7],
                [0, 4, 5, 7],
                [0, 4, 6, 7],
                [1, 0, 2, 6],
                [1, 0, 4, 6],
                [1, 3, 2, 6],
                [1, 3, 7, 6],
                [1, 5, 4, 6],
                [1, 5, 7, 6],
                [2, 3, 1, 5],
                [2, 3, 7, 5],
                [2, 0, 1, 5],
                [2, 0, 4, 5],
                [2, 6, 7, 5],
                [2, 6, 4, 5],
                [3, 2, 0, 4],
                [3, 2, 6, 4],
                [3, 1, 0, 4],
                [3, 1, 5, 4],
            ]
        )

    def test_test_init(self):
        # tbs = TetrahedralBandStructure(self.energies, self.kpoints, self.tetrahedra)
        pass
