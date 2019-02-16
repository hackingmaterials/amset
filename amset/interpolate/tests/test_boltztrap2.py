import os
import unittest
import numpy as np

from numpy.testing import assert_array_equal

from amset.interpolate.boltztrap2 import BoltzTraP2Interpolater
from pymatgen.io.vasp import Vasprun

test_dir = os.path.dirname(os.path.abspath(__file__))
amset_files = os.path.join(test_dir, '..', '..', '..', 'test_files',
                           'AlCuS2_mp-4979')


class TestBoltzTraP2Interpolater(unittest.TestCase):
    """Tests for interpolating a band structure using BoltzTraP2."""

    def setUp(self):
        vr = Vasprun(os.path.join(amset_files, 'vasprun.xml'))
        bs = vr.get_band_structure()
        num_electrons = vr.parameters['NELECT']

        self.kpoints = np.array(vr.actual_kpoints)
        self.interpolater = BoltzTraP2Interpolater(bs, num_electrons)

    def test_initialisation(self):
        """Test coefficients and parameters are calculated correctly."""
        self.interpolater.initialize()
        params = self.interpolater._parameters
        assert_array_equal(params[0][0], [[0, 0, 0]])
        self.assertAlmostEqual(params[1][0][0], 9.14055614)
        self.assertAlmostEqual(params[2][0][0].real, -2.33144546)

    def test_get_energies(self):
        """Test getting the interpolated energy, velocity and effective mass."""

        # test just getting energy
        energies = self.interpolater.get_energies(
            self.kpoints, 25, return_velocity=False,
            return_effective_mass=False)
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 3.852399483908641)

        # test energy + velocity
        energies, velocities = self.interpolater.get_energies(
            self.kpoints, 25, return_velocity=True,
            return_effective_mass=False)
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 3.852399483908641)
        self.assertEqual(velocities.shape, (138, 3))
        self.assertAlmostEqual(velocities[10][0], 5.401166156893334e+6,
                               places=0)

        # test energy + effective_mass
        energies, effective_masses = self.interpolater.get_energies(
            self.kpoints, 25, return_velocity=False,
            return_effective_mass=True)
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 3.852399483908641)
        self.assertEqual(effective_masses.shape, (138, 3, 3))
        self.assertAlmostEqual(effective_masses[10][0][0], 0.1121284806813247)

        # test energy + velocity + effective_mass
        energies, velocities, effective_masses = self.interpolater.get_energies(
            self.kpoints, 25, return_velocity=True,
            return_effective_mass=True)
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 3.852399483908641)
        self.assertEqual(velocities.shape, (138, 3))
        self.assertAlmostEqual(velocities[10][0], 5.401166156893334e+6,
                               places=0)
        self.assertEqual(effective_masses.shape, (138, 3, 3))
        self.assertAlmostEqual(effective_masses[10][0][0], 0.1121284806813247)

    def test_get_energies_scissor(self):
        """Test scissoring of band energies."""

        # test valence band
        energies = self.interpolater.get_energies(
            self.kpoints, 25, return_velocity=False,
            return_effective_mass=False, scissor=1.)
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 3.852399483908641 - 0.5)

        # test conduction band
        energies = self.interpolater.get_energies(
            self.kpoints, 33, return_velocity=False,
            return_effective_mass=False, scissor=1.)
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 7.301700765 + 0.5)

    def test_get_energies_multiple_bands(self):
        """Test getting the interpolated data for multiple bands."""

        # test just getting energy
        energies = self.interpolater.get_energies(
            self.kpoints, [25, 35], return_velocity=False,
            return_effective_mass=False)
        self.assertEqual(energies.shape, (2, 138,))
        self.assertAlmostEqual(energies[0][0], 3.852399483908641)
        self.assertAlmostEqual(energies[1][0], 9.594401616456384)

        # test energy + velocity
        energies, velocities = self.interpolater.get_energies(
            self.kpoints, [25, 35], return_velocity=True,
            return_effective_mass=False)
        self.assertEqual(energies.shape, (2, 138,))
        self.assertAlmostEqual(energies[0][0], 3.852399483908641)
        self.assertAlmostEqual(energies[1][0], 9.594401616456384)
        self.assertEqual(velocities.shape, (2, 138, 3))
        self.assertAlmostEqual(velocities[0][10][0], 5.401166156893334e+6,
                               places=0)
        self.assertAlmostEqual(velocities[1][10][0], 1.1686720831758736e+8,
                               places=0)

        # test energy + effective_mass
        energies, effective_masses = self.interpolater.get_energies(
            self.kpoints, [25, 35], return_velocity=False,
            return_effective_mass=True)
        self.assertEqual(energies.shape, (2, 138,))
        self.assertAlmostEqual(energies[0][0], 3.852399483908641)
        self.assertAlmostEqual(energies[1][0], 9.594401616456384)
        self.assertEqual(effective_masses.shape, (2, 138, 3, 3))
        self.assertAlmostEqual(effective_masses[0][10][0][0], 0.11212848068132)
        self.assertAlmostEqual(effective_masses[1][10][0][0], 0.004528551672350)

        # test energy + velocity + effective_mass
        energies, velocities, effective_masses = self.interpolater.get_energies(
            self.kpoints, [25, 35], return_velocity=True,
            return_effective_mass=True)
        self.assertEqual(energies.shape, (2, 138,))
        self.assertAlmostEqual(energies[0][0], 3.852399483908641)
        self.assertAlmostEqual(energies[1][0], 9.594401616456384)
        self.assertEqual(velocities.shape, (2, 138, 3))
        self.assertAlmostEqual(velocities[0][10][0], 5.401166156893334e+6,
                               places=0)
        self.assertAlmostEqual(velocities[1][10][0], 1.1686720831758736e+8,
                               places=0)
        self.assertEqual(effective_masses.shape, (2, 138, 3, 3))
        self.assertAlmostEqual(effective_masses[0][10][0][0], 0.11212848068132)
        self.assertAlmostEqual(effective_masses[1][10][0][0], 0.004528551672350)

    def test_get_energies_all_bands(self):
        # test all bands
        energies, velocities, effective_masses = self.interpolater.get_energies(
            self.kpoints, None, return_velocity=True,
            return_effective_mass=True)
        self.assertEqual(energies.shape, (96, 138,))
        self.assertAlmostEqual(energies[25][0], 3.852399483908641)
        self.assertAlmostEqual(energies[35][0], 9.594401616456384)
        self.assertEqual(velocities.shape, (96, 138, 3))
        self.assertAlmostEqual(velocities[25][10][0], 5.401166156893334e+6,
                               places=0)
        self.assertAlmostEqual(velocities[35][10][0], 1.1686720831758736e+8,
                               places=0)
        self.assertEqual(effective_masses.shape, (96, 138, 3, 3))
        self.assertAlmostEqual(effective_masses[25][10][0][0], 0.11212848068132)
        self.assertAlmostEqual(effective_masses[35][10][0][0], 0.00452855167235)

    def test_get_dos(self):
        """Test generating the interpolated DOS."""
        dos = self.interpolater.get_dos([10, 10, 10], emin=-10, emax=10,
                                        width=0.075)
        self.assertEqual(dos.shape, (20000, 2))
        self.assertEqual(dos[0][0], -10)
        self.assertAlmostEqual(dos[15000][1], 3.5362612128412807)

        # test normalising the DOS
        dos = self.interpolater.get_dos([10, 10, 10], emin=-10, emax=10,
                                        width=0.075, normalize=True)
        self.assertEqual(dos.shape, (20000, 2))
        self.assertEqual(dos[0][0], -10)
        self.assertAlmostEqual(dos[15000][1], 21.641789697605823)

        # test single parabolic band DOS
        dos = self.interpolater.get_dos([10, 10, 10], emin=-10, emax=10,
                                        width=0.075, normalize=True,
                                        minimum_single_parabolic_band=True)
        self.assertEqual(dos.shape, (20000, 2))
        self.assertEqual(dos[0][0], -10)
        self.assertAlmostEqual(dos[15000][1], 2.093830973742877)

    def test_get_extrema(self):
        """Test getting the band structure extrema."""

        # test VBM
        extrema = self.interpolater.get_extrema(31, e_cut=1.)
        np.testing.assert_array_almost_equal(extrema[0], [0., 0.0, 0.0], 10)
        np.testing.assert_array_almost_equal(extrema[1],
                                             [0.0972, 0.4028, 0.0972], 4)
        np.testing.assert_array_almost_equal(extrema[2],
                                             [0.5, 0.5, -0.5], 10)
        np.testing.assert_array_almost_equal(extrema[3],
                                             [-0.0365, 0.0365, 0.5], 4)

        # test CBM
        extrema = self.interpolater.get_extrema(32, e_cut=1.)
        np.testing.assert_array_almost_equal(extrema[0], [0., 0.0, 0.0], 10)
        np.testing.assert_array_almost_equal(extrema[1], [0., 0.0, 0.5], 10)
        np.testing.assert_array_almost_equal(extrema[2],
                                             [0.5, 0.5, -0.5], 4)
        np.testing.assert_array_almost_equal(extrema[3],
                                             [0.4167, 0.4167, -0.0019], 4)

        # def get_extrema(folder):
        #     path = os.path.join(test_dir, '..', '..', '..', 'test_files',
        #                         folder)
        #     vr = Vasprun(os.path.join(path, 'vasprun.xml'))
        #     bs = vr.get_band_structure()
        #     num_electrons = vr.parameters['NELECT']
        #     interpolater = BoltzTraP2Interpolater(bs, num_electrons)
        #     vbm = max(bs.get_vbm()['band_index'][Spin.up])
        #     return {'p': interpolater.get_extrema(vbm, e_cut=1.),
        #             'n': interpolater.get_extrema(vbm + 1, e_cut=1.)}
        #
        # inp_extrema = get_extrema('InP_mp-20351')
        # print(inp_extrema)
        # np.testing.assert_array_almost_equal(inp_extrema['n'][0],
        #                                      [0., 0.0, 0.], 10)
        # np.testing.assert_array_almost_equal(inp_extrema['n'][1],
        #                                      [0., 0.5, 0.], 10)
        # np.testing.assert_array_almost_equal(inp_extrema['p'][0],
        #                                      [0., 0.0, 0.], 10)
        # np.testing.assert_array_almost_equal(inp_extrema['p'][1],
        #                                      [-0.3843, -0.0325, 0.], 4)
