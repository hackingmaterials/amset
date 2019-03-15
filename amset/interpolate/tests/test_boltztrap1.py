import os
import unittest
import numpy as np

from amset.interpolate.boltztrap1 import BoltzTraP1Interpolater, \
    BoltzTraP1Parameters
from pymatgen.io.vasp import Vasprun

test_dir = os.path.dirname(os.path.abspath(__file__))
amset_files = os.path.join(test_dir, '..', '..', '..', 'test_files',
                           'AlCuS2_mp-4979')


class TestBoltzTraP1Interpolater(unittest.TestCase):
    """Tests for interpolating a band structure using BoltzTraP1.

    Note these tests do not cover regenerating the coefficient file.
    """

    def setUp(self):
        coeff_file = os.path.join(amset_files, 'fort.123')
        vr = Vasprun(os.path.join(amset_files, 'vasprun.xml'))
        bs = vr.get_band_structure()
        num_electrons = vr.parameters['NELECT']

        self.kpoints = np.array(vr.actual_kpoints)
        self.interpolater = BoltzTraP1Interpolater(
            bs, num_electrons, coeff_file=coeff_file, n_jobs=1)

        self.interpolater_parallel = BoltzTraP1Interpolater(
            bs, num_electrons, coeff_file=coeff_file, n_jobs=-1)

    def test_initialisation(self):
        """Test coefficients and parameters are calculated correctly."""
        self.interpolater.initialize()
        params = self.interpolater.parameters
        self.assertIsInstance(params, BoltzTraP1Parameters)

        self.assertEqual(params.num_symmetries, 16)
        self.assertEqual(params.min_band, 25)
        self.assertEqual(params.max_band, 35)
        self.assertEqual(params.allowed_ibands,
                         {25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35})
        self.assertEqual(params.coefficients[0][0], -0.11634230012183)
        self.assertEqual(params.num_star_vectors[-1], 16)

        np.testing.assert_array_equal(
            params.star_vectors[0][0], [0., 0., 0.])
        np.testing.assert_array_almost_equal(
            params.star_vector_products[1][0],
            [-1.89075, 8.94286, -4.15722], decimal=5)
        self.assertEqual(params.star_vector_products_sq[-1][-1][-1][-1],
                         512.2454464655998)

    def test_get_interpolation_coefficients(self):
        """Test getting the interpolation band coefficients."""
        # test getting all available coefficients
        coeff = self.interpolater._get_interpolation_coefficients()
        self.assertEqual(coeff.shape, (11, 2154))
        self.assertEqual(coeff[0][0], -0.11634230012183)

        # test getting one band coefficients
        coeff = self.interpolater._get_interpolation_coefficients(25)
        self.assertEqual(coeff.shape, (2154,))
        self.assertEqual(coeff[0], -0.11634230012183)

        # test getting multiple band coefficients
        coeff = self.interpolater._get_interpolation_coefficients([25, 26])
        self.assertEqual(coeff.shape, (2, 2154))
        self.assertEqual(coeff[0][0], -0.11634230012183)
        self.assertEqual(coeff[1][0], -0.10136164260112)

        # test error thrown when band out of range
        self.assertRaises(
            ValueError, self.interpolater._get_interpolation_coefficients, 24)

    def test_get_energies_non_parallel(self):
        """Test getting the interpolated energy, velocity and effective mass."""

        # test just getting energy
        energies = self.interpolater.get_energies(
            self.kpoints, 25, return_velocity=False,
            return_effective_mass=False)
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 3.8523994433224855)

        # test energy + velocity
        energies, velocities = self.interpolater.get_energies(
            self.kpoints, 25, return_velocity=True,
            return_effective_mass=False)
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 3.8523994433224855)
        self.assertEqual(velocities.shape, (138, 3))
        self.assertAlmostEqual(velocities[10][0], 5.367132495161134e+6,
                               places=0)

        # test energy + effective_mass
        energies, effective_masses = self.interpolater.get_energies(
            self.kpoints, 25, return_velocity=False,
            return_effective_mass=True)
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 3.8523994433224855)
        self.assertEqual(effective_masses.shape, (138, 3, 3))
        self.assertAlmostEqual(effective_masses[10][0][0], 0.21494134163522982)

        # test energy + velocity + effective_mass
        energies, velocities, effective_masses = self.interpolater.get_energies(
            self.kpoints, 25, return_velocity=True,
            return_effective_mass=True)
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 3.8523994433224855)
        self.assertEqual(velocities.shape, (138, 3))
        self.assertAlmostEqual(velocities[10][0], 5.367132495161134e+6,
                               places=0)
        self.assertEqual(effective_masses.shape, (138, 3, 3))
        self.assertAlmostEqual(effective_masses[10][0][0], 0.21494134163522982)

    def test_get_energies_parallel(self):
        """Test getting the interpolated energy etc using interpolation."""

        # test just getting energy
        energies = self.interpolater_parallel.get_energies(
            self.kpoints, 25, return_velocity=False,
            return_effective_mass=False)
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 3.8523994433224855)

        # test energy + velocity
        energies, velocities = self.interpolater_parallel.get_energies(
            self.kpoints, 25, return_velocity=True,
            return_effective_mass=False)
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 3.8523994433224855)
        self.assertEqual(velocities.shape, (138, 3))
        self.assertAlmostEqual(velocities[10][0], 5.367132495161134e+6,
                               places=0)

        # test energy + effective_mass
        energies, effective_masses = self.interpolater_parallel.get_energies(
            self.kpoints, 25, return_velocity=False,
            return_effective_mass=True)
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 3.8523994433224855)
        self.assertEqual(effective_masses.shape, (138, 3, 3))
        self.assertAlmostEqual(effective_masses[10][0][0], 0.21494134163522982)

        # test energy + velocity + effective_mass
        energies, velocities, effective_masses = self.interpolater_parallel \
            .get_energies(self.kpoints, 25, return_velocity=True,
                          return_effective_mass=True)
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 3.8523994433224855)
        self.assertEqual(velocities.shape, (138, 3))
        self.assertAlmostEqual(velocities[10][0], 5.367132495161134e+6,
                               places=0)
        self.assertEqual(effective_masses.shape, (138, 3, 3))
        self.assertAlmostEqual(effective_masses[10][0][0], 0.21494134163522982)

    def test_get_energies_scissor(self):
        """Test scissoring of band energies."""

        # test valence band
        energies = self.interpolater.get_energies(
            self.kpoints, 25, return_velocity=False,
            return_effective_mass=False, scissor=1.)
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 3.8523994433224855 - 0.5)

        # test conduction band
        energies = self.interpolater.get_energies(
            self.kpoints, 33, return_velocity=False,
            return_effective_mass=False, scissor=1)
        self.assertAlmostEqual(energies[0], 7.3017006426 + 0.5)

    def test_get_energies_multiple_bands(self):
        """Test getting the interpolated data for multiple bands."""

        # test just getting energy
        energies = self.interpolater.get_energies(
            self.kpoints, [25, 35], return_velocity=False,
            return_effective_mass=False)
        self.assertEqual(energies.shape, (2, 138,))
        self.assertAlmostEqual(energies[0][0], 3.8523994433224855)
        self.assertAlmostEqual(energies[1][0], 9.594401585672195)

        # test energy + velocity
        energies, velocities = self.interpolater.get_energies(
            self.kpoints, [25, 35], return_velocity=True,
            return_effective_mass=False)
        self.assertEqual(energies.shape, (2, 138,))
        self.assertAlmostEqual(energies[0][0], 3.8523994433224855)
        self.assertAlmostEqual(energies[1][0], 9.594401585672195)
        self.assertEqual(velocities.shape, (2, 138, 3))
        self.assertAlmostEqual(velocities[0][10][0], 5.367132495161134e+6,
                               places=0)
        self.assertAlmostEqual(velocities[1][10][0], 1.1671401990442546e+8,
                               places=0)

        # test energy + effective_mass
        energies, effective_masses = self.interpolater.get_energies(
            self.kpoints, [25, 35], return_velocity=False,
            return_effective_mass=True)
        self.assertEqual(energies.shape, (2, 138,))
        self.assertAlmostEqual(energies[0][0], 3.8523994433224855)
        self.assertAlmostEqual(energies[1][0], 9.594401585672195)
        self.assertEqual(effective_masses.shape, (2, 138, 3, 3))
        self.assertAlmostEqual(effective_masses[0][10][0][0], 0.214941341635229)
        self.assertAlmostEqual(effective_masses[1][10][0][0], 0.008700893167441)

        # test energy + velocity + effective_mass
        energies, velocities, effective_masses = self.interpolater.get_energies(
            self.kpoints, [25, 35], return_velocity=True,
            return_effective_mass=True)
        self.assertEqual(energies.shape, (2, 138,))
        self.assertAlmostEqual(energies[0][0], 3.8523994433224855)
        self.assertAlmostEqual(energies[1][0], 9.594401585672195)
        self.assertEqual(velocities.shape, (2, 138, 3))
        self.assertAlmostEqual(velocities[0][10][0], 5.367132495161134e+6,
                               places=0)
        self.assertAlmostEqual(velocities[1][10][0], 1.1671401990442546e+8,
                               places=0)
        self.assertEqual(effective_masses.shape, (2, 138, 3, 3))
        self.assertAlmostEqual(effective_masses[0][10][0][0], 0.214941341635229)
        self.assertAlmostEqual(effective_masses[1][10][0][0], 0.008700893167441)

    def test_get_energies_all_bands(self):
        # test all bands
        energies, velocities, effective_masses = self.interpolater.get_energies(
            self.kpoints, None, return_velocity=True,
            return_effective_mass=True)
        self.assertEqual(energies.shape, (11, 138,))
        self.assertAlmostEqual(energies[0][0], 3.8523994433224855)
        self.assertAlmostEqual(energies[10][0], 9.594401585672195)
        self.assertEqual(velocities.shape, (11, 138, 3))
        self.assertAlmostEqual(velocities[0][10][0], 5.367132495161134e+6,
                               places=0)
        self.assertAlmostEqual(velocities[10][10][0], 1.1671401990442546e+8,
                               places=0)
        self.assertEqual(effective_masses.shape, (11, 138, 3, 3))
        self.assertAlmostEqual(effective_masses[0][10][0][0], 0.214941341635229)
        self.assertAlmostEqual(effective_masses[10][10][0][0], 0.00870089316744)

    def test_get_dos(self):
        """Test generating the interpolated DOS."""
        dos = self.interpolater.get_dos([10, 10, 10], emin=-10, emax=10,
                                        width=0.075)
        self.assertEqual(dos.shape, (20000, 2))
        self.assertEqual(dos[0][0], -10)
        self.assertAlmostEqual(dos[15000][1], 3.5362928578283337)

        # test normalising the DOS
        dos = self.interpolater.get_dos([10, 10, 10], emin=-10, emax=10,
                                        width=0.075)
        self.assertEqual(dos.shape, (20000, 2))
        self.assertEqual(dos[0][0], -10)
        self.assertAlmostEqual(dos[15000][1], 7.072609599611304)

        # test SPB dos
        dos = self.interpolater.get_dos([10, 10, 10], emin=-10, emax=10,
                                        width=0.075,
                                        minimum_single_parabolic_band=True)
        self.assertEqual(dos.shape, (20000, 2))
        self.assertEqual(dos[0][0], -10)
        self.assertAlmostEqual(dos[15000][1], 0.23993839328024136)

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
