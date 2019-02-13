import os
import unittest
import numpy as np


from amset.interpolate.parabolic import ParabolicInterpolater
from pymatgen.io.vasp import Vasprun

test_dir = os.path.dirname(os.path.abspath(__file__))
amset_files = os.path.join(test_dir, '..', '..', '..', 'test_files',
                           'AlCuS2_mp-4979')


class TestParabolicInterpolater(unittest.TestCase):
    """Tests for interpolating a parabolic band structure."""

    def setUp(self):
        vr = Vasprun(os.path.join(amset_files, 'vasprun.xml'))
        bs = vr.get_band_structure()
        num_electrons = vr.parameters['NELECT']
        band_parameters = [[[[0.5, 0.5, 0.5], 0, 0.1],
                            [[0, 0, 0], 0.5, 0.2]],
                           [[[0, 0, 0], 1, 0.1]]]

        self.kpoints = np.array(vr.actual_kpoints)
        self.interpolater = ParabolicInterpolater(bs, num_electrons,
                                                  band_parameters)

    def test_initialisation(self):
        """Test coefficients and parameters are calculated correctly."""
        self.assertEqual(self.interpolater._band_mapping,
                         {30: 1, 31: 0, 32: 0, 33: 1})

    def test_get_energies(self):
        """Test getting the interpolated energy, velocity and effective mass."""

        # test just getting energy
        energies = self.interpolater.get_energies(
            self.kpoints, 31, return_velocity=False,
            return_effective_mass=False)
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 4.742)

        # test energy + velocity
        energies, velocities = self.interpolater.get_energies(
            self.kpoints, 31, return_velocity=True,
            return_effective_mass=False)
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 4.742)
        self.assertEqual(velocities.shape, (138, 3))
        self.assertAlmostEqual(velocities[10][0], 292836222.3145536)

        # test energy + effective_mass
        energies, effective_masses = self.interpolater.get_energies(
            self.kpoints, 31, return_velocity=False,
            return_effective_mass=True)
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 4.742)
        self.assertEqual(effective_masses.shape, (138, 3, 3))
        self.assertAlmostEqual(effective_masses[10][0][0], -0.2)

        # test energy + velocity + effective_mass
        energies, velocities, effective_masses = self.interpolater.get_energies(
            self.kpoints, 31, return_velocity=True,
            return_effective_mass=True)
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 4.742)
        self.assertEqual(velocities.shape, (138, 3))
        self.assertAlmostEqual(velocities[10][0], 292836222.3145536)
        self.assertEqual(effective_masses.shape, (138, 3, 3))
        self.assertAlmostEqual(effective_masses[10][0][0], -0.2)

    def test_get_energies_scissor(self):
        """Test scissoring of band energies."""

        # test valence band
        energies = self.interpolater.get_energies(
            self.kpoints, 31, return_velocity=False,
            return_effective_mass=False, scissor=1.)
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 4.742 - 0.5)

        # test conduction band
        energies = self.interpolater.get_energies(
            self.kpoints, 32, return_velocity=False,
            return_effective_mass=False, scissor=1.)
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 7.5015 + 0.5)

    def test_get_energies_multiple_bands(self):
        """Test getting the interpolated data for multiple bands."""

        # test just getting energy
        energies = self.interpolater.get_energies(
            self.kpoints, [31, 32], return_velocity=False,
            return_effective_mass=False)
        self.assertEqual(energies.shape, (2, 138,))
        self.assertAlmostEqual(energies[0][0], 4.742)
        self.assertAlmostEqual(energies[1][0], 7.5015)

        # test energy + velocity
        energies, velocities = self.interpolater.get_energies(
            self.kpoints, [31, 32], return_velocity=True,
            return_effective_mass=False)
        self.assertEqual(energies.shape, (2, 138,))
        self.assertAlmostEqual(energies[0][0], 4.742)
        self.assertAlmostEqual(energies[1][0], 7.5015)
        self.assertEqual(velocities.shape, (2, 138, 3))
        self.assertAlmostEqual(velocities[0][10][0], 292836222.3145536)
        self.assertAlmostEqual(velocities[1][10][0], 292836222.3145536)

        # test energy + effective_mass
        energies, effective_masses = self.interpolater.get_energies(
            self.kpoints, [31, 32], return_velocity=False,
            return_effective_mass=True)
        self.assertEqual(energies.shape, (2, 138,))
        self.assertAlmostEqual(energies[0][0], 4.742)
        self.assertAlmostEqual(energies[1][0], 7.5015)
        self.assertEqual(effective_masses.shape, (2, 138, 3, 3))
        self.assertAlmostEqual(effective_masses[0][10][0][0], -0.2)
        self.assertAlmostEqual(effective_masses[1][10][0][0], 0.2)

        # test energy + velocity + effective_mass
        energies, velocities, effective_masses = self.interpolater.get_energies(
            self.kpoints, [31, 32], return_velocity=True,
            return_effective_mass=True)
        self.assertEqual(energies.shape, (2, 138,))
        self.assertAlmostEqual(energies[0][0], 4.742)
        self.assertAlmostEqual(energies[1][0], 7.5015)
        self.assertEqual(velocities.shape, (2, 138, 3))
        self.assertAlmostEqual(velocities[0][10][0], 292836222.3145536)
        self.assertAlmostEqual(velocities[1][10][0], 292836222.3145536)
        self.assertEqual(effective_masses.shape, (2, 138, 3, 3))
        self.assertAlmostEqual(effective_masses[0][10][0][0], -0.2)
        self.assertAlmostEqual(effective_masses[1][10][0][0], 0.2)

    def test_get_energies_all_bands(self):
        # test all bands
        energies, velocities, effective_masses = self.interpolater.get_energies(
            self.kpoints, None, return_velocity=True,
            return_effective_mass=True)
        self.assertEqual(energies.shape, (4, 138,))
        self.assertAlmostEqual(energies[0][0], 4.242)
        self.assertAlmostEqual(energies[1][0], 4.742)
        self.assertEqual(velocities.shape, (4, 138, 3))
        self.assertAlmostEqual(velocities[0][10][0], 585672444.6291072)
        self.assertAlmostEqual(velocities[1][10][0], 292836222.3145536)
        self.assertEqual(effective_masses.shape, (4, 138, 3, 3))
        self.assertAlmostEqual(effective_masses[0][10][0][0], -0.1)
        self.assertAlmostEqual(effective_masses[1][10][0][0], -0.2)

    def test_get_dos(self):
        """Test generating the interpolated DOS."""
        dos = self.interpolater.get_dos([10, 10, 10], emin=-10, emax=10,
                                        width=0.075)
        self.assertEqual(dos.shape, (20000, 2))
        self.assertEqual(dos[0][0], -10)
        self.assertAlmostEqual(dos[10000][1], 0.04839047875200659)

        # test normalising the DOS
        dos = self.interpolater.get_dos([10, 10, 10], emin=-10, emax=10,
                                        width=0.075, normalize=True)
        self.assertEqual(dos.shape, (20000, 2))
        self.assertEqual(dos[0][0], -10)
        self.assertAlmostEqual(dos[10000][1], 0.2397190555784462)

        # test single parabolic band DOS
        dos = self.interpolater.get_dos([10, 10, 10], emin=-10, emax=10,
                                        width=0.075, normalize=True,
                                        minimum_single_parabolic_band=True)
        self.assertEqual(dos.shape, (20000, 2))
        self.assertEqual(dos[0][0], -10)
        self.assertAlmostEqual(dos[10000][1], 1.890607189170457e-06)

    def test_get_extrema(self):
        """Test getting the band structure extrema."""

        # test VBM
        extrema = self.interpolater.get_extrema(31, e_cut=1)
        self.assertEqual(len(extrema), 2)
        np.testing.assert_array_almost_equal(extrema[0], [0.5, -0.5, -1], 10)
        np.testing.assert_array_almost_equal(extrema[1], [0., 0., 0.], 10)

        # test CBM
        extrema = self.interpolater.get_extrema(32, e_cut=1.)
        self.assertEqual(len(extrema), 2)
        np.testing.assert_array_almost_equal(extrema[0], [0.5, -0.5, -1], 10)
        np.testing.assert_array_almost_equal(extrema[1], [0., 0., 0.], 10)

        # test cut-off
        extrema = self.interpolater.get_extrema(32, e_cut=0.2)
        self.assertEqual(len(extrema), 1)
        np.testing.assert_array_almost_equal(extrema[0], [0.5, -0.5, -1], 10)
