import os
import unittest
import numpy as np

from monty.serialization import loadfn

from amset.interpolate.boltztrap1 import BoltzTraP1Interpolater, \
    BoltzTraP1Parameters
from pymatgen.io.vasp import Vasprun

test_dir = os.path.dirname(os.path.abspath(__file__))
amset_files = os.path.join(test_dir, '..', '..', '..', 'test_files',
                           'AlCuS2_mp-4979')


class TestBoltzTraP1Interpolater(unittest.TestCase):
    """Tests for interpolating a band structure using BoltzTraP1.

    Note these tests do not cover regenerating the
    """

    def setUp(self):
        self.correct_energies = loadfn("bzt1_energies.json")
        self.correct_velocities = loadfn("bzt1_velocities.json")
        self.correct_effective_masses = loadfn("bzt1_effective_masses.json")

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
        self.assertEqual(params.min_band, 26)
        self.assertEqual(params.max_band, 36)
        self.assertEqual(params.allowed_ibands,
                         {26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36})
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
        coeff = self.interpolater._get_interpolation_coefficients(26)
        self.assertEqual(coeff.shape, (2154,))
        self.assertEqual(coeff[0], -0.11634230012183)

        # test getting multiple band coefficients
        coeff = self.interpolater._get_interpolation_coefficients([26, 27])
        self.assertEqual(coeff.shape, (2, 2154))
        self.assertEqual(coeff[0][0], -0.11634230012183)
        self.assertEqual(coeff[1][0], -0.10136164260112)

        # test error thrown when band out of range
        self.assertRaises(
            ValueError, self.interpolater._get_interpolation_coefficients, 25)

    def test_get_energy(self):
        """Test getting the interpolated energy, velocity and effective mass."""

        # test just getting energy
        energy = self.interpolater.get_energy(
            self.kpoints[10], 26, return_velocity=False,
            return_effective_mass=False)
        self.assertEqual(energy, self.correct_energies[10])

        # test energy + velocity
        energy, velocity = self.interpolater.get_energy(
            self.kpoints[10], 26, return_velocity=True,
            return_effective_mass=False)
        self.assertEqual(energy, self.correct_energies[10])
        np.testing.assert_array_almost_equal(
            velocity, self.correct_velocities[10])

        # test energy + effective_mass
        energy, effective_mass = self.interpolater.get_energy(
            self.kpoints[10], 26, return_velocity=False,
            return_effective_mass=True)
        self.assertEqual(energy, self.correct_energies[10])
        np.testing.assert_array_almost_equal(
            effective_mass, self.correct_effective_masses[10])

        # test energy + velocity + effective_mass
        energy, velocity, effective_mass = self.interpolater.get_energy(
            self.kpoints[10], 26, return_velocity=True,
            return_effective_mass=True)
        self.assertEqual(energy, self.correct_energies[10])
        np.testing.assert_array_almost_equal(
            velocity, self.correct_velocities[10])
        np.testing.assert_array_almost_equal(
            effective_mass, self.correct_effective_masses[10])

    def test_get_energies_non_parallel(self):
        """Test getting the interpolated energy, velocity and effective mass."""

        # test just getting energy
        energies = self.interpolater.get_energies(
            self.kpoints, 26, return_velocity=False,
            return_effective_mass=False)
        np.testing.assert_array_almost_equal(energies, self.correct_energies)

        # test energy + velocity
        energies, velocities = self.interpolater.get_energies(
            self.kpoints, 26, return_velocity=True,
            return_effective_mass=False)
        np.testing.assert_array_almost_equal(energies, self.correct_energies)
        np.testing.assert_array_almost_equal(
            velocities, self.correct_velocities)

        # test energy + effective_mass
        energies, effective_masses = self.interpolater.get_energies(
            self.kpoints, 26, return_velocity=False,
            return_effective_mass=True)
        np.testing.assert_array_almost_equal(energies, self.correct_energies)
        np.testing.assert_array_almost_equal(
            effective_masses, self.correct_effective_masses)

        # test energy + velocity + effective_mass
        energies, velocities, effective_masses = self.interpolater.get_energies(
            self.kpoints, 26, return_velocity=True,
            return_effective_mass=True)
        np.testing.assert_array_almost_equal(energies, self.correct_energies)
        np.testing.assert_array_almost_equal(
            velocities, self.correct_velocities)
        np.testing.assert_array_almost_equal(
            effective_masses, self.correct_effective_masses)

    def test_get_energies_parallel(self):
        """Test getting the interpolated energy etc using interpolation."""

        # test just getting energy
        energies = self.interpolater_parallel.get_energies(
            self.kpoints, 26, return_velocity=False,
            return_effective_mass=False)
        np.testing.assert_array_almost_equal(energies, self.correct_energies)

        # test energy + velocity
        energies, velocities = self.interpolater_parallel.get_energies(
            self.kpoints, 26, return_velocity=True,
            return_effective_mass=False)
        np.testing.assert_array_almost_equal(energies, self.correct_energies)
        np.testing.assert_array_almost_equal(
            velocities, self.correct_velocities)

        # test energy + effective_mass
        energies, effective_masses = self.interpolater_parallel.get_energies(
            self.kpoints, 26, return_velocity=False,
            return_effective_mass=True)
        np.testing.assert_array_almost_equal(energies, self.correct_energies)
        np.testing.assert_array_almost_equal(
            effective_masses, self.correct_effective_masses)

        # test energy + velocity + effective_mass
        energies, velocities, effective_masses = self.interpolater_parallel \
            .get_energies(self.kpoints, 26, return_velocity=True,
                          return_effective_mass=True)
        np.testing.assert_array_almost_equal(energies, self.correct_energies)
        np.testing.assert_array_almost_equal(
            velocities, self.correct_velocities)
        np.testing.assert_array_almost_equal(
            effective_masses, self.correct_effective_masses)
