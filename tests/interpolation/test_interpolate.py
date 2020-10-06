import os
import unittest

import numpy as np
from numpy.testing import assert_array_equal
from pymatgen import Spin
from pymatgen.io.vasp import Vasprun

from amset.interpolation.bandstructure import Interpolator
from amset.log import initialize_amset_logger

test_dir = os.path.dirname(os.path.abspath(__file__))
gaas_files = os.path.join(test_dir, "..", "..", "..", "examples", "GaAs")
tin_dioxide_files = os.path.join(test_dir, "..", "..", "..", "examples", "SnO2")
si_files = os.path.join(test_dir, "..", "..", "..", "examples", "Si")
pbs_files = os.path.join(test_dir, "..", "..", "..", "examples", "PbS")


@unittest.skip("Outdated tests")
class TestBoltzTraP2Interpolater(unittest.TestCase):
    """Tests for interpolating a band structure using BoltzTraP2."""

    def setUp(self):
        vr = Vasprun(
            os.path.join(gaas_files, "vasprun.xml.gz"), parse_projected_eigen=True
        )
        bs = vr.get_band_structure()
        num_electrons = vr.parameters["NELECT"]

        self.kpoints = np.array(vr.actual_kpoints)
        self.interpolater = Interpolator(
            bs, num_electrons, interpolate_projections=True, interpolation_factor=1
        )

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
            self.kpoints, 25, return_velocity=False, curvature=False
        )
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 3.852399483908641)

        # test energy + velocity
        energies, velocities = self.interpolater.get_energies(
            self.kpoints, 25, return_velocity=True, curvature=False
        )
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 3.852399483908641)
        self.assertEqual(velocities.shape, (138, 3))
        self.assertAlmostEqual(velocities[10][0], 5.401166156893334e6, places=0)

        # test energy + effective_mass
        energies, effective_masses = self.interpolater.get_energies(
            self.kpoints, 25, return_velocity=False, curvature=True
        )
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 3.852399483908641)
        self.assertEqual(effective_masses.shape, (138, 3, 3))
        self.assertAlmostEqual(effective_masses[10][0][0], 0.2242569613626494)

        # test energy + velocity + effective_mass
        energies, velocities, effective_masses = self.interpolater.get_energies(
            self.kpoints, 25, return_velocity=True, curvature=True
        )
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 3.852399483908641)
        self.assertEqual(velocities.shape, (138, 3))
        self.assertAlmostEqual(velocities[10][0], 5.401166156893334e6, places=0)
        self.assertEqual(effective_masses.shape, (138, 3, 3))
        self.assertAlmostEqual(effective_masses[10][0][0], 0.2242569613626494)

    def test_get_energies_scissor(self):
        """Test scissoring of band energies."""

        # test valence band
        energies = self.interpolater.get_energies(
            self.kpoints, 25, return_velocity=False, curvature=False, scissor=1.0
        )
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 3.852399483908641 - 0.5)

        # test conduction band
        energies = self.interpolater.get_energies(
            self.kpoints, 33, return_velocity=False, curvature=False, scissor=1.0
        )
        self.assertEqual(energies.shape, (138,))
        self.assertAlmostEqual(energies[0], 7.301700765 + 0.5)

    def test_get_energies_multiple_bands(self):
        """Test getting the interpolated data for multiple bands."""
        # test just getting energy
        energies = self.interpolater.get_energies(
            self.kpoints, [25, 35], return_velocity=False, curvature=False
        )
        self.assertEqual(energies.shape, (2, 138))
        self.assertAlmostEqual(energies[0][0], 3.852399483908641)
        self.assertAlmostEqual(energies[1][0], 9.594401616456384)

        # test energy + velocity
        energies, velocities = self.interpolater.get_energies(
            self.kpoints, [25, 35], return_velocity=True, curvature=False
        )
        self.assertEqual(energies.shape, (2, 138))
        self.assertAlmostEqual(energies[0][0], 3.852399483908641)
        self.assertAlmostEqual(energies[1][0], 9.594401616456384)
        self.assertEqual(velocities.shape, (2, 138, 3))
        self.assertAlmostEqual(velocities[0][10][0], 5.401166156893334e6, places=0)
        self.assertAlmostEqual(velocities[1][10][0], 1.1686720831758736e8, places=0)

        # test energy + effective_mass
        energies, effective_masses = self.interpolater.get_energies(
            self.kpoints, [25, 35], return_velocity=False, curvature=True
        )
        self.assertEqual(energies.shape, (2, 138))
        self.assertAlmostEqual(energies[0][0], 3.852399483908641)
        self.assertAlmostEqual(energies[1][0], 9.594401616456384)
        self.assertEqual(effective_masses.shape, (2, 138, 3, 3))
        self.assertAlmostEqual(effective_masses[0][10][0][0], 0.224256961362649)
        self.assertAlmostEqual(effective_masses[1][10][0][0], 0.009057103344700)

        # test energy + velocity + effective_mass
        energies, velocities, effective_masses = self.interpolater.get_energies(
            self.kpoints, [25, 35], return_velocity=True, curvature=True
        )
        self.assertEqual(energies.shape, (2, 138))
        self.assertAlmostEqual(energies[0][0], 3.852399483908641)
        self.assertAlmostEqual(energies[1][0], 9.594401616456384)
        self.assertEqual(velocities.shape, (2, 138, 3))
        self.assertAlmostEqual(velocities[0][10][0], 5.401166156893334e6, places=0)
        self.assertAlmostEqual(velocities[1][10][0], 1.1686720831758736e8, places=0)
        self.assertEqual(effective_masses.shape, (2, 138, 3, 3))
        self.assertAlmostEqual(effective_masses[0][10][0][0], 0.224256961362649)
        self.assertAlmostEqual(effective_masses[1][10][0][0], 0.009057103344700)

    def test_get_energies_all_bands(self):
        # test all bands
        energies, velocities, effective_masses = self.interpolater.get_energies(
            self.kpoints, None, return_velocity=True, curvature=True
        )
        self.assertEqual(energies.shape, (96, 138))
        self.assertAlmostEqual(energies[25][0], 3.852399483908641)
        self.assertAlmostEqual(energies[35][0], 9.594401616456384)
        self.assertEqual(velocities.shape, (96, 138, 3))
        self.assertAlmostEqual(velocities[25][10][0], 5.401166156893334e6, places=0)
        self.assertAlmostEqual(velocities[35][10][0], 1.1686720831758736e8, places=0)
        self.assertEqual(effective_masses.shape, (96, 138, 3, 3))
        self.assertAlmostEqual(effective_masses[25][10][0][0], 0.22425696136264)
        self.assertAlmostEqual(effective_masses[35][10][0][0], 0.00905710334470)

    def test_get_dos(self):
        """Test generating the interpolated DOS."""
        dos = self.interpolater.get_dos([10, 10, 10], emin=-10, emax=10, width=0.075)
        self.assertEqual(dos.shape, (20000, 2))
        self.assertEqual(dos[0][0], -10)
        self.assertAlmostEqual(dos[15000][1], 3.5362612128412807)

    def test_get_energies_symprec(self):
        # vr = Vasprun(os.path.join(tin_dioxide_files, 'vasprun.xml.gz'),
        #              parse_projected_eigen=True)
        vr = Vasprun(
            os.path.join(pbs_files, "vasprun.xml.gz"), parse_projected_eigen=True
        )
        bs = vr.get_band_structure()
        num_electrons = vr.parameters["NELECT"]
        interpolater = Interpolator(
            bs, num_electrons, interpolate_projections=True, interpolation_factor=1
        )

        ir_kpoints, weights, kpoints, ir_kpoints_idx, ir_to_full_idx = get_kpoints(
            [13, 15, 29],
            vr.final_structure,
            boltztrap_ordering=True,
            return_full_kpoints=True,
        )

        initialize_amset_logger()

        (
            energies,
            velocities,
            curvature,
            projections,
            sym_info,
        ) = interpolater.get_energies(
            kpoints,
            None,
            return_velocity=True,
            atomic_units=True,
            return_curvature=True,
            return_projections=True,
            symprec=0.1,
            return_vel_outer_prod=True,
            return_kpoint_mapping=True,
        )

        (
            energies_no_sym,
            velocities_no_sym,
            curvature_no_sym,
            projections_no_sym,
        ) = interpolater.get_energies(
            kpoints,
            None,
            return_velocity=True,
            atomic_units=True,
            return_curvature=True,
            return_projections=True,
            return_vel_outer_prod=True,
            symprec=None,
        )

        np.testing.assert_array_equal(ir_to_full_idx, sym_info["ir_to_full_idx"])
        np.testing.assert_array_equal(ir_kpoints_idx, sym_info["ir_kpoints_idx"])

        # print(velocities[Spin.up][5, :, :, -3:])
        # print(velocities_no_sym[Spin.up][5, :, :, -3:])
        # print(sym_info["ir_to_full_idx"][-10:])

        np.testing.assert_array_almost_equal(
            energies[Spin.up], energies_no_sym[Spin.up], decimal=12
        )
        np.testing.assert_array_almost_equal(
            velocities[Spin.up], velocities_no_sym[Spin.up], decimal=12
        )
        np.testing.assert_array_almost_equal(
            curvature[Spin.up], curvature_no_sym[Spin.up], decimal=12
        )

        for l in projections[Spin.up]:
            np.testing.assert_array_almost_equal(
                projections[Spin.up][l], projections_no_sym[Spin.up][l]
            )

    def test_get_energies_interpolater(self):

        initialize_amset_logger()

        amset_data = self.interpolater.get_amset_data()

        energies, velocities, projections, sym_info = self.interpolater.get_energies(
            amset_data.kpoints,
            None,
            return_velocity=True,
            atomic_units=True,
            curvature=False,
            return_projections=True,
            symprec=0.1,
            return_vel_outer_prod=True,
            return_kpoint_mapping=True,
        )

        np.testing.assert_array_almost_equal(
            energies[Spin.up], amset_data.energies[Spin.up]
        )
        np.testing.assert_array_almost_equal(
            velocities[Spin.up], amset_data.velocities_product[Spin.up]
        )

        for l in projections[Spin.up]:
            np.testing.assert_array_almost_equal(
                projections[Spin.up][l], amset_data._projections[Spin.up][l]
            )

        np.testing.assert_array_equal(
            sym_info["ir_kpoints_idx"], amset_data.ir_kpoints_idx
        )
        np.testing.assert_array_equal(
            sym_info["ir_to_full_idx"], amset_data.ir_to_full_kpoint_mapping
        )
