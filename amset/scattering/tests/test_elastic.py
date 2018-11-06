# coding: utf-8

import os
import unittest

import numpy as np

from amset.scattering.elastic import (
    IonizedImpurityScattering, AcousticDeformationScattering,
    PiezoelectricScattering, DislocationScattering)

from monty.serialization import loadfn

test_dir = os.path.dirname(__file__)

props = {
    "epsilon_s": 12.18,
    "conc_imp": 30186274711047.44,
    "beta": 0.0013171355046518768,
    'elastic_constant': 139.7,
    'deformation_potential': 8.6,
    'piezoelectric_coeff': 0.052,
    'temperature': 300,
    'lattice_length': 10,
    'conc_dis': 0.1
}


class IonizedImpurityScatteringTest(unittest.TestCase):
    """Tests for calculating ionized impurity scattering."""

    def setUp(self):
        self.valley = loadfn(os.path.join(test_dir, 'valley.json.gz'))
        self.props = [props[p] for p in ['epsilon_s', 'conc_imp', 'beta']]

    def test_initialisation(self):
        # test that initialisation works with no errors
        IonizedImpurityScattering(True, self.valley, *self.props)

    def test_isotropic_scattering(self):
        # test isotropic scattering returns the correct rates
        imp = IonizedImpurityScattering(True, self.valley, *self.props)
        rates = imp.calculate_scattering()

        self.assertTrue(np.array_equal(rates.shape, (3020, 3)))
        self.assertAlmostEqual(rates[0][0], 1.52423861e+11, places=-3)
        self.assertAlmostEqual(rates[10][0], 1.41929972e+10, places=-3)

    def test_anisotropic_scattering(self):
        # test anisotropic scattering returns the correct rates
        imp = IonizedImpurityScattering(False, self.valley, *self.props)
        rates = imp.calculate_scattering()

        self.assertTrue(np.array_equal(rates.shape, (3020, 3)))
        self.assertAlmostEqual(rates[0][0], 1.63140197e+10, places=-3)
        self.assertAlmostEqual(rates[10][0], 6.56527665e+09, places=-3)


class AcousticDeformationScatteringTest(unittest.TestCase):
    """Tests for calculating acoustic deformation potential scattering."""

    def setUp(self):
        self.valley = loadfn(os.path.join(test_dir, 'valley.json.gz'))
        self.props = [
            props[p] for p in
            ['elastic_constant', 'deformation_potential', 'temperature']]

    def test_initialisation(self):
        # test that initialisation works with no errors
        AcousticDeformationScattering(True, self.valley, *self.props)

    def test_isotropic_scattering(self):
        # test isotropic scattering returns the correct rates
        acd = AcousticDeformationScattering(True, self.valley, *self.props)
        rates = acd.calculate_scattering()

        self.assertTrue(np.array_equal(rates.shape, (3020, 3)))
        self.assertAlmostEqual(rates[0][0], 1.83483058e+10, places=-3)
        self.assertAlmostEqual(rates[10][0], 4.09687033e+10, places=-3)

    def test_anisotropic_scattering(self):
        # test anisotropic scattering returns the correct rates
        acd = AcousticDeformationScattering(False, self.valley, *self.props)
        rates = acd.calculate_scattering()

        self.assertTrue(np.array_equal(rates.shape, (3020, 3)))
        self.assertAlmostEqual(rates[0][0], 1.91184789e+10, places=-3)
        self.assertAlmostEqual(rates[10][0], 4.57608182e+10, places=-3)


class PiezoelectricScatteringTest(unittest.TestCase):
    """Tests for calculating piezoelectric scattering."""

    def setUp(self):
        self.valley = loadfn(os.path.join(test_dir, 'valley.json.gz'))
        self.props = [props[p] for p in
                      ['epsilon_s', 'piezoelectric_coeff', 'temperature']]

    def test_initialisation(self):
        # test that initialisation works with no errors
        PiezoelectricScattering(True, self.valley, *self.props)

    def test_isotropic_scattering(self):
        # test isotropic scattering returns the correct rates
        pie = PiezoelectricScattering(True, self.valley, *self.props)
        rates = pie.calculate_scattering()

        self.assertTrue(np.array_equal(rates.shape, (3020, 3)))
        self.assertAlmostEqual(rates[0][0], 2.37029001e+11, places=-3)
        self.assertAlmostEqual(rates[10][0], 9.74438252e+10, places=-3)

    def test_anisotropic_scattering(self):
        # test anisotropic scattering returns the correct rates
        pie = PiezoelectricScattering(False, self.valley, *self.props)
        rates = pie.calculate_scattering()

        self.assertTrue(np.array_equal(rates.shape, (3020, 3)))
        self.assertAlmostEqual(rates[0][0], 1.51270944e+11, places=-3)
        self.assertAlmostEqual(rates[10][0], 1.22814850e+11, places=-3)


class DislocationScatteringTest(unittest.TestCase):
    """Tests for calculating dislocation scattering."""

    def setUp(self):
        self.valley = loadfn(os.path.join(test_dir, 'valley.json.gz'))
        self.props = [props[p] for p in
                      ['epsilon_s', 'beta', 'conc_dis', 'lattice_length']]

    def test_initialisation(self):
        # test that initialisation works with no errors
        DislocationScattering(True, self.valley, *self.props)

    def test_isotropic_scattering(self):
        # test isotropic scattering returns the correct rates
        dis = DislocationScattering(True, self.valley, *self.props)
        rates = dis.calculate_scattering()

        self.assertTrue(np.array_equal(rates.shape, (3020, 3)))
        self.assertAlmostEqual(rates[0][0], 1.63812051e+06, places=-3)
        self.assertAlmostEqual(rates[10][0], 1.24028364e+05, places=-3)

    def test_anisotropic_scattering(self):
        # test anisotropic scattering returns the correct rates
        dis = DislocationScattering(False, self.valley, *self.props)

        # check warning is raised for anisotropic case
        self.assertRaises(NotImplementedError, dis.calculate_scattering)