# coding: utf-8

from __future__ import unicode_literals

import json
import logging
import numpy as np
import os
import unittest

from amset import AMSET
from tools import rel_diff

test_dir = os.path.dirname(__file__)

class AmsetTest(unittest.TestCase):
    def setUp(self):
        self.model_params = {'bs_is_isotropic': True,
                             'elastic_scatterings': ['ACD', 'IMP', 'PIE'],
                             'inelastic_scatterings': ['POP']}
        self.performance_params = {'dE_min': 0.0001, 'nE_min': 2,
                'parallel': True, 'BTE_iters': 5,'nkdos':29, 'max_nbands': 1,
                                'max_normk': 2, 'Ecut': 0.4}
        self.GaAs_params = {'epsilon_s': 12.9, 'epsilon_inf': 10.9,
                'W_POP': 8.73, 'C_el': 139.7, 'E_D': {'n': 8.6, 'p': 8.6},
                'P_PIE': 0.052, 'scissor': 0.5818}
        self.GaAs_path = os.path.join(test_dir, '..', 'test_files', 'GaAs')
        self.GaAs_cube = os.path.join(self.GaAs_path, "fort.123_GaAs_1099kp")

    def tearDown(self):
        pass


    def test_poly_bands(self):
        print('\ntesting test_poly_bands...')
        mass = 0.25
        self.model_params['poly_bands'] = [[[[0.0, 0.0, 0.0], [0.0, mass]]]]
        amset = AMSET(calc_dir=self.GaAs_path,material_params=self.GaAs_params,
                      model_params=self.model_params,
                      performance_params=self.performance_params,
                      dopings=[-2e15], temperatures=[300], k_integration=True,
                      e_integration=True, fermi_type='k',
                      loglevel=logging.ERROR)
        amset.run(self.GaAs_cube, kgrid_tp='poly_band', write_outputs=False)
        egrid = amset.egrid
        diff = abs(np.array(amset.mobility['n']['ACD'][-2e15][300]) - \
                   np.array(egrid['n']['mobility']['SPB_ACD'][-2e15][300]))
        avg = (amset.mobility['n']['ACD'][-2e15][300] + \
               egrid['n']['mobility']['SPB_ACD'][-2e15][300]) / 2
        self.assertTrue((diff / avg <= 0.01).all())


    def test_GaAs_isotropic(self):
        print('\ntesting test_GaAs_isotropic...')
        # if norm(prop)/sq3 is imposed in map_to_egrid if bs_is_isotropic
        # expected_mu = {'ACD': 68036.7, 'IMP': 82349394.9, 'PIE': 172180.7,
        #                'POP': 10113.9, 'overall': 8173.4}

        expected_mu = {'ACD': 52617.19, 'IMP': 154780.49, 'PIE': 111864.79,
                       'POP': 7706.76, 'overall': 5432.38, 'average': 6091.56}
        amset = AMSET(calc_dir=self.GaAs_path, material_params=self.GaAs_params,
                      model_params=self.model_params,
                      performance_params=self.performance_params,
                      dopings=[-2e15], temperatures=[300], k_integration=False,
                      e_integration=True, fermi_type='e',
                      loglevel=logging.ERROR)
        amset.run(self.GaAs_cube, kgrid_tp='very coarse', write_outputs=False)
        egrid = amset.egrid
        kgrid = amset.kgrid

        # check general characteristics of the grid
        self.assertEqual(kgrid['n']['velocity'][0].shape[0], 100)
        mean_v = np.mean(kgrid['n']['velocity'][0], axis=0)
        self.assertAlmostEqual(np.std(mean_v), 0.00, places=2) # isotropic BS
        self.assertAlmostEqual(mean_v[0], 32253886.41, places=1) # zeroth band

        # check mobility values
        for mu in expected_mu.keys():
            self.assertAlmostEqual(np.std( # test isotropic
                egrid['n']['mobility'][mu][-2e15][300]), 0.00, places=2)
            self.assertAlmostEqual(egrid['n']['mobility'][mu][-2e15][300][0],
                    expected_mu[mu], places=1)


    def test_GaAs_isotropic_k(self):
        print('\ntesting test_GaAs_isotropic_k...')
        # if norm(prop)/sq3 is imposed in map_to_egrid if bs_is_isotropic
        # expected_mu = {'ACD': 68036.7, 'IMP': 82349394.9, 'PIE': 172180.7,
        #                'POP': 10113.9, 'overall': 8173.4}

        expected_mu = {'ACD': 101397.69, 'IMP': 8496857.31, 'PIE': 325384.23,
                       'POP': 14431.63, 'overall': 12552.64, 'average': 12143.97}
        performance_params = dict(self.performance_params)
        amset = AMSET(calc_dir=self.GaAs_path, material_params=self.GaAs_params,
                      model_params=self.model_params,
                      performance_params=performance_params,
                      dopings=[-3e13], temperatures=[300], k_integration=True,
                      e_integration=False, fermi_type='k',
                      loglevel=logging.ERROR)
        amset.run(self.GaAs_cube, kgrid_tp='very coarse', write_outputs=False)
        mobility = amset.mobility
        kgrid = amset.kgrid

        # check mobility values
        for mu in expected_mu.keys():
            diff = np.std(mobility['n'][mu][-3e13][300])
            avg = np.mean(mobility['n'][mu][-3e13][300])
            self.assertLess(diff / avg, 0.002)
            self.assertAlmostEqual(mobility['n'][mu][-3e13][300][0],
                                   expected_mu[mu], places=2)


    def test_GaAs_anisotropic(self):
        print('\ntesting test_GaAs_anisotropic...')
        expected_mu = {'ACD': 47957.47, 'IMP': 139492.12, 'PIE': 112012.98,
                       'POP': 8436.67, 'overall': 5874.23, 'average': 6431.76}
        amset = AMSET(calc_dir=self.GaAs_path,
                      material_params=self.GaAs_params,
                      model_params={'bs_is_isotropic': False,
                             'elastic_scatterings': ['ACD', 'IMP', 'PIE'],
                             'inelastic_scatterings': ['POP']},
                      performance_params=self.performance_params,
                      dopings=[-2e15], temperatures=[300], k_integration=False,
                      e_integration=True, fermi_type='e',
                      loglevel=logging.ERROR)
        amset.run(self.GaAs_cube, kgrid_tp='very coarse', write_outputs=False)
        egrid = amset.egrid
        # check mobility values
        for mu in expected_mu.keys():
            self.assertLessEqual(np.std(  # GaAs band structure is isotropic
                egrid['n']['mobility'][mu][-2e15][300]), 0.02*\
                np.mean(egrid['n']['mobility'][mu][-2e15][300]))
            self.assertLess(rel_diff(egrid['n']['mobility'][mu][-2e15][300][0], expected_mu[mu]), 0.02)

    def test_defaults(self):
        print('\ntesting test_defaults...')
        amset = AMSET(self.GaAs_path, material_params={'epsilon_s': 12.9})
        amset.write_input_files()
        with open("material_params.json", "r") as fp:
            material_params = json.load(fp)
        with open("model_params.json", "r") as fp:
            model_params = json.load(fp)
        with open("performance_params.json", "r") as fp:
            performance_params = json.load(fp)

        self.assertEqual(material_params['epsilon_inf'], None)
        self.assertEqual(material_params['W_POP'], None)
        self.assertEqual(material_params['scissor'], 0.0)
        self.assertEqual(material_params['P_PIE'], 0.15)
        self.assertEqual(material_params['E_D'], None)
        self.assertEqual(material_params['N_dis'], 0.1)

        self.assertEqual(model_params['bs_is_isotropic'], True)
        self.assertEqual(model_params['elastic_scatterings'], ['IMP', 'PIE'])
        self.assertEqual(model_params['inelastic_scatterings'], [])

        self.assertEqual(performance_params['max_nbands'], None)
        self.assertEqual(performance_params['max_normk'], 2)
        self.assertEqual(performance_params['dE_min'], 0.0001)
        self.assertEqual(performance_params['nkdos'], 29)
        self.assertEqual(performance_params['dos_bwidth'], 0.05)
        self.assertEqual(performance_params['nkdos'], 29)


if __name__ == '__main__':
    unittest.main()