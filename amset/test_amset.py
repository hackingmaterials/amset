# coding: utf-8

from __future__ import unicode_literals
import logging
import numpy as np
import os
import unittest

from amset import AMSET

from tools import rel_diff

test_dir = os.path.dirname(__file__)
#test_dir = 'amset/amset'

class AmsetTest(unittest.TestCase):
    def setUp(self):
        self.model_params = {'bs_is_isotropic': True,
                             'elastic_scatterings': ['ACD', 'IMP', 'PIE'],
                             'inelastic_scatterings': ['POP']}
        self.performance_params = {'dE_min': 0.0001, 'nE_min': 2, 'Ecut': 0.7,
                                'parallel': True, 'BTE_iters': 5, 'nkdos':29}
        self.GaAs_params = {'epsilon_s': 12.9, 'epsilon_inf': 10.9,
                'W_POP': 8.73, 'C_el': 139.7, 'E_D': {'n': 8.6, 'p': 8.6},
                'P_PIE': 0.052, 'scissor': 0.5818}
        self.GaAs_path = os.path.join(test_dir, '..', 'test_files', 'GaAs')
        self.GaAs_cube = os.path.join(self.GaAs_path, "fort.123_GaAs_1099kp")

    def tearDown(self):
        pass


    def test_poly_bands(self):
        print('testing test_poly_bands...')
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
        print('testing test_GaAs_isotropic...')
        # if norm(prop)/sq3 is imposed in map_to_egrid if bs_is_isotropic
        # expected_mu = {'ACD': 68036.7, 'IMP': 82349394.9, 'PIE': 172180.7,
        #                'POP': 10113.9, 'overall': 8173.4}

        expected_mu = {'ACD': 48397.6, 'IMP': 58026678.3, 'PIE': 111243.3,
                       'POP': 7484.6, 'overall': 6017.7, 'average': 6124.6}
        amset = AMSET(calc_dir=self.GaAs_path, material_params=self.GaAs_params,
                      model_params=self.model_params,
                      performance_params=self.performance_params,
                      dopings=[-2e15], temperatures=[300], k_integration=True,
                      e_integration=True, fermi_type='e',
                      loglevel=logging.ERROR)
        amset.run(self.GaAs_cube, kgrid_tp='very coarse', write_outputs=False)
        egrid = amset.egrid
        kgrid = amset.kgrid

        # check general characteristics of the grid
        self.assertEqual(kgrid['n']['velocity'][0].shape[0], 292)
        mean_v = np.mean(kgrid['n']['velocity'][0], axis=0)
        self.assertAlmostEqual(np.std(mean_v), 0.00, places=2) # isotropic BS
        self.assertAlmostEqual(mean_v[0], 1.93656060e7, places=1) # zeroth band

        # check mobility values
        for mu in expected_mu.keys():
            self.assertAlmostEqual(np.std( # test isotropic
                egrid['n']['mobility'][mu][-2e15][300]), 0.00, places=2)
            self.assertAlmostEqual(egrid['n']['mobility'][mu][-2e15][300][0],
                    expected_mu[mu], places=1)


    def test_GaAs_isotropic_k(self):
        print('testing test_GaAs_isotropic_k...')
        # if norm(prop)/sq3 is imposed in map_to_egrid if bs_is_isotropic
        # expected_mu = {'ACD': 68036.7, 'IMP': 82349394.9, 'PIE': 172180.7,
        #                'POP': 10113.9, 'overall': 8173.4}

        expected_mu = {'ACD': 35746.857, 'IMP': 52907945.168, 'PIE': 112505.173,
                       'POP': 4994.131, 'overall': 4327.095, 'average': 4217.329}
        performance_params = dict(self.performance_params)
        performance_params["max_nbands"] = 1
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
        print('testing test_GaAs_anisotropic...')
        expected_mu = {'ACD': 44114.54, 'IMP': 56039469.34, 'PIE': 112262.60,
                       'POP': 8539.31, 'overall': 6743.91, 'average': 6724.98}
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
                egrid['n']['mobility'][mu][-2e15][300]), 0.01*\
                np.mean(egrid['n']['mobility'][mu][-2e15][300]))
            self.assertLess(rel_diff(egrid['n']['mobility'][mu][-2e15][300][0], expected_mu[mu]), 0.002)


    def test_GaAs_anisotropic_k(self):
        print('testing test_GaAs_isotropic_k...')
        # if norm(prop)/sq3 is imposed in map_to_egrid if bs_is_isotropic
        # expected_mu = {'ACD': 68036.7, 'IMP': 82349394.9, 'PIE': 172180.7,
        #                'POP': 10113.9, 'overall': 8173.4}

        expected_mu = {'overall': 4327.095}
        performance_params = dict(self.performance_params)
        performance_params["max_nbands"] = 1
        amset = AMSET(calc_dir=self.GaAs_path, material_params=self.GaAs_params,
                      model_params=self.model_params,
                      performance_params=performance_params,
                      dopings=[-3e13], temperatures=[300], k_integration=True,
                      e_integration=False, fermi_type='k',
                      loglevel=logging.ERROR)
        amset.run(self.GaAs_cube, kgrid_tp='very fine', write_outputs=False, test_k_anisotropic=True)
        mobility = amset.mobility
        kgrid = amset.kgrid

        # check mobility values
        for mu in expected_mu.keys():
            diff = np.std(mobility['n'][mu][-3e13][300])
            avg = np.mean(mobility['n'][mu][-3e13][300])
            self.assertLess(diff / avg, 0.002)
            diff = abs(mobility['n'][mu][-3e13][300][0] - expected_mu[mu])
            avg = (mobility['n'][mu][-3e13][300][0] + expected_mu[mu]) / 2
            self.assertTrue(diff / avg <= 0.01)


if __name__ == '__main__':
    unittest.main()