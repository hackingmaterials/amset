# coding: utf-8

from __future__ import unicode_literals
import logging
import numpy as np
import os
import unittest

from amset import AMSET

test_dir = os.path.dirname(__file__)

class AmsetTest(unittest.TestCase):
    def setUp(self):
        self.model_params = {'bs_is_isotropic': True,
                             'elastic_scatterings': ['ACD', 'IMP', 'PIE'],
                             'inelastic_scatterings': ['POP']}
        self.performance_params = {'dE_min': 0.0001, 'nE_min': 2, 'Ecut': 0.7,
                                'parallel': True, 'BTE_iters': 5, 'nkdos':29}
    def tearDown(self):
        pass

    def test_poly_bands(self):

    def test_GaAs(self):
        # if norm(prop)/sq3 is imposed in map_to_egrid if bs_is_isotropic
        # expected_mu = {'ACD': 68036.7, 'IMP': 82349394.9, 'PIE': 172180.7,
        #                'POP': 10113.9, 'overall': 8173.4}

        expected_mu = {'ACD': 48397.6, 'IMP': 58026678.3, 'PIE': 111243.3,
                       'POP': 7478.1, 'overall': 6014.1}
        cube_path = os.path.join(test_dir, '..', 'test_files', 'GaAs')
        coeff_file = os.path.join(cube_path, 'fort.123_GaAs_1099kp')
        material_params = {'epsilon_s': 12.9, 'epsilon_inf': 10.9,
                'W_POP': 8.73, 'C_el': 139.7, 'E_D': {'n': 8.6, 'p': 8.6},
                'P_PIE': 0.052, 'scissor': 0.5818}
        amset = AMSET(calc_dir=cube_path, material_params=material_params,
                      model_params=self.model_params,
                      performance_params=self.performance_params,
                      dopings=[-2e15], temperatures=[300], k_integration=True,
                      e_integration=True, fermi_type='e',
                      loglevel=logging.ERROR)
        amset.run(coeff_file, kgrid_tp='very coarse')
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

        # TODO-JF: similar tests for k-integration (e.g. isotropic mobility)


if __name__ == '__main__':
    unittest.main()