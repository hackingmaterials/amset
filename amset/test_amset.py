# coding: utf-8

import json
import logging
import numpy as np
import os
import shutil
import unittest
from amset.core import Amset
from copy import deepcopy

test_dir = os.path.dirname(__file__)

LOGLEVEL = logging.ERROR

class AmsetTest(unittest.TestCase):
    def setUp(self):
        os.makedirs(os.path.join(test_dir, 'temp_dir'), exist_ok=True)
        self.temp_dir = os.path.join(test_dir, 'temp_dir')
        self.model_params = {'bs_is_isotropic': True,
                             'elastic_scats': ['ACD', 'IMP', 'PIE'],
                             'inelastic_scats': ['POP']}
        self.performance_params = {'dE_min': 0.0001, 'nE_min': 5, "n_jobs": 1,
                'BTE_iters': 5,'nkdos':29, 'max_nbands': 1, 'max_nvalleys': 1,
                'max_normk': 2, 'Ecut': 0.4, 'fermi_kgrid_tp': 'coarse',
                                   "dos_kdensity": 200, "interpolation": "boltztrap1"
                                   }
        self.GaAs_params = {'epsilon_s': 12.9, 'epsilon_inf': 10.9,
                'W_POP': 8.73, 'C_el': 139.7, 'E_D': {'n': 8.6, 'p': 8.6},
                'P_PIE': 0.052, 'scissor': 0.5818}
        self.GaAs_path = os.path.join(test_dir, '..', 'test_files', 'GaAs_mp-2534')
        self.GaAs_vasprun = os.path.join(self.GaAs_path, 'vasprun.xml')
        self.GaAs_cube = os.path.join(self.GaAs_path, 'fort.123')
        self.InP_path = os.path.join(test_dir, '..', 'test_files', 'InP_mp-20351')
        self.InP_vasprun = os.path.join(self.InP_path, 'vasprun.xml')
        self.InP_params = {"epsilon_s": 14.7970, "epsilon_inf": 11.558,
                            "W_POP": 9.2651, "C_el": 119.18,
                            "E_D": {"n": 5.74, "p": 1.56},
                            "P_PIE": 0.052, "user_bandgap": 1.344
                            }

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


    def test_poly_bands(self):
        print('\ntesting test_poly_bands...')
        mass = 0.25
        c = -2e15
        temperatures = [300]
        model_params = deepcopy(self.model_params)
        model_params['poly_bands'] = [[[[0.0, 0.0, 0.0], [0.0, mass]]]]
        amset = Amset(calc_dir=self.temp_dir,
                      vasprun_file=self.GaAs_vasprun,
                      material_params=self.GaAs_params,
                      model_params=model_params,
                      performance_params=self.performance_params,
                      dopings=[c], temperatures=temperatures, integration='k',
                      loglevel=LOGLEVEL)
        amset.run(self.GaAs_cube, kgrid_tp='coarse')

        # check fermi level
        # density calculation source: http://hib.iiit-bh.ac.in/Hibiscus/docs/iiit/NBDB/FP008/597_Semiconductor%20in%20Equilibrium&pn%20junction1.pdf
        # density of states source: http://web.eecs.umich.edu/~fredty/public_html/EECS320_SP12/DOS_Derivation.pdf
        for T in temperatures:
            N_c = 2 * (2 * np.pi * mass * 9.11e-31 * 1.3806e-23 * T / ((6.626e-34)**2))**1.5
            expected_fermi_level = amset.cbm_vbm['n']["energy"] - (1.3806e-23 * T * np.log(N_c / (-c * 1e6)) * 6.242e18)

            diff = abs(amset.fermi_level[c][T] - expected_fermi_level)
            avg = (amset.fermi_level[c][T] + expected_fermi_level) / 2

            # setting kgrid_tp tp fine or very fine would drive the difference
            # closer to zero; however we set a loose 6.5% for quicker testing:
            self.assertTrue(diff / avg < 0.065)

            diff = abs(np.array(amset.mobility['n']['ACD'][c][T]) - \
                       np.array(amset.mobility['n']['SPB_ACD'][c][T]))
            avg = (amset.mobility['n']['ACD'][c][T] + \
                   amset.mobility['n']['SPB_ACD'][c][T]) / 2
            self.assertTrue((diff / avg <= 0.025).all())


    def test_GaAs_isotropic_E_plus_serialize(self):
        print('\ntesting test_GaAs_isotropic_E...')
        expected_mu = {'ACD': 459623.2946,
                       'IMP': 2932848.2466,
                       'PIE': 2142320.5897,
                       'POP': 16329.22998,
                       'average': 15570.668,
                       'overall': 15750.11293,
                       }
        expected_seebeck = -968.77840
        performance_params = deepcopy(self.performance_params)
        performance_params['max_nvalleys'] = 1
        amset = Amset(calc_dir=self.temp_dir,
                      vasprun_file=self.GaAs_vasprun,
                      material_params=self.GaAs_params,
                      model_params=self.model_params,
                      performance_params=performance_params,
                      dopings=[-2e15], temperatures=[300], integration='e',
                      loglevel=LOGLEVEL)
        amset.run(self.GaAs_cube, kgrid_tp='very coarse')
        kgrid = amset.kgrid

        # check general characteristics of the grid
        self.assertEqual(kgrid['n']['velocity'][0].shape[0], 78)
        mean_v = np.mean(kgrid['n']['velocity'][0], axis=0)
        self.assertLessEqual(np.std(mean_v), 50.00) # isotropic BS after removing points
        self.assertAlmostEqual(mean_v[0], 113757441.8667, places=1) # zeroth band

        # check mobility values
        for mu in expected_mu.keys():
            self.assertLessEqual(np.std( # test isotropic
                amset.mobility['n'][mu][-2e15][300])/np.mean(
                amset.mobility['n'][mu][-2e15][300]), 0.1)
            self.assertLessEqual(abs(amset.mobility['n'][mu][-2e15][300][0] / expected_mu[mu] - 1), 0.01)
        self.assertLess(abs(amset.seebeck['n'][-2e15][300][0]/expected_seebeck-1), 0.04)

        # just testing write to file methods:
        amset.as_dict()
        amset.to_file(dir_name='run_data')
        amset.to_csv()
        amset.to_json()

        # deserialization test:
        amset.from_file(path=os.path.join(amset.calc_dir, 'run_data'))


    def test_GaAs_anisotropic(self):
        print('\ntesting test_GaAs_anisotropic...')
        expected_mu = {'ACD': 378435.7259,
                       'IMP': 2525835.3985,
                       'PIE': 924827.386266,
                       'POP': 32060.566397,
                       'average': 28320.0974264,
                       'overall': 29528.9179,
                       }
        expected_seebeck = -809.72568
        amset = Amset(calc_dir=self.temp_dir,
                      vasprun_file=self.GaAs_vasprun,
                      material_params=self.GaAs_params,
                      model_params={'bs_is_isotropic': False,
                             'elastic_scats': ['ACD', 'IMP', 'PIE'],
                             'inelastic_scats': ['POP']},
                      performance_params=self.performance_params,
                      dopings=[-2e15], temperatures=[300], integration='e',
                      loglevel=LOGLEVEL)
        amset.run(self.GaAs_cube, kgrid_tp='coarse')

        # check mobility values
        for mu in expected_mu.keys():
            self.assertLessEqual(np.std(  # GaAs band structure is isotropic
                amset.mobility['n'][mu][-2e15][300]), 0.05*\
                np.mean(amset.mobility['n'][mu][-2e15][300]))
            self.assertLess(abs(amset.mobility['n'][mu][-2e15][300][0] - expected_mu[mu])/expected_mu[mu], 0.06)
        self.assertLess(abs(amset.seebeck['n'][-2e15][300][0]/expected_seebeck-1), 0.06)


    def test_GaAs_isotropic_k(self):
        print('\ntesting test_GaAs_isotropic_k...')
        expected_mu = {'ACD': 1303937.379,
                       'IMP': 2414.127,
                       'PIE': 2979626.74,
                       'POP': 18309.123,
                       'average': 2127.893,
                       'overall': 2120.471
                       }
        performance_params = dict(self.performance_params)
        performance_params['fermi_kgrid_tp'] = 'very coarse'
        amset = Amset(calc_dir=self.temp_dir,
                      vasprun_file=self.GaAs_vasprun,
                      material_params=self.GaAs_params,
                      model_params=self.model_params,
                      performance_params=performance_params,
                      dopings=[-3e13], temperatures=[300], integration='k',
                      loglevel=LOGLEVEL)
        amset.run(self.GaAs_cube, kgrid_tp='very coarse')
        mobility = amset.mobility
        self.assertAlmostEqual( # compare with normalized fermi w.r.t. the CBM
            amset.fermi_level[-3e13][300]-amset.cbm_vbm["n"]["energy"],
            -0.3429418, 3)

        # check mobility values
        for mu in expected_mu.keys():
            diff = np.std(mobility['n'][mu][-3e13][300])
            avg = np.mean(mobility['n'][mu][-3e13][300])
            self.assertLess(diff / avg, 0.005)
            self.assertLessEqual(abs(amset.mobility['n'][mu][-3e13][300][0] / expected_mu[mu] - 1), 0.01)




    def test_InP_isotropic_E(self):
        print('\ntesting test_InP_isotropic_E...')
        expected_mu = {'ACD': 495952.86374,
                       'IMP': 2066608.8522,
                       'PIE': 1344563.563591,
                       'POP': 34941.917715,
                       'average': 31384.4998,
                       'overall': 32195.345065
                       }

        amset = Amset(calc_dir=self.temp_dir,
                      vasprun_file=self.InP_vasprun,
                      material_params=self.InP_params,
                      model_params=self.model_params,
                      performance_params=self.performance_params,
                      dopings=[-2e15], temperatures=[300], integration='e',
                      loglevel=LOGLEVEL)
        amset.run(os.path.join(self.InP_path, 'fort.123'),
                  kgrid_tp='very coarse')

        # check isotropy of transport and mobility values
        for mu in expected_mu.keys():
            self.assertLessEqual(
                np.std(amset.mobility['n'][mu][-2e15][300]) / \
                np.mean(amset.mobility['n'][mu][-2e15][300]), 0.06
            )
            self.assertLessEqual(abs(amset.mobility['n'][mu][-2e15][300][0]/expected_mu[mu]-1),0.02)


    def test_defaults(self):
        print('\ntesting test_defaults...')
        cal_dir = self.temp_dir
        data_dir = os.path.join(cal_dir, "run_data")
        amset = Amset(cal_dir, material_params={'epsilon_s': 12.9})
        amset.write_input_files()
        with open(os.path.join(data_dir, "material_params.json"), "r") as fp:
            material_params = json.load(fp)
        with open(os.path.join(data_dir, "model_params.json"), "r") as fp:
            model_params = json.load(fp)
        with open(os.path.join(data_dir, "performance_params.json"), "r") as fp:
            performance_params = json.load(fp)

        self.assertEqual(material_params['epsilon_inf'], None)
        self.assertEqual(material_params['W_POP'], None)
        self.assertEqual(material_params['scissor'], 0.0)
        self.assertEqual(material_params['P_PIE'], 0.15)
        self.assertEqual(material_params['E_D'], None)
        self.assertEqual(material_params['N_dis'], 0.1)

        self.assertEqual(model_params['bs_is_isotropic'], True)
        self.assertEqual(model_params['elastic_scats'], ['IMP', 'PIE'])
        self.assertEqual(model_params['inelastic_scats'], [])

        self.assertEqual(performance_params['max_nbands'], None)
        self.assertEqual(performance_params['max_normk0'], None)
        self.assertEqual(performance_params['dE_min'], 0.0001)
        self.assertEqual(performance_params['dos_kdensity'], 5500)
        self.assertEqual(performance_params['dos_bwidth'], 0.075)


if __name__ == '__main__':
    unittest.main()
