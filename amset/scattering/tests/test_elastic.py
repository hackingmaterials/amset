# coding: utf-8

import os
import numpy as np

import unittest

from amset.scattering.elastic import IonizedImpurityScattering

from monty.serialization import loadfn


test_dir = os.path.dirname(__file__)


class IonizedImpurityTest(unittest.TestCase):

    def setUp(self):
        data = loadfn(os.path.join(test_dir, 'imp_very_fine.json.gz'))
        self.isotropic = data['isotropic']
        self.kpoints = data['kpoints']
        self.kpoints_norm = data['kpoints_norm']
        self.velocities = data['velocities']
        self.velocities_norm = data['velocities_norm']
        self.a_contrib = data['a_contrib']
        self.c_contrib = data['c_contrib']
        self.epsilon_s = data['epsilon_s']
        self.conc_imp = data['conc_imp']
        self.beta = data['beta']
        self.angle_k_prime_mapping = data['angle_k_prime_mapping']

    def test_initialisation(self):
        # test that initialisation works with no errors
        imp = IonizedImpurityScattering(
            self.isotropic, self.kpoints, self.kpoints_norm, self.velocities,
            self.velocities_norm, self.a_contrib, self.c_contrib,
            self.epsilon_s, self.conc_imp, self.beta, self.angle_k_prime_mapping
        )

    def test_isotropic_scattering(self):
        # test isotropic scattering returns the correct rates
        imp = IonizedImpurityScattering(
            True, self.kpoints, self.kpoints_norm, self.velocities,
            self.velocities_norm, self.a_contrib, self.c_contrib,
            self.epsilon_s, self.conc_imp, self.beta, self.angle_k_prime_mapping
        )

        #rates = imp.calculate_scattering()

    def test_integration_over_angle(self):
        # test isotropic scattering returns the correct rates
        imp = IonizedImpurityScattering(
            True, self.kpoints, self.kpoints_norm, self.velocities,
            self.velocities_norm, self.a_contrib, self.c_contrib,
            self.epsilon_s, self.conc_imp, self.beta, self.angle_k_prime_mapping
        )

        print(imp.integrate_over_angle(0, scipy_int=True))
        print(imp.integrate_over_angle(0, scipy_int=False))


        # rates_iso = imp.calculate_scattering()
        #
        # imp = IonizedImpurityScattering(
        #     False, self.kpoints, self.kpoints_norm, self.velocities,
        #     self.velocities_norm, self.a_contrib, self.c_contrib,
        #     self.epsilon_s, self.conc_imp, self.beta, self.angle_k_prime_mapping
        # )
        #
        # rates_aniso = imp.calculate_scattering()
        # print(rates_iso[:, 0])
        # print(rates_aniso[:, 0])
        # rate_ratio = rates_iso[:, 0] / rates_aniso[:, 0]
        # print("average: {:.4f}".format(np.mean(rate_ratio)))
        # print("std dev: {:.4f}".format(np.std(rate_ratio)))
#        print("{:.2e}".format(np.mean(np.abs())))
