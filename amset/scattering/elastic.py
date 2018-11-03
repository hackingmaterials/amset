import numpy as np
import logging

from amset.utils.constants import epsilon_0, e, hbar, pi, sq3, k_B, A_to_nm
from amset.utils.general import norm
from amset.scattering.base import AbstractScattering

from monty.serialization import dumpfn


class AbstractElasticScattering(AbstractScattering):

    def calculate_isotropic(self):
        raise NotImplementedError

    def calculate_between_kpoints(self, kpoint, kpoint_prime):
        raise NotImplementedError

    def calculate_anisotropic(self):
        rates = []
        for kpoint in range(len(self.valley.kpoints)):
            # the slope for PIE and IMP don't match with bs_is_isotropic
            summation = self.integrate_over_angle(kpoint)
            rate = np.abs(summation) * 2e-7 * pi / hbar

            rate_norm = np.linalg.norm(rate)

            if rate_norm < 100 and self.constrain_rates:
                logging.warning("Here {} rate < 1.\nrate:\n{}".format(
                    self.name, rate))

                rate = [1e10, 1e10, 1e10]

            elif rate_norm > 1e20:
                logging.warning('too large rate for {} at k={}, v={}:'.format(
                    self.name, rate, self.valley.velocities[kpoint]))

            rates.append(rate)
        return np.array(rates)

    def integrand_angle(self, kpoint, kpoint_prime, angle, **kwargs):
        """

        Args:
            kpoint (int): kpoint index
            kpoint_prime: k' index

            angle: Cosine of angle between k and k'.
        Returns:

        """
        kpoint_cart = self.valley.kpoints[kpoint]
        kpoint_prime_cart = self.valley.kpoints[kpoint_prime]
        integrand = np.array([0., 0., 0.])
        if np.array_equal(kpoint_cart, kpoint_prime_cart):
            # self-scattering is not defined but integrand must be a vector
            return integrand

        return ((1 - angle) * self.valley.kpoints_norm[kpoint_prime] ** 2 *
                self.calculate_between_kpoints(kpoint, kpoint_prime) *
                self.get_overlap_integral(kpoint, kpoint_prime, angle) /
                (self.valley.velocities_norm[kpoint_prime] / sq3)) + integrand


class IonizedImpurityScattering(AbstractElasticScattering):

    def __init__(self, isotropic, valley, epsilon_s,
                 conc_imp, beta):
        super(IonizedImpurityScattering, self).__init__(
            "IMP", isotropic, valley, constrain_rates=True)

        self.epsilon_s = epsilon_s
        self.conc_imp = conc_imp
        self.beta = beta

        # if True:
        #     data = vars(self)
        #     dumpfn(data, 'imp.json')

    def calculate_between_kpoints(self, kpoint, kpoint_prime):
        norm_diff_k = norm(self.valley.kpoints[kpoint] -
                           self.valley.kpoints[kpoint_prime])

        unit_conversion = 0.001 / e ** 2
        return (unit_conversion * e ** 4 * self.conc_imp /
                (4.0 * pi ** 2 * self.epsilon_s ** 2 * epsilon_0 ** 2 * hbar) /
                ((norm_diff_k ** 2 + self.beta ** 2) ** 2))

    def calculate_isotropic(self):
        # because of isotropic assumption, we treat the BS as 1D
        velocities = self.valley.velocities_norm / sq3

        # Note c_contrib is par_c; conc_imp is N_II
        # The following is a variation of Dingle's theory; see eq. (90) in [R]
        b_ii = ((4 * self.valley.kpoints_norm ** 2 / self.beta ** 2) /
                (1 + 4 * self.valley.kpoints_norm ** 2 / self.beta ** 2) +
                8 * (self.beta ** 2 + 2 * self.valley.kpoints_norm ** 2) /
                (self.beta ** 2 + 4 * self.valley.kpoints_norm ** 2) *
                self.valley.c_contrib ** 2 +
                (3 * self.beta ** 4 + 6 * self.beta ** 2 *
                 self.valley.kpoints_norm ** 2 - 8 *
                 self.valley.kpoints_norm ** 4) /
                ((self.beta ** 2 + 4 * self.valley.kpoints_norm ** 2) *
                 self.valley.kpoints_norm ** 2) * self.valley.c_contrib ** 4)
        d_ii = (1 + 2 * self.beta ** 2 * self.valley.c_contrib ** 2 /
                self.valley.kpoints_norm ** 2 + 3 * self.beta ** 4 *
                self.valley.c_contrib ** 4 /
                (4 * self.valley.kpoints_norm ** 4))

        # todo: what is magic number?
        rate = np.abs((e ** 4 * np.abs(self.conc_imp)) /
                      (8 * pi * velocities * self.epsilon_s ** 2 *
                       epsilon_0 ** 2 * hbar ** 2 *
                       self.valley.kpoints_norm ** 2) *
                      (d_ii * np.log(1 + 4 * self.valley.kpoints_norm ** 2 /
                                     self.beta ** 2)
                       - b_ii) * 3.89564386e27)
        return np.stack([rate, rate, rate], axis=1)


class AcousticDeformationScattering(AbstractElasticScattering):

    def __init__(self, isotropic, valley, elastic_constant,
                 deformation_potential, temperature):
        super(AcousticDeformationScattering, self).__init__(
            "ACD", isotropic, valley, constrain_rates=True)

        self.elastic_constant = elastic_constant
        self.deformation_potential = deformation_potential
        self.temperature = temperature

    def calculate_between_kpoints(self, kpoint, kpoint_prime):
        unit_conversion = 1e18 * e
        return (unit_conversion * k_B * self.temperature *
                self.deformation_potential ** 2 /
                (4.0 * pi ** 2 * hbar * self.elastic_constant))

    def calculate_isotropic(self):
        # The following two lines are from [R]: page 38, eq. (112)
        rate = ((k_B * self.temperature * self.deformation_potential ** 2 *
                self.valley.kpoints_norm ** 2) /
                (3 * pi * hbar ** 2 * self.elastic_constant * 1e9 *
                 self.valley.velocities_norm) *
                (3 - 8 * self.valley.c_contrib ** 2 + 6 *
                 self.valley.c_contrib ** 4) * e * 1e20)
        return np.stack([rate, rate, rate], axis=1)


class PiezoelectricScattering(AbstractElasticScattering):

    def __init__(self, isotropic, valley, epsilon_s,
                 piezeoelectric_coeff, temperatue):
        super(PiezoelectricScattering, self).__init__(
            "PIE", isotropic, valley, constrain_rates=True)

        self.epsilon_s = epsilon_s
        self.piezoelectric_coeff = piezeoelectric_coeff
        self.temperature = temperatue

    def calculate_between_kpoints(self, kpoint, kpoint_prime):
        norm_diff_k = norm(self.valley.kpoints[kpoint] -
                           self.valley.kpoints[kpoint_prime])
        unit_conversion = 1e9 / e
        return (unit_conversion * e ** 2 * k_B * self.temperature *
                self.piezoelectric_coeff ** 2 /
                (norm_diff_k ** 2 * 4.0 * pi ** 2 * hbar * epsilon_0 *
                 self.epsilon_s))

    def calculate_isotropic(self):
        # equation (108) of the reference [R]
        rate = ((e ** 2 * k_B * self.temperature *
                self.piezoelectric_coeff ** 2) / (
                6 * pi * hbar ** 2 * self.epsilon_s * epsilon_0 *
                self.valley.velocities_norm) *
                (3 - 6 * self.valley.c_contrib ** 2 + 4 *
                 self.valley.c_contrib ** 4) * 100 / e)
        return np.stack([rate, rate, rate], axis=1)


class DislocationScattering(AbstractElasticScattering):

    def __init__(self, isotropic, valley, epsilon_s,
                 beta, conc_dis, latt_c):
        super(DislocationScattering, self).__init__(
            "DIS", isotropic, valley, constrain_rates=True)
        # todo: option to choose lattice lattice dislocation direction
        self.epsilon_s = epsilon_s
        self.beta = beta
        self.conc_dis = conc_dis
        self.latt_c = latt_c

    def calculate_between_kpoints(self, kpoint, kpoint_prime):
        raise NotImplementedError(
            "Anisotropic dislocation scattering not implemented.")

    def calculate_isotropic(self):
        # See table 1 of the reference [A]
        rate = ((self.conc_dis * e ** 4 * self.valley.kpoints_norm) /
                (hbar ** 2 * epsilon_0 ** 2 * self.epsilon_s ** 2 *
                 (self.latt_c * A_to_nm) ** 2 * self.valley.velocities_norm) /
                (self.beta ** 4 * (1 + (4 * self.valley.kpoints_norm ** 2) /
                                   (self.beta ** 2)) ** 1.5)
                * 2.43146974985767e42 * 1.60217657 / 1e8)
        return np.stack([rate, rate, rate], axis=1)
