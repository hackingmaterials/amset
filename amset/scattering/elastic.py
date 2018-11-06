"""
This module implements elastic scattering calculators.

TODO: Give a more detailed description of what beta is.
TODO: Provide required units in docstrings.
TODO: Remove magic numbers.
"""

import numpy as np
import logging

from amset.utils.constants import epsilon_0, e, hbar, pi, sq3, k_B, A_to_nm
from amset.utils.general import norm
from amset.scattering.base import AbstractScattering


class AbstractElasticScattering(AbstractScattering):
    """Abstract class for defining elastic charge-carrier scattering processes.

    The scattering rates should be obtained from the ``calculate_scattering``
    method.

    This class implements the ``calculate_anisotropic`` and ``integrand_angle``
    functions for the elastic case.

    Scattering classes that extend this class only need to implement the
    following functions:

    - ``calculate_isotropic``: Calculates the isotropic scattering rates for all
        k-points in the valley.
    - ``calculate_scattering_between_kpoints``: Calculates the scattering
        rate between a k-point and k-point^prime.

    See the ``AbstractScattering`` docstring for more details.
    """

    def calculate_scattering_between_kpoints(self, kpoint, kpoint_prime):
        """Calculates the scattering rate between kpoint and kpoint_prime.

        Args:
            kpoint (int): The index of the first k-point.
            kpoint_prime (int): The index of the second k-point.

        Returns:
            (float): The scattering rate between the two k-points in per
            second.
        """
        raise NotImplementedError

    def calculate_isotropic(self):
        """Calculates the isotropic scattering rate for all k-points.

        Returns:
            (numpy.ndarray): The scattering rates as an (Nx3) numpy array, where
            N is the number of k-points in the valley. The second axis of the
            array refers to the cartesian lattice direction, e.g. x, y, and z.
            For the isotropic case, all three directions will be the same value.
        """
        raise NotImplementedError

    def calculate_anisotropic(self):
        """Calculates the anistropic scattering rate for all k-points.

        Depending on the ``constrain_rates`` instance variables, the rate
        will be limited to physical values.

        Returns:
            (numpy.ndarray): The scattering rates as an (Nx3) numpy array, where
            N is the number of k-points in the valley. The second axis of the
            array refers to the cartesian lattice direction, e.g. x, y, and z.
        """
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
        """Returns the evaluated integrand for the elastic scattering equations.

        The results are integrated over angles in ``integrate_over_angle``.

        Args:
            kpoint (int): The index of the first k-point.
            kpoint_prime (int): The index of the second k-point.
            angle (float): The cosine of the angle between the two k-points.

        Returns:
            (numpy.ndarray): The evaluated integrand of the elastic scattering
            equations as a (1x3) numpy array.
        """
        kpoint_cart = self.valley.kpoints[kpoint]
        kpoint_prime_cart = self.valley.kpoints[kpoint_prime]
        integrand = np.array([0., 0., 0.])
        if np.array_equal(kpoint_cart, kpoint_prime_cart):
            # self-scattering is not defined but integrand must be a vector
            return integrand

        return ((1 - angle) * self.valley.kpoints_norm[kpoint_prime] ** 2 *
                self.calculate_scattering_between_kpoints(kpoint, kpoint_prime)
                * self.get_overlap_integral(kpoint, kpoint_prime, angle) /
                (self.valley.velocities_norm[kpoint_prime] / sq3)) + integrand


class IonizedImpurityScattering(AbstractElasticScattering):
    """Class to calculate ionized impurity scattering rates.

    The scattering rates are obtained using the ``calculate_scattering``
    function. The three letter abbreviation for this scattering process is
    "IMP".

    Implements a variation of Dingle's theory. See equation 90 in [R]_ for more
    details.

    Args:
        isotropic (bool): Whether the scattering should be calculated using the
            isotropic formalism.
        valley (Valley): A valley object.
        epsilon_s (float): The static dielectric constant.
        conc_imp (float): The concentration of the impurity.
        beta (float): The value of beta.
    """

    def __init__(self, isotropic, valley, epsilon_s,
                 conc_imp, beta):
        super(IonizedImpurityScattering, self).__init__(
            "IMP", isotropic, valley, constrain_rates=True)

        self.epsilon_s = epsilon_s
        self.conc_imp = conc_imp
        self.beta = beta

    def calculate_scattering_between_kpoints(self, kpoint, kpoint_prime):
        """Calculates the scattering rate between kpoint and kpoint_prime.

        Args:
            kpoint (int): The index of the first k-point.
            kpoint_prime (int): The index of the second k-point.

        Returns:
            (float): The scattering rate between the two k-points in per
            second.
        """
        norm_diff_k = norm(self.valley.kpoints[kpoint] -
                           self.valley.kpoints[kpoint_prime])

        unit_conversion = 0.001 / e ** 2
        return (unit_conversion * e ** 4 * self.conc_imp /
                (4.0 * pi ** 2 * self.epsilon_s ** 2 * epsilon_0 ** 2 * hbar) /
                ((norm_diff_k ** 2 + self.beta ** 2) ** 2))

    def calculate_isotropic(self):
        """Calculates the isotropic scattering rate for all k-points.

        Returns:
            (numpy.ndarray): The scattering rates as an (Nx3) numpy array, where
            N is the number of k-points in the valley. The second axis of the
            array refers to the cartesian lattice direction, e.g. x, y, and z.
            For the isotropic case, all three directions will be the same value.
        """
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
    """Class to calculate acoustic deformation potential scattering rates.

    The scattering rates are obtained using the ``calculate_scattering``
    function. The three letter abbreviation for this scattering process is
    "ACD".

    The implementation follows that of equation 112 (page 38) in [R]_.

    Args:
        isotropic (bool): Whether the scattering should be calculated using the
            isotropic formalism.
        valley (Valley): A valley object.
        elastic_constant (float): The elastic constant.
        deformation_potential (float): The deformation potential of the
            valence/conduction band for p- or n-type scattering, respectively.
        temperature (float): The temperature.
    """

    def __init__(self, isotropic, valley, elastic_constant,
                 deformation_potential, temperature):
        super(AcousticDeformationScattering, self).__init__(
            "ACD", isotropic, valley, constrain_rates=True)

        self.elastic_constant = elastic_constant
        self.deformation_potential = deformation_potential
        self.temperature = temperature

    def calculate_scattering_between_kpoints(self, kpoint, kpoint_prime):
        """Calculates the scattering rate between kpoint and kpoint_prime.

        Args:
            kpoint (int): The index of the first k-point.
            kpoint_prime (int): The index of the second k-point.

        Returns:
            (float): The scattering rate between the two k-points in per
            second.
        """
        unit_conversion = 1e18 * e
        return (unit_conversion * k_B * self.temperature *
                self.deformation_potential ** 2 /
                (4.0 * pi ** 2 * hbar * self.elastic_constant))

    def calculate_isotropic(self):
        """Calculates the isotropic scattering rate for all k-points.

        Returns:
            (numpy.ndarray): The scattering rates as an (Nx3) numpy array, where
            N is the number of k-points in the valley. The second axis of the
            array refers to the cartesian lattice direction, e.g. x, y, and z.
            For the isotropic case, all three directions will be the same value.
        """
        # because of isotropic assumption, we treat the BS as 1D
        velocities = self.valley.velocities_norm / sq3

        # The following two lines are from [R]: page 38, eq. (112)
        rate = ((k_B * self.temperature * self.deformation_potential ** 2 *
                self.valley.kpoints_norm ** 2) /
                (3 * pi * hbar ** 2 * self.elastic_constant * 1e9 * velocities)
                * (3 - 8 * self.valley.c_contrib ** 2 + 6 *
                   self.valley.c_contrib ** 4) * e * 1e20)
        return np.stack([rate, rate, rate], axis=1)


class PiezoelectricScattering(AbstractElasticScattering):
    """Class to calculate piezoelectric scattering rates.

    The scattering rates are obtained using the ``calculate_scattering``
    function. The three letter abbreviation for this scattering process is
    "PIE".

    The implementation follows that of equation 108 in [R]_.

    Args:
        isotropic (bool): Whether the scattering should be calculated using the
            isotropic formalism.
        valley (Valley): A valley object.
        epsilon_s (float): The static dielectric constant.
        piezoelectric_coeff (float): The piezoelectric coefficient.
        temperature (float): The temperature.
    """

    def __init__(self, isotropic, valley, epsilon_s,
                 piezoelectric_coeff, temperature):
        super(PiezoelectricScattering, self).__init__(
            "PIE", isotropic, valley, constrain_rates=True)

        self.epsilon_s = epsilon_s
        self.piezoelectric_coeff = piezoelectric_coeff
        self.temperature = temperature

    def calculate_scattering_between_kpoints(self, kpoint, kpoint_prime):
        """Calculates the scattering rate between kpoint and kpoint_prime.

        Args:
            kpoint (int): The index of the first k-point.
            kpoint_prime (int): The index of the second k-point.

        Returns:
            (float): The scattering rate between the two k-points in per
            second.
        """
        norm_diff_k = norm(self.valley.kpoints[kpoint] -
                           self.valley.kpoints[kpoint_prime])
        unit_conversion = 1e9 / e
        return (unit_conversion * e ** 2 * k_B * self.temperature *
                self.piezoelectric_coeff ** 2 /
                (norm_diff_k ** 2 * 4.0 * pi ** 2 * hbar * epsilon_0 *
                 self.epsilon_s))

    def calculate_isotropic(self):
        """Calculates the isotropic scattering rate for all k-points.

        Returns:
            (numpy.ndarray): The scattering rates as an (Nx3) numpy array, where
            N is the number of k-points in the valley. The second axis of the
            array refers to the cartesian lattice direction, e.g. x, y, and z.
            For the isotropic case, all three directions will be the same value.
        """
        # because of isotropic assumption, we treat the BS as 1D
        velocities = self.valley.velocities_norm / sq3

        # equation (108) of the reference [R]
        rate = ((e ** 2 * k_B * self.temperature *
                self.piezoelectric_coeff ** 2) / (
                6 * pi * hbar ** 2 * self.epsilon_s * epsilon_0 * velocities) *
                (3 - 6 * self.valley.c_contrib ** 2 + 4 *
                 self.valley.c_contrib ** 4) * 100 / e)
        return np.stack([rate, rate, rate], axis=1)


class DislocationScattering(AbstractElasticScattering):
    """Class to calculate dislocation scattering rates.

    The scattering rates are obtained using the ``calculate_scattering``
    function. The anisotropic formulation for this scattering type has not been
    implemented. The three letter abbreviation for this scattering process is
    "DIS".

    The implementation follows that of table 1 in reference [A]_.


    Args:
        isotropic (bool): Whether the scattering should be calculated using the
            isotropic formalism.
        valley (Valley): A valley object.
        epsilon_s (float): The static dielectric constant.
        beta (float): The value of beta.
        conc_dis (float): The concentration of the dislocation.
        lattice_length (float): The length of the lattice vector along which the
            dislocation occurs.
    """

    def __init__(self, isotropic, valley, epsilon_s,
                 beta, conc_dis, lattice_length):
        super(DislocationScattering, self).__init__(
            "DIS", isotropic, valley, constrain_rates=True)
        self.epsilon_s = epsilon_s
        self.beta = beta
        self.conc_dis = conc_dis
        self.lattice_length = lattice_length

    def calculate_scattering_between_kpoints(self, kpoint, kpoint_prime):
        """Calculates the scattering rate between kpoint and kpoint_prime.

        Args:
            kpoint (int): The index of the first k-point.
            kpoint_prime (int): The index of the second k-point.

        Returns:
            (float): The scattering rate between the two k-points in per
            second.
        """
        raise NotImplementedError(
            "Anisotropic dislocation scattering not implemented.")

    def calculate_isotropic(self):
        """Calculates the isotropic scattering rate for all k-points.

        Returns:
            (numpy.ndarray): The scattering rates as an (Nx3) numpy array, where
            N is the number of k-points in the valley. The second axis of the
            array refers to the cartesian lattice direction, e.g. x, y, and z.
            For the isotropic case, all three directions will be the same value.
        """
        # See table 1 of the reference [A]
        velocities = self.valley.velocities_norm / sq3
        rate = ((self.conc_dis * e ** 4 * self.valley.kpoints_norm) /
                (hbar ** 2 * epsilon_0 ** 2 * self.epsilon_s ** 2 *
                 (self.lattice_length * A_to_nm) ** 2 * velocities) /
                (self.beta ** 4 * (1 + (4 * self.valley.kpoints_norm ** 2) /
                                   (self.beta ** 2)) ** 1.5)
                * 2.43146974985767e42 * 1.60217657 / 1e8)
        return np.stack([rate, rate, rate], axis=1)
