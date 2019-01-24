"""
This module defines the base class for calculating scattering rates.

TODO: reformat angle_k_prime_mapping to remove band index
TODO: check final mobilities using both methods, against isotropic
      formalism to see which one is more accurate
"""
import copy
import numpy as np

from monty.json import MSONable
from scipy.integrate import trapz

from amset.utils.constants import pi


class AbstractScattering(MSONable):
    """Abstract class for defining charge-carrier scattering processes.

    This class provides helper functions for integrating scattering rates over
    360 degrees in reciprocal space, and for calculating the overlap integral
    between k-points.

    Scattering classes that extend this class must implement the following
    methods ``calculate_isotropic`` and ``calculate_isotropic`` for calculating
    the isotropic and anisotropic scattering rates for all k-points,
    respectively, and ``integrand_angle`` for calculating the scattering
    interaction between two k-points.

    Args:
        name (str): A three letter abbreviation for the scattering type.
        isotropic (bool): Whether the scattering should be calculated using the
            isotropic formalism.
        valley (Valley): A valley object.
        constrain_rates (bool, optional): Whether to constrain the scattering
            rates to physical values.
    """

    def __init__(self, name, isotropic, valley, constrain_rates=True):
        self.name = name
        self.isotropic = isotropic
        self.valley = valley
        self.constrain_rates = constrain_rates

        if not isotropic and not valley.angle_k_prime_mapping:
            raise ValueError("valley angle_k_prime_mapping required to "
                             "calculate anisotropic scattering")

    def integrand_angle(self, kpoint, kpoint_prime, angle, **kwargs):
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
        raise NotImplementedError

    def calculate_scattering(self):
        """Calculates carrier scattering rates for all k-points in the valley.

        The formalism used depends on the ``isotropic`` instance variable.

        Returns:
            (numpy.ndarray): The scattering rates as an (Nx3) numpy array, where
            N is the number of k-points in the valley. The second axis of the
            array refers to the cartesian lattice direction, e.g. x, y, and z.
        """
        if self.isotropic:
            return self.calculate_isotropic()
        else:
            return self.calculate_anisotropic()

    def get_overlap_integral(self, kpoint, kpoint_prime, angle):
        """Calculates the overlap integral between two k-points.

        Args:
            kpoint (int): The index of the first k-point.
            kpoint_prime (int): The index of the second k-point.
            angle (float): The cosine of the angle between the two k-points.

        Returns:
            (float): The overlap integral between the two k-points.
        """
        return ((self.valley.a_contrib[kpoint] *
                 self.valley.a_contrib[kpoint_prime]) +
                (angle * self.valley.c_contrib[kpoint] *
                 self.valley.c_contrib[kpoint_prime]))**2

    def integrate_over_angle(self, kpoint, scipy_int=False, **integrand_kwargs):
        """Integrate the scattering rate over 360 degrees in reciprocal space.

        The default algorithm is to use a simple trapezoidal method
        (implemented by Alireza Faghaninia). This method gives non-negligible
        differences to the trapezoidal algorithm in scipy. Both methods have
        been implemented but the function will default to the method written
        by Alireza until the discrepancies can be resolved.

        Args:
            kpoint (int): The index of the k-point.
            scipy_int (bool, optional): Whether to use scipy for interpolation
                rather than Alireza's procedure.
            **integrand_kwargs (kwargs): Keyword arguments to be passed to
                the ``integrand_angle`` function. See the implemention of this
                method in the subclasses for more details.

        Returns:
            (numpy.ndarray): The integrated scattering rate as a (1x3) numpy
            array.
        """
        summation = 0.0
        if len(self.valley.angle_k_prime_mapping[kpoint]) == 0:
            raise ValueError(
                "enforcing scattering points did NOT work, {} at kpoint: {} "
                "is empty".format(self.valley.angle_k_prime_mapping,  kpoint))

        # relies on the fact that the mapping is already sorted by angle
        mapping = copy.deepcopy(self.valley.angle_k_prime_mapping[kpoint])

        if scipy_int:
            mapping.insert(0, [-1, 0, kpoint])
            integrands = [self.integrand_angle(kpoint, kpoint_prime, angle)
                          for angle, _, kpoint_prime in mapping]

            angles = [x[0] for x in mapping]
            summation = trapz(integrands, x=angles, axis=0)

        else:
            angle, _, kpoint_prime = mapping[0]

            current_integrand = self.integrand_angle(kpoint, kpoint_prime,
                                                     angle, **integrand_kwargs)
            ikp = 0
            dum = 0
            delta_angles = []
            dums = []
            cum_sum = []
            while ikp < len(mapping) - 1:
                delta_angle = (mapping[ikp + 1][0] -
                               mapping[ikp][0])
                delta_angles.append(delta_angle)

                loop_found = False

                if not loop_found:
                    dum = current_integrand / 2.0
                    ikp += 1

                angle, _, kpoint_prime = mapping[ikp]
                current_integrand = self.integrand_angle(
                    kpoint, kpoint_prime, angle, **integrand_kwargs)

                # alternate between left and right integration to simulate
                # integrating over the full range of ik
                if np.sum(current_integrand) == 0.0:
                    dum *= 2
                elif np.sum(dum) == 0.0:
                    dum = current_integrand
                else:
                    dum += current_integrand / 2.0

                dums.append(dum)
                cum_sum.append(dum * delta_angle)

                # if two points have sample angle,
                # delta_angle == 0 so no duplicates
                summation += dum * delta_angle

        return summation
