from scipy.integrate import trapz

import copy

import numpy as np

from multiprocessing import Pool, cpu_count


class AbstractScattering(object):
    """Abstract class defining common scattering calculation functions.

    """

    def __init__(self, name, isotropic, valley, constrain_rates=True):
        self.name = name
        self.isotropic = isotropic
        self.valley = valley
        self.constrain_rates = constrain_rates

        if not isotropic and not valley.angle_k_prime_mapping:
            raise ValueError("valley angle_k_prime_mapping required to "
                             "calculate anisotropic scattering")

    def calculate_isotropic(self):
        raise NotImplementedError

    def calculate_anisotropic(self):
        raise NotImplementedError

    def integrand_angle(self, kpoint, kpoint_prime, angle, **kwargs):
        raise NotImplementedError

    def calculate_scattering(self):
        if self.isotropic:
            return self.calculate_isotropic()
        else:
            return self.calculate_anisotropic()

    def get_overlap_integral(self, kpoint, kpoint_prime, angle):
        return ((self.valley.a_contrib[kpoint] *
                 self.valley.a_contrib[kpoint_prime]) +
                (angle * self.valley.c_contrib[kpoint] *
                 self.valley.c_contrib[kpoint_prime]))**2

    def integrate_over_angle(self, kpoint, scipy_int=False, **integrand_kwargs):
        """
        integrate numerically with a simple trapezoidal algorithm.

        Args:
            kpoint (int): the k-point index
            scipy_int (bool): Whether to use my scipy interpolation rather than
                Alireza's

        Returns (float or numpy.array): the integrated value/vector
        """
        summation = 0.0
        if len(self.valley.angle_k_prime_mapping[kpoint]) == 0:
            raise ValueError(
                "enforcing scattering points did NOT work, {} at kpoint: {} "
                "is empty".format(self.valley.angle_k_prime_mapping,  kpoint))

        # relies on the fact that the mapping is already sorted by angle
        mapping = copy.deepcopy(self.valley.angle_k_prime_mapping[kpoint])

        # todo: reformat angle_k_prime_mapping to remove band index

        # todo: check final mobilities using both methods, against isotropic
        # formalism to see which one is more accurate

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


class AbstractParallelizableScattering(object):
    """Abstract class defining common scattering calculation functions.

    """

    def __init__(self, n_jobs=-1):
        self._n_jobs = n_jobs

    def calculate(self, kpoint_index, kpoints, energies, velocities):
        raise NotImplementedError

    def calculate_scattering(self, kpoints, energies, velocities):

        def calculate_wrapper(kpoint_index):
            return self.calculate(kpoint_index, kpoints, energies, velocities)

        kpoint_indexes = range(len(kpoints))

        if self.n_jobs == 1:
            return [self.calculate(i, kpoints, energies, velocities)
                    for i in kpoint_indexes]
        else:
            with Pool(self.n_jobs) as p:
                return p.map(calculate_wrapper, kpoint_indexes)

    def set_n_jobs(self, n_jobs):
        """Set the number of processes to spawn."""
        self._n_jobs = n_jobs

    @property
    def n_jobs(self):
        return self._n_jobs if hasattr(self, '_n_jobs') else cpu_count()
