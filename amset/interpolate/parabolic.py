"""
Class to interpolate a parabolic band structure.
"""

import numpy as np

from typing import Optional, Union, List, Tuple

from amset.utils.band_structure import remove_duplicate_kpoints
from amset.utils.constants import hbar, m_e, e
from amset.utils.general import norm
from pymatgen import Spin
from amset.interpolate.base import AbstractInterpolater
from pymatgen.electronic_structure.bandstructure import BandStructure

__author__ = "Alex Ganose and Alireza Faghaninia"
__copyright__ = "Copyright 2019, HackingMaterials"
__maintainer__ = "Alex Ganose"


class ParabolicInterpolater(AbstractInterpolater):
    """Class to interpolate a parabolic band structure.

    Args:
        band_structure: A pymatgen band structure object.
        num_electrons: The number of electrons in the system.
        band_parameters: The parabolic bands definition. This should
            be formatted as a list containing the parameters for each band.
            Each band can contain multiple band extrema. The general format for
            each extrema is::

                [k-point, offset, effective_mass]

            Where k-point is the fractional coordinate of the extrema,
            offset is the offset from the reference energy (zero for a valence
            band and the band gap energy for a conduction band), and effective
            mass is the valley effective mass. For example, the following
            definition::

                [[[[0.5, 0.5, 0.5], 0, 0.1]], [[0, 0, 0], 0.5, 0.2]]]]

            Indicates a single band system with two extrema. The first
            extrema is centered at [0.5 0.5 0.5], has an offset of
            0 (e.g. it is the CBM and VBM), and has an effective mass of 0.1.
            The second extrema is centered at Gamma and is 0.5 eV higher in
            energy, with an effective mass of 0.2.
    """

    def __init__(self, band_structure: BandStructure, num_electrons: int,
                 band_parameters: List[List], **kwargs):
        super(ParabolicInterpolater, self).__init__(
            band_structure, num_electrons, **kwargs)
        self._parameters = np.array(band_parameters)
        self._nbands = len(self._parameters) * 2

        # modify the band_parameters to include all symmetrically
        # equivalent k-points (k_i) these points will be used later to
        # generate energy based on the minimum norm(k-k_i)
        for ib in range(len(band_parameters)):
            for valley in range(len(band_parameters[ib])):
                cart_kpoint = np.dot(band_structure.structure.lattice.
                                     reciprocal_lattice.matrix,
                                     band_parameters[ib][valley][0]),
                equivalent_points = self._band_structure.get_sym_eq_kpoints(
                            cart_kpoint, cartesian=True)[0]
                self._parameters[ib][valley][0] = remove_duplicate_kpoints(
                    equivalent_points)

        self._vbm = max(self._band_structure.get_vbm()['band_index'][Spin.up])

        # mapping of the band index to parabolic band coefficient index.
        # e.g. if the vbm is band 30 and there are 2 parabolic bands, the
        # mapping will be: {29: 1, 30: 0, 31: 0, 32: 1}
        # i.e. the VB and CB use the first coefficient and VB-1 and CB+1 use the
        # second coefficient.
        self._allowed_bands = np.arange(self._vbm + 1 - self._nbands / 2,
                                        self._vbm + 1 + self._nbands / 2,
                                        dtype=int)
        self._band_mapping = dict(zip(self._allowed_bands,
                                      list(range(len(band_parameters)))[::-1] +
                                      list(range(len(band_parameters)))))

    def get_energies(self, kpoints: np.ndarray,
                     iband: Optional[Union[int, List[int]]] = None,
                     scissor: float = 0.0,
                     return_velocity: bool = False,
                     return_effective_mass: bool = False
                     ) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """Gets the interpolated energies for multiple k-points in a band.

        Args:
            kpoints: The k-points in fractional coordinates.
            iband: A band index or list of band indicies for which to get the
                energies. If ``None``, the energies for all available bands will
                be returned.
            scissor: The amount by which the band gap is scissored.
            return_velocity: Whether to return the band velocities.
            return_effective_mass: Whether to return the band effective masses.

        Returns:
            The band energies as a numpy array. If iband is an integer
            (only 1 band requested), the energies will be returned as a
            np.ndarray array with shape (num_kpoints). If multiple bands are
            requested, the energies will be returned as a np.ndarray
            with shape (num_bands, num_kpoints). If ``return_velocity`` or
            ``return_effective_mass`` are ``True`` a tuple is returned,
            formatted as::

                (energies, Optional[velocities], Optional[effective_masses])

            The velocities and effective masses are given as the 1x3 trace and
            full 3x3 tensor, respectively (along cartesian directions).
        """
        # TODO: Make compatible with spin polarization
        bandgap = self._band_structure.get_band_gap()['energy'] + scissor
        e_vbm = self._band_structure.get_vbm()['energy'] - scissor / 2

        if isinstance(iband, int):
            iband = [iband]
        elif not iband:
            iband = self._allowed_bands

        if any(i not in self._allowed_bands for i in iband):
            self.log_raise(ValueError,
                           "Parabolic bands should fall within: {}–{}".format(
                               self._allowed_bands[0], self._allowed_bands[-1]))

        def get_energy(kpoint, band_index):
            kpoint = np.dot(self._band_structure.structure.lattice.
                            reciprocal_lattice.matrix, np.array(kpoint))

            min_kdist = float('inf')
            parabolic_index = self._band_mapping[band_index]

            # get the effective mass of the band closest to the k-point
            # TODO: This should really use the lower of the two energies
            #  of the extrema. I.e., ony extrema could be closer but higher in
            #  energy.
            closest_offset = None
            closest_eff_mass = None
            for ks, offset, effective_mass in self._parameters[parabolic_index]:
                # ks is k-points symmetrically equivalent to extrema
                for k in ks:
                    distance = norm(kpoint - k)
                    if distance < min_kdist:
                        min_kdist = distance
                        closest_eff_mass = effective_mass
                        closest_offset = offset

            sgn = -1 if band_index <= self._vbm else + 1
            energy = e_vbm if sgn < 0 else e_vbm + bandgap
            energy += sgn * (closest_offset + hbar ** 2 * min_kdist ** 2 /
                             (2 * m_e * closest_eff_mass) * e * 1e18)

            velocity = [hbar * min_kdist /
                        (m_e * closest_eff_mass) * 1e11 * e] * 3
            effective_mass = [[closest_eff_mass, 0.0, 0.0],
                              [0.0, closest_eff_mass, 0.0],
                              [0.0, 0.0, closest_eff_mass]]

            return energy, np.array(velocity), sgn * np.array(effective_mass)

        all_data = [[get_energy(kpoint, band_index) for kpoint in kpoints]
                    for band_index in iband]

        energies = np.array([[x[0] for x in band_data]
                             for band_data in all_data])

        velocities = np.array([[x[1] for x in band_data]
                               for band_data in all_data])
        effective_masses = np.array([[x[2] for x in band_data]
                                     for band_data in all_data])

        shape = (len(iband), len(kpoints)) if len(iband) > 1 else (
            len(kpoints),)
        to_return = [energies.reshape(shape)]

        if return_velocity:
            to_return.append(velocities.reshape(shape + velocities.shape[2:]))

        if return_effective_mass:
            to_return.append(effective_masses.reshape(
                shape + effective_masses.shape[2:]))

        return self._simplify_return_data(to_return)
