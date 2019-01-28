# coding: utf-8
import time
import warnings
from functools import partial
from itertools import starmap
from multiprocessing import cpu_count, Pool

import numpy as np

from amset.utils.constants import Ry_to_eV, hbar, A_to_m, m_to_cm, e, m_e
from amset.utils.general import outer
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

__author__ = "Francesco Ricci and Alireza Faghaninia"
__copyright__ = "Copyright 2017, HackingMaterials"
__maintainer__ = "Francesco Ricci"

"""
The fitting algorythm is the Shankland-Koelling-Wood Fourier interpolation scheme,
implemented (for example) in the BolzTraP software package (boltztrap1).

Details of the interpolation method are available in:
(1) R.N. Euwema, D.J. Stukel, T.C. Collins, J.S. DeWitt, D.G. Shankland,
    Phys. Rev. 178 (1969)  1419–1423.
(2) D.D. Koelling, J.H. Wood, J. Comput. Phys. 67 (1986) 253–262.
(3) Madsen, G. K. & Singh, D. J. Computer Physics Communications 175, 67–71 (2006).


The coefficient for fitting are indeed calculated in BoltzTraP, not in this code.
Here, we just build the star functions using those coefficients.
Then, we also calculate the energy bands for each k-point in input.
"""


def get_energy(xkpt, engre, nwave, nsym, nstv, vec, vec2=None, out_vec2=None,
               br_dir=None, cbm=True, return_dde=True):
    """
    Compute energy for a k-point from star functions
    Args:
        xkpt: k-point coordinates as array
        engre: matrix containing the coefficients of fitted band from get_interpolation_coefficients() function
        nwave: number of G vectors
        nsym: number of symmetries
        nstv: number of vectors in a star function
        vec: the star vectors for each G vector and symmetry
        vec2: dot product of star vectors with cell matrix for each G vector
            and symmetry. Needed only to compute the derivatives of energy
        br_dir: cell matrix. Needed only to compute the derivatives of energy
        cbm: True if the considered band is a conduction band. False if it is a valence band
        out_vec2: outer product of vec2 with itself. It is calculated outside to improve performances
        return_dde (bool): if true, it also returns the second derivative of
            energy used to calculate the effective mass for example.

    Returns:
        ene: the electronic energy at the k-point in input
        dene: 1st derivative of the electronic energy at the k-point in input
        ddene: 2nd derivative of the electronic energy at the k-point in input
    """
    sign = -1 if cbm == False else 1
    arg = 2 * np.pi * vec.dot(xkpt)
    tempc = np.cos(arg)
    spwre = (np.sum(tempc, axis=1) - (nsym - nstv)) / nstv

    if br_dir is not None:
        temps = np.sin(arg)
        # np.newaxis adds a new dimensions so that the shape of temperatures (nwave,2)
        # converts to (nwave,2,1) so it can be projected to vec2 (nwave, 2, 3)
        dspwre = np.sum(vec2 * temps[:, :, np.newaxis], axis=1)
        dspwre /= nstv[:, np.newaxis]
        if return_dde:
            out_tempc = out_vec2 * (-tempc[:, :, np.newaxis, np.newaxis])
            ddspwre = np.sum(out_tempc, axis=1) / nstv[:, np.newaxis,
                                                  np.newaxis]

    ene = spwre.dot(engre)
    if br_dir is not None:
        dene = np.dot(dspwre.T, engre)
        if return_dde:
            ddene = np.dot(ddspwre.T, engre)
            return sign * ene, dene, ddene
        else:
            return sign * ene, dene
    else:
        return sign * ene


class AnalyticalBands(object):
    """This class is meant to read the BoltzTraP fitted band structure
    coefficients and facilitate calculation of
    energy and its first and second derivatives w.r.t wave vector, k.

            n_jobs (int): number of processes used for interpolation.
            latt_points: all the G vectors
            nwave: number of G vectors
            nsym: number of symmetries
            symop: matrixes of the symmetry operations
            br_dir: cell matrix. Needed only to compute the derivatives of energy
            bmin: the index of the first band contained in the file
    """

    def __init__(self, coeff_file, n_jobs=-1):
        self._n_jobs = cpu_count() if n_jobs == -1 else n_jobs

        with open(coeff_file) as f:
            _, num_g_vectors, num_symmetries, _ = f.readline().split()
            self._num_g_vectors = int(num_g_vectors)
            self._num_symmetries = int(num_symmetries)

            self._cell_matrix = np.fromstring(f.readline(), sep=' ', count=3 * 3
                                              ).reshape((3, 3)).astype('float')
            self._sym_ops = np.transpose(
                np.fromstring(f.readline(), sep=' ', count=3 * 3 * 192
                              ).reshape((192, 3, 3)).astype('int'),
                axes=(0, 2, 1))

            g_vectors = np.zeros((self._num_g_vectors, 3))

            coefficients = []
            iband = []
            min_band = 0
            max_band = 0
            for i, l in enumerate(f):
                if i < self._num_g_vectors:
                    g_vectors[i] = np.fromstring(l, sep=' ')
                elif i == self._num_g_vectors:
                    min_band, max_band = np.fromstring(l, sep=' ', dtype=int)
                    iband = [self._num_g_vectors + (b - min_band + 1)
                             for b in range(min_band, max_band + 1)]
                elif i in iband:
                    coefficients.append(np.fromstring(l, sep=' '))
                    iband.pop(iband.index(i))
                    if len(iband) == 0:
                        break

        self._coefficients = np.array(coefficients)
        self._g_vectors = g_vectors
        self._min_band = min_band
        self._max_band = max_band
        self._allowed_ibands = set(range(self._min_band, self._max_band + 1))

        num_star_vectors, star_vectors, star_vector_products = \
            AnalyticalBands.get_star_functions(
                self._g_vectors, self._num_symmetries, self._sym_ops,
                self._num_g_vectors, cell_matrix=self._cell_matrix)
        self._num_star_vectors = num_star_vectors
        self._star_vectors = star_vectors
        self._star_vector_products = star_vector_products
        self._star_vector_products_sq = np.zeros((self._num_g_vectors,
                                                  max(num_star_vectors), 3, 3),
                                                 order='F')

        for nw in range(self._num_g_vectors):
            for i in range(self._num_star_vectors[nw]):
                self._star_vector_products_sq[nw, i] = outer(
                    star_vector_products[nw, i], star_vector_products[nw, i])

    @staticmethod
    def calculate_star_function(g_vector, num_symmetries, sym_ops):
        """
        Compute star function for a specific g vector
        Args:
            g_vector: G vector in real space
            num_symmetries: number of symmetries
            sym_ops: matrixes of the symmetry operations

        Returns:
            nst: number of vectors in the star function calculated for the G
            vector stg: star vectors
        """

        trial = sym_ops[:num_symmetries].dot(g_vector)
        stg = np.unique(trial.view(
            np.dtype((np.void, trial.dtype.itemsize * trial.shape[1])))).view(
            trial.dtype).reshape(-1, trial.shape[1])
        nst = len(stg)
        stg = np.concatenate((stg, np.zeros((num_symmetries - nst, 3))))
        return nst, stg

    @staticmethod
    def get_star_functions(g_vectors, num_symmetries, sym_ops, num_g_vectors,
                           cell_matrix):
        """
        Compute star functions for all G vectors and symmetries.
        Args:
            g_vectors: all the G vectors
            num_symmetries: number of symmetries
            sym_ops: matrixes of the symmetry operations
            num_g_vectors: number of G vectors
            cell_matrix: Cell matrix

        Returns:
            num_star_vectors: number of vectors in a star function for each G vector
            star_vectors: the star funcions for each G vector and symmetry
            star_vector_products: dot product of star vectors with cell matrix for each G vector
                    and symmetry. Needed only to compute the derivatives of energy
        """

        num_star_vectors = np.zeros(num_g_vectors, dtype='int')
        star_vectors = np.zeros((num_g_vectors, num_symmetries, 3), order='F')
        star_vector_products = np.zeros((num_g_vectors, num_symmetries, 3), order='F')

        for nw in range(num_g_vectors):
            num_star_vectors[nw], star_vectors[
                nw] = AnalyticalBands.calculate_star_function(
                g_vectors[nw], num_symmetries, sym_ops)
            star_vector_products[nw] = star_vectors[nw].dot(cell_matrix)

        return num_star_vectors, star_vectors, star_vector_products

    def get_interpolation_coefficients(self, iband):
        """
        Get coefficients of interpolation from a custom output file from BoltzTraP.
        Some other info are also read and provided as output.

        Args:
            iband: list of indexes of the bands to fit (starting from 1). If "A"
                is given, all the coefficients available are extracted.

        Returns:
            engre: matrix containing the coefficients of all the bands to fit
        """
        if isinstance(iband, int):
            iband = [iband]

        if iband == 'A':
            iband = sorted(self._allowed_ibands)

        if not set(iband).issubset(self._allowed_ibands):
            raise ValueError("At least one band is not in range : {}-{}. "
                             "Try increasing max_Ecut to include more "
                             "bands.".format(self._min_band, self._max_band))

        # normalise the bands to minimum band
        iband = [b - self._min_band for b in iband]

        if len(iband) == 1:
            return self._coefficients[iband][0]
        else:
            return self._coefficients[iband]

    def get_dos_from_scratch(self, structure, mesh, e_min, e_max, e_points,
                             width=0.2,
                             scissor=0.0):
        """
        Args:
        structure:       pmg object of crystal structure to calculate symmetries
        mesh:     list of integers defining the k-mesh on which the dos is required
        e_min:    starting energy (eV) of dos
        e_max:    ending energy (eV) of dos
        e_points: number of points of the get_dos
        width:    width in eV of the gaussians generated for each energy
        Returns:
        e_mesh:   energies in eV od the DOS
        dos:      density of states for each energy in e_mesh
        """
        height = 1.0 / (width * np.sqrt(2 * np.pi))
        e_mesh, step = np.linspace(e_min, e_max, num=e_points, endpoint=True,
                                   retstep=True)
        e_range = len(e_mesh)
        cbm_new_idx = None

        coefficients = self.get_interpolation_coefficients(iband="A")

        warnings.warn("The index of VBM/CBM is unknown; scissor is set to 0.0")
        scissor = 0.0

        nstv, vec = self.get_star_functions(latt_points, nsym, symop, nwave)
        ir_kpts = SpacegroupAnalyzer(structure).get_ir_reciprocal_mesh(mesh)
        ir_kpts = [k[0] for k in ir_kpts]
        weights = [k[1] for k in ir_kpts]
        w_sum = float(sum(weights))
        weights = [w / w_sum for w in weights]
        dos = np.zeros(e_range)
        for kpt, w in zip(ir_kpts, weights):
            for b in range(len(engre)):
                energy = get_energy(kpt, engre[b], nwave, nsym, nstv,
                                    vec)
                if b >= cbm_new_idx:
                    energy += scissor / 2.
                else:
                    energy -= scissor / 2.
                g = height * np.exp(-((e_mesh - energy) / width) ** 2 / 2.)
                dos += w * g
        return e_mesh, dos, len(engre)

    def get_energy(self, kpoint, iband, matrix=None, return_first_derivative=True,
                   return_second_derivative=False):
        """
        Compute energy for a k-point from star functions
        Args:
            kpoint: k-point coordinates as array
            engre: matrix containing the coefficients of fitted band from get_interpolation_coefficients() function
            nwave: number of G vectors
            nsym: number of symmetries
            nstv: number of vectors in a star function
            vec: the star vectors for each G vector and symmetry
            vec2: dot product of star vectors with cell matrix for each G vector
                and symmetry. Needed only to compute the derivatives of energy
            br_dir: cell matrix. Needed only to compute the derivatives of energy
            cbm: True if the considered band is a conduction band. False if it is a valence band
            out_vec2: outer product of vec2 with itself. It is calculated outside to improve performances
            return_second_derivative (bool): if true, it also returns the second derivative of
                energy used to calculate the effective mass for example.

        Returns:
            energy: the electronic energy at the k-point in input
            dene: 1st derivative of the electronic energy at the k-point in input
            second_derivative: 2nd derivative of the electronic energy at the k-point in input
        """
        arg = 2 * np.pi * self._star_vectors.dot(kpoint)
        cos_arg = np.cos(arg)
        spwre = (np.sum(cos_arg, axis=1) -
                 (self._num_symmetries - self._num_star_vectors)
                 ) / self._num_star_vectors

        coefficients = self.get_interpolation_coefficients(iband)

        energy = spwre.dot(coefficients)
        to_return = [energy * Ry_to_eV]

        if return_first_derivative:
            if matrix is None:
                matrix = np.eye(3)

            sin_arg = np.sin(arg)
            dspwre = np.sum(
                self._star_vector_products * sin_arg[:, :, np.newaxis], axis=1
                ) / self._num_star_vectors[:, np.newaxis]
            factor = hbar / 0.52917721067 * A_to_m * m_to_cm * Ry_to_eV
            matrix_norm = matrix / np.linalg.norm(matrix)
            first_derivative = np.dot(dspwre.T, coefficients)
            to_return.append(np.dot(matrix_norm, first_derivative) / factor)

        if return_second_derivative:
            ddspwre = np.sum(self._star_vector_products_sq * (
                -cos_arg[:, :, np.newaxis, np.newaxis]), axis=1
                ) / self._num_star_vectors[:, np.newaxis, np.newaxis]
            factor_a = 0.52917721067 ** 2 * Ry_to_eV
            factor_b = A_to_m ** 2 * hbar ** 2 / m_e
            second_derivative = np.dot(ddspwre.T, coefficients)
            to_return.append(1 / (second_derivative / factor_a) * e / factor_b)

        return tuple(to_return)

    def get_energies(self, kpoints, iband, matrix, sgn=None, scissor=0.0,
                     return_mass=True):
        """
        Args:
            kpoints ([1x3 array]): list of fractional coordinates of k-points
            iband (int): the band index for which the list of energy, velocity
                and mass is returned.
            matrix (3x3 np.ndarray): the direct lattice matrix used to convert
                the velocity (in fractional coordinates) to cartesian in
                boltztrap1 method.
            sgn (float): options are +1 for valence band and -1 for conduction
                bands sgn is ignored if scissor==0.0
            scissor (float): the amount by which the band gap is scissored.
            return_mass (bool): whether to return the effective mass values.

        Returns (tuple of energies, velocities, masses lists/np.ndarray):
            energies ([float]): energy values at kpts for a corresponding iband
            velocities ([3x1 array]): velocity vectors
            masses ([3x3 matrix]): list of effective mass tensors
        """
        if matrix is None:
            matrix = np.eye(3)
        if not sgn:
            if scissor == 0.0:
                sgn = 0.0
            else:
                raise ValueError('To apply scissor "sgn" is required: -1 or +1')

        fun = partial(self.get_energy, return_second_derivative=return_mass)
        inputs = [(k, iband) for k in kpoints]
        if self._n_jobs == 1:
            results = list(starmap(fun, inputs))
        else:
            with Pool(self._n_jobs) as p:
                results = p.starmap(fun, inputs)

        shift = sgn * scissor / 2.0
        energies = [r[0] - shift for r in results]
        velocities = [r[1] for r in results]

        if return_mass:
            masses = [r[2] for r in results]
            return energies, velocities, masses

        return energies, velocities

    def _get_energies_vectorized(self, kpoints, iband, matrix,
                                 sgn, scissor, return_mass=True):
        arg = np.tensordot(self._star_vectors, kpoints, axes=([2], [1])
                           ).transpose((2, 0, 1)) * 2 * np.pi

        # calculate energies
        cos_arg = np.cos(arg)
        spwre = (np.sum(cos_arg, axis=2) -
                 (self._num_symmetries - self._num_star_vectors)
                 ) / self._num_star_vectors
        shift = sgn * scissor / 2.0
        coefficients = self.get_interpolation_coefficients(iband)
        energies = (spwre @ coefficients) * Ry_to_eV - shift

        # calculate velocities
        sin_arg = np.sin(arg)
        dspwre = np.sum(self._star_vector_products *
                        sin_arg[:, :, :, np.newaxis],
                        axis=2) / self._num_star_vectors[:, np.newaxis]

        factor = hbar / 0.52917721067 * A_to_m * m_to_cm * Ry_to_eV
        norm_matrix = matrix / np.linalg.norm(matrix)
        velocities = dspwre.transpose((0, 2, 1)) @ coefficients
        velocities = np.abs(np.tensordot(norm_matrix, velocities,
                                         axes=([1], [1]))).T / factor

        if return_mass:
            ddspwre = np.sum(
                self._star_vector_products_sq *
                - cos_arg[:, :, :, np.newaxis, np.newaxis], axis=2
            ) / self._num_star_vectors[:, np.newaxis, np.newaxis]
            factor_a = 0.52917721067 ** 2 * Ry_to_eV
            factor_b = A_to_m ** 2 * hbar ** 2 / m_e
            masses = ddspwre.transpose((0, 3, 2, 1)) @ coefficients
            masses = 1 / (masses / factor_a) * e / factor_b
            return energies, velocities, masses

        return energies, velocities
