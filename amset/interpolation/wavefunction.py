import logging

import numba
import numpy as np
from interpolation.splines import eval_linear
from interpolation.splines import extrap_options as xto

from amset.constants import defaults
from amset.electronic_structure.kpoints import get_mesh_from_kpoint_numbers
from amset.electronic_structure.symmetry import expand_kpoints
from amset.interpolation.periodic import (
    PeriodicLinearInterpolator,
    group_bands_and_kpoints,
)
from amset.util import array_from_buffer, create_shared_array
from amset.wavefunction.common import desymmetrize_coefficients, is_ncl
from amset.wavefunction.io import load_coefficients

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

logger = logging.getLogger(__name__)


class WavefunctionOverlapCalculator(PeriodicLinearInterpolator):
    def __init__(self, nbands, data_shape, interpolators, ncl, gpoints):
        super().__init__(nbands, data_shape, interpolators)
        self.ncl = ncl
        self.gpoints = gpoints

    def to_reference(self):
        interpolator_references = self._interpolators_to_reference()
        gpoints_buffer, self.gpoints = create_shared_array(
            self.gpoints, return_shared_data=True
        )
        return (
            self.nbands,
            self.data_shape,
            interpolator_references,
            self.ncl,
            gpoints_buffer,
        )

    @classmethod
    def from_reference(
        cls, nbands, data_shape, interpolator_references, ncl, gpoints_buffer
    ):
        interpolators = cls._interpolators_from_reference(interpolator_references)
        gpoints = array_from_buffer(gpoints_buffer)
        return cls(nbands, data_shape, interpolators, ncl, gpoints)

    @classmethod
    def from_data(cls, kpoints, data, gpoints=None, gaussian=None):
        if gpoints is None:
            raise ValueError("gpoints required for initialization")
        ncl = is_ncl(data)
        grid_kpoints, mesh_dim, sort_idx = cls._grid_kpoints(kpoints)
        nbands, data_shape, interpolators = cls._setup_interpolators(
            data, grid_kpoints, mesh_dim, sort_idx, gaussian
        )
        return cls(nbands, data_shape, interpolators, ncl, gpoints)

    @classmethod
    def from_file(cls, filename):
        coeff, gpoints, kpoints, structure = load_coefficients(filename)
        return cls.from_coefficients(coeff, gpoints, kpoints, structure)

    @classmethod
    def from_coefficients(
        cls, coefficients, gpoints, kpoints, structure, symprec=defaults["symprec"]
    ):
        logger.info("Initializing wavefunction overlap calculator")

        mesh_dim = get_mesh_from_kpoint_numbers(kpoints)
        if np.product(mesh_dim) == len(kpoints):
            return cls.from_data(kpoints, coefficients, gpoints)

        full_kpoints, *symmetry_mapping = expand_kpoints(
            structure, kpoints, time_reversal=True, return_mapping=True, symprec=symprec
        )
        coefficients = desymmetrize_coefficients(
            coefficients, gpoints, kpoints, structure, *symmetry_mapping
        )
        return cls.from_data(full_kpoints, coefficients, gpoints)

    def get_coefficients(self, spin, bands, kpoints):
        interp_coeffs = self.interpolate(spin, bands, kpoints)
        if self.ncl:
            interp_coeffs /= np.linalg.norm(interp_coeffs, axis=(-2, -1))[
                ..., None, None
            ]
        else:
            interp_coeffs /= np.linalg.norm(interp_coeffs, axis=-1)[:, None]
        return interp_coeffs

    def get_overlap(self, spin, band_a, kpoint_a, band_b, kpoint_b):
        # generally, we don't want to do the interpolation for all band and
        # k-points simultaneously as this can use a lot of memory. This becomes
        # an issue when using multiprocessing. I.e., if you're parralellising over
        # 24 cores on a single node, you only have access 1/24 the total memory
        # for each core. Here we use numba jit to calculate the overlap of each
        # band/kpoint sequentially. In testing this only gives ~2x slow down vs
        # the simultaneous approach, while using an order of magnitude less memory.
        bands, kpoints, single_overlap = group_bands_and_kpoints(
            band_a, kpoint_a, band_b, kpoint_b
        )
        grid, data = self.interpolators[spin]
        v = np.concatenate([np.asarray(bands)[:, None], np.asarray(kpoints)], axis=1)

        if self.ncl:
            overlap = _get_overlap_ncl(grid, data, v, self.data_shape[0])
        else:
            overlap = _get_overlap(grid, data, v, self.data_shape[0])

        if single_overlap:
            return overlap[0]
        else:
            return overlap


@numba.njit
def _get_overlap(grid, data, points, n_coeffs):
    initial = np.zeros(n_coeffs, dtype=np.complex64)
    initial.real[:] = eval_linear(grid, data.real, points[0], xto.LINEAR)
    initial.imag[:] = eval_linear(grid, data.imag, points[0], xto.LINEAR)
    initial /= np.linalg.norm(initial)
    initial[:] = np.conj(initial)

    res = np.zeros(points.shape[0] - 1)
    final = np.zeros(n_coeffs, dtype=np.complex64)
    for i in range(1, points.shape[0]):
        final.real[:] = eval_linear(grid, data.real, points[i], xto.LINEAR)
        final.imag[:] = eval_linear(grid, data.imag, points[i], xto.LINEAR)
        final /= np.linalg.norm(final)
        res[i - 1] = np.abs(np.dot(final, initial)) ** 2
    return res


@numba.njit
def _get_overlap_ncl(grid, data, points, n_coeffs):
    initial = np.zeros((n_coeffs, 2), dtype=np.complex64)
    initial.real[:] = eval_linear(grid, data.real, points[0], xto.LINEAR).reshape(
        (n_coeffs, 2)
    )
    initial.imag[:] = eval_linear(grid, data.imag, points[0], xto.LINEAR).reshape(
        (n_coeffs, 2)
    )
    initial /= np.linalg.norm(initial)
    initial[:] = np.conj(initial)

    res = np.zeros(points.shape[0] - 1)
    final = np.zeros((n_coeffs, 2), dtype=np.complex64)
    for i in range(1, points.shape[0]):
        final.real[:] = eval_linear(grid, data.real, points[i], xto.LINEAR).reshape(
            (n_coeffs, 2)
        )
        final.imag[:] = eval_linear(grid, data.imag, points[i], xto.LINEAR).reshape(
            (n_coeffs, 2)
        )
        final /= np.linalg.norm(final)

        sum_ = 0j
        for j in range(final.shape[0]):
            sum_ += initial[j, 0] * final[j, 0] + initial[j, 1] * final[j, 1]

        res[i - 1] = abs(sum_) ** 2

    return res


class UnityWavefunctionOverlap:
    def __init__(self, *args, **kwargs):
        pass

    def to_reference(self):
        return [1, 2, 3]

    @classmethod
    def from_reference(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    @classmethod
    def from_data(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def get_overlap(self, *args, **kwargs):
        return 1
