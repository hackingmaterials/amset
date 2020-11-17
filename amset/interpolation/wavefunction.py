import logging

import numpy as np

from amset.constants import defaults
from amset.electronic_structure.kpoints import get_mesh_from_kpoint_numbers
from amset.electronic_structure.symmetry import expand_kpoints
from amset.interpolation.periodic import (
    PeriodicLinearInterpolator,
    group_bands_and_kpoints,
)
from amset.util import array_from_buffer, create_shared_array
from amset.wavefunction.common import desymmetrize_coefficients, get_overlap, is_ncl
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
        bands, kpoints, single_overlap = group_bands_and_kpoints(
            band_a, kpoint_a, band_b, kpoint_b
        )
        p = self.get_coefficients(spin, bands, kpoints)
        overlap = get_overlap(p[0], p[1:])
        if single_overlap:
            return overlap[0]
        else:
            return overlap
