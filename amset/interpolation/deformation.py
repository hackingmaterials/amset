import logging

import numpy as np

from amset.constants import defaults
from amset.deformation.common import desymmetrize_deformation_potentials
from amset.deformation.io import load_deformation_potentials
from amset.electronic_structure.kpoints import get_mesh_from_kpoint_numbers
from amset.electronic_structure.symmetry import expand_kpoints
from amset.interpolation.periodic import PeriodicLinearInterpolator

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

logger = logging.getLogger(__name__)


class DeformationPotentialInterpolator(PeriodicLinearInterpolator):
    @classmethod
    def from_file(cls, filename, scale=1.0):
        deform_potentials, kpoints, structure = load_deformation_potentials(filename)
        deform_potentials = {s: d * scale for s, d in deform_potentials.items()}
        return cls.from_deformation_potentials(deform_potentials, kpoints, structure)

    @classmethod
    def from_deformation_potentials(
        cls, deformation_potentials, kpoints, structure, symprec=defaults["symprec"]
    ):
        logger.info("Initializing deformation potential interpolator")

        mesh_dim = get_mesh_from_kpoint_numbers(kpoints)
        if np.product(mesh_dim) == len(kpoints):
            return cls.from_data(kpoints, deformation_potentials)

        full_kpoints, rotations, _, _, op_mapping, kp_mapping = expand_kpoints(
            structure, kpoints, time_reversal=True, return_mapping=True, symprec=symprec
        )
        logger.warning("Desymmetrizing deformation potentials, this could go wrong.")
        deformation_potentials = desymmetrize_deformation_potentials(
            deformation_potentials, structure, rotations, op_mapping, kp_mapping
        )
        return cls.from_data(full_kpoints, deformation_potentials)
