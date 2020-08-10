import logging

import numpy as np
from pymatgen.analysis.elasticity.strain import Strain
from pymatgen.transformations.standard_transformations import (
    DeformStructureTransformation,
)

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

logger = logging.getLogger(__name__)


def get_strain_fields():
    """Generates a list of strain fields for a rank 2 strain tensor in Voigt notation"""
    inds = [(i,) for i in range(6)]
    strain_fields = np.zeros((len(inds), 6))
    for n, i in enumerate(inds):
        np.put(strain_fields[n], i, 1)
    strain_fields[:, 3:] *= 2
    return strain_fields


def get_strains(distance=0.005):
    strain_fields = get_strain_fields()
    strains = []
    for strain_field in strain_fields:
        strains.append(Strain.from_voigt(strain_field * abs(distance)))
        strains.append(Strain.from_voigt(strain_field * -abs(distance)))
    return strains


def get_deformations(distance):
    strains = get_strains(distance=distance)
    deformations = [s.get_deformation_matrix() for s in strains]
    return deformations


def get_deformed_structures(structure, deformations):
    deformed_structures = []
    for i, deformation in enumerate(deformations):
        dst = DeformStructureTransformation(deformation)
        deformed_structures.append(dst.apply_transformation(structure))
    return deformed_structures
