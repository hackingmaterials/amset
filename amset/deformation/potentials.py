import logging

import numpy as np
from pymatgen.analysis.elasticity.strain import Deformation
from pymatgen.core.tensors import TensorMapping
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from amset.constants import defaults
from amset.electronic_structure.kpoints import (
    get_kpoint_indices,
    get_kpoints_from_bandstructure,
    get_mesh_from_kpoint_diff,
)
from amset.electronic_structure.symmetry import (
    expand_bandstructure,
    rotate_bandstructure,
)

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"


_mapping_tol = 0.002

logger = logging.getLogger(__name__)


def get_mesh_from_band_structure(bandstructure):
    kpoints = np.array([k.frac_coords for k in bandstructure.kpoints])
    mesh, is_shifted = get_mesh_from_kpoint_diff(kpoints)
    return tuple(mesh.round().astype(int)), is_shifted


def calculate_deformation(bulk_structure, deformed_structure):
    """
    Args:
        bulk_structure: undeformed structure
        deformed_structure: deformed structure

    Returns:
        deformation matrix
    """
    bulk_lattice = bulk_structure.lattice.matrix
    deformed_lattice = deformed_structure.lattice.matrix
    return np.transpose(np.dot(np.linalg.inv(bulk_lattice), deformed_lattice)).round(10)


def get_strain_mapping(bulk_structure, deformation_calculations):
    strain_mapping = TensorMapping(tol=_mapping_tol)
    for i, calc in enumerate(deformation_calculations):
        deformed_structure = calc["bandstructure"].structure
        matrix = calculate_deformation(bulk_structure, deformed_structure)
        strain = Deformation(matrix).green_lagrange_strain
        strain_mapping[strain] = calc
    return strain_mapping


def get_symmetrized_strain_mapping(
    bulk_structure, strain_mapping, symprec=defaults["symprec"]
):
    sga = SpacegroupAnalyzer(bulk_structure, symprec=symprec)
    cart_ops = sga.get_symmetry_operations(cartesian=True)
    frac_ops = sga.get_symmetry_operations(cartesian=False)
    for strain, calc in strain_mapping.items():
        # expand band structure to cover full brillouin zone, otherwise rotation won't
        # include all necessary points
        calc["bandstructure"] = expand_bandstructure(
            calc["bandstructure"], symprec=symprec
        )

    for strain, calc in strain_mapping.items():
        for cart_op, frac_op in zip(cart_ops, frac_ops):
            tstrain = strain.transform(cart_op)
            independent = tstrain.get_deformation_matrix().is_independent(_mapping_tol)
            if independent and tstrain not in strain_mapping:
                rband = rotate_bandstructure(calc["bandstructure"], frac_op)
                tcalc = {"reference": calc["reference"], "bandstructure": rband}
                strain_mapping[tstrain] = tcalc

    return strain_mapping


def get_strain_deformation_potential(
    strain,
    bulk_bandstructure,
    deformation_bandstructure,
    bulk_reference,
    deformation_reference,
):
    strain = strain.round(5)
    flat_strain = strain.ravel()
    strain_amount = flat_strain[np.abs(flat_strain).argmax()]
    ref_diff = bulk_reference - deformation_reference

    kpoints = get_kpoints_from_bandstructure(bulk_bandstructure)
    mesh, is_shifted = get_mesh_from_band_structure(bulk_bandstructure)
    indices_to_keep = get_kpoint_indices(kpoints, mesh, is_shifted=is_shifted)

    deform_kpoints = get_kpoints_from_bandstructure(deformation_bandstructure)
    deform_indices = get_kpoint_indices(deform_kpoints, mesh, is_shifted=is_shifted)

    if not set(indices_to_keep).issubset(set(deform_indices)):
        raise RuntimeError(
            "Deformation band structure doesn't contain the same k-points "
            "as the bulk band structure. Try changing symprec."
        )

    deform_map = np.full(np.max(deform_indices) + 1, -1)
    deform_map[deform_indices] = np.arange(len(deform_indices))
    select_indices = deform_map[indices_to_keep]

    energy_diff = {}
    for spin, spin_origin in bulk_bandstructure.bands.items():
        diff = spin_origin - deformation_bandstructure.bands[spin][:, select_indices]
        diff -= ref_diff
        energy_diff[spin] = np.abs(diff / strain_amount)

    return energy_diff


def calculate_deformation_potentials(bulk_calculation, strain_mapping):
    deformation_potentials = {
        s: np.zeros(b.shape + (3, 3))
        for s, b in bulk_calculation["bandstructure"].bands.items()
    }
    norm = np.zeros((3, 3))
    for strain, deformation_calculation in strain_mapping.items():
        deform = get_strain_deformation_potential(
            strain,
            bulk_calculation["bandstructure"],
            deformation_calculation["bandstructure"],
            bulk_calculation["reference"],
            deformation_calculation["reference"],
        )

        # sometimes the strain includes numerical noise, this will filter out components
        # of the strain that are noisy. In reality there should only be one or two
        # components of the strain and they should have the same magnitude
        max_strain = np.abs(strain).max()
        strain_loc = np.abs(strain) > 0.25 * max_strain
        loc_x, loc_y = np.where(strain_loc)
        for spin, spin_deform in deform.items():
            deformation_potentials[spin][:, :, loc_x, loc_y] += spin_deform[..., None]
            norm += strain_loc

    for spin in deformation_potentials:
        deformation_potentials[spin] /= norm[None, None]

    return deformation_potentials


def strain_coverage_ok(strains):
    sum_strains = np.abs(strains).sum(axis=0)
    sum_strains = sum_strains.round(5)
    return not np.any(sum_strains == 0)


def extract_bands(deformation_potentials, ibands):
    return {spin: deformation_potentials[spin][b_idx] for spin, b_idx in ibands.items()}
