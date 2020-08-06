import logging

import numpy as np

from amset.electronic_structure.kpoints import get_mesh_from_kpoint_diff

metal_str = {True: "metallic", False: "semiconducting"}

logger = logging.getLogger(__name__)


def get_mesh_from_band_structure(bandstructure):
    kpoints = np.array([k.frac_coords for k in bandstructure.kpoints])
    return tuple(get_mesh_from_kpoint_diff(kpoints).round().astype(int))


def check_calculations(original_calc, deformed_calcs):
    original_mesh = get_mesh_from_band_structure(original_calc["bandstructure"])
    original_species = tuple(original_calc["bandstructure"].structure.species)
    original_is_metal = original_calc["bandstructure"].is_metal()

    keep_calcs = []
    for calc in deformed_calcs:
        mesh = get_mesh_from_band_structure(calc["bandstructure"])
        if mesh != original_mesh:
            raise RuntimeError(
                "Calculations were not performed using the same k-point "
                "mesh\n{} != {}".format(mesh, original_mesh)
            )

        species = tuple(calc["bandstructure"].structure.species)
        if species != original_species:
            raise RuntimeError("Calculations were performed using different structures")
        is_metal = calc["bandstructure"].is_metal()

        if is_metal != original_is_metal:
            logger.warning(
                "Bulk structure is {} whereas deformed structure is {}.\n"
                "Skipping deformation.".format(
                    metal_str[original_is_metal], metal_str[is_metal]
                )
            )
        else:
            keep_calcs.append(calc)
    return keep_calcs
