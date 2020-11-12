import logging
from typing import List, Tuple, Union

import numpy as np
from pymatgen import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from spglib import spglib

from amset.constants import defaults, ktol

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

logger = logging.getLogger(__name__)


def kpoints_to_first_bz(
    kpoints: np.ndarray, tol=ktol, negative_zone_boundary: bool = True
) -> np.ndarray:
    """Translate fractional k-points to the first Brillouin zone.

    I.e. all k-points will have fractional coordinates:
        -0.5 <= fractional coordinates < 0.5

    Args:
        kpoints: The k-points in fractional coordinates.
        tol: Fractional tolerance for evaluating zone boundary points.
        negative_zone_boundary: Whether to use -0.5 (spglib convention) or
            0.5 (VASP convention) for zone boundary points.

    Returns:
        The translated k-points.
    """
    kp = kpoints - np.round(kpoints)

    # account for small rounding errors for 0.5
    round_dp = int(np.log10(1 / tol))
    krounded = np.round(kp, round_dp)

    if negative_zone_boundary:
        kp[krounded == 0.5] = -0.5
    else:
        kp[krounded == -0.5] = 0.5
    return kp


def sort_boltztrap_to_spglib(kpoints):
    sort_idx = np.lexsort(
        (
            kpoints[:, 2],
            kpoints[:, 2] < 0,
            kpoints[:, 1],
            kpoints[:, 1] < 0,
            kpoints[:, 0],
            kpoints[:, 0] < 0,
        )
    )
    boltztrap_kpoints = kpoints[sort_idx]

    sort_idx = np.lexsort(
        (
            boltztrap_kpoints[:, 0],
            boltztrap_kpoints[:, 0] < 0,
            boltztrap_kpoints[:, 1],
            boltztrap_kpoints[:, 1] < 0,
            boltztrap_kpoints[:, 2],
            boltztrap_kpoints[:, 2] < 0,
        )
    )
    return sort_idx


def get_kpoints_tetrahedral(
    kpoint_mesh: Union[float, List[int]],
    structure: Structure,
    symprec: float = defaults["symprec"],
    time_reversal_symmetry: bool = True,
) -> Tuple[np.ndarray, ...]:
    """Gets the symmetry inequivalent k-points from a k-point mesh.

    Follows the same process as SpacegroupAnalyzer.get_ir_reciprocal_mesh
    but is faster and allows returning of the full k-point mesh and mapping.

    Args:
        kpoint_mesh: The k-point mesh as a 1x3 array. E.g.,``[6, 6, 6]``.
            Alternatively, if a single value is provided this will be
            treated as a k-point spacing cut-off and the k-points will be generated
            automatically.  Cutoff is length in Angstroms and corresponds to
            non-overlapping radius in a hypothetical supercell (Moreno-Soler length
            cutoff).
        structure: A structure.
        symprec: Symmetry tolerance used when determining the symmetry
            inequivalent k-points on which to interpolate.
        time_reversal_symmetry: Whether the system has time reversal symmetry.

    Returns:
        The irreducible k-points and their weights as tuple, formatted as::

            (ir_kpoints, weights)

        If return_full_kpoints, the data will be returned as::

            (ir_kpoints, weights, kpoints, ir_kpoints_idx, ir_to_full_idx)

        Where ``ir_kpoints_idx`` is the index of the unique irreducible k-points
        in ``kpoints``. ``ir_to_full_idx`` is a list of indices that can be
        used to construct the full Brillouin zone from the ir_mesh. Note the
        ir -> full conversion will only work with calculated scalar properties
        such as energy (not vector properties such as velocity).
    """
    from amset.electronic_structure.tetrahedron import get_tetrahedra

    if isinstance(kpoint_mesh, (int, float)):
        kpoint_mesh = get_kpoint_mesh(structure, kpoint_mesh)

    atoms = AseAtomsAdaptor().get_atoms(structure)

    if not symprec:
        symprec = 1e-8

    grid_mapping, grid_address = spglib.get_ir_reciprocal_mesh(
        kpoint_mesh, atoms, symprec=symprec, is_time_reversal=time_reversal_symmetry
    )
    full_kpoints = grid_address / kpoint_mesh

    tetra, ir_tetrahedra_idx, ir_tetrahedra_to_full_idx, tet_weights = get_tetrahedra(
        structure.lattice.reciprocal_lattice.matrix,
        grid_address,
        kpoint_mesh,
        grid_mapping,
    )

    ir_kpoints_idx, ir_to_full_idx, weights = np.unique(
        grid_mapping, return_inverse=True, return_counts=True
    )
    ir_kpoints = full_kpoints[ir_kpoints_idx]

    return (
        ir_kpoints,
        weights,
        full_kpoints,
        ir_kpoints_idx,
        ir_to_full_idx,
        tetra,
        ir_tetrahedra_idx,
        ir_tetrahedra_to_full_idx,
        tet_weights,
    )


def get_kpoint_mesh(structure: Structure, cutoff_length: float, force_odd: bool = True):
    """Calculate reciprocal-space sampling with real-space cut-off."""
    reciprocal_lattice = structure.lattice.reciprocal_lattice_crystallographic

    # Get reciprocal cell vector magnitudes
    abc_recip = np.array(reciprocal_lattice.abc)

    mesh = np.ceil(abc_recip * 2 * cutoff_length).astype(int)

    if force_odd:
        mesh += (mesh + 1) % 2

    return mesh


def get_mesh_from_kpoint_diff(kpoints):
    kpoints = np.array(kpoints)

    # whether the k-point mesh is shifted or Gamma centered mesh
    is_shifted = np.min(np.linalg.norm(kpoints, axis=1)) > 1e-6

    unique_a = np.unique(kpoints[:, 0])
    unique_b = np.unique(kpoints[:, 1])
    unique_c = np.unique(kpoints[:, 2])

    if len(unique_a) == 1:
        na = 1
    else:
        na = 1 / np.min(np.diff(unique_a))

    if len(unique_b) == 1:
        nb = 1
    else:
        nb = 1 / np.min(np.diff(unique_b))

    if len(unique_c) == 1:
        nc = 1
    else:
        nc = 1 / np.min(np.diff(unique_c))

    # due to limited precission of the input k-points, the mesh is returned as a float
    return np.array([na, nb, nc]), is_shifted


def get_mesh_from_kpoint_numbers(kpoints, tol=ktol):
    round_dp = int(np.log10(1 / tol))

    kpoints = kpoints_to_first_bz(kpoints)
    round_kpoints = np.round(kpoints, round_dp)

    nx = len(np.unique(round_kpoints[:, 0]))
    ny = len(np.unique(round_kpoints[:, 1]))
    nz = len(np.unique(round_kpoints[:, 2]))

    return nx, ny, nz


def get_kpoint_indices(kpoints, mesh, is_shifted=False):
    mesh = np.array(mesh)
    shift = np.array([1, 1, 1]) if is_shifted else np.array([0, 0, 0])
    min_kpoint = -np.floor(mesh / 2).round().astype(int)
    addresses = ((kpoints + shift / (mesh * 2)) * mesh).round().astype(int)
    shifted = addresses - min_kpoint
    nyz = mesh[1] * mesh[2]
    nz = mesh[2]
    indices = shifted[:, 0] * nyz + shifted[:, 1] * nz + shifted[:, 2]
    return indices.round().astype(int)


def get_kpoints_from_bandstructure(bandstructure, cartesian=False):
    if cartesian:
        return np.array([k.coords for k in bandstructure.kpoints])
    else:
        return np.array([k.frac_coords for k in bandstructure.kpoints])
