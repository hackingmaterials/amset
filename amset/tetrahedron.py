from typing import Dict, Optional

import numpy as np

from amset.misc.util import groupby
from pymatgen import Spin, Structure

from quadpy import triangle, quadrilateral

from pymatgen.util.coord import pbc_diff

"""
     6-------7
    /|      /|
   / |     / |
  4-------5  |
  |  2----|--3
  | /     | /
  |/      |/
  0-------1

 i: vec        neighbours
 0: O          1, 2, 4
 1: a          0, 3, 5
 2: b          0, 3, 6
 3: a + b      1, 2, 7
 4: c          0, 5, 6
 5: c + a      1, 4, 7
 6: c + b      2, 4, 7
 7: c + a + b  3, 5, 6
"""

_main_diagonals = (
    (1, 1, 1),  # 0-7
    (-1, 1, 1),  # 1-6
    (1, -1, 1),  # 2-5
    (1, 1, -1),  # 3-4
)

_tetrahedron_vectors = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
)

numerical_integration_defaults = {
    "high": {
        "triangle": triangle.xiao_gimbutas_50(),
        "quadrilateral": quadrilateral.sommariva_50(),
    },
    "medium": {
        "triangle": triangle.xiao_gimbutas_06(),
        "quadrilateral": quadrilateral.sommariva_06(),
    },
    "low": {
        "triangle": triangle.centroid(),
        "quadrilateral": quadrilateral.dunavant_00(),
    },
}


def get_main_diagonal(reciprocal_lattice: np.ndarray) -> int:
    # want a list of tetrahedra as (k1, k2, k3, k4); as per the Bloechl paper,
    # the order of the k-points is irrelevant and therefore they should be ordered
    # in increasing number
    # All tetrahedrons for a specific submesh will share one common diagonal.
    # To minimise interpolaiton distances, we should choose the shortest
    # diagonal
    diagonal_lengths = []
    for diagonal in _main_diagonals:
        length = np.linalg.norm(reciprocal_lattice @ diagonal)
        diagonal_lengths.append(length)

    return int(np.argmin(diagonal_lengths))


def get_relative_tetrahedron_vertices(reciprocal_lattice):
    shortest_index = get_main_diagonal(reciprocal_lattice)

    if shortest_index == 0:
        pairs = ((1, 3), (1, 5), (2, 3), (2, 6), (4, 5), (4, 6))
        main = (0, 7)

    elif shortest_index == 1:
        pairs = ((0, 2), (0, 4), (2, 3), (3, 7), (4, 5), (5, 7))
        main = (1, 6)

    elif shortest_index == 2:
        pairs = ((0, 1), (0, 4), (1, 3), (3, 7), (4, 6), (6, 7))
        main = (2, 5)

    elif shortest_index == 3:
        pairs = ((0, 1), (0, 2), (1, 5), (2, 6), (5, 7), (6, 7))
        main = (3, 4)

    else:
        assert False

    tetras = np.sort([main + x for x in pairs])
    return _tetrahedron_vectors[tetras]


def get_tetrahedra(
    reciprocal_lattice: np.ndarray,
    grid_address: np.ndarray,
    mesh: np.ndarray,
    grid_address_mapping,
):
    tetrahedron_vertices = get_relative_tetrahedron_vertices(reciprocal_lattice)

    grid_order = [1, mesh[0], mesh[0] * mesh[1]]

    all_grid_points = np.repeat(grid_address, [24] * len(grid_address), axis=0)
    all_vertices = np.tile(tetrahedron_vertices, (len(grid_address), 1, 1))
    points = all_grid_points.reshape(all_vertices.shape) + all_vertices

    # fancy magic from phonopy to get neighboring indices given relative coordinates
    tetrahedra = np.dot(points % mesh, grid_order)

    ir_tetrahedra_vertices = grid_address_mapping[tetrahedra]
    _, ir_tetrahedra_idx, ir_tetrahedra_to_full_idx, ir_weights = np.unique(
        np.sort(ir_tetrahedra_vertices),
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )

    return tetrahedra, ir_tetrahedra_idx, ir_tetrahedra_to_full_idx, ir_weights


class TetrahedralBandStructure(object):
    def __init__(
        self,
        energies: Dict[Spin, np.ndarray],
        kpoints: np.ndarray,
        tetrahedra: np.ndarray,
        structure: Structure,
        ir_tetrahedra_idx: Optional[np.ndarray] = None,
        ir_tetrahedra_to_full_idx: Optional[np.ndarray] = None,
        ir_weights: Optional[np.ndarray] = None,
    ):
        tparams = (ir_tetrahedra_idx, ir_tetrahedra_to_full_idx, ir_weights)

        if len(set([x is None for x in tparams])) != 1:
            raise ValueError(
                "Either all or none of ir_tetrahedra_idx, "
                "ir_tetrahedra_to_full_idx and ir_weights should be set."
            )

        if ir_tetrahedra_idx is None:
            ir_tetrahedra_idx = np.arange(len(kpoints))
            ir_tetrahedra_to_full_idx = np.ones_like(ir_tetrahedra_idx)
            ir_weights = np.ones_like(ir_tetrahedra_idx)

        self.energies = energies
        self.kpoints = kpoints
        self.ir_tetrahedra_idx = ir_tetrahedra_idx
        self.ir_tetrahedra_to_full_idx = ir_tetrahedra_to_full_idx
        self.ir_weights = ir_weights

        # need to keep track of full tetrahedra to recover full k-point indices
        # when calculating scattering rates (i.e., k-k' is symmetry inequivalent).
        self.tetrahedra, _ = process_tetrahedra(tetrahedra, self.energies)

        # store irreducible tetrahedra and use energies to calculate diffs and min/maxes
        self.ir_tetrahedra, self.ir_tetrahedra_energies = process_tetrahedra(
            tetrahedra[self.ir_tetrahedra_idx], self.energies
        )

        # the remaining properties are given for each irreducible tetrahedra
        self.e21, self.e31, self.e41, self.e32, self.e42, self.e43 = get_tetrahedra_energy_diffs(
            self.ir_tetrahedra_energies
        )

        self.max_tetrahedra_energies, self.min_tetrahedra_energies = get_max_min_tetrahedra_energies(
            self.ir_tetrahedra_energies
        )

        self.cross_section_weights = get_tetrahedra_cross_section_weights(
            structure.lattice.reciprocal_lattice.matrix,
            self.kpoints,
            self.ir_tetrahedra,
            self.e21,
            self.e31,
            self.e41,
        )

        self._tetrahedron_volume = 1 / len(tetrahedra)

        self.grouped_ir_to_full = groupby(
            np.arange(len(ir_tetrahedra_to_full_idx)), ir_tetrahedra_to_full_idx
        )

    def get_intersecting_tetrahedra(self, spin, energy, band_idx=None):
        max_energies = self.max_tetrahedra_energies[spin]
        min_energies = self.min_tetrahedra_energies[spin]

        if band_idx is not None:
            mask = np.full_like(max_energies, False, dtype=bool)
            mask[band_idx] = True

            return (min_energies < energy) & (max_energies > energy) & mask

        else:
            return (min_energies < energy) & (max_energies > energy)

    def get_tetrahedra_density_of_states(
        self,
        spin,
        energy,
        return_contributions=False,
        symmetry_reduce=True,
        band_idx=None,
    ):
        tetrahedra_mask = self.get_intersecting_tetrahedra(
            spin, energy, band_idx=band_idx
        )

        if not np.any(tetrahedra_mask):
            if return_contributions:
                return [], [], [], []
            else:
                return []

        energies = self.ir_tetrahedra_energies[spin][tetrahedra_mask]
        e21 = self.e21[spin][tetrahedra_mask]
        e31 = self.e31[spin][tetrahedra_mask]
        e41 = self.e41[spin][tetrahedra_mask]
        e32 = self.e32[spin][tetrahedra_mask]
        e42 = self.e42[spin][tetrahedra_mask]
        e43 = self.e43[spin][tetrahedra_mask]
        cs_weights = self.cross_section_weights[spin][tetrahedra_mask]

        cond_a_mask = (energies[:, 0] < energy) & (energy < energies[:, 1])
        cond_b_mask = (energies[:, 1] <= energy) & (energy < energies[:, 2])
        cond_c_mask = (energies[:, 2] <= energy) & (energy < energies[:, 3])

        ee1 = energy - energies[:, 0]
        ee2 = energy - energies[:, 1]
        e4e = energies[:, 3] - energy

        tetrahedra_dos = np.zeros(len(energies))

        tetrahedra_dos[cond_a_mask] = _get_density_of_states_a(
            ee1[cond_a_mask], e21[cond_a_mask], e31[cond_a_mask], e41[cond_a_mask]
        )

        tetrahedra_dos[cond_b_mask] = _get_density_of_states_b(
            ee2[cond_b_mask],
            e21[cond_b_mask],
            e31[cond_b_mask],
            e41[cond_b_mask],
            e32[cond_b_mask],
            e42[cond_b_mask],
        )

        tetrahedra_dos[cond_c_mask] = _get_density_of_states_c(
            e4e[cond_c_mask], e41[cond_c_mask], e42[cond_c_mask], e43[cond_c_mask]
        )

        tetrahedra_dos *= self._tetrahedron_volume

        band_idx, tetrahedra_idx = np.where(tetrahedra_mask)
        tetrahedra_weights = self.ir_weights[tetrahedra_idx]

        if symmetry_reduce:
            tetrahedra_dos *= tetrahedra_weights

        else:
            # transform the mask to the full BZ
            band_idx = np.repeat(band_idx, tetrahedra_weights)
            tetrahedra_grouped = self.grouped_ir_to_full[tetrahedra_idx]
            tetrahedra_idx = np.concatenate(tetrahedra_grouped)
            tetrahedra_mask = (band_idx, tetrahedra_idx)

            # tetrahedra_mask = tetrahedra_mask[:, self.ir_tetrahedra_to_full_idx]
            # get a mask to expand the remaining properties with shape (ntetra,)
            # to the full BZ
            expand_tetrahedra = np.repeat(np.arange(len(ee1)), tetrahedra_weights)
            tetrahedra_dos = tetrahedra_dos[expand_tetrahedra]
            cond_a_mask = cond_a_mask[expand_tetrahedra]
            cond_b_mask = cond_b_mask[expand_tetrahedra]
            cond_c_mask = cond_c_mask[expand_tetrahedra]
            ee1 = ee1[expand_tetrahedra]
            ee2 = ee2[expand_tetrahedra]
            e4e = e4e[expand_tetrahedra]
            e21 = e21[expand_tetrahedra]
            e31 = e31[expand_tetrahedra]
            e41 = e41[expand_tetrahedra]
            e32 = e32[expand_tetrahedra]
            e42 = e42[expand_tetrahedra]
            e43 = e43[expand_tetrahedra]
            cs_weights = cs_weights[expand_tetrahedra]

        if return_contributions:
            frac_21 = ee1 / e21
            frac_31 = ee1 / e31
            frac_41 = ee1 / e41
            frac_32 = ee2 / e32
            frac_42 = ee2 / e42
            frac_c_41 = e4e / e41
            frac_c_42 = e4e / e42
            frac_c_43 = e4e / e43

            frac_21[(frac_21 == np.nan) | (frac_21 == np.inf)] = 1
            frac_31[(frac_31 == np.nan) | (frac_31 == np.inf)] = 1
            frac_41[(frac_41 == np.nan) | (frac_41 == np.inf)] = 1
            frac_32[(frac_32 == np.nan) | (frac_32 == np.inf)] = 1
            frac_42[(frac_42 == np.nan) | (frac_42 == np.inf)] = 1
            frac_c_41[(frac_c_41 == np.nan) | (frac_c_41 == np.inf)] = 1
            frac_c_42[(frac_c_42 == np.nan) | (frac_c_42 == np.inf)] = 1
            frac_c_43[(frac_c_43 == np.nan) | (frac_c_43 == np.inf)] = 1

            contributions = (
                cond_a_mask,
                cond_b_mask,
                cond_c_mask,
                frac_21,
                frac_31,
                frac_41,
                frac_32,
                frac_42,
                frac_c_41,
                frac_c_42,
                frac_c_43,
            )

            return tetrahedra_dos, tetrahedra_mask, cs_weights, contributions

        else:
            return tetrahedra_dos

    def get_density_of_states(
        self, emin, emax, npoints, weights=None, sum_spins=False, band_idx=None
    ):
        energies = np.linspace(emin, emax, npoints)

        dos = {}
        for spin in self.energies.keys():
            if isinstance(weights, dict):
                # weights given for each spin channel
                spin_weights = weights[spin]
            else:
                spin_weights = weights

            if isinstance(band_idx, dict):
                # band indices given for each spin channel
                spin_band_idx = band_idx[spin]
            else:
                spin_band_idx = band_idx

            emesh, dos[spin] = self.get_spin_density_of_states(
                spin, energies, weights=spin_weights, band_idx=spin_band_idx
            )

        if sum_spins:
            if Spin.down in dos:
                dos = dos[Spin.up] + dos[Spin.down]
            else:
                dos = dos[Spin.up]

        return energies, dos

    def get_spin_density_of_states(self, spin, energies, weights=None, band_idx=None):
        if weights is None:
            dos = np.zeros_like(energies)
            for i, energy in enumerate(energies):
                dos[i] = np.sum(
                    self.get_tetrahedra_density_of_states(
                        spin, energy, band_idx=band_idx
                    )
                )

        else:
            dos = np.zeros(weights.shape[2:] + energies.shape)

            for i, energy in enumerate(energies):
                tet_dos, tet_mask, _, tet_contributions = self.get_tetrahedra_density_of_states(
                    spin,
                    energy,
                    return_contributions=True,
                    symmetry_reduce=False,
                    band_idx=band_idx,
                )

                if len(tet_dos) == 0:
                    continue

                # get masks needed to find the weights inside the tetrahedra
                property_mask, _, _, _ = self.get_masks(spin, tet_mask)

                # get the weights of the tetrahedron vertices
                vert_weights = weights[property_mask]

                # now get the average weighting for each tetrahedron cross section
                tet_weights = get_cross_section_values(vert_weights, *tet_contributions)

                # finally, weight the dos and sum overall tetrahedra
                # TODO: Don't hard code in the None, None
                dos[..., i] = np.sum(tet_weights * tet_dos[:, None, None], axis=0)

        return energies, np.asarray(dos)

    def get_masks(self, spin, tetrahedra_mask):
        # mask needs to be generated with symmetry_reduce=False

        band_idxs = tetrahedra_mask[0]

        # property_mask can be used to get the values of a band and k-dependent
        # property at all tetrahedra vertices for each tetrahedron selected by the
        # mask for example: energies[property_mask]
        property_mask = (
            np.repeat(band_idxs[:, None], 4, axis=1),
            self.tetrahedra[spin][tetrahedra_mask],
        )

        # band_kpoint_mask can be used to get the inequivalent band, k-point
        # combinations that are required to get the properties for all tetrahedra
        # vertices. Can be used to avoid calculation of band and k-point dependent
        # properties at all bands k-points and k-points (instead you only calculate
        # the properties at the bands and k-points involved in the tetrahedra)
        # and therefore reduce computational expense
        band_kpoint_mask = np.full(self.energies[spin].shape, False)
        band_kpoint_mask[property_mask] = True

        # kpoint_mask is the k-point contribution to band_kpoint_mask which allows
        # mixing of band and k-point dependent properties and just k-dependent
        # properties
        band_mask, kpoint_mask = np.where(band_kpoint_mask)

        return property_mask, band_kpoint_mask, band_mask, kpoint_mask


def _get_density_of_states_a(ee1, e21, e31, e41):
    return 3 * ee1 ** 2 / (e21 * e31 * e41)


def _get_density_of_states_b(ee2, e21, e31, e41, e32, e42):
    return 3 * (e21 + 2 * ee2 - (e42 + e31) * ee2 ** 2 / (e32 * e42)) / (e31 * e41)


def _get_density_of_states_c(e4e, e41, e42, e43):
    return 3 * e4e ** 2 / (e41 * e42 * e43)


def get_cross_section_values(
    property_values,
    cond_a_mask,
    cond_b_mask,
    cond_c_mask,
    frac_21,
    frac_31,
    frac_41,
    frac_32,
    frac_42,
    frac_c_41,
    frac_c_42,
    frac_c_43,
    average=True,
):
    # property_values is given for each tetrahedra
    # property can be scalar or vector
    p21 = property_values[:, 1] - property_values[:, 0]
    p31 = property_values[:, 2] - property_values[:, 0]
    p41 = property_values[:, 3] - property_values[:, 0]
    p32 = property_values[:, 2] - property_values[:, 1]
    p42 = property_values[:, 3] - property_values[:, 1]
    p43 = property_values[:, 3] - property_values[:, 2]

    prop_shape = property_values.shape[2:]

    if average:
        # remove the 4 vertices from shape
        value_shape = property_values.shape[:1] + prop_shape
    else:
        # shape is (ntetrahedra, 4, prop_shape)
        # the 4 is to account for the intersections of the cross section and tetrahedra
        # the number of intersections can be either 3 or 4;
        # for cross sections with 3 interesections, the final intersection will be
        # an array of 0's with the shape of prop_shape
        value_shape = property_values.shape[:1] + (4,) + prop_shape

    values = np.zeros(value_shape)

    if len(prop_shape) != 0:
        # property is multi-dimensional
        new_shape = (len(values),) + tuple([1] * len(prop_shape))
        frac_21 = frac_21.reshape(new_shape)
        frac_31 = frac_31.reshape(new_shape)
        frac_41 = frac_41.reshape(new_shape)
        frac_32 = frac_32.reshape(new_shape)
        frac_42 = frac_42.reshape(new_shape)
        frac_c_41 = frac_c_41.reshape(new_shape)
        frac_c_42 = frac_c_42.reshape(new_shape)
        frac_c_43 = frac_c_43.reshape(new_shape)

    values[cond_a_mask] = _get_cross_section_values_a(
        property_values[cond_a_mask],
        p21[cond_a_mask],
        p31[cond_a_mask],
        p41[cond_a_mask],
        frac_21[cond_a_mask],
        frac_31[cond_a_mask],
        frac_41[cond_a_mask],
        average=average,
    )

    values[cond_b_mask] = _get_cross_section_values_b(
        property_values[cond_b_mask],
        p31[cond_b_mask],
        p41[cond_b_mask],
        p32[cond_b_mask],
        p42[cond_b_mask],
        frac_31[cond_b_mask],
        frac_41[cond_b_mask],
        frac_32[cond_b_mask],
        frac_42[cond_b_mask],
        average=average,
    )

    values[cond_c_mask] = _get_cross_section_values_c(
        property_values[cond_c_mask],
        p41[cond_c_mask],
        p42[cond_c_mask],
        p43[cond_c_mask],
        frac_c_41[cond_c_mask],
        frac_c_42[cond_c_mask],
        frac_c_43[cond_c_mask],
        average=average,
    )

    return values


def _get_cross_section_values_a(
    property_values, p21, p31, p41, frac_21, frac_31, frac_41, average=False
):
    # these are the property values at the intersection of the cross section section
    # with the tetrahedron edges
    interp_21 = p21 * frac_21 + property_values[:, 0]
    interp_31 = p31 * frac_31 + property_values[:, 0]
    interp_41 = p41 * frac_41 + property_values[:, 0]

    if average:
        return (interp_21 + interp_31 + interp_41) / 3

    else:
        return np.stack(
            [interp_21, interp_31, interp_41, np.zeros_like(interp_21)], axis=1
        )


def _get_cross_section_values_b(
    property_values,
    p31,
    p41,
    p32,
    p42,
    frac_31,
    frac_41,
    frac_32,
    frac_42,
    average=False,
):
    # these are the property values at the intersection of the cross section section
    # with the tetrahedron edges
    interp_31 = p31 * frac_31 + property_values[:, 0]
    interp_41 = p41 * frac_41 + property_values[:, 0]
    interp_32 = p32 * frac_32 + property_values[:, 1]
    interp_42 = p42 * frac_42 + property_values[:, 1]

    if average:
        return (interp_31 + interp_41 + interp_32 + interp_42) / 4

    else:
        # the order of these matters for calculating the cross sectional
        # area. the first and last points are at opposite ends of the quadrilateral
        return np.stack([interp_31, interp_32, interp_41, interp_42], axis=1)


def _get_cross_section_values_c(
    property_values, p41, p42, p43, frac_41, frac_42, frac_43, average=False
):
    # these are the property values at the intersection of the cross section
    # with the tetrahedron edges
    interp_41 = property_values[:, 3] - p41 * frac_41
    interp_42 = property_values[:, 3] - p42 * frac_42
    interp_43 = property_values[:, 3] - p43 * frac_43

    if average:
        return (interp_41 + interp_42 + interp_43) / 3

    else:
        return np.stack(
            [interp_41, interp_42, interp_43, np.zeros_like(interp_41)], axis=1
        )


def get_projected_intersections(intersections):
    # project the intersections coordinates onto a two dimensional plane

    # take two of the triangle/quadrilateral sides and calculate the unit vectors
    # the first of these vectors will be a basis vector
    axis_a = intersections[:, 1] - intersections[:, 0]
    axis_a /= np.linalg.norm(axis_a, axis=1)[:, None]

    other_side = intersections[:, 2] - intersections[:, 0]

    # find the vector normal to the plane of the triangle/quadrilateral, this will be a
    # basis vector, however all points will have the same magnitude in this direction
    axis_c = np.cross(axis_a, other_side)
    axis_c /= np.linalg.norm(axis_c, axis=1)[:, None]

    # find the final basis vector by finding the normal to the a and c basis vectors
    # no need to normalise as it is the cross product of two orthogonal unit vectors
    axis_b = np.cross(axis_a, axis_c)

    # define a transformation matrix to transform the points to our new basis
    transform = np.linalg.inv(np.stack([axis_a, axis_b, axis_c], axis=2))

    # finally transform the intersection coordinates. This syntax is equivalent to
    # np.dot(transform[0], intersections[0][0]) for all intersections in all triangles/
    # quadrilaterals.
    return np.einsum("ikj,ilj->ilk", transform, intersections)


def integrate_function_over_cross_section(
    function_generator,
    intersections,
    cond_a_mask,
    cond_b_mask,
    cond_c_mask,
    precision="medium",
    return_shape=None,
    cross_section_weights=None,
):
    triangle_scheme = numerical_integration_defaults[precision]["triangle"]
    quadrilateral_scheme = numerical_integration_defaults[precision]["quadrilateral"]

    if cross_section_weights is None:
        cross_section_weights = np.ones(len(intersections))

    if return_shape:
        function_values = np.zeros(return_shape + (len(intersections),))
    else:
        function_values = np.zeros(len(intersections))

    ninter = len(intersections)
    z_coords_sq = intersections[:, 2, 2] ** 2

    # intersections now has shape nvert, ntet, 2 (i.e., x, y coords)
    intersections = intersections[:, :, :2].transpose(1, 0, 2)
    triangle_mask = cond_a_mask | cond_c_mask

    if np.any(triangle_mask):
        function = function_generator(z_coords_sq[triangle_mask])
        function_values[..., triangle_mask] = triangle_scheme.integrate(
            function, intersections[:3, triangle_mask]
        )

    if np.any(cond_b_mask):
        function = function_generator(z_coords_sq[cond_b_mask])
        function_values[..., cond_b_mask] = quadrilateral_scheme.integrate(
            function, intersections.reshape((2, 2, ninter, 2))[:, :, cond_b_mask]
        )

    function_values *= cross_section_weights

    return function_values


def process_tetrahedra(tetrahedra, energies):
    all_tetrahedra = {}
    all_tetrahedra_energies = {}

    for spin, spin_energies in energies.items():
        data_shape = (len(spin_energies),) + tetrahedra.shape
        spin_tetrahedra = np.zeros(data_shape, dtype=int)
        spin_tetrahedra_energies = np.zeros(data_shape)

        for band_idx, band_energies in enumerate(spin_energies):
            band_tetrahedra_energies = band_energies[tetrahedra]

            sort_idx = np.argsort(band_tetrahedra_energies, axis=1)
            spin_tetrahedra_energies[band_idx, ...] = np.take_along_axis(
                band_tetrahedra_energies, sort_idx, axis=1
            )
            spin_tetrahedra[band_idx, ...] = np.take_along_axis(
                tetrahedra, sort_idx, axis=1
            )

        all_tetrahedra[spin] = spin_tetrahedra
        all_tetrahedra_energies[spin] = spin_tetrahedra_energies

    return all_tetrahedra, all_tetrahedra_energies


def get_tetrahedra_energy_diffs(tetrahedra_energies):
    e21 = {}
    e31 = {}
    e41 = {}
    e32 = {}
    e42 = {}
    e43 = {}

    for spin, s_tetrahedra_energies in tetrahedra_energies.items():
        # each energy difference has the shape nbands, ntetrahedra
        e21[spin] = s_tetrahedra_energies[:, :, 1] - s_tetrahedra_energies[:, :, 0]
        e31[spin] = s_tetrahedra_energies[:, :, 2] - s_tetrahedra_energies[:, :, 0]
        e41[spin] = s_tetrahedra_energies[:, :, 3] - s_tetrahedra_energies[:, :, 0]
        e32[spin] = s_tetrahedra_energies[:, :, 2] - s_tetrahedra_energies[:, :, 1]
        e42[spin] = s_tetrahedra_energies[:, :, 3] - s_tetrahedra_energies[:, :, 1]
        e43[spin] = s_tetrahedra_energies[:, :, 3] - s_tetrahedra_energies[:, :, 2]

    return e21, e31, e41, e32, e42, e43


def get_max_min_tetrahedra_energies(tetrahedra_energies):
    max_tetrahedra_energies = {}
    min_tetrahedra_energies = {}

    for spin, s_tetrahedra_energies in tetrahedra_energies.items():
        max_tetrahedra_energies[spin] = np.max(s_tetrahedra_energies, axis=2)
        min_tetrahedra_energies[spin] = np.min(s_tetrahedra_energies, axis=2)

    return max_tetrahedra_energies, min_tetrahedra_energies


def get_tetrahedra_cross_section_weights(
    reciprocal_lattice, kpoints, tetrahedra, e21, e31, e41
):
    # weight (b) defined by equation 3.4 in https://doi.org/10.1002/pssb.2220540211
    # this weight is not the Bloechl weights but a scaling needed to obtain the
    # DOS directly from the tetrahedron cross section
    cross_section_weights = {}

    # volume is 6 * the volume of one tetrahedron
    volume = np.linalg.det(reciprocal_lattice) / len(kpoints)
    for spin, s_tetrahedra in tetrahedra.items():
        tetrahedra_kpoints = kpoints[s_tetrahedra]

        k1 = pbc_diff(tetrahedra_kpoints[:, :, 1], tetrahedra_kpoints[:, :, 0])
        k2 = pbc_diff(tetrahedra_kpoints[:, :, 2], tetrahedra_kpoints[:, :, 0])
        k3 = pbc_diff(tetrahedra_kpoints[:, :, 3], tetrahedra_kpoints[:, :, 0])

        k1_cart = np.dot(k1, reciprocal_lattice)
        k2_cart = np.dot(k2, reciprocal_lattice)
        k3_cart = np.dot(k3, reciprocal_lattice)

        contragradient = np.stack(
            [
                np.cross(k2_cart, k3_cart) / volume,
                np.cross(k3_cart, k1_cart) / volume,
                np.cross(k1_cart, k2_cart) / volume,
            ],
            axis=2,
        )

        energies = np.stack([e21[spin], e31[spin], e41[spin]], axis=2)
        b_vector = np.sum(contragradient * energies[..., None], axis=2)

        cross_section_weights[spin] = 1 / np.linalg.norm(b_vector, axis=2)

    return cross_section_weights
