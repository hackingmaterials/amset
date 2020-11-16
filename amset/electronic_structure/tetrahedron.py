import logging
import time
from typing import Dict, List, Optional, Union

import numpy as np
from pymatgen import Spin, Structure
from pymatgen.util.coord import pbc_diff

from amset.constants import numeric_types
from amset.log import log_time_taken
from amset.util import (
    array_from_buffer,
    create_shared_array,
    create_shared_dict_array,
    dict_array_from_buffer,
    get_progress_bar,
    groupby,
)

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

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

logger = logging.getLogger(__name__)


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
        ir_kpoints_idx: np.ndarray,
        ir_kpoint_mapping: np.ndarray,
        ir_kpoint_weights: np.ndarray,
        tetrahedra: Dict[Spin, np.ndarray],
        ir_tetrahedra: Dict[Spin, np.ndarray],
        ir_tetrahedra_energies: Dict[Spin, np.ndarray],
        ir_tetrahedra_idx: np.ndarray,
        ir_tetrahedra_to_full_idx: np.ndarray,
        ir_tetrahedra_weights: np.ndarray,
        e21: Dict[Spin, np.ndarray],
        e31: Dict[Spin, np.ndarray],
        e41: Dict[Spin, np.ndarray],
        e32: Dict[Spin, np.ndarray],
        e42: Dict[Spin, np.ndarray],
        e43: Dict[Spin, np.ndarray],
        max_tetrahedra_energies: Dict[Spin, np.ndarray],
        min_tetrahedra_energies: Dict[Spin, np.ndarray],
        cross_section_weights: Dict[Spin, np.ndarray],
        tetrahedron_volume: float,
        weights_cache: Optional[Dict[Spin, np.ndarray]] = None,
        weights_mask_cache: Optional[Dict[Spin, np.ndarray]] = None,
        energies_cache: Optional[Dict[Spin, np.ndarray]] = None,
    ):
        self.energies = energies
        self.kpoints = kpoints
        self.ir_kpoints_idx = ir_kpoints_idx
        self.ir_kpoint_mapping = ir_kpoint_mapping
        self.ir_kpoint_weights = ir_kpoint_weights
        self.tetrahedra = tetrahedra
        self.ir_tetrahedra = ir_tetrahedra
        self.ir_tetrahedra_energies = ir_tetrahedra_energies
        self.ir_tetrahedra_idx = ir_tetrahedra_idx
        self.ir_tetrahedra_to_full_idx = ir_tetrahedra_to_full_idx
        self.ir_tetrahedra_weights = ir_tetrahedra_weights
        self.e21 = e21
        self.e31 = e31
        self.e41 = e41
        self.e32 = e32
        self.e42 = e42
        self.e43 = e43
        self.max_tetrahedra_energies = max_tetrahedra_energies
        self.min_tetrahedra_energies = min_tetrahedra_energies
        self.cross_section_weights = cross_section_weights
        self._tetrahedron_volume = tetrahedron_volume
        self._weights_cache = {} if weights_cache is None else weights_cache
        self._weights_mask_cache = (
            {} if weights_mask_cache is None else weights_mask_cache
        )
        self._energies_cache = {} if energies_cache is None else energies_cache

        self.grouped_ir_to_full = groupby(
            np.arange(len(ir_tetrahedra_to_full_idx)), ir_tetrahedra_to_full_idx
        )
        self._ir_weights_shape = {
            s: (len(energies[s]), len(ir_kpoints_idx)) for s in energies
        }

    def to_reference(self):
        energies_buffer, self.energies = create_shared_dict_array(
            self.energies, return_shared_data=True
        )
        kpoints_buffer, self.kpoints = create_shared_array(
            self.kpoints, return_shared_data=True
        )
        ir_kpoints_idx_buffer, self.ir_kpoints_idx = create_shared_array(
            self.ir_kpoints_idx, return_shared_data=True
        )
        ir_kpoint_mapping_buffer, self.ir_kpoint_mapping = create_shared_array(
            self.ir_kpoint_mapping, return_shared_data=True
        )
        ir_kpoint_weights_buffer, self.ir_kpoint_weights = create_shared_array(
            self.ir_kpoint_weights, return_shared_data=True
        )
        tetrahedra_buffer, self.tetrahedra = create_shared_dict_array(
            self.tetrahedra, return_shared_data=True
        )
        ir_tetrahedra_buffer, self.ir_tetrahedra = create_shared_dict_array(
            self.ir_tetrahedra, return_shared_data=True
        )
        (
            ir_tetrahedra_energies_buffer,
            self.ir_tetrahedra_energies,
        ) = create_shared_dict_array(
            self.ir_tetrahedra_energies, return_shared_data=True
        )
        ir_tetrahedra_idx_buffer, self.ir_tetrahedra_idx = create_shared_array(
            self.ir_tetrahedra_idx, return_shared_data=True
        )
        (
            ir_tetrahedra_to_full_idx_buffer,
            self.ir_tetrahedra_to_full_idx,
        ) = create_shared_array(self.ir_tetrahedra_to_full_idx, return_shared_data=True)
        ir_tetrahedra_weights_buffer, self.ir_tetrahedra_weights = create_shared_array(
            self.ir_tetrahedra_weights, return_shared_data=True
        )
        e21_buffer, self.e21 = create_shared_dict_array(
            self.e21, return_shared_data=True
        )
        e31_buffer, self.e31 = create_shared_dict_array(
            self.e31, return_shared_data=True
        )
        e41_buffer, self.e41 = create_shared_dict_array(
            self.e41, return_shared_data=True
        )
        e32_buffer, self.e32 = create_shared_dict_array(
            self.e32, return_shared_data=True
        )
        e42_buffer, self.e42 = create_shared_dict_array(
            self.e42, return_shared_data=True
        )
        e43_buffer, self.e43 = create_shared_dict_array(
            self.e43, return_shared_data=True
        )
        (
            max_tetrahedra_energies_buffer,
            self.max_tetrahedra_energies,
        ) = create_shared_dict_array(
            self.max_tetrahedra_energies, return_shared_data=True
        )
        (
            min_tetrahedra_energies_buffer,
            self.min_tetrahedra_energies,
        ) = create_shared_dict_array(
            self.min_tetrahedra_energies, return_shared_data=True
        )
        (
            cross_section_weights_buffer,
            self.cross_section_weights,
        ) = create_shared_dict_array(
            self.cross_section_weights, return_shared_data=True
        )
        weights_cache_buffer, self._weights_cache = create_shared_dict_array(
            self._weights_cache, return_shared_data=True
        )
        weights_mask_cache_buffer, self._weights_mask_cache = create_shared_dict_array(
            self._weights_mask_cache, return_shared_data=True
        )
        energies_cache_buffer, self._energies_cache = create_shared_dict_array(
            self._energies_cache, return_shared_data=True
        )

        return (
            energies_buffer,
            kpoints_buffer,
            ir_kpoints_idx_buffer,
            ir_kpoint_mapping_buffer,
            ir_kpoint_weights_buffer,
            tetrahedra_buffer,
            ir_tetrahedra_buffer,
            ir_tetrahedra_energies_buffer,
            ir_tetrahedra_idx_buffer,
            ir_tetrahedra_to_full_idx_buffer,
            ir_tetrahedra_weights_buffer,
            e21_buffer,
            e31_buffer,
            e41_buffer,
            e32_buffer,
            e42_buffer,
            e43_buffer,
            max_tetrahedra_energies_buffer,
            min_tetrahedra_energies_buffer,
            cross_section_weights_buffer,
            self._tetrahedron_volume,
            weights_cache_buffer,
            weights_mask_cache_buffer,
            energies_cache_buffer,
        )

    @classmethod
    def from_reference(
        cls,
        energies_buffer,
        kpoints_buffer,
        ir_kpoints_idx_buffer,
        ir_kpoint_mapping_buffer,
        ir_kpoint_weights_buffer,
        tetrahedra_buffer,
        ir_tetrahedra_buffer,
        ir_tetrahedra_energies_buffer,
        ir_tetrahedra_idx_buffer,
        ir_tetrahedra_to_full_idx_buffer,
        ir_tetrahedra_weights_buffer,
        e21_buffer,
        e31_buffer,
        e41_buffer,
        e32_buffer,
        e42_buffer,
        e43_buffer,
        max_tetrahedra_energies_buffer,
        min_tetrahedra_energies_buffer,
        cross_section_weights_buffer,
        tetrahedron_volume,
        weights_cache_buffer,
        weights_mask_cache_buffer,
        energies_cache_buffer,
    ):
        return cls(
            dict_array_from_buffer(energies_buffer),
            array_from_buffer(kpoints_buffer),
            array_from_buffer(ir_kpoints_idx_buffer),
            array_from_buffer(ir_kpoint_mapping_buffer),
            array_from_buffer(ir_kpoint_weights_buffer),
            dict_array_from_buffer(tetrahedra_buffer),
            dict_array_from_buffer(ir_tetrahedra_buffer),
            dict_array_from_buffer(ir_tetrahedra_energies_buffer),
            array_from_buffer(ir_tetrahedra_idx_buffer),
            array_from_buffer(ir_tetrahedra_to_full_idx_buffer),
            array_from_buffer(ir_tetrahedra_weights_buffer),
            dict_array_from_buffer(e21_buffer),
            dict_array_from_buffer(e31_buffer),
            dict_array_from_buffer(e41_buffer),
            dict_array_from_buffer(e32_buffer),
            dict_array_from_buffer(e42_buffer),
            dict_array_from_buffer(e43_buffer),
            dict_array_from_buffer(max_tetrahedra_energies_buffer),
            dict_array_from_buffer(min_tetrahedra_energies_buffer),
            dict_array_from_buffer(cross_section_weights_buffer),
            tetrahedron_volume,
            dict_array_from_buffer(weights_cache_buffer),
            dict_array_from_buffer(weights_mask_cache_buffer),
            dict_array_from_buffer(energies_cache_buffer),
        )

    @classmethod
    def from_data(
        cls,
        energies: Dict[Spin, np.ndarray],
        kpoints: np.ndarray,
        tetrahedra: np.ndarray,
        structure: Structure,
        ir_kpoints_idx: np.ndarray,
        ir_kpoint_mapping: np.ndarray,
        ir_tetrahedra_idx: Optional[np.ndarray] = None,
        ir_tetrahedra_to_full_idx: Optional[np.ndarray] = None,
        ir_tetrahedra_weights: Optional[np.ndarray] = None,
    ):
        logger.info("Initializing tetrahedron band structure")
        t0 = time.perf_counter()

        tparams = (ir_tetrahedra_idx, ir_tetrahedra_to_full_idx, ir_tetrahedra_weights)
        if len(set([x is None for x in tparams])) != 1:
            raise ValueError(
                "Either all or none of ir_tetrahedra_idx, ir_tetrahedra_to_full_idx and"
                " ir_tetrahedra_weights should be set."
            )

        if ir_tetrahedra_idx is None:
            ir_tetrahedra_idx = np.arange(len(kpoints))
            ir_tetrahedra_to_full_idx = np.ones_like(ir_tetrahedra_idx)
            ir_tetrahedra_weights = np.ones_like(ir_tetrahedra_idx)

        ir_tetrahedra_to_full_idx = ir_tetrahedra_to_full_idx
        ir_kpoints_idx = ir_kpoints_idx
        ir_kpoint_mapping = ir_kpoint_mapping

        _, ir_kpoint_weights = np.unique(ir_kpoint_mapping, return_counts=True)

        # need to keep track of full tetrahedra to recover full k-point indices
        # when calculating scattering rates (i.e., k-k' is symmetry inequivalent).
        full_tetrahedra, _ = process_tetrahedra(tetrahedra, energies)

        # store irreducible tetrahedra and use energies to calculate diffs and min/maxes
        ir_tetrahedra, ir_tetrahedra_energies = process_tetrahedra(
            tetrahedra[ir_tetrahedra_idx], energies
        )

        # the remaining properties are given for each irreducible tetrahedra
        (e21, e31, e41, e32, e42, e43) = get_tetrahedra_energy_diffs(
            ir_tetrahedra_energies
        )

        (
            max_tetrahedra_energies,
            min_tetrahedra_energies,
        ) = get_max_min_tetrahedra_energies(ir_tetrahedra_energies)

        cross_section_weights = get_tetrahedra_cross_section_weights(
            structure.lattice.reciprocal_lattice.matrix,
            kpoints,
            ir_tetrahedra,
            e21,
            e31,
            e41,
        )

        tetrahedron_volume = 1 / len(tetrahedra)

        log_time_taken(t0)
        return cls(
            energies,
            kpoints,
            ir_kpoints_idx,
            ir_kpoint_mapping,
            ir_kpoint_weights,
            full_tetrahedra,
            ir_tetrahedra,
            ir_tetrahedra_energies,
            ir_tetrahedra_idx,
            ir_tetrahedra_to_full_idx,
            ir_tetrahedra_weights,
            e21,
            e31,
            e41,
            e32,
            e42,
            e43,
            max_tetrahedra_energies,
            min_tetrahedra_energies,
            cross_section_weights,
            tetrahedron_volume,
        )

    def get_connected_kpoints(self, kpoint_idx: Union[int, List[int], np.ndarray]):
        """Given one or more k-point indices, get a list of all k-points that are in
        the same tetrahedra

        Args:
            kpoint_idx: One or mode k-point indices.

        Returns:
            A list of k-point indices that are in the same tetrahedra.
        """
        if isinstance(kpoint_idx, numeric_types):
            kpoint_idx = [kpoint_idx]

        tetrahedra = self.tetrahedra[Spin.up][0]
        return np.unique(tetrahedra[np.isin(tetrahedra, kpoint_idx).any(axis=1)])

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
        tetrahedra_weights = self.ir_tetrahedra_weights[tetrahedra_idx]

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
        self,
        energies=None,
        integrand=None,
        sum_spins=False,
        band_idx=None,
        use_cached_weights=False,
        progress_bar=False,
    ):
        if energies is None:
            from amset.constants import defaults, ev_to_hartree

            min_e = np.min([np.min(e) for e in self.energies.values()])
            max_e = np.max([np.max(e) for e in self.energies.values()])
            energies = np.arange(min_e, max_e, defaults["dos_estep"] * ev_to_hartree)

        dos = {}
        for spin in self.energies.keys():
            if isinstance(integrand, dict):
                # integrand given for each spin channel
                spin_integrand = integrand[spin]
            else:
                spin_integrand = integrand

            if isinstance(band_idx, dict):
                # band indices given for each spin channel
                spin_band_idx = band_idx[spin]
            else:
                spin_band_idx = band_idx

            if spin_integrand is not None:
                if spin_integrand.shape[:2] != self.energies[spin].shape:
                    raise ValueError(
                        "Unexpected integrand shape, should be (nbands, nkpoints, ...)"
                    )

                nbands = len(spin_integrand)
                integrand_shape = spin_integrand.shape[2:]
                n_ir_kpoints = len(self.ir_kpoints_idx)
                new_integrand = np.zeros((nbands, n_ir_kpoints) + integrand_shape)

                flat_k = np.tile(self.ir_kpoint_mapping, nbands)
                flat_b = np.repeat(np.arange(nbands), len(self.ir_kpoint_mapping))
                flat_integrand = spin_integrand.reshape((-1,) + integrand_shape)
                # flat_integrand = spin_integrand.reshape(-1, 3, 3)

                # sum integrand at all symmetry equivalent points, new_integrand
                # has shape (nbands, n_ir_kpoints)
                np.add.at(new_integrand, (flat_b, flat_k), flat_integrand)
                spin_integrand = new_integrand

            emesh, dos[spin] = self.get_spin_density_of_states(
                spin,
                energies,
                integrand=spin_integrand,
                band_idx=spin_band_idx,
                use_cached_weights=use_cached_weights,
                progress_bar=progress_bar,
            )

        if sum_spins:
            if Spin.down in dos:
                dos = dos[Spin.up] + dos[Spin.down]
            else:
                dos = dos[Spin.up]

        return energies, dos

    def get_spin_density_of_states(
        self,
        spin,
        energies,
        integrand=None,
        band_idx=None,
        use_cached_weights=False,
        progress_bar=False,
    ):
        # integrand should have the shape (nbands, n_ir_kpts, ...)
        # the integrand should have been summed at all equivalent k-points
        # TODO: add support for variable shaped integrands
        if integrand is None:
            dos = np.zeros_like(energies)
        else:
            integrand_shape = integrand.shape[2:]
            dos = np.zeros((len(energies),) + integrand_shape)

        if use_cached_weights:
            if self._weights_cache is None:
                raise ValueError("No integrand have been cached")

            all_weights = self._weights_cache[spin]
            all_weights_mask = self._weights_mask_cache[spin]
            energies = self._energies_cache[spin]
        else:
            all_weights = []
            all_weights_mask = []

        nbands = len(self.energies[spin])
        kpoint_multiplicity = np.tile(self.ir_kpoint_weights, (nbands, 1))

        if band_idx is not None and integrand is not None:
            integrand = integrand[band_idx]

        if band_idx is not None and integrand is None:
            kpoint_multiplicity = kpoint_multiplicity[band_idx]

        energies_iter = list(enumerate(energies))
        if progress_bar:
            energies_iter = get_progress_bar(iterable=energies_iter, desc="DOS")

        for i, energy in energies_iter:
            if use_cached_weights:
                weights = all_weights[i]
                weights_mask = all_weights_mask[i]
            else:
                weights = self.get_energy_dependent_integration_weights(spin, energy)
                weights_mask = weights != 0
                all_weights.append(weights)
                all_weights_mask.append(weights_mask)

            if band_idx is not None:
                weights = weights[band_idx]
                weights_mask = weights_mask[band_idx]

            if integrand is None:
                dos[i] = np.sum(
                    weights[weights_mask] * kpoint_multiplicity[weights_mask]
                )

            else:
                # expand weights to match the dimensions of the integrand
                expand_axis = [1 + i for i in range(len(integrand.shape[2:]))]
                expand_weights = np.expand_dims(weights[weights_mask], axis=expand_axis)

                # don't need to include the k-point multiplicity as this is included by
                # pre-summing the integrand at symmetry equivalent points
                dos[i] = np.sum(expand_weights * integrand[weights_mask], axis=0)

        if not use_cached_weights:
            self._weights_cache[spin] = np.array(all_weights)
            self._weights_mask_cache[spin] = np.array(all_weights_mask)
            self._energies_cache[spin] = energies

        return energies, np.asarray(dos)

    def get_energy_dependent_integration_weights(self, spin, energy):
        integration_weights = np.zeros(self._ir_weights_shape[spin])
        tetrahedra_mask = self.get_intersecting_tetrahedra(spin, energy)

        if not np.any(tetrahedra_mask):
            return integration_weights

        energies = self.ir_tetrahedra_energies[spin][tetrahedra_mask]
        e21 = self.e21[spin][tetrahedra_mask]
        e31 = self.e31[spin][tetrahedra_mask]
        e41 = self.e41[spin][tetrahedra_mask]
        e32 = self.e32[spin][tetrahedra_mask]
        e42 = self.e42[spin][tetrahedra_mask]
        e43 = self.e43[spin][tetrahedra_mask]

        cond_a_mask = (energies[:, 0] < energy) & (energy < energies[:, 1])
        cond_b_mask = (energies[:, 1] <= energy) & (energy < energies[:, 2])
        cond_c_mask = (energies[:, 2] <= energy) & (energy < energies[:, 3])

        ee1 = energy - energies[:, 0]
        ee2 = energy - energies[:, 1]
        ee3 = energy - energies[:, 2]
        e2e = energies[:, 1] - energy
        e3e = energies[:, 2] - energy
        e4e = energies[:, 3] - energy

        kpoints_idx = self.ir_tetrahedra[spin][tetrahedra_mask]
        ir_kpoints_idx = self.ir_kpoint_mapping[kpoints_idx]

        # calculate the integrand for each vertices
        vert_weights = np.zeros_like(energies)
        vert_weights[cond_a_mask] = _get_energy_dependent_weight_a(
            ee1[cond_a_mask],
            e2e[cond_a_mask],
            e3e[cond_a_mask],
            e4e[cond_a_mask],
            e21[cond_a_mask],
            e31[cond_a_mask],
            e41[cond_a_mask],
        )

        vert_weights[cond_b_mask] = _get_energy_dependent_weight_b(
            ee1[cond_b_mask],
            ee2[cond_b_mask],
            e3e[cond_b_mask],
            e4e[cond_b_mask],
            e31[cond_b_mask],
            e41[cond_b_mask],
            e32[cond_b_mask],
            e42[cond_b_mask],
        )

        vert_weights[cond_c_mask] = _get_energy_dependent_weight_c(
            ee1[cond_c_mask],
            ee2[cond_c_mask],
            ee3[cond_c_mask],
            e4e[cond_c_mask],
            e41[cond_c_mask],
            e42[cond_c_mask],
            e43[cond_c_mask],
        )

        # finally, get the integrand for each ir_kpoint by summing over all
        # tetrahedra and multiplying by the tetrahedra multiplicity and
        # tetrahedra weight; Finally, divide by the k-point multiplicity
        # to get the final weight
        band_idx, tetrahedra_idx = np.where(tetrahedra_mask)

        # include tetrahedra multiplicity
        vert_weights *= self.ir_tetrahedra_weights[tetrahedra_idx][:, None]

        flat_ir_kpoints = np.ravel(ir_kpoints_idx)
        flat_ir_weights = np.ravel(vert_weights)
        flat_bands = np.repeat(band_idx, 4)

        # sum integrand, note this sums in place and is insanely fast
        np.add.at(integration_weights, (flat_bands, flat_ir_kpoints), flat_ir_weights)
        integration_weights *= (
            self._tetrahedron_volume / self.ir_kpoint_weights[None, :]
        )

        return integration_weights

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


def _get_energy_dependent_weight_a(ee1, e2e, e3e, e4e, e21, e31, e41):
    c = ee1 ** 2 / (e21 * e31 * e41)
    i1 = c * (e2e / e21 + e3e / e31 + e4e / e41)
    i2 = c * (ee1 / e21)
    i3 = c * (ee1 / e31)
    i4 = c * (ee1 / e41)
    return np.stack([i1, i2, i3, i4], axis=1)


def _get_energy_dependent_weight_b(ee1, ee2, e3e, e4e, e31, e41, e32, e42):
    c = (ee1 * e4e) / (e31 * e41 * e42)
    x = e3e / e31
    y = e4e / e42
    z = ee2 / (e32 * e42)
    zx = z * x
    k = ee1 / e31
    n = ee2 / e42

    i1 = c * (x + e4e / e41) + z * x ** 2
    i2 = c * y + zx * (e3e / e32 + y)
    i3 = c * k + zx * (k + ee2 / e32)
    i4 = c * (ee1 / e41 + n) + zx * n
    return np.stack([i1, i2, i3, i4], axis=1)


def _get_energy_dependent_weight_c(ee1, ee2, ee3, e4e, e41, e42, e43):
    c = e4e ** 2 / (e41 * e42 * e43)
    i1 = c * e4e / e41
    i2 = c * e4e / e42
    i3 = c * e4e / e43
    i4 = c * (ee1 / e41 + ee2 / e42 + ee3 / e43)
    return np.stack([i1, i2, i3, i4], axis=1)


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
    basis = np.stack([axis_a, axis_b, axis_c], axis=2)
    transform = np.linalg.inv(basis)

    # finally transform the intersection coordinates. This syntax is equivalent to
    # np.dot(transform[0], intersections[0][0]) for all intersections in all triangles/
    # quadrilaterals.
    return np.einsum("ikj,ilj->ilk", transform, intersections), basis


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
    # this weight is not the Bloechl integrand but a scaling needed to obtain the
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
