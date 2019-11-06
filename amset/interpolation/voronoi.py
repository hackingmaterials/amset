import itertools
import logging
import math
import sys
import time
from typing import Optional

import numpy as np
from multiprocessing import cpu_count

from joblib import Parallel, delayed
from scipy.spatial.qhull import Voronoi, ConvexHull
from sklearn.neighbors.ball_tree import BallTree
from tqdm import tqdm

from amset import amset_defaults
from amset.constants import output_width
from amset.misc.log import log_time_taken
from amset.misc.util import gen_even_slices, groupby
from pymatgen.core.lattice import Lattice
from pymatgen.util.coord import lattice_points_in_supercell

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"
__date__ = "June 21, 2019"

pdefaults = amset_defaults["performance"]
logger = logging.getLogger(__name__)


class PeriodicVoronoi(object):
    """

    Say no existing packages (scipy.qhull, tess, pyvoro) can handle 250,000+
    points in a reasonable amount of time with reasonable memory requirements.
    Note this class works because most of the points are on a regular grid so
    it is valid to calculate the Voronoi diagram in blocks.
    """

    def __init__(
        self,
        reciprocal_lattice: Lattice,
        original_points: np.ndarray,
        original_dim: np.ndarray,
        extra_points: np.ndarray,
        ir_to_full_idx: Optional[np.ndarray] = None,
        extra_ir_points_idx: Optional[np.ndarray] = None,
        nworkers: int = pdefaults["nworkers"],
    ):
        """

        Add a warning about only using the symmetry options if you are sure your
        extra k-points have been symmetrized

        Args:
            original_points:
            nworkers:
        """
        self._nworkers = nworkers if nworkers != -1 else cpu_count()
        self._final_points = np.concatenate([original_points, extra_points])
        self._reciprocal_lattice = reciprocal_lattice

        if ir_to_full_idx is None:
            ir_to_full_idx = np.arange(len(original_points) + len(extra_points))

        if extra_ir_points_idx is None:
            extra_ir_points_idx = np.arange(len(extra_points))

        logger.debug("Initializing periodic Voronoi calculator")
        all_points = np.concatenate((original_points, extra_points))

        logger.debug("  ├── getting supercell k-points")
        supercell_points = get_supercell_points(all_points)
        supercell_idxs = np.arange(supercell_points.shape[0])

        # filter points far from the zone boundary, this will lead to errors for
        # very small meshes < 5x5x5 but we are not interested in those
        mask = ((supercell_points > -0.75) & (supercell_points < 0.75)).all(axis=1)
        supercell_points = supercell_points[mask]
        supercell_idxs = supercell_idxs[mask]

        # want points in cartesian space so we can define a regular spherical
        # cutoff even if reciprocal lattice is not cubic. If we used a
        # fractional cutoff, the cutoff regions would not be spherical
        logger.debug("  ├── getting cartesian points")
        cart_points = reciprocal_lattice.get_cartesian_coords(supercell_points)
        cart_extra_points = reciprocal_lattice.get_cartesian_coords(
            extra_points[extra_ir_points_idx])

        # small cutoff is slightly larger than the max regular grid spacing
        # means at least 1 neighbour point will always be included in each
        # direction, need to find cartesian length which covers the longest direction
        # of the mesh
        spacing = 1 / original_dim
        body_diagonal = reciprocal_lattice.get_cartesian_coords(spacing)
        xy = reciprocal_lattice.get_cartesian_coords([spacing[0], spacing[1], 0])
        xz = reciprocal_lattice.get_cartesian_coords([spacing[0], 0, spacing[2]])
        yz = reciprocal_lattice.get_cartesian_coords([0, spacing[1], spacing[2]])

        len_diagonal = np.linalg.norm(body_diagonal)
        len_xy = np.linalg.norm(xy)
        len_xz = np.linalg.norm(xz)
        len_yz = np.linalg.norm(yz)

        small_cutoff = (np.max([len_diagonal, len_xy, len_xz, len_yz]) * 1.6)
        big_cutoff = (small_cutoff * 1.77)

        logger.debug("  ├── initializing ball tree")

        # use BallTree for quickly evaluating which points are within cutoffs
        tree = BallTree(cart_points)

        # big points are those which surround the extra points within the big cutoff
        # (including the extra points themselves)
        logger.debug("  ├── calculating points in big radius")
        big_points_idx = _query_radius_iteratively(tree, cart_extra_points, big_cutoff)

        # Voronoi points are those we actually include in the Voronoi diagram
        self._voronoi_points = cart_points[big_points_idx]

        # small points are the points in all_points (i.e., original + extra points) for
        # which we want to calculate the Voronoi volumes. Outside the small cutoff, the
        # weights will just be the regular grid weight.
        logger.debug("  └── calculating points in small radius")
        small_points_idx = _query_radius_iteratively(
            tree, cart_extra_points, small_cutoff)

        # get the irreducible small points
        small_points_in_all_points = supercell_idxs[small_points_idx] % len(all_points)
        mapping = ir_to_full_idx[small_points_in_all_points]
        unique_mappings, ir_idx = np.unique(mapping, return_index=True)
        small_points_idx = small_points_idx[ir_idx]

        # get a mapping to go from the ir small points to the full BZ.
        groups = groupby(np.arange(len(all_points)), ir_to_full_idx)
        grouped_ir = groups[unique_mappings]
        counts = [len(g) for g in grouped_ir]
        self._expand_ir = np.repeat(np.arange(len(ir_idx)), counts)

        # get the indices of the expanded ir_small_points in all_points
        self._volume_in_final_idx = np.concatenate(grouped_ir)

        # get the indices of ir_small_points_idx (i.e., the points for which we will
        # calculate the volume) in voronoi_points
        self._volume_points_idx = _get_loc(big_points_idx, small_points_idx)

        # Prepopulate the final volumes array. By default, each point has the
        # volume of the original mesh. Note: at this point, the extra points
        # will have zero volume. This will array will be updated by
        # compute_volumes
        self._volume = reciprocal_lattice.volume
        self._final_volumes = np.full(
            len(all_points), self._volume / len(original_points))
        self._final_volumes[len(original_points):] = 0
        self._final_volumes[self._volume_in_final_idx] = 0

        # from pymatgen import Structure
        # s = Structure(
        #     reciprocal_lattice.matrix * 10,
        #     ['H'] * len(self._volume_points_idx),
        #     reciprocal_lattice.get_fractional_coords(self._voronoi_points[self._volume_points_idx]) / 3 + 0.5,
        # )
        # s.to(filename="volume-points.cif")
        #
        # s = Structure(
        #     reciprocal_lattice.matrix * 10,
        #     ['H'] * len(self._voronoi_points),
        #     reciprocal_lattice.get_fractional_coords(self._voronoi_points) / 3 + 0.5,
        # )
        # s.to(filename="voronoi-points.cif")

    def compute_volumes(self):
        logger.info("Calculating k-point Voronoi diagram:")
        logger.debug(
            "  ├── num k-points near extra points: {}".format(len(self._voronoi_points))
        )
        t0 = time.perf_counter()

        # after some testing it seems like sorting the points before calculating
        # the Voronoi diagram can speed things up by > 1000x when there are many
        # points
        sorted_idx = np.argsort(self._voronoi_points, axis=0)[:, 1]

        # voro = Voronoi(self._voronoi_points[sorted_idx], qhull_options="Qbb Qc Qz")
        voro = Voronoi(self._voronoi_points[sorted_idx], qhull_options="Qbb Qc Qz")

        # need to unsort regions to get correct points
        inv_sorted_idx = np.argsort(sorted_idx)
        regions = voro.point_region[inv_sorted_idx][self._volume_points_idx]
        indices = np.array(voro.regions)[regions]
        vertices = [voro.vertices[i] for i in indices]

        log_time_taken(t0)

        volumes = self._final_volumes.copy()

        # divide volumes by reciprocal lattice volume to get the fractional volume
        volumes[self._volume_in_final_idx] = self._get_voronoi_volumes(
            indices, vertices
        )[self._expand_ir]

        zero_vols = volumes == 0
        if zero_vols.any():
            logger.warning("{} volumes are zero".format(np.sum(zero_vols)))

        inf_vols: np.ndarray = volumes == np.inf
        if inf_vols.any():
            logger.warning("{} volumes are infinite".format(inf_vols.sum()))

        sum_volumes = volumes.sum()
        vol_diff = (sum_volumes - self._volume) / self._volume

        if abs(vol_diff) > 1e-7:
            logger.warning(
                "Sum of weights does not equal 1 (diff = {:.3f} "
                "%)... renormalising weights".format(vol_diff * 100)
            )
            volumes /= sum_volumes

        return volumes

    def _get_voronoi_volumes(self, indices, vertices) -> np.ndarray:
        logger.info("Calculating k-point weights:")

        voronoi_info = tqdm(
            list(zip(indices, vertices)),
            total=len(indices),
            ncols=output_width,
            desc="    ├── progress",
            file=sys.stdout,
            bar_format="{l_bar}{bar}| {elapsed}<{remaining}{postfix}",
        )

        t0 = time.perf_counter()
        volumes = Parallel(n_jobs=self._nworkers, prefer="processes")(
            delayed(_get_volume)(idx, verts) for idx, verts in voronoi_info
        )
        log_time_taken(t0)
        return np.array(volumes)


def _get_volume(indices, vertices):
    if -1 in indices:
        # some regions can be open
        return np.inf
    else:
        return ConvexHull(vertices, qhull_options="Qt C-0").volume


def _get_loc(x, y):
    """
    Based on https://stackoverflow.com/questions/8251541

    Args:
        x:
        y:

    Returns:

    """
    # len(x) > len(y)
    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y)
    yindex = np.take(index, sorted_index, mode="clip")
    mask = x[yindex] == y
    return yindex[mask]


def get_supercell_points(points):
    vals = (-1, 0, 1)
    images = np.array(list(itertools.product(vals, vals, vals)))

    repeated_points = np.tile(points, (len(images), 1))
    repeated_images = np.repeat(images, len(points), axis=0)

    return repeated_images + repeated_points


def _query_radius_iteratively(tree: BallTree, points: np.ndarray, cutoff: float,
                              max_points_per_split=10000):
    # return np.concatenate(tree.query_radius(points, cutoff), axis=0)
    # iterative querying when we have lots of points to avoid memory issues
    unique_neighbor_idxs = np.array([], dtype=int)
    npoints = len(points)
    nsplits = math.ceil(npoints / max_points_per_split)

    i = 0
    for split in gen_even_slices(npoints, nsplits):
        i += 1
        idxs = np.concatenate(tree.query_radius(points[split], cutoff), axis=0)
        unique_neighbor_idxs = np.unique(np.concatenate((idxs, unique_neighbor_idxs)))

    return np.asarray(unique_neighbor_idxs)
