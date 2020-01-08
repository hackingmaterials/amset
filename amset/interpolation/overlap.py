import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist

from amset.constants import numeric_types
from amset.kpoints import expand_kpoints, get_mesh_dim_from_kpoints


class OverlapCalculator(object):
    def __init__(self, structure, kpoints, projections, distance_function="cosine"):
        # distance function can be anything supported by the cdist metric argument,
        # including callable functions
        self.distance_function = distance_function

        # k-points have to be on a regular grid, even if only the irreducible part of
        # the grid is used
        full_kpoints, ir_to_full_idx = expand_kpoints(structure, kpoints)
        mesh_dim = get_mesh_dim_from_kpoints(full_kpoints)

        round_dp = int(np.log10(1 / 1e-6))
        full_kpoints = np.round(full_kpoints, round_dp)

        # get the indices to sort the k-points on the Z, then Y, then X columns
        sort_idx = np.lexsort(
            (full_kpoints[:, 2], full_kpoints[:, 1], full_kpoints[:, 0])
        )

        # put the kpoints into a 3D grid so that they can be indexed as
        # kpoints[ikx][iky][ikz] = [kx, ky, kz]
        grid_kpoints = full_kpoints[sort_idx].reshape(mesh_dim + (3,))

        x = grid_kpoints[:, 0, 0, 0]
        y = grid_kpoints[0, :, 0, 1]
        z = grid_kpoints[0, 0, :, 2]

        # TODO: Expand the k-point mesh to account for periodic boundary conditions
        self.interpolators = {}
        for spin, spin_projections in projections.items():
            nbands, nkpoints = spin_projections.shape[:2]
            nprojections = np.product(spin_projections.shape[2:])

            flat_projections = spin_projections.reshape((nbands, nkpoints, -1))
            expand_projections = flat_projections[:, ir_to_full_idx]

            # sort the projections then reshape them into the grid. The projections
            # can now be indexed as projections[iband][ikx][iky][ikz]
            sorted_projections = expand_projections[:, sort_idx]
            grid_shape = (nbands,) + mesh_dim + (nprojections,)
            grid_projections = sorted_projections.reshape(grid_shape)

            self.interpolators[spin] = RegularGridInterpolator(
                (np.arange(nbands), x, y, z), grid_projections, bounds_error=False
            )

    def get_overlap(self, spin, band_a, kpoint_a, band_b, kpoint_b):
        # k-points should be in fractional
        kpoint_a = np.asarray(kpoint_a)
        kpoint_b = np.asarray(kpoint_b)

        v1 = np.array([[band_a] + kpoint_a.tolist()])

        single_overlap = False
        if isinstance(band_b, numeric_types):
            # only one band index given

            if len(kpoint_b.shape) > 1:
                # multiple k-point indices given
                band_b = np.array([band_b] * len(kpoint_b))

            else:
                band_b = np.array([band_b])
                kpoint_b = [kpoint_b]
                single_overlap = True

        else:
            band_b = np.asarray(band_b)

        # v2 now has shape of (nkpoints_b, 4)
        v2 = np.concatenate([band_b[:, None], kpoint_b], axis=1)

        # get a big array of all the k-points to interpolate
        all_v = np.vstack([v1, v2])

        # get the interpolate projections for the k-points; p1 is the projections for
        # kpoint_a, p2 is a list of projections for the kpoint_bs
        p1, *p2 = self.interpolators[spin](all_v)

        # finally, get the overlap using the specified distance function
        overlap = 1 - cdist([p1], p2, metric=self.distance_function)[0]

        if single_overlap:
            return overlap[0]
        else:
            return overlap
