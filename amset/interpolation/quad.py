import math

import numpy as np
from monty.serialization import loadfn
from pkg_resources import resource_filename

# Format of quad schemes.
# high:
#   triangle: xiao_gimbutas_50
#   quad: sommariva_55
# medium:
#   triangle: xiao_gimbutas_06
#   quad: sommariva_06
# low:
#   triangle: centroid
#   quad: dunavant_00
QUAD_SCHEMES = loadfn(resource_filename("amset.interpolation.quad", "quad.json"))


def get_triangle_vol(simplex):
    # compute all edge lengths
    edges = np.subtract(simplex[:, None], simplex[None, :])
    ei_dot_ej = np.einsum("...k,...k->...", edges, edges)

    j = simplex.shape[0] - 1
    a = np.empty((j + 2, j + 2) + ei_dot_ej.shape[2:])
    a[1:, 1:] = ei_dot_ej
    a[0, 1:] = 1.0
    a[1:, 0] = 1.0
    a[0, 0] = 0.0

    a = np.moveaxis(a, (0, 1), (-2, -1))
    det = np.linalg.det(a)

    vol = np.sqrt((-1.0) ** (j + 1) / 2**j / math.factorial(j) ** 2 * det)
    return vol


def transform_triangle(points, simplex):
    """Transform the points `xi` from the reference simplex onto `simplex`."""
    return np.dot(simplex, points)


def get_quad_vol(xi, cube):
    """Get the determinant of the transformation matrix."""
    # Like transform(), simplify here and form the determinant explicitly.
    d = xi.shape[0]

    one_mp_xi = np.stack([0.5 * (1.0 - xi), 0.5 * (1.0 + xi)], axis=1)

    # Build the Jacobi matrix row by row.
    J = []
    for k in range(d):
        a = one_mp_xi.copy()
        a[k, 0, :] = -0.5
        a[k, 1, :] = +0.5
        a0 = n_outer(a)
        J.append(np.tensordot(a0, cube, axes=(range(d), range(d))).T)

    J = np.array(J)
    J = np.moveaxis(J, (0, 1), (-2, -1))
    out = np.linalg.det(J)
    return abs(out) * 4


def transform_quad(xi, cube):
    """Transform the points `xi` from the reference cube to `cube`."""
    one_mp_xi = np.stack([0.5 * (1.0 - xi), 0.5 * (1.0 + xi)], axis=1)
    a = n_outer(one_mp_xi)
    d = xi.shape[0]
    return np.tensordot(a, cube, axes=(range(d), range(d)))


def n_outer(a):
    """Given a list (tuple, array) of arrays, this method computes their outer product."""
    # <https://stackoverflow.com/a/45376730/353337>
    d = len(a)

    # If the elements are more than one-dimensional, assert that the extra
    # dimensions are all equal.
    s0 = a[0].shape
    for arr in a:
        assert s0[1:] == arr.shape[1:]

    out = a[0]
    for k in range(1, d):
        # Basically outer products. Checkout `np.outer`'s implementation for
        # comparison.
        out = np.multiply(
            # Insert a newaxis after k `:`
            out[(slice(None),) * k + (np.newaxis,)],
            # Insert a newaxis at the beginning
            a[k][np.newaxis],
        )
    return out
