import numpy as np

from amset.utils.general import norm

"""
Functions relevant only to integration='k' method. If integration is set to 'e'
when Amset is instantiated, none of these methods are relevant.
"""

def grid_norm(grid):
    """
            *** a method used only by "k"-integration method.

    Args:
        grid:

    Returns:

    """
    return (grid[:,:,:,0]**2 + grid[:,:,:,1]**2 + grid[:,:,:,2]**2) ** 0.5


def generate_k_mesh_axes(important_pts, kgrid_tp='coarse', one_list=True):
    """
                *** a method used only by "k"-integration method.

    Args:
        important_pts:
        kgrid_tp:
        one_list:

    Returns:

    """
    points_1d = {dir: [] for dir in ['x', 'y', 'z']}
    for center in important_pts:
        for dim, dir in enumerate(['x', 'y', 'z']):
            points_1d[dir].append(center[dim])

            if not one_list:
                for step, nsteps in [[0.002, 2], [0.005, 4], [0.01, 4], [0.05, 2], [0.1, 5]]:
                    for i in range(nsteps - 1):
                        points_1d[dir].append(center[dim] - (i + 1) * step)
                        points_1d[dir].append(center[dim] + (i + 1) * step)
            else:
                if kgrid_tp == 'extremely fine':
                    mesh = [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.0045,
                            0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03,
                            0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.25]
                elif kgrid_tp == 'super fine':
                    mesh = [0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.007,
                            0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.25]
                elif kgrid_tp == 'very fine':
                    mesh = [0.001, 0.002, 0.004, 0.007, 0.01, 0.02, 0.03,
                            0.05, 0.07, 0.1, 0.15, 0.25]
                elif kgrid_tp == 'fine':
                    mesh = [0.001, 0.004, 0.01, 0.02, 0.03,
                            0.05, 0.11, 0.25]
                elif kgrid_tp == 'coarse':
                    mesh = [0.001, 0.005, 0.01, 0.02, 0.05, 0.15]
                    # mesh = [0.003, 0.01, 0.05, 0.15]
                elif kgrid_tp == 'very coarse':
                    mesh = [0.001, 0.01]
                else:
                    raise ValueError('Unsupported value for kgrid_type: {}'.format(kgrid_tp))
                for step in mesh:
                    points_1d[dir].append(center[dim] + step)
                    points_1d[dir].append(center[dim] - step)
    return points_1d


def create_grid(points_1d):
    """
                *** a method used only by "k"-integration method.

    Args:
        points_1d:

    Returns:

    """
    for dir in ['x', 'y', 'z']:
        points_1d[dir].sort()
    grid = np.zeros((len(points_1d['x']), len(points_1d['y']), len(points_1d['z']), 3))
    for i, x in enumerate(points_1d['x']):
        for j, y in enumerate(points_1d['y']):
            for k, z in enumerate(points_1d['z']):
                grid[i, j, k, :] = np.array([x, y, z])
    return grid


def array_to_kgrid(grid):
    """
                *** a method used only by "k"-integration method.

    Args:
        grid (np.array): 4d numpy array, where last dimension is vectors
            in a 3d grid specifying fractional position in BZ
    Returns:
        a list of [kx, ky, kz] k-point coordinates compatible with Amset
    """
    kgrid = []
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for k in range(grid.shape[2]):
                kgrid.append(grid[i,j,k])
    return kgrid


def normalize_array(grid):
    """
                *** a method used only by "k"-integration method.

    Args:
        grid:

    Returns:

    """
    N = grid.shape
    norm_grid = np.zeros(N)
    for i in range(N[0]):
        for j in range(N[1]):
            for k in range(N[2]):
                vec = grid[i, j, k]
                if norm(vec) == 0:
                    norm_grid[i, j, k] = [0, 0, 0]
                else:
                    norm_grid[i, j, k] = vec / norm(vec)
    return norm_grid
