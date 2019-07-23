import os
import unittest
import numpy as np


from amset.interpolation.voronoi import PeriodicVoronoi
from amset.misc.util import get_dense_kpoint_mesh_spglib
from pymatgen import Lattice

test_dir = os.path.dirname(os.path.abspath(__file__))
amset_files = os.path.join(test_dir, '..', '..', '..', 'test_files',
                           'AlCuS2_mp-4979')


class TestPeriodicVoronoi(unittest.TestCase):
    """Tests for interpolating a band structure using BoltzTraP2."""

    def test_get_volumes(self):
        # mesh = np.array([100, 100, 100])
        # kpts = get_dense_kpoint_mesh_spglib(mesh, spg_order=True, shift=0.)
        # extra_points = np.array([[0.001, 0, 0], [0.002, 0, 0], [0.003, 0, 0]])
        # pv = PeriodicVoronoi(Lattice([3, 0, 0, 0, 3, 0, 0, 0, 3]),
        #                      kpts, mesh, extra_points)
        # vols = pv.compute_volumes()
        # idx = np.argsort(vols)
        # all_k = np.concatenate([kpts, extra_points])
        # print(all_k[idx][:12])
        # print(vols[idx][:12])

        mesh = np.array([5, 5, 5])

        def get_kpoints(m, shift=0.):
            k = get_dense_kpoint_mesh_spglib(m, spg_order=True, shift=shift)
            sort_idx = np.lexsort((k[:, 2], k[:, 1], k[:, 0]))
            grid = k[sort_idx].reshape(tuple(m) + (3,))
            grid = grid[:, :, 0].reshape((m[0] * m[1], 3))
            return grid[:, :2]

        kpts = get_kpoints(mesh)

        dmesh = np.array([5, 5, 5])
        extra = get_kpoints(dmesh, shift=-0.)
        extra /= mesh[:2]
        extra += [0, 0]

        dmesh = np.array([5, 5, 5])
        lextra = get_kpoints(dmesh, shift=-0.)
        lextra /= mesh[:2]
        lextra += [0.2, 0.2]

        dmesh = np.array([3, 3, 3])
        jextra = get_kpoints(dmesh, shift=-0.)
        jextra /= mesh[:2]
        jextra += [0, 0.2]

        dmesh = np.array([7, 7, 7])
        pextra = get_kpoints(dmesh, shift=-0.)
        pextra /= mesh[:2]
        pextra += [-0.20, 0.2]

        kpts = np.concatenate([kpts, extra, lextra, jextra, pextra])

        from scipy.spatial import Voronoi, voronoi_plot_2d
        vor = Voronoi(kpts)
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        voronoi_plot_2d(vor)

        plt.xlim((-0.5, 0.5))
        plt.ylim((-0.5, 0.5))
        plt.show()

        # mesh = np.array([
        #     [0, 0, 0][]
        #
        # ])
        # kpts = np.array([
        #     [0, 0],
        #     [1/3, 0], [1/3, -1/3], [1/3, 1/3],
        #     [-1/3, 0], [-1/3, -1/3], [-1/3, 1/3],
        #     [0, 1/3], [0, -1/3]
        # ])
        # print(kpts)
        # extra_points = np.array([
        #     [1/18, 0], [2/18, 0], [3/18, 0],
        #     [-1/18, 0], [-2/18, 0], [-3/18, 0],
        #     [0, 1/18], [0, 2/18], [0, 3/18],
        #     [0, -1/18], [0, -2/18], [0, -3/18]
        # ])
        # all_k = np.concatenate([kpts, extra_points])
        #
        # from scipy.spatial import Voronoi, voronoi_plot_2d
        # vor = Voronoi(all_k)
        # import matplotlib
        # matplotlib.use("TkAgg")
        # import matplotlib.pyplot as plt
        # voronoi_plot_2d(vor)
        #
        # plt.xlim((-0.5, 0.5))
        # plt.ylim((-0.5, 0.5))
        # plt.show()


