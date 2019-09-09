import os
import unittest
import numpy as np

from amset.interpolation.densify import BandDensifier
from amset.interpolation.interpolate import Interpolater
from amset.interpolation.voronoi import PeriodicVoronoi
from amset.misc.log import initialize_amset_logger
from amset.misc.util import get_dense_kpoint_mesh_spglib
from pymatgen import Lattice
from pymatgen.io.vasp import Vasprun

test_dir = os.path.dirname(os.path.abspath(__file__))
amset_files = os.path.join(test_dir, '..', '..', '..', 'examples',
                           'Si')


class TestPeriodicVoronoi(unittest.TestCase):
    """Tests for interpolating a band structure using BoltzTraP2."""

    def test_densify(self):
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

        initialize_amset_logger(log_error_traceback=True)

        vr = Vasprun(os.path.join(amset_files, "vasprun.xml.gz"),
                     parse_projected_eigen=True)
        bs = vr.get_band_structure()
        inter = Interpolater(bs, vr.parameters["NELECT"],
                             interpolate_projections=True,
                             interpolation_factor=5)
        amset_data = inter.get_amset_data(energy_cutoff=2, bandgap=1.33)
        amset_data.calculate_dos()
        amset_data.set_doping_and_temperatures(doping=np.array([1e13]),
                                               temperatures=np.array([300]))
        amset_data.calculate_fd_cutoffs(fd_tolerance=0.000001)

        densifier = BandDensifier(inter, amset_data, energy_cutoff=2,
                                  bandgap=1.33)
        # print(amset_data.ir_to_full_kpoint_mapping[:200])
        # print(max(amset_data.ir_to_full_kpoint_mapping))
        # print(len(amset_data.ir_to_full_kpoint_mapping))
        print("IR Kpoints idx max", max(amset_data.ir_kpoints_idx))
        amset_data.set_extra_kpoints(*densifier.densify(0.008))
        print(amset_data.ir_to_full_kpoint_mapping.max())
        print(len(amset_data.ir_kpoints))
        print(len(amset_data.ir_kpoints_idx))
        print(max(amset_data.ir_kpoints_idx))
        # print(max(amset_data.ir_to_full_kpoint_mapping))
        # print(len(amset_data.ir_to_full_kpoint_mapping))

        # x = extra_kpts[0][2]
        all_kpoints = amset_data.full_kpoints
        weights = amset_data.kpoint_weights
        mask = (all_kpoints[:, 2] == 0.)

        center_points = all_kpoints[mask][:, :2]
        center_labels = map("{:.3g}".format, weights[mask])

        from scipy.spatial import Voronoi, voronoi_plot_2d
        vor = Voronoi(center_points)
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        voronoi_plot_2d(vor)

        # plt.xlim((-0.48, -0.38))
        # plt.ylim((-0.48, -0.38))
        ax = plt.gca()

        for i, txt in enumerate(center_labels):
            kx = center_points[i][0]
            ky = center_points[i][1]
            if -0.38 > kx > -0.48 and -0.38 > ky > -0.48:
                ax.annotate(str(txt), (kx, ky))

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


