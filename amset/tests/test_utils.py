import numpy as np
import os
import unittest

from amset.utils.constants import comp_to_dirname
from amset.utils.band_structure import kpts_to_first_BZ, get_closest_k, \
    remove_duplicate_kpoints, get_bindex_bspin
from amset.utils.band_interpolation import get_energy_args, interpolate_bs, \
    get_bs_extrema
from pymatgen.io.vasp import Vasprun

tdir = os.path.join(os.path.dirname(__file__), '..', '..', 'test_files')
vruns = {c: Vasprun(os.path.join(tdir, comp_to_dirname[c], 'vasprun.xml'))
         for c in comp_to_dirname}
coeff_files = {c: os.path.join(tdir, comp_to_dirname[c], 'fort.123')
               for c in comp_to_dirname}

CHECK_BOLTZTRAP2 = True


class AmsetToolsTest(unittest.TestCase):
    def setUp(self):
        pass

    def listalmostequal(self, list1, list2, places=3):
        for l1, l2 in zip(list1, list2):
            self.assertAlmostEqual(l1, l2, places=places)

    def test_get_bs_extrema(self):
        extrema = get_bs_extrema(bs=vruns['GaAs'].get_band_structure(),
                                 coeff_file=coeff_files['GaAs'], Ecut=1.0)
        # first conduction band extrema:
        self.listalmostequal(extrema['n'][0], [.0, .0, .0], 10)
        self.listalmostequal(extrema['n'][1], [.0, -0.4701101, .0], 4)
        self.assertEqual(len(extrema['n']), 2)
        # last valence band extrema
        self.listalmostequal(extrema['p'][0], [.0, .0, .0], 10)
        self.listalmostequal(extrema['p'][1], [-0.31693, -0.04371, 0.], 4)
        self.assertEqual(len(extrema['p']), 2)

        si_extrema = get_bs_extrema(bs=vruns['Si'].get_band_structure(),
                                    coeff_file=coeff_files['Si'], Ecut=1.0)
        self.listalmostequal(si_extrema['n'][0], [0.419204, 0.419204, 0.], 4)
        self.listalmostequal(si_extrema['n'][1], [-0.4638, -0.4638, -0.4638], 4)
        self.listalmostequal(si_extrema['p'][0], [0.0, 0.0, 0.0], 10)
        self.listalmostequal(si_extrema['p'][1], [-0.226681, -0.049923, 0.], 3)

        pbte_extrema = get_bs_extrema(bs=vruns['PbTe'].get_band_structure(),
                                      coeff_file=coeff_files['PbTe'], Ecut=1.0)
        self.listalmostequal(pbte_extrema['n'][0], [0., 0.5, 0.], 10)
        self.listalmostequal(pbte_extrema['n'][1], [.1522, -.0431, .1522], 4)
        self.listalmostequal(pbte_extrema['p'][0], [0., 0.5, 0.], 10)
        self.listalmostequal(pbte_extrema['p'][1], [.4784, -.2709, .2278], 3)
        self.listalmostequal(pbte_extrema['p'][2], [.162054, .162054, 0.], 3)

        inp_extrema = get_bs_extrema(bs=vruns['InP'].get_band_structure(),
                                     coeff_file=coeff_files['InP'], Ecut=1.0)
        self.listalmostequal(inp_extrema['n'][0], [0., 0.0, 0.], 10)
        self.listalmostequal(inp_extrema['n'][1], [0., 0.5, 0.], 10)
        self.listalmostequal(inp_extrema['p'][0], [0., 0.0, 0.], 10)
        self.listalmostequal(inp_extrema['p'][1], [-0.3843, -0.0325, 0.], 4)

        alcus2_extrema = get_bs_extrema(bs=vruns['AlCuS2'].get_band_structure(),
                                        coeff_file=coeff_files['AlCuS2'],
                                        Ecut=1.0)
        self.listalmostequal(alcus2_extrema['n'][0], [0., 0.0, 0.0], 10)
        self.listalmostequal(alcus2_extrema['n'][1], [0., 0.0, 0.5], 10)
        self.listalmostequal(alcus2_extrema['n'][2], [-0.49973, -0.49973, 0.],
                             4)
        self.listalmostequal(alcus2_extrema['n'][3],
                             [0.49047, 0.49047, 0.49818], 4)
        self.listalmostequal(alcus2_extrema['p'][0], [0., 0.0, 0.0], 10)
        self.listalmostequal(alcus2_extrema['p'][1], [0.28291, 0., -0.40218], 4)
        self.listalmostequal(alcus2_extrema['p'][2], [-0.25765, 0.25148, 0.], 4)
        self.listalmostequal(alcus2_extrema['p'][3], [-0.49973, -0.49973, 0.],
                             4)

        in2o3_extrema = get_bs_extrema(bs=vruns['In2O3'].get_band_structure(),
                                       coeff_file=coeff_files['In2O3'],
                                       Ecut=1.0)
        self.listalmostequal(in2o3_extrema['n'][0], [0., 0.0, 0.0], 10)
        self.listalmostequal(in2o3_extrema['p'][0], [0., 0.09631, 0.0], 4)
        self.listalmostequal(in2o3_extrema['p'][1], [0.30498, 0.30498, 0.18299],
                             4)

    def test_kpts_to_first_BZ(self):
        kpts_orig = [[0.51, 1.00, -0.50], [1.40, -1.20, 0.49]]
        kpts_trns = [[-0.49, 0.00, -0.50], [0.40, -0.20, 0.49]]
        # self.assertListEqual() #doesn't work as they differ at 7th decimal
        for ik, k in enumerate(kpts_to_first_BZ(kpts_orig)):
            np.testing.assert_array_almost_equal(kpts_trns[ik], k, 7)
        self.assertTrue(isinstance(kpts_to_first_BZ(kpts_orig), list))

    def test_get_closest_k(self):
        kpts = np.array([[0.51, -0.5, 0.5], [0.4, 0.5, 0.51]])
        np.testing.assert_array_equal([0.4, 0.5, 0.51],
                                      get_closest_k(np.array([0.5, 0.5, 0.5]),
                                                    kpts, return_diff=False))
        np.testing.assert_array_almost_equal([0.1, 0.0, -0.01],
                                             get_closest_k(
                                                 np.array([0.5, 0.5, 0.5]),
                                                 kpts, return_diff=True))

    def test_remove_duplicate_kpoints(self):
        kpts_orig = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.00999],
                     [0.25, 0.25, 0.25], [0.25, 0.25, 0.25],
                     [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5]]
        kpts_out = [[0.0, 0.0, 0.00999],
                    [0.25, 0.25, 0.25],
                    [0.5, 0.5, -0.5]]
        # print(remove_duplicate_kpoints(kpts_orig))
        self.assertListEqual(kpts_out, remove_duplicate_kpoints(kpts_orig))

    def test_interpolate_bs(self):
        bs = vruns['GaAs'].get_band_structure()
        vbm_idx, vbm_bidx = get_bindex_bspin(bs.get_vbm(), is_cbm=False)
        cbm_idx, cbm_bidx = get_bindex_bspin(bs.get_cbm(), is_cbm=True)
        dft_vbm = bs.get_vbm()['energy']
        dft_vb = np.array(bs.bands[vbm_bidx][vbm_idx]) - dft_vbm
        dft_cb = np.array(bs.bands[cbm_bidx][cbm_idx]) - dft_vbm

        # in boltztrap1 interpolation the first band is 1st not 0th
        vbm_idx += 1
        cbm_idx += 1

        kpts = [k.frac_coords for k in bs.kpoints]
        matrix = vruns['GaAs'].lattice.matrix

        # get the interpolation parameters
        interp_params1 = get_energy_args(coeff_files['GaAs'],
                                         [vbm_idx, cbm_idx])

        # calculate and check the last valence and the first conduction bands:
        vb_en1, vb_vel1, vb_masses1 = interpolate_bs(kpts, interp_params1,
                                                     iband=0,
                                                     method="boltztrap1",
                                                     scissor=0.0, matrix=matrix)
        cb_en1, cb_vel1, cb_masses1 = interpolate_bs(kpts, interp_params1,
                                                     iband=1,
                                                     method="boltztrap1",
                                                     scissor=0.0, matrix=matrix)

        vbm = np.max(vb_en1)
        vb_en1 -= vbm
        cb_en1 -= vbm
        interp_gap1 = min(cb_en1) - max(vb_en1)
        self.assertAlmostEqual(bs.get_band_gap()['energy'], interp_gap1, 4)
        self.assertAlmostEqual(interp_gap1, 0.1899, 4)

        # check exact match between DFT energy and interpolated band energy
        self.assertAlmostEqual(np.mean(vb_en1 - dft_vb), 0.0, 4)
        self.assertAlmostEqual(np.std(vb_en1 - dft_vb), 0.0, 4)
        self.assertAlmostEqual(np.mean(cb_en1 - dft_cb), 0.0, 4)
        self.assertAlmostEqual(np.std(cb_en1 - dft_cb), 0.0, 4)

        # check the average of the velocity vectors; not isotropic since not
        # all sym. eq. kpoints are sampled
        expected_vb_v = [37199316.52376, 64230953.3495545, 30966751.7547101]
        expected_cb_v = [63838832.347664, 78291298.7589355, 69109280.002242]
        self.listalmostequal(np.mean(vb_vel1, axis=0), expected_vb_v, 0)
        self.listalmostequal(np.mean(cb_vel1, axis=0), expected_cb_v, 0)

        if CHECK_BOLTZTRAP2:
            from amset.utils.pymatgen_loader_for_bzt2 import PymatgenLoader
            from BoltzTraP2 import sphere, fite
            bz2_data = PymatgenLoader.from_vasprun(vruns['GaAs'])
            equivalences = sphere.get_equivalences(atoms=bz2_data.atoms,
                                                   nkpt=len(
                                                       bz2_data.kpoints) * 5,
                                                   magmom=None)
            lattvec = bz2_data.get_lattvec()
            coeffs = fite.fitde3D(bz2_data, equivalences)
            interp_params2 = (equivalences, lattvec, coeffs)
            vb_en2, vb_vel2, vb_masses2 = interpolate_bs(kpts, interp_params2,
                                                         iband=vbm_idx,
                                                         method="boltztrap2",
                                                         scissor=0.0,
                                                         matrix=matrix)
            cb_en2, cb_vel2, cb_masses2 = interpolate_bs(kpts, interp_params2,
                                                         iband=cbm_idx,
                                                         method="boltztrap2",
                                                         scissor=0.0,
                                                         matrix=matrix)
            vbm2 = np.max(vb_en2)
            vb_en2 -= vbm2
            cb_en2 -= vbm2
            interp_gap2 = min(cb_en2) - max(vb_en2)
            self.assertAlmostEqual(interp_gap1, interp_gap2, 4)
            self.assertAlmostEqual(np.mean(vb_en1 - vb_en2), 0.0, 4)
            self.assertAlmostEqual(np.std(vb_en1 - vb_en2), 0.0, 4)
            self.assertAlmostEqual(np.mean(cb_en1 - cb_en2), 0.0, 4)
            self.assertAlmostEqual(np.std(cb_en1 - cb_en2), 0.0, 4)
            vb_avg2 = np.mean(vb_vel2, axis=0)
            cb_avg2 = np.mean(cb_vel2, axis=0)
            for i in range(3):
                self.assertLessEqual(1 - abs(vb_avg2[i] / expected_vb_v[i]),
                                     0.001)
                self.assertLessEqual(1 - abs(cb_avg2[i] / expected_cb_v[i]),
                                     0.001)


if __name__ == '__main__':
    unittest.main()
