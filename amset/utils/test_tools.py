import numpy as np
import os
import unittest
from amset.utils.tools import kpts_to_first_BZ, get_closest_k, \
    remove_duplicate_kpoints, get_energy_args, get_bindex_bspin, \
    interpolate_bs, get_bs_extrema
from pymatgen.io.vasp import Vasprun

comps = {
    'GaAs': 'GaAs_mp-2534',
    'Si': 'Si_mp-149',
    'PbTe': 'PbTe_mp-19717',
    'InP': 'InP_mp-20351',
    'AlCuS2': 'AlCuS2_mp-4979',
    'In2O3': 'In2O3_mp-22598',
}
tdir = os.path.join(os.path.dirname(__file__), '..', '..', 'test_files')
vruns = {c: Vasprun(os.path.join(tdir, comps[c], 'vasprun.xml')) for c in comps}
coeff_files = {c: os.path.join(tdir, comps[c], 'fort.123') for c in comps}

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
        self.listalmostequal(extrema['n'][1], [.5, .5, .5], 10)
        self.listalmostequal(extrema['n'][2], [-0.4414894, -.4414894, .0], 4)
        self.assertEqual(len(extrema['n']), 3)
        # last valence band extrema
        self.listalmostequal(extrema['p'][0], [.0, .0, .0], 10)
        self.listalmostequal(extrema['p'][1], [-.2786, -.0459, .0], 4)
        self.listalmostequal(extrema['p'][2], [0.39561, 0.11702 , 0.162898], 4)
        self.assertEqual(len(extrema['p']), 3)

        Si_extrema = get_bs_extrema(bs=vruns['Si'].get_band_structure(),
                                    coeff_file=coeff_files['Si'], Ecut=1.0)
        self.listalmostequal(Si_extrema['n'][0], [0.419204, 0.419204, 0.], 4)
        self.listalmostequal(Si_extrema['n'][1], [-0.4638, -0.4638, -0.4638],4)
        self.listalmostequal(Si_extrema['p'][0], [0.0, 0.0, 0.0], 10)
        self.listalmostequal(Si_extrema['p'][1], [-0.226681, -0.049923, 0.], 3)

        PbTe_extrema = get_bs_extrema(bs=vruns['PbTe'].get_band_structure(),
                                coeff_file=coeff_files['PbTe'], Ecut=1.0)
        self.listalmostequal(PbTe_extrema['n'][0], [0. , 0.5, 0. ], 10)
        self.listalmostequal(PbTe_extrema['n'][1], [.1522, -.0431,  .1522], 4)
        self.listalmostequal(PbTe_extrema['p'][0], [0. , 0.5, 0. ], 10)
        self.listalmostequal(PbTe_extrema['p'][1], [.4784, -.2709,  .2278], 3)
        self.listalmostequal(PbTe_extrema['p'][2], [.162054 , .162054, 0.], 3)

        InP_extrema = get_bs_extrema(bs=vruns['InP'].get_band_structure(),
                                coeff_file=coeff_files['InP'], Ecut=1.0)
        self.listalmostequal(InP_extrema['n'][0], [0. , 0.0, 0. ], 10)
        self.listalmostequal(InP_extrema['n'][1], [0. , 0.5, 0. ], 10)
        self.listalmostequal(InP_extrema['p'][0], [0. , 0.0, 0. ], 10)
        self.listalmostequal(InP_extrema['p'][1], [-0.3843 , -0.0325,  0.], 4)

        AlCuS2_extrema = get_bs_extrema(bs=vruns['AlCuS2'].get_band_structure(),
                                coeff_file=coeff_files['AlCuS2'], Ecut=1.0)
        self.listalmostequal(AlCuS2_extrema['n'][0], [0. , 0.0, 0.0 ], 10)
        self.listalmostequal(AlCuS2_extrema['n'][1], [0. , 0.0, 0.5 ], 10)
        self.listalmostequal(AlCuS2_extrema['n'][2], [-0.49973, -0.49973,  0.], 4)
        self.listalmostequal(AlCuS2_extrema['n'][3], [0.49047, 0.49047, 0.49818], 4)
        self.listalmostequal(AlCuS2_extrema['p'][0], [0. , 0.0, 0.0 ], 10)
        self.listalmostequal(AlCuS2_extrema['p'][1], [0.28291, 0., -0.40218], 4)
        self.listalmostequal(AlCuS2_extrema['p'][2], [-0.25765, 0.25148, 0.], 4)
        self.listalmostequal(AlCuS2_extrema['p'][3], [-0.49973, -0.49973, 0.], 4)

        In2O3_extrema = get_bs_extrema(bs=vruns['In2O3'].get_band_structure(),
                                coeff_file=coeff_files['In2O3'], Ecut=1.0)
        self.listalmostequal(In2O3_extrema['n'][0], [0. , 0.0, 0.0 ], 10)
        self.listalmostequal(In2O3_extrema['p'][0], [0. , 0.09631, 0.0 ], 4)
        self.listalmostequal(In2O3_extrema['p'][1], [0.30498, 0.30498, 0.18299], 4)


    def test_kpts_to_first_BZ(self):
        kpts_orig = [[0.51, 1.00, -0.50], [1.40, -1.20, 0.49]]
        kpts_trns = [[-0.49, 0.00, -0.50], [0.40, -0.20, 0.49]]
        # self.assertListEqual() #doesn't work as they differ at 7th decimal
        for ik, k in enumerate(kpts_to_first_BZ(kpts_orig)):
            np.testing.assert_array_almost_equal(kpts_trns[ik], k, 7)
        self.assertTrue(isinstance(kpts_to_first_BZ(kpts_orig), list))


    def test_get_closest_k(self):
        kpts = np.array([[0.51, -0.5, 0.5], [0.4, 0.5, 0.51]])
        np.testing.assert_array_equal([0.4 , 0.5, 0.51],
            get_closest_k(np.array([0.5, 0.5, 0.5]), kpts, return_diff=False))
        np.testing.assert_array_almost_equal([0.1 , 0.0, -0.01],
            get_closest_k(np.array([0.5, 0.5, 0.5]), kpts, return_diff=True))


    def test_remove_duplicate_kpoints(self):
        kpts_orig = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.00999],
                     [0.25, 0.25, 0.25], [0.25, 0.25, 0.25],
                     [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, -0.5]]
        kpts_out = [[0.0, 0.0, 0.00999],
                    [0.25, 0.25, 0.25],
                    [0.5, 0.5, -0.5]]
        # print(remove_duplicate_kpoints(kpts_orig))
        self.assertListEqual(kpts_out, remove_duplicate_kpoints(kpts_orig))


    def test_interpolate_bs(self, check_bzt2=False):
        bs = vruns['GaAs'].get_band_structure()
        vbm_idx, vbm_bidx = get_bindex_bspin(bs.get_vbm(), is_cbm=False)
        cbm_idx, cbm_bidx = get_bindex_bspin(bs.get_cbm(), is_cbm=True)
        dft_vbm = bs.get_vbm()['energy']
        dft_vb = np.array(bs.bands[vbm_bidx][vbm_idx]) - dft_vbm
        dft_cb = np.array(bs.bands[cbm_bidx][cbm_idx]) - dft_vbm
        vbm_idx += 1 # in boltztrap1 interpolation the first band is 1st not 0th
        cbm_idx += 1

        kpts = [k.frac_coords for k in bs.kpoints]
        matrix = vruns['GaAs'].lattice.matrix

        # get the interpolation parameters
        interp_params1 = get_energy_args(coeff_files['GaAs'], [vbm_idx, cbm_idx])

        # calculate and check the last valence and the first conduction bands:
        vb_en1, vb_vel1, vb_masses1 = interpolate_bs(kpts, interp_params1,
                iband=0, method="boltztrap1", scissor=0.0, matrix=matrix)
        cb_en1, cb_vel1, cb_masses1 = interpolate_bs(kpts, interp_params1,
                iband=1, method="boltztrap1", scissor=0.0, matrix=matrix)

        vbm = np.max(vb_en1)
        vb_en1 -= vbm
        cb_en1 -= vbm
        interp_gap1 = min(cb_en1) - max(vb_en1)
        self.assertAlmostEqual(bs.get_band_gap()['energy'], interp_gap1, 4)
        self.assertAlmostEqual(interp_gap1, 0.9582, 4)

        # check exact match between DFT energy and interpolated band energy
        self.assertAlmostEqual(np.mean(vb_en1 - dft_vb), 0.0, 4)
        self.assertAlmostEqual(np.std(vb_en1 - dft_vb), 0.0, 4)
        self.assertAlmostEqual(np.mean(cb_en1 - dft_cb), 0.0, 4)
        self.assertAlmostEqual(np.std(cb_en1 - dft_cb), 0.0, 4)

        # check the average of the velocity vectors; not isotropic since not all sym. eq. kpoints are sampled
        expected_vb_v = [36110459.736, 67090934.345, 38192774.737]
        expected_cb_v = [68706796.0747, 73719673.252, 84421427.422]
        self.listalmostequal(np.mean(vb_vel1, axis=0), expected_vb_v, 0)
        self.listalmostequal(np.mean(cb_vel1, axis=0), expected_cb_v,0)

        if check_bzt2:
            from amset.utils.pymatgen_loader_for_bzt2 import PymatgenLoader
            from BoltzTraP2 import sphere, fite
            bz2_data = PymatgenLoader(vruns['GaAs'])
            equivalences = sphere.get_equivalences(bz2_data.atoms,
                                                   len(bz2_data.kpoints) * 10)
            lattvec = bz2_data.get_lattvec()
            coeffs = fite.fitde3D(bz2_data, equivalences)
            interp_params2 = (equivalences, lattvec, coeffs)
            vb_en2, vb_vel2, vb_masses2 = interpolate_bs(kpts, interp_params2,
                iband=vbm_idx, method="boltztrap2", scissor=0.0, matrix=matrix)
            cb_en2, cb_vel2, cb_masses2 = interpolate_bs(kpts, interp_params2,
                iband=cbm_idx, method="boltztrap2", scissor=0.0, matrix=matrix)
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
                self.assertLessEqual(1-abs(vb_avg2[i]/expected_vb_v[i]), 0.001)
                self.assertLessEqual(1-abs(cb_avg2[i] / expected_cb_v[i]), 0.001)


if __name__ == '__main__':
    unittest.main()