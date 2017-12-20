from pymatgen import MPRester
from pylab import plot,show, scatter
import numpy as np

from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.io.vasp import Vasprun
from pymatgen.symmetry.bandstructure import HighSymmKpath

from analytical_band_from_BZT import get_energy
from pymatgen import Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tools import get_energy_args, calc_analytical_energy, norm, \
    get_bindex_bspin, get_bs_extrema
from constants import hbar, m_e, Ry_to_eV, A_to_m, m_to_cm, A_to_nm, e, k_B,\
                        epsilon_0, default_small_E, dTdz, sq3

api = MPRester("fDJKEZpxSyvsXdCt")

def retrieve_bs(coeff_file, bs, ibands, cbm):
    # sp=bs.bands.keys()[0]
    engre, nwave, nsym, nstv, vec, vec2, out_vec2, br_dir = get_energy_args(coeff_file, ibands)

    #you can use a for loop along a certain list of k-points.
    for i, iband in enumerate(ibands):
        en = []
        sym_line_kpoints = [k.frac_coords for k in bs.kpoints]
        for kpt in sym_line_kpoints:
            e, v, m = get_energy(kpt, engre[i], nwave, nsym, nstv, vec, vec2=vec2, out_vec2=out_vec2, br_dir=br_dir, cbm=cbm)
            en.append(e*13.605)

        # plot(np.array(bs.bands[sp])[iband-1,:].T-bs.efermi) # from MP
        # plot(np.array(bs.bands[sp])[iband-2,:].T-bs.efermi) # from MP
        # plot(np.array(bs.bands[sp])[iband-3,:].T-bs.efermi) # from MP
        plot(en, color='b') # interpolated by BoltzTraP
    show()




if __name__ == "__main__":
    # user inputs
    PbTe_id = 'mp-19717' # valence_idx = 9
    Si_id = 'mp-149' # valence_idx = 4
    GaAs_id = 'mp-2534' # valence_idx = ?
    SnSe2_id = "mp-665"

    Si_bs = api.get_bandstructure_by_material_id(Si_id)
    bs = Vasprun('GaAs_28electron_ncsf_line_vasprun.xml').get_band_structure(
        kpoints_filename='GaAs_28electron_KPOINTS', line_mode=True)
    GaAs_st = api.get_structure_by_material_id(GaAs_id)

    bs.structure =  GaAs_st
    Si_bs.structure = api.get_structure_by_material_id(Si_id)
    print(bs.get_sym_eq_kpoints([0.5, 0.5, 0.5]))
    print(bs.get_sym_eq_kpoints([ 0.5,  0.,  0.5]))

    # vbm_idx = bs.get_vbm()['band_index'][Spin.up][0]
    vbm_idx, _ = get_bindex_bspin(Si_bs.get_vbm(), is_cbm=False)
    print('vbm band index (vbm_idx): {}'.format(vbm_idx))
    ibands = [1, 2] # in this notation, 1 is the last valence band
    ibands = [i + vbm_idx for i in ibands]

    PbTe_coeff_file = '../test_files/PbTe/fort.123'
    Si_coeff_file = "../test_files/Si/Si_fort.123"
#    GaAs_coeff_file = "../test_files/GaAs/fort.123_GaAs_sym_23x23x23"
    GaAs_coeff_file = "../test_files/GaAs/fort.123_GaAs_1099kp"
    # SnSe2_coeff_file = "/Users/alirezafaghaninia/Dropbox/Berkeley_Lab_Work/Yanzhong_Pei/SnSe2/boltztrap_vdw_dense/boltztrap/fort.123"
    # SnSe2_coeff_file = "/Users/alirezafaghaninia/Documents/boltztrap_examples/SnSe2/boltztrap_vdw_soc/boltztrap/fort.123"
    # SnSe2_coeff_file = "/Users/alirezafaghaninia/Documents/boltztrap_examples/SnSe2/boltztrap_vdw_better_geom_dense/boltztrap/fort.123"

    # retrieve_bs(coeff_file=PbTe_coeff_file, bs=bs, ibands=ibands)
    # retrieve_bs(coeff_file=Si_coeff_file, bs=Si_bs, ibands=ibands, cbm=True)

    # retrieve_bs(coeff_file=GaAs_coeff_file, bs=bs, ibands=ibands, cbm=False)

    # retrieve_bs(coeff_file=SnSe2_coeff_file, bs=bs, ibands=[11, 12, 13, 14])
    # retrieve_bs(coeff_file=SnSe2_coeff_file, bs=bs, ibands=[24, 25, 26, 27])

    # extrema = get_bs_extrema(bs, coeff_file=GaAs_coeff_file, nk_ibz=17, v_cut=1e4, min_normdiff=0.1, Ecut=0.5, nex_max=20)
    extrema = get_bs_extrema(Si_bs, coeff_file=Si_coeff_file, nk_ibz=25, v_cut=1e4, min_normdiff=0.1, Ecut=0.5, nex_max=20)
    print(extrema)
