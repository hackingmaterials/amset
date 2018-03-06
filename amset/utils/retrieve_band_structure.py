from pymatgen import MPRester
import numpy as np
import os
from matminer import PlotlyFig
import BoltzTraP2.dft
from BoltzTraP2 import sphere, fite

# from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.io.vasp import Vasprun
# from pymatgen.symmetry.bandstructure import HighSymmKpath

from amset.utils.analytical_band_from_BZT import get_energy
# from pymatgen import Spin
# from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from amset.utils.tools import get_energy_args, get_bindex_bspin
# from amset.utils.constants import hbar, m_e, Ry_to_eV, A_to_m, m_to_cm, A_to_nm, e, k_B,\
#                         epsilon_0, default_small_E, dTdz, sq3


api = MPRester("fDJKEZpxSyvsXdCt")


def retrieve_bs_boltztrap1(coeff_file, bs, ibands, cbm):
    engre, nwave, nsym, nstv, vec, vec2, out_vec2, br_dir = get_energy_args(coeff_file, ibands)

    #you can use a for loop along a certain list of k-points.
    pf = PlotlyFig(filename='boltztrap1')
    plot_data =[]
    names = []
    for i, iband in enumerate(ibands):
        en = []
        sym_line_kpoints = [k.frac_coords for k in bs.kpoints]
        for kpt in sym_line_kpoints:
            e, v, m = get_energy(kpt, engre[i], nwave, nsym, nstv, vec, vec2=vec2, out_vec2=out_vec2, br_dir=br_dir, cbm=cbm)
            en.append(e*13.605)

        plot_data.append((np.linspace(0, len(en)), en))
        names.append('band {}'.format(iband))
    pf.xy(plot_data, names=names)


def retrieve_bs_boltztrap2(vrun_path, ibands):
    pf = PlotlyFig(filename='boltztrap2')
    bz_data = BoltzTraP2.dft.DFTData(vrun_path, derivatives=False)
    equivalences = sphere.get_equivalences(bz_data.atoms, len(bz_data.kpoints) * 10)
    lattvec = bz_data.get_lattvec()
    coeffs = fite.fitde3D(bz_data, equivalences)

    kpts = np.array([[0., 0., 0.], [0.5, 0.4, 0.3]])
    energies = []
    velocities = []
    fitted = fite.getBands(kp=kpts, equivalences=equivalences,
                           lattvec=lattvec, coeffs=coeffs)

    EE = fitted[0]*13.605
    print(EE.shape)
    print(EE)

    v = fitted[1]
    print(v.shape)
    # v[abs(v) < 1e-10] = 0
    print(v)

if __name__ == "__main__":
    # user inputs
    DIR = os.path.dirname(__file__)
    PbTe_id = 'mp-19717' # valence_idx = 9
    Si_id = 'mp-149' # valence_idx = 4
    GaAs_id = 'mp-2534' # valence_idx = ?
    SnSe2_id = "mp-665"

    test_dir = os.path.join(DIR, '../../test_files')

    # Si_bs = api.get_bandstructure_by_material_id(Si_id)
    bs = Vasprun(os.path.join(test_dir, 'GaAs/28_electrons_line/vasprun.xml')).get_band_structure(
        kpoints_filename=os.path.join(test_dir, 'GaAs/28_electrons_line/KPOINTS'), line_mode=True)
    # GaAs_st = api.get_structure_by_material_id(GaAs_id)
    #
    # bs.structure =  GaAs_st
    # # Si_bs.structure = api.get_structure_by_material_id(Si_id)
    # Si_bs.structure = Vasprun(os.path.join(test_dir, "Si/vasprun.xml")).final_structure
    # print(bs.get_sym_eq_kpoints([0.5, 0.5, 0.5]))
    # print(bs.get_sym_eq_kpoints([ 0.5,  0.,  0.5]))

    # vbm_idx, _ = get_bindex_bspin(Si_bs.get_vbm(), is_cbm=False)
    vbm_idx, _ = get_bindex_bspin(bs.get_vbm(), is_cbm=False)
    print('vbm band index (vbm_idx): {}'.format(vbm_idx))
    ibands = [1, 2] # in this notation, 1 is the last valence band
    ibands = [i + vbm_idx for i in ibands]

    PbTe_coeff_file = os.path.join(test_dir, 'PbTe/fort.123')
    Si_coeff_file = os.path.join(test_dir, "Si/Si_fort.123")
    GaAs_coeff_file = os.path.join(test_dir, "GaAs/fort.123_GaAs_1099kp")

    # retrieve_bs_boltztrap1(coeff_file=PbTe_coeff_file, bs=bs, ibands=ibands)
    # retrieve_bs_boltztrap1(coeff_file=Si_coeff_file, bs=Si_bs, ibands=ibands, cbm=True)

    # retrieve_bs_boltztrap1(coeff_file=GaAs_coeff_file, bs=bs, ibands=ibands, cbm=True)
    retrieve_bs_boltztrap2(os.path.join(test_dir, 'GaAs'), ibands=[13, 14])
    # retrieve_bs_boltztrap2(os.path.join(test_dir, 'GaAs/28_electrons_line'), ibands=[13, 14])
    # retrieve_bs_boltztrap2('/Users/alirezafaghaninia/Documents/py3/py3_codes/BoltzTraP2-18.1.2/scratch/Si_data', ibands=[4, 5])

    # retrieve_bs_boltztrap1(coeff_file=SnSe2_coeff_file, bs=bs, ibands=[11, 12, 13, 14])
    # retrieve_bs_boltztrap1(coeff_file=SnSe2_coeff_file, bs=bs, ibands=[24, 25, 26, 27])

    # extrema = get_bs_extrema(bs, coeff_file=GaAs_coeff_file, nk_ibz=17, v_cut=1e4, min_normdiff=0.1, Ecut=0.5, nex_max=20)
    # extrema = get_bs_extrema(Si_bs, coeff_file=Si_coeff_file, nk_ibz=31, v_cut=1e4, min_normdiff=0.15, Ecut=0.25, nex_max=20)
    # print(extrema)
