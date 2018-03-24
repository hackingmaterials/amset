from amset.utils.constants import hbar, A_to_m, m_to_cm, Ry_to_eV, \
    Hartree_to_eV, m_e, e
from amset.utils.pymatgen_loader_for_bzt2 import PMG_Vasprun_Loader, PMG_BS_Loader
from pymatgen import MPRester
import numpy as np
import os
from time import time

from matminer import PlotlyFig
import BoltzTraP2.dft
from BoltzTraP2 import sphere, fite

# from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.io.vasp import Vasprun
# from pymatgen.symmetry.bandstructure import HighSymmKpath

from amset.utils.analytical_band_from_BZT import get_energy
# from pymatgen import Spin
# from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from amset.utils.tools import get_energy_args, get_bindex_bspin, norm

# from amset.utils.constants import hbar, m_e, Ry_to_eV, A_to_m, m_to_cm, A_to_nm, e, k_B,\
#                         epsilon_0, default_small_E, dTdz, sq3

api = MPRester("fDJKEZpxSyvsXdCt")


def retrieve_bs_boltztrap1(coeff_file, bs, ibands, cbm, matrix=None):
    engre, nwave, nsym, nstv, vec, vec2, out_vec2, br_dir = get_energy_args(coeff_file, ibands)
    pf = PlotlyFig(filename='Energy-bt1')
    plot_data =[]
    v_data = []
    mass_data = []
    trace_names = []
    mass_unit_conversion = e / Ry_to_eV / A_to_m**2 * (hbar*2*np.pi)**2 / m_e
    for i, iband in enumerate(ibands):
        en = []
        vel = []
        masses = []
        sym_line_kpoints = [k.frac_coords for k in bs.kpoints]
        print(len(sym_line_kpoints))
        for kpt in sym_line_kpoints:
            ee, de, dde = get_energy(kpt, engre[i], nwave, nsym, nstv, vec, vec2=vec2, out_vec2=out_vec2, br_dir=br_dir, cbm=cbm)
            en.append(ee*Ry_to_eV)
            v = matrix.dot(de)
            transformed_dde = dde/ 0.52917721067
            # vel.append(norm(v) / (hbar*2*np.pi)* Ry_to_eV/ 0.52917721067 * A_to_m * m_to_cm)
            vel.append(np.linalg.norm(v) / (hbar*2*np.pi)* Ry_to_eV/ 0.52917721067 * A_to_m * m_to_cm)
            masses.append(1/(transformed_dde.trace()/3.0) * mass_unit_conversion)
        plot_data.append((np.linspace(0, len(en)), en))
        v_data.append((en, vel))
        mass_data.append((en, masses))
        trace_names.append('band {}'.format(iband))
    # print('boltztrap1')
    # print(en[:10])
    # print(vel[:10])
    pf.xy(plot_data, names=[n for n in trace_names])
    pf2 = PlotlyFig(filename='Velocity-bt1')
    pf2.xy(v_data, names=[n for n in trace_names])
    pf3 = PlotlyFig(filename='mass-bt1')
    pf3.xy(mass_data, names=[n for n in trace_names])


def retrieve_bs_boltztrap2(vrun_path, bs, ibands):
    ibands = [i-1 for i in ibands]
    pf = PlotlyFig(filename='Energy-bt2')
    sym_line_kpoints = [k.frac_coords for k in bs.kpoints]

    # bz_data = BoltzTraP2.dft.DFTData(vrun_path, derivatives=False)
    bz_data = PMG_Vasprun_Loader(os.path.join(vrun_path, 'vasprun.xml'))
    # bz_data = PMG_BS_Loader(bs, nelect=28)
    equivalences = sphere.get_equivalences(bz_data.atoms, len(bz_data.kpoints) * 10)
    lattvec = bz_data.get_lattvec()
    print('lattvec:\n{}'.format(lattvec))
    coeffs = fite.fitde3D(bz_data, equivalences)

    # kpts = np.array([[0., 0., 0.], [0.5, 0.4, 0.3]])
    kpts = np.array(sym_line_kpoints)

    fitted = fite.getBands(kp=kpts, equivalences=equivalences,
                           lattvec=lattvec, coeffs=coeffs)

    EE = fitted[0]*Hartree_to_eV
    print(EE.shape)


    v = fitted[1]
    print(v.shape) # (3, nkpoints, nbands)
    vel = np.linalg.norm(v, axis=0) * Hartree_to_eV / hbar * A_to_m * m_to_cm
    # print(v[:, 0, :].T) # velocities at the first k-point
    # print(v[:, :, 0].T) # velocities at the first band

    print(fitted[2].shape) # (3, 3, nkpoints, nbands)
    plot_data = []
    v_data = []
    names = []
    EE = EE - np.max(EE[ibands[0], :]) # normalize w.r.t VBM as it is in boltztrap1
    for iband in ibands:
        plot_data.append((np.linspace(0, EE.shape[1]), EE[iband, :]))
        v_data.append((EE[iband, :], vel[:, iband]))
        names.append('band {}'.format(iband+1))
    # print('here ibands', ibands)
    # print('boltztrap2:')
    # print(EE[iband, :10])
    # print(vel[:10, iband])
    pf.xy(plot_data, names=[n for n in names])
    pf2 = PlotlyFig(filename='Velocity-bt2')
    pf2.xy(v_data, names=[n for n in names])

    mass_data = []
    mass_unit_conversion = e / Hartree_to_eV / A_to_m**2 * hbar**2 / m_e
    for iband in ibands:
        masses = []
        for i in range(fitted[2].shape[2]):
            dde = fitted[2][:, :, i, iband].trace()/3.0
            # masses.append(1/(dde) * hbar ** 2/ m_e / A_to_m ** 2 * e * Hartree_to_eV)
            masses.append(1/dde * mass_unit_conversion)
        mass_data.append((EE[iband, :], masses))
    pf3 = PlotlyFig(filename='mass-bt2')
    pf3.xy(mass_data, names=[n for n in names])
    print('mass tensor at the point {} of band {}:\n{}'.format(kpts[0], iband, 1.0/fitted[2][:, :, 0, iband]* mass_unit_conversion))



if __name__ == "__main__":
    # user inputs
    DIR = os.path.dirname(__file__)
    PbTe_id = 'mp-19717' # valence_idx = 9
    Si_id = 'mp-149' # valence_idx = 4
    GaAs_id = 'mp-2534' # valence_idx = ?
    SnSe2_id = "mp-665"

    test_dir = os.path.join(DIR, '../../test_files')

    # Si_bs = api.get_bandstructure_by_material_id(Si_id)
    # vrun = Vasprun(os.path.join(test_dir, 'GaAs/28_electrons_line/vasprun.xml'))
    vrun = Vasprun(os.path.join(test_dir, 'GaAs/nscf-uniform/vasprun.xml'))
    bs = vrun.get_band_structure(
        kpoints_filename=os.path.join(test_dir, 'GaAs/28_electrons_line/KPOINTS'), line_mode=True)
    rec_matrix = vrun.final_structure.lattice.reciprocal_lattice.matrix
    dir_matrix = vrun.final_structure.lattice.matrix
    st = vrun.final_structure
    print('reciprocal lattice matrix from Vasprun:\n{}'.format(rec_matrix))
    print('direct lattice matrix from Vasprun:\n{}'.format(dir_matrix))
    print('lattice constants: {}, {}, {}'.format(st.lattice.a, st.lattice.b, st.lattice.c))

    InP_root = '/Users/alirezafaghaninia/Documents/py3/py3_codes/thermoelectrics_work/thermoelectrics_work/amset_examples/InP_mp-20351'
    # bs = Vasprun(os.path.join(root, 'vasprun.xml')).get_band_structure(
    #     kpoints_filename=os.path.join(root, 'KPOINTS'))
    # InP_st = api.get_structure_by_material_id('mp-20351')
    # bs.structure = InP_st

    # GaAs_st = api.get_structure_by_material_id(GaAs_id)
    #
    # bs.structure =  GaAs_st
    # # Si_bs.structure = api.get_structure_by_material_id(Si_id)
    # Si_bs.structure = Vasprun(os.path.join(test_dir, "Si/vasprun.xml")).final_structure

    vbm_idx, _ = get_bindex_bspin(bs.get_vbm(), is_cbm=False)
    print('vbm band index (vbm_idx): {}'.format(vbm_idx))
    ibands = [1, 2] # in this notation, 1 is the last valence band
    ibands = [i + vbm_idx for i in ibands]

    PbTe_coeff_file = os.path.join(test_dir, 'PbTe/fort.123')
    Si_coeff_file = os.path.join(test_dir, "Si/Si_fort.123")
    # GaAs_coeff_file = os.path.join(test_dir, "GaAs/fort.123_GaAs_1099kp")
    GaAs_coeff_file = os.path.join(test_dir, "GaAs/nscf-uniform/boltztrap/fort.123")

    # start_time = time()
    # retrieve_bs_boltztrap1(coeff_file=PbTe_coeff_file, bs=bs, ibands=ibands)
    # retrieve_bs_boltztrap1(coeff_file=Si_coeff_file, bs=Si_bs, ibands=ibands, cbm=True)

    retrieve_bs_boltztrap1(coeff_file=GaAs_coeff_file, bs=bs, ibands=ibands, cbm=True, matrix=dir_matrix)

    # retrieve_bs_boltztrap1(coeff_file=SnSe2_coeff_file, bs=bs, ibands=[11, 12, 13, 14])
    # print("Boltztrap1 total time: {}".format(time() - start_time))

    # start_time = time()

    # retrieve_bs_boltztrap2(os.path.join(test_dir, 'GaAs'), ibands=ibands)
    # retrieve_bs_boltztrap2(os.path.join(test_dir, 'ZnS_391_vrun'), ibands=ibands)
    # retrieve_bs_boltztrap2(os.path.join(DIR, '../../../BoltzTraP2-18.1.2/data/Si.vasp'), ibands=[3, 4])
    # retrieve_bs_boltztrap2(root, ibands=[2, 3])
    # retrieve_bs_boltztrap2(InP_root, bs=bs, ibands=ibands)
    retrieve_bs_boltztrap2(os.path.join(test_dir, 'GaAs/nscf-uniform'), bs=bs, ibands=ibands)

    # print("Boltztrap2 total time: {}".format(time() - start_time))

    # extrema = get_bs_extrema(bs, coeff_file=GaAs_coeff_file, nk_ibz=17, v_cut=1e4, min_normdiff=0.1, Ecut=0.5, nex_max=20)
