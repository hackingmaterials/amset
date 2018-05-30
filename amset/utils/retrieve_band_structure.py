from time import time

from amset.utils.pymatgen_loader_for_bzt2 import PymatgenLoader
from pymatgen import MPRester
import numpy as np
import os
from matminer import PlotlyFig
from BoltzTraP2 import sphere, fite
from pymatgen.io.vasp import Vasprun
from amset.utils.tools import get_energy_args, get_bindex_bspin, norm, \
    interpolate_bs, create_grid, generate_k_mesh_axes, array_to_kgrid, \
    kpts_to_first_BZ
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

api = MPRester("fDJKEZpxSyvsXdCt")


def retrieve_bs_boltztrap1(coeff_file, bs, ibands, matrix=None):
    interp_params = get_energy_args(coeff_file, ibands)
    pf = PlotlyFig(filename='Energy-bt1')
    plot_data =[]
    v_data = []
    mass_data = []
    trace_names = []
    Eref = 0.0

    sg = SpacegroupAnalyzer(bs.structure)
    nkk = 7
    kpts_and_weights = sg.get_ir_reciprocal_mesh(mesh=(nkk, nkk, nkk),
                                                 is_shift=[0, 0, 0])
    initial_ibzkpt = [i[0] for i in kpts_and_weights]
    vels = {iband: [] for iband in ibands}
    sym_line_kpoints = []
    # print(initial_ibzkpt)
    points_1d = generate_k_mesh_axes(important_pts=[[0.0, 0.0, 0.0]], kgrid_tp='coarse')
    kgrid_array = create_grid(points_1d)
    kpts = array_to_kgrid(kgrid_array)
    sym_line_kpoints = kpts
    sym_line_kpoints = []
    for k in kpts:
        # sym_line_kpoints += [np.dot(matrix.T, k)] # definitely not
        # sym_line_kpoints += [np.dot(k, matrix.T)] # definitely not
        # sym_line_kpoints += [np.dot(matrix.T, np.dot(k, matrix))] #1
        sym_line_kpoints += [np.dot(np.dot(matrix.T, k), matrix)] # same as #1
        # sym_line_kpoints += [np.dot(np.dot(matrix.T, k), matrix.T)]
        # sym_line_kpoints += [np.dot(bs.structure.lattice.reciprocal_lattice.matrix.T, k)]
    # sym_line_kpoints = kpts_to_first_BZ(sym_line_kpoints)


    # for k in initial_ibzkpt:
    #     # sym_line_kpoints += [k]
    #     sym_line_kpoints += [np.dot(matrix, k)]
    #     # sym_line_kpoints += list(bs.get_sym_eq_kpoints(k))

    print(sym_line_kpoints)

    for i, iband in enumerate(ibands):
        # sym_line_kpoints = [k.frac_coords for k in bs.kpoints]

        en, vel, masses = interpolate_bs(sym_line_kpoints, interp_params, iband=i,
                       method="boltztrap1", scissor=0.0, matrix=matrix, n_jobs=-1)
        # vels[iband] = np.array([np.dot(matrix.T, v) for v in vel])
        vels[iband] = np.array([np.dot(np.eye(3), v) for v in vel])

        vel = np.linalg.norm(vel, axis=1)
        masses = [mass.trace()/3.0 for mass in masses]
        if i==0:
            Eref = max(en)
        en = [E - Eref for E in en]
        plot_data.append((np.linspace(0, len(en)), en))
        v_data.append((en, vel))
        mass_data.append((en, masses))
        trace_names.append('band {}'.format(iband))

    print('average p-velocity:', np.mean(vels[9], axis=0))
    print('average n-velocity:', np.mean(vels[10], axis=0))
    quit()

    # print(v_data[1][1])
    # pf.xy(plot_data, names=[n for n in trace_names])
    # pf2 = PlotlyFig(filename='Velocity-bt1')
    # pf2.xy(v_data, names=[n for n in trace_names])
    # pf3 = PlotlyFig(filename='mass-bt1')
    # pf3.xy(mass_data, names=[n for n in trace_names])


def retrieve_bs_boltztrap2(vrun_path, bs, ibands):
    pf = PlotlyFig(filename='Energy-bt2')
    sym_line_kpoints = [k.frac_coords for k in bs.kpoints]
    # vrun = Vasprun(os.path.join(vrun_path, 'vasprun.xml'))
    # bz_data = PMG_Vasprun_Loader(vrun)
    bz_data = PymatgenLoader.from_files(os.path.join(vrun_path, 'vasprun.xml'))
    equivalences = sphere.get_equivalences(bz_data.atoms, len(bz_data.kpoints) * 5)
    lattvec = bz_data.get_lattvec()
    print('lattvec:\n{}'.format(lattvec))
    coeffs = fite.fitde3D(bz_data, equivalences)
    kpts = np.array(sym_line_kpoints)
    interp_params = (equivalences, lattvec, coeffs)
    plot_data = []
    v_data = []
    names = []
    mass_data = []
    eref = 0.0
    for ith, iband in enumerate(ibands):
        en, vel, masses = interpolate_bs(kpts, interp_params, iband=iband,
                       method="boltztrap2", matrix=lattvec*0.529177)
        if ith==0:
            eref = np.max(en)
        en -= eref
        plot_data.append((np.linspace(0, len(en)), en ))
        v_data.append((en, np.linalg.norm(vel, axis=1)))
        mass_data.append((en, [mass.trace()/3.0 for mass in masses]))
        names.append('band {}'.format(iband+1))
    print(v_data[1][1])
    pf.xy(plot_data, names=[n for n in names])
    pf2 = PlotlyFig(filename='Velocity-bt2')
    pf2.xy(v_data, names=[n for n in names])
    pf3 = PlotlyFig(filename='mass-bt2')
    pf3.xy(mass_data, names=[n for n in names])


if __name__ == "__main__":
    # user inputs
    DIR = os.path.dirname(__file__)
    test_dir = os.path.join(DIR, '../../test_files')

    # # GaAs
    vrun_path = 'GaAs/nscf-uniform/vasprun.xml'
    kpoints_path = 'GaAs/28_electrons_line/KPOINTS'
    cube_path = "GaAs/nscf-uniform/fort.123"
    #
    #### PbTe
    compound = 'PbTe'

    #### InP
    compound = 'InP_mp-20351'


    vrun_path = '{}/vasprun.xml'.format(compound)
    kpoints_path = '{}/KPOINTS'.format(compound)
    cube_path = '{}/fort.123'.format(compound)


    vrun = Vasprun(os.path.join(test_dir, vrun_path))
    bs = vrun.get_band_structure(
        kpoints_filename=os.path.join(test_dir, kpoints_path), line_mode=True)

    rec_matrix = vrun.final_structure.lattice.reciprocal_lattice.matrix
    dir_matrix = vrun.final_structure.lattice.matrix

    # all_ks = bs.get_sym_eq_kpoints(kpoint=np.dot(dir_matrix, np.array([0.1, 0.3, 0.2])), cartesian=True)
    # print([norm(k) for k in all_ks])
    # # print([np.dot(dir_matrix, k) for k in all_ks])
    # # print([norm(np.dot(dir_matrix, k)) for k in all_ks])
    # quit()

    st = vrun.final_structure
    print('reciprocal lattice matrix from Vasprun:\n{}'.format(rec_matrix))
    print('direct lattice matrix from Vasprun:\n{}'.format(dir_matrix))
    print('lattice constants: {}, {}, {}'.format(st.lattice.a, st.lattice.b, st.lattice.c))

    vbm_idx, _ = get_bindex_bspin(bs.get_vbm(), is_cbm=False)
    cbm_idx, _ = get_bindex_bspin(bs.get_cbm(), is_cbm=True)
    ibands = [vbm_idx+1, cbm_idx+1]

    coeff_file = os.path.join(test_dir, cube_path)
    start_time = time()
    retrieve_bs_boltztrap1(coeff_file=coeff_file, bs=bs, ibands=ibands, matrix=dir_matrix)

    # retrieve_bs_boltztrap1(coeff_file=SnSe2_coeff_file, bs=bs, ibands=[11, 12, 13, 14])
    print("Boltztrap1 total time: {}".format(time() - start_time))

    # start_time = time()

    # retrieve_bs_boltztrap2(os.path.join(test_dir, 'GaAs/nscf-uniform'), bs=bs, ibands=ibands)
    retrieve_bs_boltztrap2(os.path.join(test_dir, compound), bs=bs, ibands=ibands)

    # print("Boltztrap2 total time: {}".format(time() - start_time))

    # extrema = get_bs_extrema(bs, coeff_file=GaAs_coeff_file, nk_ibz=17, v_cut=1e4, min_normdiff=0.1, Ecut=0.5, nex_max=20)
