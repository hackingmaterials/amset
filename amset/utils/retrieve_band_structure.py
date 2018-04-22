from time import time

from amset.utils.pymatgen_loader_for_bzt2 import PymatgenLoader
from pymatgen import MPRester
import numpy as np
import os
from matminer import PlotlyFig
from BoltzTraP2 import sphere, fite
from pymatgen.io.vasp import Vasprun
from amset.utils.tools import get_energy_args, get_bindex_bspin, norm, \
    interpolate_bs

api = MPRester("fDJKEZpxSyvsXdCt")


def retrieve_bs_boltztrap1(coeff_file, bs, ibands, matrix=None):
    interp_params = get_energy_args(coeff_file, ibands)
    pf = PlotlyFig(filename='plots/Energy-bt1')
    plot_data =[]
    v_data = []
    mass_data = []
    trace_names = []
    Eref = 0.0
    for i, iband in enumerate(ibands):
        sym_line_kpoints = [k.frac_coords for k in bs.kpoints]
        en, vel, masses = interpolate_bs(sym_line_kpoints, interp_params, iband=i,
                       method="boltztrap1", scissor=0.0, matrix=matrix, n_jobs=-1)
        vel = np.linalg.norm(vel, axis=1)
        masses = [mass.trace()/3.0 for mass in masses]
        if i==0:
            Eref = max(en)
        en = [E - Eref for E in en]
        plot_data.append((np.linspace(0, len(en)), en))
        v_data.append((en, vel))
        mass_data.append((en, masses))
        trace_names.append('band {}'.format(iband))
    pf.xy(plot_data, names=[n for n in trace_names])
    pf2 = PlotlyFig(filename='plots/Velocity-bt1')
    pf2.xy(v_data, names=[n for n in trace_names])
    pf3 = PlotlyFig(filename='plots/mass-bt1')
    pf3.xy(mass_data, names=[n for n in trace_names])


def retrieve_bs_boltztrap2(vrun_path, bs, ibands):
    pf = PlotlyFig(filename='plots/Energy-bt2')
    sym_line_kpoints = [k.frac_coords for k in bs.kpoints]
    # vrun = Vasprun(os.path.join(vrun_path, 'vasprun.xml'))
    # bz_data = PMG_Vasprun_Loader(vrun)
    bz_data = PymatgenLoader.from_files(os.path.join(vrun_path, 'vasprun.xml'))
    equivalences = sphere.get_equivalences(bz_data.atoms, len(bz_data.kpoints) * 10)
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
    pf.xy(plot_data, names=[n for n in names])
    pf2 = PlotlyFig(filename='plots/Velocity-bt2')
    pf2.xy(v_data, names=[n for n in names])
    pf3 = PlotlyFig(filename='plots/mass-bt2')
    pf3.xy(mass_data, names=[n for n in names])


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

    start_time = time()
    # retrieve_bs_boltztrap1(coeff_file=PbTe_coeff_file, bs=bs, ibands=ibands)
    # retrieve_bs_boltztrap1(coeff_file=Si_coeff_file, bs=Si_bs, ibands=ibands, cbm=True)

    retrieve_bs_boltztrap1(coeff_file=GaAs_coeff_file, bs=bs, ibands=ibands, cbm=True, matrix=dir_matrix)

    # retrieve_bs_boltztrap1(coeff_file=SnSe2_coeff_file, bs=bs, ibands=[11, 12, 13, 14])
    print("Boltztrap1 total time: {}".format(time() - start_time))

    # start_time = time()

    # retrieve_bs_boltztrap2(os.path.join(test_dir, 'GaAs'), ibands=ibands)
    # retrieve_bs_boltztrap2(os.path.join(test_dir, 'ZnS_391_vrun'), ibands=ibands)
    # retrieve_bs_boltztrap2(os.path.join(DIR, '../../../BoltzTraP2-18.1.2/data/Si.vasp'), ibands=[3, 4])
    # retrieve_bs_boltztrap2(root, ibands=[2, 3])
    # retrieve_bs_boltztrap2(InP_root, bs=bs, ibands=ibands)
    retrieve_bs_boltztrap2(os.path.join(test_dir, 'GaAs/nscf-uniform'), bs=bs, ibands=ibands)

    # print("Boltztrap2 total time: {}".format(time() - start_time))

    # extrema = get_bs_extrema(bs, coeff_file=GaAs_coeff_file, nk_ibz=17, v_cut=1e4, min_normdiff=0.1, Ecut=0.5, nex_max=20)
