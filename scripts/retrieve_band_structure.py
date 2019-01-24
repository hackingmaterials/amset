import numpy as np
import os
from time import time
from amset.utils.constants import comp_to_dirname
from amset.utils.pymatgen_loader_for_bzt2 import PymatgenLoader
from matminer import PlotlyFig
from BoltzTraP2 import sphere, fite
from pymatgen.io.vasp import Vasprun
from amset.utils.band_interpolation import get_energy_args, interpolate_bs
from amset.utils.band_structure import get_bindex_bspin

"""
This script is to compare the energy, velocity and effective mass calculated 
from the band structures interpolated via BoltzTraP1 vs. BoltzTraP2 to check
their consistency. If KPOINTS file is fed to the Vasprun.get_bandstructure() 
method, one can compare the band structures in line-mode which is more visually 
appealing.
"""

def retrieve_bs_boltztrap1(coeff_file, bs, ibands, matrix=None):
    interp_params = get_energy_args(coeff_file, ibands)
    pf = PlotlyFig(filename='Energy-bt1')
    plot_data =[]
    v_data = []
    mass_data = []
    trace_names = []
    Eref = 0.0
    vels = {iband: [] for iband in ibands}
    sym_line_kpoints = []

    for i, iband in enumerate(ibands):
        sym_line_kpoints = [k.frac_coords for k in bs.kpoints]

        en, vel, masses = interpolate_bs(sym_line_kpoints, interp_params, iband=i,
                       method="boltztrap1", scissor=0.0, matrix=matrix, n_jobs=-1)
        vel = np.linalg.norm(vel, axis=1)
        masses = [mass.trace()/3.0 for mass in masses]
        if i==0:
            Eref = max(en)
        en = [E - Eref for E in en]
        plot_data.append((list(range(len(en))), en))
        v_data.append((en, vel))
        mass_data.append((en, masses))
        trace_names.append('band {}'.format(iband))


    pf.xy(plot_data, names=[n for n in trace_names], labels=[sym_line_kpoints])
    pf2 = PlotlyFig(filename='Velocity-bt1')
    pf2.xy(v_data, names=[n for n in trace_names])
    pf3 = PlotlyFig(filename='mass-bt1')
    pf3.xy(mass_data, names=[n for n in trace_names])


def retrieve_bs_boltztrap2(vrun, bs, ibands, matrix=None):
    pf = PlotlyFig(filename='Energy-bt2')
    sym_line_kpoints = [k.frac_coords for k in bs.kpoints]
    bz_data = PymatgenLoader.from_vasprun(vrun)
    equivalences = sphere.get_equivalences(atoms=bz_data.atoms, nkpt=len(bz_data.kpoints) * 5, magmom=None)
    lattvec = bz_data.get_lattvec()
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
                       method="boltztrap2", matrix=matrix)
                        # method = "boltztrap2", matrix = lattvec * 0.529177)
        if ith==0:
            eref = np.max(en)
        en -= eref
        plot_data.append((list(range(len(en))), en ))
        v_data.append((en, np.linalg.norm(vel, axis=1)))
        mass_data.append((en, [mass.trace()/3.0 for mass in masses]))
        names.append('band {}'.format(iband+1))
    pf.xy(plot_data, names=[n for n in names])
    pf2 = PlotlyFig(filename='Velocity-bt2')
    pf2.xy(v_data, names=[n for n in names])
    pf3 = PlotlyFig(filename='mass-bt2')
    pf3.xy(mass_data, names=[n for n in names])

if __name__ == "__main__":
    # user inputs
    COMPOUND = 'GaAs' # You can try: GaAs, Si, PbTe, InP, AlCuS2, In2O3
    DIR = os.path.dirname(__file__)
    test_dir = os.path.join(DIR, '../test_files')
    vruns = {c: Vasprun(os.path.join(test_dir, comp_to_dirname[c],
                                     'vasprun.xml')) for c in comp_to_dirname}
    coeff_files = {c: os.path.join(test_dir, comp_to_dirname[c],
                                   'fort.123') for c in comp_to_dirname}

    vrun = vruns[COMPOUND]
    cube_path = coeff_files[COMPOUND]

    # bs = vrun.get_band_structure(
    #     kpoints_filename=os.path.join(test_dir, kpoints_path), line_mode=True)
    bs = vrun.get_band_structure()

    rec_matrix = vrun.final_structure.lattice.reciprocal_lattice.matrix
    dir_matrix = vrun.final_structure.lattice.matrix

    st = vrun.final_structure

    vbm_idx, _ = get_bindex_bspin(bs.get_vbm(), is_cbm=False)
    cbm_idx, _ = get_bindex_bspin(bs.get_cbm(), is_cbm=True)
    ibands = [vbm_idx+1, cbm_idx+1]

    coeff_file = os.path.join(test_dir, cube_path)
    start_time = time()
    retrieve_bs_boltztrap1(coeff_file=coeff_file, bs=bs, ibands=ibands, matrix=dir_matrix)

    retrieve_bs_boltztrap2(vrun, bs=bs, ibands=ibands, matrix=dir_matrix)
