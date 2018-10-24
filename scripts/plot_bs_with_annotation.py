from amset.core import Amset
from amset.utils.constants import A_to_nm, comp_to_dirname
from amset.utils.detect_peaks import detect_peaks
from amset.utils.tools import get_bindex_bspin, interpolate_bs, \
    get_energy_args, kpts_to_first_BZ
from matminer import PlotlyFig
import numpy as np
import os
import pandas as pd
from pymatgen.io.vasp import Vasprun, Spin
from pymatgen.symmetry.bandstructure import HighSymmKpath
import plotly.graph_objs as go
from thermoelectrics_work.amset_examples.material_params import global_dopings, \
    global_temperatures

abs_dir = os.path.dirname(__file__)

"""
This script plots the interpolated band structure with kpoint coordinates as 
labels to help easily detect the important pockets. It can be used to visually 
test the output of get_bs_extrema function that is used in Amset to determine 
the band extrema that contribute to the transport.
"""

# INPUTS
COMPOUND = 'PbTe'
LINE_DENSITY = 30


if __name__ == '__main__':
    test_dir = os.path.join(abs_dir, '..', 'test_files')
    vrun_file = os.path.join(test_dir, comp_to_dirname[COMPOUND], 'vasprun.xml')
    coeff_file = os.path.join(test_dir, comp_to_dirname[COMPOUND], 'fort.123')
    vrun = Vasprun(vrun_file)
    formula = vrun.final_structure.composition.reduced_formula

    dopings = [1e20, -1e20]
    temperatures = global_temperatures[formula]
    word_map = {'p': 'vb0', 'n': 'cb0'}

    amset = Amset(calc_dir='.', material_params={'epsilon_s': 12.9},
              temperatures=temperatures, dopings=dopings,
              performance_params={'Ecut': 1.0}
                  )
    amset.read_vrun(vasprun_file=vrun_file)
    amset.update_cbm_vbm_dos(coeff_file)
    extrema = amset.find_all_important_points(coeff_file,
                                              nbelow_vbm=0,
                                              nabove_cbm=0,
                                              interpolation="boltztrap1",
                                              line_density=LINE_DENSITY
                                              )

    hsk = HighSymmKpath(vrun.final_structure)
    hs_kpoints , _ = hsk.get_kpoints(line_density=LINE_DENSITY)
    hs_kpoints = kpts_to_first_BZ(hs_kpoints)

    bs = vrun.get_band_structure()
    bsd = {}
    bsd['kpoints'] = hs_kpoints
    hs_kpoints = np.array(hs_kpoints)

    bsd['str_kpts'] = [str(k) for k in bsd['kpoints']]
    bsd['cartesian kpoints (1/nm)'] = [amset.get_cartesian_coords(k)/A_to_nm for k in bsd['kpoints']]
    bsd['normk'] = np.linalg.norm(bsd['cartesian kpoints (1/nm)'], axis=1)

    cbm_idx, cbm_spin = get_bindex_bspin(bs.get_cbm(), is_cbm=True)
    vbmd = bs.get_vbm()
    vbm_idx, vbm_spin = get_bindex_bspin(vbmd, is_cbm=False)
    vbm = vbmd['energy']
    print(np.array(bs.bands[Spin.up]).shape)

    interp_params = get_energy_args(coeff_file, ibands=[vbm_idx+1, vbm_idx+2])
    # bsd['vb0'] = bs.bands[vbm_spin][vbm_idx] - vbm
    bsd['vb0'], _, _ = interpolate_bs(bsd['kpoints'], interp_params, iband=0,
                           method="boltztrap1", scissor=0.0, matrix=vrun.final_structure.lattice.matrix, n_jobs=-1)
    vbm = max(bsd['vb0'])
    bsd['vb0'] -= vbm

    # bsd['cb0'] = bs.bands[cbm_spin][cbm_idx] - vbm
    bsd['cb0'], _, _ = interpolate_bs(bsd['kpoints'], interp_params, iband=1,
                           method="boltztrap1", scissor=0.0, matrix=vrun.final_structure.lattice.matrix, n_jobs=-1)
    bsd['cb0'] -= vbm
    cbm = min(bsd['cb0'])

    bs_df = pd.DataFrame.from_dict(bsd)

    pf = PlotlyFig(bs_df, x_title='index', y_title='Energy (eV)',
                   filename='interpolated_line-mode')
    plt = pf.xy([(bs_df.index, 'vb0'), (bs_df.index, 'cb0')], labels='str_kpts', return_plot=True)

    extrema_data_x = []
    extrema_data_y = []
    extrema_data_labels = []

    initial_extrema_data_x = []
    initial_extrema_data_y = []
    initial_extrema_data_labels = []

    for iband, tp in enumerate(['p', 'n']):
        energies, _, _ = interpolate_bs(extrema[tp], interp_params, iband=iband,
                                          method="boltztrap1", scissor=0.0,
                                          matrix=vrun.final_structure.lattice.matrix,
                                          n_jobs=-1)
        for k in extrema[tp]:
            idx = np.argmin([np.linalg.norm(hs_kpoints-k, axis=1)])
            if np.linalg.norm(hs_kpoints[idx]-k) < 0.05:
                extrema_data_x.append(idx)
        extrema_data_labels += [str(k.tolist()) for k in extrema[tp]]
        extrema_data_y += list(energies - vbm)

        intial_extrema_idx = detect_peaks(bs_df[word_map[tp]], mph=None, mpd=1,
                                   valley=iband == 1)
        initial_extrema_data_x += list(intial_extrema_idx)
        initial_extrema_data_y += (bs_df[word_map[tp]][intial_extrema_idx]-vbm).tolist()
        initial_extrema_data_labels += [str(k.tolist()) for k in hs_kpoints[intial_extrema_idx]]

    for idx in bs_df.index:
        for tp in ['p', 'n']:
            if bs_df['str_kpts'][idx] in extrema[tp]:
                extrema_data_x.append(idx)
                extrema_data_y.append(bs_df[word_map[tp]][idx])
                extrema_data_labels.append(bs_df['str_kpts'][idx])


    plt['data'].append(
    go.Scatter(
        x = initial_extrema_data_x,
        y = initial_extrema_data_y,
        mode = 'markers',
        name = 'initial extrema',
        text = initial_extrema_data_labels,
        marker = {'size': 15, 'symbol': 'diamond-open', 'color': 'rgb(0, 255, 0)',
                  'line': {'width': 3}}
    )
    )

    plt['data'].append(
    go.Scatter(
        x = extrema_data_x,
        y = extrema_data_y,
        mode = 'markers',
        name = 'selected extrema',
        text = extrema_data_labels,
        marker = {'size': 20, 'symbol': 'circle-open', 'color': 'rgb(255, 0, 0)',
                  'line': {'width': 4}}
    )
    )

    pf.create_plot(plt)