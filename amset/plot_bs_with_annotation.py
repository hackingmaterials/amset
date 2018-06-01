from amset.core import AMSET
from amset.utils.constants import A_to_nm
from amset.utils.tools import get_bindex_bspin, interpolate_bs, get_energy_args
from matminer import PlotlyFig
import numpy as np
import os
import pandas as pd
from pymatgen.io.vasp import Vasprun, Spin
from pymatgen.symmetry.bandstructure import HighSymmKpath
from scipy.signal import find_peaks_cwt

abs_dir = os.path.dirname(__file__)

"""
This script helps plot the band structure with kpoint coordinates as labels to
help easily detect the important pockets. It can be used to visually test
the output of get_bs_extrema function.
"""

# vrun_file = os.path.join(abs_dir, "../test_files/GaAs/28_electrons_line/vasprun.xml")
# vrun_file = os.path.join(abs_dir, "../test_files/GaAs/nscf-uniform/vasprun.xml")
# coeff_file = os.path.join(abs_dir, "../test_files/GaAs/nscf-uniform/fort.123")

vrun_file = os.path.join(abs_dir, "../test_files/Si/vasprun.xml")
coeff_file = os.path.join(abs_dir, "../test_files/Si/Si_fort.123")

vrun_file = os.path.join(abs_dir, "../test_files/PbTe/vasprun.xml")
coeff_file = os.path.join(abs_dir, "../test_files/PbTe/fort.123")

amset = AMSET(calc_dir='.', material_params={'epsilon_s': 12.9})
amset.read_vrun(vasprun_file=vrun_file)
# extrema = amset.get_bs_extrema(bs=self.GaAs_vrun.get_band_structure(),
#             coeff_file=self.GaAs_cube, nbelow_vbm=0, nabove_cbm=0)


vrun = Vasprun(vrun_file)
hsk = HighSymmKpath(vrun.final_structure)
hs_kpoints , _ = hsk.get_kpoints(line_density=30)

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

# extrema_idx = find_peaks_cwt(bsd['cb0'], widths=np.arange(0.03, 0.2, 0.02))

bs_df = pd.DataFrame.from_dict(bsd)

pf = PlotlyFig(bs_df, x_title='index', y_title='Energy (eV)')
pf.xy([(bs_df.index, 'vb0'), (bs_df.index, 'cb0')], labels='str_kpts')
# pf.xy([(bs_df.index, 'vb0'), (bs_df.index, 'cb0')], labels='normk')

