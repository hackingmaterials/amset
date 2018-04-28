from amset.core import AMSET
from amset.utils.tools import get_bindex_bspin
from matminer import PlotlyFig
import numpy as np
import os
import pandas as pd
from pymatgen.io.vasp import Vasprun, Spin

dir = os.path.dirname(__file__)

"""
This script helps plot the band structure with kpoint coordinates as labels to
help easily detect the important pockets
"""

vrun_file = os.path.join(dir, "../test_files/GaAs/28_electrons_line/vasprun.xml")


amset = AMSET(calc_dir='.', material_params={'epsilon_s': 12.9})
amset.read_vrun(vasprun_file=vrun_file)
# extrema = amset.get_bs_extrema(bs=self.GaAs_vrun.get_band_structure(),
#             coeff_file=self.GaAs_cube, nbelow_vbm=0, nabove_cbm=0)




vrun = Vasprun(vrun_file)
bs = vrun.get_band_structure()
bsd = {}
bsd['kpoints'] = [k.frac_coords for k in bs.kpoints]
bsd['str_kpts'] = [str(k) for k in bsd['kpoints']]
bsd['cartesian kpoints'] = [amset.get_cartesian_coords(k) for k in bsd['kpoints']]
bsd['normk'] = np.linalg.norm(bsd['cartesian kpoints'], axis=1)

cbm_idx, cbm_spin = get_bindex_bspin(bs.get_cbm(), is_cbm=True)
vbmd = bs.get_vbm()
vbm_idx, vbm_spin = get_bindex_bspin(vbmd, is_cbm=False)
vbm = vbmd['energy']
print(np.array(bs.bands[Spin.up]).shape)

bsd['vb0'] = bs.bands[vbm_spin][vbm_idx] - vbm
bsd['cb0'] = bs.bands[cbm_spin][cbm_idx] - vbm

bs_df = pd.DataFrame.from_dict(bsd)

pf = PlotlyFig(bs_df, x_title='index', y_title='Energy (eV)')
pf.xy([(bs_df.index, 'vb0'), (bs_df.index, 'cb0')], labels='str_kpts')
# pf.xy([(bs_df.index, 'vb0'), (bs_df.index, 'cb0')], labels='normk')

