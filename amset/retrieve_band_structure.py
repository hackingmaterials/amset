from pymatgen import MPRester
from pylab import plot,show, scatter
import numpy as np
from analytical_band_from_BZT import get_energy
from pymatgen import Spin
from tools import get_energy_args

api = MPRester("fDJKEZpxSyvsXdCt")


def retrieve_bs(coeff_file, bs, ibands):
    # sp=bs.bands.keys()[0]
    engre, nwave, nsym, nstv, vec, vec2, out_vec2, br_dir = get_energy_args(coeff_file, ibands)

    #you can use a for loop along a certain list of k-points.
    for i, iband in enumerate(ibands):
        en = []
        sym_line_kpoints = [k.frac_coords for k in bs.kpoints]
        for kpt in sym_line_kpoints:
            # cbm = False
            cbm = True
            e = get_energy(kpt, engre[i], nwave, nsym, nstv, vec, vec2=vec2, out_vec2=out_vec2, br_dir=br_dir, cbm=cbm)
            en.append(e*13.605)

        # plot(np.array(bs.bands[sp])[iband-1,:].T-bs.efermi) # from MP
        # plot(np.array(bs.bands[sp])[iband-2,:].T-bs.efermi) # from MP
        # plot(np.array(bs.bands[sp])[iband-3,:].T-bs.efermi) # from MP
        plot(en, color='b')                                            # interpolated by BoltzTraP
    show()

if __name__ == "__main__":
    # user inputs
    PbTe_id = 'mp-19717' # valence_idx = 9
    Si_id = 'mp-149' # valence_idx = 4
    GaAs_id = 'mp-2534' # valence_idx = 14
    SnSe2_id = "mp-665"

    bs = api.get_bandstructure_by_material_id(SnSe2_id)
    vbm_idx = bs.get_vbm()['band_index'][Spin.up][0]
    ibands = [1, 2] # in this notation, 0 is the last valence band
    ibands = [i+vbm_idx+1 for i in ibands]


    PbTe_coeff_file = '../test_files/PbTe/fort.123'
    Si_coeff_file = "../test_files/Si/Si_fort.123"
#    GaAs_coeff_file = "../test_files/GaAs/fort.123_GaAs_sym_23x23x23"
    GaAs_coeff_file = "../test_files/GaAs/fort.123_GaAs_1099kp"
    # SnSe2_coeff_file = "/Users/alirezafaghaninia/Dropbox/Berkeley_Lab_Work/Yanzhong_Pei/SnSe2/boltztrap_vdw_dense/boltztrap/fort.123"
    # SnSe2_coeff_file = "/Users/alirezafaghaninia/Documents/boltztrap_examples/SnSe2/boltztrap_vdw_soc/boltztrap/fort.123"
    # SnSe2_coeff_file = "/Users/alirezafaghaninia/Documents/boltztrap_examples/SnSe2/boltztrap_vdw_better_geom_dense/boltztrap/fort.123"

    # retrieve_bs(coeff_file=PbTe_coeff_file, bs=bs, ibands=ibands)
    retrieve_bs(coeff_file=GaAs_coeff_file, bs=bs, ibands=ibands)
    # retrieve_bs(coeff_file=SnSe2_coeff_file, bs=bs, ibands=[11, 12, 13, 14])
    # retrieve_bs(coeff_file=SnSe2_coeff_file, bs=bs, ibands=[24, 25, 26, 27])
