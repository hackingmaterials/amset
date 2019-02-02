# coding: utf-8
import numpy as np
from amset.utils.band_interpolation import get_energy_args, interpolate_bs
from pymatgen.io.vasp.outputs import Vasprun
from amset.utils.analytical_band_from_bzt1 import Analytical_bands

# includes bands 26-36
interp_params = get_energy_args("fort.123", 'A')
vr = Vasprun('vasprun.xml')
vr.actual_kpoints
kpts = np.array(vr.actual_kpoints)

energies, velocities, effective_masses = interpolate_bs(
    kpts, interp_params, 0, matrix=vr.final_structure.lattice.matrix,
    return_mass=True)

ab = Analytical_bands(coeff_file='fort.123')
bs = vr.get_band_structure()
emin = -10
emax = 10
emesh, dos, dos_nbands, bmin = ab.get_dos_from_scratch(
    vr.final_structure, [10, 10, 10], emin, emax,
    int(round((emax-emin)/0.001)), 0.075, 0., 32)
