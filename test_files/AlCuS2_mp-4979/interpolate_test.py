# coding: utf-8
import numpy as np
from amset.utils.band_interpolation import get_energy_args, interpolate_bs
from pymatgen.io.vasp.outputs import Vasprun

# includes bands 26-36
interp_params = get_energy_args("fort.123", 'A')
vr = Vasprun('vasprun.xml')
vr.actual_kpoints
kpts = np.array(vr.actual_kpoints)

energies, velocities, effective_masses = interpolate_bs(
    kpts, interp_params, 0, matrix=vr.final_structure.lattice.matrix,
    return_mass=True)

