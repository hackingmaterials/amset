import numpy as np
from amset.interpolate import BoltzTraP2Interpolater
from amset.utils.constants import k_B, A_to_nm, hbar, e
from pymatgen import Spin
from pymatgen.io.vasp import Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.interpolate import griddata

deformation_potential = 9.
elastic_constant = 98.
temperature = 300
symprec = 0.01
vasprun_file = "vasprun.xml"
kpoint_mesh = (30, 30, 30)
ecut = 1.5

vasprun = Vasprun(vasprun_file, parse_projected_eigen=True)

band_structure = vasprun.get_band_structure()
num_electrons = vasprun.parameters['NELECT']

interpolater = BoltzTraP2Interpolater(band_structure, num_electrons)

sga = SpacegroupAnalyzer(band_structure.structure, symprec=symprec)
mesh_data = np.array(sga.get_ir_reciprocal_mesh(kpoint_mesh))
kpoints = np.asarray(list(map(list, mesh_data[:, 0])))
weights = mesh_data[:, 1]
normalization_factor = 1

# will require adjusting for scissor operators
if band_structure.is_metal():
    min_energy = band_structure.efermi - ecut
    max_energy = band_structure.efermi + ecut
else:
    min_energy = band_structure.get_vbm()['energy'] - ecut
    max_energy = band_structure.get_cbm()['energy'] + ecut

# fix for spin-polarization
bands_in_range = np.any((band_structure.bands[Spin.up] > min_energy) &
                        (band_structure.bands[Spin.up] < max_energy), axis=1)
iband = np.where(bands_in_range)[0]
energies, velocities = interpolater.get_energies(kpoints, iband=iband)

# interpolate orbital projections
projections = band_structure.projections[Spin.up]
orig_kpoints = np.array([k.frac_coords for k in band_structure.kpoints])
orig_cart_kpoints = np.dot(band_structure.structure.lattice.reciprocal_lattice.
                           matrix, orig_kpoints) / A_to_nm
new_cart_kpoints = np.dot(band_structure.structure.lattice.reciprocal_lattice.
                          matrix, kpoints) / A_to_nm


def get_projections_for_band(band_index):
    band_proj = projections[band_index]
    s_orbital = np.sum(band_proj, axis=2)[:, 0]

    if band_proj.shape[1] > 5:
        # lm decomposed projections therefore sum across px, py, and pz
        p_orbital = np.sum(np.sum(band_proj, axis=2)[:, 1:4], axis=1)
    else:
        p_orbital = np.sum(band_proj, axis=2)[:, 1]

    return s_orbital, p_orbital


s_contribs = []
p_contribs = []
for band in iband:
    s_proj, p_proj = get_projections_for_band(band)
    s_interp = griddata(points=orig_cart_kpoints, values=s_proj,
                        xi=new_cart_kpoints, method='nearest')
    p_interp = griddata(points=orig_cart_kpoints, values=p_proj,
                        xi=new_cart_kpoints, method='nearest')
    s_contribs.append(s_interp)
    p_contribs.append(p_interp)

s_contribs = np.array(s_contribs)
p_contribs = np.array(p_contribs)

# calculate ACD scattering
norm_kpoints = np.linalg.norm(new_cart_kpoints, axis=1)
acd_rate = ((k_B * temperature * deformation_potential ** 2 * norm_kpoints ** 2)
            / (3 * np.pi * hbar ** 2 * elastic_constant * 1e9 * velocities)
            * (3 - 8 * p_contribs ** 2 + 6 * p_contribs ** 4) * e * 1e20)

lifetimes = 1 / acd_rate
print(lifetimes.shape)
