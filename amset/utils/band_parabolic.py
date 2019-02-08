
import numpy as np

from amset.utils.constants import hbar, m_e, e, A_to_nm, pi
from amset.utils.general import norm
from matplotlib.pylab import show, scatter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

"""
Methods for generating parabolic band structure and density of states.
"""


def get_parabolic_energy(kpt, parabolic_bands, type, ib=0, bandgap=1, all_values = False):
    """
    Args:
        kpt (list): coordinates of a given k-point in the actual cartesian coordinates and NOT fractional coordinates
        parabolic_bands [[lists]]: each member of the first list represents a band: in each band a list of lists of lists
            should contain a list of two-member lists: the two members are: the coordinates of extrema k-point and its
            symmetrically equivalent points and another two members list of
            [first member: energy offset from the main extremum (i.e. CBM/VBM), second member: the effective mass]
            example parabolic_bands = [[ [[[0.5, 0.5, 0.5]], [0, 0.1]], [[[0, 0, 0]], [0.5, 0.2]]]] represents a
            band structure with a single band; this parabolic band (hbar**2k**2/2m*) at point X
            where k is norm(k-[0.5,0.5,0.5]) if the k-point is closer to X than Gamma. Additionally, this band has
            another extremum at an energy level 0.5 eV above the first/main extremum/CBM. If type is "p" the band structure
            would be a mirror image. VBM is always set to 0.0 eV and the CBM is set to the bandgap

        type (str): "n" or "p"
        ib (int): the band index, 0 is for the first band and maximum allowed value is len(parabolic_bands)-1
        bandgap (float): the targetted band gap of the band structure
    Returns:
    """
    # The sign of energy from type; e.g. p-type energies are negative (VBM=0.0)
    kpt = np.array(kpt)
    sgn = (-1)**(["p", "n"].index(type)+1)
    band_shapes = parabolic_bands[ib]
    min_kdist = 1e32
    allE = []
    for ks, c in band_shapes:
        #ks: all symmetrically equivalent k-points to the extremum k in 1st BZ
        for k in ks:
            distance = norm(kpt-k)
            allE.append(bandgap * ["p", "n"].index(type) + sgn * (c[0] + \
                    hbar ** 2 * (distance) ** 2 / (2 * m_e * c[1]) * e * 1e18))
            if distance < min_kdist:
                min_kdist = distance
                coefficients = c
    eff_m = coefficients[1]
    energy = sgn * bandgap/2.0
    energy += sgn*(coefficients[0] + hbar**2 * min_kdist**2 / (2*m_e*eff_m) * e*1e18) # last part is unit conv. to eV
    v = hbar*min_kdist/(m_e*eff_m) *1e11*e #last part is unit conversion to cm/s
    if not all_values:
        return energy, np.array([v, v, v]), sgn*np.array(
            [[eff_m, 0.0, 0.0], [0.0, eff_m, 0.0], [0.0, 0.0, eff_m]])
    else:
        return allE, min_kdist


def get_dos_from_parabolic_bands(st, reclat_matrix, mesh, e_min, e_max, e_points, parabolic_bands, bandgap, width=0.1, SPB_DOS=False, all_values=False):
    """
    Args:
    st:       pmg object of crystal structure to calculate symmetries
    mesh:     list of integers defining the k-mesh on which the dos is required
    e_min:    starting energy (eV) of dos
    e_max:    ending energy (eV) of dos
    e_points: number of points of the get_dos
    width:    width in eV of the gaussians generated for each energy
    Returns:
    e_mesh:   energies in eV od the DOS
    dos:      density of states for each energy in e_mesh
    """
    height = 1.0 / (width * np.sqrt(2 * np.pi))
    e_mesh, step = np.linspace(e_min, e_max,num=e_points, endpoint=True, retstep=True)
    e_range = len(e_mesh)
    ir_kpts_n_weights = SpacegroupAnalyzer(st).get_ir_reciprocal_mesh(mesh)
    ir_kpts = [k[0] for k in ir_kpts_n_weights]
    weights = [k[1] for k in ir_kpts_n_weights]

    ir_kpts = [reclat_matrix.get_cartesian_coords(k)/A_to_nm for k in ir_kpts]

    w_sum = float(sum(weights))
    dos = np.zeros(e_range)

    if SPB_DOS:
        volume = st.volume
        for band in parabolic_bands:
            for valley in band:
                offset = valley[-1][0] # each valley has a list of k-points (valley[0]) and [offset, m*] (valley[1]) info
                m_eff = valley[-1][1]
                degeneracy = len(valley[0])
                for ie, energy in enumerate(e_mesh):
                    dos_temp = volume/(2*pi**2)*(2*m_e*m_eff/hbar**2)**1.5 * 1e-30/e**1.5
                    if energy <= -bandgap/2.0-offset:
                        # dos_temp *= (-energy-offset)**0.5
                        dos_temp *= (-energy+bandgap/2.0+offset)**0.5
                    elif energy >=bandgap/2.0+offset:
                        # dos_temp *= (energy-bandgap-offset)**0.5
                        dos_temp *= (energy-bandgap/2.0-offset)**0.5
                    else:
                        dos_temp = 0
                    dos[ie] += dos_temp * degeneracy
    else:
        all_energies = []
        all_ks = []
        for kpt,w in zip(ir_kpts,weights):
            for tp in ["n", "p"]:
                for ib in range(len(parabolic_bands)):
                    if all_values:
                        energy_list, k_dist = get_parabolic_energy(kpt, parabolic_bands, tp, ib=ib, bandgap=bandgap,
                                                           all_values=all_values)
                        all_energies += energy_list
                        all_ks += [k_dist]*len(energy_list)
                        for energy in energy_list:
                            g = height * np.exp(-((e_mesh - energy) / width) ** 2 / 2.)
                            dos += w/w_sum * g
                    else:
                        energy, v, m_eff = get_parabolic_energy(kpt, parabolic_bands, tp, ib=ib, bandgap=bandgap, all_values=all_values)
                        g = height * np.exp(-((e_mesh - energy) / width) ** 2 / 2.)
                        dos += w/w_sum * g
        if all_values:
            scatter(all_ks, all_energies)
            show()
    return e_mesh,dos

