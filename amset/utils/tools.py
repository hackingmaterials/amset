import logging
import numpy as np
import scipy
from scipy.optimize import basinhopping

from amset.utils.analytical_band_from_BZT import Analytical_bands, outer, get_energy
from amset.utils.constants import hbar, m_e, Ry_to_eV, A_to_m, m_to_cm, A_to_nm, e, k_B,\
                        epsilon_0, default_small_E, dTdz, sq3
from math import pi, log

from pymatgen import Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath


class AmsetError(Exception):
    """
    Exception class for AMSET. Raised when AMSET gives an error.
    """

    def __init__(self, msg):
        self.msg = msg
        logging.error(self.msg)

    def __str__(self):
        return "AmsetError : " + self.msg

def remove_from_grid(grid, grid_rm_list):
    """deletes dictionaries storing properties that are no longer needed from
    a given grid (i.e. kgrid or egrid)"""
    for tp in ["n", "p"]:
        for rm in grid_rm_list:
            try:
                del (grid[tp][rm])
            except:
                pass
    return grid


def norm(v):
    """method to quickly calculate the norm of a vector (v: 1x3 or 3x1) as numpy.linalg.norm is slower for this case"""
    return (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5


def grid_norm(grid):
    return (grid[:,:,:,0]**2 + grid[:,:,:,1]**2 + grid[:,:,:,2]**2) ** 0.5

def kpts_to_first_BZ(kpts):
    for i, k in enumerate(kpts):
        for alpha in range(3):
            if k[alpha] > 0.5:
                k[alpha] -= 1
            if k[alpha] < -0.5:
                k[alpha] += 1
        kpts[i] = k
    return kpts

def generate_k_mesh_axes(important_pts, kgrid_tp='coarse', one_list=True):
    points_1d = {dir: [] for dir in ['x', 'y', 'z']}

    for center in important_pts:
        for dim, dir in enumerate(['x', 'y', 'z']):
            points_1d[dir].append(center[dim])

            if not one_list:
                for step, nsteps in [[0.002, 2], [0.005, 4], [0.01, 4], [0.05, 2], [0.1, 5]]:
                    for i in range(nsteps - 1):
                        points_1d[dir].append(center[dim] - (i + 1) * step)
                        points_1d[dir].append(center[dim] + (i + 1) * step)

            else:
                if kgrid_tp == 'extremely fine':
                    mesh = [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.0045, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03,
                            0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.25]
                elif kgrid_tp == 'super fine':
                    mesh = [0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.01, 0.02, 0.03,
                            0.05, 0.07, 0.1, 0.15, 0.25]
                elif kgrid_tp == 'very fine':
                    mesh = [0.001, 0.002, 0.004, 0.007, 0.01, 0.02, 0.03,
                            0.05, 0.07, 0.1, 0.15, 0.25]
                elif kgrid_tp == 'fine':
                    mesh = [0.001, 0.004, 0.01, 0.02, 0.03,
                            0.05, 0.11, 0.25]
                elif kgrid_tp == 'coarse':
                    mesh = [0.001, 0.005, 0.01, 0.02, 0.05, 0.15]
                elif kgrid_tp == 'very coarse':
                    mesh = [0.001, 0.01]
                elif kgrid_tp == 'uniform':
                    mesh = np.linspace(0.001, 0.50, 15)
                elif kgrid_tp == 'test uniform':
                    mesh = np.linspace(0.001, 0.50, 7)
                elif kgrid_tp == 'test':
                    mesh = [0.001, 0.005, 0.01, 0.02, 0.03,
                            0.04, 0.05, 0.06, 0.07, 0.1, 0.15, 0.2, 0.25, 0.4, 0.5]
                else:
                    raise ValueError('Unsupported value for kgrid_tp: {}'.format(kgrid_tp))
                for step in mesh:
                    points_1d[dir].append(center[dim] + step)
                    points_1d[dir].append(center[dim] - step)
    print('included points in the mesh: {}'.format(points_1d))

    # ensure all points are in "first BZ" (parallelepiped)
    for dir in ['x', 'y', 'z']:
        for ik1d in range(len(points_1d[dir])):
            if points_1d[dir][ik1d] > 0.5:
                points_1d[dir][ik1d] -= 1
            if points_1d[dir][ik1d] < -0.5:
                points_1d[dir][ik1d] += 1

    # remove duplicates
    for dir in ['x', 'y', 'z']:
        points_1d[dir] = list(set(np.array(points_1d[dir]).round(decimals=14)))

    return points_1d


def create_grid(points_1d):
    for dir in ['x', 'y', 'z']:
        points_1d[dir].sort()
    grid = np.zeros((len(points_1d['x']), len(points_1d['y']), len(points_1d['z']), 3))
    for i, x in enumerate(points_1d['x']):
        for j, y in enumerate(points_1d['y']):
            for k, z in enumerate(points_1d['z']):
                grid[i, j, k, :] = np.array([x, y, z])
    return grid


def array_to_kgrid(grid):
    """
    Args:
        grid (np.array): 4d numpy array, where last dimension is vectors
            in a 3d grid specifying fractional position in BZ
    Returns:
        a list of [kx, ky, kz] k-point coordinates compatible with AMSET
    """
    kgrid = []
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for k in range(grid.shape[2]):
                kgrid.append(grid[i,j,k])
    return kgrid


def normalize_array(grid):
    N = grid.shape
    norm_grid = np.zeros(N)
    for i in range(N[0]):
        for j in range(N[1]):
            for k in range(N[2]):
                vec = grid[i, j, k]
                if norm(vec) == 0:
                    norm_grid[i, j, k] = [0, 0, 0]
                else:
                    norm_grid[i, j, k] = vec / norm(vec)
    return norm_grid


def f0(E, fermi, T):
    """returns the value of Fermi-Dirac at equilibrium for E (energy), fermi [level] and T (temperature)"""
    exponent = (E - fermi) / (k_B * T)
    if exponent > 40:
        return 0.0
    elif exponent < -40:
        return 1.0
    else:
        return 1 / (1 + np.exp(exponent))


def df0dE(E, fermi, T):
    """returns the energy derivative of the Fermi-Dirac equilibrium distribution"""
    exponent = (E - fermi) / (k_B * T)
    if exponent > 40 or exponent < -40:  # This is necessary so at too low numbers python doesn't return NaN
        return 0.0
    else:
        return -1 / (k_B * T) * np.exp((E - fermi) / (k_B * T)) / (1 + np.exp((E - fermi) / (k_B * T))) ** 2


def cos_angle(v1, v2):
    """
    returns cosine of the angle between two 3x1 or 1x3 vectors
    """
    norm_v1, norm_v2 = norm(v1), norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 1.0  # In case of the two points are the origin, we assume 0 degree; i.e. no scattering: 1-X==0
    else:
        return np.dot(v1, v2) / (norm_v1 * norm_v2)


def fermi_integral(order, fermi, T, initial_energy=0, wordy=False):
    """
    returns the Fermi integral (e.g. for calculating single parabolic band acoustic phonon mobility
    Args:
        order (int): the order of integral
        fermi (float): the actual Fermi level of the band structure (not relative to CBM/VBM):
        T (float): the temperature
        initial_energy (float): the actual CBM/VBM energy in eV
        wordy (bool): whether to print out the integrals or not
    """
    fermi = fermi - initial_energy
    integral = 0.
    nsteps = 100000.0
    # TODO: 1000000 works better (converges!) but for faster testing purposes we use larger steps
    # emesh = np.linspace(0.0, 30*k_B*T, nsteps) # We choose 20kBT instead of infinity as the fermi distribution will be 0
    emesh = np.linspace(0.0, 30 * k_B * T,
                        nsteps)  # We choose 20kBT instead of infinity as the fermi distribution will be 0
    dE = (emesh[-1] - emesh[0]) / (nsteps - 1.0)
    for E in emesh:
        integral += dE * (E / (k_B * T)) ** order / (1. + np.exp((E - fermi) / (k_B * T)))

    if wordy:
        print("order {} fermi integral at fermi={} and {} K".format(order, fermi, T))
        print(integral)
    return integral


def GB(x, eta):
    """Gaussian broadening. At very small eta values (e.g. 0.005 eV) this function goes to the dirac-delta of x.
    Args:
        x (float): the mean value of the nomral distribution
        eta (float): the standard deviation of the normal distribution
        """

    return 1 / np.pi * 1 / eta * np.exp(-(x / eta) ** 2)

    ## although both expressions conserve the final transport properties, the one below doesn't conserve the scat. rates
    # return np.exp(-(x/eta)**2)


def calculate_Sio_list(tp, c, T, ib, once_called, kgrid, cbm_vbm, epsilon_s, epsilon_inf):
    S_i_list = [0.0 for ik in kgrid[tp]["kpoints"][ib]]
    S_i_th_list = [0.0 for ik in kgrid[tp]["kpoints"][ib]]
    S_o_list = [0.0 for ik in kgrid[tp]["kpoints"][ib]]
    S_o_th_list = [0.0 for ik in kgrid[tp]["kpoints"][ib]]

    for ik in range(len(kgrid[tp]["kpoints"][ib])):
        S_i_list[ik], S_i_th_list[ik], S_o_list[ik], S_o_th_list[ik] = \
            calculate_Sio(tp, c, T, ib, ik, once_called, kgrid, cbm_vbm, epsilon_s, epsilon_inf)

    return [S_i_list, S_i_th_list, S_o_list, S_o_th_list]


def calculate_Sio(tp, c, T, ib, ik, once_called, kgrid, cbm_vbm, epsilon_s, epsilon_inf):
    """calculates and returns the in and out polar optical phonon inelastic scattering rates. This function
        is defined outside of the AMSET class to enable parallelization.
    Args:
        tp (str): the type of the bands; "n" for the conduction and "p" for the valence bands
        c (float): the carrier concentration
        T (float): the temperature
        ib (int): the band index
        ik (int): the k-point index
        once_called (bool): whether this function was once called hence S_o and S_o_th calculated once or not
        kgrid (dict): the main kgrid variable in AMSET (AMSET.kgrid)
        cbm_vbm (dict): the dict containing information regarding the cbm and vbm (from AMSET.cbm_vbm)
        epsilon_s (float): static dielectric constant
        epsilon_inf (float): high-frequency dielectric constant
        """
    S_i = [np.array([1e-32, 1e-32, 1e-32]), np.array([1e-32, 1e-32, 1e-32])]
    S_i_th = [np.array([1e-32, 1e-32, 1e-32]), np.array([1e-32, 1e-32, 1e-32])]
    S_o = [np.array([1e-32, 1e-32, 1e-32]), np.array([1e-32, 1e-32, 1e-32])]
    S_o_th = [np.array([1e-32, 1e-32, 1e-32]), np.array([1e-32, 1e-32, 1e-32])]

    # v = kgrid[tp]["norm(v)"][ib][ik] / sq3  # 3**0.5 is to treat each direction as 1D BS
    # v = kgrid[tp]["norm(v)"][ib][ik]  # 20180306: still not sure about /sq3 and whether it's necessary it has been /sq3 for a long time and is tested more
    k = kgrid[tp]["norm(k)"][ib][ik]
    a = kgrid[tp]["a"][ib][ik]
    c_ = kgrid[tp]["c"][ib][ik]
    f = kgrid[tp]["f"][c][T][ib][ik]
    f_th = kgrid[tp]["f_th"][c][T][ib][ik]
    N_POP = kgrid[tp]["N_POP"][c][T][ib][ik]

    for j, X_Epm in enumerate(["X_Eplus_ik", "X_Eminus_ik"]):
        # bypass k-points that cannot have k_plus or k_minus associated with them
        if tp == "n" and X_Epm == "X_Eminus_ik" and kgrid[tp]["energy"][ib][ik] - hbar * \
                kgrid[tp]["W_POP"][ib][ik] < cbm_vbm[tp]["energy"]:
            continue

        if tp == "p" and X_Epm == "X_Eplus_ik" and kgrid[tp]["energy"][ib][ik] + hbar * \
                kgrid[tp]["W_POP"][ib][ik] > cbm_vbm[tp]["energy"]:
            continue

        # TODO: see how does dividing by counted affects results, set to 1 to test: #20170614: in GaAs,
        # they are all equal anyway (at least among the ones checked)
        # ACTUALLY this is not true!! for each ik I get different S_i values at different k_prm

        counted = len(kgrid[tp][X_Epm][ib][ik])
        for X_ib_ik in kgrid[tp][X_Epm][ib][ik]:
            X, ib_pm, ik_pm = X_ib_ik
            k_pm = kgrid[tp]["norm(k)"][ib_pm][ik_pm]
            abs_kdiff = abs(k_pm - k)
            if abs_kdiff < 1e-4:
                counted -= 1
                continue
            # if abs(kgrid[tp]['energy'][ib_pm][ik_pm] - \
            #                kgrid[tp]['energy'][ib][ik]) < \
            #                         hbar * kgrid[tp]["W_POP"][ib][ik] / 2:
            #     counted -= 1
            #     continue

            g_pm = kgrid[tp]["g"][c][T][ib_pm][ik_pm]
            g_pm_th = kgrid[tp]["g_th"][c][T][ib_pm][ik_pm]
            v_pm = kgrid[tp]["norm(v)"][ib_pm][ik_pm] / sq3  # 3**0.5 is to treat each direction as 1D BS
            # v_pm = kgrid[tp]["norm(v)"][ib_pm][ik_pm] # 20180306: still not sure about /sq3 and whether it's necessary
            a_pm = kgrid[tp]["a"][ib_pm][ik_pm]
            c_pm = kgrid[tp]["c"][ib_pm][ik_pm]

            if tp == "n":
                f_pm = kgrid[tp]["f"][c][T][ib_pm][ik_pm]
                f_pm_th = kgrid[tp]["f_th"][c][T][ib_pm][ik_pm]
            else:
                f_pm = 1 - kgrid[tp]["f"][c][T][ib_pm][ik_pm]
                f_pm_th = 1 - kgrid[tp]["f_th"][c][T][ib_pm][ik_pm]


            A_pm = a * a_pm + c_ * c_pm * (k_pm ** 2 + k ** 2) / (2 * k_pm * k)
            beta_pm = (e ** 2 * kgrid[tp]["W_POP"][ib_pm][ik_pm]) / (4 * pi * hbar * v_pm) * \
                      (1 / (epsilon_inf * epsilon_0) - 1 / (epsilon_s * epsilon_0)) * 6.2415093e20

            if not once_called:
                lamb_opm = beta_pm * (
                    A_pm ** 2 * log((k_pm + k) / (abs_kdiff)) - A_pm * c_ * c_pm - a * a_pm * c_ * c_pm)
                # because in the scalar form k+ or k- is supposed to be unique, here we take average
                S_o[j] += (N_POP + j + (-1) ** j * f_pm) * lamb_opm
                S_o_th[j] += (N_POP + j + (-1) ** j * f_pm_th) * lamb_opm

            lamb_ipm = beta_pm * (
                (k_pm ** 2 + k ** 2) / (2 * k * k_pm) * \
                A_pm ** 2 * log((k_pm + k) / (abs_kdiff)) - A_pm ** 2 - c_ ** 2 * c_pm ** 2 / 3)
            S_i[j] += (N_POP + (1 - j) + (-1) ** (1 - j) * f) * lamb_ipm * g_pm
            S_i_th[j] += (N_POP + (1 - j) + (-1) ** (1 - j) * f_th) * lamb_ipm * g_pm_th

        if counted > 0:
            S_i[j] /= counted
            S_i_th[j] /= counted
            S_o[j] /= counted
            S_o_th[j] /= counted

    return [sum(S_i), sum(S_i_th), sum(S_o), sum(S_o_th)]


def get_closest_k(kpoint, ref_ks, return_diff=False, threshold = 0.001):
    """
    returns the list of difference between kpoints. If return_diff True, then
        for a given kpoint the minimum distance among distances with ref_ks is
        returned or just the reference kpoint that results if not return_diff
    Args:
        kpoint (1x3 array): the coordinates of the input k-point
        ref_ks ([1x3 array]): list of reference k-points from which the
            distance with initial_ks are calculated
        return_diff (bool): if True, the minimum distance is returned
    Returns (1x3 array):
    """
    # kpoint = np.array(kpoint) # not necessary?
    # ref_ks = np.array(ref_ks) # not necessary?
    # print('here inputs')
    # print(kpoint)
    # print(ref_ks)
    assert len(list(kpoint)) == 3
    assert len(list(ref_ks[0])) == 3
    assert isinstance(ref_ks[0][0], (float, list))
    norms = []
    for ki in ref_ks:
        norm_diff = norm(ki - kpoint)
        if norm_diff > threshold:
            norms.append(norm_diff)
        else:
            norms.append(1e10)
    min_dist_ik = np.array(norms).argmin()
    # min_dist_ik = np.array([norm(ki - kpoint) for ki in ref_ks]).argmin()
    # print('here min dist')
    # print(min_dist_ik)
    # print(kpoint)
    # print(ref_ks[min_dist_ik])
    if return_diff:
        return kpoint - ref_ks[min_dist_ik]
    else:
        return ref_ks[min_dist_ik]

def remove_duplicate_kpoints(kpts, dk=0.0001):
    """kpts (list of list): list of coordinates of electrons
     ALWAYS return either a list or ndarray: BE CONSISTENT with the input!!!

     Attention: it is better to call this method only once as calculating the norms takes time.
     """
    rm_list = []

    kdist = [norm(k) for k in kpts]
    ktuple = list(zip(kdist, kpts))
    ktuple.sort(key=lambda x: x[0])
    kpts = [tup[1] for tup in ktuple]

    i = 0
    while i < len(kpts) - 1:
        j = i
        while j < len(kpts) - 1 and ktuple[j + 1][0] - ktuple[i][0] < dk:

            # for i in range(len(kpts)-2):
            # if kpts[i][0] == kpts[i+1][0] and kpts[i][1] == kpts[i+1][1] and kpts[i][2] == kpts[i+1][2]:

            if (abs(kpts[i][0] - kpts[j + 1][0]) < dk or abs(kpts[i][0]) == abs(kpts[j + 1][0]) == 0.5) and \
                    (abs(kpts[i][1] - kpts[j + 1][1]) < dk or abs(kpts[i][1]) == abs(kpts[j + 1][1]) == 0.5) and \
                    (abs(kpts[i][2] - kpts[j + 1][2]) < dk or abs(kpts[i][2]) == abs(kpts[j + 1][2]) == 0.5):
                rm_list.append(j + 1)
            j += 1
        i += 1
    kpts = np.delete(kpts, rm_list, axis=0)
    kpts = list(kpts)
    return kpts


def find_fermi_SPB(cbm_vbm, c, T, tolerance=0.001, tolerance_loose=0.03, alpha=0.02, max_iter=1000):
    tp = get_tp(c)
    sgn = np.sign(c)
    m_eff = np.prod(cbm_vbm[tp]["eff_mass_xx"]) ** (1.0 / 3.0)
    c *= sgn
    initial_energy = cbm_vbm[tp]["energy"]
    fermi = initial_energy + 0.02
    iter = 0
    for iter in range(max_iter):
        calc_doping = 4 * pi * (2 * m_eff * m_e * k_B * T / hbar ** 2) ** 1.5 * fermi_integral(0.5, fermi, T,
                                                                                               initial_energy) * 1e-6 / e ** 1.5
        fermi += alpha * sgn * (calc_doping - c) / abs(c + calc_doping) * fermi
        relative_error = abs(calc_doping - c) / abs(c)
        if relative_error <= tolerance:
            # This here assumes that the SPB generator set the VBM to 0.0 and CBM=  gap + scissor
            if sgn < 0:
                return fermi
            else:
                return -(fermi - initial_energy)
    if relative_error > tolerance:
        raise ValueError("could NOT find a corresponding SPB fermi level after {} itenrations".format(max_iter))


def get_tp(c):
    """returns "n" for n-tp or negative carrier concentration or "p" (p-tp)."""
    if c < 0:
        return "n"
    elif c > 0:
        return "p"
    else:
        raise ValueError("The carrier concentration cannot be zero! AMSET stops now!")


def get_angle(v1, v2):
    x = cos_angle(v1,v2)
    if x < -1:
        x = -1
    elif x > 1:
        x = 1
    return np.arccos(x)


def sort_angles(vecs):
    sorted_vecs = []
    indexes = range(len(vecs))
    final_idx = []
    vecs = list(vecs)
    while len(vecs) > 1:
        angles = [get_angle(vecs[0], vecs[i]) for i in range(len(vecs))]
        sort_idx = np.argsort(angles)
        vecs = [vecs[i] for i in sort_idx]
        indexes = [indexes[i] for i in sort_idx]
        sorted_vecs.append(vecs.pop(0)) # reduntant step for the first element
        final_idx.append(indexes.pop(0))
    vecs.extend(sorted_vecs)
    indexes.extend(final_idx)
    return np.array(vecs), indexes


def rel_diff(num1, num2):
    diff = abs(num1 - num2)
    avg = (num1 + num2) / 2
    return diff / avg


def get_energy_args(coeff_file, ibands):
    """
    Args:
        coeff_file (str): the address to the cube (*.123) file
        ibands ([int]): list of band numbers to be calculated; note that the
            first band index is 1 not 0
    Returns (tuple): necessary inputs for calc_analytical_energy or get_energy
    """
    analytical_bands = Analytical_bands(coeff_file=coeff_file)
    logging.debug('ibands in get_energy_args: {}'.format(ibands))
    try:
        engre, latt_points, nwave, nsym, nsymop, symop, br_dir = \
            analytical_bands.get_engre(iband=ibands)
    except TypeError as e:
        raise AmsetError('try reducing Ecut to include fewer bands')

    nstv, vec, vec2 = analytical_bands.get_star_functions(
            latt_points, nsym, symop, nwave, br_dir=br_dir)
    out_vec2 = np.zeros((nwave, max(nstv), 3, 3))
    for nw in range(nwave):
        for i in range(nstv[nw]):
            out_vec2[nw, i] = outer(vec2[nw, i], vec2[nw, i])
    return engre, nwave, nsym, nstv, vec, vec2, out_vec2, br_dir


def calc_analytical_energy(kpt, engre, nwave, nsym, nstv, vec, vec2, out_vec2,
                           br_dir, sgn, scissor=0.0):
    """
    Args:
        kpt ([1x3 array]): fractional coordinates of the k-point
        engre, nwave, nsym, stv, vec, vec2, out_vec2, br_dir: all obtained via
            get_energy_args
        sgn (int): options are +1 for valence band and -1 for conduction bands
            sgn is basically ignored (doesn't matter) if scissor==0.0
        scissor (float): the amount by which the band gap is modified/scissored
    Returns:

    """
    energy, de, dde = get_energy(kpt, engre, nwave, nsym, nstv, vec, vec2,
                                 out_vec2, br_dir=br_dir)
    energy = energy * Ry_to_eV - sgn * scissor / 2.0
    velocity = abs(de / hbar * A_to_m * m_to_cm * Ry_to_eV)
    effective_m = hbar ** 2 / (
        dde * 4 * pi ** 2) / m_e / A_to_m ** 2 * e * Ry_to_eV
    return energy, velocity, effective_m


def get_bindex_bspin(extremum, is_cbm):
    """
    Returns the band index and spin of band extremum

    Args:
        extremum (dict): dictionary containing the CBM/VBM, i.e. output of
            Bandstructure.get_cbm()
        is_cbm (bool): whether the extremum is the CBM or not
    """

    idx = int(is_cbm) - 1  # 0 for CBM and -1 for VBM
    try:
        bidx = extremum["band_index"][Spin.up][idx]
        bspin = Spin.up
    except IndexError:
        bidx = extremum["band_index"][Spin.down][idx]
        bspin = Spin.down
    return bidx, bspin


def insert_intermediate_kpoints(kpts, n=2):
    """
    Insert n k-points in between each two k-points from kpts and return the
        latter bigger list. This can be used for example to make a finer-mesh
        HighSymmKpath
    Args:
        kpts ([[float]]): list of coordinates of k-points
        n (int): the number of k-points inserted between each pair of k-points
    Returns ([[float]]): the final list of k-point coordinates
    """
    n += 1
    new_kpts = []
    for i in range(len(kpts)-1):
        step = (kpts[i+1] - kpts[i])/n
        for j in range(n):
            new_kpts.append(kpts[i] + j*step)
    new_kpts.append(kpts[-1])
    return new_kpts


def get_bs_extrema(bs, coeff_file, nk_ibz=17, v_cut=1e4, min_normdiff=0.05,
                    Ecut=None, nex_max=0, return_global=False, niter=5,
                   nbelow_vbm= 0, nabove_cbm=0):
    """
    returns a dictionary of p-type (valence) and n-type (conduction) band
        extrema k-points by looking at the 1st and 2nd derivatives of the bands
    Args:
        bs (pymatgen BandStructure object): must containt Structure and have
            the same number of valence electrons and settings as the vasprun.xml
            from which coeff_file is generated.
        coeff_file (str): path to the cube file from BoltzTraP run
        nk_ibz (int): maximum number of k-points in one direction in IBZ
        v_cut (float): threshold under which the derivative is assumed 0 [cm/s]
        min_normdiff (float): the minimum allowed distance norm(fractional k)
            in extrema; this is important to avoid numerical instability errors
        Ecut (float or dict): max energy difference with CBM/VBM allowed for
            extrema
        nex_max (int): max number of low-velocity kpts tested for being extrema
        return_global (bool): in addition to the extrema, return the actual
            CBM (global minimum) and VBM (global maximum) w/ their k-point
        niter (int): number of iterations in basinhoopping for finding the
            global extremum
        nbelow_vbm (int): # of bands below the last valence band
        nabove_vbm (int): # of bands above the first conduction band
    Returns (dict): {'n': list of extrema fractional coordinates, 'p': same}
    """
    #TODO: MAJOR cleanup needed in this function; also look into parallelizing get_analytical_energy at all kpts if it's time consuming
    #TODO: if decided to only include one of many symmetrically equivalent extrema, write a method to keep only one of symmetrically equivalent extrema as a representative
    Ecut = Ecut or 10*k_B*300
    if not isinstance(Ecut, dict):
        Ecut = {'n': Ecut, 'p': Ecut}
    actual_cbm_vbm={'n': {}, 'p': {}}
    vbm_idx, _ = get_bindex_bspin(bs.get_vbm(), is_cbm=False)
    # vbm_idx = bs.get_vbm()['band_index'][Spin.up][0]
    ibands = [1-nbelow_vbm, 2+nabove_cbm]  # in this notation, 1 is the last valence band
    ibands = [i + vbm_idx for i in ibands]
    ibz = HighSymmKpath(bs.structure)
    sg = SpacegroupAnalyzer(bs.structure)
    kmesh = sg.get_ir_reciprocal_mesh(mesh=(nk_ibz, nk_ibz, nk_ibz))
    kpts = [k_n_w[0] for k_n_w in kmesh]
    kpts.extend(insert_intermediate_kpoints(list(ibz.kpath['kpoints'].values()), n=10))



    # grid = {'energy': [], 'velocity': [], 'mass': [], 'normv': []}
    extrema = {'n': [], 'p': []}
    engre, nwave, nsym, nstv, vec, vec2, out_vec2, br_dir = get_energy_args(
        coeff_file=coeff_file, ibands=ibands)

    cbmk = np.array(bs.get_cbm()['kpoint'].frac_coords)
    vbmk = np.array(bs.get_cbm()['kpoint'].frac_coords)
    bounds = [(-0.5,0.5), (-0.5,0.5), (-0.5,0.5)]
    func = lambda x: calc_analytical_energy(x, engre[1], nwave,
            nsym, nstv, vec, vec2, out_vec2, br_dir, sgn=-1, scissor=0)[0]
    opt = basinhopping(func, x0=cbmk, niter=niter, T=0.1, minimizer_kwargs={'bounds': bounds})
    kpts.append(opt.x)

    func = lambda x: -calc_analytical_energy(x, engre[0], nwave,
            nsym, nstv, vec, vec2, out_vec2, br_dir, sgn=+1, scissor=0)[0]
    opt = basinhopping(func, x0=vbmk, niter=niter, T=0.1, minimizer_kwargs={'bounds': bounds})
    kpts.append(opt.x)
    for iband in range(len(ibands)):
        is_cb = [False, True][iband]
        tp = ['p', 'n'][iband]
        energies = []
        velocities = []
        normv = []
        masses = []

######################################################################
        # kpts_list = [np.array([ 0.,  0.,  0.]),
        #              [-0.5, -0.5, -0.5],
        #              [-0.5, 0.0, 0.0],
        #              [0., -0.5, 0.],
        #              [0., 0., -0.5],
        #              [0.5, 0.5, 0.5],
        #              [0.5, 0.0, 0.0],
        #              [0., 0.5, 0.],
        #              [0., 0., 0.5],
        #              np.array([0., 0., 0.]),
        #              [ 0.5,  0.,  0.5],
        #             [0., -0.5, -0.5],
        #              [0.5 , 0.5 , 0.],
        #              [-0.5, 0., -0.5],
        #              [0., 0.5, 0.5],
        #              [-0.5, -0.5, 0.]
        #              ]


 #        kpts_list = [ [0.0, 0.0, 0.0],
 #            [ 0.  ,  0.44,  0.44],
 # [ 0.44,  0.44  ,0.  ],
 # [ 0. ,  -0.44, -0.44],
 # [-0.44 ,-0.44,  0.  ],
 # [ 0.44 , 0.  ,  0.44],
 # [-0.44 , 0.  , -0.44],
 # [ 0. ,  -0.44 ,-0.44],
 # [-0.44 ,-0.44 , 0.  ],
 # [ 0. ,   0.44 , 0.44],
 # [ 0.44,  0.44 , 0.  ],
 # [-0.44 , 0. ,  -0.44],
 # [ 0.44 , 0. ,   0.44] ,
 #   [ 0.5,  0. ,  0.5],
 # [ 0. ,  0.5,  0.5],
 # [ 0.5 , 0.5,  0. ],
 # [-0.5 , 0.  ,-0.5],
 # [ 0. , -0.5 ,-0.5],
 # [-0.5 ,-0.5 , 0. ]     ]
 #        print('here kpts_list:')
 #        print(kpts_list)
 #        print('here the energies')
 #        for ik, kpt in enumerate(kpts_list):
 #            en, v, mass = calc_analytical_energy(kpt, engre[iband], nwave,
 #                nsym, nstv, vec, vec2, out_vec2, br_dir, sgn=1, scissor=0)
 #            print en


######################################################################

        for ik, kpt in enumerate(kpts):
            en, v, mass = calc_analytical_energy(kpt, engre[iband], nwave,
                nsym, nstv, vec, vec2, out_vec2, br_dir, sgn=1, scissor=0)
            energies.append(en)
            velocities.append(abs(v))
            normv.append(norm(v))
            masses.append(mass.trace() / 3)
        indexes = np.argsort(normv)
        energies = [energies[i] for i in indexes]
        normv = [normv[i] for i in indexes]
        velocities = [velocities[i] for i in indexes]
        masses = [masses[i] for i in indexes]
        kpts = [np.array(kpts[i]) for i in indexes]

        # print('here')
        # cbmk = np.array([ 0.44,  0.44,  0.  ])
        # print(np.vstack((bs.get_sym_eq_kpoints(cbmk),bs.get_sym_eq_kpoints(-cbmk))))
        # cbmk = np.array([ 0.5,  0. ,  0.5])
        # print(np.vstack((bs.get_sym_eq_kpoints(cbmk),bs.get_sym_eq_kpoints(-cbmk))))

        # print('here values')
        # print energies[:10]
        # print normv[:10]
        # print kpts[:10]
        # print masses[:10]
        if is_cb:
            iextrem = np.argmin(energies)
            extremum0 = energies[iextrem]  # extremum0 is numerical CBM here
            actual_cbm_vbm[tp]['energy'] = extremum0
            actual_cbm_vbm[tp]['kpoint'] = kpts[iextrem]
            # The following is in case CBM doesn't have a zero numerical norm(v)
            closest_cbm = get_closest_k(kpts[iextrem], np.vstack((bs.get_sym_eq_kpoints(cbmk),bs.get_sym_eq_kpoints(-cbmk))))
            if norm(np.array(kpts[iextrem]) - closest_cbm) < min_normdiff and \
                            abs(bs.get_cbm()['energy']-extremum0) < 0.05:
                extrema['n'].append(cbmk)
            else:
                extrema['n'].append(kpts[iextrem])
        else:
            iextrem = np.argmax(energies)
            extremum0 = energies[iextrem]
            actual_cbm_vbm[tp]['energy'] = extremum0
            actual_cbm_vbm[tp]['kpoint'] = kpts[iextrem]
            closest_vbm = get_closest_k(kpts[iextrem], np.vstack((bs.get_sym_eq_kpoints(vbmk),bs.get_sym_eq_kpoints(-vbmk))))
            if norm(np.array(kpts[iextrem]) - closest_vbm) < min_normdiff and \
                            abs(bs.get_vbm()['energy']-extremum0) < 0.05:
                extrema['p'].append(vbmk)
            else:
                extrema['p'].append(kpts[iextrem])



        if normv[0] > v_cut:
            raise ValueError('No extremum point (v<{}) found!'.format(v_cut))
        for i in range(0, len(kpts[:nex_max])):
            # if (velocities[i] > v_cut).all() :
            if normv[i] > v_cut:
                break
            else:
                far_enough = True
                for k in extrema[tp]:
                    if norm(get_closest_k(kpts[i], np.vstack((bs.get_sym_eq_kpoints(k),bs.get_sym_eq_kpoints(-k))), return_diff=True)) <= min_normdiff:
                    # if norm(kpts[i] - k) <= min_normdiff:
                        far_enough = False
                if far_enough \
                        and abs(energies[i] - extremum0) < Ecut[tp] \
                        and masses[i] * ((-1) ** (int(is_cb) + 1)) >= 0:
                    extrema[tp].append(kpts[i])
    if not return_global:
        return extrema
    else:
        return extrema, actual_cbm_vbm


def get_dos_boltztrap2(params, st, mesh, estep, vbmidx = None, width=0.2, scissor=0.0):
    from BoltzTraP2 import fite
    (equivalences, lattvec, coeffs) = params
    ir_kpts = SpacegroupAnalyzer(st).get_ir_reciprocal_mesh(mesh)
    ir_kpts = [k[0] for k in ir_kpts]
    weights = [k[1] for k in ir_kpts]
    w_sum = float(sum(weights))
    weights = [w / w_sum for w in weights]

    fitted = fite.getBands(np.array(ir_kpts), equivalences=equivalences,
                           lattvec=lattvec, coeffs=coeffs)
    energies = fitted[0]*Ry_to_eV  # shape==(bands, nkpoints)
    nbands = energies.shape[0]
    if vbmidx:
        energies[vbmidx + 1:, :] += scissor / 2.
        energies[:vbmidx + 1, :] -= scissor / 2.
    e_min = np.min(energies)
    e_max = np.max(energies)
    height = 1.0 / (width * np.sqrt(2 * np.pi))
    e_points = int(round((e_max - e_min) / estep))
    e_mesh, step = np.linspace(e_min, e_max, num=e_points, endpoint=True,
                               retstep=True)
    e_range = len(e_mesh)
    dos = np.zeros(e_range)

    for ik, w in enumerate(weights):
        for b in range(nbands):
            g = height * np.exp(
                -((e_mesh - energies[b, ik]) / width) ** 2 / 2.)
            dos += w * g
    return e_mesh, dos, nbands