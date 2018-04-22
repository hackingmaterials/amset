import logging
from multiprocessing import Pool, cpu_count
import numpy as np
import os
import sys
import warnings

from amset.utils.analytical_band_from_BZT import Analytical_bands, outer, \
    get_energy
from amset.utils.constants import hbar, m_e, e, k_B, epsilon_0, sq3, \
    Hartree_to_eV, Ry_to_eV, A_to_m, m_to_cm
from math import pi, log

from pymatgen import Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import pbc_diff

try:
    import BoltzTraP2
    import BoltzTraP2.dft
    from BoltzTraP2 import sphere, fite
except ImportError:
    warnings.warn('BoltzTraP2 not imported; "boltztrap2" interpolation not available.')


class AmsetError(Exception):
    """
    Exception class for AMSET. Raised when AMSET gives an error.
    """

    def __init__(self, logger, msg):
        self.msg = msg
        logger.error(self.msg)

    def __str__(self):
        return "AmsetError : " + self.msg


def setup_custom_logger(name, filepath, filename, level=None):
    """
    Custom logger with both screen and file handlers. This is particularly
    useful if there are other programs (e.g. BoltzTraP2) that call on logging
    in which case the log results and their levels are distict and clear.
    Args:
        name (str): logger name to distinguish between different codes.
        filepath (str): path to the folder where the logfile is meant to be
        filename (str): log file filename
        level (int): log level in logging package; example: logging.DEBUG
    Returns: a logging instance with customized formatter and handlers
    """
    level = level or logging.DEBUG
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler(os.path.join(filepath, filename), mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(screen_handler)
    logger.addHandler(handler)
    return logger


def remove_from_grid(grid, grid_rm_list):
    """
    Deletes dictionaries storing properties that are no longer needed from
    a given grid (i.e. kgrid or egrid)
    """
    for tp in ["n", "p"]:
        for rm in grid_rm_list:
            try:
                del (grid[tp][rm])
            except:
                pass
    return grid


def norm(v):
    """
    Quickly calculates the norm of a vector (v: 1x3 or 3x1) as np.linalg.norm
    can be slower if called individually for each vector.
    """
    return (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5


def grid_norm(grid):
    return (grid[:,:,:,0]**2 + grid[:,:,:,1]**2 + grid[:,:,:,2]**2) ** 0.5


def kpts_to_first_BZ(kpts):
    """
    Brings a list of k-points to the 1st Brillouin Zone (BZ);
    i.e. -0.5 <= the fractional coordinates <= 0.5

    Args:
        kpts ([3x1 list or array]): list of k-points fractional coordinates

    Returns ([3x1 list or array]): list of transformed coordinates
    """
    new_kpts = []
    for i, k in enumerate(kpts):
        for alpha in range(3):
            while k[alpha] > 0.50:
                k[alpha] -= 1.00
            while k[alpha] < -0.50:
                k[alpha] += 1.00
        new_kpts.append(k)
    return new_kpts


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
                    mesh = [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.0045,
                            0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03,
                            0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.25]
                elif kgrid_tp == 'super fine':
                    mesh = [0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.007,
                            0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.25]
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
                    mesh = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05,
                            0.06, 0.07, 0.1, 0.15, 0.2, 0.25, 0.4, 0.5]
                else:
                    raise ValueError('Unsupported value for kgrid_tp: {}'.format(kgrid_tp))
                for step in mesh:
                    points_1d[dir].append(center[dim] + step)
                    points_1d[dir].append(center[dim] - step)
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
    """
    Returns the value of Fermi-Dirac at equilibrium for E (energy),
    fermi [level] and T (temperature)
    """
    return 1. / (1. + np.exp((E - fermi) / (k_B * T)))


def df0dE(E, fermi, T):
    """
    Returns the energy derivative of the Fermi-Dirac equilibrium distribution
    """
    exponent = (E - fermi) / (k_B * T)
    if exponent > 40 or exponent < -40:  # This is necessary so at too low numbers python doesn't return NaN
        return 0.0
    else:
        return -1 / (k_B * T) * np.exp((E - fermi) / (k_B * T)) / (1 + np.exp((E - fermi) / (k_B * T))) ** 2


def cos_angle(v1, v2):
    """
    Returns cosine of the angle between two 3x1 or 1x3 vectors
    """
    norm_v1, norm_v2 = norm(v1), norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 1.0  # In case of the two points are the origin, we assume 0 degree; i.e. no scattering: 1-X==0
    else:
        return np.dot(v1, v2) / (norm_v1 * norm_v2)


def fermi_integral(order, fermi, T, initial_energy=0):
    """
    Returns the Fermi integral
    (e.g. for calculating single parabolic band acoustic phonon mobility)

    Args:
        order (int): the order of integral
        fermi (float): absolute band structure fermi (not relative to CBM/VBM)
        T (float): the temperature in kelvin
        initial_energy (float): the actual CBM/VBM energy in eV
        wordy (bool): whether to print out the integrals or not
    """
    fermi = fermi - initial_energy
    integral = 0.
    nsteps = 100000.0
    # TODO: 1e6 works better (converges!) but for faster test we use 1e5
    emesh = np.linspace(0.0, 30 * k_B * T, nsteps)
    dE = (emesh[-1] - emesh[0]) / (nsteps - 1.0)
    for E in emesh:
        integral += dE*(E / (k_B*T))**order / (1.+np.exp((E-fermi) / (k_B*T)))
    return integral


def GB(x, eta):
    """
    Gaussian broadening. At very small eta values (e.g. 0.005 eV) this function
    goes to the dirac-delta of x.

    Args:
        x (float): the mean value of the nomral distribution
        eta (float): the standard deviation of the normal distribution
    """
    return 1 / np.pi * 1 / eta * np.exp(-(x / eta) ** 2)


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
    """
    Calculates the polar optical phonon "in" and "out" scattering rates.
    This method is defined outside of the AMSET class to enable parallelization
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

    k = kgrid[tp]["norm(k)"][ib][ik]
    a = kgrid[tp]["a"][ib][ik]
    c_ = kgrid[tp]["c"][ib][ik]
    f = kgrid[tp]["f"][c][T][ib][ik]
    f_th = kgrid[tp]["f_th"][c][T][ib][ik]
    N_POP = kgrid[tp]["N_POP"][c][T][ib][ik]

    for j, X_Epm in enumerate(["X_Eplus_ik", "X_Eminus_ik"]):
        if tp == "n" and X_Epm == "X_Eminus_ik" and kgrid[tp]["energy"][ib][ik] - hbar * \
                kgrid[tp]["W_POP"][ib][ik] < cbm_vbm[tp]["energy"]:
            continue
        if tp == "p" and X_Epm == "X_Eplus_ik" and kgrid[tp]["energy"][ib][ik] + hbar * \
                kgrid[tp]["W_POP"][ib][ik] > cbm_vbm[tp]["energy"]:
            continue
        counted = len(kgrid[tp][X_Epm][ib][ik])
        for X_ib_ik in kgrid[tp][X_Epm][ib][ik]:
            X, ib_pm, ik_pm = X_ib_ik
            k_pm = kgrid[tp]["norm(k)"][ib_pm][ik_pm]
            abs_kdiff = abs(k_pm - k)
            if abs_kdiff < 1e-4:
                # avoid rate blow-up (e.g. due to self-scattering)
                counted -= 1
                continue
            if abs(kgrid[tp]['energy'][ib_pm][ik_pm] - \
                           kgrid[tp]['energy'][ib][ik]) < \
                                    hbar * kgrid[tp]["W_POP"][ib][ik] / 2.0:
                counted -= 1
                continue
            g_pm = kgrid[tp]["g"][c][T][ib_pm][ik_pm]
            g_pm_th = kgrid[tp]["g_th"][c][T][ib_pm][ik_pm]
            v_pm = kgrid[tp]["norm(v)"][ib_pm][ik_pm] / sq3  # 3**0.5 is to treat each direction as 1D BS
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
                (k_pm**2 + k**2) / (2*k*k_pm) * A_pm**2 *\
                log((k_pm + k) / (abs_kdiff)) - A_pm**2 - c_**2 * c_pm** 2 / 3.0)
            S_i[j] += (N_POP + (1 - j) + (-1)**(1 - j) * f) * lamb_ipm * g_pm
            S_i_th[j] += (N_POP + (1 - j) + (-1)**(1 - j) * f_th) * lamb_ipm * g_pm_th
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
    if len(list(kpoint)) != 3 or len(list(ref_ks[0])) != 3:
        raise AmsetError('k-point coordinates must be 3-dimensional')
    # assert isinstance(ref_ks[0][0], (float, list))
    norms = []
    for ki in ref_ks:
        norm_diff = norm(ki - kpoint)
        if norm_diff > threshold:
            norms.append(norm_diff)
        else:
            norms.append(1e10)
    min_dist_ik = np.array(norms).argmin()
    if return_diff:
        return kpoint - ref_ks[min_dist_ik]
    else:
        return ref_ks[min_dist_ik]


def remove_duplicate_kpoints(kpts, dk=0.01):
    """
    Removes duplicate points from a list of k-points. Note that it is better
    to call this method only once as calculating the norms scales poorly.
    Args:
        kpts ([np.ndarray or list]): list of k-point coordinates
    Returns: kpts but with duplicate points removed.
    """
    rm_list = []
    for i in range(len(kpts) - 1):
        for j in range(i + 1, len(kpts)):
            if np.allclose(pbc_diff(kpts[i], kpts[j]), [0, 0, 0], atol=dk):
                rm_list.append(j)
                break
    return [list(k) for k in np.delete(kpts, rm_list, axis=0)]


def find_fermi_SPB(cbm_vbm, c, T, tolerance=0.01, alpha=0.02, max_iter=1000):
    """
    Not tested! Returns the fermi level based on single parabolic band (SPB)
    assumption. Note that this function is currently not tested and not used
    in AMSET
    Args:
        cbm_vbm (dict):
        c (float):
        T (float):
        tolerance (float):
        alpha (float):
        max_iter (int):
    Returns (float): the fermi level under SPB assumption.
    """
    tp = get_tp(c)
    sgn = np.sign(c)
    m_eff = np.prod(cbm_vbm[tp]["eff_mass_xx"]) ** (1.0 / 3.0)
    c *= sgn
    initial_energy = cbm_vbm[tp]["energy"]
    fermi = initial_energy + 0.02
    for iter in range(max_iter):
        calc_doping = 4*pi* (2*m_eff*m_e*k_B*T / hbar**2)**1.5 * \
                    fermi_integral(0.5, fermi, T,initial_energy)*1e-6 / e**1.5
        fermi += alpha * sgn * (calc_doping - c) / abs(c + calc_doping) * fermi
        relative_error = abs(calc_doping - c) / abs(c)
        if relative_error <= tolerance:
            # This here assumes that the SPB generator set the VBM to 0.0 and CBM=gap + scissor
            if sgn < 0:
                return fermi
            else:
                return -(fermi - initial_energy)
    if relative_error > tolerance:
        raise ValueError("could NOT find a corresponding SPB fermi level after {} itenrations".format(max_iter))


def get_tp(c):
    """
    Returns "n" for n-type (electrons majority carrier, c<0) or "p" (p-type).
    """
    if c < 0:
        return "n"
    elif c > 0:
        return "p"
    else:
        raise ValueError("The carrier concentration cannot be zero! AMSET stops now!")


def get_angle(v1, v2):
    """
    Returns the actual angles (in radian) between 2 vectors not its cosine.
    """
    x = cos_angle(v1,v2)
    if x < -1:
        x = -1
    elif x > 1:
        x = 1
    return np.arccos(x)


def sort_angles(vecs):
    """
    Sort a list of vectors based on their pair angles using a greedy algorithm.
    Args:
        vecs ([nx1 list or numpy.ndarray]): list of nd vectors
    Returns (sorted vecs, indexes of the initial vecs that result in sorted vecs):
    """
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


def get_energy_args(coeff_file, ibands):
    """
    Args:
        coeff_file (str): the address to the cube (*.123) file
        ibands ([int]): list of band numbers to be calculated; note that the
            first band index is 1 not 0
    Returns (tuple): necessary inputs for calc_analytical_energy or get_energy
    """
    analytical_bands = Analytical_bands(coeff_file=coeff_file)
    try:
        engre, latt_points, nwave, nsym, nsymop, symop, br_dir = \
            analytical_bands.get_engre(iband=ibands)
    except TypeError as e:
        raise ValueError('try reducing Ecut to include fewer bands', e)

    nstv, vec, vec2 = analytical_bands.get_star_functions(
            latt_points, nsym, symop, nwave, br_dir=br_dir)
    out_vec2 = np.zeros((nwave, max(nstv), 3, 3))
    for nw in range(nwave):
        for i in range(nstv[nw]):
            out_vec2[nw, i] = outer(vec2[nw, i], vec2[nw, i])
    return engre, nwave, nsym, nstv, vec, vec2, out_vec2, br_dir


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



def get_dos_boltztrap2(params, st, mesh, estep=0.001, vbmidx=None,
                       width=0.2, scissor=0.0):
    """
    Calculates the density of states (DOS) based on boltztrap2 interpolation.
    Args:
        params (list/tuple): parameters required for boltztrap2 interpolation
        st (pymatgen Structure object): required for generating irriducible
            brillouin zone mesh)
        mesh (a 3x1 list or np.ndarray): the k-grid; e.g. [13, 15, 11]
        estep (float): small energy step, the smaller better but more expensive
        vbmidx (int): the index of the valence band maximum assuming the index
            of the first band is 0
        width (float): energy bandwidth/smearing parameter.
        scissor (float): the intended change to the current band gap
    Returns (tuple): in the same order: 1) list of enegy values 2) list of
        densities at those energy values and 3) number of bands considered
    """
    from BoltzTraP2 import fite
    (equivalences, lattvec, coeffs) = params
    ir_kpts = SpacegroupAnalyzer(st).get_ir_reciprocal_mesh(mesh)
    ir_kpts = [k[0] for k in ir_kpts]
    weights = [k[1] for k in ir_kpts]
    w_sum = float(sum(weights))
    weights = [w / w_sum for w in weights]

    energies, _, _ = fite.getBands(np.array(ir_kpts), equivalences=equivalences,
                           lattvec=lattvec, coeffs=coeffs)
    energies *= Hartree_to_eV  # shape==(bands, nkpoints)
    nbands = energies.shape[0]
    if vbmidx:
        ib_start = max(0, vbmidx-4)
        ib_end = min(energies.shape[0], vbmidx+1+4)
        energies[vbmidx + 1:, :] += scissor / 2.
        energies[:vbmidx + 1, :] -= scissor / 2.
        energies = energies[ib_start:ib_end, :]
        nbands = ib_end - ib_start
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
            g = height * np.exp(-((e_mesh - energies[b, ik]) / width) ** 2 / 2.)
            dos += w * g
    return e_mesh, dos, nbands


def interpolate_bs(kpts, interp_params, iband, sgn=None, method="boltztrap1",
                   scissor=0.0, matrix=None, n_jobs=1):
    """
    Args:
        kpts ([1x3 array]): list of fractional coordinates of k-points
        interp_params (tuple): a tuple or list containing positional
            arguments fed to the interpolation method.
            e.g. for boltztrap1:
                engre, nwave, nsym, stv, vec, vec2, out_vec2, br_dir
            and for boltztrap2:
                (equivalences, lattvec, coeffs)
        iband (int): the band index for which the list of energy, velocity
            and mass is returned. If "boltztrap2" method is used, this is the
            actual band index while if "boltztrap1" methid is used, this is the
            ith band among the bands that were included in the fit (i.e. when
            get_energy_args is called)
        sgn (float): options are +1 for valence band and -1 for conduction bands
            sgn is basically ignored (doesn't matter) if scissor==0.0
        method (str): the interpolation method. Current options are
            "boltztrap1", "boltztrap2"
        scissor (float): the amount by which the band gap is modified/scissored
        matrix (3x3 np.ndarray): the direct lattice matrix used to convert
            the velocity (in fractional coordinates) to cartesian in
            boltztrap1 method.
        n_jobs (int): number of processes used in boltztrap1 interpolation
    Returns (tuple of energies, velocities, masses lists/np.ndarray):
    """
    if matrix is None:
        matrix = np.eye(3)
    if not sgn:
        if scissor == 0.0:
            sgn=0.0
        else:
            raise ValueError('To apply scissor "sgn" is required: -1 or +1')
    if method=="boltztrap1":
        engre, nwave, nsym, nstv, vec, vec2, out_vec2, br_dir = interp_params
        energies = []
        velocities = []
        masses = []
        if n_jobs == 1:
            results = []
            for kpt in kpts:
                energy, de, dde = get_energy(kpt, engre[iband], nwave, nsym,
                                             nstv, vec, vec2, out_vec2, br_dir)
                results.append((energy, de, dde))
        else:
            if n_jobs == -1:
                n_jobs = cpu_count()
            inputs = [(kpt, engre[iband], nwave, nsym, nstv, vec, vec2,
                                            out_vec2, br_dir) for kpt in kpts]
            with Pool(n_jobs if n_jobs != -1 else cpu_count()) as p:
                results = p.starmap(get_energy, inputs)
        for energy, de, dde in results:
            energy = energy * Ry_to_eV - sgn * scissor / 2.0
            velocity = abs(np.dot(matrix, de.T).T) / (hbar * 2 * pi) / 0.52917721067 * A_to_m * m_to_cm * Ry_to_eV
            effective_m = 1/(dde/ 0.52917721067) * e / Ry_to_eV / A_to_m**2 * (hbar*2*np.pi)**2 / m_e
            energies.append(energy)
            velocities.append(velocity)
            masses.append(effective_m)
    elif method=="boltztrap2":
        if n_jobs == 1:
            warnings.warn('n_jobs={}: Parallel not implemented w/ boltztrap2'
                          .format(n_jobs))
        fitted = fite.getBands(np.array(kpts), *interp_params)
        energies = fitted[0][iband - 1] * Hartree_to_eV - sgn * scissor / 2.
        velocities = fitted[1][:, :, iband - 1].T * Hartree_to_eV / hbar * A_to_m * m_to_cm
        masses = fitted[2][:, :, :, iband - 1].T* e / Hartree_to_eV / A_to_m**2 * hbar**2/m_e
    else:
        raise AmsetError('Unsupported interpolation "{}"'.format(method))
    return energies, velocities, masses