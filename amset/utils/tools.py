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
from detect_peaks import detect_peaks
from math import pi, log
from pymatgen import Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.util.coord import pbc_diff

try:
    import BoltzTraP2
    import BoltzTraP2.dft
    from BoltzTraP2 import sphere, fite
except ImportError:
    warnings.warn('BoltzTraP2 not imported; "boltztrap2" interpolation not available.')


class AmsetError(Exception):
    """
    Exception class for AMSET. Raised when AMSET gives an error. The purpose
    of this class is to be explicit about the exceptions raised specifically
    due to AMSET input/output requirements instead of a generic ValueError, etc
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
                    # mesh = [0.003, 0.01, 0.05, 0.15]
                elif kgrid_tp == 'very coarse':
                    mesh = [0.001, 0.01]
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


def calculate_Sio(tp, c, T, ib, ik, once_called, kgrid, cbm_vbm, epsilon_s, epsilon_inf):
    """
    Calculates the polar optical phonon "in" and "out" scattering rates.
    This method is defined outside of the AMSET class to enable parallelization

    Args:
        tp (str): type of the bands
            options: "n" for the conduction and "p" for the valence bands
        c (float): the carrier concentration in 1/cm3
        T (float): the temperature in Kelvin
        ib (int): the band index
        ik (int): the k-point index
        once_called (bool): whether this function was once called hence S_o and
            S_o_th calculated once or not. Caches already calculated properties
        kgrid (dict): the main kgrid variable in AMSET (AMSET.kgrid)
        cbm_vbm (dict): from AMSET.cbm_vbm, containing cbm and vbm energy
        epsilon_s (float): static dielectric constant
        epsilon_inf (float): high-frequency dielectric constant

    Returns ([four 3x1 lists]):
        the overall vectors for S_i, S_i_th, S_o, and S_o_th
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
            # v_pm = kgrid[tp]["velocity"][ib_pm][ik_pm] # 3**0.5 is to treat each direction as 1D BS
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


def get_closest_k(kpoint, ref_ks, return_diff=False, exclude_self=False):
    """
    returns the list of difference between kpoints. If return_diff True, then
        for a given kpoint the minimum distance among distances with ref_ks is
        returned or just the reference kpoint that results if not return_diff

    Args:
        kpoint (1x3 array): the coordinates of the input k-point
        ref_ks ([1x3 array]): list of reference k-points from which the
            distance with initial_ks are calculated
        return_diff (bool): if True, the minimum distance is returned
        exclude_self (bool): if kpoint is already repeated in ref_ks, exclude
            that from ref_ks

    Returns (1x3 array):
    """
    if len(list(kpoint)) != 3 or len(list(ref_ks[0])) != 3:
        raise AmsetError('k-point coordinates must be 3-dimensional')
    norms = [norm(ki-kpoint) for ki in ref_ks]
    if exclude_self:
        norms = [norm if norm>0.001 else 1e10 for norm in norms]
    min_dist_ik = np.array(norms).argmin()
    if return_diff:
        return kpoint - ref_ks[min_dist_ik]
    else:
        return ref_ks[min_dist_ik]


def remove_duplicate_kpoints(kpts, dk=0.01, periodic=True):
    """
    Removes duplicate points from a list of k-points.

    Args:
        kpts ([np.ndarray or list]): list of k-point coordinates

    Returns ([np.ndarray or list]):
        kpts but with duplicate points removed.
    """
    rm_list = []
    if periodic:
        diff_func = pbc_diff
    else:
        diff_func = np.subtract
    # identify and remove duplicates from the list of equivalent k-points:
    for i in range(len(kpts) - 1):
        for j in range(i + 1, len(kpts)):
            if np.allclose(diff_func(kpts[i], kpts[j]), [0, 0, 0], atol=dk):
                rm_list.append(i)
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

    Returns (float):
        the fermi level under SPB assumption.
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

    energies, _ = fite.getBands(np.array(ir_kpts), equivalences=equivalences,
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
        energies ([float]): energy values at kpts for a corresponding iband
        velocities ([3x1 array]): velocity vectors
        masses ([3x3 matrix]): list of effective mass tensors
    """
    #TODO: effective mass is still inconsistent between btp1 and btp2 w/o any transformation used since it is not used in AMSET ok but has to be checked with the right transformation
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
            inputs = [(kpt, engre[iband], nwave, nsym, nstv, vec, vec2,
                                            out_vec2, br_dir) for kpt in kpts]
            with Pool(n_jobs if n_jobs != -1 else cpu_count()) as p:
                results = p.starmap(get_energy, inputs)
        for energy, de, dde in results:
            energy = energy * Ry_to_eV - sgn * scissor / 2.0
            velocity = abs(np.dot(matrix/np.linalg.norm(matrix), de)) / hbar / 0.52917721067 * A_to_m * m_to_cm * Ry_to_eV # this results in btp1-btp2 consistency but ONLY IF matrix is None
            effective_m = 1/(dde/ 0.52917721067**2*Ry_to_eV) * e / A_to_m**2 * hbar**2 / m_e
            energies.append(energy)
            velocities.append(velocity)
            masses.append(effective_m)
    elif method=="boltztrap2":
        if n_jobs != 1:
            warnings.warn('n_jobs={}: Parallel not implemented w/ boltztrap2'
                          .format(n_jobs))
        equivalences, lattvec, coeffs = interp_params
        fitted = fite.getBands(np.array(kpts), equivalences, lattvec, coeffs, curvature=True)
        energies = fitted[0][iband - 1] * Hartree_to_eV - sgn * scissor / 2.
        velocities = abs(np.matmul(matrix/np.linalg.norm(matrix), fitted[1][:, iband - 1, :]).T) * Hartree_to_eV / hbar * A_to_m * m_to_cm / 0.52917721067
        try:
            masses = 1/(fitted[2][:, :, iband - 1, :].T/ 0.52917721067**2*Hartree_to_eV)* e / A_to_m**2 * hbar**2/m_e
        except IndexError:
            warnings.warn("The boltztrap2 fite.getBands version does not return "
                          " effective mass. The tensors will be all zeros.")
            masses = np.array([np.zeros((3, 1))]*len(kpts))
    else:
        raise AmsetError("Unsupported interpolation method: {}".format(method))
    return energies, velocities, masses


def get_bs_extrema(bs, coeff_file=None, interp_params=None, method="boltztrap1",
                   line_density=30, min_normdiff=sq3/10.0,
                   Ecut=None, eref=None, return_global=False, n_jobs=-1,
                   nbelow_vbm=0, nabove_cbm=0, scissor=0.0):
    """
    Returns a dictionary of p-type (valence) and n-type (conduction) band
        extrema k-points by looking at the 1st and 2nd derivatives of the bands

    Args:
        bs (pymatgen BandStructure object): must contain Structure and have
            the same number of valence electrons and settings as the vasprun.xml
            from which coeff_file is generated.
        coeff_file (str): path to the cube file from BoltzTraP run
        line_density (int): maximum number of k-points between each two
            consecutive high-symmetry k-points
        v_cut (float): threshold under which the derivative is assumed 0 [cm/s]
        min_normdiff (float): the minimum allowed distance norm(fractional k)
            in extrema; this is important to avoid numerical instability errors
        Ecut (float or dict): max energy difference with CBM/VBM allowed for
            extrema. Valid examples: 0.25 or {'n': 0.5, 'p': 0.25} , ...
        eref (dict): BandStructure global VBM/CBM used as a global reference
            energy for Ecut. Example: {'n': 6.0, 'p': 5.0}. Ignored if None in
            which case maximum/minimum of the current valence/conduction used
        return_global (bool): in addition to the extrema, return the actual
            CBM (global minimum) and VBM (global maximum) w/ their k-point
        n_jobs (int): number of processors used in boltztrap1 interpolation
        nbelow_vbm (int): # of bands below the last valence band
        nabove_vbm (int): # of bands above the first conduction band
        scissor (float): the amount by which the band gap is altered/scissored.

    Returns (dict): {'n': list of extrema fractional coordinates, 'p': same}
    """
    Ecut = Ecut or 10 * k_B * 300
    if not isinstance(Ecut, dict):
        Ecut = {'n': Ecut, 'p': Ecut}
    if eref is None:
        global_extrema = {'n': {}, 'p': {}}
    else:
        cbmk = np.array(bs.get_cbm()['kpoint'].frac_coords)
        vbmk = np.array(bs.get_vbm()['kpoint'].frac_coords)
        global_extrema = {'n': {'energy': eref['n'], 'kpoint': cbmk},
                          'p': {'energy': eref['p'], 'kpoint': vbmk}}
    final_extrema = {'n': [], 'p': []}
    hsk = HighSymmKpath(bs.structure)
    hs_kpoints, _ = hsk.get_kpoints(line_density=line_density)
    hs_kpoints = kpts_to_first_BZ(hs_kpoints)
    vbm_idx, _ = get_bindex_bspin(bs.get_vbm(), is_cbm=False)
    cbm_idx, _ = get_bindex_bspin(bs.get_cbm(), is_cbm=True)

    if method == "boltztrap1" and interp_params is None:
        interp_params = get_energy_args(coeff_file=coeff_file,
                                        ibands=[vbm_idx + 1 - nbelow_vbm,
                                                cbm_idx + 1 + nabove_cbm])

    for ip, tp in enumerate(["p", "n"]): # hence iband == 0 or 1
        if method=="boltztrap1":
            iband = ip
        else:
            iband = ip*(cbm_idx+nabove_cbm) + (1-ip)*(vbm_idx-nbelow_vbm) + 1
        band , _, _ = interpolate_bs(hs_kpoints, interp_params, iband=iband,
                                      method=method, scissor=scissor,
                                      matrix=bs.structure.lattice.matrix,
                                      n_jobs=n_jobs, sgn=(-1)**iband)
        global_ext_idx = (1-iband) * np.argmax(band) + iband * np.argmin(band)
        if eref is None:
            global_extrema[tp]['energy'] = band[global_ext_idx]
            global_extrema[tp]['kpoint'] = hs_kpoints[global_ext_idx]
        extrema_idx = detect_peaks(band, mph=None, mpd=min_normdiff,
                               valley=ip==1)

        extrema_init = []
        for idx in extrema_idx:
            k_localext = hs_kpoints[idx]
            if abs(band[idx] - global_extrema[tp]['energy']) < Ecut[tp]:
                far_enough = True
                for kp in extrema_init:
                    kp = np.array(kp)
                    if norm(get_closest_k(k_localext,
                                          np.vstack(
                                              (bs.get_sym_eq_kpoints(-kp),
                                               bs.get_sym_eq_kpoints(kp))),
                                          return_diff=True)) <= min_normdiff:
                        far_enough = False
                if far_enough:
                    extrema_init.append(k_localext)

        # check to see if one of the actual high-symm k-points can be used
        hisymks = list(hsk.kpath['kpoints'].values())
        all_hisymks = []
        for kp in hisymks:
            all_hisymks += list(np.vstack((bs.get_sym_eq_kpoints(-kp),
                                           bs.get_sym_eq_kpoints(kp))))
        for k_ext_found in extrema_init:
            kp = get_closest_k(k_ext_found, all_hisymks, return_diff=False)
            if norm(kp - k_ext_found) < min_normdiff/2.0:
                final_extrema[tp].append(kp)
            else:
                final_extrema[tp].append(k_ext_found)
        # sort the extrema based on their energy (i.e. importance)
        subband, _, _ = interpolate_bs(final_extrema[tp], interp_params, iband=iband,
                                    method=method, scissor=scissor,
                                    matrix=bs.structure.lattice.matrix,
                                    n_jobs=n_jobs, sgn=(-1) ** iband)
        sorted_idx = np.argsort(subband)
        if iband==0:
            sorted_idx = sorted_idx[::-1]
        final_extrema[tp] = [final_extrema[tp][i] for i in sorted_idx]

    if return_global:
        return final_extrema, global_extrema
    else:
        return final_extrema


def get_dft_orbitals(vasprun, bidx, lorbit):
    """
    The contribution from s and p orbitals at a given band for kpoints
    that were used in the DFT run (from which vasprun.xml is read). This is
    just to parse the total orbital contributions out of Vasprun depending
    on LORBIT.

    Args:
        vasprun (pymatgen Vasprun):
        bidx (idx): band index
        lorbit (int): the LORBIT flag that was used when vasprun.xml generated

    Returns:
        ([float], [float]) two lists: s&p orbital scores at the band # bidx
    """
    projected = vasprun.projected_eigenvalues
    nk = len(vasprun.actual_kpoints)
    # projected indexes : Spin; kidx; bidx; s,py,pz,px,dxy,dyz,dz2,dxz,dx2
    s_orbital = [0.0] * nk
    p_orbital = [0.0] * nk
    for ik in range(nk):
        s_orbital[ik] = sum(projected[Spin.up][ik][bidx])[0]
        if lorbit == 10:
            p_orbital[ik] = sum(projected[Spin.up][ik][bidx])[1]
        elif lorbit == 11:
            p_orbital[ik] = sum(sum(projected[Spin.up][ik][bidx])[1:4])
        else:
            raise AmsetError('Not sure what to do with lorbit={}'.format(lorbit))
    return s_orbital, p_orbital


def generate_adaptive_kmesh(bs, important_points, kgrid_tp, ibz=True):
    """
    Returns a kpoint mesh surrounding the important k-points in the
    conduction (n-type) and valence bands (p-type). This mesh is adaptive,
    meaning that the mesh is much finer closer to these "important" points
    than it is further away. This saves tremendous computational time as
    the points closer to the extremum are the most important ones dictating
    the transport properties.

    Args:
        bs (pymatgen.BandStructure): the bandstructure with bs.structure
            present that are used for structural symmetry and getting the
            symmetrically equivalent points
        important_points ({"n": [[3x1 array]], "p": [[3x1 array]]}): list
            of fractional coordinates of extrema for n-type and p-type
        kgrid_tp (str): determines how coarse/fine the k-mesh would be.
            options: "very coarse", "coarse", "fine", "very fine"
        ibz (bool): whether to generate the k-mesh based on a scaled k-mesh
            in Irreducible Brillouin Zone (True, recommended) or custom
            intervals only in the directions (+/-) of ibz kpoints.

    Returns ({"n": [[3x1 array]], "p": [[3x1 array]]}):
        list of k-points (k-mesh) may be different for conduction or
        valence bands.
    """
    if ibz:
        kpts = {}
        kgrid_tp_map = {'very coarse': 4,
                        'coarse': 8,
                        'fine': 14,
                        'very fine': 19,
                        'super fine': 25,
                        'extremely fine': 33
                        }
        nkk = kgrid_tp_map[kgrid_tp]
        sg = SpacegroupAnalyzer(bs.structure)
        kpts_and_weights = sg.get_ir_reciprocal_mesh(mesh=(nkk, nkk, nkk),
                                                     is_shift=[0, 0, 0])
        kpts_and_weights_init = sg.get_ir_reciprocal_mesh(mesh=(5, 5, 5),
                                                     is_shift=[0, 0, 0])
        initial_ibzkpt_init = [i[0] for i in kpts_and_weights_init]
        initial_ibzkpt0 = np.array([i[0] for i in kpts_and_weights])

        # this is to cover the whole BZ in an adaptive manner in case of high-mass isolated valleys
        initial_ibzkpt0 = np.array(initial_ibzkpt_init + \
                                   list(initial_ibzkpt0/5.0) + \
                                   list(initial_ibzkpt0/20.0) )
        for tp in ['p', 'n']:
            tmp_kpts = []
            for important_point in important_points[tp]:
                initial_ibzkpt = initial_ibzkpt0 + important_point
                for k in initial_ibzkpt:
                    tmp_kpts += list(bs.get_sym_eq_kpoints(k))
            kpts[tp] = tmp_kpts
    else:
        if kgrid_tp == "fine":
            mesh = np.array(
                [0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.25])
            nkk = 15
        elif kgrid_tp == "coarse":
            mesh = np.array([0.001, 0.005, 0.03, 0.1])
            nkk = 10
        elif kgrid_tp == 'very coarse':
            mesh = np.array([0.001, 0.01])
            nkk = 5
        else:
            raise AmsetError('Unsupported kgrid_tp: {}'.format(kgrid_tp))
        # just to find a set of + or - kpoint signs when ibzkpt is generated:
        sg = SpacegroupAnalyzer(bs.structure)
        kpts_and_weights = sg.get_ir_reciprocal_mesh(mesh=(nkk, nkk, nkk),
                                                     is_shift=[0, 0, 0])
        initial_ibzkpt = [i[0] for i in kpts_and_weights]
        step_signs = [[np.sign(k[0]), np.sign(k[1]), np.sign(k[2])] for k in
                      initial_ibzkpt]
        step_signs = remove_duplicate_kpoints(step_signs, periodic=False)

        # actually generating the mesh seaprately for n- and p-type
        kpts = {'n': [], 'p': []}
        for tp in ["n", "p"]:
            for k_extremum in important_points[tp]:
                for kx_sign, ky_sign, kz_sign in step_signs:
                    for kx in k_extremum[0] + kx_sign*mesh:
                        for ky in k_extremum[1] + ky_sign*mesh:
                            for kz in k_extremum[2] + kz_sign*mesh:
                                kpts[tp].append(np.array([kx, ky, kz]))
            tmp_kpts = []
            for k in kpts[tp]:
                tmp_kpts += list(bs.get_sym_eq_kpoints(k))
            kpts[tp] = remove_duplicate_kpoints(tmp_kpts, dk=0.0009)
    return kpts


def create_plots(x_title, y_title, show_interactive, save_format, c, tp,
                 file_suffix, fontsize, ticksize, path, margins, fontfamily,
                 plot_data, names=None, labels=None, x_label_short='',
                 y_label_short=None, mode='markers', y_axis_type='linear', title=None):
    """
    A wrapper private function with args mostly consistent with
    matminer.figrecipes.PlotlyFig but slightly better handling of plot
    file saving (e.g. incorporating temperature and concentration in
    the filename, etc).
    """
    from matminer.figrecipes.plot import PlotlyFig
    tp_title = {"n": "conduction band(s)", "p": "valence band(s)"}
    if title is None:
        title = '{} for {}, c={}'.format(y_title, tp_title[tp], c)
    if y_label_short is None:
        y_label_short = y_title
    if show_interactive:
        if not x_label_short:
            filename = os.path.join(path, "{}_{}.{}".format(y_label_short, file_suffix, 'html'))
        else:
            filename = os.path.join(path, "{}_{}_{}.{}".format(y_label_short, x_label_short, file_suffix, 'html'))
        pf = PlotlyFig(x_title=x_title, y_title=y_title, y_scale=y_axis_type,
                        title=title, fontsize=fontsize,
                       mode='offline', filename=filename, ticksize=ticksize,
                        margins=margins, fontfamily=fontfamily)
        pf.xy(plot_data, names=names, labels=labels, modes=mode)
    if save_format is not None:
        if not x_label_short:
            filename = os.path.join(path, "{}_{}.{}".format(y_label_short, file_suffix, save_format))
        else:
            filename = os.path.join(path, "{}_{}_{}.{}".format(y_label_short, x_label_short, file_suffix, save_format))
        pf = PlotlyFig(x_title=x_title, y_title=y_title,
                        title=title, fontsize=fontsize,
                        mode='static', filename=filename, ticksize=ticksize,
                        margins=margins, fontfamily=fontfamily)
        pf.xy(plot_data, names=names, labels=labels, modes=mode)