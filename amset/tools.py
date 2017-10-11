import numpy as np
from constants import hbar, m_e, e, k_B, epsilon_0, sq3
from math import pi, log

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
                if kgrid_tp == 'poly_band':
                    mesh = [0.002, 0.004, 0.007, 0.01, 0.016, 0.025, 0.035,
                            0.05, 0.07, 0.1, 0.15, 0.25, 0.35, 0.5]
                elif kgrid_tp == 'super fine':
                    mesh = [0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.01, 0.02, 0.03,
                            0.05, 0.07, 0.1, 0.15, 0.25, 0.35, 0.5]
                elif kgrid_tp == 'very fine':
                    mesh = [0.001, 0.002, 0.004, 0.007, 0.01, 0.02, 0.03,
                            0.05, 0.07, 0.1, 0.15, 0.25, 0.35, 0.5]
                elif kgrid_tp == 'fine':
                    mesh = [0.004, 0.01, 0.02, 0.03,
                            0.05, 0.07, 0.1, 0.15, 0.25, 0.35, 0.5]
                elif kgrid_tp == 'coarse':
                    mesh = [0.001, 0.005, 0.01, 0.05, 0.15, 0.5]
                elif kgrid_tp == 'very coarse':
                    mesh = [0.001, 0.01, 0.1, 0.5]
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
    if E - fermi > 5:
        return 0.0
    elif E - fermi < -5:
        return 1.0
    else:
        return 1 / (1 + np.exp((E - fermi) / (k_B * T)))


def df0dE(E, fermi, T):
    """returns the energy derivative of the Fermi-Dirac equilibrium distribution"""
    if E - fermi > 5 or E - fermi < -5:  # This is necessary so at too low numbers python doesn't return NaN
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
        print "order {} fermi integral at fermi={} and {} K".format(order, fermi, T)
        print integral
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

    v = kgrid[tp]["norm(v)"][ib][ik] / sq3  # 3**0.5 is to treat each direction as 1D BS
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
            if abs(kgrid[tp]['energy'][ib_pm][ik_pm] - \
                           kgrid[tp]['energy'][ib][ik]) < \
                                    hbar * kgrid[tp]["W_POP"][ib][ik] / 2:
                print abs(kgrid[tp]['energy'][ib_pm][ik_pm] - \
                           kgrid[tp]['energy'][ib][ik])
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
                    A_pm ** 2 * log((k_pm + k) / (abs_kdiff + 1e-4)) - A_pm * c_ * c_pm - a * a_pm * c_ * c_pm)
                # because in the scalar form k+ or k- is suppused to be unique, here we take average
                S_o[j] += (N_POP + j + (-1) ** j * f_pm) * lamb_opm
                S_o_th[j] += (N_POP + j + (-1) ** j * f_pm_th) * lamb_opm

            lamb_ipm = beta_pm * (
                (k_pm ** 2 + k ** 2) / (2 * k * k_pm) * \
                A_pm ** 2 * log((k_pm + k) / (abs_kdiff + 1e-4)) - A_pm ** 2 - c_ ** 2 * c_pm ** 2 / 3)
            S_i[j] += (N_POP + (1 - j) + (-1) ** (1 - j) * f) * lamb_ipm * g_pm
            S_i_th[j] += (N_POP + (1 - j) + (-1) ** (1 - j) * f_th) * lamb_ipm * g_pm_th

        if counted > 0:
            S_i[j] /= counted
            S_i_th[j] /= counted
            S_o[j] /= counted
            S_o_th[j] /= counted

    return [sum(S_i), sum(S_i_th), sum(S_o), sum(S_o_th)]


def remove_duplicate_kpoints(kpts, dk=0.0001):
    """kpts (list of list): list of coordinates of electrons
     ALWAYS return either a list or ndarray: BE CONSISTENT with the input!!!

     Attention: it is better to call this method only once as calculating the norms takes time.
     """
    rm_list = []

    kdist = [norm(k) for k in kpts]
    ktuple = zip(kdist, kpts)
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