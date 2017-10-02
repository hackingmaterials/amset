import numpy as np
from scipy.constants.codata import value as _cd
from math import log, pi


k_B = _cd("Boltzmann constant in eV/K")
epsilon_0 = 8.854187817e-12  # dielectric constant in vacuum [C**2/m**2N]
sq3 = 3 ** 0.5
hbar = _cd('Planck constant in eV s') / (2 * pi)
e = _cd('elementary charge')

def norm(v):
    """method to quickly calculate the norm of a vector (v: 1x3 or 3x1) as numpy.linalg.norm is slower for this case"""
    return (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5


def grid_norm(grid):
    return (grid[:,:,:,0]**2 + grid[:,:,:,1]**2 + grid[:,:,:,2]**2) ** 0.5


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
    # S_o = np.array([self.gs, self.gs, self.gs])

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
            g_pm = kgrid[tp]["g"][c][T][ib_pm][ik_pm]
            g_pm_th = kgrid[tp]["g_th"][c][T][ib_pm][ik_pm]
            v_pm = kgrid[tp]["norm(v)"][ib_pm][ik_pm] / sq3  # 3**0.5 is to treat each direction as 1D BS
            k_pm = kgrid[tp]["norm(k)"][ib_pm][ik_pm]
            abs_kdiff = abs(k_pm - k)
            if abs_kdiff < 1e-4:
                counted -= 1
                continue
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