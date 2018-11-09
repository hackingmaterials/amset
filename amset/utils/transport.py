
import numpy as np

from amset.utils.constants import k_B, hbar, m_e, e, pi, sq3, epsilon_0
from math import log


"""
Contains method for calculating electron distribution and optical phonon
scattering rate, etc.
"""


def f0(E, fermi, T):
    """
    Returns the value of Fermi-Dirac distribution at equilibrium.

    Args:
        E (float): energy in eV
        fermi (float): the Fermi level with the same reference as E (in eV)
        T (float): the absolute temperature in Kelvin.

    Returns (0<float<1):
        The occupation calculated by Fermi dirac
    """
    return 1. / (1. + np.exp((E - fermi) / (k_B * T)))


def df0dE(E, fermi, T):
    """
    Returns the energy derivative of the Fermi-Dirac equilibrium distribution

    Args: see Args for f0(E, fermi, T)

    Returns (float<0): the energy derivative of the Fermi-Dirac distribution.
    """
    exponent = (E - fermi) / (k_B * T)
    if exponent > 40 or exponent < -40:  # This is necessary so at too low numbers python doesn't return NaN
        return 1e-32
    else:
        return -1 / (k_B * T) * np.exp((E - fermi) / (k_B * T)) / (1 + np.exp((E - fermi) / (k_B * T))) ** 2



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


def free_e_dos(E, m_eff=0.001):
    """
    The density of states (dos) of a free-electron with effective mass m_eff
        that consequently has a parabolic band.

    Args:
        E (float): energy in eV relative to the CBM (or VBM for valence)
        m_eff (float): unitless effective mass

    Returns (float): the dos
    """
    volume = 4./3.*pi * (2*m_eff*m_e*E/(hbar**2*1e10*e))**1.5
    return volume/(.2*pi**2)*(2*m_e*m_eff/hbar**2)**1.5*1e-30/e**1.5*E**0.5


def find_fermi_SPB(cbm_vbm, c, T, tolerance=0.01, alpha=0.02, max_iter=1000):
    """
    Not tested! Returns the fermi level based on single parabolic band (SPB)
    assumption. Note that this function is currently not tested and not used
    in Amset

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
    This method is defined outside of the Amset class to enable parallelization

    Args:
        tp (str): type of the bands
            options: "n" for the conduction and "p" for the valence bands
        c (float): the carrier concentration in 1/cm3
        T (float): the temperature in Kelvin
        ib (int): the band index
        ik (int): the k-point index
        once_called (bool): whether this function was once called hence S_o and
            S_o_th calculated once or not. Caches already calculated properties
        kgrid (dict): the main kgrid variable in Amset (Amset.kgrid)
        cbm_vbm (dict): from Amset.cbm_vbm, containing cbm and vbm energy
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
            if abs_kdiff < 1e-4 or k<1e-4 or k_pm<1e-4:
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


def get_tp(c):
    """
    Returns "n" for n-type (electrons majority carrier, c<0) or "p" (p-type).
    """
    if c < 0:
        return "n"
    elif c > 0:
        return "p"
    else:
        raise ValueError("The carrier concentration cannot be zero! Amset stops now!")

