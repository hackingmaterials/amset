from math import log

import numpy as np

from amset.constants import k_B, hbar, m_e, e

"""
Contains method for calculating electron distribution and optical phonon
scattering rate, etc.
"""


def f0(energy, fermi, temperature):
    """
    Returns the value of Fermi-Dirac distribution at equilibrium.

    Args:
        energy (float): energy in eV
        fermi (float): the Fermi level with the same reference as E (in eV)
        temperature (float): the absolute temperature in Kelvin.

    Returns (0<float<1):
        The occupation calculated by Fermi dirac
    """
    return 1. / (1. + np.exp((energy - fermi) / (k_B * temperature)))


def df0de(energy, fermi, temperature):
    """
    Returns the energy derivative of the Fermi-Dirac equilibrium distribution

    Args: see Args for f0(energy, fermi, temperature)

    Returns (float<0): the energy derivative of the Fermi-Dirac distribution.
    """
    exponent = (energy - fermi) / (k_B * temperature)
    # This is necessary so at too low numbers python doesn't return NaN
    if exponent > 40 or exponent < -40:
        return 1e-32
    else:
        return -1 / (k_B * temperature) * \
               np.exp((energy - fermi) / (k_B * temperature)) / (
                       1 + np.exp((energy - fermi) / (k_B * temperature))) ** 2


def fermi_integral(order, fermi, temperature, initial_energy=0):
    """
    Returns the Fermi integral
    (e.g. for calculating single parabolic band acoustic phonon mobility)

    Args:
        order (int): the order of integral
        fermi (float): absolute band structure fermi (not relative to CBM/VBM)
        temperature (float): the temperature in kelvin
        initial_energy (float): the actual CBM/VBM energy in eV
    """
    fermi = fermi - initial_energy
    integral = 0.
    nsteps = 100000

    # TODO: 1e6 works better (converges!) but for faster test we use 1e5
    emesh = np.linspace(0.0, 30 * k_B * temperature, nsteps)
    de = (emesh[-1] - emesh[0]) / (nsteps - 1.0)
    for energy in emesh:
        integral += de * (energy / (k_B * temperature)) ** order / (
                1. + np.exp((energy - fermi) / (k_B * temperature)))
    return integral


def free_e_dos(energy, m_eff=0.001):
    """
    The density of states (dos) of a free-electron with effective mass m_eff
        that consequently has a parabolic band.

    Args:
        energy (float): energy in eV relative to the CBM (or VBM for valence)
        m_eff (float): unitless effective mass

    Returns (float): the dos
    """
    volume = 4. / 3. * pi * (
            2 * m_eff * m_e * energy / (hbar ** 2 * 1e10 * e)) ** 1.5
    return volume / (.2 * pi ** 2) * (
            2 * m_e * m_eff /
            hbar ** 2) ** 1.5 * 1e-30 / e ** 1.5 * energy ** 0.5


def calculate_sio(tp, c, temperature, ib, ik, once_called, kgrid, cbm_vbm,
                  epsilon_s, epsilon_inf):
    """
    Calculates the polar optical phonon "in" and "out" scattering rates.
    This method is defined outside of the Amset class to enable parallelization

    Args:
        tp (str): type of the bands
            options: "n" for the conduction and "p" for the valence bands
        c (float): the carrier concentration in 1/cm3
        temperature (float): the temperature in Kelvin
        ib (int): the band index
        ik (int): the k-point index
        once_called (bool): whether this function was once called hence s_o and
            S_o_th calculated once or not. Caches already calculated properties
        kgrid (dict): the main kgrid variable in Amset (Amset.kgrid)
        cbm_vbm (dict): from Amset.cbm_vbm, containing cbm and vbm energy
        epsilon_s (float): static dielectric constant
        epsilon_inf (float): high-frequency dielectric constant

    Returns ([four 3x1 lists]):
        the overall vectors for s_i, s_i_th, s_o, and S_o_th
    """
    s_i = [np.array([1e-32, 1e-32, 1e-32]), np.array([1e-32, 1e-32, 1e-32])]
    s_o = [np.array([1e-32, 1e-32, 1e-32]), np.array([1e-32, 1e-32, 1e-32])]

    k = kgrid[tp]["norm(k)"][ib][ik]
    a = kgrid[tp]["a"][ib][ik]
    c_ = kgrid[tp]["c"][ib][ik]
    f = kgrid[tp]["f"][c][temperature][ib][ik]
    n_pop = kgrid[tp]["N_POP"][c][temperature][ib][ik]

    for j, X_Epm in enumerate(["X_Eplus_ik", "X_Eminus_ik"]):
        if tp == "n" and X_Epm == "X_Eminus_ik" and kgrid[tp]["energy"][ib][
            ik] - hbar * \
                kgrid[tp]["W_POP"][ib][ik] < cbm_vbm[tp]["energy"]:
            continue
        if tp == "p" and X_Epm == "X_Eplus_ik" and kgrid[tp]["energy"][ib][
            ik] + hbar * \
                kgrid[tp]["W_POP"][ib][ik] > cbm_vbm[tp]["energy"]:
            continue
        counted = len(kgrid[tp][X_Epm][ib][ik])

        for X_ib_ik in kgrid[tp][X_Epm][ib][ik]:
            X, ib_pm, ik_pm = X_ib_ik
            k_pm = kgrid[tp]["norm(k)"][ib_pm][ik_pm]
            abs_kdiff = abs(k_pm - k)

            if abs_kdiff < 1e-4 or k < 1e-4 or k_pm < 1e-4:
                # avoid rate blow-up (e.g. due to self-scattering)
                counted -= 1
                continue
            if abs(kgrid[tp]['energy'][ib_pm][ik_pm] -
                   kgrid[tp]['energy'][ib][ik]) < \
                    hbar * kgrid[tp]["W_POP"][ib][ik] / 2.0:
                counted -= 1
                continue

            g_pm = kgrid[tp]["g"][c][temperature][ib_pm][ik_pm]
            # 3**0.5 is to treat each direction as 1D BS
            v_pm = kgrid[tp]["norm(v)"][ib_pm][ik_pm] / sq3
            a_pm = kgrid[tp]["a"][ib_pm][ik_pm]
            c_pm = kgrid[tp]["c"][ib_pm][ik_pm]

            if tp == "n":
                f_pm = kgrid[tp]["f"][c][temperature][ib_pm][ik_pm]
            else:
                f_pm = 1 - kgrid[tp]["f"][c][temperature][ib_pm][ik_pm]

            A_pm = a * a_pm + c_ * c_pm * (k_pm ** 2 + k ** 2) / (2 * k_pm * k)
            beta_pm = (e ** 2 * kgrid[tp]["W_POP"][ib_pm][ik_pm]
                       ) / (4 * pi * hbar * v_pm) * \
                      (1 / (epsilon_inf * epsilon_0) - 1 /
                       (epsilon_s * epsilon_0)) * 6.2415093e20

            if not once_called:
                lamb_opm = beta_pm * (A_pm ** 2 * log((k_pm + k) / (
                    abs_kdiff)) - A_pm * c_ * c_pm - a * a_pm * c_ * c_pm)
                # because in the scalar form k+ or k- is supposed to be unique,
                # here we take average
                s_o[j] += (n_pop + j + (-1) ** j * f_pm) * lamb_opm

            lamb_ipm = beta_pm * ((k_pm ** 2 + k ** 2) / (2 * k * k_pm) *
                                  A_pm ** 2 * log((k_pm + k) / abs_kdiff) -
                                  A_pm ** 2 - c_ ** 2 * c_pm ** 2 / 3.0)
            s_i[j] += (n_pop + (1 - j) + (-1) ** (1 - j) * f) * lamb_ipm * g_pm

        if counted > 0:
            s_i[j] /= counted
            s_o[j] /= counted
    return [sum(s_i), sum(s_o)]
