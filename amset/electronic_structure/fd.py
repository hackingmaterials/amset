import numpy as np


def fd(e, mu, kb_t):
    """Compute Fermi-Dirac occupancies.

    Args:
        e: array of energies
        mu: single value of the chemical potential
        kb_t: thermal energy at the temperature of interest

    Returns:
        An array with the same shape as the "e" argument containing the average
        Fermi-Dirac occupancies for each energy level.
    """
    if kb_t == 0.0:
        delta = e - mu
        nruter = np.where(delta < 0.0, 1.0, 0.0)
        nruter[np.isclose(delta, 0.0)] = 0.5
    else:
        x = np.asarray((e - mu) / kb_t)
        nruter = 1.0 / (np.exp(x) + 1.0)
    return nruter


def dfddx(x):
    """Compute the derivative of the Fermi-Dirac occupancies with respect to
    the recentered and scaled energy (e - mu) / kBT

    Args:
        x: recentered and scaled energies at which to compute the derivative

    Returns:
        An array with the same shape as x containing the derivatives.
    """
    c = np.cosh(0.5 * np.asarray(x))
    nruter = -0.25 / c / c
    return nruter


def dfdde(e, mu, kb_t):
    """Compute the derivative of Fermi-Dirac occupancies with respect to
    the energy.

    Args:
        e: array of energies
        mu: single value of the chemical potential
        kb_t: thermal energy at the temperature of interest

    Returns:
        An array with the same shape as the "e" argument containing the
        derivatives of the average Fermi-Dirac occupancies for each energy
        level.
    """
    factor = (e - mu) / kb_t
    dfde = dfddx(factor) / kb_t
    return dfde
