import numpy as np

from amset.constants import boltzmann_au
from amset.electronic_structure.fd import fd

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"


def calculate_inverse_screening_length_sq(amset_data, dielectric):
    inverse_screening_length_sq = np.zeros(amset_data.fermi_levels.shape)

    tdos = amset_data.dos.tdos
    energies = amset_data.dos.energies
    fermi_levels = amset_data.fermi_levels
    vol = amset_data.structure.volume

    for n, t in np.ndindex(inverse_screening_length_sq.shape):
        ef = fermi_levels[n, t]
        temp = amset_data.temperatures[t]
        f = fd(energies, ef, temp * boltzmann_au)
        integral = np.trapz(tdos * f * (1 - f), x=energies)
        inverse_screening_length_sq[n, t] = (
            integral * 4 * np.pi / (dielectric * boltzmann_au * temp * vol)
        )

    return inverse_screening_length_sq
