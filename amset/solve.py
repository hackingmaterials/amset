import numpy as np
from BoltzTraP2 import units
from BoltzTraP2.bandlib import fermiintegrals, BTPDOS, calc_Onsager_coefficients
from monty.json import MSONable

from amset.constants import e
from amset.core import ElectronicStructure


class BTESolver(MSONable):

    def __init__(self, solve_scatter_mobilities: bool = False):
        self.solve_scatter_mobilities = solve_scatter_mobilities

    def solve_bte(self, electronic_structure: ElectronicStructure):
        if not all([electronic_structure.doping,
                    electronic_structure.temperatures,
                    electronic_structure.scattering_rates]):
            raise ValueError("Electronic structure must contain dopings "
                             "temperatures and scattering rates")

        if self.solve_scatter_mobilities:
            raise ValueError("Scatter limited mobilities not implemented yet")

        energies = np.vstack([electronic_structure.energies[spin]
                              for spin in electronic_structure.spins])
        vv = np.vstack([electronic_structure.velocities_product[spin]
                        for spin in electronic_structure.spins])

        mobility
        for n, t in np.ndindex((len(electronic_structure.doping),
                                len(electronic_structure.temperatures))):
            sum_rates = [np.sum(
                electronic_structure.scattering_rates[s][:, n, t], axis=0)
                for s in electronic_structure.spins]
            lifetimes = 1 / np.vstack(sum_rates)

            # obtain the Fermi integrals the temperature and doping
            epsilon, dos, vvdos, cdos = BTPDOS(
                energies, vv, scattering_model=lifetimes,
                npts=len(electronic_structure.dos.energies))

            fermi_level = [5.940700 * units.eV]
            temp = np.array([300.])

            carriers, l0, l1, l2, lm11 = fermiintegrals(
                epsilon, dos, vvdos, mur=fermi_level, Tr=temp,
                dosweight=electronic_structure.dos_weight)

            volume = (electronic_structure.structure.lattice.volume *
                      units.Angstrom ** 3)

            # Rescale the carrier count into a volumetric density in cm**(-3)
            carriers = ((-carriers[0, ...] - electronic_structure.dos.nelecs) /
                        (volume / (units.Meter / 100.) ** 3))

            # Compute the Onsager coefficients from those Fermi integrals
            sigma, seebeck, kappa, hall = calc_Onsager_coefficients(
                l0, l1, l2, fermi_level, temp, volume)

            sigma = sigma[0, ...]
            mobility = sigma * 0.01 / (e * carriers[0])
            print("mobility: {}".format(mobility[0]))
