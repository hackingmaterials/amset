import logging

import scipy
from os.path import join as joinpath
from typing import Optional, Dict, List

import numpy as np

from monty.json import MSONable
from monty.serialization import dumpfn
from scipy.ndimage import gaussian_filter1d

from BoltzTraP2 import units
from BoltzTraP2.bandlib import DOS, FD, dFDde

from amset.constants import hbar, hartree_to_ev, m_to_cm, A_to_m
from amset.util import groupby, cast_dict
from amset.log import log_list
from amset.utils.transport import df0de
from pymatgen import Spin, Structure
from pymatgen.electronic_structure.dos import FermiDos, Dos, f0

logger = logging.getLogger(__name__)
_kpt_str = '[{k[0]:.5f} {k[1]:.5f} {k[2]:.5f}]'


class AmsetData(MSONable):

    def __init__(self,
                 structure: Structure,
                 energies: Dict[Spin, np.ndarray],
                 vvelocities_product: Dict[Spin, np.ndarray],
                 projections: Dict[Spin, Dict[str, np.ndarray]],
                 kpoint_mesh: np.ndarray,
                 full_kpoints: np.ndarray,
                 ir_kpoints: np.ndarray,
                 ir_kpoint_weights: np.ndarray,
                 ir_kpoints_idx: np.ndarray,
                 ir_to_full_kpoint_mapping: np.ndarray,
                 efermi: float,
                 is_metal: bool,
                 soc: bool,
                 dos: Optional[FermiDos] = None,
                 dos_weight: Optional[int] = None,
                 vb_idx: Optional[Dict[Spin, int]] = None,
                 conductivity: Optional[np.ndarray] = None,
                 seebeck: Optional[np.ndarray] = None,
                 electronic_thermal_conductivity: Optional[np.ndarray] = None,
                 mobility: Optional[Dict[str, np.ndarray]] = None,
                 fermi_dirac: Optional[np.ndarray] = None
                 ):
        self.structure = structure
        self.energies = energies
        self.velocities_product = vvelocities_product
        self.kpoint_mesh = kpoint_mesh
        self.full_kpoints = full_kpoints
        self.ir_kpoints = ir_kpoints
        self.ir_kpoint_weights = ir_kpoint_weights
        self.ir_kpoints_idx = ir_kpoints_idx
        self.ir_to_full_kpoint_mapping = ir_to_full_kpoint_mapping
        self._projections = projections
        self._efermi = efermi
        self._soc = soc
        self.conductivity = conductivity
        self.seebeck = seebeck
        self.electronic_thermal_conductivity = electronic_thermal_conductivity
        self.mobility = mobility
        self.fermi_dirac = fermi_dirac

        self.dos = dos
        self.dos_weight = dos_weight
        self.is_metal = is_metal
        self.vb_idx = None if is_metal else vb_idx
        self.spins = self.energies.keys()
        self.kpoint_norms = np.linalg.norm(full_kpoints, axis=1)

        self.a_factor = {}
        self.c_factor = {}

        for spin in self.spins:
            self.a_factor[spin] = (
                projections[spin]["s"] / (projections[spin]["s"] ** 2 +
                                          projections[spin]["p"] ** 2) ** 0.5)
            self.c_factor[spin] = (1 - self.a_factor[spin] ** 2) ** 0.5

        self.scattering_rates = None
        self.scattering_labels = None
        self.doping = None
        self.temperatures = None
        self.fermi_levels = None
        self.electron_conc = None
        self.hole_conc = None
        self.f = None
        self.dfde = None
        self.dfdk = None

        self.grouped_ir_to_full = groupby(
            np.arange(len(full_kpoints)), ir_to_full_kpoint_mapping)

    def calculate_dos(self, dos_estep: float = 0.01, dos_width: float = None):
        """

        Args:
            dos_estep: The DOS energy step, where smaller numbers give more
                accuracy but are more expensive.
            dos_width: The DOS gaussian smearing width in eV.
        """

        all_energies = np.vstack([self.energies[spin] for spin in self.spins])
        all_energies /= units.eV  # convert from Hartree to eV for DOS

        # add a few multiples of dos_width to emin and emax to account for tails
        pad = dos_width if dos_width else 0
        dos_emin = np.min(all_energies) - pad * 5
        dos_emax = np.max(all_energies) + pad * 5
        npts = int(round((dos_emax - dos_emin) / dos_estep))

        logger.debug("DOS parameters:")
        log_list(["emin: {:.2f} eV".format(dos_emin),
                  "emax: {:.2f} eV".format(dos_emax),
                  "broadening width: {} eV".format(dos_width)])

        emesh, densities = DOS(all_energies.T, erange=(dos_emin, dos_emax),
                               npts=npts)

        if dos_width:
            densities = gaussian_filter1d(densities, dos_width /
                                          (emesh[1] - emesh[0]))

        # integrate up to Fermi level to get number of electrons
        efermi = self._efermi / units.eV
        energy_mask = emesh <= efermi + pad
        nelect = scipy.trapz(densities[energy_mask], emesh[energy_mask])

        logger.debug("Intrinsic DOS Fermi level: {:.4f}".format(efermi))
        logger.debug("DOS contains {:.3f} electrons".format(nelect))

        dos = Dos(efermi, emesh, {Spin.up: densities})
        self.dos_weight = 1 if self._soc or len(self.spins) == 2 else 2
        self.dos = FermiDos(dos, structure=self.structure,
                            dos_weight=self.dos_weight)

    def set_doping_and_temperatures(self,
                                    doping: np.ndarray,
                                    temperatures: np.ndarray):
        if not self.dos:
            raise RuntimeError(
                "The DOS should be calculated (AmsetData.calculate_dos) before "
                "setting doping levels.")

        self.doping = doping
        self.temperatures = temperatures

        self.fermi_levels = np.zeros((len(doping), len(temperatures)))
        self.electron_conc = np.zeros((len(doping), len(temperatures)))
        self.hole_conc = np.zeros((len(doping), len(temperatures)))

        logger.info("Calculated Fermi levels:")

        fermi_level_info = []
        for n, t in np.ndindex(self.fermi_levels.shape):
            # do minus -c as FermiDos treats negative concentrations as electron
            # doping and +ve as hole doping (the opposite to amset).
            self.fermi_levels[n, t], self.electron_conc[n, t], \
                self.hole_conc[n, t] = self.dos.get_fermi(
                    -doping[n], temperatures[t], rtol=1e-4, precision=10,
                    return_electron_hole_conc=True)

            fermi_level_info.append("{:.2g} cm⁻³ & {} K: {:.4f} eV".format(
                doping[n], temperatures[t], self.fermi_levels[n, t]))

        log_list(fermi_level_info)

        # calculate Fermi dirac distributions and derivatives for each Fermi
        # level
        self.f = {s: np.zeros(self.fermi_levels.shape +
                              self.energies[s].shape)
                  for s in self.spins}
        self.dfde = {s: np.zeros(self.fermi_levels.shape +
                                 self.energies[s].shape)
                     for s in self.spins}
        self.dfdk = {s: np.zeros(self.fermi_levels.shape +
                                 self.energies[s].shape)
                     for s in self.spins}

        matrix_norm = (self.structure.lattice.matrix /
                       np.linalg.norm(self.structure.lattice.matrix))
        factor = hartree_to_ev * m_to_cm * A_to_m / (hbar * 0.52917721067)
        for spin in self.spins:
            for n, t in np.ndindex(self.fermi_levels.shape):
                self.f[spin][n, t] = f0(
                    self.energies[spin] / units.eV,
                    self.fermi_levels[n, t],
                    self.temperatures[t])
                self.dfde[spin][n, t] = df0de(
                    self.energies[spin] / units.eV,
                    self.fermi_levels[n, t],
                    self.temperatures[t])

                # velocities product has shape (nbands, 3, 3, nkpoints)
                # we want the diagonal of the 3x3 matrix for each k and band
                # after diagonalization shape is nbands, nkpoints, 3
                # v = np.diagonal(np.sqrt(self.velocities_product[spin]),
                #                 axis1=1, axis2=2)
                # v = v.transpose((0, 2, 1))
                # v = np.abs(np.matmul(matrix_norm, v)) * factor
                # v = v.transpose((0, 2, 1))
                # self.dfdk[spin][n, t] = np.linalg.norm(
                #     self.dfde[spin][n, t][..., None] * v * hbar, axis=2)

    def set_scattering_rates(self,
                             scattering_rates: Dict[Spin, np.ndarray],
                             scattering_labels: List[str]):
        for spin in self.spins:
            expected_shape = ((len(self.doping), len(self.temperatures)) +
                              self.energies[spin].shape)
            if scattering_rates[spin].shape[1:] != expected_shape:
                raise ValueError(
                    "Shape of scattering_type rates array does not match the "
                    "number of dopings, temperatures, bands, or kpoints")

            if scattering_rates[spin].shape[0] != len(scattering_labels):
                raise ValueError(
                    "Number of scattering_type rates does not match number of "
                    "scattering_type labels")

        self.scattering_rates = scattering_rates
        self.scattering_labels = scattering_labels

    def to_file(self,
                directory: str = '.',
                prefix: Optional[str] = None,
                write_mesh: bool = True,
                file_format: str = 'json',
                suffix_mesh: bool = True):
        if (self.conductivity is None or self.seebeck is None or
                self.electronic_thermal_conductivity is None):
            raise ValueError("Cannot write AmsetData to file, transport "
                             "properties not set")

        if not prefix:
            prefix = ''
        else:
            prefix += '_'

        if suffix_mesh:
            suffix = "_{}".format("x".join(map(str, self.kpoint_mesh)))
        else:
            suffix = ''

        if file_format in ["json", "yaml"]:
            data = {"doping": self.doping,
                    "temperature": self.temperatures,
                    "fermi_levels": self.fermi_levels,
                    "conductivity": self.conductivity,
                    "seebeck": self.seebeck,
                    "electronic_thermal_conductivity":
                        self.electronic_thermal_conductivity,
                    "mobility": self.mobility}

            if write_mesh:
                data.update({
                    "energies": {s.value: e / units.eV for s, e in
                                 self.energies.items()},
                    "kpoints": self.full_kpoints,
                    "dos": self.dos,
                    "efermi": self._efermi,
                    "vb_idx": cast_dict(self.vb_idx),
                    "scattering_rates": cast_dict(self.scattering_rates),
                    "scattering_labels": self.scattering_labels})

            filename = joinpath(directory, "{}amset_data{}.{}".format(
                prefix, suffix, file_format))
            dumpfn(data, filename)

        elif file_format in ["csv", "txt"]:
            # don't write the data as JSON, instead write raw text files
            data = []

            triu = np.triu_indices(3)
            for n, t in np.ndindex(len(self.doping), len(self.temperatures)):
                row = [self.doping[n], self.temperatures[t],
                       self.fermi_levels[n, t]]
                row.extend(self.conductivity[n, t][triu])
                row.extend(self.seebeck[n, t][triu])
                row.extend(self.electronic_thermal_conductivity[n, t][triu])

                if self.mobility is not None:
                    for mob in self.mobility.values():
                        row.extend(mob[n, t][triu])
                data.append(row)

            headers = ["doping[cm^-3]", "temperature[K]", "Fermi_level[eV]"]

            # TODO: confirm unit of kappa
            for prop, unit in [("cond", "S/m"), ("seebeck", "µV/K"),
                               ("kappa", "?")]:
                headers.extend(["{}_{}[{}]".format(prop, d, unit)
                                for d in ("xx", "xy", "xz", "yy", "yz", "zz")])

            if self.mobility is not None:
                for name in self.mobility.keys():
                    headers.extend(
                        ["{}_mobility_{}[cm^2/V.s]".format(name, d)
                         for d in ("xx", "xy", "xz", "yy", "yz", "zz")])

            filename = joinpath(directory, "{}amset_transport{}.{}".format(
                prefix, suffix, file_format))
            np.savetxt(filename, data, header=" ".join(headers))

            if not write_mesh:
                return

            # write separate files for k-point mesh, energy mesh and
            # temp/doping dependent scattering rate mesh
            kpt_file = joinpath(directory, "{}amset_k_mesh{}.{}".format(
                prefix, suffix, file_format))
            data = np.array([[i + 1, str(_kpt_str.format(k=kpt))] for i, kpt in
                             enumerate(self.full_kpoints)], dtype=object)
            np.savetxt(kpt_file, data, fmt=["%d", "%s"],
                       header="k-point_index frac_kpt_coord")

            # write energy mesh
            energy_file = joinpath(directory, "{}amset_e_mesh{}.{}".format(
                prefix, suffix, file_format))
            with open(energy_file, 'w') as f:
                for spin in self.spins:
                    eshape = self.energies[spin].shape
                    b_idx = np.repeat(np.arange(eshape[0]), eshape[1])
                    k_idx = np.tile(np.arange(eshape[1]), eshape[0])
                    data = np.column_stack(
                        (b_idx, k_idx, self.energies[spin].flatten()))

                    np.savetxt(f, data, fmt="%d %d %.8f",
                               header="band_index kpt_index energy[eV]")
                    f.write("\n")

            # write scatter mesh
            scatter_file = joinpath(directory, "{}amset_scatter{}.{}".format(
                prefix, suffix, file_format))
            # + 1 accounts for total rate column
            labels = self.scattering_labels + ["total"]
            fmt = "%d %d" + " %.5g" * len(labels)
            header = ["band_index", "kpt_index"]
            header += ["{}_rate[s^-1]".format(label) for label in labels]
            header = " ".join(header)
            with open(scatter_file, 'w') as f:
                for spin in self.spins:
                    for n, t in np.ndindex((len(self.doping),
                                            len(self.temperatures))):
                        f.write("# n = {:g} cm^-3, T = {} K E_F = {} eV\n"
                                .format(self.doping[n], self.temperatures[t],
                                        self.fermi_levels[n, t]))
                        shape = self.energies[spin].shape
                        b_idx = np.repeat(np.arange(shape[0]), shape[1]) + 1
                        k_idx = np.tile(np.arange(shape[1]), shape[0]) + 1
                        cols = [b_idx, k_idx]
                        cols.extend(self.scattering_rates[spin][:, n, t]
                                    .reshape(len(self.scattering_labels), -1))

                        # add "total rate" column
                        cols.extend(np.sum(self.scattering_rates[spin][:, n, t],
                                           axis=0).reshape(1, -1))
                        data = np.column_stack(cols)

                        np.savetxt(f, data, fmt=fmt, header=header)
                        f.write("\n")

        else:
            raise ValueError("Unrecognised output format: {}".format(
                file_format))
