import logging

from os.path import join as joinpath
from typing import Optional, Dict, List, Tuple

import numpy as np

from monty.json import MSONable
from monty.serialization import dumpfn
from BoltzTraP2 import units
from BoltzTraP2.fd import dFDde, FD

from pymatgen import Spin, Structure

from amset.constants import hbar, hartree_to_ev, m_to_cm, A_to_m
from amset.misc.util import groupby, cast_dict
from amset.misc.log import log_list
from amset.dos import FermiDos, ADOS
from amset import amset_defaults as defaults

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"
__date__ = "June 21, 2019"

logger = logging.getLogger(__name__)
_kpt_str = '[{k[0]:.5f} {k[1]:.5f} {k[2]:.5f}]'
pdefaults = defaults["performance"]


class AmsetData(MSONable):

    def __init__(self,
                 structure: Structure,
                 energies: Dict[Spin, np.ndarray],
                 vvelocities_product: Dict[Spin, np.ndarray],
                 projections: Dict[Spin, Dict[str, np.ndarray]],
                 kpoint_mesh: np.ndarray,
                 full_kpoints: np.ndarray,
                 ir_kpoints: np.ndarray,
                 ir_kpoints_idx: np.ndarray,
                 ir_to_full_kpoint_mapping: np.ndarray,
                 efermi: float,
                 is_metal: bool,
                 soc: bool,
                 kpoint_weights: Optional[np.ndarray] = None,
                 dos: Optional[FermiDos] = None,
                 vb_idx: Optional[Dict[Spin, int]] = None,
                 conductivity: Optional[np.ndarray] = None,
                 seebeck: Optional[np.ndarray] = None,
                 electronic_thermal_conductivity: Optional[np.ndarray] = None,
                 mobility: Optional[Dict[str, np.ndarray]] = None,
                 fd_cutoffs: Optional[Tuple[float, float]] = None,
                 scissor: Optional[Tuple] = None
                 ):
        self.structure = structure
        self.energies = energies
        self.velocities_product = vvelocities_product
        self.kpoint_mesh = kpoint_mesh
        self.full_kpoints = full_kpoints
        self.ir_kpoints = ir_kpoints
        self.ir_kpoints_idx = ir_kpoints_idx
        self.ir_to_full_kpoint_mapping = ir_to_full_kpoint_mapping
        self._projections = projections
        self._efermi = efermi
        self._soc = soc
        self.conductivity = conductivity
        self.seebeck = seebeck
        self.electronic_thermal_conductivity = electronic_thermal_conductivity
        self.mobility = mobility
        self.scissor = scissor if scissor else 0

        self.dos = dos
        self.is_metal = is_metal
        self.vb_idx = None if is_metal else vb_idx
        self.spins = self.energies.keys()
        self.kpoint_norms = np.linalg.norm(full_kpoints, axis=1)

        self.a_factor = {}
        self.c_factor = {}
        self._calculate_orbital_factors()

        if kpoint_weights is None:
            self.kpoint_weights = np.ones(len(full_kpoints)) / len(full_kpoints)
        else:
            self.kpoint_weights = kpoint_weights

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
        self.fd_cutoffs = fd_cutoffs

        self.grouped_ir_to_full = groupby(
            np.arange(len(full_kpoints)), ir_to_full_kpoint_mapping)

    def calculate_dos(self, estep: float = pdefaults["dos_estep"]):
        """
        Args:
            estep: The DOS energy step in eV, where smaller numbers give more
                accuracy but are more expensive.
        """
        emin = np.min([np.min(spin_eners) for spin_eners in
                       self.energies.values()])
        emax = np.max([np.max(spin_eners) for spin_eners in
                       self.energies.values()])
        epoints = int(round((emax - emin) / (estep * units.eV)))

        logger.debug("DOS parameters:")
        log_list(["emin: {:.2f} eV".format(emin / units.eV),
                  "emax: {:.2f} eV".format(emax / units.eV),
                  "n points: {}".format(epoints)])

        dos = {}
        emesh = None
        for spin in self.spins:
            kpoint_weights = np.tile(self.kpoint_weights,
                                     (len(self.energies[spin]), 1))
            emesh, dos[spin] = ADOS(
                self.energies[spin].T, erange=(emin, emax), npts=epoints,
                weights=kpoint_weights.T)

        dos_weight = 1 if self._soc or len(self.spins) == 2 else 2
        self.dos = FermiDos(
            self._efermi, emesh, dos, self.structure, atomic_units=True,
            dos_weight=dos_weight)

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
            self.fermi_levels[n, t], self.electron_conc[n, t], \
                self.hole_conc[n, t] = self.dos.get_fermi(
                    doping[n], temperatures[t], tol=1e-5, precision=10,
                    return_electron_hole_conc=True)

            fermi_level_info.append("{:.2g} cm⁻³ & {} K: {:.4f} eV".format(
                doping[n], temperatures[t], self.fermi_levels[n, t] / units.eV))

        log_list(fermi_level_info)
        self._calculate_fermi_functions()

    def _calculate_fermi_functions(self):
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
                self.f[spin][n, t] = FD(
                    self.energies[spin],
                    self.fermi_levels[n, t],
                    self.temperatures[t] * units.BOLTZMANN)
                self.dfde[spin][n, t] = dFDde(
                    self.energies[spin],
                    self.fermi_levels[n, t],
                    self.temperatures[t] * units.BOLTZMANN)
                # velocities product has shape (nbands, 3, 3, nkpoints)
                # we want the diagonal of the 3x3 matrix for each k and band
                # after diagonalization shape is nbands, nkpoints, 3
                v = np.diagonal(np.sqrt(self.velocities_product[spin]),
                                axis1=1, axis2=2)
                v = v.transpose((0, 2, 1))
                v = np.abs(np.matmul(matrix_norm, v)) * factor
                v = v.transpose((0, 2, 1))
                self.dfdk[spin][n, t] = np.linalg.norm(
                    self.dfde[spin][n, t][..., None] * v * hbar, axis=2)

    def calculate_fd_cutoffs(self, fd_tolerance: Optional[float] = 0.01,
                             cutoff_pad: float = 0.):
        energies = self.dos.energies

        # three fermi integrals govern transport properties:
        #   1. df/de controls conductivity and mobility
        #   2. (e-u) * df/de controls Seebeck
        #   3. (e-u)^2 df/de controls electronic thermal conductivity
        # take the absolute sum of the integrals across all doping and
        # temperatures. this gives us the energies that are important for
        # transport

        weights = np.zeros(energies.shape)
        for n, t in np.ndindex(self.fermi_levels.shape):
            ef = self.fermi_levels[n, t]
            temp = self.temperatures[t]

            dfde = -dFDde(energies, ef, temp * units.BOLTZMANN)
            sigma_int = np.abs(dfde)
            seeb_int = np.abs((energies - ef) * dfde)
            ke_int = np.abs((energies - ef) ** 2 * dfde)

            # normalize the transport integrals and sum
            nt_weights = sigma_int / sigma_int.max()
            nt_weights += seeb_int / seeb_int.max()
            nt_weights += ke_int / ke_int.max()
            weights = np.maximum(weights, nt_weights)

        if not self.is_metal:
            # weights should be zero in the band gap as there will be no density
            vb_bands = [self.energies[s][:self.vb_idx[s] + 1] for s in self.spins]
            cb_bands = [self.energies[s][self.vb_idx[s] + 1:] for s in self.spins]
            vbm_e = np.max(vb_bands)
            cbm_e = np.min(cb_bands)
            weights[(energies > vbm_e) & (energies < cbm_e)] = 0

        weights /= np.max(weights)
        cumsum = np.cumsum(weights)
        cumsum /= np.max(cumsum)

        if fd_tolerance:
            min_cutoff = energies[cumsum < fd_tolerance / 2].max()
            max_cutoff = energies[cumsum > 1 - fd_tolerance / 2].min()
        else:
            min_cutoff = energies.min()
            max_cutoff = energies.max()

        min_cutoff -= cutoff_pad
        max_cutoff += cutoff_pad

        logger.info("Calculated Fermi–Dirac cut-offs:")
        log_list(["min: {:.3f} eV".format(min_cutoff / units.eV),
                  "max: {:.3f} eV".format(max_cutoff / units.eV)])
        self.fd_cutoffs = (min_cutoff, max_cutoff)

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

    def set_extra_kpoints(self,
                          extra_kpoints: np.ndarray,
                          extra_energies: Dict[Spin, np.ndarray],
                          extra_vvelocities: Dict[Spin, np.ndarray],
                          extra_projections: Dict[Spin, np.ndarray],
                          kpoint_weights: np.ndarray,
                          ir_kpoints_idx: np.ndarray,
                          ir_to_full_idx: np.ndarray):
        if len(self.full_kpoints) + len(extra_kpoints) != len(kpoint_weights):
            raise ValueError("Total number of k-points (full_kpoints + "
                             "extra_kpoints) does not equal number of kpoint "
                             "weights")

        self.kpoint_weights = kpoint_weights
        new_ir_idx = len(self.full_kpoints) + ir_kpoints_idx

        # add the extra data to the storage arrays
        self.full_kpoints = np.concatenate((self.full_kpoints, extra_kpoints))
        self.energies = {
            s: np.concatenate((self.energies[s], extra_energies[s]), axis=1)
            for s in self.spins}
        self.velocities_product = {
            s: np.concatenate((self.velocities_product[s],
                               extra_vvelocities[s]), axis=3)
            for s in self.spins}
        self._projections = {
            s: {
                l: np.concatenate((self._projections[s][l],
                                   extra_projections[s][l]), axis=1)
                for l in self._projections[s]
            } for s in self.spins}
        self._calculate_orbital_factors()

        self.ir_kpoints = np.concatenate((self.ir_kpoints,
                                          extra_kpoints[ir_kpoints_idx]))
        self.ir_kpoints_idx = np.concatenate((self.ir_kpoints_idx, new_ir_idx))

        # add additional indices to the end of the mapping. E.g. if
        # mapping was originally [1, 2, 2, 3, 3] and we have 3 extra k-points
        # the new mapping will be [1, 2, 2, 3, 3, 4, 5, 6]
        self.ir_to_full_kpoint_mapping = np.concatenate(
            (self.ir_to_full_kpoint_mapping,
             ir_to_full_idx + self.ir_to_full_kpoint_mapping.max() + 1))

        # recalculate kpoint norms and grouping
        self.kpoint_norms = np.linalg.norm(self.full_kpoints, axis=1)
        self.grouped_ir_to_full = groupby(
            np.arange(len(self.full_kpoints)), self.ir_to_full_kpoint_mapping)

        self._calculate_fermi_functions()

    def _calculate_orbital_factors(self):
        for spin in self.spins:
            self.a_factor[spin] = (self._projections[spin]["s"] / (
                    self._projections[spin]["s"] ** 2 +
                    self._projections[spin]["p"] ** 2) ** 0.5)
            self.c_factor[spin] = (1 - self.a_factor[spin] ** 2) ** 0.5

    def to_file(self,
                directory: str = '.',
                prefix: Optional[str] = None,
                write_mesh: bool = defaults["output"]["write_mesh"],
                file_format: str = defaults["output"]["file_format"],
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
            if self.kpoint_mesh is not None:
                mesh = "x".join(map(str, self.kpoint_mesh))
            else:
                # user supplied k-points
                mesh = len(self.full_kpoints)
            suffix = "_{}".format(mesh)
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
                ir_rates = {spin: rates[:, :, :, :, self.ir_kpoints_idx]
                            for spin, rates in self.scattering_rates.items()}
                ir_energies = {spin: energies[:, self.ir_kpoints_idx] / units.eV
                               for spin, energies in self.energies.items()}
                data.update({
                    "energies": cast_dict(ir_energies),
                    "kpoints": self.full_kpoints,
                    "kpoint_weights": self.kpoint_weights,
                    "ir_kpoints": self.ir_kpoints,
                    "ir_to_full_kpoint_mapping": self.ir_to_full_kpoint_mapping,
                    "dos": self.dos,
                    "efermi": self._efermi,
                    "vb_idx": cast_dict(self.vb_idx),
                    "scattering_rates": cast_dict(ir_rates),
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

            # todo: update this to only write the irreducible energies & rates

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
