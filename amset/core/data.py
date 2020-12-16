import logging
import time
from collections import defaultdict
from os.path import join as joinpath
from typing import Dict, List, Optional

import numpy as np
from monty.json import MSONable
from monty.serialization import dumpfn
from pymatgen import Spin, Structure
from tabulate import tabulate

from amset.constants import bohr_to_cm, boltzmann_au, cm_to_bohr
from amset.constants import defaults as defaults
from amset.constants import ev_to_hartree, hartree_to_ev, spin_name
from amset.electronic_structure.common import (
    get_angstrom_structure,
    get_cbm_energy,
    get_vbm_energy,
)
from amset.electronic_structure.dos import FermiDos
from amset.electronic_structure.fd import dfdde
from amset.electronic_structure.tetrahedron import TetrahedralBandStructure
from amset.interpolation.momentum import MRTACalculator
from amset.io import write_mesh
from amset.log import log_list, log_time_taken
from amset.util import cast_dict_list, groupby, tensor_average

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

logger = logging.getLogger(__name__)
_kpt_str = "[{k[0]:.5f} {k[1]:.5f} {k[2]:.5f}]"


class AmsetData(MSONable):
    def __init__(
        self,
        structure: Structure,
        energies: Dict[Spin, np.ndarray],
        vvelocities_product: Dict[Spin, np.ndarray],
        velocities: Dict[Spin, np.ndarray],
        kpoint_mesh: np.ndarray,
        kpoints: np.ndarray,
        ir_kpoints_idx: np.ndarray,
        ir_to_full_kpoint_mapping: np.ndarray,
        tetrahedra: np.ndarray,
        ir_tetrahedra_info: np.ndarray,
        efermi: float,
        num_electrons: float,
        is_metal: bool,
        soc: bool,
        vb_idx: Optional[Dict[Spin, int]] = None,
    ):
        self.structure = structure
        self.velocities_product = vvelocities_product
        self.kpoint_mesh = kpoint_mesh
        self.intrinsic_fermi_level = efermi
        self.ir_to_full_kpoint_mapping = ir_to_full_kpoint_mapping
        self._soc = soc
        self.num_electrons = num_electrons
        self.is_metal = is_metal
        self.vb_idx = vb_idx
        self.spins = list(energies.keys())
        self.velocities = {s: v.transpose((0, 2, 1)) for s, v in velocities.items()}

        self.dos = None
        self.scattering_rates = None
        self.scattering_labels = None
        self.doping = None
        self.temperatures = None
        self.fermi_levels = None
        self.electron_conc = None
        self.hole_conc = None
        self.conductivity = None
        self.seebeck = None
        self.electronic_thermal_conductivity = None
        self.mobility = None
        self.overlap_calculator = None
        self.mrta_calculator = None
        self.fd_cutoffs = None

        self.grouped_ir_to_full = groupby(
            np.arange(len(kpoints)), ir_to_full_kpoint_mapping
        )

        self.tetrahedral_band_structure = TetrahedralBandStructure.from_data(
            energies,
            kpoints,
            tetrahedra,
            structure,
            ir_kpoints_idx,
            ir_to_full_kpoint_mapping,
            *ir_tetrahedra_info
        )

        logger.info("Initializing momentum relaxation time factor calculator")
        self.mrta_calculator = MRTACalculator.from_data(kpoints, self.velocities)

    @property
    def energies(self):
        return self.tetrahedral_band_structure.energies

    @property
    def kpoints(self):
        return self.tetrahedral_band_structure.kpoints

    @property
    def ir_kpoints(self):
        return self.kpoints[self.tetrahedral_band_structure.ir_kpoints_idx]

    @property
    def ir_kpoints_idx(self):
        return self.tetrahedral_band_structure.ir_kpoints_idx

    def set_overlap_calculator(self, overlap_calculator):
        if overlap_calculator is not None:
            equal = check_nbands_equal(overlap_calculator, self)
            if not equal:
                raise RuntimeError(
                    "Overlap calculator does not have the correct number of bands\n"
                    "If using wavefunction coefficients, ensure they were generated "
                    "using the same energy_cutoff (not encut)"
                )

        self.overlap_calculator = overlap_calculator

    def calculate_dos(
        self,
        estep: float = defaults["dos_estep"],
        progress_bar: bool = defaults["print_log"],
    ):
        """
        Args:
            estep: The DOS energy step in eV, where smaller numbers give more
                accuracy but are more expensive.
            progress_bar: Show a progress bar for DOS calculation.
        """
        emin = np.min([np.min(spin_eners) for spin_eners in self.energies.values()])
        emax = np.max([np.max(spin_eners) for spin_eners in self.energies.values()])
        epoints = int(round((emax - emin) / (estep * ev_to_hartree)))
        energies = np.linspace(emin, emax, epoints)
        dos_weight = 1 if self._soc or len(self.spins) == 2 else 2

        logger.info("DOS parameters:")
        log_list(
            [
                "emin: {:.2f} eV".format(emin * hartree_to_ev),
                "emax: {:.2f} eV".format(emax * hartree_to_ev),
                "dos weight: {}".format(dos_weight),
                "n points: {}".format(epoints),
            ]
        )

        logger.info("Generating tetrahedral DOS:")
        t0 = time.perf_counter()
        emesh, dos = self.tetrahedral_band_structure.get_density_of_states(
            energies=energies, progress_bar=progress_bar
        )
        log_time_taken(t0)

        num_electrons = self.num_electrons if self.is_metal else None

        self.dos = FermiDos(
            self.intrinsic_fermi_level,
            emesh,
            dos,
            self.structure,
            atomic_units=True,
            dos_weight=dos_weight,
            num_electrons=num_electrons,
        )

    def set_doping_and_temperatures(self, doping: np.ndarray, temperatures: np.ndarray):
        if not self.dos:
            raise RuntimeError(
                "The DOS should be calculated (AmsetData.calculate_dos) before "
                "setting doping levels."
            )

        if doping is None:
            # Generally this is for metallic systems; here we use the intrinsic Fermi
            # level
            self.doping = [0]
            print("doping is none")
        else:
            self.doping = doping * (1 / cm_to_bohr) ** 3

        self.temperatures = temperatures

        self.fermi_levels = np.zeros((len(doping), len(temperatures)))
        self.electron_conc = np.zeros((len(doping), len(temperatures)))
        self.hole_conc = np.zeros((len(doping), len(temperatures)))

        fermi_level_info = []
        tols = np.logspace(-5, 0, 6)
        for n, t in np.ndindex(self.fermi_levels.shape):
            for i, tol in enumerate(tols):
                # Finding the Fermi level is quite fickle. Enumerate multiple
                # tolerances and use the first one that works!
                try:
                    if self.doping[n] == 0:
                        self.fermi_levels[n, t] = self.dos.get_fermi_from_num_electrons(
                            self.num_electrons,
                            temperatures[t],
                            tol=tol / 1000,
                            precision=10,
                        )
                    else:
                        (
                            self.fermi_levels[n, t],
                            self.electron_conc[n, t],
                            self.hole_conc[n, t],
                        ) = self.dos.get_fermi(
                            self.doping[n],
                            temperatures[t],
                            tol=tol,
                            precision=10,
                            return_electron_hole_conc=True,
                        )
                    break
                except ValueError:
                    if i == len(tols) - 1:
                        raise ValueError(
                            "Could not calculate Fermi level position."
                            "Try a denser k-point mesh."
                        )
                    else:
                        pass

            fermi_level_info.append(
                (doping[n], temperatures[t], self.fermi_levels[n, t] * hartree_to_ev)
            )

        table = tabulate(
            fermi_level_info,
            headers=("conc [cm⁻³]", "temp [K]", "E_fermi [eV]"),
            numalign="right",
            stralign="center",
            floatfmt=(".2e", ".1f", ".4f"),
        )
        logger.info("Calculated Fermi levels:")
        logger.info(table)

    def calculate_fd_cutoffs(
        self,
        fd_tolerance: Optional[float] = 0.01,
        cutoff_pad: float = 0.0,
        max_moment: int = 2,
        mobility_rates_only: bool = False,
    ):
        energies = self.dos.energies
        vv = {s: v.transpose((0, 3, 1, 2)) for s, v in self.velocities_product.items()}
        _, vvdos = self.tetrahedral_band_structure.get_density_of_states(
            energies, integrand=vv, sum_spins=True, use_cached_weights=True
        )
        vvdos = tensor_average(vvdos)
        # vvdos = np.array(self.dos.get_densities())

        # three fermi integrals govern transport properties:
        #   1. df/de controls conductivity and mobility
        #   2. (e-u) * df/de controls Seebeck
        #   3. (e-u)^2 df/de controls electronic thermal conductivity
        # take the absolute sum of the integrals across all doping and
        # temperatures. this gives us the energies that are important for
        # transport
        if fd_tolerance:

            def get_min_max_cutoff(cumsum):
                min_idx = np.where(cumsum < fd_tolerance / 2)[0].max()
                max_idx = np.where(cumsum > (1 - fd_tolerance / 2))[0].min()
                return energies[min_idx], energies[max_idx]

            min_cutoff = np.inf
            max_cutoff = -np.inf
            for n, t in np.ndindex(self.fermi_levels.shape):
                ef = self.fermi_levels[n, t]
                temp = self.temperatures[t]
                dfde = -dfdde(energies, ef, temp * boltzmann_au)

                for moment in range(max_moment + 1):
                    weight = np.abs((energies - ef) ** moment * dfde)
                    weight_dos = weight * vvdos
                    weight_cumsum = np.cumsum(weight_dos)
                    weight_cumsum /= np.max(weight_cumsum)

                    cmin, cmax = get_min_max_cutoff(weight_cumsum)
                    min_cutoff = min(cmin, min_cutoff)
                    max_cutoff = max(cmax, max_cutoff)

                    # import matplotlib.pyplot as plt
                    # ax = plt.gca()
                    # plt.plot(energies / units.eV, weight / weight.max())
                    # plt.plot(energies / units.eV, vvdos / vvdos.max())
                    # plt.plot(energies / units.eV, weight_dos / weight_dos.max())
                    # plt.plot(energies / units.eV, weight_cumsum / weight_cumsum.max())
                    # ax.set(xlim=(4, 7.5))
                    # plt.show()

        else:
            min_cutoff = energies.min()
            max_cutoff = energies.max()

        if mobility_rates_only:
            vbm = get_vbm_energy(self.energies, self.vb_idx)
            cbm = get_cbm_energy(self.energies, self.vb_idx)
            mid_gap = (cbm + vbm) / 2
            if np.all(self.doping < 0):
                # only electron mobility so don't calculate valence band rates
                min_cutoff = max(min_cutoff, mid_gap)
            elif np.all(self.doping < 0):
                # only hole mobility so don't calculate conudction band rates
                max_cutoff = min(max_cutoff, mid_gap)

        min_cutoff -= cutoff_pad
        max_cutoff += cutoff_pad

        logger.info("Calculated Fermi–Dirac cut-offs:")
        log_list(
            [
                "min: {:.3f} eV".format(min_cutoff * hartree_to_ev),
                "max: {:.3f} eV".format(max_cutoff * hartree_to_ev),
            ]
        )
        self.fd_cutoffs = (min_cutoff, max_cutoff)

    def set_scattering_rates(
        self, scattering_rates: Dict[Spin, np.ndarray], scattering_labels: List[str]
    ):
        for spin in self.spins:
            s = (len(self.doping), len(self.temperatures)) + self.energies[spin].shape
            if scattering_rates[spin].shape[1:] != s:
                raise ValueError(
                    "Shape of scattering_type rates array does not match the "
                    "number of dopings, temperatures, bands, or kpoints"
                )

            if scattering_rates[spin].shape[0] != len(scattering_labels):
                raise ValueError(
                    "Number of scattering_type rates does not match number of "
                    "scattering_type labels"
                )

        self.scattering_rates = scattering_rates
        self.scattering_labels = scattering_labels

    def fill_rates_outside_cutoffs(self, fill_value=None):
        if self.scattering_rates is None:
            raise ValueError("Scattering rates must be set before being filled")

        min_fd, max_fd = self.fd_cutoffs
        snt_fill = fill_value
        for spin, spin_energies in self.energies.items():
            mask = (spin_energies < min_fd) | (spin_energies > max_fd)
            any_in_mask = np.any(~mask)  # sometimes no rates calculated for spin
            rate_info = defaultdict(list)
            for s, n, t in np.ndindex(self.scattering_rates[spin].shape[:3]):
                if fill_value is None and any_in_mask:
                    # get average log rate inside cutoffs
                    snt_fill = np.log(self.scattering_rates[spin][s, n, t, ~mask])
                    snt_fill = np.exp(snt_fill.mean())
                elif fill_value is None:
                    # no rates have been calculated for this spin channel so cannot
                    # average them. In this rare case we use a rate of 10^14
                    snt_fill = 1e14

                rate_info[self.scattering_labels[s]].append(snt_fill)
                self.scattering_rates[spin][s, n, t, mask] = snt_fill

            if len(self.spins) == 1:
                logger.info("Filling scattering rates [s⁻¹] outside FD cutoffs with:")
            else:
                logger.info(
                    "Filling {} scattering rates [s⁻¹] outside FD cutoffs "
                    "with:".format(spin_name[spin])
                )

            headers = ["conc [cm⁻³]", "temp [K]"]
            headers += ["{}".format(s) for s in self.scattering_labels]
            rate_table = []
            for i, (n, t) in enumerate(np.ndindex(self.fermi_levels.shape)):
                col = [self.doping[n] * (1 / bohr_to_cm) ** 3, self.temperatures[t]]
                col += [rate_info[s][i] for s in self.scattering_labels]
                rate_table.append(col)

            table = tabulate(
                rate_table,
                headers=headers,
                numalign="right",
                stralign="center",
                floatfmt=[".2e", ".1f"] + [".2e"] * len(self.scattering_labels),
            )
            logger.info(table)

    def set_transport_properties(
        self,
        conductivity: np.ndarray,
        seebeck: np.ndarray,
        electronic_thermal_conductivity: np.ndarray,
        mobility: Optional[np.ndarray] = None,
    ):
        self.conductivity = conductivity
        self.seebeck = seebeck
        self.electronic_thermal_conductivity = electronic_thermal_conductivity
        self.mobility = mobility

    def to_dict(self, include_mesh=defaults["write_mesh"]):
        data = {
            "doping": (self.doping * cm_to_bohr ** 3).round(),
            "temperatures": self.temperatures,
            "fermi_levels": self.fermi_levels * hartree_to_ev,
            "conductivity": self.conductivity,
            "seebeck": self.seebeck,
            "electronic_thermal_conductivity": self.electronic_thermal_conductivity,
            "mobility": self.mobility,
        }

        if include_mesh:
            rates = self.scattering_rates
            energies = self.energies
            vel = self.velocities

            ir_rates = {s: r[..., self.ir_kpoints_idx] for s, r in rates.items()}
            ir_energies = {
                s: e[:, self.ir_kpoints_idx] * hartree_to_ev
                for s, e in energies.items()
            }
            ir_vel = {s: v[:, self.ir_kpoints_idx] for s, v in vel.items()}

            mesh_data = {
                "energies": ir_energies,
                "kpoints": self.kpoints,
                "ir_kpoints": self.ir_kpoints,
                "ir_to_full_kpoint_mapping": self.ir_to_full_kpoint_mapping,
                "efermi": self.intrinsic_fermi_level * hartree_to_ev,
                "vb_idx": self.vb_idx,
                "num_electrons": self.num_electrons,
                # "dos": self.dos,  # TODO: Convert dos to eV
                "velocities": ir_vel,  # TODO: convert units
                "scattering_rates": ir_rates,
                "scattering_labels": self.scattering_labels,
                "is_metal": self.is_metal,
                "fd_cutoffs": (
                    self.fd_cutoffs[0] * hartree_to_ev,
                    self.fd_cutoffs[1] * hartree_to_ev,
                ),
                "structure": get_angstrom_structure(self.structure),
                "soc": self._soc,
                "doping": data["doping"],
                "temperatures": data["temperatures"],
                "fermi_levels": data["fermi_levels"],
            }
            data["mesh"] = mesh_data
        return data

    def to_data(self):
        data = []

        triu = np.triu_indices(3)
        for n, t in np.ndindex(len(self.doping), len(self.temperatures)):
            row = [
                self.doping[n] * cm_to_bohr ** 3,
                self.temperatures[t],
                self.fermi_levels[n, t] * hartree_to_ev,
            ]
            row.extend(self.conductivity[n, t][triu])
            row.extend(self.seebeck[n, t][triu])
            row.extend(self.electronic_thermal_conductivity[n, t][triu])

            if self.mobility is not None:
                for mob in self.mobility.values():
                    row.extend(mob[n, t][triu])
            data.append(row)

        headers = ["doping[cm^-3]", "temperature[K]", "Fermi_level[eV]"]
        ds = ("xx", "xy", "xz", "yy", "yz", "zz")

        # TODO: confirm unit of kappa
        for prop, unit in [("cond", "S/m"), ("seebeck", "µV/K"), ("kappa", "?")]:
            headers.extend(["{}_{}[{}]".format(prop, d, unit) for d in ds])

        if self.mobility is not None:
            for name in self.mobility.keys():
                headers.extend(["{}_mobility_{}[cm^2/V.s]".format(name, d) for d in ds])

        return data, headers

    def to_file(
        self,
        directory: str = ".",
        prefix: Optional[str] = None,
        write_mesh_file: bool = defaults["write_mesh"],
        file_format: str = defaults["file_format"],
        suffix_mesh: bool = True,
    ):
        if self.conductivity is None:
            raise ValueError("Can't write AmsetData, transport properties not set")

        if not prefix:
            prefix = ""
        else:
            prefix += "_"

        if suffix_mesh:
            suffix = "_{}".format("x".join(map(str, self.kpoint_mesh)))
        else:
            suffix = ""

        if file_format in ["json", "yaml"]:
            data = self.to_dict()
            data = cast_dict_list(data)

            filename = joinpath(
                directory, "{}transport{}.{}".format(prefix, suffix, file_format)
            )
            dumpfn(data, filename, indent=4)

        elif file_format in ["csv", "txt"]:
            # don't write the data as JSON, instead write raw text files
            data, headers = self.to_data()
            filename = joinpath(
                directory, "{}transport{}.{}".format(prefix, suffix, file_format)
            )
            np.savetxt(filename, data, header=" ".join(headers))
        else:
            raise ValueError("Unrecognised output format: {}".format(file_format))

        if write_mesh_file:
            mesh_data = self.to_dict(include_mesh=True)["mesh"]
            mesh_filename = joinpath(directory, "{}mesh{}.h5".format(prefix, suffix))
            write_mesh(mesh_data, filename=mesh_filename)
            return filename, mesh_filename
        else:
            return filename


def check_nbands_equal(interpolator, amset_data):
    nbands_equal = [
        amset_data.energies[s].shape[0] == interpolator.nbands[s]
        for s in amset_data.spins
    ]
    return np.all(nbands_equal)
