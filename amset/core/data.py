import logging
import time
from os.path import join as joinpath
from typing import Dict, List, Optional, Tuple

import numpy as np
from BoltzTraP2 import units
from BoltzTraP2.fd import dFDde
from monty.json import MSONable
from monty.serialization import dumpfn

from amset.constants import amset_defaults as defaults
from amset.constants import cm_to_bohr
from amset.electronic_structure.dos import FermiDos
from amset.electronic_structure.overlap import OverlapCalculator
from amset.electronic_structure.tetrahedron import TetrahedralBandStructure
from amset.log import log_list, log_time_taken
from amset.util import cast_dict, groupby
from pymatgen import Spin, Structure

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"
__date__ = "June 21, 2019"

logger = logging.getLogger(__name__)
_kpt_str = "[{k[0]:.5f} {k[1]:.5f} {k[2]:.5f}]"


class AmsetData(MSONable):
    def __init__(
        self,
        structure: Structure,
        energies: Dict[Spin, np.ndarray],
        vvelocities_product: Dict[Spin, np.ndarray],
        effective_mass: Dict[Spin, np.ndarray],
        projections: Dict[Spin, Dict[str, np.ndarray]],
        kpoint_mesh: np.ndarray,
        full_kpoints: np.ndarray,
        ir_kpoints: np.ndarray,
        ir_kpoints_idx: np.ndarray,
        ir_to_full_kpoint_mapping: np.ndarray,
        tetrahedra: np.ndarray,
        ir_tetrahedra_info: np.ndarray,
        efermi: float,
        num_electrons: float,
        is_metal: bool,
        soc: bool,
        overlap_calculator: OverlapCalculator,
        dos: Optional[FermiDos] = None,
        vb_idx: Optional[Dict[Spin, int]] = None,
        conductivity: Optional[np.ndarray] = None,
        seebeck: Optional[np.ndarray] = None,
        electronic_thermal_conductivity: Optional[np.ndarray] = None,
        mobility: Optional[Dict[str, np.ndarray]] = None,
        fd_cutoffs: Optional[Tuple[float, float]] = None,
    ):
        self.structure = structure
        self.energies = energies
        self.velocities_product = vvelocities_product
        self.effective_mass = effective_mass
        self.kpoint_mesh = kpoint_mesh
        self.full_kpoints = full_kpoints
        self.ir_kpoints = ir_kpoints
        self.ir_kpoints_idx = ir_kpoints_idx
        self.ir_to_full_kpoint_mapping = ir_to_full_kpoint_mapping
        self._projections = projections
        self.intrinsic_fermi_level = efermi
        self._soc = soc
        self.num_electrons = num_electrons
        self.conductivity = conductivity
        self.seebeck = seebeck
        self.electronic_thermal_conductivity = electronic_thermal_conductivity
        self.mobility = mobility
        self.overlap_calculator = overlap_calculator

        self.dos = dos
        self.is_metal = is_metal
        self.vb_idx = None if is_metal else vb_idx
        self.spins = self.energies.keys()

        self.a_factor = {}
        self.c_factor = {}
        self._calculate_orbital_factors()

        self.scattering_rates = None
        self.scattering_labels = None
        self.doping = None
        self.temperatures = None
        self.fermi_levels = None
        self.electron_conc = None
        self.hole_conc = None
        self.fd_cutoffs = fd_cutoffs

        self.grouped_ir_to_full = groupby(
            np.arange(len(full_kpoints)), ir_to_full_kpoint_mapping
        )

        self.tetrahedral_band_structure = TetrahedralBandStructure(
            energies,
            full_kpoints,
            tetrahedra,
            structure,
            ir_kpoints_idx,
            ir_to_full_kpoint_mapping,
            *ir_tetrahedra_info
        )

    def calculate_dos(self, estep: float = defaults["dos_estep"]):
        """
        Args:
            estep: The DOS energy step in eV, where smaller numbers give more
                accuracy but are more expensive.
        """
        emin = np.min([np.min(spin_eners) for spin_eners in self.energies.values()])
        emax = np.max([np.max(spin_eners) for spin_eners in self.energies.values()])
        epoints = int(round((emax - emin) / (estep * units.eV)))
        energies = np.linspace(emin, emax, epoints)
        dos_weight = 1 if self._soc or len(self.spins) == 2 else 2

        logger.debug("DOS parameters:")
        log_list(
            [
                "emin: {:.2f} eV".format(emin / units.eV),
                "emax: {:.2f} eV".format(emax / units.eV),
                "dos weight: {}".format(dos_weight),
                "n points: {}".format(epoints),
            ]
        )

        logger.debug("Generating tetrahedral DOS:")
        t0 = time.perf_counter()
        emesh, dos = self.tetrahedral_band_structure.get_density_of_states(
            energies=energies, progress_bar=True
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
                        self.fermi_levels[n, t], self.electron_conc[
                            n, t
                        ], self.hole_conc[n, t] = self.dos.get_fermi(
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
                "{:.2g} cm⁻³ & {} K: {:.4f} eV".format(
                    doping[n], temperatures[t], self.fermi_levels[n, t] / units.eV
                )
            )

        logger.info("Calculated Fermi levels:")
        log_list(fermi_level_info)

    def calculate_fd_cutoffs(
        self, fd_tolerance: Optional[float] = 0.01, cutoff_pad: float = 0.0
    ):
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
            vb_bands = [
                np.max(self.energies[s][: self.vb_idx[s] + 1]) for s in self.spins
            ]
            cb_bands = [
                np.min(self.energies[s][self.vb_idx[s] + 1 :]) for s in self.spins
            ]
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
        log_list(
            [
                "min: {:.3f} eV".format(min_cutoff / units.eV),
                "max: {:.3f} eV".format(max_cutoff / units.eV),
            ]
        )
        self.fd_cutoffs = (min_cutoff, max_cutoff)

    def set_scattering_rates(
        self, scattering_rates: Dict[Spin, np.ndarray], scattering_labels: List[str]
    ):
        for spin in self.spins:
            expected_shape = (len(self.doping), len(self.temperatures)) + self.energies[
                spin
            ].shape
            if scattering_rates[spin].shape[1:] != expected_shape:
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

    def _calculate_orbital_factors(self):
        for spin in self.spins:
            self.a_factor[spin] = (
                self._projections[spin]["s"]
                / (
                    self._projections[spin]["s"] ** 2
                    + self._projections[spin]["p"] ** 2
                )
                ** 0.5
            )
            self.c_factor[spin] = (1 - self.a_factor[spin] ** 2) ** 0.5

    def to_file(
        self,
        directory: str = ".",
        prefix: Optional[str] = None,
        write_mesh: bool = defaults["write_mesh"],
        file_format: str = defaults["file_format"],
        suffix_mesh: bool = True,
    ):
        if (
            self.conductivity is None
            or self.seebeck is None
            or self.electronic_thermal_conductivity is None
        ):
            raise ValueError(
                "Cannot write AmsetData to file, transport " "properties not set"
            )

        if not prefix:
            prefix = ""
        else:
            prefix += "_"

        if suffix_mesh:
            if self.kpoint_mesh is not None:
                mesh = "x".join(map(str, self.kpoint_mesh))
            else:
                # user supplied k-points
                mesh = len(self.full_kpoints)
            suffix = "_{}".format(mesh)
        else:
            suffix = ""

        if file_format in ["json", "yaml"]:
            data = {
                "doping": self.doping,
                "temperature": self.temperatures,
                "fermi_levels": self.fermi_levels,
                "conductivity": self.conductivity,
                "seebeck": self.seebeck,
                "electronic_thermal_conductivity": self.electronic_thermal_conductivity,
                "mobility": self.mobility,
            }

            if write_mesh:
                ir_rates = {
                    spin: rates[:, :, :, :, self.ir_kpoints_idx]
                    for spin, rates in self.scattering_rates.items()
                }
                ir_energies = {
                    spin: energies[:, self.ir_kpoints_idx] / units.eV
                    for spin, energies in self.energies.items()
                }
                data.update(
                    {
                        "energies": cast_dict(ir_energies),
                        "kpoints": self.full_kpoints,
                        "kpoint_weights": self.kpoint_weights,
                        "ir_kpoints": self.ir_kpoints,
                        "ir_to_full_kpoint_mapping": self.ir_to_full_kpoint_mapping,
                        "dos": self.dos,
                        "efermi": self.intrinsic_fermi_level,
                        "vb_idx": cast_dict(self.vb_idx),
                        "scattering_rates": cast_dict(ir_rates),
                        "scattering_labels": self.scattering_labels,
                    }
                )

            filename = joinpath(
                directory, "{}amset_data{}.{}".format(prefix, suffix, file_format)
            )
            dumpfn(data, filename)

        elif file_format in ["csv", "txt"]:
            # don't write the data as JSON, instead write raw text files
            data = []

            triu = np.triu_indices(3)
            for n, t in np.ndindex(len(self.doping), len(self.temperatures)):
                row = [self.doping[n], self.temperatures[t], self.fermi_levels[n, t]]
                row.extend(self.conductivity[n, t][triu])
                row.extend(self.seebeck[n, t][triu])
                row.extend(self.electronic_thermal_conductivity[n, t][triu])

                if self.mobility is not None:
                    for mob in self.mobility.values():
                        row.extend(mob[n, t][triu])
                data.append(row)

            headers = ["doping[cm^-3]", "temperature[K]", "Fermi_level[eV]"]

            # TODO: confirm unit of kappa
            for prop, unit in [("cond", "S/m"), ("seebeck", "µV/K"), ("kappa", "?")]:
                headers.extend(
                    [
                        "{}_{}[{}]".format(prop, d, unit)
                        for d in ("xx", "xy", "xz", "yy", "yz", "zz")
                    ]
                )

            if self.mobility is not None:
                for name in self.mobility.keys():
                    headers.extend(
                        [
                            "{}_mobility_{}[cm^2/V.s]".format(name, d)
                            for d in ("xx", "xy", "xz", "yy", "yz", "zz")
                        ]
                    )

            filename = joinpath(
                directory, "{}amset_transport{}.{}".format(prefix, suffix, file_format)
            )
            np.savetxt(filename, data, header=" ".join(headers))

            if write_mesh:
                logger.warning("Writing mesh data as txt or csv not supported")

        else:
            raise ValueError("Unrecognised output format: {}".format(file_format))
