import os
import copy
import datetime
import logging
import time
from functools import partial

import numpy as np

from pathlib import Path
from os.path import join as joinpath
from typing import Optional, Any, Dict, Union

from BoltzTraP2 import units
from tabulate import tabulate

from monty.json import MSONable
from memory_profiler import memory_usage

from amset.constants import hbar, bohr_to_cm
from pymatgen import Structure
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.io.vasp import Vasprun
from amset import __version__
from amset.constants import amset_defaults
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.string import unicodeify, unicodeify_spacegroup

from amset.interpolation.interpolate import Interpolater
from amset.scattering.calculate import ScatteringCalculator
from amset.transport import solve_boltzman_transport_equation
from amset.util import (
    validate_settings,
    tensor_average,
    load_settings_from_file,
    write_settings_to_file,
)
from amset.log import log_banner, log_list, initialize_amset_logger

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

logger = logging.getLogger(__name__)
_kpt_str = "[{k[0]:.2f}, {k[1]:.2f}, {k[2]:.2f}]"


class AmsetRunner(MSONable):
    def __init__(
        self,
        band_structure: BandStructure,
        num_electrons: int,
        settings: Dict[str, Any],
    ):
        self._band_structure = band_structure
        self._num_electrons = num_electrons

        # set materials and performance parameters
        # if the user doesn't specify a value then use the default
        self.settings = copy.deepcopy(amset_defaults)
        self.settings.update(settings)

        if self.settings["print_log"]:
            initialize_amset_logger()

    def run(
        self,
        directory: Union[str, Path] = ".",
        prefix: Optional[str] = None,
        return_usage_stats: bool = False,
    ):
        mem_usage, (amset_data, usage_stats) = memory_usage(
            partial(self._run_wrapper, directory=directory, prefix=prefix),
            max_usage=True,
            retval=True,
            interval=0.1,
            include_children=False,
            multiprocess=True,
        )

        log_banner("END")

        logger.info("Timing and memory usage:")
        timing_info = [
            "{} time: {:.4f} s".format(name, t) for name, t in usage_stats.items()
        ]
        log_list(timing_info + ["max memory: {:.1f} MB".format(mem_usage)])

        now = datetime.datetime.now()
        logger.info(
            "amset exiting on {} at {}".format(
                now.strftime("%d %b %Y"), now.strftime("%H:%M")
            )
        )

        if return_usage_stats:
            usage_stats["max memory"] = mem_usage[0]
            return amset_data, usage_stats

        else:
            return amset_data

    def _run_wrapper(
        self, directory: Union[str, Path] = ".", prefix: Optional[str] = None
    ):
        tt = time.perf_counter()
        _log_amset_intro()
        _log_settings(self)
        _log_structure_information(
            self._band_structure.structure, self.settings["symprec"]
        )
        _log_band_structure_information(self._band_structure)

        log_banner("INTERPOLATION")
        t0 = time.perf_counter()

        interpolater = Interpolater(
            self._band_structure,
            num_electrons=self._num_electrons,
            interpolation_factor=self.settings["interpolation_factor"],
            soc=self.settings["soc"],
            interpolate_projections=True,
        )

        amset_data = interpolater.get_amset_data(
            energy_cutoff=self.settings["energy_cutoff"],
            scissor=self.settings["scissor"],
            bandgap=self.settings["bandgap"],
            symprec=self.settings["symprec"],
            nworkers=self.settings["nworkers"],
        )

        timing = {"interpolation": time.perf_counter() - t0}

        pop_frequency = self.settings["pop_frequency"]
        scattering_type = self.settings["scattering_type"]
        cutoff_pad = 0
        if pop_frequency and "POP" in scattering_type or scattering_type == "auto":
            # convert from THz to angular frequency in Hz
            pop_frequency = pop_frequency * 1e12 * 2 * np.pi

            # use the phonon energy to pad the fermi dirac cutoffs, this is because
            # pop scattering from a kpoints, k, to kpoints with energies above and below
            # k. We therefore need k-points above and below to be within the cut-offs
            # otherwise scattering cannot occur
            cutoff_pad = pop_frequency * hbar * units.eV

        log_banner("DOS")
        amset_data.calculate_dos(estep=self.settings["dos_estep"])
        amset_data.set_doping_and_temperatures(
            self.settings["doping"], self.settings["temperatures"]
        )
        amset_data.calculate_fd_cutoffs(self.settings["fd_tol"], cutoff_pad=cutoff_pad)

        log_banner("SCATTERING")
        t0 = time.perf_counter()

        scatter = ScatteringCalculator(
            self.settings,
            amset_data,
            scattering_type=scattering_type,
            use_symmetry=self.settings["symprec"] is not None,
        )

        amset_data.set_scattering_rates(
            scatter.calculate_scattering_rates(), scatter.scatterer_labels
        )

        timing["scattering"] = time.perf_counter() - t0

        log_banner("TRANSPORT")
        t0 = time.perf_counter()
        sep_scats = self.settings["separate_scattering_mobilities"]
        transport_properties = solve_boltzman_transport_equation(
            amset_data,
            separate_scattering_mobilities=sep_scats,
            calculate_mobility=self.settings["calculate_mobility"],
        )
        amset_data.set_transport_properties(*transport_properties)
        timing["transport"] = time.perf_counter() - t0

        log_banner("RESULTS")
        _log_results_summary(amset_data, self.settings)

        abs_dir = os.path.abspath(directory)
        logger.info("Writing results to {}".format(abs_dir))
        t0 = time.perf_counter()

        if not os.path.exists(abs_dir):
            os.makedirs(abs_dir)

        if self.settings["write_input"]:
            self.write_settings(abs_dir)

        amset_data.to_file(
            directory=abs_dir,
            write_mesh=self.settings["write_mesh"],
            prefix=prefix,
            file_format=self.settings["file_format"],
        )

        timing["writing"] = time.perf_counter() - t0
        timing["total"] = time.perf_counter() - tt

        return amset_data, timing

    @staticmethod
    def from_vasprun(
        vasprun: Union[str, Path, Vasprun], settings: Dict[str, Any]
    ) -> "AmsetRunner":
        """Initialise an AmsetRunner from a Vasprun.

        The nelect and soc options will be determined from the Vasprun
        automatically.

        Args:
            vasprun: Path to a vasprun or a Vasprun pymatgen object.
            settings: AMSET settings.

        Returns:
            An :obj:`AmsetRunner` instance.
        """
        if not isinstance(vasprun, Vasprun):
            vasprun = Vasprun(vasprun, parse_projected_eigen=True)

        band_structure = vasprun.get_band_structure()
        nelect = vasprun.parameters["NELECT"]
        settings["soc"] = vasprun.parameters["LSORBIT"]

        settings = validate_settings(settings)

        return AmsetRunner(band_structure, nelect, settings)

    @staticmethod
    def from_directory(
        directory: Union[str, Path] = ".",
        vasprun: Optional[Union[str, Path]] = None,
        settings_file: Optional[Union[str, Path]] = None,
        settings_override: Optional[Dict[str, Any]] = None,
    ):
        if not vasprun:
            vr_file = joinpath(directory, "vasprun.xml")
            vr_file_gz = joinpath(directory, "vasprun.xml.gz")

            if os.path.exists(vr_file):
                vasprun = Vasprun(vr_file, parse_projected_eigen=True)
            elif os.path.exists(vr_file_gz):
                vasprun = Vasprun(vr_file_gz, parse_projected_eigen=True)
            else:
                msg = "No vasprun.xml found in {}".format(directory)
                logger.error(msg)
                raise FileNotFoundError(msg)

        if not settings_file:
            settings_file = joinpath(directory, "settings.yaml")
        settings = load_settings_from_file(settings_file)

        if settings_override:
            settings.update(settings_override)

        return AmsetRunner.from_vasprun(vasprun, settings)

    def write_settings(self, directory: str = ".", prefix: Optional[str] = None):
        if prefix is None:
            prefix = ""
        else:
            prefix += "_"

        filename = joinpath(directory, "{}amset_settings.yaml".format(prefix))
        write_settings_to_file(self.settings, filename)


def _log_amset_intro():
    now = datetime.datetime.now()
    logger.info(
        """
               █████╗ ███╗   ███╗███████╗███████╗████████╗
              ██╔══██╗████╗ ████║██╔════╝██╔════╝╚══██╔══╝
              ███████║██╔████╔██║███████╗█████╗     ██║
              ██╔══██║██║╚██╔╝██║╚════██║██╔══╝     ██║
              ██║  ██║██║ ╚═╝ ██║███████║███████╗   ██║
              ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝   ╚═╝

                                                  v{}

    A. Ganose, A. Faghaninia, J. Park, F. Ricci, R. Woods-Robinson,
    J. Frost,  K. Persson, G. Hautier, A. Jain, in prep.


amset starting on {} at {}""".format(
            __version__, now.strftime("%d %b %Y"), now.strftime("%H:%M")
        )
    )


def _log_structure_information(structure: Structure, symprec):
    log_banner("STRUCTURE")
    logger.info("Structure information:")

    comp = structure.composition
    lattice = structure.lattice
    formula = comp.get_reduced_formula_and_factor(iupac_ordering=True)[0]

    if not symprec:
        symprec = 0.01

    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    spg = unicodeify_spacegroup(sga.get_space_group_symbol())

    comp_info = [
        "formula: {}".format(unicodeify(formula)),
        "# sites: {}".format(structure.num_sites),
        "space group: {}".format(spg),
    ]
    log_list(comp_info)

    logger.info("Lattice:")
    lattice_info = [
        "a, b, c [Å]: {:.2f}, {:.2f}, {:.2f}".format(*lattice.abc),
        "α, β, γ [°]: {:.0f}, {:.0f}, {:.0f}".format(*lattice.angles),
    ]
    log_list(lattice_info)


def _log_settings(runner: AmsetRunner):
    log_banner("SETTINGS")
    logger.info("Run parameters:")
    p = ["{}: {}".format(k, v) for k, v in runner.settings.items() if v is not None]
    log_list(p)


def _log_band_structure_information(band_structure: BandStructure):
    log_banner("BAND STRUCTURE")

    info = [
        "# bands: {}".format(band_structure.nb_bands),
        "# k-points: {}".format(len(band_structure.kpoints)),
        "Fermi level: {:.3f} eV".format(band_structure.efermi),
        "spin polarized: {}".format(band_structure.is_spin_polarized),
        "metallic: {}".format(band_structure.is_metal()),
    ]
    logger.info("Input band structure information:")
    log_list(info)

    if band_structure.is_metal():
        return

    logger.info("Band gap:")
    band_gap_info = []

    bg_data = band_structure.get_band_gap()
    if not bg_data["direct"]:
        band_gap_info.append("indirect band gap: {:.3f} eV".format(bg_data["energy"]))

    direct_data = band_structure.get_direct_band_gap_dict()
    direct_bg = min((spin_data["value"] for spin_data in direct_data.values()))
    band_gap_info.append("direct band gap: {:.3f} eV".format(direct_bg))

    direct_kpoint = []
    for spin, spin_data in direct_data.items():
        direct_kindex = spin_data["kpoint_index"]
        kpt_str = _kpt_str.format(k=band_structure.kpoints[direct_kindex].frac_coords)
        direct_kpoint.append(kpt_str)

    band_gap_info.append("direct k-point: {}".format(", ".join(direct_kpoint)))
    log_list(band_gap_info)

    vbm_data = band_structure.get_vbm()
    cbm_data = band_structure.get_cbm()

    logger.info("Valence band maximum:")
    _log_band_edge_information(band_structure, vbm_data)

    logger.info("Conduction band minimum:")
    _log_band_edge_information(band_structure, cbm_data)


def _log_band_edge_information(band_structure, edge_data):
    """Log data about the valence band maximum or conduction band minimum.

    Args:
        band_structure: A band structure.
        edge_data (dict): The :obj:`dict` from ``bs.get_vbm()`` or
            ``bs.get_cbm()``
    """
    if band_structure.is_spin_polarized:
        spins = edge_data["band_index"].keys()
        b_indices = [
            ", ".join([str(i + 1) for i in edge_data["band_index"][spin]])
            + "({})".format(spin.name.capitalize())
            for spin in spins
        ]
        b_indices = ", ".join(b_indices)
    else:
        b_indices = ", ".join([str(i + 1) for i in edge_data["band_index"][Spin.up]])

    kpoint = edge_data["kpoint"]
    kpoint_str = _kpt_str.format(k=kpoint.frac_coords)

    info = [
        "energy: {:.3f} eV".format(edge_data["energy"]),
        "k-point: {}".format(kpoint_str),
        "band indices: {}".format(b_indices),
    ]
    log_list(info)


def _log_results_summary(amset_data, output_parameters):
    results_summary = []

    doping = [d * (1 / bohr_to_cm) ** 3 for d in amset_data.doping]
    temps = amset_data.temperatures

    if output_parameters["calculate_mobility"] and not amset_data.is_metal:
        logger.info(
            "Average conductivity (σ), Seebeck (S) and mobility (μ)" " results:"
        )
        headers = ("conc [cm⁻³]", "temp [K]", "σ [S/m]", "S [µV/K]", "μ [cm²/Vs]")
        for c, t in np.ndindex(amset_data.fermi_levels.shape):
            results = (
                doping[c],
                temps[t],
                tensor_average(amset_data.conductivity[c, t]),
                tensor_average(amset_data.seebeck[c, t]),
                tensor_average(amset_data.mobility["overall"][c, t]),
            )
            results_summary.append(results)

    else:
        logger.info("Average conductivity (σ) and Seebeck (S) results:")
        headers = ("conc [cm⁻³]", "temp [K]", "σ [S/m]", "S [µV/K]")
        for c, t in np.ndindex(amset_data.fermi_levels.shape):
            results = (
                doping[c],
                temps[t],
                tensor_average(amset_data.conductivity[c, t]),
                tensor_average(amset_data.seebeck[c, t]),
            )
            results_summary.append(results)

    table = tabulate(
        results_summary,
        headers=headers,
        numalign="right",
        stralign="center",
        floatfmt=(".2e", ".1f", ".2e", ".2e", ".1f"),
    )
    logger.info(table)

    if output_parameters["separate_scattering_mobilities"] and not amset_data.is_metal:
        labels = amset_data.scattering_labels
        logger.info("Mobility breakdown by scattering mechanism, in cm²/Vs:")
        headers = ["conc [cm⁻³]", "temp [K]"] + labels

        results_summary = []
        for c, t in np.ndindex(amset_data.fermi_levels.shape):
            results = [doping[c] + temps[t]]
            results += [tensor_average(amset_data.mobility[s][c, t]) for s in labels]
            results_summary.append(results)

        table = tabulate(
            results_summary,
            headers=headers,
            numalign="right",
            stralign="center",
            floatfmt=[".2e", ".1f"] + [".2e"] * len(labels),
        )
        logger.info(table)
