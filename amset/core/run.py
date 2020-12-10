import copy
import datetime
import logging
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from memory_profiler import memory_usage
from monty.json import MSONable
from pymatgen import Structure, loadfn
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp import Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.string import unicodeify, unicodeify_spacegroup
from tabulate import tabulate

from amset import __version__
from amset.constants import bohr_to_cm, ev_to_hartree, hbar, numeric_types
from amset.core.transport import solve_boltzman_transport_equation
from amset.electronic_structure.common import get_band_structure
from amset.interpolation.bandstructure import Interpolator
from amset.interpolation.projections import ProjectionOverlapCalculator
from amset.interpolation.wavefunction import WavefunctionOverlapCalculator
from amset.io import load_settings, write_settings
from amset.log import initialize_amset_logger, log_banner, log_list
from amset.scattering.calculate import ScatteringCalculator, basic_scatterers
from amset.util import tensor_average, validate_settings

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

logger = logging.getLogger(__name__)
_kpt_str = "[{k[0]:.2f}, {k[1]:.2f}, {k[2]:.2f}]"


class Runner(MSONable):
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
        self.settings = validate_settings(settings)

    def run(
        self,
        directory: Union[str, Path] = ".",
        return_usage_stats: bool = False,
        prefix: Optional[str] = None,
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
        timing_info = ["{} time: {:.4f} s".format(n, t) for n, t in usage_stats.items()]
        log_list(timing_info + ["max memory: {:.1f} MB".format(mem_usage)])

        this_date = datetime.datetime.now().strftime("%d %b %Y")
        this_time = datetime.datetime.now().strftime("%H:%M")
        logger.info("amset exiting on {} at {}".format(this_date, this_time))

        if return_usage_stats:
            usage_stats["max memory"] = mem_usage
            return amset_data, usage_stats
        else:
            return amset_data

    def _run_wrapper(
        self, directory: Union[str, Path] = ".", prefix: Optional[str] = None
    ):
        if self.settings["print_log"] or self.settings["write_log"]:
            if self.settings["write_log"]:
                log_file = "{}_amset.log".format(prefix) if prefix else "amset.log"
            else:
                log_file = False

            initialize_amset_logger(
                directory=directory,
                filename=log_file,
                print_log=self.settings["print_log"],
            )

        self._check_wavefunction()

        tt = time.perf_counter()
        _log_amset_intro()
        _log_settings(self)
        _log_structure_information(
            self._band_structure.structure, self.settings["symprec"]
        )
        _log_band_structure_information(self._band_structure)

        amset_data, interpolation_time = self._do_interpolation()
        timing = {"interpolation": interpolation_time}

        amset_data, dos_time = self._do_dos(amset_data)
        timing["dos"] = dos_time

        amset_data, scattering_time = self._do_scattering(amset_data)
        timing["scattering"] = scattering_time

        if isinstance(self.settings["fd_tol"], numeric_types):
            amset_data, timing = self._do_fd_tol(amset_data, directory, prefix, timing)
        else:
            amset_data, timing = self._do_many_fd_tol(
                amset_data, self.settings["fd_tol"], directory, prefix, timing
            )

        timing["total"] = time.perf_counter() - tt
        return amset_data, timing

    def _do_fd_tol(self, amset_data, directory, prefix, timing):
        amset_data.fill_rates_outside_cutoffs()
        amset_data, transport_time = self._do_transport(amset_data)
        timing["transport"] = transport_time

        filepath, writing_time = self._do_writing(amset_data, directory, prefix)
        timing["writing"] = writing_time

        return amset_data, timing

    def _do_many_fd_tol(self, amset_data, fd_tols, directory, prefix, timing):
        prefix = "" if prefix is None else prefix + "_"
        cutoff_pad = _get_cutoff_pad(
            self.settings["pop_frequency"], self.settings["scattering_type"]
        )
        orig_rates = copy.deepcopy(amset_data.scattering_rates)
        mobility_rates_only = self.settings["mobility_rates_only"]

        for fd_tol in sorted(fd_tols)[::-1]:
            # do smallest cutoff last, so the final amset_data is the best result
            for spin in amset_data.spins:
                amset_data.scattering_rates[spin][:] = orig_rates[spin][:]

            amset_data.calculate_fd_cutoffs(
                fd_tol, cutoff_pad=cutoff_pad, mobility_rates_only=mobility_rates_only
            )
            fd_prefix = prefix + "fd-{}".format(fd_tol)
            _, timing = self._do_fd_tol(amset_data, directory, fd_prefix, timing)
            timing["transport ({})".format(fd_tol)] = timing.pop("transport")
            timing["writing ({})".format(fd_tol)] = timing.pop("writing")

        return amset_data, timing

    def _check_wavefunction(self):
        if (
            not Path(self.settings["wavefunction_coefficients"]).exists()
            and not self.settings["use_projections"]
        ):
            raise ValueError(
                "Could not find wavefunction coefficients. To run AMSET, the \n"
                "wavefunction coefficients should first be extracted from a WAVECAR \n"
                "file using the 'amset wave' command. See the documentation for more \n"
                "details: https://hackingmaterials.lbl.gov/amset/\n\n"
                "Alternatively, to use the band structure orbital projections to \n"
                "approximate overlap, set 'use_projections' option to true."
            )
        elif self.settings["use_projections"] and not self._band_structure.projections:
            raise ValueError(
                "use_projections is set to true but calculation does not contain\n"
                "orbital projections. Ensure VASP was run with 'LORBIT = 11'\n"
                "Alternatively, use wavefunction coefficients to calculate overlap.\n"
                "Wavefunction coefficients can be extracted from a VASP WAVECAR\n"
                "file using the 'amset wave' command. See the documentation for more\n"
                "details: https://hackingmaterials.lbl.gov/amset/\n\n"
            )
        elif self.settings["use_projections"]:
            logger.info(
                "Using orbital projections to approximate wavefunction overlap. This "
                "can result in inaccurate results. I hope you know what you are doing."
            )

    def _do_interpolation(self):
        log_banner("INTERPOLATION")
        t0 = time.perf_counter()

        interpolater = Interpolator(
            self._band_structure,
            num_electrons=self._num_electrons,
            interpolation_factor=self.settings["interpolation_factor"],
            soc=self.settings["soc"],
        )

        amset_data = interpolater.get_amset_data(
            energy_cutoff=self.settings["energy_cutoff"],
            scissor=self.settings["scissor"],
            bandgap=self.settings["bandgap"],
            symprec=self.settings["symprec"],
            nworkers=self.settings["nworkers"],
        )

        if set(self.settings["scattering_type"]).issubset(set(basic_scatterers)):
            overlap_calculator = None
        elif self.settings["use_projections"]:
            overlap_calculator = ProjectionOverlapCalculator.from_band_structure(
                self._band_structure,
                energy_cutoff=self.settings["energy_cutoff"],
                symprec=self.settings["symprec"],
            )
        else:
            overlap_calculator = WavefunctionOverlapCalculator.from_file(
                self.settings["wavefunction_coefficients"]
            )
        amset_data.set_overlap_calculator(overlap_calculator)

        return amset_data, time.perf_counter() - t0

    def _do_dos(self, amset_data):
        log_banner("DOS")
        t0 = time.perf_counter()

        amset_data.calculate_dos(
            estep=self.settings["dos_estep"], progress_bar=self.settings["print_log"]
        )
        amset_data.set_doping_and_temperatures(
            self.settings["doping"], self.settings["temperatures"]
        )

        cutoff_pad = _get_cutoff_pad(
            self.settings["pop_frequency"], self.settings["scattering_type"]
        )

        if isinstance(self.settings["fd_tol"], numeric_types):
            fd_tol = self.settings["fd_tol"]
        else:
            fd_tol = min(self.settings["fd_tol"])

        mob_only = self.settings["mobility_rates_only"]
        amset_data.calculate_fd_cutoffs(
            fd_tol, cutoff_pad=cutoff_pad, mobility_rates_only=mob_only
        )
        return amset_data, time.perf_counter() - t0

    def _do_scattering(self, amset_data):
        log_banner("SCATTERING")
        t0 = time.perf_counter()

        cutoff_pad = _get_cutoff_pad(
            self.settings["pop_frequency"], self.settings["scattering_type"]
        )

        scatter = ScatteringCalculator(
            self.settings,
            amset_data,
            cutoff_pad,
            scattering_type=self.settings["scattering_type"],
            progress_bar=self.settings["print_log"],
            cache_wavefunction=self.settings["cache_wavefunction"],
            nworkers=self.settings["nworkers"],
        )

        amset_data.set_scattering_rates(
            scatter.calculate_scattering_rates(), scatter.scatterer_labels
        )
        return amset_data, time.perf_counter() - t0

    def _do_transport(self, amset_data):
        log_banner("TRANSPORT")
        t0 = time.perf_counter()
        transport_properties = solve_boltzman_transport_equation(
            amset_data,
            separate_mobility=self.settings["separate_mobility"],
            calculate_mobility=self.settings["calculate_mobility"],
            progress_bar=self.settings["print_log"],
        )
        amset_data.set_transport_properties(*transport_properties)
        return amset_data, time.perf_counter() - t0

    def _do_writing(self, amset_data, directory, prefix):
        log_banner("RESULTS")
        _log_results_summary(amset_data, self.settings)

        abs_dir = os.path.abspath(directory)
        t0 = time.perf_counter()

        if not os.path.exists(abs_dir):
            os.makedirs(abs_dir)

        if self.settings["write_input"]:
            self.write_settings(abs_dir)

        if self.settings["file_format"] is None:
            full_filename = None
        else:
            filename = amset_data.to_file(
                directory=abs_dir,
                write_mesh_file=self.settings["write_mesh"],
                prefix=prefix,
                file_format=self.settings["file_format"],
            )

            if isinstance(filename, tuple):
                full_filename = "\nand\n".join(
                    [str(Path(abs_dir) / f) for f in filename]
                )
            else:
                full_filename = Path(abs_dir) / filename

            logger.info("Results written to:\n{}".format(full_filename))
        return full_filename, time.perf_counter() - t0

    @staticmethod
    def from_vasprun(
        vasprun: Union[str, Path, Vasprun], settings: Dict[str, Any]
    ) -> "Runner":
        """Initialise an AmsetRunner from a Vasprun.

        The nelect and soc options will be determined from the Vasprun
        automatically.

        Args:
            vasprun: Path to a vasprun or a Vasprun pymatgen object.
            settings: AMSET settings.

        Returns:
            A Runner object that can be used to calculate transport properties.
        """
        if not isinstance(vasprun, Vasprun):
            vasprun = Vasprun(vasprun, parse_projected_eigen=True)

        zwk_option = settings.get("zero_weighted_kpoints", "keep")
        band_structure = get_band_structure(vasprun, zero_weighted=zwk_option)
        nelect = vasprun.parameters["NELECT"]
        settings["soc"] = vasprun.parameters["LSORBIT"]

        return Runner(band_structure, nelect, settings)

    @staticmethod
    def from_directory(
        directory: Union[str, Path] = ".",
        input_file: Optional[Union[str, Path]] = None,
        settings_file: Optional[Union[str, Path]] = None,
        settings_override: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize amset Runner from a directory.

        If input_file or settings_file are not specified, the code will look in the
        specified directory for these files.

        Args:
            directory: A directory.
            input_file: An input file path, can either be vasprun.xml(.gz) or a
                band_structure_data.json(.gz) file containing the keys:

                - "nelect" (int): The number of electrons in the system.
                - "band_structure (BandStructure)": A pymatgen band structure object.
            settings_file: Path to settings file.
            settings_override: Settings that will be used to override the settings
                in the settings file.

        Returns:
            A Runner instance that can be used to run amset.
        """
        directory = Path(directory)

        if not settings_file:
            settings_file = directory / "settings.yaml"
        settings = load_settings(settings_file)

        if settings_override:
            settings.update(settings_override)

        run_type, input_file = _get_run_type(directory, input_file)
        if run_type == "vasprun":
            return Runner.from_vasprun(input_file, settings)
        elif run_type == "band_structure":
            data = loadfn(input_file)
            nelect = data["nelect"]
            band_structure = data["band_structure"]
            return Runner(band_structure, nelect, settings)
        else:
            raise ValueError(f"Unrecognised run type: {run_type}")

    def write_settings(self, directory: str = ".", prefix: Optional[str] = None):
        prefix = "" if prefix is None else f"{prefix}_"
        filename = Path(directory) / f"{prefix}amset_settings.yaml"
        write_settings(self.settings, filename)


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

    A. Ganose, J. Park, A. Faghaninia, R. Woods-Robinson,
    A. Jain, in prep.


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
        symprec = 1e-32

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


_tensor_str = """
    │   [[{:6.2f} {:6.2f} {:6.2f}]
    │    [{:6.2f} {:6.2f} {:6.2f}]
    │    [{:6.2f} {:6.2f} {:6.2f}]]"""

_elastic_tensor_str = """
    │   [[{:6.1f} {:6.1f} {:6.1f} {:6.1f} {:6.1f} {:6.1f}]
    │    [{:6.1f} {:6.1f} {:6.1f} {:6.1f} {:6.1f} {:6.1f}]
    │    [{:6.1f} {:6.1f} {:6.1f} {:6.1f} {:6.1f} {:6.1f}]
    │    [{:6.1f} {:6.1f} {:6.1f} {:6.1f} {:6.1f} {:6.1f}]
    │    [{:6.1f} {:6.1f} {:6.1f} {:6.1f} {:6.1f} {:6.1f}]
    │    [{:6.1f} {:6.1f} {:6.1f} {:6.1f} {:6.1f} {:6.1f}]]"""

_piezo_tensor_str = """
    │   [[{:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f}]
    │    [{:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f}]
    │    [{:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f}]]"""


def _log_settings(runner: Runner):
    from pymatgen.core.tensors import Tensor

    def ff(prop):
        # format tensor properties
        if isinstance(prop, np.ndarray):
            if prop.shape == (3, 3):
                return _tensor_str.format(*prop.ravel())
            elif prop.shape == (3, 3, 3):
                return _piezo_tensor_str.format(*Tensor(prop).voigt.ravel())
            elif prop.shape == (3, 3, 3, 3):
                return _elastic_tensor_str.format(*Tensor(prop).voigt.ravel())

        return prop

    log_banner("SETTINGS")
    logger.info("Run parameters:")
    p = ["{}: {}".format(k, ff(v)) for k, v in runner.settings.items() if v is not None]
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

    if output_parameters["separate_mobility"] and not amset_data.is_metal:
        labels = amset_data.scattering_labels
        logger.info("Mobility breakdown by scattering mechanism, in cm²/Vs:")
        headers = ["conc [cm⁻³]", "temp [K]"] + labels

        results_summary = []
        for c, t in np.ndindex(amset_data.fermi_levels.shape):
            results = [doping[c], temps[t]]
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


def _get_cutoff_pad(pop_frequency, scattering_type):
    cutoff_pad = 0
    if pop_frequency and ("POP" in scattering_type or scattering_type == "auto"):
        # convert from THz to angular frequency in Hz
        pop_frequency = pop_frequency * 1e12 * 2 * np.pi

        # use the phonon energy to pad the fermi dirac cutoffs, this is because
        # pop scattering from a kpoints, k, to kpoints with energies above and below
        # k. We therefore need k-points above and below to be within the cut-offs
        # otherwise scattering cannot occur
        cutoff_pad = pop_frequency * hbar * ev_to_hartree
    return cutoff_pad


def _get_run_type(directory: Path, input_file: Optional[str]):
    if input_file is None:
        vr_files = list(directory.glob("*vasprun*"))
        bs_files = list(directory.glob("*band_structure_data*"))
        if len(vr_files) > 0:
            input_file = vr_files[0]
        elif len(bs_files) > 0:
            input_file = bs_files[0]
        else:
            raise ValueError(
                "Could not find vasprun.xml, vasprun.xml.gz or band_structure_data.json"
                f" in {directory}"
            )

    input_file = str(input_file)
    if "xml" in input_file:
        run_type = "vasprun"
    elif "json" in input_file:
        run_type = "band_structure"
    else:
        raise ValueError(f"Could not determine input file type: {input_file}")

    return run_type, input_file
