import os
import copy
import datetime
import logging
import time
from functools import partial

import numpy as np

from pathlib import Path
from os.path import join as joinpath
from typing import Optional, Any, Dict, Union, List

from tabulate import tabulate

from monty.json import MSONable
from memory_profiler import memory_usage

from pymatgen import Structure
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.io.vasp import Vasprun
from amset import __version__, amset_defaults
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.string import unicodeify

from amset.interpolate import Interpolater
from amset.scatter import ScatteringCalculator
from amset.transport import TransportCalculator
from amset.util import validate_settings, star_log, tensor_average, log_list, \
    unicodeify_spacegroup, write_settings_to_file, load_settings_from_file

logger = logging.getLogger(__name__)
_kpt_str = '[{k[0]:.2f}, {k[1]:.2f}, {k[2]:.2f}]'


class AmsetRunner(MSONable):

    def __init__(self,
                 band_structure: BandStructure,
                 num_electrons: int,
                 material_properties: Dict[str, Any],
                 doping: Optional[Union[List, np.ndarray]] = None,
                 temperatures: Optional[Union[List, np.ndarray]] = None,
                 scattering_type: Optional[Union[str, List[str],
                                                 float]] = "auto",
                 performance_parameters: Optional[Dict[str, float]] = None,
                 output_parameters: Optional[Dict[str, Any]] = None,
                 interpolation_factor: int = 10,
                 scissor: Optional[float] = None,
                 user_bandgap: Optional[float] = None,
                 soc: bool = False):
        self._band_structure = band_structure
        self._num_electrons = num_electrons
        self.scattering_type = scattering_type
        self.interpolation_factor = interpolation_factor
        self.scissor = scissor
        self.user_bandgap = user_bandgap
        self.soc = soc
        self.doping = doping
        self.temperatures = temperatures

        if self.doping is None:
            self.doping = np.concatenate([np.logspace(16, 21, 6),
                                          -np.logspace(16, 21, 6)])

        if self.temperatures is None:
            self.temperatures = np.array([300])

        # set materials and performance parameters
        # if the user doesn't specify a value then use the default
        params = copy.deepcopy(amset_defaults)
        self.performance_parameters = params["performance"]
        self.material_properties = params["material"]
        self.output_parameters = params["output"]
        self.performance_parameters.update(performance_parameters)
        self.material_properties.update(material_properties)
        self.output_parameters.update(output_parameters)

    def run(self,
            directory: Union[str, Path] = '.',
            prefix: Optional[str] = None,
            write_input: bool = True,
            write_mesh: bool = True,
            return_usage_stats: bool = False):
        mem_usage, (amset_data, usage_stats) = memory_usage(
            partial(self._run_wrapper, directory=directory, prefix=prefix,
                    write_input=write_input, write_mesh=write_mesh),
            include_children=True, max_usage=True, retval=True, interval=1,
            multiprocess=True)

        star_log("END")

        logger.info("Timing and memory usage:")
        timing_info = ["{} time: {:.4f} s".format(name, t)
                       for name, t in usage_stats.items()]
        log_list(timing_info + ["max memory: {:.1f} MB".format(mem_usage[0])])

        now = datetime.datetime.now()
        logger.info("amset exiting on {} at {}".format(
            now.strftime("%d %b %Y"), now.strftime("%H:%M")))

        if return_usage_stats:
            usage_stats["max memory"] = mem_usage[0]
            return amset_data, usage_stats

        else:
            return amset_data

    def _run_wrapper(self,
                     directory: Union[str, Path] = '.',
                     prefix: Optional[str] = None,
                     write_input: bool = True,
                     write_mesh: bool = True):
        tt = time.perf_counter()
        _log_amset_intro()
        _log_settings(self)

        # initialize scattering first so we can check that materials properties
        # and desired scattering mechanisms are consistent
        scatter = ScatteringCalculator(
            self.material_properties,
            scattering_type=self.scattering_type,
            energy_tol=self.performance_parameters["energy_tol"],
            g_tol=self.performance_parameters["g_tol"],
            use_symmetry=self.performance_parameters["symprec"] is not None,
            nworkers=self.performance_parameters["nworkers"])

        _log_structure_information(self._band_structure.structure,
                                   self.performance_parameters["symprec"])
        _log_band_structure_information(self._band_structure)

        star_log("INTERPOLATION")
        t0 = time.perf_counter()

        interpolater = Interpolater(
            self._band_structure, num_electrons=self._num_electrons,
            interpolation_factor=self.interpolation_factor, soc=self.soc,
            interpolate_projections=True)

        amset_data = interpolater.get_amset_data(
            energy_cutoff=self.performance_parameters["energy_cutoff"],
            scissor=self.scissor, bandgap=self.user_bandgap,
            symprec=self.performance_parameters["symprec"],
            nworkers=self.performance_parameters["nworkers"])

        timing = {"interpolation": time.perf_counter() - t0}

        star_log("DOS")
        amset_data.calculate_dos(
            dos_estep=self.performance_parameters["dos_estep"],
            dos_width=self.performance_parameters["dos_width"])
        amset_data.set_doping_and_temperatures(self.doping, self.temperatures)

        star_log("SCATTERING")
        t0 = time.perf_counter()

        amset_data.set_scattering_rates(
            scatter.calculate_scattering_rates(amset_data),
            [m.name for m in scatter.scatterers])

        timing["scattering"] = time.perf_counter() - t0

        star_log('BTE')
        t0 = time.perf_counter()

        solver = TransportCalculator(
            separate_scattering_mobilities=self.output_parameters[
                "separate_scattering_mobilities"],
            calculate_mobility=self.output_parameters["calculate_mobility"])
        (amset_data.conductivity, amset_data.seebeck,
         amset_data.electronic_thermal_conductivity,
         amset_data.mobility) = solver.solve_bte(amset_data)

        timing["BTE"] = time.perf_counter() - t0

        star_log('RESULTS')

        results_summary = []
        if self.output_parameters["calculate_mobility"]:
            logger.info("Average conductivity (σ), Seebeck (S) and mobility (μ)"
                        " results:")
            headers = ("conc [cm⁻³]", "temp [K]", "σ [S/m]", "S [µV/K]",
                       "μ [cm²/Vs]")
            for c, t in np.ndindex(amset_data.fermi_levels.shape):
                results_summary.append(
                    (self.doping[c], self.temperatures[t],
                     tensor_average(amset_data.conductivity[c, t]),
                     tensor_average(amset_data.seebeck[c, t]),
                     tensor_average(amset_data.mobility["overall"][c, t])))

        else:
            logger.info("Average conductivity (σ) and Seebeck (S) results:")
            headers = ("conc [cm⁻³]", "temp [K]", "σ [S/m]", "S [µV/K]")
            for c, t in np.ndindex(amset_data.fermi_levels.shape):
                results_summary.append(
                    (self.doping[c], self.temperatures[t],
                     tensor_average(amset_data.conductivity[c, t]),
                     tensor_average(amset_data.seebeck[c, t]),
                     tensor_average(amset_data.mobility["overall"][c, t])))

        logger.info(tabulate(
            results_summary, headers=headers, numalign="right",
            stralign="center", floatfmt=(".2g", ".1f", ".2g", ".2g", ".1f")))

        abs_dir = os.path.abspath(directory)
        logger.info("Writing data to {}".format(abs_dir))
        if write_input:
            self.write_settings(abs_dir)

        amset_data.to_file(
            directory=abs_dir, write_mesh=write_mesh, prefix=prefix,
            file_format=self.output_parameters["file_format"])

        timing["total"] = time.perf_counter() - tt

        return amset_data, timing

    @staticmethod
    def from_vasprun(vasprun: Union[str, Path, Vasprun],
                     material_parameters: Dict[str, Any],
                     **kwargs) -> "AmsetRunner":
        """Initialise an AmsetRunner from a Vasprun.

        The nelect and soc options will be determined from the Vasprun
        automatically.

        Args:
            vasprun: Path to a vasprun or a Vasprun pymatgen object.
            material_parameters: TODO
            **kwargs: Other parameters to be passed to the AmsetRun constructor
                except ``nelect`` and ``soc``.

        Returns:
            An :obj:`AmsetRunner` instance.
        """
        if not isinstance(vasprun, Vasprun):
            vasprun = Vasprun(vasprun, parse_projected_eigen=True)

        band_structure = vasprun.get_band_structure()
        soc = vasprun.parameters["LSORBIT"]
        nelect = vasprun.parameters["NELECT"]

        return AmsetRunner(band_structure, nelect, material_parameters,
                           soc=soc, **kwargs)

    @staticmethod
    def from_vasprun_and_settings(vasprun, settings):
        settings = validate_settings(settings)

        return AmsetRunner.from_vasprun(
            vasprun, settings["material"],
            doping=settings["general"]["doping"],
            temperatures=settings["general"]["temperatures"],
            scattering_type=settings["general"]["scattering_type"],
            performance_parameters=settings["performance"],
            output_parameters=settings["output"],
            interpolation_factor=settings["general"]["interpolation_factor"],
            scissor=settings["general"]["scissor"],
            user_bandgap=settings["general"]["bandgap"])

    @staticmethod
    def from_directory(directory: Union[str, Path] = '.',
                       vasprun: Optional[Union[str, Path]] = None,
                       settings_file: Optional[Union[str, Path]] = None):
        if not vasprun:
            vr_file = joinpath(directory, 'vasprun.xml')
            vr_file_gz = joinpath(directory, 'vasprun.xml.gz')

            if os.path.exists(vr_file):
                vasprun = Vasprun(vr_file)
            elif os.path.exists(vr_file_gz):
                vasprun = Vasprun(vr_file_gz)
            else:
                msg = 'No vasprun.xml found in {}'.format(directory)
                logger.error(msg)
                raise FileNotFoundError(msg)

        if not settings_file:
            settings_file = joinpath(directory, "settings.yaml")
        settings = load_settings_from_file(settings_file)

        return AmsetRunner.from_vasprun_and_settings(vasprun, settings)

    def write_settings(self,
                       directory: str = '.',
                       prefix: Optional[str] = None):
        if prefix is None:
            prefix = ''
        else:
            prefix += '_'

        general_settings = {
            "scissor": self.scissor,
            "bandgap": self.user_bandgap,
            "interpolation_factor": self.interpolation_factor,
            "scattering_type": self.scattering_type,
            "doping": self.doping,
            "temperatures": self.temperatures}

        settings = {"general": general_settings,
                    "material": self.material_properties,
                    "performance": self.performance_parameters,
                    "output": self.output_parameters}

        filename = joinpath(directory, "{}amset_settings.yaml".format(prefix))
        write_settings_to_file(settings, filename)


def _log_amset_intro():
    now = datetime.datetime.now()
    logger.info("""
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
        __version__, now.strftime("%d %b %Y"), now.strftime("%H:%M")))


def _log_structure_information(structure: Structure, symprec):
    star_log("STRUCTURE")
    logger.info("Structure information:")

    formula = structure.composition.get_reduced_formula_and_factor(
        iupac_ordering=True)[0]

    if not symprec:
        symprec = 0.01

    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    log_list(["formula: {}".format(unicodeify(formula)),
              "# sites: {}".format(structure.num_sites),
              "space group: {}".format(unicodeify_spacegroup(
                  sga.get_space_group_symbol()))])

    logger.info("Lattice:")
    log_list([
        "a, b, c [Å]: {:.2f}, {:.2f}, {:.2f}".format(*structure.lattice.abc),
        "α, β, γ [°]: {:.0f}, {:.0f}, {:.0f}".format(*structure.lattice.angles)]
    )


def _log_settings(runner: AmsetRunner):
    star_log("SETTINGS")

    logger.info("Run parameters:")
    run_params = [
        "doping: {}".format(", ".join(map("{:g}".format, runner.doping))),
        "temperatures: {}".format(", ".join(map(str, runner.temperatures))),
        "interpolation_factor: {}".format(runner.interpolation_factor),
        "scattering_type: {}".format(runner.scattering_type),
        "soc: {}".format(runner.soc)]

    if runner.user_bandgap:
        run_params.append("bandgap: {}".format(runner.user_bandgap))

    if runner.scissor:
        run_params.append("scissor: {}".format(runner.scissor))

    log_list(run_params)

    logger.info("Performance parameters:")
    log_list(["{}: {}".format(k, v) for k, v in
              runner.performance_parameters.items()])

    logger.info("Output parameters:")
    log_list(["{}: {}".format(k, v) for k, v in
              runner.output_parameters.items()])

    logger.info("Material properties:")
    log_list(["{}: {}".format(k, v) for k, v in
              runner.material_properties.items() if v is not None])


def _log_band_structure_information(band_structure: BandStructure):
    star_log("BAND STRUCTURE")

    logger.info("Input band structure information:")
    log_list([
        "# bands: {}".format(band_structure.nb_bands),
        "# k-points: {}".format(len(band_structure.kpoints)),
        "Fermi level: {:.3f} eV".format(band_structure.efermi),
        "spin polarized: {}".format(band_structure.is_spin_polarized),
        "metallic: {}".format(band_structure.is_metal())])

    if band_structure.is_metal():
        return

    logger.info("Band gap:")
    band_gap_info = []

    bg_data = band_structure.get_band_gap()
    if not bg_data['direct']:
        band_gap_info.append(
            'indirect band gap: {:.3f} eV'.format(bg_data['energy']))

    direct_data = band_structure.get_direct_band_gap_dict()
    direct_bg = min((spin_data['value'] for spin_data in direct_data.values()))
    band_gap_info.append('direct band gap: {:.3f} eV'.format(direct_bg))

    direct_kpoint = []
    for spin, spin_data in direct_data.items():
        direct_kindex = spin_data['kpoint_index']
        direct_kpoint.append(_kpt_str.format(
            k=band_structure.kpoints[direct_kindex].frac_coords))

    band_gap_info.append("direct k-point: {}".format(", ".join(direct_kpoint)))
    log_list(band_gap_info)

    vbm_data = band_structure.get_vbm()
    cbm_data = band_structure.get_cbm()

    logger.info('\nValence band maximum:')
    _log_band_edge_information(band_structure, vbm_data)

    logger.info('\nConduction band minimum:')
    _log_band_edge_information(band_structure, cbm_data)


def _log_band_edge_information(band_structure, edge_data):
    """Log data about the valence band maximum or conduction band minimum.

    Args:
        band_structure: A band structure.
        edge_data (dict): The :obj:`dict` from ``bs.get_vbm()`` or
            ``bs.get_cbm()``
    """
    if band_structure.is_spin_polarized:
        spins = edge_data['band_index'].keys()
        b_indices = [', '.join([str(i+1) for i in
                                edge_data['band_index'][spin]])
                     + '({})'.format(spin.name.capitalize()) for spin in spins]
        b_indices = ', '.join(b_indices)
    else:
        b_indices = ', '.join([str(i+1) for i in
                               edge_data['band_index'][Spin.up]])

    kpoint = edge_data['kpoint']
    kpoint_str = _kpt_str.format(k=kpoint.frac_coords)

    log_list(["energy: {:.3f} eV".format(edge_data['energy']),
              "k-point: {}".format(kpoint_str),
              "band indices: {}".format(b_indices)])
