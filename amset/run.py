import os
import copy
import datetime
import logging


import numpy as np

from pathlib import Path
from os.path import join as joinpath
from typing import Optional, Any, Dict, Union, List

from tabulate import tabulate

from monty.json import MSONable

from amset.interpolate import Interpolater
from amset.io import load_settings_from_file
from amset.scatter import ScatteringCalculator
from amset.bte import BTESolver
from amset.util import validate_settings, star_log, tensor_average, log_list
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.io.vasp import Vasprun
from amset import __version__, amset_defaults

logger = logging.getLogger(__name__)


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
        self._scissor = scissor
        self._user_bandgap = user_bandgap
        self._soc = soc
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
            write_mesh: bool = True):
        _log_amset_intro()
        # _log_structure_information(self._band_structure)
        # _log_settings(self)

        # initialize scattering first so we can check that materials properties
        # and desired scattering mechanisms are consistent
        scatter = ScatteringCalculator(
            self.material_properties, self.doping, self.temperatures,
            scattering_type=self.scattering_type,
            energy_tol=self.performance_parameters["energy_tol"],
            g_tol=self.performance_parameters["g_tol"],
            use_symmetry=self.performance_parameters["symprec"] is not None,
            nworkers=self.performance_parameters["nworkers"])

        star_log("INTERPOLATION")

        interpolater = Interpolater(
            self._band_structure, num_electrons=self._num_electrons,
            interpolation_factor=self.interpolation_factor, soc=self._soc,
            interpolate_projections=True)

        amset_data = interpolater.get_amset_data(
            energy_cutoff=self.performance_parameters["energy_cutoff"],
            scissor=self._scissor, bandgap=self._user_bandgap,
            symprec=self.performance_parameters["symprec"],
            nworkers=self.performance_parameters["nworkers"])

        star_log("DOS")

        amset_data.calculate_dos(
            dos_estep=self.performance_parameters["dos_estep"],
            dos_width=self.performance_parameters["dos_width"])
        amset_data.set_doping_and_temperatures(
            self.doping, self.temperatures)

        star_log("SCATTERING")

        amset_data.set_scattering_rates(
            scatter.calculate_scattering_rates(amset_data),
            [m.name for m in scatter.scatterers])

        star_log('BTE')

        solver = BTESolver(
            separate_scattering_mobilities=self.output_parameters[
                "separate_scattering_mobilities"],
            calculate_mobility=self.output_parameters["calculate_mobility"])
        sigma, seebeck, kappa, mobility = solver.solve_bte(
            amset_data)

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
                     tensor_average(sigma[c, t]), tensor_average(seebeck[c, t]),
                     tensor_average(mobility["overall"][c, t])))

        else:
            logger.info("Average conductivity (σ) and Seebeck (S) results:")
            headers = ("conc [cm⁻³]", "temp [K]", "σ [S/m]", "S [µV/K]")
            for c, t in np.ndindex(amset_data.fermi_levels.shape):
                results_summary.append(
                    (self.doping[c], self.temperatures[t],
                     tensor_average(sigma[c, t]), tensor_average(seebeck[c, t]),
                     tensor_average(mobility["overall"][c, t])))

        logger.info(tabulate(
            results_summary, headers=headers, numalign="center",
            stralign="center", floatfmt=(".2g", ".1f", ".2g", ".2g", ".1f")))

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
